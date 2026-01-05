import logging
import os
from pathlib import Path

# Set cache directories before importing HF/lm_eval libraries
os.environ.setdefault("HF_HOME", "./hf-cache")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import wandb
from accelerate import PartialState
from datasets import load_dataset
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from peft import PeftModel
from pydantic import validate_call
from pydantic_config import parse_argv
from torchao.quantization import quantize_
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import EvalConfig, Tee


PERPLEXITY_DATASETS = {
    "wikitext": ("wikitext", "wikitext-2-raw-v1", "test"),
    "c4": ("allenai/c4", "en", "validation"),
}


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    dataset: str | None = "wikitext",
    max_length: int = 1024,
    stride: int = 512,
    max_samples: int | None = None,
) -> float | None:
    """Compute perplexity on a dataset using sliding window.

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Dataset name ("wikitext", "c4") or None to skip
        max_length: Maximum sequence length for each window
        stride: Stride between windows (smaller = more overlap, more accurate)
        max_samples: Max samples to use (for large datasets like C4)

    Returns:
        Perplexity value, or None if dataset is None
    """
    if dataset is None:
        return None

    if dataset not in PERPLEXITY_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset}. Choose from {list(PERPLEXITY_DATASETS.keys())}"
        )

    ds_name, ds_config, ds_split = PERPLEXITY_DATASETS[dataset]
    ds = load_dataset(ds_name, ds_config, split=ds_split)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    # Concatenate all text
    text_key = "text" if "text" in ds.column_names else ds.column_names[0]
    text = "\n\n".join(ds[text_key])

    # Tokenize
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    device = next(model.parameters()).device
    nlls = []
    prev_end_loc = 0

    for begin_loc in tqdm(
        range(0, seq_len, stride), desc=f"Perplexity ({dataset})", leave=False
    ):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # tokens to compute loss on

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        # Only compute loss on new tokens (not overlapping ones)
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc >= seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / prev_end_loc)
    return ppl.item()


def get_latest_checkpoint(output_dir: Path) -> tuple[Path, int]:
    """Find the latest checkpoint in the given output directory."""
    checkpoint = max(
        output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])
    )
    step = int(checkpoint.name.split("-")[1])
    return checkpoint, step


def load_model(
    path: str,
    dtype: torch.dtype,
    quant_config=None,
    base_model: str | None = None,
):
    """Load a model, optionally with quantization via torchao.

    If base_model is provided, path is treated as a LoRA adapter directory.
    The adapter is merged before quantization.
    """
    if base_model is not None:
        # Load base model in full precision, merge LoRA, then quantize if needed
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)

    if quant_config is not None:
        quantize_(model, quant_config)
    return model


def run_lm_eval(
    model,
    tokenizer,
    task_list: list[str],
    num_fewshot: int | None = None,
) -> dict:
    """Run lm-evaluation-harness on specified tasks."""
    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size="auto",
    )
    return {
        task: results["results"][task].get(
            "acc,none", results["results"][task].get("acc_norm,none")
        )
        for task in task_list
    }


@validate_call
def main(cfg: EvalConfig) -> None:
    state = PartialState()
    own_wandb_run = wandb.run is None

    if state.is_main_process:
        if own_wandb_run:
            wandb.init(project=cfg.wandb_project, name="eval", tags=cfg.tags)
        Tee.redirect_stdout_stderr("./eval.log")
    else:
        logging.disable(logging.WARNING)
        os.environ["TQDM_DISABLE"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build column list: tasks + avg + optional perplexity
    columns = ["model"] + cfg.tasks + ["avg"]
    if cfg.perplexity_dataset:
        columns.append(f"ppl_{cfg.perplexity_dataset}")

    # header
    if state.is_main_process:
        header = " | ".join(f"{c[:8]:>8s}" for c in columns)
        print(header)
        table = wandb.Table(columns=columns)
    else:
        table = None

    def eval_and_log(name: str, model):
        task_results = run_lm_eval(model, tokenizer, cfg.tasks)
        ppl = compute_perplexity(model, tokenizer, dataset=cfg.perplexity_dataset)

        # Build metrics dict with keys matching columns (except "model")
        metrics = {
            **task_results,
            "avg": sum(task_results.values()) / len(task_results),
        }
        if ppl is not None:
            metrics[f"ppl_{cfg.perplexity_dataset}"] = ppl

        if state.is_main_process:
            assert table is not None
            row = [name] + [metrics[c] for c in columns[1:]]
            print(
                " | ".join(
                    f"{v:8.4f}" if isinstance(v, float) else f"{v:>8s}" for v in row
                )
            )
            table.add_data(*row)
            for key, value in metrics.items():
                wandb.summary[f"eval/{name}/{key}"] = value

        del model

    if cfg.eval_teacher:
        # Teacher model (unquantized)
        eval_and_log("teacher", cfg.load_model())
        torch.cuda.empty_cache()

        # Teacher model (quantized) - PTQ baseline
        eval_and_log(
            "teacher_ptq",
            cfg.load_quant_model("ptq"),
        )
        torch.cuda.empty_cache()

    # Evaluate each LoRA adapter
    for lora_path in cfg.lora_paths:
        checkpoint, step = get_latest_checkpoint(lora_path)
        model = cfg.load_model()
        model = PeftModel.from_pretrained(model, str(checkpoint))
        model = model.merge_and_unload()
        quantize_(model, cfg._get_torchao_config())
        eval_and_log(f"{lora_path.stem}/{step}", model)
        del model
        torch.cuda.empty_cache()

    if state.is_main_process:
        wandb.log({"eval_results": table})  # Shows in Charts
        wandb.summary["eval_results"] = table  # Queryable via runs.summary
        if own_wandb_run:
            wandb.finish()


if __name__ == "__main__":
    main(EvalConfig(**parse_argv()))
