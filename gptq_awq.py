"""Evaluate GPTQ and AWQ quantized models for comparison with torchao distillation."""

import os

os.environ.setdefault("HF_HOME", "./hf-cache")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import wandb
from datasets import Dataset, load_dataset
from pydantic_config import parse_argv
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import QuantEvalConfig, Tee
from eval import run_lm_eval


def get_calibration_dataset(cfg: QuantEvalConfig, tokenizer):
    """Prepare calibration data for quantization."""
    ds = load_dataset(cfg.dataset_name, split="train", streaming=True)
    examples = []
    for i, sample in enumerate(ds):
        if i >= cfg.num_calibration_samples:
            break
        # Handle different dataset formats
        if "messages" in sample:
            text = tokenizer.apply_chat_template(
                sample["messages"], tokenize=False, add_generation_prompt=False
            )
        elif "text" in sample:
            text = sample["text"]
        else:
            text = str(sample[list(sample.keys())[0]])

        if len(text) > 256:
            examples.append({"text": text[: cfg.max_seq_length * 4]})

    ds = Dataset.from_list(examples)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=cfg.max_seq_length,
            truncation=True,
            add_special_tokens=True,
        )

    return ds.map(tokenize, remove_columns=ds.column_names)


def eval_gptq(cfg: QuantEvalConfig, tokenizer) -> dict:
    """Quantize model with GPTQ via llmcompressor and evaluate."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    print(f"\n{'=' * 60}")
    print("Evaluating GPTQ INT4")
    print(f"{'=' * 60}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=cfg.dtype, device_map="auto"
    )

    ds = get_calibration_dataset(cfg, tokenizer)

    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=cfg.max_seq_length,
        num_calibration_samples=cfg.num_calibration_samples,
    )

    model.save_pretrained(cfg.output_dir, save_compressed=True)
    tokenizer.save_pretrained(cfg.output_dir)

    results = run_lm_eval(model, tokenizer, cfg.tasks)
    del model
    torch.cuda.empty_cache()
    return results


def eval_awq(cfg: QuantEvalConfig, tokenizer) -> dict:
    """Quantize model with AWQ via llmcompressor and evaluate."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier

    print(f"\n{'=' * 60}")
    print("Evaluating AWQ INT4")
    print(f"{'=' * 60}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=cfg.dtype, device_map="auto"
    )

    ds = get_calibration_dataset(cfg, tokenizer)

    recipe = AWQModifier(
        targets="Linear", scheme="W4A16_ASYM", ignore=["lm_head"], duo_scaling="both"
    )

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=cfg.max_seq_length,
        num_calibration_samples=cfg.num_calibration_samples,
    )

    results = run_lm_eval(model, tokenizer, cfg.tasks)
    del model
    torch.cuda.empty_cache()
    return results


def main(cfg: QuantEvalConfig):
    wandb.init(project=cfg.wandb_project, name="gptq-awq-eval", tags=cfg.tags)
    Tee.redirect_stdout_stderr("./gptq_awq.log")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build results table
    columns = ["method"] + cfg.tasks + ["avg"]
    table = wandb.Table(columns=columns)

    all_results = {}

    try:
        all_results["GPTQ INT4"] = eval_gptq(cfg, tokenizer)
    except Exception as e:
        print(f"GPTQ eval failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        all_results["AWQ INT4"] = eval_awq(cfg, tokenizer)
    except Exception as e:
        print(f"AWQ eval failed: {e}")
        import traceback

        traceback.print_exc()

    # Log results
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}\n")

    header = "| Method | " + " | ".join(cfg.tasks) + " | Avg |"
    separator = "|" + "|".join(["---"] * (len(cfg.tasks) + 2)) + "|"
    print(header)
    print(separator)

    for method, results in all_results.items():
        avg = sum(results.values()) / len(results)
        row = f"| {method} | " + " | ".join(f"{results[t]:.3f}" for t in cfg.tasks)
        row += f" | {avg:.3f} |"
        print(row)

        # Log to wandb
        table.add_data(method, *[results[t] for t in cfg.tasks], avg)
        for task, score in results.items():
            wandb.summary[f"eval/{method}/{task}"] = score
        wandb.summary[f"eval/{method}/avg"] = avg

    wandb.log({"eval_results": table})
    wandb.summary["eval_results"] = table

    print("\n(For reference from README:)")
    print("| torchao PTQ | 0.509 | 0.816 | 0.527 | 0.674 | 0.678 | 0.641 |")
    print("| Off-policy  | 0.517 | 0.824 | 0.538 | 0.688 | 0.682 | 0.650 |")
    print("| On-policy   | 0.513 | 0.821 | 0.531 | 0.690 | 0.684 | 0.648 |")

    wandb.finish()


if __name__ == "__main__":
    main(QuantEvalConfig(**parse_argv()))
