import os
import logging
import torch
import wandb
from accelerate import PartialState
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from pydantic import validate_call
from pydantic_config import parse_argv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torchao.quantization import quantize_

from config import EvalConfig, Tee


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
    num_fewshot: int = 0,
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

    if state.is_main_process:
        wandb.init(project=cfg.wandb_project, name="eval_comparison", tags=["eval"])
        Tee.redirect_stdout_stderr("./eval.log")
    else:
        logging.disable(logging.WARNING)
        os.environ["TQDM_DISABLE"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # header
    if state.is_main_process:
        print(f"{'model':25s} | " + " | ".join(f"{t[:8]:>8s}" for t in cfg.tasks))
        table = wandb.Table(columns=["model"] + cfg.tasks)
    else:
        table = None

    def eval_and_log(name: str, model):
        res = run_lm_eval(model, tokenizer, cfg.tasks)
        if state.is_main_process:
            assert table is not None
            print(f"{name:25s} | " + " | ".join(f"{res[t]:8.4f}" for t in cfg.tasks))
            table.add_data(name, *[res[t] for t in cfg.tasks])
            for task in cfg.tasks:
                wandb.summary[f"{name}/{task}"] = res[task]

        del model

    # # Teacher model (unquantized)
    # eval_and_log("teacher", cfg.load_model())
    # torch.cuda.empty_cache()

    # # # Teacher model (quantized) - PTQ baseline
    # eval_and_log(
    #     "teacher_ptq",
    #     cfg.load_quant_model("ptq"),
    # )
    # torch.cuda.empty_cache()

    # Evaluate each LoRA adapter
    for lora_path in cfg.lora_paths:
        model = cfg.load_model()
        model = PeftModel.from_pretrained(model, str(lora_path))
        model = model.merge_and_unload()
        quantize_(model, cfg._get_torchao_config())
        eval_and_log(lora_path.stem, model)
        del model
        torch.cuda.empty_cache()

    if state.is_main_process:
        wandb.log({"eval_results": table})
        wandb.finish()


if __name__ == "__main__":
    # print(parse_argv())
    raw = {
        "lora_paths": [
            # "qwen_kd_baseline/checkpoint-300",
            # "qwen_onpolicy_4b_int4/checkpoint-300",
            "qwen_kd_half/checkpoint-300",
            # "qwen_offpolicy_4b_bnb_fp4/checkpoint-300",
            "qwen_onpolicy_4b_bnb_fp4/checkpoint-300",
        ],
        "tasks": "arc_challenge",
    }
    main(EvalConfig(**raw))
