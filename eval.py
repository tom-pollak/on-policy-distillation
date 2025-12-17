import torch
import wandb
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from pydantic import validate_call
from pydantic_config import parse_argv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torchao.quantization import quantize_

from config import EvalConfig


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
        model = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=dtype, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, device_map="auto"
        )

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
def main(conf: EvalConfig = EvalConfig()) -> None:
    wandb.init(
        project="on-policy-distillation", name="eval_comparison", job_type="eval"
    )

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(conf.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = conf.get_quant_config()

    # header
    print(f"{'model':25s} | " + " | ".join(f"{t[:8]:>8s}" for t in conf.tasks))
    table = wandb.Table(columns=["model"] + conf.tasks)

    def eval_and_log(name: str, model):
        res = run_lm_eval(model, tokenizer, conf.tasks)
        # log
        print(f"{name:25s} | " + " | ".join(f"{res[t]:8.4f}" for t in conf.tasks))
        table.add_data(name, *[res[t] for t in conf.tasks])
        for task in conf.tasks:
            wandb.summary[f"{name}/{task}"] = res[task]

        del model
        torch.cuda.empty_cache()

    # Teacher model (unquantized)
    eval_and_log("teacher", load_model(conf.model_name, dtype))

    # Teacher model (quantized) - PTQ baseline
    eval_and_log(
        "teacher_ptq",
        load_model(conf.model_name, dtype, quant_config=quant_config),
    )

    # Evaluate each LoRA adapter
    for lora_path in conf.lora_paths:
        eval_and_log(
            lora_path.stem,
            load_model(
                str(lora_path),
                dtype,
                quant_config=quant_config,
                base_model=conf.model_name,
            ),
        )

    wandb.log({"eval_results": table})
    wandb.finish()


if __name__ == "__main__":
    main(**parse_argv())
