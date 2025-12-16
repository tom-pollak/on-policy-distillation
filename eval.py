from pathlib import Path
import torch
import wandb
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from pydantic import validate_call
from pydantic_config import parse_argv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pydantic_config import BaseConfig


class EvalConfig(BaseConfig):
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    kd_student_dir: Path = Path("./qwen_kd_baseline")
    onpolicy_student_dir: Path = Path("./qwen_onpolicy_kd")  # merged + re-quantized
    tasks: list[str] = ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "mmlu"]


def load_model(
    path: str,
    dtype: torch.dtype,
    quantize_4bit: bool = False,
    base_model: str | None = None,
):
    """Load a model, optionally with 4-bit quantization.

    If base_model is provided, path is treated as a LoRA adapter directory.
    The adapter is merged before quantization.
    """
    if base_model is not None:
        # Load base model in full precision, merge LoRA, then quantize if needed
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()

        if quantize_4bit:
            # Re-load with quantization after saving merged weights
            import tempfile

            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir)
                del model
                torch.cuda.empty_cache()
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                return AutoModelForCausalLM.from_pretrained(
                    tmp_dir, quantization_config=bnb_config
                )
        return model

    if quantize_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb_config
        )
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, device_map="auto"
    )


def run_lm_eval(
    path,
    dtype,
    quantize,
    tokenizer,
    task_list: list[str],
    num_fewshot: int = 0,
    base_model: str | None = None,
) -> dict:
    """Run lm-evaluation-harness on specified tasks."""
    model = load_model(path, dtype, quantize_4bit=quantize, base_model=base_model)
    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size="auto",
    )
    del model
    return {
        task: results["results"][task].get(
            "acc,none", results["results"][task].get("acc_norm,none")
        )
        for task in task_list
    }


def do_log(tasks, table, name, res):
    print(f"{name:25s} | " + " | ".join(f"{res[t]:8.4f}" for t in tasks))
    table.add_data(name, *[res[t] for t in tasks])
    for task in tasks:
        wandb.summary[f"{name}/{task}"] = res[task]


@validate_call
def main(conf: EvalConfig = EvalConfig()) -> None:
    wandb.init(
        project="on-policy-distillation", name="eval_comparison", job_type="eval"
    )

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(conf.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # (path, quantize, base_model) - base_model is set for LoRA adapters
    models = {
        "teacher": (conf.model_name, False, None),
        "student_ptq_4bit": (conf.model_name, True, None),
        "student_kd": (conf.kd_student_dir, True, conf.model_name),
        "student_onpolicy": (conf.onpolicy_student_dir, True, conf.model_name),
    }

    # header
    print(f"{'model':25s} | " + " | ".join(f"{t[:8]:>8s}" for t in conf.tasks))
    table = wandb.Table(columns=["model"] + conf.tasks)
    for name, (path, quantize, base_model) in models.items():
        res = run_lm_eval(
            path, dtype, quantize, tokenizer, conf.tasks, base_model=base_model
        )
        do_log(conf.tasks, table, name, res)
        torch.cuda.empty_cache()

    wandb.log({"eval_results": table})
    wandb.finish()


if __name__ == "__main__":
    main(**parse_argv())
