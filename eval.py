import torch
import wandb
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from pydantic import validate_call
from pydantic_config import parse_argv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import EvalConfig

DEFAULT_TASKS = ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "mmlu"]


def load_model(path: str, dtype: torch.dtype, quantize_4bit: bool = False):
    """Load a model, optionally with 4-bit quantization."""
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


def run_lm_eval(model, tokenizer, task_list: list[str], num_fewshot: int = 0) -> dict:
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

    dtype = torch.bfloat16 if conf.bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(conf.teacher_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    models = {
        "teacher": (conf.teacher_model_name, False),
        "student_ptq_4bit": (conf.ptq_student_name, True),
        "student_offpolicy": (conf.kd_student_dir, True),
        "student_onpolicy": (conf.onpolicy_student_dir, True),
    }

    header = f"{'model':25s} | " + " | ".join(f"{t[:8]:>8s}" for t in DEFAULT_TASKS)
    print(header)
    table = wandb.Table(columns=["model"] + DEFAULT_TASKS)
    for name, (path, quantize) in models.items():
        model = load_model(path, dtype, quantize_4bit=quantize)

        downstream = run_lm_eval(model, tokenizer, DEFAULT_TASKS)

        # log
        print(
            f"{name:25s} | "
            + " | ".join(f"{downstream[t]:8.4f}" for t in DEFAULT_TASKS)
        )
        table.add_data(name, *[downstream[t] for t in DEFAULT_TASKS])
        for task in DEFAULT_TASKS:
            wandb.summary[f"{name}/{task}"] = downstream[task]

        del model
        torch.cuda.empty_cache()

    wandb.log({"eval_results": table})
    wandb.finish()


if __name__ == "__main__":
    main(**parse_argv())
