"""Evaluate GPTQ and AWQ quantized models for comparison with torchao distillation."""

import os

os.environ.setdefault("HF_HOME", "./hf-cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, AwqConfig

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
TASKS = ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "mmlu"]


def run_lm_eval(model, tokenizer, tasks: list[str]) -> dict:
    """Run lm-evaluation-harness on specified tasks."""
    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(model=lm, tasks=tasks, batch_size="auto")
    return {
        task: results["results"][task].get(
            "acc,none", results["results"][task].get("acc_norm,none")
        )
        for task in tasks
    }


def get_calibration_dataset(tokenizer, num_samples: int = 128, max_length: int = 2048):
    """Get calibration dataset for quantization."""
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    examples = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        text = sample["text"]
        if len(text) > 256:
            examples.append(text[:max_length])
    return examples


def eval_gptq(model_name: str) -> dict:
    """Quantize model with GPTQ and evaluate using optimum."""
    from optimum.gptq import GPTQQuantizer

    print(f"\n{'=' * 60}")
    print("Evaluating GPTQ INT4")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )

    # Get calibration data
    calibration_texts = get_calibration_dataset(tokenizer)
    calibration_data = [
        tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        for text in calibration_texts
    ]

    # Quantize with GPTQ
    quantizer = GPTQQuantizer(bits=4, group_size=128, desc_act=False, dataset=calibration_data)
    model = quantizer.quantize_model(model, tokenizer)

    results = run_lm_eval(model, tokenizer, TASKS)
    del model
    torch.cuda.empty_cache()
    return results


def eval_awq(model_name: str) -> dict:
    """Quantize model with AWQ and evaluate."""
    from awq import AutoAWQForCausalLM

    print(f"\n{'=' * 60}")
    print("Evaluating AWQ INT4")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and quantize
    model = AutoAWQForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.quantize(
        tokenizer,
        quant_config={
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM",
        },
    )

    results = run_lm_eval(model.model, tokenizer, TASKS)
    del model
    torch.cuda.empty_cache()
    return results


def eval_torchao_ptq(model_name: str) -> dict:
    """Evaluate torchao INT4 PTQ baseline."""
    from torchao.quantization import Int4WeightOnlyConfig, quantize_

    print(f"\n{'=' * 60}")
    print("Evaluating torchao INT4 PTQ")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    quantize_(model, Int4WeightOnlyConfig())

    results = run_lm_eval(model, tokenizer, TASKS)
    del model
    torch.cuda.empty_cache()
    return results


def eval_fp16(model_name: str) -> dict:
    """Evaluate FP16 teacher baseline."""
    print(f"\n{'=' * 60}")
    print("Evaluating FP16 Teacher")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    results = run_lm_eval(model, tokenizer, TASKS)
    del model
    torch.cuda.empty_cache()
    return results


def print_results(all_results: dict[str, dict]):
    """Print results in markdown table format."""
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}\n")

    # Header
    header = "| Method | " + " | ".join(TASKS) + " | Avg |"
    separator = "|" + "|".join(["---"] * (len(TASKS) + 2)) + "|"
    print(header)
    print(separator)

    # Rows
    for method, results in all_results.items():
        avg = sum(results.values()) / len(results)
        row = f"| {method} | " + " | ".join(f"{results[t]:.3f}" for t in TASKS)
        row += f" | {avg:.3f} |"
        print(row)


def main():
    all_results = {}

    # Run evaluations
    try:
        all_results["FP16 Teacher"] = eval_fp16(MODEL_NAME)
    except Exception as e:
        print(f"FP16 eval failed: {e}")

    try:
        all_results["torchao INT4"] = eval_torchao_ptq(MODEL_NAME)
    except Exception as e:
        print(f"torchao eval failed: {e}")

    try:
        all_results["GPTQ INT4"] = eval_gptq(MODEL_NAME)
    except Exception as e:
        print(f"GPTQ eval failed: {e}")

    try:
        all_results["AWQ INT4"] = eval_awq(MODEL_NAME)
    except Exception as e:
        print(f"AWQ eval failed: {e}")

    # Print comparison
    print_results(all_results)

    # Add distillation results for comparison
    print("\n(For reference, your distillation results:)")
    print("| Off-policy | 0.517 | 0.824 | 0.538 | 0.688 | 0.682 | 0.650 |")
    print("| On-policy  | 0.513 | 0.821 | 0.531 | 0.690 | 0.684 | 0.648 |")


if __name__ == "__main__":
    main()
