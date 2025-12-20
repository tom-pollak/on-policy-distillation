"""Debug script to find problematic samples around step 303."""

import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from config import TrainConfig

# Training parameters
NUM_GPUS = 8
STEP_WITH_ERROR = 303

cfg = TrainConfig()

# Calculate samples per step
samples_per_micro_batch = cfg.per_device_train_batch_size * NUM_GPUS
samples_per_step = samples_per_micro_batch * cfg.gradient_accumulation_steps

print(f"per_device_train_batch_size: {cfg.per_device_train_batch_size}")
print(f"gradient_accumulation_steps: {cfg.gradient_accumulation_steps}")
print(f"num_gpus: {NUM_GPUS}")
print(f"samples_per_step: {samples_per_step}")
print()

print(f"Error at step: {STEP_WITH_ERROR}")
print()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def analyze_sample(idx, example):
    """Analyze a single sample for potential issues."""
    messages = example.get("messages", [])
    issues = []

    if len(messages) == 0:
        issues.append("NO_MESSAGES")
        return {"idx": idx, "issues": issues, "messages": messages}

    # Check for empty content
    for i, m in enumerate(messages):
        content = m.get("content", "")
        if not content or not content.strip():
            issues.append(f"EMPTY_CONTENT_MSG_{i}_{m.get('role', 'unknown')}")

    # Extract prompt the same way as DataCollatorForChatML: all messages except last
    prompt_msgs = messages[:-1]

    if len(prompt_msgs) == 0:
        issues.append("NO_PROMPT_MESSAGES")
        return {"idx": idx, "issues": issues, "messages": messages}

    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)

        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

        prompt_len = len(prompt_tokens)
        full_len = len(full_tokens)
        response_len = min(full_len, cfg.max_length) - prompt_len

        if prompt_len == 0:
            issues.append("ZERO_PROMPT_TOKENS")

        if prompt_len >= cfg.max_length:
            issues.append(f"PROMPT_EXCEEDS_MAX_LENGTH({prompt_len}>={cfg.max_length})")

        if response_len < 32:  # min_response_tokens default
            issues.append(f"INSUFFICIENT_RESPONSE_TOKENS({response_len}<32)")

        # Check if prompt_text is effectively empty after encoding
        if not prompt_text.strip():
            issues.append("EMPTY_PROMPT_TEXT")

        # KEY BUG CHECK: collator sets prompt_ids=[] when completion >= max_length
        completion_len = full_len - prompt_len
        if completion_len >= cfg.max_length:
            issues.append(f"COMPLETION_EXCEEDS_MAX_LENGTH({completion_len}>={cfg.max_length})")

    except Exception as e:
        issues.append(f"TOKENIZATION_ERROR: {str(e)}")
        return {"idx": idx, "issues": issues, "messages": messages}

    return {
        "idx": idx,
        "issues": issues,
        "prompt_len": prompt_len,
        "full_len": full_len,
        "response_len": response_len,
        "completion_len": full_len - prompt_len,  # raw completion before truncation
        "num_messages": len(messages),
        "roles": [m["role"] for m in messages],
        "content_lengths": [len(m.get("content", "")) for m in messages],
        "prompt_text_preview": prompt_text[:200] if prompt_text else "",
        "messages": messages,
    }


# Load raw dataset (before filtering)
print("Loading raw dataset...")
raw_dataset = load_dataset(cfg.dataset_name, split="train")
print(f"Raw dataset size: {len(raw_dataset)}")

# Apply the same filtering as train.py
print("Applying filters...")


def filter_dataset(dataset, tokenizer, max_length, min_response_tokens=32):
    def more_than_one_message(example):
        return len(example.get("messages", [])) > 0

    def non_empty_message(example):
        return all(m.get("content", "").strip() for m in example["messages"])

    def has_room_for_response(example):
        messages = example["messages"]
        prompt_msgs = messages[:-1]  # match collator: all except last
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        full_len = len(tokenizer.encode(full_text, add_special_tokens=False))
        response_len = min(full_len, max_length) - prompt_len
        return (
            prompt_len < max_length - min_response_tokens
            and response_len >= min_response_tokens
        )

    filters = [more_than_one_message, non_empty_message, has_room_for_response]
    return dataset.filter(lambda x: all(f(x) for f in filters), num_proc=os.cpu_count())


filtered_dataset = filter_dataset(raw_dataset, tokenizer, max_length=cfg.max_length)
print(f"Filtered dataset size: {len(filtered_dataset)}")
print()

# Simulate shuffled DataLoader (Trainer shuffles with seed)
print("Simulating shuffled DataLoader...")
g = torch.Generator()
g.manual_seed(cfg.seed)  # seed=42
dataset_size = len(filtered_dataset)
shuffled_indices = torch.randperm(dataset_size, generator=g).tolist()

# With DistributedSampler, each GPU gets every Nth sample
# GPU i gets indices: i, i+8, i+16, ...
# At step 303, each GPU has consumed 303 * batch * grad_accum samples
samples_per_gpu_per_step = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps

# Get indices for all GPUs at step 303 (and a few steps around it)
step_indices = []
for step in range(STEP_WITH_ERROR - 1, STEP_WITH_ERROR + 2):  # steps 302, 303, 304
    for rank in range(NUM_GPUS):
        rank_indices = shuffled_indices[rank::NUM_GPUS]
        start = step * samples_per_gpu_per_step
        end = start + samples_per_gpu_per_step
        if end <= len(rank_indices):
            step_indices.extend(rank_indices[start:end])

step_indices = list(set(step_indices))  # dedupe
print(f"Checking {len(step_indices)} unique indices around step {STEP_WITH_ERROR}")
print()

# Analyze these specific samples
problematic_samples = []
all_analyses = []

for idx in step_indices:
    analysis = analyze_sample(idx, filtered_dataset[idx])
    all_analyses.append(analysis)

    if analysis["issues"]:
        problematic_samples.append(analysis)
        print(f"FOUND ISSUE at idx {idx}: {analysis['issues']}")

print()
print(f"Total samples analyzed: {len(all_analyses)}")
print(f"Problematic samples found: {len(problematic_samples)}")

# Save results
output_file = "debug_batch_results.json"
results = {
    "config": {
        "step_with_error": STEP_WITH_ERROR,
        "num_gpus": NUM_GPUS,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "samples_per_step": samples_per_step,
        "max_length": cfg.max_length,
        "seed": cfg.seed,
    },
    "shuffled_indices_checked": step_indices,
    "dataset_sizes": {"raw": len(raw_dataset), "filtered": len(filtered_dataset)},
    "problematic_samples": problematic_samples,
    "all_analyses": all_analyses,
}

with open(output_file, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {output_file}")

# Also print summary statistics
if all_analyses:
    prompt_lens = [a.get("prompt_len", 0) for a in all_analyses if "prompt_len" in a]
    response_lens = [a.get("response_len", 0) for a in all_analyses if "response_len" in a]

    if prompt_lens:
        print(f"\nPrompt length stats in range:")
        print(f"  min: {min(prompt_lens)}, max: {max(prompt_lens)}, avg: {sum(prompt_lens)/len(prompt_lens):.1f}")

    if response_lens:
        print(f"Response length stats in range:")
        print(f"  min: {min(response_lens)}, max: {max(response_lens)}, avg: {sum(response_lens)/len(response_lens):.1f}")
