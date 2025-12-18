"""Debug script to find invalid samples that cause empty tensor errors in GKD training."""

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


model_name = "Qwen/Qwen3-4B-Instruct-2507"
dataset_name = "allenai/tulu-3-sft-mixture"

print(f"Loading tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading dataset {dataset_name}...")
dataset = load_dataset(dataset_name, split="train")
print(f"Dataset size: {len(dataset)}")

# Apply the same filter as train.py
dataset = dataset.filter(lambda x: len(x.get("messages", "")) > 0)
print(f"After basic filter: {len(dataset)}")

invalid_samples = []
empty_prompt_samples = []
empty_tokenization_samples = []

print("\nScanning for problematic samples...")
for idx in tqdm(range(int(len(dataset) * 0.1))):
    sample = dataset[idx]
    messages = sample.get("messages", [])

    # Check 1: Empty or missing messages
    if not messages:
        invalid_samples.append((idx, "empty_messages", sample))
        continue

    # Check 2: No user message (GKD needs a prompt to generate from)
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        invalid_samples.append((idx, "no_user_message", sample))
        continue

    # Check 3: Empty user content
    empty_user = [m for m in user_msgs if not m.get("content", "").strip()]
    if empty_user:
        empty_prompt_samples.append((idx, "empty_user_content", sample))
        continue

    # Check 4: Simulate GKD's prompt extraction
    # GKD extracts all messages except the last assistant response as the prompt
    prompt_messages = []
    for m in messages:
        if m.get("role") == "assistant":
            # Keep all but stop before last assistant message
            break
        prompt_messages.append(m)

    if not prompt_messages:
        empty_prompt_samples.append((idx, "no_prompt_messages", sample))
        continue

    # Check 5: Test tokenization of the prompt
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer(prompt_text, return_tensors="pt")

        if tokens["input_ids"].shape[1] == 0:
            empty_tokenization_samples.append(
                (idx, "empty_tokenization", sample, prompt_text)
            )
    except Exception as e:
        invalid_samples.append((idx, f"tokenization_error: {e}", sample))

# Report findings
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"\nInvalid samples: {len(invalid_samples)}")
for idx, reason, sample in invalid_samples[:10]:
    print(f"  [{idx}] {reason}: {sample.get('messages', [])[:1]}")

print(f"\nEmpty prompt samples: {len(empty_prompt_samples)}")
for idx, reason, sample in empty_prompt_samples[:10]:
    print(f"  [{idx}] {reason}: {sample.get('messages', [])[:2]}")

print(f"\nEmpty tokenization samples: {len(empty_tokenization_samples)}")
for item in empty_tokenization_samples[:10]:
    idx, reason, sample = item[0], item[1], item[2]
    prompt_text = item[3] if len(item) > 3 else ""
    print(f"  [{idx}] {reason}")
    print(f"       messages: {sample.get('messages', [])[:2]}")
    print(f"       prompt_text: {repr(prompt_text[:200])}")

total_bad = (
    len(invalid_samples) + len(empty_prompt_samples) + len(empty_tokenization_samples)
)
print(f"\nTotal problematic samples: {total_bad}")

if total_bad > 0:
    print("\n" + "=" * 60)
    print("SUGGESTED FIX")
    print("=" * 60)
    print("""
Add this filter to train.py after loading the dataset:

def valid_sample(sample, tokenizer):
messages = sample.get("messages", [])
if not messages:
    return False

# Need at least one user message with content
user_msgs = [m for m in messages if m.get("role") == "user"]
if not user_msgs or not all(m.get("content", "").strip() for m in user_msgs):
    return False

# Extract prompt (messages before first assistant response)
prompt_messages = []
for m in messages:
    if m.get("role") == "assistant":
        break
    prompt_messages.append(m)

if not prompt_messages:
    return False

# Check tokenization produces non-empty output
try:
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    tokens = tokenizer(prompt_text, return_tensors="pt")
    return tokens["input_ids"].shape[1] > 0
except:
    return False

dataset = dataset.filter(lambda x: valid_sample(x, tokenizer))
""")
