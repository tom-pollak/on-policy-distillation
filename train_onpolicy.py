import math
import random
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from pydantic import validate_call
from pydantic_config import parse_argv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

from config import OnPolicyKDConfig


def reverse_kl_on_generated(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    prompt_lengths: List[int],
    temperature: float,
) -> torch.Tensor:
    """
    Compute reverse KL(student || teacher) on tokens after the prompt.

    student_logits, teacher_logits: [B, L, V]
    prompt_lengths: length of each prompt (in tokens)
    """
    B, L, V = student_logits.shape
    total_kl = 0.0
    count = 0

    for i in range(B):
        pl = prompt_lengths[i]
        start = max(pl - 1, 0)
        if start >= L - 1:
            continue
        s_slice = student_logits[i, start:-1, :] / temperature
        t_slice = teacher_logits[i, start:-1, :] / temperature

        log_s = F.log_softmax(s_slice, dim=-1)
        prob_t = F.softmax(t_slice, dim=-1)
        kl = F.kl_div(log_s, prob_t, reduction="batchmean", log_target=False)
        total_kl = total_kl + kl
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=student_logits.device)
    return total_kl / count


@validate_call
def main(conf: OnPolicyKDConfig) -> None:
    # -----------------------------
    # Setup & seeding
    # -----------------------------
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed_all(conf.seed)

    dtype = torch.bfloat16 if conf.bf16 else torch.float16

    # -----------------------------
    # Tokenizer & dataset
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(conf.teacher_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_dataset(conf.dataset_name, split="train")
    if conf.max_train_samples is not None:
        raw_dataset = raw_dataset.select(range(conf.max_train_samples))

    def extract_prompt_messages(messages):
        # Drop last assistant/model message if present; keep everything else as the "prompt".
        if len(messages) >= 2 and messages[-1]["role"] in ("assistant", "model"):
            return messages[:-1]
        return messages

    def collate_prompts(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts_texts = []
        prompt_msg_lists = []

        for ex in batch:
            msgs = extract_prompt_messages(ex["messages"])
            if not msgs:
                msgs = ex["messages"]
            prompt_msg_lists.append(msgs)

            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,  # append assistant header
            )
            prompts_texts.append(text)

        tokenized = tokenizer(
            prompts_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=conf.max_seq_length - conf.max_new_tokens,
        )
        return {
            "prompt_input_ids": tokenized["input_ids"],
            "prompt_attention_mask": tokenized["attention_mask"],
            "prompt_msgs": prompt_msg_lists,
        }

    prompt_loader = DataLoader(
        raw_dataset,
        batch_size=conf.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_prompts,
    )

    # -----------------------------
    # Models: teacher FP, student 4-bit + LoRA
    # -----------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    teacher_model = AutoModelForCausalLM.from_pretrained(
        conf.teacher_model_name,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    student_base = AutoModelForCausalLM.from_pretrained(
        conf.student_model_name,
        quantization_config=bnb_config,
        device_map=None,
        trust_remote_code=True,
    ).to(device)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student_model = get_peft_model(student_base, lora_config)
    student_model.train()
    student_model.gradient_checkpointing_enable()

    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=conf.learning_rate)

    # Rough schedule estimate
    num_batches_per_epoch = len(raw_dataset) // conf.per_device_train_batch_size
    num_update_steps_per_epoch = max(
        1, math.ceil(num_batches_per_epoch / conf.gradient_accumulation_steps)
    )
    num_train_epochs = max(1, conf.max_train_steps // num_update_steps_per_epoch)
    t_total = conf.max_train_steps
    warmup_steps = int(conf.warmup_ratio * t_total)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # -----------------------------
    # Training loop
    # -----------------------------
    global_step = 0

    for epoch in range(num_train_epochs):
        if global_step >= conf.max_train_steps:
            break

        for batch in prompt_loader:
            if global_step >= conf.max_train_steps:
                break

            prompt_input_ids = batch["prompt_input_ids"].to(device)
            prompt_attention_mask = batch["prompt_attention_mask"].to(device)

            # 1) On-policy rollouts from quantized student
            with torch.no_grad():
                gen_outputs = student_model.generate(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    max_new_tokens=conf.max_new_tokens,
                    do_sample=True,
                    top_p=0.95,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
                sequences = gen_outputs.sequences  # [B, L_full]

            # Compute prompt lengths in token space
            prompt_lengths: List[int] = []
            for i in range(sequences.size(0)):
                prompt_ids = prompt_input_ids[i]
                pl = (prompt_ids != tokenizer.pad_token_id).sum().item()
                prompt_lengths.append(pl)

            sequences = sequences.to(device)
            attention_mask_full = (
                (sequences != tokenizer.pad_token_id).long().to(device)
            )

            # 2) Forward both models on full sequences
            student_outputs = student_model(
                input_ids=sequences,
                attention_mask=attention_mask_full,
            )
            student_logits = student_outputs.logits

            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=sequences,
                    attention_mask=attention_mask_full,
                )
                teacher_logits = teacher_outputs.logits

            # 3) Reverse KL on generated region
            loss = reverse_kl_on_generated(
                student_logits, teacher_logits, prompt_lengths, temperature=1.0
            )
            loss = loss / conf.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % conf.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % 50 == 0:
                print(f"step {global_step} | loss {loss.item():.4f}")

            global_step += 1

    student_model.save_pretrained(str(conf.output_dir))
    tokenizer.save_pretrained(str(conf.output_dir))
    print(f"Saved on-policy KD student to {conf.output_dir}")


if __name__ == "__main__":
    main(**parse_argv())
