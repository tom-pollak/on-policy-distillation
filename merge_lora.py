import torch
from peft import PeftModel
from pydantic import validate_call
from pydantic_config import parse_argv
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MergeConfig


@validate_call
def main(conf: MergeConfig = MergeConfig()) -> None:
    dtype = torch.bfloat16 if conf.bf16 else torch.float16

    print(f"Loading base model {conf.base_model_name} in full precision...")
    base_model = AutoModelForCausalLM.from_pretrained(
        conf.base_model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {conf.lora_dir}...")
    model = PeftModel.from_pretrained(base_model, str(conf.lora_dir))

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {conf.output_dir}...")
    model.save_pretrained(str(conf.output_dir))

    tokenizer = AutoTokenizer.from_pretrained(conf.base_model_name)
    tokenizer.save_pretrained(str(conf.output_dir))

    print("Done! Merged model saved.")
    print(f"\nTo load as 4-bit for eval, use:")
    print(f"  model = AutoModelForCausalLM.from_pretrained(")
    print(f"      '{conf.output_dir}',")
    print(f"      quantization_config=bnb_config,")
    print(f"      device_map='auto',")
    print(f"  )")


if __name__ == "__main__":
    main(**parse_argv())
