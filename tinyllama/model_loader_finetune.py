from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch

def load_quantized_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16
    )

    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    return model, tokenizer, generator