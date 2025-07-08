import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_model_components(model_id):

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    return model, tokenizer, generator
