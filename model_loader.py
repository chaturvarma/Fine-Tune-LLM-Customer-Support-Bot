import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

def model_loader(model_id):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        # Configure 4-bit quantization for more efficient GPU memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load the model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map={"": 0},
            quantization_config=bnb_config,
            torch_dtype=torch.float16
        )
        
        device = torch.device("cuda")

    else:
        # Load the model without quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
        
        device = torch.device("cpu")

    # Create a text generation pipeline
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    return model, tokenizer, generator, device