import yaml
from data_splitting import load_and_split_dataset
from data.field_labels import categories, intents
from model_loader import model_loader
from huggingface_hub import login

# TinyLlama imports
from tinyllama.inference_prompting import general_inference_tinyllama
from tinyllama.lora_finetuning import finetune_category_tinyllama, finetune_intent_tinyllama, evaluate_model_tinyllama

# Llama 3.2 3B imports
from llama_3_2_3b.inference_prompting import general_inference_llama_3_2_3b
from llama_3_2_3b.lora_finetuning import finetune_category_llama_3_2_3b, finetune_intent_llama_3_2_3b, evaluate_model_llama_3_2_3b

# Mistral 7B v0.2 imports
from mistral_7b_v0_2.inference_prompting import general_inference_mistral_7b_v0_2
from mistral_7b_v0_2.lora_finetuning import finetune_category_mistral_7b_v0_2, finetune_intent_mistral_7b_v0_2, evaluate_model_mistral_7b_v0_2

# Gemma 7B IT imports
from gemma_7b.inference_prompting import general_inference_gemma_7b
from gemma_7b.lora_finetuning import finetune_category_gemma_7b, finetune_intent_gemma_7b, evaluate_model_gemma_7b

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Hugging Face token and login
hf_token = config["hf_key"]
login(token=hf_token)

# Dataset and model settings
dataset_size = config["dataset_size"]
selected_model = config["model"].lower()

# LoRA and training parameters
lora_parameters = config.get("lora_parameters", {})
training_parameters = config.get("training_parameters", {})

dataset_path = './data/dataset.csv'

# Load and split the dataset
df_train, df_val, df_test = load_and_split_dataset(dataset_path, dataset_size)

# ==================================================
# Functions to handle each model
# ==================================================

def run_tinyllama():
    """
    Model 1: TinyLlama
    """
    model_id = "Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct"

    # Load the initial model, tokenizer, and text generation pipeline
    model_initial, tokenizer_initial, generator, device = model_loader(model_id)

    # Case 1: Performing inference using general prompting (no fine-tuning)
    accuracy_category, accuracy_intent, predictions_inference = general_inference_tinyllama(df_test, generator, categories, intents)
    print(f"Accuracy without Fine-tuning [TinyLlama] on categories: {accuracy_category:.2f}%")
    print(f"Accuracy without Fine-tuning [TinyLlama] on intents: {accuracy_intent:.2f}%")

    # Case 2: Performing fine-tuning with LoRA
    model_category_tinyllama = finetune_category_tinyllama(df_train, model_initial, tokenizer_initial, device, lora_params=lora_parameters, training_params=training_parameters)
    model_intent_tinyllama = finetune_intent_tinyllama(df_train, model_initial, tokenizer_initial, device, intents, lora_params=lora_parameters, training_params=training_parameters)
    accuracy_category, accuracy_intent, predictions_finetune = evaluate_model_tinyllama(df_test, model_category_tinyllama, model_intent_tinyllama, tokenizer_initial, device, categories, intents)
    print(f"Accuracy with LoRa Fine-tuning [TinyLlama] on categories: {accuracy_category:.2f}%")
    print(f"Accuracy with LoRa Fine-tuning [TinyLlama] on intents: {accuracy_intent:.2f}%")


def run_llama_3_2_3b():
    """
    Model 2: Llama 3.2 3B
    """
    model_id = "meta-llama/Llama-3.2-3B-Instruct"

    # Load the initial model, tokenizer, and text generation pipeline
    model_initial, tokenizer_initial, generator, device = model_loader(model_id)

    # Case 1: Performing inference using general prompting (no fine-tuning)
    accuracy_category, accuracy_intent, predictions_inference = general_inference_llama_3_2_3b(df_test, generator, categories, intents)
    print(f"Accuracy without Fine-tuning [Llama 3.2 3B] on categories: {accuracy_category:.2f}%")
    print(f"Accuracy without Fine-tuning [Llama 3.2 3B] on intents: {accuracy_intent:.2f}%")

    # Case 2: Performing fine-tuning with LoRA
    model_category_llama_3_2_3b = finetune_category_llama_3_2_3b(df_train, model_initial, tokenizer_initial, device, lora_params=lora_parameters, training_params=training_parameters)
    model_intent_llama_3_2_3b = finetune_intent_llama_3_2_3b(df_train, model_initial, tokenizer_initial, device, intents, lora_params=lora_parameters, training_params=training_parameters)
    accuracy_category, accuracy_intent, predictions_finetune = evaluate_model_llama_3_2_3b(df_test, model_category_llama_3_2_3b, model_intent_llama_3_2_3b, tokenizer_initial, device, categories, intents)
    print(f"Accuracy with LoRa Fine-tuning [Llama 3.2 3B] on categories: {accuracy_category:.2f}%")
    print(f"Accuracy with LoRa Fine-tuning [Llama 3.2 3B] on intents: {accuracy_intent:.2f}%")


def run_mistral_7b_v0_2():
    """
    Model 3: Mistral 7B v0.2
    """
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Load the initial model, tokenizer, and text generation pipeline
    model_initial, tokenizer_initial, generator, device = model_loader(model_id)

    # Case 1: Performing inference using general prompting (no fine-tuning)
    accuracy_category, accuracy_intent, predictions_inference = general_inference_mistral_7b_v0_2(df_test, generator, categories, intents)

    print(f"Accuracy without Fine-tuning [Mistral 7B v0.2] on categories: {accuracy_category:.2f}%")
    print(f"Accuracy without Fine-tuning [Mistral 7B v0.2] on intents: {accuracy_intent:.2f}%")

    # Case 2: Performing fine-tuning with LoRA
    model_category_llama_3_2_3b = finetune_category_mistral_7b_v0_2(df_train, model_initial, tokenizer_initial, device, lora_params=lora_parameters, training_params=training_parameters)
    model_intent_llama_3_2_3b = finetune_intent_mistral_7b_v0_2(df_train, model_initial, tokenizer_initial, device, intents, lora_params=lora_parameters, training_params=training_parameters)
    accuracy_category, accuracy_intent, predictions_finetune = evaluate_model_mistral_7b_v0_2(df_test, model_category_llama_3_2_3b, model_intent_llama_3_2_3b, tokenizer_initial, device, categories, intents)
    print(f"Accuracy with LoRa Fine-tuning [Mistral 7B v0.2] on categories: {accuracy_category:.2f}%")
    print(f"Accuracy with LoRa Fine-tuning [Mistral 7B v0.2] on intents: {accuracy_intent:.2f}%")


def run_gemma_7b():
    """
    Model 4: Gemma 7B IT
    """
    model_id = "google/gemma-7b-it"
    
    # Load the initial model, tokenizer, and text generation pipeline
    model_initial, tokenizer_initial, generator, device = model_loader(model_id)

    # Case 1: Performing inference using general prompting (no fine-tuning)
    accuracy_category, accuracy_intent, predictions_inference = general_inference_gemma_7b(df_test, generator, categories, intents)

    print(f"Accuracy without Fine-tuning [Gemma 7B] on categories: {accuracy_category:.2f}%")
    print(f"Accuracy without Fine-tuning [Gemma 7B] on intents: {accuracy_intent:.2f}%")

    # Case 2: Performing fine-tuning with LoRA
    model_category_llama_3_2_3b = finetune_category_gemma_7b(df_train, model_initial, tokenizer_initial, device, lora_params=lora_parameters, training_params=training_parameters)
    model_intent_llama_3_2_3b = finetune_intent_gemma_7b(df_train, model_initial, tokenizer_initial, device, intents, lora_params=lora_parameters, training_params=training_parameters)
    accuracy_category, accuracy_intent, predictions_finetune = evaluate_model_gemma_7b(df_test, model_category_llama_3_2_3b, model_intent_llama_3_2_3b, tokenizer_initial, device, categories, intents)
    print(f"Accuracy with LoRa Fine-tuning [Gemma 7B] on categories: {accuracy_category:.2f}%")
    print(f"Accuracy with LoRa Fine-tuning [Gemma 7B] on intents: {accuracy_intent:.2f}%")


# ==================================================
# Dispatch based on selected model
# ==================================================
if selected_model == "tinyllama":
    run_tinyllama()
elif selected_model == "llama_3_2_3b":
    run_llama_3_2_3b()
elif selected_model == "mistral_7b_v0_2":
    run_mistral_7b_v0_2()
elif selected_model == "gemma_7b":
    run_gemma_7b()
else:
    raise ValueError(f"Unknown model: {selected_model}")