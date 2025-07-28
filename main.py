from data_splitting import load_and_split_dataset
from data.field_labels import categories, intents
from model_loader import model_loader
from huggingface_hub import login
from tinyllama.inference_prompting import general_inference_tinyllama
from tinyllama.lora_finetuning import finetune_category_tinyllama, finetune_intent_tinyllama, evaluate_model_tinyllama
from llama_3_2_3b.inference_prompting import general_inference_llama_3_2_3b
from llama_3_2_3b.lora_finetuning import finetune_category_llama_3_2_3b, finetune_intent_llama_3_2_3b, evaluate_model_llama_3_2_3b

# Hugging face requires gated access for some models
hf_token = "your_key"
login(token=hf_token)

dataset_path = './data/dataset.csv'
dataset_size = 10000 # No of samples to apply. Use full dataset (27k rows) by setting to None

# Load and split the dataset
df_train, df_val, df_test = load_and_split_dataset(dataset_path, dataset_size)


# ==================================================
# Model 1: TinyLlama
# ==================================================
model_id = "Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct"

# Load the initial model, tokenizer, and text generation pipeline
model_initial, tokenizer_initial, generator, device = model_loader(model_id)

# Case 1: Performing inference using general prompting (no fine-tuning)
accuracy_category, accuracy_intent, predictions_inference = general_inference_tinyllama(df_test, generator, categories, intents)
print(f"Accuracy without Fine-tuning [TinyLlama] on categories: {accuracy_category:.2f}%")
print(f"Accuracy without Fine-tuning [TinyLlama] on intents: {accuracy_intent:.2f}%")

# Case 2: Performing fine-tuning with LoRA
model_category_tinyllama = finetune_category_tinyllama(df_train, model_initial, tokenizer_initial, device)
model_intent_tinyllama = finetune_intent_tinyllama(df_train, model_initial, tokenizer_initial, device, intents)
accuracy_category, accuracy_intent, predictions_finetune = evaluate_model_tinyllama(df_test, model_category_tinyllama, model_intent_tinyllama, tokenizer_initial, device, categories, intents)
print(f"Accuracy with LoRa Fine-tuning [TinyLlama] on categories: {accuracy_category:.2f}%")
print(f"Accuracy with LoRa Fine-tuning [TinyLlama] on intents: {accuracy_intent:.2f}%")


# ==================================================
# Model 2: Llama
# ==================================================
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Load the initial model, tokenizer, and text generation pipeline
model_initial, tokenizer_initial, generator, device = model_loader(model_id)

# Case 1: Performing inference using general prompting (no fine-tuning)
accuracy_category, accuracy_intent, predictions_inference = general_inference_llama_3_2_3b(df_test, generator, categories, intents)
print(f"Accuracy without Fine-tuning [Llama 3.2 3B] on categories: {accuracy_category:.2f}%")
print(f"Accuracy without Fine-tuning [Llama 3.2 3B] on intents: {accuracy_intent:.2f}%")

# Case 2: Performing fine-tuning with LoRA [Llama 3.2 3B]: {accuracy:.2f}%")
model_category_llama_3_2_3b = finetune_category_llama_3_2_3b(df_train, model_initial, tokenizer_initial, device)
model_intent_llama_3_2_3b = finetune_intent_llama_3_2_3b(df_train, model_initial, tokenizer_initial, device, intents)
accuracy_category, accuracy_intent, predictions_finetune = evaluate_model_llama_3_2_3b(df_test, model_category_llama_3_2_3b, model_intent_llama_3_2_3b, tokenizer_initial, device, categories, intents)
print(f"Accuracy with LoRa Fine-tuning [Llama 3.2 3B] on categories: {accuracy_category:.2f}%")
print(f"Accuracy with LoRa Fine-tuning [Llama 3.2 3B] on intents: {accuracy_intent:.2f}%")

# ==================================================
# Model 3: Mistral
# ==================================================


# ==================================================
# Model 4: Gemma
# ==================================================