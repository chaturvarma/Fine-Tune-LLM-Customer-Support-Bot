from data_splitting import load_and_split_dataset
from field_labels import categories
from model_loader import load_model_components
from model_loader_finetune import load_quantized_model
from inference_prompting import general_prompting
from custom_finetuning import generate_input_output_pair, custom_finetune_tinyllama
from lora_finetuning import finetune_train, evaluate_model_on_test_set
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = '../dataset.csv'
model_id = "Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct"
dataset_size = 5000 # No of samples to apply. Use full dataset (27k rows) by setting to None

# Load and split the dataset into training, validation, and test sets
df_train, df_val, df_test = load_and_split_dataset(dataset_path, dataset_size)

# Load the model, tokenizer, and generator
model, tokenizer, generator = load_model_components(model_id)
model_finetune, tokenizer_finetune, generator_finetune = load_quantized_model(model_id)

# Case 1: Performing inference using general prompting (no fine-tuning)
predictions, total, correct, accuracy = general_prompting(df_test, generator, categories)
print(f"Accuracy with general prompting: {accuracy:.2f}%")

# Case 2: Performing fine-tuning with LoRA
finetune_train(df_train, model_finetune, tokenizer_finetune)
predictions, correct, total, accuracy = evaluate_model_on_test_set(df_test, categories, model_finetune, tokenizer_finetune, device)
print(f"Accuracy with LoRa Fine-Tuning: {accuracy:.2f}%")

# Case 3: Performing custom fine-tuning with AdamW
custom_batch_size = 4
custom_lr = 1e-5
custom_decay = 0.01
custom_epochs = 5
custom_pair_data = generate_input_output_pair(df_train, tokenizer_finetune)
custom_finetune_tinyllama(
    df_train, model_finetune, tokenizer_finetune, custom_pair_data, 
    device, custom_batch_size, custom_lr, custom_decay, custom_epochs
)

# Case 4: Performing fine-tuning using reinforcement learning