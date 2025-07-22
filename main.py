from data_splitting import load_and_split_dataset
from data.field_labels import categories
from model_loader import model_loader
import torch

from tinyllama.inference_prompting import general_prompting_tinyllama
from tinyllama.custom_finetuning import generate_input_output_pair_tinyllama, custom_finetune_tinyllama
from tinyllama.lora_finetuning import finetune_train_tinyllama, evaluate_model_on_test_set_tinyllama

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = './data/dataset.csv'
dataset_size = 10000 # No of samples to apply. Use full dataset (27k rows) by setting to None

# Load and split the dataset
df_train, df_val, df_test = load_and_split_dataset(dataset_path, dataset_size)


# ==================================================
# Model 1: TinyLlama
# ==================================================
model_id = "Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct"
model, tokenizer, generator = model_loader(model_id)

# Case 1: Performing inference using general prompting (no fine-tuning)
predictions, total, correct, accuracy = general_prompting_tinyllama(df_test, generator, categories)
print(f"Accuracy with General prompting [TinyLlama]: {accuracy:.2%}")

# Case 2: Performing fine-tuning with LoRA
finetune_train_tinyllama(df_train, model, tokenizer)
predictions, correct, total, accuracy = evaluate_model_on_test_set_tinyllama(df_test, categories, model, tokenizer, device)
print(f"Accuracy with LoRa Fine-Tuning [TinyLlama]: {accuracy:.2f}%")

# Case 3: Performing custom fine-tuning with AdamW
custom_batch_size = 4
custom_lr = 1e-5
custom_decay = 0.01
custom_epochs = 5
custom_pair_data = generate_input_output_pair_tinyllama(df_train, tokenizer)
custom_finetune_tinyllama(
    df_train, model, tokenizer, custom_pair_data,
    device, custom_batch_size, custom_lr, custom_decay, custom_epochs
)

# Case 4: Performing fine-tuning using reinforcement learning