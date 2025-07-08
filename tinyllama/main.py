from data_splitting import load_and_split_dataset
from field_labels import categories
from model_loader import load_model_components
from inference_prompting import general_prompting

dataset_path = '../dataset.csv'
model_id = "Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct"
# Number of samples to use for quick. Use full dataset (27k rows) by setting to None
dataset_size = 500

# Load and split the dataset into training, validation, and test sets
df_train, df_val, df_test = load_and_split_dataset(dataset_path, dataset_size)

# Load the model, tokenizer, and generator
model, tokenizer, generator = load_model_components(model_id)

# Case 1: Perform inference using general prompting (no fine-tuning)
predictions, total, correct, accuracy = general_prompting(df_test, generator, categories)
print(f"Accuracy with general prompting: {accuracy:.2f}%")