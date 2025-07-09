from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import re

def format_chat(example, tokenizer):
    example["text"] = f"Instruction: {example['instruction']}\nLabeled Category: {example['category']}{tokenizer.eos_token}"
    return example

def tokenize_function(example, tokenizer, max_length=512):
    tokenized = tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_length)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def finetune_train(df_train, model, tokenizer):
    dataset = Dataset.from_pandas(df_train[["instruction", "category"]])
    dataset = dataset.map(lambda x: format_chat(x, tokenizer))
    
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./tinyllama-lora-supportbot",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        fp16=True,
        label_names=["labels"]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()


def predict_output(text, model, tokenizer, device):
    prompt = f"Instruction: {text}\nLabeled Category:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output

def extract_category(text, categories):
    match = re.search(r"Labeled Category:\s*(.*)", text, re.IGNORECASE)
    if match:
        label_section = match.group(1).strip().upper()

        categories_upper = [cat.upper() for cat in categories]

        for cat in sorted(categories_upper, key=len, reverse=True):
            if cat in label_section:
                return cat

    return "NOT FOUND"

def evaluate_model_on_test_set(df_test, categories, model, tokenizer, device):
    predictions = []
    correct = 0
    total = len(df_test)

    for _, row in df_test.iterrows():
        instruction = row["instruction"]
        true_category = row["category"].strip().upper()

        pred = predict_output(instruction, model, tokenizer, device)
        pred_category = extract_category(pred, categories)
        pred_category = pred_category.strip().upper()
        predictions.append(pred)

        print("Predicted text: ", pred)
        print("Predicted Category: ", pred_category)
        print("Correct Category: ", true_category)
        print("")

        if pred_category == true_category:
            correct += 1

    accuracy = (correct / total) * 100

    return predictions, correct, total, accuracy