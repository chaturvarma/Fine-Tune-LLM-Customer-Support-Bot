import re
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Format the training data for category classification
def format_chat_category(example, tokenizer):
    example["text"] = (
        f"Instruction: {example['instruction']}\n"
        f"Labeled Category: {example['category']}{tokenizer.eos_token}"
    )
    return example

# Format the training data for intent classification (includes list of possible intents for that category)
def format_chat_intent(example, tokenizer, intents):
    category = example["category"]
    category_intents = intents.get(category.upper(), [])
    intents_str = ", ".join(category_intents)
    example["text"] = (
        f"Instruction: {example['instruction']}\n"
        f"Category: {category}\n"
        f"Possible Intents: {intents_str}\n"
        f"Labeled Intent: {example['intent']}{tokenizer.eos_token}"
    )
    return example

# Tokenize the formatted text for model input
def tokenize_function(example, tokenizer, max_length=512):
    tokenized = tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_length)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Fine-tune for category classification
def finetune_category_llama_3_2_3b(df_train, model, tokenizer, device):
    dataset = Dataset.from_pandas(df_train[["instruction", "category"]])
    dataset = dataset.map(lambda x: format_chat_category(x, tokenizer))
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir="./tinyllama-lora-category-classifier",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=3,
        logging_steps=100,
        save_strategy="epoch",
        report_to="none",
        fp16=True,
        label_names=["labels"]
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset,
        tokenizer=tokenizer, data_collator=data_collator
    )
    
    trainer.train()
    return model

# Fine-tune for intent classification
def finetune_intent_llama_3_2_3b(df_train, model, tokenizer, device, intents):
    dataset = Dataset.from_pandas(df_train[["instruction", "category", "intent"]])
    dataset = dataset.map(lambda x: format_chat_intent(x, tokenizer, intents))
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir="./tinyllama-lora-intent-classifier",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=3,
        logging_steps=100,
        save_strategy="epoch",
        report_to="none",
        fp16=True,
        label_names=["labels"]
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset,
        tokenizer=tokenizer, data_collator=data_collator
    )
    
    trainer.train()
    return model

# Generate category prediction from instruction
def predict_output_category(text, model, tokenizer, device):
    prompt = (
        f"Instruction: {text}\n"
        f"Labeled Category:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate intent prediction using predicted category and instruction
def predict_output_intent(text, predicted_category, model, tokenizer, device, intents):
    possible_intents = intents.get(predicted_category.upper(), [])
    if not possible_intents:
        possible_intents = [intent for sub in intents.values() for intent in sub]
    
    intents_str = ", ".join(possible_intents)
    prompt = (
        f"Instruction: {text}\n"
        f"Category: {predicted_category}\n"
        f"Possible Intents: {intents_str}\n"
        f"Labeled Intent:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract the predicted category string from the model output
def extract_category(text, categories):
    match = re.search(r"Labeled Category:\s*(.*)", text, re.IGNORECASE)

    if match:
        label_section = match.group(1).strip().upper()
    
        for cat in categories:
            if cat in label_section:
                return cat
                
    return "NOT FOUND"

# Extract the predicted intent string from the model output
def extract_intent(text, intents):
    match = re.search(r"Labeled Intent:\s*(.*)", text, re.IGNORECASE)
    
    if match:
        label_section = match.group(1).strip().lower()
        
        for intent in intents:
            if intent in label_section:
                return intent
                
    return "NOT FOUND"

# Evaluate both models on a test dataset
def evaluate_model_llama_3_2_3b(df_test, model_category, model_intent, tokenizer, device, categories, intents):
    predictions = []
    correct_category = 0
    correct_intent = 0
    total = len(df_test)
    
    all_intents_flat = [intent.lower() for sublist in intents.values() for intent in sublist]
    
    for _, row in df_test.iterrows():
        instruction = row["instruction"]
        actual_category = row["category"].strip().upper()
        actual_intent = row["intent"].strip().lower()
        
        output_category = predict_output_category(instruction, model_category, tokenizer, device)
        pred_category = extract_category(output_category, categories).strip().upper()

        category_intents = intents.get(pred_category, [])

        if len(category_intents) == 1:
            pred_intent = category_intents[0].strip().lower()
        else:
            output_intent = predict_output_intent(instruction, pred_category, model_intent, tokenizer, device, intents)
            pred_intent = extract_intent(output_intent, all_intents_flat).strip().lower()
        
        if pred_category == actual_category:
            correct_category += 1
        if pred_intent == actual_intent:
            correct_intent += 1
            
        predictions.append({
            "instruction": instruction,
            "actual_category": actual_category,
            "predicted_category": pred_category,
            "actual_intent": actual_intent,
            "predicted_intent": pred_intent,
        })
        
    accuracy_category = (correct_category / total) * 100
    accuracy_intent = (correct_intent / total) * 100
    
    return accuracy_category, accuracy_intent, predictions