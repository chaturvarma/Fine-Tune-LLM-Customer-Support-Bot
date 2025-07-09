import torch
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

def generate_input_output_pair(df_train, tokenizer, max_length=512):
    prompts = [
        [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": "Labeled Category:"}
        ]
        for instruction in df_train['instruction']
    ]
    responses = df_train['category'].tolist()

    chat_templates = tokenizer.apply_chat_template(prompts, continue_final_message=True, tokenize=False)
    full_response_text = [
        (chat_template + " " + target_response + tokenizer.eos_token)
        for chat_template, target_response in zip(chat_templates, responses)
    ]

    input_ids_tokenized = tokenizer(
        full_response_text,
        return_tensors="pt",
        add_special_tokens=False,
        padding="max_length",
        max_length=max_length,
        truncation=True
    )["input_ids"]

    labels_tokenized = tokenizer(
        [" " + response + tokenizer.eos_token for response in responses],
        add_special_tokens=False,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=input_ids_tokenized.shape[1]
    )["input_ids"]

    labels_tokenized_fixed = torch.where(labels_tokenized != tokenizer.pad_token_id, labels_tokenized, -100)
    labels_tokenized_fixed[:, -1] = -100

    input_ids_tokenized_left_shifted = input_ids_tokenized[:, :-1]
    labels_tokenized_right_shifted = labels_tokenized_fixed[:, 1:]

    attention_mask = input_ids_tokenized_left_shifted != tokenizer.pad_token_id

    return {
        "input_ids": input_ids_tokenized_left_shifted,
        "attention_mask": attention_mask,
        "labels": labels_tokenized_right_shifted
    }

def custom_finetune_tinyllama(df_train, model, tokenizer, data, device, batch_size=2, lr=1e-5, decay=0.01, epochs=3):
    dataset = TensorDataset(data["input_ids"], data["attention_mask"], data["labels"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=decay)

    for epoch in range(epochs):
        for input_ids, attention_mask, labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if torch.isnan(loss):
               continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"Loss: {loss.item():.4f}")