def general_prompting(df_test, generator, categories):
    correct = 0
    total = len(df_test)
    predictions = []
    
    # Structuring the prompt
    system_prompt = {
        "role": "system",
        "content": (
            "You are an AI assistant that classifies user support queries into one of the following categories:\n\n"
            f"{chr(10).join(f'- {c}' for c in categories)}\n\n"
            "Your task is to read the user's query and return only the most relevant category name from the list above.\n"
            "Do not provide explanations or additional text—only output the category name."
        )
    }
    
    # Iterating through each sample
    for step, (_, row) in enumerate(df_test.iterrows(), 1):
        user_prompt = {
            "role": "user",
            "content": row["instruction"]
        }

        assistant_prompt = {
            "role": "assistant",
            "content": "Labeled Category:"
        }

        messages = [system_prompt, user_prompt, assistant_prompt]
        prompt = "\n".join(f"<|{msg['role']}|>\n{msg['content']}" for msg in messages)

        output = generator(prompt, max_new_tokens=10)[0]["generated_text"]
        
        # Match one of the predefined categories in the output
        predicted_category = "UNKNOWN"
        for cat in categories:
            if cat in output:
                predicted_category = cat
                break
            
        # Compare prediction to the actual label
        actual_category = row["category"]
        is_correct = predicted_category == actual_category
        correct += is_correct
        predictions.append({
            "instruction": row["instruction"],
            "actual": actual_category,
            "predicted": predicted_category,
            "correct": is_correct
        })

    accuracy = correct / total * 100

    return predictions, total, correct, accuracy