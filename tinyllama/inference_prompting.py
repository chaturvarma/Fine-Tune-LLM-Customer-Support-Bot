def general_inference_tinyllama(df_test, generator, categories):
    correct = 0
    total = len(df_test)
    predictions = []

    system_prompt = {
        "role": "system",
        "content": (
            "You are an AI assistant that classifies user support queries into ONE and ONLY ONE of the following categories:\n\n"
            f"{chr(10).join(f'- {c}' for c in categories)}\n\n"
            "INSTRUCTIONS:\n"
            "- Read the user's query carefully.\n"
            "- Select the ONE category from the list that BEST matches the query.\n"
            "- ONLY return the category name, EXACTLY as it appears in the list.\n"
            "- DO NOT invent new categories."
            "- DO NOT include any extra words, explanations, punctuation, or formatting.\n\n"
        )
    }

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

        predicted_category = "UNKNOWN"
        for cat in categories:
            if cat in output:
                predicted_category = cat
                break

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

    return accuracy