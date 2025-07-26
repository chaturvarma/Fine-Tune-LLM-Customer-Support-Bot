import re

# Format the prompt for category classification
def format_category_prompt(instruction, categories):
    system_prompt = {
        "role": "system",
        "content": (
            "You are an AI assistant that classifies user support queries into ONE and ONLY ONE of the following categories:\n\n"
            f"{chr(10).join(f'- {c}' for c in categories)}\n\n"
            "INSTRUCTIONS:\n"
            "- Read the user's query carefully.\n"
            "- Select the ONE category from the list that BEST matches the query.\n"
            "- ONLY return the category name, EXACTLY as it appears in the list.\n"
            "- DO NOT invent new categories.\n"
            "- DO NOT include any extra words, explanations, punctuation, or formatting.\n\n"
        )
    }

    user_prompt = {
        "role": "user",
        "content": instruction
    }

    assistant_prompt = {
        "role": "assistant",
        "content": "Labeled Category:"
    }
     
    return [system_prompt, user_prompt, assistant_prompt]

# Format the prompt for intent classification
def format_intent_prompt(instruction, intents_for_category):
    system_prompt = {
        "role": "system",
        "content": (
            "You are an AI assistant that selects the most appropriate intent for a user query.\n\n"
            "Available intents:\n"
            f"{chr(10).join(f'- {intent}' for intent in intents_for_category)}\n\n"
            "INSTRUCTIONS:\n"
            "- Read the user's query carefully.\n"
            "- Select the ONE intent from the list that BEST matches the query.\n"
            "- ONLY return the intent name, EXACTLY as it appears in the list.\n"
            "- DO NOT invent new intents.\n"
            "- DO NOT include any extra words, explanations, punctuation, or formatting.\n\n"
        )
    }

    user_prompt = {
        "role": "user",
        "content": instruction
    }

    assistant_prompt = {
        "role": "assistant",
        "content": "Labeled Intent:"
    }

    return [system_prompt, user_prompt, assistant_prompt]

# Predict the category of a user query
def predict_category(instruction, categories, generator):
    messages = format_category_prompt(instruction, categories)
    prompt = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)
    output = generator(prompt, max_new_tokens=10)[0]["generated_text"]

    # Extract category name from the generated output
    match = re.search(r"Labeled Category:\s*(.*)", output, re.IGNORECASE)
    if match:
        label_section = match.group(1).strip().upper()
        
        for cat in categories:
            if cat in label_section:
                return cat
            
    return "NOT FOUND"

# Predict the intent of a query based on the predicted category
def predict_intent(predicted_category, instruction, intents, generator):
    if predicted_category not in intents:
        all_intents = [intent for sublist in intents.values() for intent in sublist]
        intents_for_cat = all_intents
    else:
        intents_for_cat = intents[predicted_category]
        
    # If there's only one possible intent, return it directly    
    if len(intents_for_cat) == 1:
        return intents_for_cat[0]

    messages = format_intent_prompt(instruction, intents_for_cat)
    prompt = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)
    output = generator(prompt, max_new_tokens=15)[0]["generated_text"]

    # Extract intent from the generated output
    match = re.search(r"Labeled Intent:\s*(.*)", output, re.IGNORECASE)
    if match:
        label_section = match.group(1).strip().lower()
        
        for intent in intents_for_cat:
            if intent in label_section:
                return intent

    return "NOT FOUND"

# Evaluate category and intent predictions on test data
def general_inference_tinyllama(df_test, generator, categories, intents):
    correct_category = 0
    correct_intent = 0
    total = len(df_test)
    predictions = []

    for _, row in df_test.iterrows():
        instruction = row["instruction"]
        actual_category = row["category"].upper()
        actual_intent = row["intent"].lower()

        predicted_category = predict_category(instruction, categories, generator)
        is_correct_category = predicted_category == actual_category
        correct_category += is_correct_category

        predicted_intent = predict_intent(predicted_category, instruction, intents, generator)
        is_correct_intent = predicted_intent == actual_intent
        correct_intent += is_correct_intent

        predictions.append({
            "instruction": instruction,
            "actual_category": actual_category,
            "predicted_category": predicted_category,
            "actual_intent": actual_intent,
            "predicted_intent": predicted_intent,
        })

    accuracy_category = correct_category / total * 100
    accuracy_intent = correct_intent / total * 100

    return accuracy_category, accuracy_intent, predictions