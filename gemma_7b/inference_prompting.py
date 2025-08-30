import re

# Format the prompt for category classification
def format_category_prompt(instruction, categories):
    system_prompt = {
        "role": "system",
        "content": (
            "You are an AI assistant that classifies a user support query into one of the predefined categories.\n\n"
            "### Available Categories:\n"
            f"{chr(10).join(f'- {c}' for c in categories)}\n\n"
            "### Instructions:\n"
            "1. Carefully read the user’s query.\n"
            "2. Choose the **one category** from the list that best matches the query.\n"
            "3. Return only the category name — exactly as shown in the list.\n\n"
            "### Rules:\n"
            "- Do not invent new categories.\n"
            "- Do not include explanations, punctuation, or any extra words.\n"
            "- Output just the category name."
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
            "You are an AI assistant that selects the correct intent for a user query based on a given category.\n\n"
            "### Available Intents:\n"
            f"{chr(10).join(f'- {intent}' for intent in intents_for_category)}\n\n"
            "### Instructions:\n"
            "1. Read the user query carefully.\n"
            "2. Pick the **single intent** from the list that best matches the query.\n"
            "3. Return only the intent name — exactly as written above.\n\n"
            "### Rules:\n"
            "- Do not make up new intents.\n"
            "- Do not add any extra text, punctuation, or explanations.\n"
            "- Only return the intent name."
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
def general_inference_gemma_7b(df_test, generator, categories, intents):
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