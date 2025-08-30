# Fine-Tuning Open Source LLMs for Customer Support Bot

In this project, I explored several open-source language models including **LLaMA**, **Mistral**, **Gemma**, and **TinyLLaMA** to tackle the task of predicting category and support intent for customer support queries on bitext data. Moving beyond basic prompt-based inference, I implemented fine-tuning using **LoRA (Low-Rank Adaptation)**, which significantly enhanced model performance. Through iterative tuning and evaluation, I was able to improve classification accuracy from an initial 20% to over 50–60%, ready for real-world support query purposes.

This project is currently part of a larger project where I am developing a fully functional support chatbot with live chat capabilities. The system uses the company’s knowledge base to generate contextually relevant responses to user queries. If the user requests to speak with a human agent, the models developed in this project can be used to identify the appropriate support category and intent, transferring the conversation to the right support representative

## Introduction

In current support systems, customers are often required to navigate multiple options, either by clicking through a series of buttons or pressing numbers on a phone (e.g., "Press 1 for billing, Press 2 for technical support") to reach the right department. While this was acceptable a decade ago, today's AI capabilities allow us to significantly reduce such manual effort. By automatically understanding and classifying the query using language models, the support experience becomes more seamless. This also helps avoid common issues like misrouted calls or chats due to incorrect user selections, ultimately enhancing both efficiency and customer satisfaction.

The aim of this project is to use a customer's initial support query to automatically predict the relevant support category (such as "ORDER", "ACCOUNT") and intent (such as "cancel_order", "get_invoice"). The goal is to route the query to the appropriate support department with minimal user friction, enabling faster and more accurate responses. This reduces and streamlines the process as previously discussed, minimizing the need for manual input or navigation and improving the overall customer support experience.

In this project, I experimented with multiple open-source models including LLaMA, Mistral, Gemma, and TinyLLaMA to make the solution flexible for different use cases. While the experiments were heavily based on a bitext dataset, the same fine-tuning approach can be applied to your own data for similar tasks.

For clarity and modularity:

- Each model has its own directory `model_name/` containing fine-tuning and inference logic specific to that model.
- The `data/` folder contains the dataset along with its structure and fields.
- Prediction outputs are stored as JSON files in the `json_outputs/` directory.
- The `models/` directory contains LoRA adapter weights trained for each model, which can be merged with the base Hugging Face model for usage.
- Training and testing notebooks are located in the `notebooks/` folder, so that you can directly import them and run it for yourself in Kaggle, Google Colab, or a local Jupyter environment.

## Technologies Used

### Libraries & Frameworks

- **Transformers (Hugging Face)** — For model loading, training, and inference
- **PEFT (Parameter-Efficient Fine-Tuning)** — For efficient fine-tuning using LoRA
- **Bitsandbytes** — For 4-bit and 8-bit quantized model support
- **Datasets** — For loading and preprocessing training data
- **Accelerate** — To simplify multi-GPU and mixed precision training
- **Torch (PyTorch)** — Framework used for model training and integration with Hugging Face ecosystem

### Models

The following open-source language models were explored and fine-tuned in this project:

- **LLaMA 3.2 3B Instruct** - Meta’s LLaMA 3 model with 3.2 billion parameters
- **Mistral 7B Instruct v0.2** - Robust 7B parameter instruction-tuned model and multilingual support
- **Gemma 7B IT** - Google’s 7B parameter instruction-tuned model
- **TinyLLaMA** (Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct) - Lightweight 1.1B parameter model

## Dataset

This project uses a bitext customer support dataset, available at `data/dataset.csv`, containing 26,872 rows. Each row includes the following columns:

- **instruction**: The user's support query (e.g., a question or request).
- **category**: The high-level category the query belongs to.
- **intent**: The specific user intent within that category.
- **response**: A pre-written agent response (not used in this project).

Since the aim of this project is to route the support query to the appropriate department, only the instruction, category, and intent columns are used. The response column and any other data are excluded from training and inference.

Here is a sample entry from the dataset:

```json
{
  "instruction": "i need assistance canceling purchase {{Order Number}}",
  "category": "ORDER",
  "intent": "cancel_order"
}
```

Each instruction falls into one of the following predefined categories and intents:

```python
categories = [
    "ORDER", "SHIPPING", "CANCEL", "INVOICE",
    "PAYMENT", "REFUND", "FEEDBACK", "CONTACT",
    "ACCOUNT", "DELIVERY", "SUBSCRIPTION"
]
```

```python
intents = {
    "ORDER": ["cancel_order", "change_order", "place_order", "track_order"],
    "SHIPPING": ["change_shipping_address", "set_up_shipping_address"],
    "CANCEL": ["check_cancellation_fee"],
    "INVOICE": ["check_invoice", "get_invoice"],
    "PAYMENT": ["check_payment_methods", "payment_issue"],
    "REFUND": ["check_refund_policy", "get_refund", "track_refund"],
    "FEEDBACK": ["complaint", "review"],
    "CONTACT": ["contact_customer_service", "contact_human_agent"],
    "ACCOUNT": ["create_account", "delete_account", "edit_account", "recover_password", "registration_problems", "switch_account"],
    "DELIVERY": ["delivery_options", "delivery_period"],
    "SUBSCRIPTION": ["newsletter_subscription"]
}
```

## Implementation

Since each model varies slightly in terms of prompt structure and usage, the relevant implementation files for each model are organized under their respective folders, named as `model_name/`

The core utility scripts are placed at the root level:

- **model_loader.py** — Handles loading of models, tokenizers, and generation pipeline with support for 4-bit quantization (when GPU is available).
- **data_splitting.py** — Splits the dataset into training and testing sets.
- **main.py** — Central entry point that imports all modules and runs the training, inference, and evaluation processes.

Here is a simple overview of the entire workflow:

1. The bitext dataset `data/dataset.csv` is loaded and split into train and validation sets using data_splitting.py.

2. Hugging Face models are loaded using `model_loader.py`. If a GPU is available, 4-bit quantization via bitsandbytes is applied for faster and memory-efficient execution. It returns the model, tokenizer, and a text generation pipeline (generator).

3. The generator is used to predict the support category and intent using each model's `inference_prompting.py`, based on the `instruction` field in the validation set. This step is used to test baseline accuracy through inference without fine-tuning.

4. For all model sizes, including LLaMA, Mistral, and TinyLLaMA, two separate models were fine-tuned: one for **category classification** and one for **intent classification**. Each model was trained independently using its respective `lora_finetuning.py`, with LoRA adapters applied to the base model

5. The fine-tuned models were evaluated on the validation set to compute the final classification accuracy for both category and intent prediction.

6. The main.py script demonstrates end-to-end usage of all supported models. You can customize it by commenting out unused models or retaining only the one you want to test or deploy.

## Results

The table below compares the inference accuracy (zero-shot prompting) with accuracy after LoRA-based fine-tuning, evaluated on the test dataset:

| Model                 | Category (Before -> After) | Intent (Before -> After) |
| --------------------- | -------------------------- | ------------------------ |
| TinyLlama             | 22% -> 55%                 | 14% -> 30%               |
| Llama 3.2 3B Instruct | 54% -> 72%                 | 43% -> 60%               |
| Mistral 7B v0.2       | 70% -> 85%                 | 63% -> 73%               |
| Gemma 7B IT           | 46% -> 66%                 | 26% -> 40%               |

## Conclusion

Different models offer trade-offs between performance, accuracy, and resource requirements, making them suitable for different types of use cases:

- TinyLLaMA is ideal for lightweight applications where computational resources are limited and high accuracy is not critical

- LLaMA 3 and Mistral 7B are better suited for use cases where high-quality predictions are essential and sufficient compute is available

- Gemma 7B strikes can be a good fit for privacy-focused or on-premise deployments where Google’s tooling ecosystem is already in use.

Overall, Mistral stands out as my choice because it provides high accuracy, comparatively faster load times, and can be adapted to multilingual environments

## Future Work

In the next phase, I plan to explore reinforcement learning–based fine-tuning techniques and experiment with additional Hugging Face models to further optimize response accuracy and adaptability in dynamic customer support environments
