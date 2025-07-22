# Fine-Tuning Open Source LLMs for Customer Support Bot

In this project, I explored the use of multiple open-source language models—starting with **TinyLlama**—to fine-tune a customer support AI that can classify user queries into predefined support categories. 

The goal was to enhance the accuracy and usefulness of a support bot by leveraging fine-tuning techniques, particularly **LoRA (Low-Rank Adaptation)**, to improve its classification capabilities beyond general prompting.

## 🛠️ Tech Stack

- **Transformers (Hugging Face)** — For model loading, training, and inference
- **PEFT (Parameter-Efficient Fine-Tuning)** — For efficient fine-tuning using LoRA
- **Bitsandbytes** — For 4-bit and 8-bit quantized model support
- **Datasets** — For loading and preprocessing training data
- **TinyLlama (Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct)** — Base model used
- **Accelerate** — To simplify multi-GPU and mixed precision training
- **Torch (PyTorch)** — Core deep learning framework for model training

## 📊 Dataset

The **Bitext Customer Support Dataset** was used for training and evaluation. It consists of real-world customer support queries, each labeled with a specific support category. The dataset is suitable for multi-class classification and provides a good benchmark for real-world support bot use cases.

The dataset contains **26.9k rows** and **11 unique categories**, as specified in the `category` column. The input data for each example is taken from the `instruction` column, which contains the actual customer support query


## 📈 Results

| Method                     | Accuracy |
|----------------------------|----------|
| General Prompting | ~22%     |
| LoRA Fine-Tuned Model         | ~50–60%  |

Fine-tuning with LoRA showed a significant improvement over zero-shot prompting. The model became noticeably more consistent and accurate in identifying the correct support category after fine-tuning

## 🧪 Future Work

I am currently experimenting with **full fine-tuning** using the **AdamW optimizer**, aiming for even better performance. In the next phase, I plan to explore **reinforcement learning based fine-tuning techniques** to further optimize model responses in dynamic customer support environments
