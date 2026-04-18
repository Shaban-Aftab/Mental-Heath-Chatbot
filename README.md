# 💙 Mental Health Support Chatbot (Fine-Tuned AI)

This repository contains a professional End-to-End AI pipeline. It demonstrates the complete process of acquiring natural human dialogue, fine-tuning a base Large Language Model (`distilgpt2`) using PyTorch and Hugging Face's `SFTTrainer`, and deploying the resulting empathetic model to a Streamlit public interface.

## 🚀 Project Overview

Base AI models act entirely like autocomplete engines. To turn a model into an empathetic therapist, it must be taught the "rules" of conversation. 

This project downloads the official **Facebook Empathetic Dialogues dataset**, dynamically formats thousands of natural conversations with `<|user|>` and `<|bot|>` tags, and performs gradient descent (fine-tuning) on a Kaggle GPU environment to bake empathetic responses directly into the model's neural weights.

## 📂 Repository Contents

| File | Description |
|---|---|
| `mhs-chatbot.ipynb` | The master Kaggle Jupyter Notebook containing the full training pipeline (Dataset Extraction, Tokenizer formatting, `SFTConfig`, and GPU execution). |
| `app.py` | The frontend UI built in Streamlit. It fetches the fine-tuned model securely from the Hugging Face hub and manages user sessions. |
| `requirements.txt` | Dependency configuration for deploying the frontend securely to Streamlit Community Cloud. |
| `empatheticdialogues.tar.gz` | The raw untouched dataset archive obtained directly from Facebook servers, ensuring reproducible data ingestion without API deprecation errors. |

## 🛠️ The Architecture Stack
* **Deep Learning:** `torch`, `transformers`, `trl` (Transformer Reinforcement Learning)
* **Data Processing:** `pandas`, `datasets`, `tarfile`
* **Cloud GPU Backend:** Kaggle (Nvidia T4 x2)
* **Model Hosting:** Hugging Face Hub (Bypassing GitHub's 100MB file limit)
* **Frontend UI:** Streamlit 

## 🚨 Safety & Usage Warning
This chatbot was fine-tuned for educational purposes in prompt engineering, formatting constraints (e.g. `repetition_penalty` fixing), and LLM deployment. **It is an AI experiment and is completely unqualified to provide professional medical/psychiatric advice.** The `app.py` script strictly filters crisis keywords and redirects users to emergency crisis lifelines.
