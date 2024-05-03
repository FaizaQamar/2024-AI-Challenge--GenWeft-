# 2024 AI Challenge- GenWeft

# Hawaar Al-Zaki

## Table of contents
* [Introduction](#introduction)
* [Features](#features)
* [Technologies](#technologies)
* [Getting Started](#getting-started)
* [Evaluation](#evaluation)
* [Feedback](#feedback)
  
## Introduction 

Welcome to Hawaar al-Zaki! In today's digital landscape, accessing reliable information on Islamic teachings can be challenging. To address this, we've developed a multilingual question answering system that provides quick and accurate responses to inquiries about the Quran and Ahadith.

Our goal is to make Islamic knowledge more accessible and trustworthy, particularly for those not fluent in Arabic or Islamic scholarship. In this README, we'll give you a concise overview of Hawaar al-Zaki, including its purpose, target audience, competitive analysis, and evaluation metrics. 

## Features

- Multilingual question answering system using Quranic translations, Tafseer, and Ahadith from Sahah-e-Sittah
- Utilizes Mistral-7B language model and LangChain retrieval agent
- Accessible globally for general public, students of Islam, and newly reverted individuals
- RAG methodology ensures quick and accurate responses
- Potential for future enhancements including accuracy improvements and personalization features

## Technologies

- Programming Language: Python (version: 3.10.4)
- UI Framework: Flask (version: 13.0.0)
- Generative AI: Mistral7B
- Framework: langchain, huggingface_hub, torch

  
## Getting Started 
1. Run requirements.txt file to install dependencies:
  `pip install -r requirements.txt`
2. Obtain your Hugging Face API token from your account settings. User Access Tokens are recommended for authentication. Learn more [here]([url](https://huggingface.co/docs/hub/en/security-tokens)). 
3. Edit app.py and replace the token in line 25 with your Hugging Face API token.
   `os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token_here"`
4. Execute `python app.py` to start the Flask app.

## Evaluation 
Hawaar Al-Zaki demonstrates exceptional performance across various evaluation metrics, showing impressive precision, recall, and F1-score metrics, coupled with a remarkable human evaluation score of 91.8%. Our model distinguishes itself significantly from leading platforms, setting a new standard for efficiency, accessibility, and user engagement.
![Hawaar Al-Zaki Logo](https://github.com/FaizaQamar/2024-AI-Challenge--GenWeft-/blob/main/static/evaluation1.PNG)

## Feedback

We value your feedback! If you have any query or suggestion, you may contact us at [fqamar.dphd18seecs@seecs.edu.pk](mailto:fqamar.dphd18seecs@seecs.edu.pk).
