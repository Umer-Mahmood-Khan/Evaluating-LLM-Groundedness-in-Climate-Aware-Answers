# 🧠 RAG vs. ChatGPT: Evaluating LLM Groundedness in Climate-Aware Answers

This project compares the performance of a pure ChatGPT model vs a RAG-augmented pipeline in answering climate-impact questions on cotton crops in Pakistan.

## 🚀 Features
- LangChain-based pipeline
- FAISS vector search
- OpenAI GPT-3.5 or GPT-4 models
- Evaluation on metrics like Accuracy, Clarity, and Use of Retrieved Content
- Streamlit dashboard for visualization

## 📊 Example Use Case
> "What is the impact of a 1°C temperature rise on cotton lint yield in Pakistan?"

## 🏗 Screenshots & Visuals

![Architecture Flowchart](images/screencaption.png)  
*Figure: Streamlit Dashboard*}  

---
## 🛠 Tech Stack
- Python 3.10
- LangChain
- OpenAI API
- FAISS
- Streamlit + Plotly

## 🔧 Setup

```bash
conda env create -f environment.yaml
conda activate rag_evaluation_env
