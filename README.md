# 🧠 LLM Evaluation Lab

A hands-on experimental framework for analyzing and evaluating Large Language Model (LLM) behavior under different conditions such as multilingual prompts, prompt perturbations, retrieval augmentation, and model variability.

---

## 🎯 Project Goal

This project explores how LLM outputs change when:
- Prompts are translated into different languages
- Input phrasing is slightly modified
- Model randomness (temperature) is introduced
- Different LLM sources are used (local vs API)

The focus is not on building a production application, but on understanding and measuring LLM behavior as a probabilistic system.

---

## 🧩 Core Idea

Traditional evaluation metrics like Accuracy or F1-score are not sufficient for generative models.

This project introduces a key concept:

### 🔍 Intent Drift

Intent Drift is defined as:

> The divergence between the semantic meaning of an input prompt and the behavioral consistency of the model’s output under different conditions.

We measure:
- Semantic similarity between inputs
- Consistency of outputs across repeated runs
- Sensitivity to language and prompt changes
- Model stability under randomness

---

## 🏗️ System Architecture

The system follows an experimental pipeline:

Input Prompt (multilingual / perturbed)
↓
LLM (API or local model via Ollama)
↓
Generated Response
↓
Evaluation Layer
- Semantic Similarity
- Intent Drift Score
- Variability Metrics
↓
Logging & Visualization

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Sentence Transformers
- Matplotlib
- Seaborn
- FAISS (vector similarity search)
- OpenAI API (optional)
- Ollama (local LLM execution)

---

## 📁 Project Structure

llm-evaluation-lab/
├── src/
│   ├── llm/
│   │   ├── ollama_client.py
│   │   ├── api_client.py
│   ├── evaluation/
│   │   ├── similarity.py
│   │   ├── intent_drift.py
│   │   ├── metrics.py
│   │   ├── variability.py
│   ├── rag/
│   │   ├── retriever.py
│   │   ├── pipeline.py
│   │   ├── chunking.py
│   ├── experiments/
│   │   ├── multilingual_tests.py
│   │   ├── temperature_tests.py
│   │   ├── rag_tests.py
│   ├── utils/
│   │   ├── logger.py
│   │   ├── config.py
│   │   ├── embeddings.py
│
├── notebooks/
│   ├── day1_variability.ipynb
│   ├── day2_intent_drift.ipynb
│   ├── day3_rag_experiments.ipynb
│   ├── day4_final_system.ipynb
│
├── experiments/
│   ├── logs.json
│   ├── drift_results.csv
│   ├── similarity_scores.csv
│
├── data/
│   ├── documents/
│   ├── prompts/
│
├── visuals/
│   ├── drift_plots.png
│   ├── similarity_heatmap.png

---

## 🔬 Key Experiments

### Day 1 — Variability Exploration
- Repeated prompt execution
- Output randomness analysis
- Stability measurement across runs

### Day 2 — Intent Drift Analysis
- Multilingual prompt testing
- Semantic similarity vs behavioral divergence
- Introduction of Intent Drift metric

### Day 3 — RAG Sensitivity
- Retrieval-Augmented Generation experiments
- Context injection impact on output stability
- Evaluation of retrieval noise effects

### Day 4 — Integrated Evaluation System
- Unified evaluation pipeline
- Combination of all metrics
- Local vs API model comparison

---

## 📊 Metrics Defined

- Semantic Similarity → cosine similarity of embeddings
- Variability Score → output instability across repeated runs
- Intent Drift Score → divergence between input meaning and output behavior
- Retrieval Sensitivity → impact of external context on response stability

---

## ⚙️ How to Run

Install dependencies:

pip install -r requirements.txt

Run experiments:

python src/experiments/multilingual_tests.py

Or launch notebooks:

jupyter notebook notebooks/

---

## 💡 Key Insight

Large Language Models should not only be evaluated on correctness, but also on:

- Stability across repeated runs
- Sensitivity to input variations
- Semantic consistency across languages
- Robustness under retrieval augmentation

This project treats LLMs as probabilistic systems rather than deterministic tools.

---

## 🚀 Future Improvements

- Add Streamlit dashboard for real-time visualization
- Extend Intent Drift into a formal evaluation metric
- Compare multiple LLM providers side-by-side
- Add experiment tracking (e.g., MLflow integration)
- Deploy as an interactive research tool

---

## 👤 Author

Data Scientist focused on:
- Machine Learning systems
- LLM evaluation frameworks
- Experimental AI workflows
- Applied data science in consulting environments