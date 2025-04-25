# 🍽️ Yelp Recommendation Chatbot

A GPT-2 based conversational assistant that provides restaurant recommendations based on real Yelp reviews. Built with fine-tuning and reinforcement learning, and tracked using MLflow.

---

## 🧠 Overview

This project explores how large language models can be adapted for task-specific recommendation using:

- **GPT-2 Fine-Tuning** on structured Yelp data
- **Reinforcement Learning (Policy Gradient)** for targeted reward optimization
- **Custom Evaluation Metrics**: Relevance, Completeness, Semantic Similarity
- **MLflow**: Logging all training metrics and experiments

---

## 🗂️ Repository Structure

```
restaurant-chatbot/
├── src/
│   ├── models/
│   │   └── chatbot.py             # GPT-2 chatbot class
│   ├── training/
│   │   ├── train_baseline.py      # Fine-tuning
│   │   └── train_rl.py            # Reinforcement learning
│   └── evaluation/
│   │   └── evaluate_responses.py  # Custom metrics
├── data/
│   ├── train.json
│   └── val.json
├── download_data.py              # Yelp preprocessor
├── train_in_colab.ipynb          # End-to-end training
└── README.md
```

---

## 📦 Setup & Preprocessing

### 1. Get Yelp Data

Download from [Yelp Dataset](https://www.yelp.com/dataset)

- `yelp_academic_dataset_review.json`
- `yelp_academic_dataset_business.json`

### 2. Process Data

```bash
python download_data.py \
  --input_file data/yelp_academic_dataset_review.json \
  --business_file data/yelp_academic_dataset_business.json \
  --output_dir data \
  --max_reviews 10000
```

Generates:

- `data/train.json`
- `data/val.json`

3. Open `train_in_colab.ipynb` in Google Colab

4. Upload Project Files within Google Colab:
   ```
   restaurant-chatbot/
   ├── src/
   │   ├── models/
   │   │   └── chatbot.py
   │   ├── training/
   │   │   ├── train_baseline.py
   │   │   └── train_rl.py
   │   └── evaluation/
   │       └── evaluate_responses.py
   └── data/
       ├── train.json
       └── val.json
   ```

5. Run Training:
   - Follow the notebook cells in order
   - The models will be saved under `models/` folder
   - Download the files after training completes

---

## 🧪 Experiments Summary

| Model Variant | Semantic | Relevance | Complete | Overall |
|---------------|----------|-----------|----------|---------|
| Vanilla GPT-2 | 0.474    | 0.467     | 0.300    | 0.560   |
| Fine-Tuned    | 0.596    | 0.667     | 0.650    | 0.728   |
| RL-Default    | 0.638    | 0.733     | 0.600    | 0.743   |

---

## 📦 Deployment (Separate Repo)

This repo focuses on model development and training.

➡️ Deployment via Flask + Docker + Google Cloud is included in [deployed-restaurant-bot](https://github.com/Illiaminerva/deployed-restaurant-bot)

---

## ✅ Dependencies

- Python 3.11+
- torch, transformers, mlflow
- sentence-transformers
- Flask (for deployment)

