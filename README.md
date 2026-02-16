# 🔍 Social Media Sentiment Analysis For Brand Monitoring

A comparative sentiment analysis project on 1.6M Twitter tweets 
using TF-IDF + Logistic Regression vs RoBERTa transformer — 
evaluating when traditional ML outperforms deep learning on 
short-text classification tasks.

---

## 🎯 Project Overview

This project implements binary sentiment classification on 
Twitter data using two contrasting approaches — a traditional 
ML pipeline and a pretrained transformer model — to understand 
model-data fit in real-world NLP scenarios.

Dataset: **1.6M tweets** (800K positive, 800K negative)

---

## ⚙️ Tech Stack

Python | Scikit-learn | HuggingFace Transformers | RoBERTa |
TF-IDF | NLTK | PyTorch | Pandas | NumPy | Matplotlib | 
Seaborn | Jupyter Notebook

---

## 🧠 Preprocessing Pipeline

- Lowercasing + URL removal
- @mention replacement with USER token
- Hashtag cleaning (keeping text, removing #)
- Digit removal + whitespace normalization
- Custom stopword removal
- RegexpTokenizer + POS-aware Lemmatization 
  (WordNetLemmatizer)

---

## 🤖 Models Implemented

### 1. TF-IDF + Logistic Regression (Baseline)
- TF-IDF vectorization with 500K features
- Bigram support (ngram_range = 1,2)
- Trained on 1.28M tweets (80/20 split)
- Evaluated on 320,000 tweets

### 2. RoBERTa (Transformer)
- Pretrained: `cardiffnlp/twitter-roberta-base-sentiment`
- Twitter-specific pretrained weights
- Evaluated on balanced 1,000 tweet subset
- Softmax over negative/positive scores 
  (neutral class ignored)

---

## 📊 Model Performance Comparison

### TF-IDF + Logistic Regression
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Negative | 81% | 80% | 80% |
| Positive | 80% | 82% | 81% |
| **Overall Accuracy** | | | **81%** |

### RoBERTa (twitter-roberta-base-sentiment)
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Negative | 75% | 71% | 73% |
| Positive | 72% | 76% | 74% |
| **Overall Accuracy** | | | **74%** |

---

## 🔍 Key Insights

- **Logistic Regression outperformed RoBERTa** (81% vs 74%)
  on full dataset evaluation
- TF-IDF with **bigrams and 500K features** captured 
  Twitter-specific patterns effectively
- RoBERTa was evaluated on a **1K subset** vs LR on 
  **320K samples** — scale difference impacts results
- Key learning: **Model-data fit matters more than 
  model complexity** — a well-tuned baseline can 
  beat transformers on domain-specific short text
- Twitter-specific pretrained RoBERTa still achieved 
  competitive performance without any fine-tuning

---

## 🧪 Sample Predictions
```python
# Logistic Regression
tweet = "I hate the new design of your website!"
→ Predicted Sentiment: Negative ✅

# RoBERTa
tweet = "I #hatedata science brain #dsbrain"
→ Predicted Sentiment: Negative ✅
```

---

## 🚀 Future Enhancements

- Fine-tune RoBERTa on full 1.6M dataset for fair comparison
- Add neutral class for 3-way sentiment classification
- Build real-time brand monitoring dashboard (Streamlit)
- Integrate Twitter/X API for live sentiment tracking
- Experiment with DistilBERT for faster inference

---

## 📧 Contact

Feel free to connect for collaboration, internships, 
or project discussions.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/kokila-m-ai-ds/)
[![Email](https://img.shields.io/badge/Email-Contact-red)](mailto:kokilakoki3376@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/kokilamariyayi)
