# Sentiment Analysis using Logistic Regression and RoBERTa

## Project Overview
This project implements sentiment analysis on Twitter text using two different
approaches: a traditional machine learning model and a transformer-based
deep learning model.

The objective is to compare baseline performance with a transformer model
on the same sentiment classification task.

---

## Models Implemented

### Logistic Regression
- TF-IDF feature extraction
- Used as a baseline model for sentiment classification

### RoBERTa
- Pretrained transformer-based language model
- Fine-tuned for binary sentiment classification

---

## Evaluation Metrics
The models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score

---

## Results

 Logistic Regression --  **81%** 
 RoBERTa -- **74%** 

> Note: Results are based on the preprocessing steps, train–test split,
> and hyperparameter settings used in the notebooks.

---

## Key Observations
- Logistic Regression achieved higher accuracy due to effective TF-IDF
  feature representation on short text data.
- RoBERTa requires careful fine-tuning and larger computational resources
  to outperform traditional models.
- Transformer models do not always guarantee better performance without
  extensive tuning and sufficient training data.

---
