# Naive Bayes Text Classification

This project implements a **Naive Bayes text classifier from scratch** in Python, with support for **Laplace smoothing**. It is evaluated on two real-world datasets: **Movie Reviews (sentiment analysis)** and **20 Newsgroups (topic classification)**. The project explores preprocessing strategies, model implementation details, performance comparisons, and visual analysis.

---

## Project Structure

```
NaiveBayes-Classifier/
├── data/
│   ├── reviews_polarity_test.csv
│   ├── reviews_polarity_train.csv
│   ├── newsgroups_test.csv
│   ├── newsgroups_train.csv
├── visuals/
│   ├── c1.png            # Confusion matrix (No smoothing)
│   ├── c2.png            # Confusion matrix (Laplace smoothing)
│   ├── c3.png            # Metrics comparison (Movie Review)
│   ├── c4.png            # Metrics comparison (Newsgroups)
│   ├── ld1.png - ld4.png # Distribution plots
│   └── t.png             # Summary comparison table
├── NaiveBayes.ipynb
├── NaiveBayes.oobn
├── README.md
└── requirements.txt
```

---

## Algorithm Overview

### Naive Bayes (Multinomial)

- **Training**: Computes log prior and likelihood probabilities.
- **Prediction**: Uses log-sum of prior and likelihoods to choose the class with the maximum posterior.
- **Laplace Smoothing**: Applied to avoid zero probabilities and improve generalization to unseen words.
- **Out-of-Vocabulary (OOV)** Handling: Small constant added to handle unseen tokens in test set.

---

## Implementation Details

- **Language**: Python
- **Libraries**: `nltk`, `pandas`, `numpy`, `seaborn`, `matplotlib`
- **Preprocessing**:
  - Lowercasing, punctuation and number removal
  - Tokenization using `punkt`
  - Lemmatization using WordNet and POS tagging
  - Word splitting for concatenated tokens (using `wordninja`)
  - Optional: Stemming (not used due to worse performance)
- **Features**: Unigrams (word frequency counts per document)

---

## Results Summary

| Dataset        | Accuracy (No Smoothing) | Accuracy (Laplace Smoothing) |
|----------------|--------------------------|-------------------------------|
| Movie Reviews  | 0.60                     | 0.84                          |
| News Groups    | 0.29                     | 0.64                          |

### Key Observations:
- **Laplace smoothing significantly improves classification**, especially when encountering rare or unseen words.
- **Movie Reviews** show better performance due to:
  - Binary classification
  - Balanced label distribution
  - Clear sentiment-based vocabulary
- **Newsgroups** show poorer performance due to:
  - Multi-class imbalance
  - Overlapping word distributions between topics (e.g., "Christian" vs "misc")
  - Data skew towards certain classes

---

## Visualizations

- Confusion matrices for both models (with and without smoothing)
- Metric comparisons: accuracy, precision, recall, F1
- Label and word count distributions across datasets
- Summary comparison table (`t.png`)

See the `/visuals` folder for all plots.

---

## How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Download NLTK Resources

Run this once to get necessary NLP tools:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### 3. Run Notebook or Python Scripts

You can either:
- Use `notebook.ipynb` for step-by-step interactive analysis, or
- Execute scripts from `src/` if modular automation is preferred.

---

## Key Equations

### Without Laplace:

\[
P(w|c) = \frac{\text{count}(w, c)}{\sum_{w'} \text{count}(w', c)}
\]

### With Laplace Smoothing:

\[
P(w|c) = \frac{\text{count}(w, c) + 1}{\text{total words in class} + |V|}
\]

---

## Advantages of Naive Bayes

- Simple and interpretable
- Fast for both training and prediction
- Works well with high-dimensional, sparse text data
- Efficient even on relatively small datasets

---

## Drawbacks

- Assumes feature independence (rarely true in real-world text)
- Struggles with overlapping vocabulary between classes
- Sensitive to class imbalance and out-of-vocabulary words

---

## References

- **NLTK Documentation**: https://www.nltk.org/
- **20 Newsgroups Dataset**: https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
- **Naive Bayes Theory**: https://en.wikipedia.org/wiki/Naive_Bayes_classifier

---
