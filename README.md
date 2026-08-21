# 📰 Fake News Detection using NLP

> A comparative machine learning study for detecting fake and real news through text representation, model, and feature experimentation.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NLP](https://img.shields.io/badge/Domain-NLP-green)
![Machine Learning](https://img.shields.io/badge/Focus-Machine%20Learning-orange)

---

## Overview

The rapid spread of misinformation across digital platforms makes automated fake news detection an important text classification problem.

This project develops an end-to-end NLP pipeline for classifying news articles as **Fake** or **True**. Rather than relying on a single model and reporting its accuracy, the project investigates how different **text representations, machine learning models, and handcrafted features** affect classification performance.

The project follows an experimental approach: the dataset is first investigated for redundancy, missing information, and potential leakage; preprocessing decisions are then made based on these observations; multiple representations and classifiers are evaluated through controlled experiments; and the resulting models are further examined using feature interpretation and error analysis.

The objective is not only to identify a high-performing model, but to understand **which choices work, why they work, and where the resulting system may fail**.

---

## Project Architecture

```text
Raw Fake / True News Data
          │
          ▼
   Dataset Investigation
          │
          ▼
 Data Cleaning & Preparation
          │
          ▼
   Text Preprocessing
          │
          ▼
 ┌─────────────────────────────┐
 │    Feature Representation   │
 │                             │
 │  BoW  │  TF-IDF  │ Word2Vec │
 └─────────────────────────────┘
          │
          ▼
     Model Training
          │
     ┌────┴────┐
     ▼         ▼
 Classifier   Feature
 Comparison   Experiments
     │         │
     └────┬────┘
          ▼
   Evaluation & Analysis
          │
          ▼
 Comparison & Model Decision
          │
          ▼
    Recommended Approach
```

The project follows an iterative NLP experimentation pipeline rather than a single-model workflow. Raw news data is investigated and prepared before text preprocessing and feature representation. Multiple representations and classifiers are then evaluated through controlled experiments, with additional handcrafted features tested against baseline models.

---

## Dataset & Initial Investigation

### Dataset Overview

The project uses a dataset containing **Fake and True news articles** with the following initial fields:

- `title`
- `text`
- `subject`
- `date`
- `label`

The original merged dataset contained **44,898 articles**, with an approximately balanced distribution between the two classes.

### Investigation → Reasoning → Decision

| Investigation | Observation | Decision |
|---|---|---|
| Class distribution | Fake and True classes were reasonably balanced | No resampling applied |
| Duplicate analysis | 5,793 duplicate rows were identified | Remove duplicates |
| Empty article bodies | 631 rows had empty `text` | Retain available title information |
| `subject` analysis | Strong association with the target was observed | Remove to avoid metadata leakage |
| `date` analysis | Represents metadata rather than article content | Remove |
| Title information | Titles remained available for articles with empty bodies | Combine title + text |

After duplicate removal, the working dataset contained **39,105 unique articles**.

The initial investigation was therefore used to identify redundancy, missing information, and potential sources of leakage before model development rather than treating the dataset as immediately model-ready.

### Dataset

The four dataset files used in this project are available separately in the Google Drive folder.

[📂 Download Dataset](https://drive.google.com/drive/folders/1CohI9KEV8QDBfzOutuTjbh1swHd7Wjc9?usp=drive_link)

---

## Text Preprocessing & Cleaning

Once the dataset was prepared, the next step was to convert the raw news content into a consistent textual representation.

### Preprocessing Pipeline

```text
Title + Article Text
        ↓
      Lowercase
        ↓
 Capture useful signals
 (! count, text length, URL presence)
        ↓
 Normalize URLs
        ↓
 Remove HTML
        ↓
 Remove punctuation / special characters
        ↓
      Tokenization
        ↓
   Stopword Removal
        ↓
      Stemming
```

Preprocessing was treated as **information management rather than simple noise removal**. Potentially useful signals were captured before the corresponding textual information was discarded.

### Key Decisions

**Title + Text**

Empty article bodies were identified during dataset investigation. Instead of discarding these records, available title information was retained by combining the title and article text.

**URLs**

URL presence was captured before URLs were normalized and removed from the textual representation, allowing URL-related information to be investigated separately.

**Punctuation**

Exclamation marks were counted before punctuation removal so that their potential class-level signal was not lost.

**Stopwords & Stemming**

Tokenization, stopword removal, and Porter stemming were applied to reduce unnecessary vocabulary variation before vectorization.

---

## Feature Representation & Engineering

After preprocessing, the next question was:

> **How should the cleaned text be represented so that machine learning models can learn from it?**

Instead of assuming one representation was best, three approaches were investigated:

- **Bag-of-Words (BoW)**
- **TF-IDF**
- **Word2Vec**

### Representation Experiment

To isolate the effect of text representation, **Logistic Regression was kept fixed** while the representation was changed.

```text
             Logistic Regression
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
       BoW         TF-IDF      Word2Vec
        │            │            │
        └────────────┼────────────┘
                     ▼
               Compare Results
```

This controlled setup allows differences in performance to be attributed primarily to the representation rather than simultaneous changes in the classifier.

### Handcrafted Features

The project also investigated whether simple, human-designed signals could add information beyond the text representation.

Candidate features included:

- `exclamation_count`
- `text_length`
- `has_url`

The analysis showed substantial class-level differences in some of these features. However, **not every candidate feature was automatically added to the final model**.

The enhanced experiments used:

> **TF-IDF + `exclamation_count` + `text_length`**

This allowed the effect of additional features to be evaluated against a baseline rather than assuming that more features would necessarily improve performance.

---

## Model Experiments

With the representations established, the next question was:

> **How does the choice of classifier affect performance when the text representation is kept fixed?**

### Classifier Comparison

Using **TF-IDF**, four classifiers were evaluated:

- K-Nearest Neighbors
- Logistic Regression
- Random Forest
- Neural Network

```text
                    TF-IDF
                      │
          ┌───────────┼───────────┬───────────┐
          ▼           ▼           ▼           ▼
         KNN          LR          RF          NN
          │           │           │           │
          └───────────┴───────────┴───────────┘
                      ▼
                 Compare Models
```

The purpose was to compare different learning approaches under the same text representation.

### Baseline vs Enhanced Models

The handcrafted features were then tested against baseline TF-IDF models.

```text
                 TF-IDF Baseline
                       │
                ┌──────┴──────┐
                ▼             ▼
       Logistic Regression  Random Forest
                │             │
                └──────┬──────┘
                       ▼
             + Handcrafted Features
                       │
                ┌──────┴──────┐
                ▼             ▼
          Enhanced LR    Enhanced RF
```

This experiment tested whether the additional handcrafted signals provided meaningful improvement over the original text-based models.

---

## Evaluation Strategy

The models were evaluated using multiple complementary metrics rather than relying on accuracy alone:

- **Accuracy** — overall correct predictions
- **Precision** — correctness of positive predictions
- **Recall** — ability to identify actual class instances
- **F1-score** — balance between precision and recall
- **Confusion Matrix** — class-wise prediction errors
- **Train vs Test Accuracy** — generalization and potential overfitting

The evaluation therefore considered three aspects:

```text
Overall Performance
        +
Class-wise Performance
        +
Generalization
(train vs test)
```

This provided the basis for comparing models and making the final model-selection decision.

---

## Results & Model Decision

### Representation Comparison

Using the same **Logistic Regression** classifier:

| Representation | Test Accuracy |
|---|---:|
| **BoW** | **99.37%** |
| TF-IDF | 98.43% |
| Word2Vec | 98.07% |

**Observation:** BoW achieved the highest test accuracy on this dataset, while the difference between the representations was relatively small.

### Classifier Comparison

Using **TF-IDF**:

| Model | Accuracy | F1 (Fake) | F1 (True) |
|---|---:|---:|---:|
| KNN | 85.79% | 82.88% | 87.86% |
| Logistic Regression | 98.43% | 98.27% | 98.56% |
| Random Forest | 97.20% | 96.87% | 97.47% |
| Neural Network | **98.75%** | **98.63%** | **98.85%** |

KNN performed substantially worse than the other evaluated classifiers, while Logistic Regression and Neural Network produced the strongest results with TF-IDF.

### Did Handcrafted Features Help?

| Model | Baseline | Enhanced |
|---|---:|---:|
| Logistic Regression | 98.43% | 98.45% |
| Random Forest | 97.20% | **97.66%** |

The additional handcrafted features produced only a marginal improvement for Logistic Regression, while the improvement was more noticeable for Random Forest.

### Final Model Decision

> **Highest accuracy ≠ automatically selected model.**

Although **BoW + Logistic Regression** achieved the highest raw test accuracy of **99.37%**, **TF-IDF + Logistic Regression** was selected as the recommended deployment-oriented approach because it showed a smaller train–test gap and more conservative generalization behaviour.

```text
Highest Raw Accuracy
        │
        ▼
   BoW + Logistic Regression
        │
        ├── Excellent test performance
        └── Larger train–test gap
                    │
                    ▼
          Consider Generalization
                    │
                    ▼
       TF-IDF + Logistic Regression
                    │
                    ▼
        Recommended Approach
```

The final choice therefore reflects **performance together with generalization**, rather than simply selecting the largest accuracy value.

---

## Explainability & Error Analysis

After comparing model performance, the next question was:

> **What is the model actually learning, and where does it fail?**

### Feature Interpretation

Logistic Regression coefficients were examined to understand which terms were associated with each class.

One important finding was the strong influence of the term **`reuter`** toward the True class.

This did not necessarily indicate that the word itself represents truthfulness. Instead, it suggested that the model could be learning **source-specific patterns** present in the dataset.

### Error Analysis

Misclassified articles were examined to understand failure cases, including:

- False positives — True articles predicted as Fake
- False negatives — Fake articles predicted as True
- Unusual writing styles
- Satire or opinion-like content
- Cases where linguistic patterns overlapped between classes

This analysis showed why aggregate metrics alone are insufficient for understanding model behaviour.

> **Performance tells us how well the model predicts; explainability and error analysis help us understand what it learned and where that performance may come from.**

---

## Limitations & Lessons

### Dataset / Source Bias

The most important limitation identified during analysis was **source-related bias**.

True articles in the dataset were primarily sourced from Reuters, and `reuter` emerged as one of the strongest features associated with the True class. This indicates that the model may learn source or writing-style patterns alongside more general linguistic signals.

Therefore:

> **The reported performance should be interpreted as performance on this dataset, not as a guarantee of real-world fake-news detection accuracy.**

### What the Experiments Taught Us

- More features do not automatically produce better models.
- The highest test accuracy does not automatically make a model the best deployment choice.
- Dataset construction can strongly influence what a model learns.
- Error analysis remains important even when aggregate metrics are very high.
- Model selection should consider both predictive performance and generalization behaviour.

---

## Future Work

The next development stages are driven by the limitations and findings of the experiments.

### 🌐 Web Application

Deploy the selected **TF-IDF + Logistic Regression** pipeline as a web application where users can submit news text and receive a prediction.

### 🧠 Transformer-based Models

Experiment with contextual language models such as:

- BERT
- DistilBERT

to investigate whether contextual representations improve upon traditional text representations.

### 🔎 Explainability

Extend coefficient-based interpretation using tools such as:

- SHAP
- LIME

for more detailed prediction-level explanations.

### 🌍 Better Generalization

Evaluate the system across multiple independent news sources to reduce the possibility of source-specific learning.

### 🔗 Additional Feature Experiments

Further investigate whether features such as `has_url` improve generalization when incorporated into controlled experiments, rather than simply exploiting dataset-specific differences.

---

## Tech Stack

### Language
- Python

### Data & NLP
- Pandas
- NumPy
- NLTK
- Gensim

### Machine Learning
- Scikit-learn

### Visualization
- Matplotlib
- Seaborn

### Development
- Jupyter Notebook

---

## Project Structure

```text
Fake_News_Detection/
│
├── data/
│   ├── Source/
│   └── Final/
│
├── models/
├── notebooks/
│   └── Fake_News_Detection_854216.ipynb
│
├── outputs/
├── report/
│
├── .gitignore
└── README.md
```

- `data/` — dataset workflow and prepared data
- `notebooks/` — experimental implementation
- `models/` — reserved for future trained model artifacts
- `outputs/` — reserved for selected experiment outputs
- `report/` — project report and presentation

---

## Installation & How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Samay1011/Fake_News_Detection.git
cd Fake_News_Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

The dataset is not included in the repository.

The four dataset files used in the project are available in the Google Drive folder:

[📂 Download Dataset](https://drive.google.com/drive/folders/1CohI9KEV8QDBfzOutuTjbh1swHd7Wjc9?usp=drive_link)

Place the required files in the corresponding `data/` directories before running the notebook.

### 4. Run the notebook

Open:

```text
notebooks/Fake_News_Detection_854216.ipynb
```

and execute the notebook sequentially.

---

## Dataset & References

### Dataset

The project uses a Fake/True news dataset containing article titles, article text, subject metadata, dates, and binary labels.

The dataset is provided separately through Google Drive and is not stored directly in the repository.

### Project Documentation

The repository also contains the project report and presentation under:

```text
report/
```

---

## Project Status

| Component | Status |
|---|---|
| Dataset investigation | ✅ Complete |
| NLP preprocessing | ✅ Complete |
| Text representation experiments | ✅ Complete |
| Model comparison | ✅ Complete |
| Feature experiments | ✅ Complete |
| Error analysis | ✅ Complete |
| Model selection | ✅ Complete |
| Web application | 🚧 Planned |

---

## Author

**Samay Talwar**  
Computer Science Undergraduate | Machine Learning & AI

[GitHub](https://github.com/Samay1011)
