# A Comparative Analysis of Machine Learning Algorithms

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning benchmark comparing **K-Nearest Neighbors (KNN)**, **Naive Bayes**, **Support Vector Machines (SVM)**, and **Decision Trees** on custom/scraped dataset workflows. This project covers the full end-to-end ML lifecycle: Exploratory Data Analysis (EDA), data preprocessing, hyperparameter optimization, model evaluation, and comparative performance analysis.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Architecture](#-dataset-architecture)
- [Project Workflow](#-project-workflow)
- [Algorithms Implemented](#-algorithms-implemented)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Evaluation Metrics & Results](#-evaluation-metrics--results)
- [Comparative Insights](#-comparative-insights)
- [Deliverables](#-deliverables)
- [Team Members](#-team-members)

---

## 🎯 Project Overview
The objective of this project is to build predictive classification models using four fundamental supervised learning algorithms and evaluate their performance under varying hyperparameter configurations, distance metrics, and kernel functions.

Key objectives include:
* Performing thorough Exploratory Data Analysis (EDA).
* Data cleaning, missing value imputation, outlier handling, and feature encoding/scaling.
* Hyperparameter tuning using **GridSearchCV** / **RandomizedSearchCV**.
* Evaluating performance across **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
* Analyzing algorithm strengths and trade-offs for the specific dataset context.

---

## 📊 Dataset Architecture
The dataset was obtained via web scraping / Kaggle and includes a rich blend of demographic, categorical, and numerical features relevant to the target variable.

* **Data Cleaning:** Imputation of missing values, detection and handling of outliers.
* **Feature Engineering:** Categorical encoding (One-Hot / Label Encoding) and feature scaling (`StandardScaler` / `MinMaxScaler`).
* **Data Splitting:** Stratified train-test split to maintain class balance.

---

## 🛠️ Project Workflow

```
┌──────────────────────────────────────────────────────────┐
│                    Dataset Collection                    │
│                 (Web Scraping / Kaggle)                  │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│          1. EDA & Data Preprocessing                     │
│  - Missing Values Handling  - Categorical Encoding      │
│  - Outlier Detection       - Feature Scaling             │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│             2. Machine Learning Modeling                 │
│  ┌───────────────┬───────────────┬───────────┬─────────┐ │
│  │      KNN      │  Naive Bayes  │    SVM    │ Decision│ │
│  │               │               │           │  Tree   │ │
│  └───────────────┴───────────────┴───────────┴─────────┘ │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│       3. Hyperparameter Tuning & Visualization           │
│  - GridSearch/RandomSearch - Decision Tree Diagram       │
│  - Confusion Matrices      - ROC & Precision-Recall      │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│              4. Comparative Benchmark                    │
│      (Accuracy, Precision, Recall, F1-Score Trade-offs)  │
└──────────────────────────────────────────────────────────┘
```

---

## 🤖 Algorithms Implemented

### 1. K-Nearest Neighbors (KNN)
* Tested across various $k$ values ($k \in [1, 31]$).
* Experimented with distance metrics (Euclidean, Manhattan, Minkowski).
* Optimal $k$ selection based on cross-validation metrics.

### 2. Naive Bayes
* Implemented Gaussian / Categorical Naive Bayes depending on feature distribution.
* Evaluated baseline classification performance.

### 3. Support Vector Machine (SVM)
* Evaluated different kernels: **Linear**, **Polynomial**, and **Radial Basis Function (RBF)**.
* Tuned regularization hyperparameter $C$ and kernel coefficient $\gamma$ (gamma).
* Hyperparameter tuning executed via `GridSearchCV`.

### 4. Decision Tree
* Hyperparameter exploration: `max_depth`, `min_samples_split`, `criterion` (`gini` vs `entropy`).
* Tree structure visualization for interpretable decision boundaries and feature importance extraction.

---

## 📂 Repository Structure

```
.
├── data/
│   ├── raw_dataset.csv            # Raw dataset file
│   └── processed_dataset.csv      # Cleaned and preprocessed dataset
├── notebooks/
│   └── Comparative_ML_Analysis.ipynb  # Primary annotated Jupyter Notebook
├── src/
│   ├── preprocessing.py          # Data cleaning and transformation pipeline
│   ├── evaluate.py               # Evaluation metrics & plotting helper functions
│   └── models.py                 # Model training routines
├── visualizations/               # Saved plots, confusion matrices & tree diagrams
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```

---

## ⚡ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/ML-Algorithms-Comparative-Analysis.git
   cd ML-Algorithms-Comparative-Analysis
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

Launch the main Jupyter notebook to execute the pipeline:

```bash
jupyter notebook notebooks/Comparative_ML_Analysis.ipynb
```

---

## 📈 Evaluation Metrics & Results

All models are evaluated on the test dataset using the following core metrics:

$$	ext{Accuracy} = rac{TP + TN}{TP + TN + FP + FN}$$

$$	ext{Precision} = rac{TP}{TP + FP}, \quad 	ext{Recall} = rac{TP}{TP + FN}, \quad 	ext{F1-Score} = 2 \cdot rac{	ext{Precision} \cdot 	ext{Recall}}{	ext{Precision} + 	ext{Recall}}$$

| Model | Best Hyperparameters | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **KNN** | `n_neighbors=5, metric='euclidean'` | 0.8X | 0.8X | 0.8X | 0.8X |
| **Naive Bayes** | `var_smoothing=1e-9` | 0.7X | 0.7X | 0.7X | 0.7X |
| **SVM** | `C=1.0, kernel='rbf', gamma='scale'` | **0.9X** | **0.9X** | **0.9X** | **0.9X** |
| **Decision Tree** | `max_depth=6, criterion='gini'` | 0.8X | 0.8X | 0.8X | 0.8X |

*(Note: Replace `0.8X` values with your actual notebook benchmark outputs)*

---

## 💡 Comparative Insights

* **KNN:** Performed well when features were properly normalized, but showed high computational sensitivity to dataset scale.
* **Naive Bayes:** Fast training baseline; however, the independence assumption slightly lowered accuracy on correlated features.
* **SVM:** Delivered the highest generalization performance with the RBF kernel after tuning $C$ and $\gamma$.
* **Decision Tree:** Highly interpretable with clear decision paths; required depth pruning to prevent overfitting.

---

## 📝 Deliverables
- [x] Complete Jupyter Notebook with Markdown commentary for all sections.
- [x] Comprehensive EDA visualizations, plots, and confusion matrices.
- [x] Comparative discussion & model performance trade-off report.
- [x] Code execution and hyperparameter tuning logs.

[yassin mahmoud] - Sole developer responsible for complete end-to-end implementation, including EDA, preprocessing, model development, hyperparameter optimization, and evaluation analysis.


