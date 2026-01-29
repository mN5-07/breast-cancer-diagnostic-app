---
title: Breast Cancer Predictor
emoji: 🎗️
colorFrom: pink
colorTo: purple
sdk: docker
pinned: true           # Changed to true so it shows up prominently
license: mit
short_description: Educational breast cancer prediction demo

---

<p align="center">
  <img src="https://via.placeholder.com/800x400/FFB6C1/FFFFFF?text=Breast+Cancer+Predictor+App" alt="App Screenshot" width="80%"/>
  <br/>
  <em>Educational Streamlit app — not for medical diagnosis</em>
</p>

## 🎗️ Project Description

This is an **interactive web demo** built with Streamlit that lets users input cell nuclei measurements and receive a prediction of whether a breast tumor is **benign** or **malignant**.

- **Model**: Logistic Regression (selected for best balance of performance + interpretability)
- **Dataset**: UCI Breast Cancer Wisconsin (Diagnostic) — 569 samples, 30 features
- **Performance**: ~97–98% accuracy on hold-out test set (5-fold CV mean: 97.1%)
- **Most predictive features**: concavity_worst, texture_worst, symmetry_worst, concave points_worst

**Important**:  
⚠️ **This is an educational tool only.** It is **not** a medical diagnostic device. Never use model predictions for real medical decisions. Always consult a qualified healthcare professional.

Full training pipeline, EDA, model comparison, and feature importance analysis available in the main repository:  
🔗 **[Link to main GitHub repo]** *(replace with your actual repo URL)*

## 🚀 How to Use the App

1. Adjust the sliders in the sidebar (grouped by Mean, Std Error, Worst measurements)
2. Use the **Quick Start** buttons for typical cases:
   - Dataset Median
   - Typical Benign
   - Borderline
   - Typical Malignant
3. Click **Predict Tumor Type**
4. View:
   - Prediction (Benign / Malignant)
   - Malignancy probability (risk score)
   - Confidence gauge
   - Highlighted key feature values

<p align="center">
  <img src="https://via.placeholder.com/600x400/98FB98/333333?text=Prediction+Result+Example" alt="Prediction Example" width="60%"/>
</p>

## 🔍 Key Findings – Top Predictive Features

From permutation importance on the test set (Logistic Regression):

| Rank | Feature                | Mean Importance | Interpretation                              |
|------|------------------------|------------------|---------------------------------------------|
| 1    | concavity_worst        | 0.025           | Severity of deepest indentations (most important) |
| 2    | texture_worst          | 0.023           | Variation in gray-scale values (extreme)    |
| 3    | symmetry_worst         | 0.020           | Asymmetry in extreme measurements           |
| 4    | concave points_worst   | 0.012           | Number of deep indentations (extreme)       |
| 5    | concave points_mean    | 0.005           | Average number of indentations              |

**Insight**: "Worst" (extreme) values consistently outperform mean and standard error features — aligns with pathology knowledge that malignant tumors show more irregular/aggressive characteristics.

## 📸 Screenshots

### Main Interface with Input Sliders
<p align="center">
  <img src="images\homepage.png" alt="App Interface" width="70%"/>
</p>

### Prediction Result Example (Malignant)
<p align="center">
  <img src="images\typicalmalignant.png" alt="Malignant Prediction" width="70%"/>
</p>

### Benign Prediction
<p align="center">
  <img src="images\typicalbenign.png" alt="Benign Prediction" width="70%"/>
</p>

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: scikit-learn (Logistic Regression + StandardScaler)
- **Deployment**: Hugging Face Spaces (Docker)
- **Training pipeline**: Jupyter notebooks (EDA → preprocessing → model comparison → feature importance)

## Ethical & Responsible Use

- Model trained on small public dataset (n=569)
- No clinical validation performed
- Predictions are probabilistic estimates — **not** substitutes for biopsy, imaging, or expert diagnosis
- Always prioritize professional medical evaluation

## Deployment Notes

The app is deployed using Streamlit Community Cloud.
For production-grade deployment, the app can be containerized and hosted on
platforms such as Hugging Face Spaces, Fly.io, or AWS ECS.
