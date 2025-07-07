# heart-disease-prediction
It is used to predict heart disease using data

# ❤️ Heart Disease Prediction using Neural Networks (Keras + TensorFlow)

This project uses a neural network built with Keras and TensorFlow to predict the presence of heart disease based on a set of medical attributes from the UCI Heart Disease dataset.

---

## 📊 Dataset

The dataset used is `heart.csv`, which includes medical attributes such as:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Max heart rate
- Exercise-induced angina
- Oldpeak
- Slope of ST segment
- Number of major vessels
- Thalassemia
- Target (0 = No disease, 1 = Disease)

---

## 🧠 Model Architecture

A simple feedforward neural network using **Keras Sequential API**:

- **Input Layer**: 13 features
- **Hidden Layer 1**: 8 neurons, ReLU activation
- **Hidden Layer 2**: 14 neurons, ReLU activation
- **Output Layer**: 1 neuron, Sigmoid activation (for binary classification)

---

## ⚙️ Technologies Used

- Python 3
- TensorFlow / Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---
