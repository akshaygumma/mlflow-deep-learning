# ANN Model with MLflow Tracking, Hyperparameter Tuning, and Deployment

This project implements an **Artificial Neural Network (ANN)** using **Keras** for regression tasks, combined with:
- 🔹 **MLflow Tracking** for experiment logging
- 🔹 **Hyperopt** for hyperparameter optimization
- 🔹 **MLflow Model Registry** for model versioning
- 🔹 **MLflow Serving (REST API)** for production inference
- 🔹 **Docker containerization** for scalable cloud deployment

The goal is to provide a full end-to-end machine learning workflow from model building to deployment.

---

## 🚀 Features

✅ Build a Keras-based ANN for regression  
✅ Normalize input data using Keras preprocessing layers  
✅ Tune hyperparameters (`learning rate`, `momentum`, etc.) using **Hyperopt**  
✅ Track parameters, metrics, and artifacts using **MLflow UI**  
✅ Register the best model to **MLflow Model Registry**  
✅ Serve the model via **MLflow REST API**  
✅ Containerize the model serving endpoint using **Docker**

---

## ⚙️ Technologies Used

- **Python 3.x**
- **TensorFlow / Keras**
- **MLflow**
- **Hyperopt**
- **Docker**
- **Sklearn (for data splitting and preprocessing)**

---

## 🧠 Model Details

The ANN model architecture:
- Input Layer: normalization (mean + variance from training set)
- Dense Layer 1: 64 units, ReLU activation
- Output Layer: 1 unit (regression output)

The model is compiled with:
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** SGD with tunable learning rate and momentum
- **Metric:** Root Mean Squared Error (RMSE)

---

## 🔬 Hyperparameter Tuning

We use **Hyperopt + TPE (Tree-structured Parzen Estimator)** for tuning:
- Learning rate (`lr`)
- Momentum (`momentum`)

Example hyperopt config:
```python
space = {
    'lr': hp.uniform('lr', np.log(1e-5),np.log(1e-1)),
    'momentum': hp.uniform('momentum', 0.0, 0.99)
}
