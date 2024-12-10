import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, load_iris, load_diabetes, load_breast_cancer, load_digits, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from mpl_toolkits.mplot3d import Axes3D

# Utility functions for datasets
@st.cache_data
def load_dataset(app_type, dataset_choice):
    if dataset_choice == "Synthetic":
        if app_type == "Classification":
            X, y = make_classification(
                n_samples=1000,
                n_features=3,
                n_informative=2,  # Number of informative features
                n_redundant=0,    # Number of redundant features
                n_repeated=0,     # Number of repeated features
                n_classes=2,
                random_state=42
            )
        else:
            X, y = make_regression(
                n_samples=1000,
                n_features=3,
                noise=0.1,
                random_state=42
            )
    elif dataset_choice == "Iris (Classification)":
        data = load_iris()
        X, y = data.data, data.target
    elif dataset_choice == "Diabetes (Regression)":
        data = load_diabetes()
        X, y = data.data, data.target
    elif dataset_choice == "California Housing (Regression)":
        data = fetch_california_housing()
        X, y = data.data, data.target
    elif dataset_choice == "Breast Cancer (Classification)":
        data = load_breast_cancer()
        X, y = data.data, data.target
    elif dataset_choice == "Digits (Classification)":
        data = load_digits()
        X, y = data.data, data.target
    else:
        raise ValueError("Invalid dataset choice!")

    # Normalize the dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into training and testing sets
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Loss functions
def compute_loss(y, y_pred, loss_type):
    if loss_type == "Mean Squared Error (MSE)":
        return np.mean((y - y_pred)**2)
    elif loss_type == "Cross-Entropy Loss":
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    elif loss_type == "Hinge Loss":
        return np.mean(np.maximum(0, 1 - y * y_pred))
    else:
        raise ValueError("Invalid loss function!")

# Optimization algorithms
def gradient_descent(X, y, lr, epochs):
    loss_values = []
    weights = np.zeros(X.shape[1])
    for epoch in range(epochs):
        gradient = -2 * X.T @ (y - X @ weights) / len(X)
        weights -= lr * gradient
        loss = compute_loss(y, X @ weights, "Mean Squared Error (MSE)")
        loss_values.append(loss)
    return weights, loss_values

def mini_batch_gd(X, y, lr, epochs, batch_size):
    loss_values = []
    weights = np.zeros(X.shape[1])
    for epoch in range(epochs):
        perm = np.random.permutation(len(X))
        for i in range(0, len(X), batch_size):
            batch_indices = perm[i:i+batch_size]
            X_batch, y_batch = X[batch_indices], y[batch_indices]
            gradient = -2 * X_batch.T @ (y_batch - X_batch @ weights) / batch_size
            weights -= lr * gradient
        loss = compute_loss(y, X @ weights, "Mean Squared Error (MSE)")
        loss_values.append(loss)
    return weights, loss_values

def rmsprop(X, y, lr, epochs, beta=0.9, epsilon=1e-8):
    loss_values = []
    weights = np.zeros(X.shape[1])
    cache = np.zeros_like(weights)
    for epoch in range(epochs):
        gradient = -2 * X.T @ (y - X @ weights) / len(X)
        cache = beta * cache + (1 - beta) * gradient**2
        weights -= lr * gradient / (np.sqrt(cache) + epsilon)
        loss = compute_loss(y, X @ weights, "Mean Squared Error (MSE)")
        loss_values.append(loss)
    return weights, loss_values

def sgd(X, y, lr, epochs, batch_size):
    loss_values = []
    weights = np.zeros(X.shape[1])
    for epoch in range(epochs):
        perm = np.random.permutation(len(X))
        for i in range(0, len(X), batch_size):
            batch_indices = perm[i:i+batch_size]
            X_batch, y_batch = X[batch_indices], y[batch_indices]
            gradient = -2 * X_batch.T @ (y_batch - X_batch @ weights) / batch_size
            weights -= lr * gradient
        loss = np.mean((y - X @ weights)**2)
        loss_values.append(loss)
    return weights, loss_values

def adam(X, y, lr, epochs, batch_size):
    loss_values = []
    weights = np.zeros(X.shape[1])
    m, v = np.zeros_like(weights), np.zeros_like(weights)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for epoch in range(epochs):
        perm = np.random.permutation(len(X))
        for i in range(0, len(X), batch_size):
            batch_indices = perm[i:i+batch_size]
            X_batch, y_batch = X[batch_indices], y[batch_indices]
            gradient = -2 * X_batch.T @ (y_batch - X_batch @ weights) / batch_size
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient**2
            m_hat, v_hat = m / (1 - beta1**(epoch+1)), v / (1 - beta2**(epoch+1))
            weights -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        loss = np.mean((y - X @ weights)**2)
        loss_values.append(loss)
    return weights, loss_values

# Visualization
def visualize_loss_landscape(X, y, weights_range=(-10, 10), steps=50):
    w1 = np.linspace(weights_range[0], weights_range[1], steps)
    w2 = np.linspace(weights_range[0], weights_range[1], steps)
    W1, W2 = np.meshgrid(w1, w2)
    Loss = np.zeros_like(W1)
    for i in range(steps):
        for j in range(steps):
            weights = np.array([W1[i, j], W2[i, j]])
            Loss[i, j] = compute_loss(y, X @ weights, "Mean Squared Error (MSE)")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(W1, W2, Loss, cmap='viridis')
    ax.set_xlabel("Weight 1")
    ax.set_ylabel("Weight 2")
    ax.set_zlabel("Loss")
    st.pyplot(fig)

# Streamlit interface

st.title("Optimization Algorithms Visualization")
app_type = st.sidebar.selectbox("Application Type", ["Classification", "Regression"])
dataset_choice = st.sidebar.selectbox("Dataset", ["Synthetic", "Iris (Classification)", "Breast Cancer (Classification)", "Digits (Classification)", "Diabetes (Regression)", "California Housing (Regression)"])
optimizer = st.sidebar.selectbox("Optimizer", ["Gradient Descent", "Stochastic Gradient Descent (SGD)", "Mini-Batch Gradient Descent", "RMSprop", "Adam"])
loss_function = st.sidebar.selectbox("Loss Function", ["Mean Squared Error (MSE)", "Cross-Entropy Loss", "Hinge Loss"])
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 1.0, 0.01, step=0.0001)
epochs = st.sidebar.slider("Epochs", 1, 500, 100)
batch_size = st.sidebar.slider("Batch Size (For Mini-Batch GD)", 1, 256, 32)

# Load data and run optimizer
X_train, X_test, y_train, y_test = load_dataset(app_type, dataset_choice)
if optimizer == "Gradient Descent":
    weights, losses = gradient_descent(X_train, y_train, learning_rate, epochs)
elif optimizer == "Stochastic Gradient Descent (SGD)":
    weights, losses = sgd(X_train, y_train, learning_rate, epochs, batch_size)
elif optimizer == "Mini-Batch Gradient Descent":
    weights, losses = mini_batch_gd(X_train, y_train, learning_rate, epochs, batch_size)
elif optimizer == "RMSprop":
    weights, losses = rmsprop(X_train, y_train, learning_rate, epochs)
elif optimizer == "Adam":
    weights, losses = adam(X_train, y_train, learning_rate, epochs, batch_size)

# Show results
st.subheader("Loss Convergence")
st.line_chart(losses)

st.subheader("Loss Landscape Visualization")
if X_train.shape[1] >= 2:
    visualize_loss_landscape(X_train[:, :2], y_train)

st.sidebar.markdown("## Selected Dataset")
if dataset_choice == "Iris (Classification)":
    st.sidebar.write("**Iris Dataset**: A dataset for classifying iris flower species.")
elif dataset_choice == "Diabetes (Regression)":
    st.sidebar.write("**Diabetes Dataset**: A regression dataset for predicting disease progression.")
elif dataset_choice == "Breast Cancer (Classification)":
    st.sidebar.write("**Breast Cancer Dataset**: A classification dataset for cancer diagnosis.")
elif dataset_choice == "Digits (Classification)":
    st.sidebar.write("**Digits Dataset**: A dataset for recognizing handwritten digits.")
elif dataset_choice == "California Housing (Regression)":
    st.sidebar.write("**California Housing**: A regression dataset for predicting house prices in California.")
