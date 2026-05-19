# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:51:33 2026

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:46:55 2026

@author: Lenovo

DLBAIPCV01 - Computer Vision Project
Task 1: Handwritten Digit Recognition
Comparing LeCun et al. (1989) CNN vs. Chollet (2021) CNN on MNIST

Works in PyCharm and Spyder / Anaconda

Install dependencies (run once in terminal):
    pip install tensorflow numpy matplotlib seaborn scikit-learn
"""

# =========================
# IMPORT LIBRARIES
# =========================
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    AveragePooling2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# LOAD DATASET
# =========================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("=" * 55)
print("  MNIST Dataset Summary")
print("=" * 55)
print("Training images shape:", x_train.shape)
print("Test images shape    :", x_test.shape)
print("Number of classes    :", len(np.unique(y_train)))
print("Pixel value range    :", x_train.min(), "-", x_train.max())

# =========================
# PREPROCESS DATA
# =========================

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# Reshape to CNN format (samples, height, width, channels)
x_train = x_train.reshape((60000, 28, 28, 1))
x_test  = x_test.reshape((10000, 28, 28, 1))

# One-hot encoding for both models
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test,  10)

# =========================
# DISPLAY SAMPLE IMAGES
# =========================
plt.figure(figsize=(12, 4))
plt.suptitle("Figure 1 - MNIST Sample Images (one per digit)", fontsize=12)

for digit in range(10):
    idx = np.where(y_train.flatten() == digit)[0][0]
    plt.subplot(2, 5, digit + 1)
    plt.imshow(x_train[idx].reshape(28, 28), cmap="gray")
    plt.title(f"Digit: {digit}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("fig1_mnist_samples.png", dpi=150, bbox_inches="tight")
plt.show()

# =========================================================
# MODEL 1 - CHOLLET (2021) - Modern Keras CNN
# =========================================================
# Source: Chollet, F. (2021). Simple MNIST convnet.
# https://keras.io/examples/vision/mnist_convnet/
# Reproduced exactly - no modifications made.
#
# Architecture:
#   Conv2D(32, 3x3, ReLU) -> MaxPool(2x2) ->
#   Conv2D(64, 3x3, ReLU) -> MaxPool(2x2) ->
#   Flatten -> Dropout(0.5) -> Dense(10, softmax)
# =========================================================

modern_model = Sequential(name="Chollet_2021")

modern_model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=(28, 28, 1)
))

modern_model.add(MaxPooling2D(pool_size=(2, 2)))

modern_model.add(Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation='relu'
))

modern_model.add(MaxPooling2D(pool_size=(2, 2)))

modern_model.add(Flatten())

modern_model.add(Dropout(0.5))

modern_model.add(Dense(10, activation='softmax'))

# Compile
modern_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n========================")
print("MODEL 1 - Chollet (2021)")
print("========================")
modern_model.summary()

# Train
start_time = time.time()

history_modern = modern_model.fit(
    x_train,
    y_train_cat,
    epochs=15,          # 15 epochs as in Chollet (2021)
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

modern_training_time = time.time() - start_time

# Evaluate
modern_loss, modern_accuracy = modern_model.evaluate(
    x_test, y_test_cat, verbose=0
)

print("\nChollet (2021) Results:")
print("  Test Loss    :", round(modern_loss, 4))
print("  Test Accuracy:", round(modern_accuracy, 4))
print("  Training Time:", round(modern_training_time, 1), "seconds")

# =========================================================
# MODEL 2 - LeCUN et al. (1989) - Historical CNN
# =========================================================
# Source: LeCun et al. (1989). Handwritten digit recognition
# with a back-propagation network. NeurIPS 1989.
#
# Original architecture:
#   Input 16x16 -> H1: 12 maps 5x5 tanh -> subsampling ->
#   sparse connections -> H2: 12 maps 5x5 tanh -> subsampling ->
#   H3: 30 units tanh -> Output: 10 RBF units
#
# Adaptations and justifications:
#   1. Input kept at 28x28 (MNIST native size).
#      Resizing to 16x16 would discard spatial information.
#   2. AveragePooling2D replaces learned subsampling.
#      Closest Keras equivalent to LeCun's 2x2 averaging layer.
#   3. Dropout(0.3) replaces sparse H1->H2 connections (Table 1).
#      Achieves the same regularisation effect (Srivastava 2014).
#   4. Dense(10, softmax) replaces 10 RBF output units.
#      Softmax with categorical cross-entropy is the modern
#      equivalent and produces stable gradients.
#   5. SGD(lr=0.01, momentum=0.9) matches original back-propagation
#      with momentum used in the paper.
#   6. Batch size 32 approximates the original online (batch=1)
#      stochastic gradient descent for practical runtime.
# =========================================================

lecun_model = Sequential(name="LeCun_1989_approx")

# H1 - 12 feature maps, 5x5 receptive fields, tanh
lecun_model.add(Conv2D(
    filters=12,
    kernel_size=(5, 5),
    activation='tanh',
    padding='valid',
    input_shape=(28, 28, 1)
))

# Subsampling layer (AveragePooling approximates learned subsampling)
lecun_model.add(AveragePooling2D(pool_size=(2, 2)))

# Dropout replaces sparse H1->H2 connections from Table 1 of the paper
lecun_model.add(Dropout(0.3))

# H2 - 12 feature maps, 5x5 receptive fields, tanh
lecun_model.add(Conv2D(
    filters=12,
    kernel_size=(5, 5),
    activation='tanh',
    padding='valid'
))

# Subsampling layer
lecun_model.add(AveragePooling2D(pool_size=(2, 2)))

lecun_model.add(Flatten())

# H3 - 30 fully connected units, tanh (identical to paper)
lecun_model.add(Dense(30, activation='tanh'))

# Output - 10 units softmax (replaces RBF units)
lecun_model.add(Dense(10, activation='softmax'))

# Compile - SGD with momentum matches original back-propagation
lecun_model.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n==============================")
print("MODEL 2 - LeCun (1989) approx.")
print("==============================")
lecun_model.summary()

# Train
start_time = time.time()

history_lecun = lecun_model.fit(
    x_train,
    y_train_cat,
    epochs=23,          # ~23 passes as in original paper
    batch_size=32,      # mini-batch SGD approximating online learning
    validation_split=0.1,
    verbose=1
)

lecun_training_time = time.time() - start_time

# Evaluate
lecun_loss, lecun_accuracy = lecun_model.evaluate(
    x_test, y_test_cat, verbose=0
)

print("\nLeCun (1989) approx. Results:")
print("  Test Loss    :", round(lecun_loss, 4))
print("  Test Accuracy:", round(lecun_accuracy, 4))
print("  Training Time:", round(lecun_training_time, 1), "seconds")

# =========================================================
# FINAL COMPARISON TABLE
# =========================================================

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)
print(f"{'Model':<28} {'Accuracy':>10} {'Loss':>10} {'Time(s)':>10}")
print("-" * 60)
print(f"{'Chollet (2021)':<28} {modern_accuracy:>10.4f} {modern_loss:>10.4f} {modern_training_time:>10.1f}")
print(f"{'LeCun (1989) approx.':<28} {lecun_accuracy:>10.4f} {lecun_loss:>10.4f} {lecun_training_time:>10.1f}")
print("=" * 60)

# =========================================================
# FIGURE 2 - TRAINING CURVES (Accuracy + Loss)
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Figure 2 - Training Curves: Chollet (2021) vs. LeCun (1989)", fontsize=12)

# Accuracy
axes[0].plot(history_modern.history['accuracy'],     color='steelblue',  label='Chollet - train')
axes[0].plot(history_modern.history['val_accuracy'], color='steelblue',  label='Chollet - val',  linestyle='--')
axes[0].plot(history_lecun.history['accuracy'],      color='darkorange', label='LeCun - train')
axes[0].plot(history_lecun.history['val_accuracy'],  color='darkorange', label='LeCun - val',    linestyle='--')
axes[0].set_title("Accuracy per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history_modern.history['loss'],     color='steelblue',  label='Chollet - train')
axes[1].plot(history_modern.history['val_loss'], color='steelblue',  label='Chollet - val',  linestyle='--')
axes[1].plot(history_lecun.history['loss'],      color='darkorange', label='LeCun - train')
axes[1].plot(history_lecun.history['val_loss'],  color='darkorange', label='LeCun - val',    linestyle='--')
axes[1].set_title("Loss per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fig2_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# =========================================================
# FIGURE 3 - CONFUSION MATRICES
# =========================================================

# Get predictions
y_pred_modern = np.argmax(modern_model.predict(x_test, verbose=0), axis=1)
y_pred_lecun  = np.argmax(lecun_model.predict(x_test,  verbose=0), axis=1)

cm_modern = confusion_matrix(y_test, y_pred_modern)
cm_lecun  = confusion_matrix(y_test, y_pred_lecun)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figure 3 - Confusion Matrices (Test Set)", fontsize=12)

sns.heatmap(cm_modern, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=range(10), yticklabels=range(10),
            linewidths=0.5, linecolor='white')
axes[0].set_title("Chollet (2021)")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

sns.heatmap(cm_lecun, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=range(10), yticklabels=range(10),
            linewidths=0.5, linecolor='white')
axes[1].set_title("LeCun (1989) approx.")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.savefig("fig3_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# =========================================================
# CLASSIFICATION REPORTS
# =========================================================

print("\n=== Chollet (2021) - Classification Report ===")
print(classification_report(y_test, y_pred_modern,
                             target_names=[str(i) for i in range(10)]))

print("\n=== LeCun (1989) approx. - Classification Report ===")
print(classification_report(y_test, y_pred_lecun,
                             target_names=[str(i) for i in range(10)]))

# =========================================================
# FIGURE 4 - PREDICTION EXAMPLES (Modern model)
# =========================================================

plt.figure(figsize=(12, 5))
plt.suptitle("Figure 4 - Prediction Examples (Chollet model on test set)", fontsize=12)

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    predicted = y_pred_modern[i]
    true_label = y_test[i]
    color = 'green' if predicted == true_label else 'red'
    plt.title(f"P:{predicted} T:{true_label}", color=color, fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.savefig("fig4_predictions.png", dpi=150, bbox_inches="tight")
plt.show()

# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 55)
print("  All done!")
print("  4 figures saved in the same folder as this script:")
print("    fig1_mnist_samples.png")
print("    fig2_training_curves.png")
print("    fig3_confusion_matrices.png")
print("    fig4_predictions.png")
print("  Copy the numbers above into your report Table 3.2.")
print("=" * 55)