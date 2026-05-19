# Computer-vision-task-1
python code
# DLBAIPCV01 – Computer Vision | Task 1
## Handwritten Digit Recognition: LeCun (1989) vs. Chollet (2021) on MNIST
IU International University of Applied Sciences – B.A. Applied Artificial Intelligence

---

## What this does
Trains and compares two CNN architectures on the MNIST handwritten digit dataset:
- **Chollet (2021)** – modern Keras CNN with ReLU, MaxPooling, Adam
- **LeCun (1989) approx.** – re-implementation of the original 1989 backpropagation network

---

## Results
| Model | Accuracy | Loss | Time |
|---|---|---|---|
| Chollet (2021) | 99.15% | 0.0222 | 100.9s |
| LeCun (1989) approx. | 98.77% | 0.0366 | 141.7s |

---

## How to run
pip install tensorflow numpy matplotlib seaborn scikit-learn
python 15epochs_code.py

Outputs 4 figures automatically: MNIST samples, training curves, confusion matrices, prediction examples.

---

## Built with
Python · TensorFlow/Keras · NumPy · Matplotlib · Seaborn · Scikit-learn
