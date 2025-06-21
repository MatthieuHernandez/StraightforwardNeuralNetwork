import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def custom_sigmoid(x):
    return (np.tanh(x / 2.0) + 1.0) / 2.0

def custom_sigmoid_derivative(x):
    return (1.0 - np.pow(np.tanh(x / 2.0), 2.0)) / 4.0

x = np.linspace(-10, 10, 400)

y = sigmoid(x)
y_prime = sigmoid_derivative(x)

print("*--- 34 ----")
print("s  = ", sigmoid(34), "c_s  = ", custom_sigmoid(34))
print("s' = ", sigmoid_derivative(34), "c_s' = ", custom_sigmoid_derivative(34))

print("*--- -4 ----")
print("s  = ", sigmoid(-4), "c_s  = ", custom_sigmoid(-4))
print("s' = ", sigmoid_derivative(-4), "c_s' = ", custom_sigmoid_derivative(-4))

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(x, sigmoid(x), label='Sigmoid', color='blue')
axs[0].plot(x, sigmoid_derivative(x), label='Derivative', color='red', linestyle='--')
axs[0].set_title("Sigmoid and Derivative")
axs[0].grid(True)
axs[0].legend()

axs[1].plot(x, custom_sigmoid(x), label='ReLU', color='green')
axs[1].plot(x, custom_sigmoid_derivative(x), label='Derivative', color='orange', linestyle='--')
axs[1].set_title("Custom Sigmoid and Derivative")
axs[1].grid(True)
axs[1].legend()

# Mise en page
plt.tight_layout()
plt.show()