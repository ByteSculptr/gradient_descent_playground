
import numpy as np
import matplotlib.pyplot as plt

def plot_descent(f, x_history):
    x_vals = np.linspace(min(x_history)-1, max(x_history)+1, 400)
    y_vals = f(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='f(x)')
    plt.plot(x_history, [f(x) for x in x_history], 'ro-', label='Descent Path')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gradient Descent Optimization')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_convergence(f_history):
    plt.figure(figsize=(10, 6))
    plt.plot(f_history, 'b-', label='f(x) vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.title('Function Value over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()
