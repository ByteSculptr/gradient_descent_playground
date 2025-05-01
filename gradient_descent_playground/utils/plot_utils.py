import numpy as np
import matplotlib.pyplot as plt

def plot_descent(f, x_history, ax):
    x_vals = np.linspace(min(x_history)-1, max(x_history)+1, 400)
    y_vals = f(x_vals)

    ax.plot(x_vals, y_vals, label='f(x)')
    ax.plot(x_history, [f(x) for x in x_history], 'ro-', label='Descent Path')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Gradient Descent Optimization')
    ax.legend()
    ax.grid(True)

def plot_convergence(f_history, ax):
    ax.plot(f_history, 'b-', label='f(x) vs Iteration')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('f(x)')
    ax.set_title('Function Value over Iterations')
    ax.legend()
    ax.grid(True)
