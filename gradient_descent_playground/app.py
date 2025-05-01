import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from functions import get_function
from optimizers.gradient_descent import gradient_descent
from utils.plot_utils import plot_descent, plot_convergence

# Streamlit Inputs
function_name = st.selectbox("Select Function", ["parabola", "polynomial", "sine"])
initial_x = st.slider("Initial x", -5.0, 5.0, 2.0)
learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
num_iterations = st.slider("Number of Iterations", 10, 1000, 100)

# Load function and derivative
f, df = get_function(function_name)

# Run optimization
x_history, f_history = gradient_descent(f, df, initial_x, learning_rate, num_iterations)

# Visualizations
plot_descent(f, x_history)
plot_convergence(f_history)
