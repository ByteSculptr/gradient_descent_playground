import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from functions import get_function
from optimizers.gradient_descent import gradient_descent
from utils.plot_utils import plot_descent, plot_convergence

# Streamlit Inputs
st.title("Gradient Descent Playground")

function_name = st.selectbox("Select Function", ["parabola", "polynomial", "sine"])
initial_x = st.slider("Initial x", -5.0, 5.0, 2.0)
learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
num_iterations = st.slider("Number of Iterations", 10, 1000, 100)

# Button to trigger the execution of the code
run_button = st.button("Run Gradient Descent")

if run_button:
    # Load function and derivative
    f, df = get_function(function_name)

    # Run optimization
    x_history, f_history = gradient_descent(f, df, initial_x, learning_rate, num_iterations)

    # Output Results
    st.subheader("Results")
    st.write(f"Initial x: {initial_x}")
    st.write(f"Final x: {x_history[-1]}")
    st.write(f"Final f(x): {f_history[-1]}")

    # Visualizations
    st.subheader(f"Optimization Path for {function_name.capitalize()}")
    plot_descent(f, x_history)

    st.subheader("Convergence of Function Value over Iterations")
    plot_convergence(f_history)
