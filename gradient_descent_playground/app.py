import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from functions import get_function
from optimizers.gradient_descent import gradient_descent
from utils.plot_utils import plot_descent, plot_convergence

# Streamlit Inputs
st.title("Gradient Descent Playground")

# Dropdown for function selection
function_name = st.selectbox("Select Function", ["parabola", "polynomial", "sine"])

# Sliders for user inputs
initial_x = st.slider("Initial x", -5.0, 5.0, 2.0)
learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
num_iterations = st.slider("Number of Iterations", 10, 1000, 100)

# Button to trigger the execution of the code
run_button = st.button("Run Gradient Descent")

if run_button:
    # Load function and derivative
    f, df = get_function(function_name)

    # Run optimization using gradient descent
    x_history, f_history = gradient_descent(f, df, initial_x, learning_rate, num_iterations)

    # Display the results of the gradient descent optimization
    st.subheader("Results")
    st.write(f"Initial x: {initial_x}")
    st.write(f"Final x: {x_history[-1]}")
    st.write(f"Final f(x): {f_history[-1]}")

    # Show the optimization path as a graph
    st.subheader(f"Optimization Path for {function_name.capitalize()}")

    # Call the plotting function and use st.pyplot to display the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_descent(f, x_history, ax)
    st.pyplot(fig)

    # Show the convergence of the function value over iterations
    st.subheader("Convergence of Function Value over Iterations")

    # Call the plotting function and use st.pyplot to display the plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_convergence(f_history, ax2)
    st.pyplot(fig2)
