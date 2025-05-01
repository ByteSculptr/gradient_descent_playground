
def gradient_descent(f, df, initial_x, learning_rate, num_iterations):
    x = initial_x
    x_history = [x]
    f_history = [f(x)]

    for _ in range(num_iterations):
        grad = df(x)
        x = x - learning_rate * grad
        x_history.append(x)
        f_history.append(f(x))

    return x_history, f_history
