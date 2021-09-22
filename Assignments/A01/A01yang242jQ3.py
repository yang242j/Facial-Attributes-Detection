# =============================================================================
# #
# #  Created on Sun Sep 18 2021
# #
# #  ENSE 885AU: A01Q3
# #
# #  @author: Jingkang Yang <yang242j@uregina.ca>
# #
# #  Execution Command: python A01yang242jQ3.py
# #
# #  Variable Defination:
# #
# #     x is a matrix with 100 vectors of size 10
# #     y is a vector of size 100
# #
# =============================================================================
# IMPORTS
import numpy as np

# Generate y list corresponding to the given x matrix
def y_gen(x_matrix):
    y_list = []
    for x_vector_10 in x_matrix:
        xi_sum = 0
        # Multiply each xi by its index+1 and sum together
        for xi_index in range(len(x_vector_10)):
            xi_sum += x_vector_10[xi_index] * (xi_index+1)
        # Add 0.1*N(0,1) to each yi
        yi = xi_sum + 0.1*np.random.normal(0, 1)
        y_list.append(yi)
    return y_list

# Calculate the l2 loss
def l2_loss(y_pred, yi):
    if (yi.shape == y_pred.shape):
        loss = 0
        for i in range(len(yi)):
            loss += (y_pred[i] - yi[i]) ** 2
        return loss/len(yi)
    else:
        print("l2_loss ERROR")

# Full Gradient Descent:
# Hypothesis: y = wTx + b with bias term
# Initialization: Initialize w and bias to vectors with all zeros
# Learning Rate (alpha): First attempt using 0.1, and increase or decreaase 0.01 each time
# Stop Criterion: To minimize l2 loss, mean squared error, by minimize the loss difference.
def FGD(x, y, alpha=0.1):
    n = 0
    w = np.zeros(x[0].shape)
    b = np.zeros(y.shape)
    loss_old, loss = 0.0, 0.0
    done = False
    while (done == False):
        # Predict and get loss value
        y_predict = np.array(
            [np.dot(w.transpose(), x[i]) + b[i] for i in range(len(x))])  # New y_predict
        loss_old = loss  # Old l2 loss
        loss = l2_loss(y_predict, y)  # New l2 loss

        # Stop if the loss difference is small enough
        done = True if (abs(loss_old - loss) <= 0.00001) else False

        # Get new w and b
        w_slope = 0
        b_slope = []
        for i in range(len(x)):
            w_slope += 2*(np.dot(w.transpose(), x[i]) + b[i] - y[i])*x[i]/len(x)
            b_slope_i = 2*(np.dot(w.transpose(), x[i]) + b[i] - y[i])/len(x)
            b_slope.append(b_slope_i)
        b_slope = np.array(b_slope)
        step_size_w = w_slope * alpha
        step_size_b = b_slope * alpha
        w -= step_size_w
        b -= step_size_b
        n += 1
    print("FGD Final iteration: ", n-1)
    print("FGD predicted weights: ", w)
    print("FGD predicted bias: ", b)
    return w, b

# Stochastic Gradient Descent:
# Hypothesis: y = wTx + b with bias term
# Initialization: Initialize w to all zeros, and bias term to 0.0
# Learning Rate (eta): First attempt using 0.1, and increase or decreaase 0.01 each time
# Stop Criterion: To minimize l2 loss, mean squared error, by minimize the step-size of both variables.
def SGD(x, y, eta=6.5):
    n = 1
    w = np.zeros(x[0].shape)
    b = 0.0
    done = False
    while (done == False):
        # Generate one random sample index
        sample_index = np.random.choice(range(len(x)))

        # Get new w and b
        step_size_w = 2 * eta * \
            (np.dot(w.transpose(), x[sample_index]) +
             b - y[sample_index]) * x[sample_index] / len(x)
        step_size_b = 2 * eta * \
            (np.dot(w.transpose(), x[sample_index]) +
             b - y[sample_index]) / len(x)
        w -= step_size_w
        b -= step_size_b

        # stoping Criterion: minimize the step size for both variables
        done = True if (np.all(abs(step_size_w) <= 0.00001) and abs(step_size_b) <= 0.00001) else False
        n += 1
    print("SGD Final iteration: ", n-1)
    print("SGD predicted weights: ", w)
    print("SGD predicted bias: ", b)
    return w, b


# Generate 100 synthetic data ponts (x, y)
x = np.random.uniform(0, 1, (100, 10))
y = np.array(y_gen(x))

# Implement full gradient descent
FGD_w, FGD_b = FGD(x, y)

# Implement stochastic gradient descent
SGD_w, SGD_b = SGD(x, y)
