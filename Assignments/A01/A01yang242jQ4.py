# =============================================================================
# #
# #  Created on Sun Sep 18 2021
# #
# #  ENSE 885AU: A01Q4
# #
# #  @author: Jingkang Yang <yang242j@uregina.ca>
# #
# #  Execution Command: python A01yang242jQ4.py
# #
# #  Variable Defination:
# #
# #     w_opt = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# #     x is 1000 data ponts by sampling uniformly at random over the unit sphere
# #         and then removing those that have margin gama smaller then 0.1
# #     y = sign(w_opt^T x)
# #
# =============================================================================
# IMPORTS
import numpy as np

# Define gama individual, the distance of a sample to the decision boundary
def gama_indi(vector_xi):
    return abs(np.dot(w_opt.transpose(), vector_xi))

# Define remove_x_to_increase_gama, removing those of x that have margin gama smaller than min_gama
def remove_x_to_increase_gama(matrix_x, min_gama):
    gama_list = []
    for i in range(len(matrix_x)):
        if (gama_indi(matrix_x[i]) < min_gama):
            gama_list.append(i)
    new_matrix_x = np.delete(matrix_x, gama_list, axis=0)
    return new_matrix_x

# Define the perceptron algorithm
def perceptron(matrix_x, y_original):
    # Initialize w
    w = np.zeros(matrix_x[0].shape) # w = [0,0,0,0,0,0,0,0,0,0]
    y_pred = np.empty(y_original.shape)
    
    n = 1
    done = False
    while not done:
        # Random pick one sample to predict
        sample_index = np.random.choice(range(len(x)))

        # Predict y
        y_pred[sample_index] = np.sign(np.dot(w.transpose(), matrix_x[sample_index]))

        # Correcting
        if (y_pred[sample_index] != y_original[sample_index] and y_original[sample_index] > 0):
            w += matrix_x[sample_index]
        elif (y_pred[sample_index] != y_original[sample_index] and y_original[sample_index] < 0):
            w -= matrix_x[sample_index]

        # Stoping Criterion
        done = True if (np.array_equal(y_pred, y_original)) else False
        #print("Error rate: ", np.mean(y_pred != y_original))
        n += 1

    print("Perceptron Final Iteration: ", n-1)
    print("New weights: ", w)
    return w


# Generate data sets
w_opt = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
x = np.random.uniform(-1, 1, (1000, 10))
print("Init shape of x: ", x.shape)
x = remove_x_to_increase_gama(x, 0.1)
print("New shape of x: ", x.shape)
new_gama = np.min(np.array([gama_indi(xi) for xi in x]))
print("Gama of new x: ", new_gama)
y = np.array([np.sign(np.dot(w_opt.transpose(), x[i])) for i in range(len(x))])
print("Shape of y: ", y.shape)

# Implement Perceptron algorithm
w = perceptron(x, y)
