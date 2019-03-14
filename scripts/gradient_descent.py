import pandas as pd
import numpy as np


'''
following function transforms a pandas df into the format we want for our algorithm
'''


def gradient_format(df, target, features):
    y = np.array(df[target]).reshape((-1, 1))
    X = np.array(df[features]).reshape((y.shape[0], -1))
    return X, y


'''
computes the cost of a prediction, also implements regularization
'''


def compute_cost(X, y, thetas, ld):
    if ld < 0:
        raise ValueError("lambda not greater or equal 0.")
    m = y.shape[0]
    cost = 1/(2*m) * np.sum(np.square(np.dot(X, thetas) - y), axis=0) + ld/(2*m) * float(sum(np.square(thetas[1:])))
    return cost[0]


'''
calculates the partial derivative dJ/dtheta_j where J is the cost function and theta_j the j'th parameter for theta, our
linear regression parameters. Includes regularization. 
'''


def derivative(X, y, thetas, ld):
    m = y.shape[0]
    err = np.dot(X, thetas) - y
    der = np.dot(err.T, X).T
    der[1:] = der[1:] + ld/m * thetas[1:]
    return der


'''
implements gradient descent. There are also options to select a maximum number of iterations, or a convergence 
threshold. For example: threshold 0.1 means that the algorithm stops once max(theta_prev - theta) < 0.1 
for each theta_j (theta_prev is the theta parameter of the previous iteration).
'''


def gradient_descent(X, y, alpha, ld=0, max_iterations=-1, threshold=-1):
    if ld < 0:
        raise ValueError("lambda not greater or equal 0.")
    if not ((max_iterations > 0) != (threshold > 0)):
        raise ValueError("max_iterations (range of(0, )) or threshold (range of (0,1)) must be specified")

    num_parameters = X.shape[1]
    num_trainings = X.shape[0]
    thetas = np.asarray([0.0 for i in range(num_parameters)]).reshape((num_parameters, -1))

    if max_iterations > 0:
        for i in range(max_iterations):
            thetas = thetas - alpha/num_trainings * derivative(X, y, thetas, ld)

    if threshold > 0:
        thetas_prev = thetas + 1
        while np.max(np.abs(thetas_prev - thetas)) > threshold:
            thetas_prev = thetas
            thetas = thetas - alpha / num_trainings * derivative(X, y, thetas, ld)

    print("Cost:")
    print(compute_cost(X, y, thetas, ld))
    print("Thetas:")
    return thetas


# starting to test:


test = {
    1: [1, 1, 1, 1, 1],
    2: [1, 4, 3, 2, 4],
    3: [6, 7, 2, 3, 5],
    4: [8, 9, 3, 3, 2],
    5: [3, 6, 4, 3, 2]
}

test_df = pd.DataFrame(test)
X1, y1 = gradient_format(test_df, 5, [1, 2, 3, 4])

X_test = np.asarray(
    [
        [1, 1, 6, 8],
        [1, 4, 7, 9],
        [1, 3, 2, 3],
        [1, 2, 3, 3],
        [1, 4, 5, 2]
    ],
    dtype="float64"
)

y_test = np.asarray(
    [
    [3, 6, 4, 3, 2],
    ],
    dtype="float64"
).T

x = gradient_descent(X_test, y_test, 0.003, threshold=10e-12)
print(x)

x2 = gradient_descent(X1, y1, 0.003,ld=0.6, threshold=10e-12)
print(x2)

x3 = gradient_descent(X1, y1, 0.003,ld=0.6, max_iterations=50000)
print(x3)
