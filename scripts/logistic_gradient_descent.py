import numpy as np
import pandas as pd



'''
following function transforms dataframe into a frame thats usable for multiclass classification.
also returns a dictionary that shows to which column in the transformed targets the values from the 
target column got mapped. Also returns the original target column mapped with the new labels (int values
starting from 0 for the first class)
'''


def transform_multiclass(df, target, features):
    class_map = dict()
    classes = df[target].unique().tolist()
    c = 0
    y_s = []
    for el in classes:
        class_map[el] = c
        y_c = (df[target] == el).astype("int").values
        y_s.append(y_c)
        c += 1

    X = np.array(df[features])
    y = np.array(y_s).T
    X = X.reshape(X.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    y_original = np.array(df[target].map(class_map)).reshape(y.shape[0], -1)

    return X, y, y_original, class_map

'''
following function is a simpler function that transforms a dataframe for binary classification
'''


def transform_data(df, target, features):
    y = np.array(df[target]).reshape((-1, 1))
    X = np.array(df[features]).reshape((y.shape[0], -1))
    return X, y


'''
returns the cost of a predictions. is used for checking if the algorithms run right. Measures how "good" the 
predictions are.
'''


def cost(X, y, thetas, ld=0):  # returns cost of a prediction
    if ld < 0:
        raise ValueError("lambda not greater or equal 0.")
    m = y.shape[0]
    ones = np.ones(shape=(m, 1))
    sig = sigmoid(X, thetas)

    c = -1 / m * np.sum(y * np.log(sig) + (ones - y) * np.log(ones - sig)) + ld/(2*m) * float(sum(np.square(thetas[1:])))

    return c


'''
calculates the sigmoid activation function for all predictions, that maps values onto the space (0,1).
afterwards (depending on the treshold) this is rounded to 0 or 1 and decides the predicted label
'''


def sigmoid(X, thetas):  # calculates the sigmoid activation values
    sig = 1 / (1 + np.exp(-np.dot(X, thetas)))

    return sig


'''
calculates the derivative dJ/dtheta_j to use in gradient descent updates. contains an option for regularization.
does not regularize theta_0.
'''


def derivative(X, y, thetas, ld=0):  # calculates the derivatives for gradient descent
    if ld < 0:
        raise ValueError("lambda not greater or equal 0.")
    m = X.shape[0]
    error = sigmoid(X, thetas) - y

    der = (1/m * np.dot(error.T, X).T).reshape((thetas.shape[0], -1))
    der[1:, :] = der[1:, :] + ld/m * thetas[1:, :]
    #der[1:] = der[1:] + ld/m * thetas[1:]
    return der


'''
following function runs gradient descent for binary classifcation. I decided to write two seperate functions to keep 
everything compact and easier to understand. If a multiclass classification is needed, use the other function.
This algorithm also offers the possibility to specify either a number of maximum iterations or a threshold for making 
sure it converges. For example: threshold 0.1 means that the algorithm stops once max(theta_prev - theta) < 0.1 
for each theta_j (theta_prev is the theta parameter of the previous iteration).
'''


def gradient_descent_logistic(X, y, alpha,ld=0, max_iterations=-1, threshold=-1):
    if ld < 0:
        raise ValueError("lambda not greater or equal 0.")
    if not ((max_iterations > 0) != (threshold > 0)):
        raise ValueError("max_iterations (range of(0, )) or threshold (range of (0,1)) must be specified")

    num_parameters = X.shape[1]
    thetas = np.asarray([0 for i in range(num_parameters)]).reshape((num_parameters, -1))

    if max_iterations > 0:
        for i in range(max_iterations):
            thetas = thetas - alpha * derivative(X, y, thetas, ld)

    if threshold > 0:
        thetas_prev = thetas + 1
        while np.max(np.abs(thetas_prev - thetas)) > threshold:
            thetas_prev = thetas
            thetas = thetas - alpha * derivative(X, y, thetas, ld)

    print("Cost: " + str(cost(X, y, thetas, ld)))
    print("Theta:")
    print(thetas)
    return thetas


'''
runs gradient descent for one-vs-all multiclass classification. this function needs the prepared data that is returned
by the transformation function in the beginning of the script. This algorithm also offers the possibility to specify 
either a number of maximum iterations or a threshold for making sure it converges. For example: 
threshold 0.1 means that the algorithm stops once max(theta_prev - theta) < 0.1 for each theta_ij 
(theta_prev is the theta parameter of the previous iteration).
'''


def gradient_descent_logistic_mc(X, y, alpha, ld=0, max_iterations=-1, threshold=-1):
    if ld < 0:
        raise ValueError("lambda not greater or equal 0.")
    if not ((max_iterations > 0) != (threshold > 0)):
        raise ValueError("max_iterations (range of(0, )) or threshold (range of (0,1)) must be specified")

    num_classes = y.shape[1]
    num_parameters = X.shape[1]

    thetas = np.zeros(shape=(num_parameters, num_classes)).reshape((num_parameters, -1))

    if max_iterations > 0:
        for i in range(max_iterations):
            thetas = thetas - alpha * derivative(X, y, thetas, ld)
        #for j in range(num_classes):
        #    for i in range(max_iterations):
        #        theta_j = thetas[:, j].reshape(num_parameters, -1)
        #        theta_j = theta_j - alpha * derivative(X, y[:, [j]], theta_j, ld)
        #        thetas[:, [j]] = theta_j

    if threshold > 0:
        thetas_prev = thetas + 1
        while np.max(np.abs(thetas_prev - thetas)) > threshold:
            thetas_prev = thetas
            thetas = thetas - alpha * derivative(X, y, thetas, ld)

    print("Theta:")
    print(thetas)
    return thetas


'''
predicts the label for the selected thetas. When multiclass is used, this will return a 1d-array thats mapped
like the dictionary returned by the transform_multiclass function.
'''


def predict(X, thetas, mc):
    p = sigmoid(X, thetas)
    if mc is False:
        return np.rint(p)
    elif mc is True:
        if thetas.shape[1] == 1:
            return np.rint(p)
        else:
            return np.argmax(p, 1).reshape(X.shape[0], -1)


'''
following function returns the accuracy of the predictions. The function takes the argument y_original, 
 because for multiclass it needs the class labels.
'''


def accuracy(predictions, y_original):
    return float(sum(predictions == y_original)/y_original.shape[0])


# starting to test:

X_ = np.array(
    [
        [1, 1, 2, 3],
        [1, 3, 2, 3],
        [1, 4, 5, 5],
        [1, 1, 2, 3]
    ]
)

y_ = np.array(
    [
        [1],
        [0],
        [0],
        [1]
    ]
).reshape((-1, 1))

test_df = pd.DataFrame(X_)
test_df.columns = ["a", "b", "c", "d"]

X__, y__, y_orig, mapped = transform_multiclass(test_df, "d", ["a", "b", "c"])

print("Binary with max iterations & no regularization")
test_thetas = gradient_descent_logistic(X_, y_, 0.01, max_iterations=10000)
preds = predict(X_, test_thetas, mc=False)
print(accuracy(preds, y_))

print("Binary with max iterations & regularization")
test_thetas2 = gradient_descent_logistic(X_, y_, 0.01, ld=0.05, max_iterations=10000)
preds2 = predict(X_, test_thetas2, mc=False)
print(accuracy(preds2, y_))

print("Binary with threshold & regularization:")
test_thetas2 = gradient_descent_logistic(X_, y_, 0.01, ld=0.05, threshold=10e-6)
preds2 = predict(X_, test_thetas2, mc=False)
print(accuracy(preds2, y_))

print("Multiclass with max iterations & regularization")
test_thetas3 = gradient_descent_logistic_mc(X__, y__, 0.01, ld=0.05, max_iterations = 10000)
preds3 = predict(X__, test_thetas3, mc=True)
print(accuracy(preds3, y_orig))

print("Multiclass with threshold & regularization")
test_thetas3 = gradient_descent_logistic_mc(X__, y__, 0.01, ld=0.05, threshold=10e-6)
preds3 = predict(X__, test_thetas3, mc=True)
print(accuracy(preds3, y_orig))
