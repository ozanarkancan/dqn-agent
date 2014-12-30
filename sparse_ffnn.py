import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def KL_divergence(x, y):
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


def initialize(input_size, hidden_size, output_size):
    # we'll choose weights uniformly from the interval [-r, r]
    r1 = np.sqrt(6) / np.sqrt(hidden_size + input_size + 1)
    r2 = np.sqrt(6) / np.sqrt(hidden_size + output_size + 1)
    W1 = np.random.random((hidden_size, input_size)) * 2 * r1 - r1
    W2 = np.random.random((output_size, hidden_size)) * 2 * r2 - r2

    b1 = np.zeros(hidden_size, dtype=np.float64)
    b2 = np.zeros(output_size, dtype=np.float64)

    theta = np.concatenate((W1.reshape(input_size * hidden_size),
                            W2.reshape(hidden_size * output_size),
                            b1.reshape(hidden_size),
                            b2.reshape(output_size)))

    return theta


# input_size: the number of input units
# hidden_size: the number of hidden units
# output_size: the number of output units (number of actions)
# lambda_: weight decay parameter
# sparsity_param: The desired average activation for the hidden units (denoted in the lecture
#                            notes by the greek alphabet rho, which looks like a lower-case "p").
# beta: weight of sparsity penalty term
# data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example.
#
# The input theta is a vector (because minFunc expects the parameters to be a vector).
# We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
# follows the notation convention of the lecture notes.
# Returns: (cost,gradient) tuple
def cost(theta, input_size, hidden_size, output_size,
                            lambda_, sparsity_param, beta, data, output):
    # The input theta is a vector (because minFunc expects the parameters to be a vector).
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.

    W1 = theta[0:input_size * hidden_size].reshape(hidden_size, input_size)
    W2 = theta[input_size * hidden_size:input_size * hidden_size + hidden_size * output_size].reshape(output_size, hidden_size)
    b1 = theta[input_size * hidden_size + hidden_size * output_size:input_size * hidden_size + hidden_size * output_size + hidden_size]
    b2 = theta[input_size * hidden_size + hidden_size * output_size + hidden_size:]

    # Number of training examples
    m = data.shape[1]

    # Forward propagation
    z2 = W1.dot(data) + np.tile(b1, (m, 1)).transpose()
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + np.tile(b2, (m, 1)).transpose()
    h = sigmoid(z3)

    # Sparsity
    rho_hat = np.sum(a2, axis=1) / m
    rho = np.tile(sparsity_param, hidden_size)

    # Cost function
    cost = np.sum((h - output) ** 2) / (2 * m) + \
           (lambda_ / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2)) + \
           beta * np.sum(KL_divergence(rho, rho_hat))

    # Backprop
    sparsity_delta = np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), (m, 1)).transpose()

    delta3 = -(output - h) * sigmoid_prime(z3)
    delta2 = (W2.transpose().dot(delta3) + beta * sparsity_delta) * sigmoid_prime(z2)
    W1grad = delta2.dot(data.transpose()) / m + lambda_ * W1
    W2grad = delta3.dot(a2.transpose()) / m + lambda_ * W2
    b1grad = np.sum(delta2, axis=1) / m
    b2grad = np.sum(delta3, axis=1) / m

    # After computing the cost and gradient, we will convert the gradients back
    # to a vector format (suitable for minFunc).  Specifically, we will unroll
    # your gradient matrices into a vector.
    grad = np.concatenate((W1grad.reshape(input_size * hidden_size),
                           W2grad.reshape(hidden_size * output_size),
                           b1grad.reshape(hidden_size),
                           b2grad.reshape(output_size)))

    return cost, grad


def predict(theta, input_size, hidden_size, output_size, data):
    """
    :param theta: trained weights from the network
    :param hidden_size: the number of input units
    :param hidden_size: the number of hidden units
    :param output_size: the number of output units (number of actions)
    :param data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example.
    """

    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.

    W1 = theta[0:input_size * hidden_size].reshape(hidden_size, input_size)
    W2 = theta[input_size * hidden_size:input_size * hidden_size + hidden_size * output_size].reshape(output_size, hidden_size)
    b1 = theta[input_size * hidden_size + hidden_size * output_size:input_size * hidden_size + hidden_size * output_size + hidden_size]
    b2 = theta[input_size * hidden_size + hidden_size * output_size + hidden_size:]

    # Number of training examples
    m = data.shape[1]

    # Forward propagation
    z2 = W1.dot(data) + np.tile(b1, (m, 1)).transpose()
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + np.tile(b2, (m, 1)).transpose()
    h = sigmoid(z3)

    return h
