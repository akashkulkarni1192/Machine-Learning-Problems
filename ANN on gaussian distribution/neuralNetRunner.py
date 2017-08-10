import numpy as np
import matplotlib.pyplot as plt


def create_dataset(n_input):
    X1 = np.random.randn(n_input, 2) + np.array([0, -2])
    X2 = np.random.randn(n_input, 2) + np.array([2, 2])
    X3 = np.random.randn(n_input, 2) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])
    Y = np.array([0] * n_input + [1] * n_input + [2] * n_input)
    return X, Y


def plot_graph(X, Y):
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
    plt.show()


def configure_nn_architecture(X):
    N, D = X.shape
    M = 3
    K = 3
    return D, M, K


def initialize_weights(D, M, K):
    W1 = np.random.randn(D, M)
    B1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    B2 = np.random.randn(K)
    return W1, B1, W2, B2


def sigmoid(z):
    Z = 1 / (1 + np.exp(-z))
    return Z


def softmax(a):
    expA = np.exp(a)
    A = expA / np.sum(expA, axis=1, keepdims=True)
    return A


def feed_forward_propogate(X, W1, B1, W2, B2):
    A_hidden = X.dot(W1) + B1
    Z = sigmoid(A_hidden)
    A_final = Z.dot(W2) + B2
    P = softmax(A_final)
    return P, Z


def calculate_accuracy(P, Y):
    n_correct = 0
    n_total = 0

    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1

    return float(n_correct) / n_total


def display_accuracy(P, Y):
    P_output = np.argmax(P, axis=1)
    accuracy = calculate_accuracy(P_output, Y)
    print('Accuracy : {0} '.format(accuracy))


def calculate_cost(T, Y):
    cost = T * np.log(Y)
    return np.sum(cost)


def derivative_W2(T, Y, Z):
    gradient = Z.T.dot(T - Y)
    return gradient


def derivative_B2(T, Y):
    gradient = (T - Y).sum(axis=0)
    return gradient


def derivative_W1(W2, X, Y, Z, T):
    N, M = Z.shape
    N, D = X.shape
    gradient = np.random.randn(D, M)
    # for n in range(N):
    #     for k in range(K):
    #         for m in range(M):
    #             for d in range(D):
    #                 gradient[d, m] += ((T[n, k] - Y[n, k]) * W2[m, k] * Z[n, m] * (1 - Z[n, m]) * X[n, d])

    #attempt2
    gradient = X.T.dot((T - Y).dot(W2.T) * Z * (1 - Z))
    return gradient

def derivative_B1(T, Y, Z, W2):
    gradient = (T - Y).dot(W2.T) * Z * (1 - Z)
    return np.sum(gradient, axis=0)


def back_propogate(W1, B1, W2, B2, X, Y_actual, D, M, K):
    N = len(Y_actual)

    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y_actual[i]] = 1

    costs = []
    total_gradient_steps = 100000
    alpha = 0.001
    for step in range(total_gradient_steps):
        Y, Z = feed_forward_propogate(X, W1, B1, W2, B2)
        if (step % 100 == 0):
            cost = calculate_cost(T, Y)
            costs.append(cost)
            #print("Cost : {0}".format(cost))
            display_accuracy(Y, Y_actual)
        W2 += alpha*derivative_W2(T, Y, Z)
        B2 += alpha*derivative_B2(T, Y)
        W1 += alpha*derivative_W1(W2, X, Y, Z, T)
        B1 += alpha*derivative_B1(T, Y, Z, W2)
    plt.plot(costs)
    plt.show
    return W2, B2, W1, B1

if __name__ == '__main__':
    ninputs_per_class = 500  # number of input
    X, Y = create_dataset(ninputs_per_class)
    plot_graph(X, Y)
    D, M, K = configure_nn_architecture(X)
    W1, B1, W2, B2 = initialize_weights(D, M, K)
    W2, B1, W2, B2 = back_propogate(W1, B1, W2, B2, X, Y, D, M, K)
    print("End")

