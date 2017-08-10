import pandas as pandu
import numpy as numpu
import matplotlib.pyplot as plt


def compute_cost(x_matrix, theta, y_matrix):
    hypo_matrix = numpu.dot(x_matrix, theta)
    cost = 0.5 * 97 * numpu.sum(numpu.square(numpu.subtract(hypo_matrix, y_matrix)), axis=0)
    print(cost)


def gradient_descent(x_matrix, theta, y_matrix, num_of_iter, alpha):
    m = len(x_matrix)
    temp_theta = numpu.matrix.copy(theta)
    for i in range(num_of_iter):
        hypo_matrix = numpu.dot(x_matrix, theta)
        for j in range(len(theta)):
            theta[j][0] = temp_theta[j][0] - alpha / m * numpu.sum(
                numpu.multiply(numpu.subtract(hypo_matrix, y_matrix), numpu.row_stack(x_matrix[:, j])))
        temp_theta = numpu.matrix.copy(theta)
    return theta


def predict(input_value, theta):
    input_matrix = numpu.column_stack([numpu.ones(1), input_value])
    output_value = numpu.dot(input_matrix, theta)
    return output_value


def plot_graph(real_theta, x, y):
    plt.xlabel('Population')
    plt.ylabel('Profit')
    plt.plot(x[:, 1], numpu.dot(x, real_theta))
    plt.scatter(x[:, 1], y, edgecolors='red')
    plt.show()


def run():
    names = ['population', 'profit']
    dataset = pandu.read_csv('ex1data1.txt', names=names)
    array = dataset.values

    # initialize values
    x = array[:, 0]  # get idea sim
    x = numpu.column_stack([numpu.ones(97), x])
    y = array[:, 1]
    y = numpu.row_stack(y)
    theta = numpu.zeros(shape=(2, 1))

    compute_cost(x, theta, y)
    real_theta = gradient_descent(x, theta, y, 1500, 0.01)
    print(real_theta)

    plot_graph(real_theta, x, y)

    input_value = input('Enter population input : ')
    output_value = predict(float(input_value), real_theta)
    print('Output profile : ' + str(output_value[0][0] * 10000))


if __name__ == '__main__':
    run()
