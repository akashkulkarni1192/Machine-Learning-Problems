import pandas as pandu
import numpy as numpu
import mainrunner

import matplotlib.pyplot as plt

def normalize_feature(x):
    x_norm = numpu.matrix.copy(x)
    mean_x = numpu.mean(x_norm,0)
    std_x = numpu.std(x_norm,0)
    for i in range(1,len(x_norm[0])):
        x_norm[:,i] = (x[:,i] - mean_x[i])/std_x[i]
    return x_norm

def predict(theta):
    size = input('Enter size of house : ')
    no_of_bedrooms = input('Enter no of bedrooms : ')
    input_matrix = numpu.column_stack([1, float(size), float(no_of_bedrooms)])
    profit = numpu.dot(input_matrix,theta)
    return profit

def normalEq(x,y,theta):
    xTranspose = numpu.transpose(x)
    theta = numpu.dot(numpu.linalg.inv(numpu.dot(xTranspose,x)),numpu.dot(xTranspose,y))
    return theta

def run():
    names = ['size', 'no_of_bedrooms', 'profit']
    dataset = pandu.read_csv('ex1data2.txt', names=names)
    array = dataset.values

    # initialize values
    x = array[:, 0:2]  # get idea sim
    x = numpu.column_stack([numpu.ones(len(x)), x])
    y = array[:, 2]
    y = numpu.row_stack(y)

    n = len(x[0])
    theta = numpu.zeros(shape=(n, 1))
    x_norm = normalize_feature(x)

    # Gradient Descent
    GDtheta = mainrunner.gradient_descent(x_norm, theta, y, 400, 0.01)
    print("theta from gradient descent: \n", GDtheta)

    profit = predict(GDtheta)
    print("profit predicted by gradient descent: ", profit)

    # Normal Equation
    theta = numpu.zeros(shape=(n, 1))
    NEtheta = normalEq(x, y, theta)
    print("theta from normal equation: \n", NEtheta)

if __name__ == '__main__':
    run()
