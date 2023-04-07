"""
Author: 201901809
Description: Implementation of gradient descent algorithm
             based on 2D random data, that resembles
             linear scatter plot 
"""

import random
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd

"""
#1. Implementation of first requirement.

# This code is highlighted because the random data
# is already generated and stored in random_data.csv. 
# To generate new, remove the multi-line comments. 

# Define the slope and y-intercept of the line
m = 2
b = 5

# Define the number of data points to generate
num_points = 30

# Generate the x and y values
x_values = []
y_values = []
for i in range(num_points):
    x = random.uniform(0, 10)
    y = m * x + b + random.uniform(-1, 1)
    x_values.append(x)
    y_values.append(y)

# Create file to store the data points in .csv file, as per requirement
"""

# This file stores the random data points after they are generated below
filename = 'random_data.csv'

"""

# Open file in write mode(w) 
with open(filename, 'w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)

    # Write data to the file
    writer.writerow(['X', 'Y'])
    for i in range(len(x_values)):
        writer.writerow([x_values[i], y_values[i]])

print(f'{filename} created successfully!') 
"""

# 2. Gradient Descent algorithm implementation 

# Define the learning rate and number of iterations
learning_rate = 0.01
num_iterations = 1000

# Define the initial values for the parameters(c and m)
w_0 = 0.0
w_1 = 0.0

# Load the data from the CSV file
data = []
with open('random_data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip header row
    for row in reader:
        data.append([float(row[0]), float(row[1])])

# Define the cost function
def cost_function(w_1, w_0, data):
    total_cost = 0
    for point in data:
        x, y = point
        y_pred = w_1 * x + w_0
        total_cost += (y_pred - y) ** 2
    return total_cost / len(data)

# Define the gradient function
def gradient_function(w_1, w_0, data):
    grad_m = 0
    grad_c = 0
    for point in data:
        x, y = point
        y_pred = w_1 * x + w_0
        grad_m += 2 * (y_pred - y) * x
        grad_c += 2 * (y_pred - y)
    grad_m /= len(data)
    grad_c /= len(data)
    return grad_m, grad_c

# Define the gradient descent algorithm
def gradient_descent(data, learning_rate, num_iterations):
    w_1 = 0
    w_0 = 0
    for i in range(num_iterations):
        grad_m, grad_c = gradient_function(w_1, w_0, data)
        w_1 -= learning_rate * grad_m
        w_0 -= learning_rate * grad_c
    return w_1, w_0

w_1, w_0 = gradient_descent(data, learning_rate, num_iterations)

# 3. Displaying the random data points along with the line of best hypothesis

# Loading data from csv
data = pd.read_csv(filename)

# Extracting the data point
x_values = data['X'].values
y_values = data['Y'].values

# Print the parameters of the linear regression model
print(f"w_0 = {w_0}, w_1 = {w_1}")

# Compute the predicted values
predicted_values = w_0 + w_1 * x_values

# Plot the data points and the linear regression line
plt.scatter(x_values, y_values, color='purple', label='random data points')
plt.plot(x_values, predicted_values, color='yellow', label='best hypothesis')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.title('2D Linear Regression Model')
plt.show()

