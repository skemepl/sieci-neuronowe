##SOM (Self-Organizing Map).
## wykorzystane materiały: https://www.geeksforgeeks.org/python/self-organising-maps-kohonen-maps/

import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
x = iris['data'] # 4 cechy
y = iris['target'] # 3 grupy
feature_names = iris['feature_names']
target_names = iris['target_names']

print("Liczba próbek:", x.shape[0])
print("Liczba cech:", x.shape[1])
print("Nazwy cech:",feature_names)
print("Klasy:", target_names)
print("Rozkład klas:", np.bincount(y))

x_min = x.min(axis=0)
x_max = x.max(axis=0)
x_norm = (x - x_min) / (x_max - x_min)

n_inputs = x_norm.shape[1]
n_neurons = 3

weights = np.random.rand(n_neurons, n_inputs)
print("Wagi początkowe:\n", weights)

def euclidean_distance(x, w):
    return np.sum((x - w) ** 2)

def find_winner(x, weights):
    distances = []
    for j in range(weights.shape[0]):
        d = euclidean_distance(x, weights[j])
        distances.append(d)
    winner_index = np.argmin(distances)
    return winner_index

def update_weights(weights, x, winner_index, learning_rate):
    weights[winner_index] = weights[winner_index] + learning_rate * (x - weights[winner_index])
    return weights

n_epochs = 50
learning_rate = 0.5

for epoch in range(n_epochs):
    learning = learning_rate * (1 - epoch / n_epochs)

    for i in range(x_norm.shape[0]):
        sample = x_norm[i]
        winner = find_winner(sample, weights)
        weights = update_weights(weights, sample, winner, learning)

    print(f"Epoka {epoch+1}/{n_epochs}, lr={learning:.3f}")

assigments = []

for i in range(x_norm.shape[0]):
    x = x_norm[i]
    winner = find_winner(x, weights)
    assigments.append(winner)

assigments = np.array(assigments)

for j in range(n_neurons):
    print(f"Neuron {j}:")
    indices = np.where(assigments == j)[0]
    print("  liczba próbek:", len(indices))
    print("  rozkład klas:", np.bincount(y[indices]))