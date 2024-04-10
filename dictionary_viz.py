from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import random

mnist = fetch_openml('mnist_784', version=1)
#X = mnist.data.to_numpy()
X, y = mnist.data.to_numpy(), mnist.target.to_numpy()

# Select ONLY the 7s
X_7 = X[y == '7']

# Reshape the images to 28x28
X = X.reshape(-1, 28, 28)

def select_random_squares(image, seed):
    random.seed(seed)
    rows = list(range(24))
    cols = list(range(24))
    random.shuffle(rows)
    random.shuffle(cols)
    indices = list(zip(rows, cols))
    squares = []
    for row, col in indices:
        square = image[row:row+5, col:col+5]
        squares.append(square.flatten())
    return np.array(squares)

random_seed = 12

# Create the MNIST5x5 dataset
MNIST5x5 = np.array([select_random_squares(image, random_seed) for image in X])
MNIST5x5 = MNIST5x5.reshape(-1, 25)

# Normalize the data
#MNIST5x5 = MNIST5x5 / 255.0

n_clusters = 400
kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
kmeans.fit(MNIST5x5)

# Get the cluster centers (dictionary)
dictionary = kmeans.cluster_centers_

# Reshape the dictionary for visualization
dictionary_reshaped = dictionary.reshape(n_clusters, 5, 5)

# Visualize the dictionary
plt.figure(figsize=(20, 20))
for i in range(n_clusters):
    plt.subplot(20, 20, i + 1)
    plt.imshow(dictionary_reshaped[i], cmap='gray', interpolation='nearest')
    plt.axis('off')
plt.tight_layout()
plt.show()
