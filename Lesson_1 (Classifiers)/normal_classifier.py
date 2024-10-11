# Download iris data if it does not exist
# ! runs a bash command, so !ls will run the ls command on the virtual machine
import os
import numpy as np
import requests
import matplotlib.pyplot as plt
from utils import *

if not os.path.exists("iris.data"):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    response = requests.get(url)

    with open("iris.data", "wb") as file:
        file.write(response.content)
    
# Print the file contents
for line in open("iris.data"):
    print(line.strip())

(x, y, classes) = read_dataset("iris.data")
print(x.shape)
print(y.shape)
print(classes)

for class_label in np.unique(y):
    print("\nClass", classes[class_label])
    x_class = x[y == class_label]
    print(x_class.min(axis=0))
    print(x_class.max(axis=0))
    print(x_class.mean(axis=0))
    print(np.median(x_class, axis=0))
    print(x_class.std(axis=0))

feature_names = ["Sepal length", "Sepal width", "Petal length", "Petal width"]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid of subplots

# First subplot: Sepal Length vs Sepal Width
axs[0, 0].scatter(x[:, 0], x[:, 1], c=y)
axs[0, 0].set_xlabel(feature_names[0])
axs[0, 0].set_ylabel(feature_names[1])
axs[0, 0].set_title('Sepal Length vs Sepal Width')

# Second subplot: Sepal Width vs Petal Length
axs[0, 1].scatter(x[:, 1], x[:, 2], c=y)
axs[0, 1].set_xlabel(feature_names[1])
axs[0, 1].set_ylabel(feature_names[2])
axs[0, 1].set_title('Sepal Width vs Petal Length')

# Third subplot: Petal Length vs Petal Width
axs[1, 0].scatter(x[:, 2], x[:, 3], c=y)
axs[1, 0].set_xlabel(feature_names[2])
axs[1, 0].set_ylabel(feature_names[3])
axs[1, 0].set_title('Petal Length vs Petal Width')

# Fourth subplot: Sepal Length vs Petal Width
axs[1, 1].scatter(x[:, 0], x[:, 3], c=y)
axs[1, 1].set_xlabel(feature_names[0])
axs[1, 1].set_ylabel(feature_names[3])
axs[1, 1].set_title('Sepal Length vs Petal Width')


plt.tight_layout()

current_dir = os.getcwd()
filename = os.path.join(current_dir, 'Lesson_1/iris_plots.png')
plt.savefig(filename)

# Show the plots
plt.show()