import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

# Generate a toy dataset
# There is no need to understand the details,
# but I'm generating a random dataset from 4*x_1 + 2.5*x_2 + 1.5
# and adding some noise to the output

seed = 60012
rg = default_rng(seed)
weights = np.array([4, 2.5, 1.5])
n_samples = 100
x = rg.random((n_samples, 2))*10.0
x = np.hstack((x, np.ones((n_samples, 1))))
y = np.matmul(x, weights)

# add noise to y
# comment these out if you want to work with a perfectly clean dataset
noise = rg.standard_normal(y.shape)
y = y + noise

x_train = x[:80, :2]
y_train = y[:80]
x_test = x[80:, :2]
y_test = y[80:]

# Plot the training set
fig = plt.figure()
ax = fig.add_subplot(projection='3d') # enable 3D
ax.scatter(x_train[:,0], x_train[:,1], y_train, c="red")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel('y')

# Plot the plane - you are aiming for your algorithm to recover this plane
fig = plt.figure()
ax = fig.add_subplot(projection='3d') # enable 3D
x_plane = np.linspace(0,10,10)
y_plane = np.linspace(0,10,10)
x_plane, y_plane = np.meshgrid(x_plane, y_plane)
z = weights[0] * x_plane + weights[1] * y_plane + weights[2]
surf = ax.plot_surface(x_plane, y_plane, z)

plt.show()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class MultipleLinearRegression:
    def __init__(self, n_input_vars, random_generator=default_rng()):
        """ Constructor

        Args:
            n_input_vars (int): Number of features (including bias)
            random_generator (RandomGenerator): A random generator
        """

        # we include the bias as an additional weight here
        self.w = random_generator.standard_normal(n_input_vars)
        self.w[-1] = 0. # set the bias to 0

    def forward(self, x):
        """ Perform forward pass given an input x

        Args:
            x (np.ndarray): shape (N, K) where
                            - N is the number of instances,
                            - K is the number of features (including bias)

        Returns:
            np.ndarray: the output of the model given the current weights
        """

        return np.matmul(x, self.w) # matmul performs dot product
        # return x @ self.w  # same as above
        # return np.matmul(self.w.T, x.T)  # alternative solution

    def loss(self, x, y):
        """ Compute the loss for an input x

        Args:
            x (np.ndarray): shape (N, K) where
                            - N is the number of instances,
                            - K is the number of features (including bias)
            y (np.ndarray): shape (N, ), the ground truth output labels for
                            each of the instance in x

        Returns:
            np.ndarray: shape (N,), the output of the model given the current weights
                        for each instance in x
        """

        y_hat = self.forward(x)
        return (y_hat - y)**2

    def gradient(self, x, y):
        """ Compute partial derivatives wrt w and b

        Args:
            x (np.ndarray): shape (N, K) where
                            - N is the number of instances,
                            - K is the number of features (including bias)
            y (np.ndarray): shape (N, ), the ground truth output labels for
                            each of the instance in x

        Returns:
            np.ndarray: shape (N, K) containing the partial derivatives
                            wrt the K weights, for each N instance
        """
        y_hat = self.forward(x)
        diff = y_hat - y
        grad = x * diff[:, np.newaxis]
        return grad

n_input = 2
model = MultipleLinearRegression(n_input+1, rg)

learning_rate = 0.0001
n_epochs = 1000

# concat the constant 1 to each instance for the bias
x_train_ext = np.hstack((x_train, np.ones((x_train.shape[0], 1))))

for epoch in range(n_epochs):
    error = 0.5 * model.loss(x_train_ext, y_train).sum()
    grad = model.gradient(x_train_ext, y_train).sum(axis=0)
    model.w = model.w - learning_rate * grad
    print(f"Epoch: {epoch}\t w: {model.w}\t L: {error:.4f}")