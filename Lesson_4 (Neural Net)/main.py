from numpy.random import default_rng
import numpy as np

x_train = np.array([1.0, 1.2, 2.0, 3.5, 4.0, 5.0])
y_train = np.array([3.1, 3.5, 5.0, 7.9, 9.1, 10.9])
x_test = np.array([2.5, 3.0, 4.5])
y_test = np.array([6.0, 7.0, 10.1])

class SimpleLinearRegression:
    def __init__(self, random_generator=default_rng()):
        # initialise the slope with a random value drawn from a standard normal 
        # distribution (mean=0, stddev=1)
        self.w = random_generator.standard_normal()

        # initialise bias to 0 
        self.b = 0

    def forward(self, x):
        """ Perform forward pass given an input x

        Args:
            x (float): input instance

        Returns:
            float: the output of the model given the current weights
        """

        return self.w * x + self.b

    def loss(self, x, y):
        """ Simulating loss functions """
        y_hat = self.forward(x)
        return (y_hat - y)**2

    def gradient(self, x, y):
        """ Compute partial derivatives wrt w and b

        Args:
            x (float): input instance
            y (float): ground truth output

        Returns:
            tuple: (float, float)
                - the first element will be dL/dw
                - the second element will be dL/db
        """
        y_hat = self.forward(x)
        return ((y_hat - y) * x, (y_hat - y))

model = SimpleLinearRegression()

learning_rate = 0.01
n_epochs = 100

for epoch in range(n_epochs):
    error = 0.0
    grad_w = 0.0
    grad_b = 0.0
    for (x, y) in zip(x_train, y_train):
        (dLdw, dLdb) = model.gradient(x, y)
        grad_w += dLdw
        grad_b += dLdb
        error += model.loss(x, y)
    model.w = model.w - learning_rate * grad_w
    model.b = model.b - learning_rate * grad_b
    print(f"Epoch: {epoch}\t w: {model.w:.2f}\t b: {model.b:.2f}\t L: {error:.4f}")


# to store all losses for later use
losses = []

# the parameters to search
weights = np.arange(0, 4.1, 0.2) 
biases = np.arange(0, 2.1, 0.2)

# for storing the loss in a matrix for visualisation later
loss_matrix = np.zeros((len(weights), len(biases)))

# compute loss for each (w,b) combination
for i, w in enumerate(weights):
    for j, b in enumerate(biases):
        print(f"(w={w:.1f}, b={b:.1f})")
        
        # setup weights of model
        model.w = w
        model.b = b

        sum_loss = 0
        # for each example
        for (x, y) in zip(x_train, y_train):
            # compute the loss for this example
            single_loss = model.loss(x, y)

            # and add it to the sum
            sum_loss += single_loss

            # print out the values just to make sure everything is working correctly
            y_hat = model.forward(x)
            print(f"    x: {x}, y: {y}, y_hat: {y_hat:.1f}, loss: {single_loss:.2f}")

        # print out the sum of individual losses
        # I multiplied by 0.5 to be consistent with the equation earlier, 
        # but this is not necessary in practice as this is a constant
        print(f"    Loss = {(0.5 * sum_loss):.4f}\n")

        # store the losses and the corresponding (w,b) for later use
        losses.append((0.5*sum_loss, w, b))

        # store the losses in a matrix form for visualisation later
        loss_matrix[i,j] = 0.5 * sum_loss

# find combination with minimum loss
(min_loss, best_w, best_b) = min(losses, key=lambda x:x[0])
print("BEST:")
print(f"w={best_w}, b={best_b}, loss={min_loss:.4f}")