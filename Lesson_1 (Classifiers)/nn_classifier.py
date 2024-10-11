import numpy as np

class NearestNeighbourClassifier:
    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])

    def fit(self, x, y):
        """ Fit the training data to the classifier.

        Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)
        y (np.ndarray): Class labels, numpy array with shape (N,)
        """
        self.x = x
        self.y = y

    def predict(self, x):
        """ Perform prediction given some examples.

        Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)

        Returns:
        y (np.ndarray): Predicted class labels, numpy array with shape (N,)
        """ 
        # TODO: Complete this method to predict the class of the 
        # nearest neighbour given a set of test instance
        y = np.zeros(len(x))
        
