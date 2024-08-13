# Download iris data if it does not exist
# ! runs a bash command, so !ls will run the ls command on the virtual machine
import os

if not os.path.exists("iris.data"):
    !wget -O iris.data https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data