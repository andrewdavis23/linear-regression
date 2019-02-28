import numpy as np
import math

def splitData(fileName,ratio):

    data = np.genfromtxt(fileName, dtype=float, comments='#', delimiter=',')

    partition = math.ceil(data.shape[0] * ratio)

    np.random.shuffle(data)

    trainSet, testSet = data[:partition,:], data[partition:,:]

    return trainSet, testSet
