import numpy
import math
def sigmoid(x:float):
    return numpy.true_divide(1,1+math.exp(-x))

def sigmoid_derivative(x:float):
    return numpy.true_divide(math.exp(-x),(1+math.exp(-x))**2)