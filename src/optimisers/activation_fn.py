import numpy
import math
def sigmoid(x_vect:list[float], derivative=False):
    op_vect = []
    if not derivative:
        for x in x_vect:
            op_vect.append(numpy.true_divide(1,1+math.exp(-x)))
    else:
        for x in x_vect:
            op_vect.append(numpy.true_divide(math.exp(-x),(1+math.exp(-x))**2))
    return op_vect

def sigmoid_derivative(x:float):
    raise Exception('Deprecated')

def softmax(vect: list[float], derivative=False):
    denom = numpy.sum([math.exp(x) for x in vect])
    op_vect = []
    if not derivative:
        for el in vect:
            op_vect.append(numpy.divide(math.exp(el),denom))
        return op_vect  
    
    d_denom = denom ** 2
    for el in vect:
        numerator = (denom - math.exp(el))*math.exp(el)
        op_vect.append(numpy.divide(numerator, d_denom))
    return op_vect

