##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
    A0 = random.random((m, n))

    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0