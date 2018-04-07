pass
from stats232a.layers import *


def fc_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    out, cache = None, None
    
    ###########################################################################
    # TODO: Implement fc-relu forward pass.                                   #
    ###########################################################################
    out_fc,cache_1 = fc_forward(x,w,b)
    out,cache_2 = relu_forward(out_fc)
    cache = [cache_1,cache_2]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return out, cache


def fc_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    dx, dw, db = None, None, None
    
    ###########################################################################
    # TODO: Implement the fc-relu backward pass.                              #
    ###########################################################################
    out_relu_b= relu_backward(dout,cache[1])
    dx,dw,db = fc_backward(out_relu_b,cache[0])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return dx, dw, db
