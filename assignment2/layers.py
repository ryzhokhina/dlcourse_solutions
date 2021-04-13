import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """

    loss = reg_strength*(np.sum(W**2))
    grad = 2*reg_strength*W

    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # Implement softmax
    if predictions.ndim == 1:
        f = predictions-predictions.max()
        return np.exp(f)/np.sum(np.exp(f))
    else:
        max = predictions.max(axis = 1)
        max= max[:,None]
        f = predictions-max
        sums = np.sum(np.exp(f),axis = 1)
        return np.exp(f)/sums[:,None]
    

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # Implement cross-entropy
    # Your final implementation shouldn't have any loops
    
    if probs.ndim == 1:
        return -np.log(probs[target_index])
    else:
        batch_size = probs.shape[0]
        hpqs = (-np.log(probs[range(batch_size),target_index.T])).mean()
        return hpqs

def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    p = softmax(predictions)
    dprediction = p.copy()
    if predictions.ndim == 1:
        dprediction[target_index] -= 1.0
    else:
        batch_size=predictions.shape[0]
        dprediction[range(batch_size),target_index.T] -= 1.0
        dprediction=dprediction/batch_size
    loss = cross_entropy_loss(p,target_index)
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = X.copy()
        y = np.where(self.X>0, self.X,0)
        return y

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        grad = np.where(self.X>0, 1,0)
        d_result = grad*d_out
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X.copy()
        Z = np.dot(X,self.W.value)+ self.B.value
        return Z

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.W.grad = np.dot( self.X.T, d_out)
        self.B.grad = np.dot(np.ones([1, d_out.shape[0]]), d_out)
        
        d_result = np.dot(d_out, self.W.value.T)
        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}
