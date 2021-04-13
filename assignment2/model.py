import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        # Create necessary layers
        self.reg = reg
        self.FCL1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.RL1 = ReLULayer()
        self.FCL2 = FullyConnectedLayer( hidden_layer_size, n_output)
        
    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        # Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        #for param in self.params().values():
        #    param.grad = 0
        
        out = self.FCL1.forward(X)
        out = self.RL1.forward(out)
        out = self.FCL2.forward(out)
        
        loss, grad_loss = softmax_with_cross_entropy(out, y)
        
        out = self.FCL2.backward(grad_loss)
        out = self.RL1.backward(out)
        out = self.FCL1.backward(out)
     
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        loss_reg, l2_grad = l2_regularization(self.params()["W1"].value,self.reg)
        
        
        self.params()["W1"].grad += l2_grad
        loss += loss_reg
        
        loss_reg, l2_grad = l2_regularization(self.params()["W2"].value,self.reg)
        
        self.params()["W2"].grad += l2_grad
        loss += loss_reg

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        
        pred = np.dot(X,self.params()["W1"].value) + self.params()["B1"].value
        pred= np.where(pred>0, pred,0)
        pred = np.dot(pred,self.params()["W2"].value) + self.params()["B2"].value
        
        pred =softmax(pred)
        pred = np.argmax(pred, axis = 1)
        
        return pred

    def params(self):

        # Implement aggregating all of the params
        result = {
            "W1": self.FCL1.params()["W"], 
            "B1": self.FCL1.params()["B"], 
            "W2": self.FCL2.params()["W"], 
            "B2": self.FCL2.params()["B"] }
        return result
