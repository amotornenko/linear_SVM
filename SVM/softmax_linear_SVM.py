import numpy as np
from .linear_SVM import LinearSVM

class SoftMaxLinearSVM(LinearSVM):

  def predict(self, x):
    """
    Use the SVM linear model to make a prediction on the imput x.

    Inputs:
    - x: (np.array) data in shape (N, D) used for a rpediction; N - number of samples, D - size of each sample;

    Returns:
    - y_pred: (np.array) predicted labels for the data in x; y_pred is a vector of size (N), each element 
    corresponds to a prediction.
    """
    y_pred = np.zeros(x.shape[0])

    scores = x.dot(self.W)
    exp = np.exp(scores)
    probab = exp/np.sum(exp)
    y_pred = np.argmax(probab, axis=-1)
    return y_pred

  def loss(self, x, y, reg, use_numpy = False, reg_type=2):
    """
    Computes the loss function and its grtadient.

    Inputs:
    - x_batch: (np.array) data in shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: (np.array) a numpy vector of shape (N) containing labels for the minibatch.
    - reg: (float) regularization strength.
    - use_numpy: (bool) use numpy arrays for a speed-up (vectorization).

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """

    return self.svm_loss_numpy(x,y,reg)

    n_samples, D = x.shape
    n_classes = np.max(y) + 1 # number of classes, assume they start from 0

    if self.W is None:
      # randomly initialze W
      self.W = 0.001 * np.random.randn(D, n_classes)

    scores = x.dot(self.W)
    exp = np.exp(scores)
    probab = exp/(np.sum(exp, axis=1)).reshape((exp.shape[0],1))

    correct_lbl_indices = (np.arange(n_samples), y)
    correct_probab = probab[correct_lbl_indices]

    loss = -np.log(correct_probab)

    loss = np.sum(loss)/n_samples

    dW = np.zeros(self.W.shape)

    gradient = probab
    gradient[np.arange(n_samples),y] -= 1

    dW = x.T.dot(gradient)

    # naive loss
    # for i in range(n_samples):
    #   scores = x[i].dot(self.W)

    #   p = np.exp(scores)
    #   p = p/np.sum(p)

    #   gradient = p.reshape(1,-1)
    #   gradient[0, y[i]] += -1

    #   dW += x[i].reshape(-1,1).dot(gradient)

    dW /= n_samples

    loss += reg * np.sum(self.W * self.W)
    dW += 2 * reg * self.W

    return loss, dW

  def svm_loss_numpy(self, x, y, reg):
    """
    SVM Hinge loss function, using numpy arrays.
    Provides a speed-up because of the numpy array vectorisation. 

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - x: (np.array) data in shape (N, D) containing a minibatch of data.
    - y: (np.array) data in shape (N,) containing training labels; y[i] = c means
      that x[i] has label c, where 0 <= c < C.
    - reg: (float) regularization factor

    Returns a tuple of:
    - (float) loss as single float
    - (np.array) gradient d(loss)/dW of the same shape as W
    """
    n_samples = x.shape[0]
    loss = 0.0
    dW = np.zeros(self.W.shape) # initialize the gradient as zero

    scores = x.dot(self.W)
    
    #print(scores, scores.shape)
    
    correct_lbl_indices = (np.arange(n_samples), y)
    correct_scores = scores[correct_lbl_indices]

    exp = np.exp(scores)
    expSum        = np.sum(exp, axis=1)
    pCorrect      = np.exp(correct_scores) / expSum
    loss         -= np.sum(np.log(pCorrect))
    

    # Average loss by the number of samples in the mini-batch
    # i.e. divide by the number of samples, the same goes for dW 
    loss /= n_samples

    # Add regularization to the loss.
    loss += reg * np.sum(self.W*self.W)

    probab = exp/(expSum).reshape((exp.shape[0],1))

    correct_lbl_indices = (np.arange(n_samples), y)
    correct_probab = probab[correct_lbl_indices]

    loss = -np.log(correct_probab)

    loss = np.sum(loss)/n_samples

    dW = np.zeros(self.W.shape)

    gradient = probab
    gradient[correct_lbl_indices] -= 1

    dW = x.T.dot(gradient)

    # Average the gradient
    dW /= n_samples

    # Take regularization into account
    dW += 2 * reg * self.W
    

    return loss, dW