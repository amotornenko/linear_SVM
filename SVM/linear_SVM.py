import numpy as np

class LinearSVM():

  def __init__(self, W=None):
    self.W = W

  def train(self, x, y, validate=False, x_val=None, y_val=None, learning_rate=1e-3, reg=1e-5, n_epochs=100,
            batch_size=200, log_period=-1, use_numpy=False):
    """
    

    Inputs:
    - x: (np.array) training data in shape (N,D): N - number of samples, D - size of each sample;
    - y: (np.array) discrete labels of shape (N);
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    (np.array) of losses in shape (num_iters) corresponding for each iteration
    """

    n_samples, D = x.shape
    n_classes = np.max(y) + 1 # number of classes, assume they start from 0
    if self.W is None:
      # randomly initialze W
      self.W = 0.001 * np.random.randn(D, n_classes)


    # here we store loss history    
    loss_history = []

    # iterate with Stochastic Gradien Descent to optimize W
    for it in range(n_epochs):
      idxs = np.random.choice(range(n_samples), size=batch_size, replace=True)
      x_batch = x[idxs, :]
      y_batch = y[idxs]

      # calculate loss and gradient simultaneously
      loss, grad = self.loss(x_batch, y_batch, reg, use_numpy=use_numpy)
      loss_history.append(loss)

      # perform parameter update
      self.W = self.W - grad * learning_rate

      if log_period > 0 and it % log_period == 0:
        print('iteration %d / %d: loss %f' % (it, n_epochs, loss))
        if validate:
          y_val_pred = self.predict(x_val)
          val_accuracy = np.mean(y_val == y_val_pred)
          print('validation accuracy %.2f' % (val_accuracy))

    return loss_history

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
    y_pred = np.argmax(scores, axis=-1)
    return y_pred

  def loss(self, x_batch, y_batch, reg, use_numpy = False):
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
    if use_numpy:
      return self.svm_loss_numpy(x_batch, y_batch, reg)
    else:
      return self.svm_loss_naive(x_batch, y_batch, reg)

  def svm_loss_naive(self, x, y, reg):
    """
    SVM Hinge loss function, naive implementation (with for loops).

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
    # initialize the gradient
    dW = np.zeros(self.W.shape)

    # compute the loss and the gradient
    n_classes = self.W.shape[1]
    n_samples = x.shape[0]
    loss = 0.0
    for i in range(n_samples):
      scores = x[i].dot(self.W)
      correct_class_score = scores[y[i]]
      for j in range(n_classes):
        if j == y[i]:
          continue
        margin = scores[j] - correct_class_score + 1

        if margin > 0:
          loss += margin

          # compute dW
          # loss = max(0, x*w_j - x*w_y + 1)
          # hence derivative is
          dW[:,j] += x[i]
          dW[:,y[i]] -= x[i]

    # Average loss by the number of samples in the mini-batch
    # i.e. divide by the number of samples, the same goes for dW 
    loss /= n_samples
    dW /= n_samples

    # Take regularization into account
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
    
    correct_lbl_indices = (np.arange(n_samples), y)
    correct_scores = scores[correct_lbl_indices]

    margins = np.maximum(0, scores - correct_scores.reshape(n_samples, 1) + 1)
    margins[correct_lbl_indices] = 0
    loss += np.sum(margins)

    # Average loss by the number of samples in the mini-batch
    # i.e. divide by the number of samples, the same goes for dW 
    loss /= n_samples

    # Add regularization to the loss.
    loss += reg * np.sum(self.W * self.W)


    #we calculate how many times each x contributes to loss
    n_entries = np.zeros(self.W.shape)
    n_entries[margins > 0] = 1

    # for each margin > 0 "correct" weight contributes "-1" time 
    row_sum = np.sum(n_entries, axis=1)
    n_entries[correct_lbl_indices] = -row_sum.T

    dW = np.dot(x.T, n_entries)

    # Average the gradient
    dW /= n_samples

    # Take regularization into account
    dW += 2 * reg * self.W

    return loss, dW
