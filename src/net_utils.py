'''
@Author: Dat Nguyen 
@Version of Date: 2019. 
Code inspired by course cs231n
'''

import numpy as np


################################################################
################################################################
"""Outline
affind_forward()
affind_backward() 
relu_forward()
relu_backward()
batchnorm_forward()
batchnorm_backward()
layernorm_forward()
layernorm_backward()
dropout_forward()
dropout_backward()
"""
################################################################
################################################################
def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.(Hidden layer size)

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = x.reshape(x.shape[0], -1).dot(w) + b #reshape into vector for ease of computing 
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    '''mynote
    dout/dy = dout(variable name)
    y = x dot w + b 
    dout/dx = dout/dy * dy/dx(upstream * local grad) 
        dy/dx = w
    dout/db = dout/dy * dy/db = dout * 1.0
    dout/dw = dout/dy * dy/dw
    '''
    x_shape = x.shape
    x = x.reshape(x.shape[0],-1) # turn input to vector

    y = x.dot(w) #Shape: (N x M) same as dout

    #We want to retain spatial info and shape so reshape it 
    dx = dout.dot(w.T).reshape(x_shape) #(NxM) dot (MxD) = (N x D) output.
    dw = x.T.dot(dout) #(DxN) dot (NxM) = (DxM)
    db = np.sum(dout * 1.0, axis= 0)#Stack row
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    relu = lambda x : np.maximum(0,x)
    out = relu(x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = dout
    dx[x <= 0.0] = 0    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(x, axis=0) #Shape: (1 x D) or (D,)
        sample_var = np.var(x, axis=0) #Shape (1 x D) or (D,)
        sample_norm = (x - sample_mean) / np.sqrt(sample_var + eps) #SHape (N x D)

        #scale and shift back with learnable parameter
        out = gamma*sample_norm + beta #Shape N x D 

        #Update running avg var
        # running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        # running_var = momentum * running_var + (1 - momentum) * sample_var
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var

        #Need cast for backward pass. No need to recompute sample_norm
        cache = (x,sample_mean, sample_var, eps, sample_norm, gamma)
    elif mode == 'test':
        test_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * test_norm + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    
    

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

def batchnorm_backward(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    """
    dx, dgamma, dbeta = None, None, None
    x,sample_mean, sample_var, eps, sample_norm, gamma = cache # unpack
    x_hat = sample_norm
    mu_s = sample_mean
    var_s = sample_var
    N, D = x.shape
    dgamma = np.sum(dout * sample_norm, axis=0)  #stack horizontal Shape (D,)
    dbeta = np.sum(dout * 1.0, axis=0)# Stack horizontal: Shape. (D,) 

    #Compute dL/dx
    dx_hat = dout * gamma 

    #dL/dmu1 = dL/dx_hat * dxhat/dmu1
    dxhat_dmu1 = 1/(np.sqrt(var_s + eps))
    dmu1 = dx_hat * dxhat_dmu1

    #(1/x) gate
    #dL/ivar = dL/dx_hat * dx_hat/divar
    #x_hat = (x - mu)/sqrt(var - e)
    #dx_hat/divar = dx_hat/d(1/x) = (x-mu)
    #ivar: inverse of var: (1/x) gate
    divar = np.sum(dx_hat * (x-mu_s), axis=0) #Shape: (D,)

    # dsqrtvar = divar * (-(var_s-eps)**(-1))

    #sqrt(x + e): gate
    #dL/dvar = dL/divar * divar/dvar
    #divar/dvar = (1/2) * (x+e)**(-1/2)
    dvar = divar * (-0.5) * (var_s + eps)**(-3/2)

    #Summation gate. Step up dimension
    #dL/dsq = dL/dvar * dvar/dsq 
    #dvar/dsq: derivative of summation gate
    dsq = (1/N) * np.ones((N,D)) * dvar

    #(^2) gate: 
    #sigmoid^2 = var
    #dL/dmu2 = dL/dsq * dsq/dmu2
    # y = x^2 
    #dsq/dmu2 = 2x = 2(x-mu) 
    dmu2 = 2 * (x - mu_s) * dsq 


    #At minus node of the computation graph
    dmu  = (-1.0) * np.sum(dmu1 + dmu2,axis=0) # When 2 grads meet just add them
    dx1 = (1.0) * (dmu1 + dmu2)

    dx2 = (1/N) * np.ones((N,D)) * dmu
    dx = dx1 + dx2
    return dx, np.sum(dgamma), np.sum(dbeta)

def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    N = x.shape[0]
    mu_l = x.mean(axis=1,keepdims=True)
    var_l = x.var(axis=1,keepdims=True)
    x_hat = (x - mu_l) / np.sqrt(var_l + eps) #Layer norm
    out = gamma * x_hat + beta

    #Store cache needed for backward pass
    cache = x, gamma, beta, mu_l, var_l, eps, x_hat
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x, gamma, beta, mu_d, var_d, eps, x_hat = cache # unpack
    N, D = x.shape
    # This 2 is unchange from batch norm. 
    dgamma = np.sum(dout * x_hat, axis=0)  #stack horizontal Shape (D,)
    dbeta = np.sum(dout * 1.0, axis=0)# Stack horizontal: Shape. (D,) 

    #Compute dL/dx
    dx_hat = dout * gamma 

    #dL/dmu1 = dL/dx_hat * dxhat/dmu1
    dxhat_dmu1 = 1/(np.sqrt(var_d + eps))
    dmu1 = dx_hat * dxhat_dmu1

    #(1/x) gate
    #dL/ivar = dL/dx_hat * dx_hat/divar
    #x_hat = (x - mu)/sqrt(var - e)
    #dx_hat/divar = dx_hat/d(1/x) = (x-mu)
    #ivar: inverse of var: (1/x) gate
    divar = np.sum(dx_hat * (x-mu_d), axis=1,keepdims=True) #Shape: (D,)

    # dsqrtvar = divar * (-(var_s-eps)**(-1))

    #sqrt(x + e): gate
    #dL/dvar = dL/divar * divar/dvar
    #divar/dvar = (1/2) * (x+e)**(-1/2)
    dvar = divar * (-0.5) * (var_d + eps)**(-3/2)

    #Summation gate. Step up dimension
    #dL/dsq = dL/dvar * dvar/dsq 
    #dvar/dsq: derivative of summation gate
    dsq = (1/D) * np.ones((N,D)) * dvar

    #(^2) gate: 
    #sigmoid^2 = var
    #dL/dmu2 = dL/dsq * dsq/dmu2
    # y = x^2 
    #dsq/dmu2 = 2x = 2(x-mu) 
    dmu2 = 2 * (x - mu_d) * dsq 


    #At minus node of the computation graph
    dmu  = (-1.0) * np.sum(dmu1 + dmu2,axis=1,keepdims=True) # When 2 grads meet just add them
    dx1 = (1.0) * (dmu1 + dmu2)

    dx2 = (1/D) * np.ones((N,D)) * dmu
    dx = dx1 + dx2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p)/p
        out = x*mask #Drop those unlucky neuron
    elif mode == 'test':
        out = x # Nothing to do. 

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    return dx





#################################################################
#################################################################
"""
Optimizer
	- Adam
	- sgd
	- sgd + momentum 
	- RMS prop
"""
#################################################################
#################################################################
"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.

    https://www.coursera.org/lecture/deep-neural-network/gradient-descent-with-momentum-y0m1f
    Moving average to avoid oscilation. 
    momentum: mu
    Vdw = mu * Vdw + (1-mu)dw
    Vdb = mu * Vdb + (1-mu)db 

    mu * Vdw: slow down 2nd term()
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))#if exist return it. else return np.zeros_like(w)

    next_w = None
    #http://cs231n.github.io/neural-networks-3/#sgd
    v = config['momentum'] * v - config['learning_rate'] * dw # integrate velocity
    next_w = w + v #integerate position
    config['velocity'] = v

    return next_w, config



def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.

    cache = decay_rate * cache + (1 - decay_rate) * dx**2
    x += - learning_rate * dx / (np.sqrt(cache) + eps)
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))

    next_w = None
    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dw **2
    next_w = w + (-1.0)* config['learning_rate'] * dw / (np.sqrt(config['cache']) + config['epsilon'])

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))

    ###NOTE: Changed this line form 0 -> 1: solve NAN error.
    config.setdefault('t', 1)## number of current iteration

    next_w = None
    '''# look here for code generialize
    # t is your iteration counter going from 1 to infinity
    m = beta1*m + (1-beta1)*dx #1st moment
    mt = m / (1-beta1**t) # add unbias term to avoid 2nd moment being small in first few pass
    v = beta2*v + (1-beta2)*(dx**2) #second moment
    vt = v / (1-beta2**t) # add unbias term to avoid 2nd moment being small in first few pass
    x += - learning_rate * mt / (np.sqrt(vt) + eps)
    '''
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dw
    mt = config['m'] / (1 - config['beta1']**config['t'])
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2'])*(dw**2)
    vt = config['v'] / (1 - config['beta2']**config['t'])
    next_w = w + (-1.0) * config['learning_rate']  * mt / (np.sqrt(vt) + config['epsilon']) 

    return next_w, config