import numpy as np


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
    dx[x <= 0.0] = 0
    return dx


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
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
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
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x,sample_mean, sample_var, eps, sample_norm, gamma = cache # unpack
    x_hat = sample_norm
    mu_s = sample_mean
    var_s = sample_var
    N = dout.shape[0]
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    dgamma = np.sum(dout * sample_norm, axis=0)  #stack horizontal Shape (D,)
    dbeta = np.sum(dout * 1.0, axis=0)# Stack horizontal: Shape. (D,) 

    #Compute dL/dx 
    #x_hat = sample_norm
    dx_hat = dout * gamma 
    dvar = np.sum(dx_hat * (x - mu_s) * (-0.5) * (var_s + eps)**(-3/2), axis=0)
    dmu = np.sum(dx_hat*-1/np.sqrt(var_s+eps),axis=0) + dvar * (-2 * (x-mu_s)).sum(axis=0)/N 
    dx = dx_hat * (1/np.sqrt(var_s + eps)) + dvar * 2*(x-mu_s)/N + dmu/N
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
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


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
            - Padding help retain the spatial shape of the input after convolution. (My note)

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']  
    H_p = 1 + (H + 2*pad - HH) // stride 
    W_p = 1 + (W + 2*pad - WW) // stride
    ###########################################################################
    #                          #
    # Hint: you can use the function np.pad for padding.                      #
    #   import numpy as np
    #
    #   image_stack = np.ones((2, 2, 18))
    #
    #   extra_left, extra_right = 1, 2
    #   extra_top, extra_bottom = 3, 1
    #
    #   np.pad(image_stack, ((0, 0),(extra_left, extra_right), (extra_top, extra_bottom)),
    #        mode='constant', constant_values=3) 
    # #With nd array of shape N, RGB_DEPTH, D_1, D_2,...
    # c = np.pad(image_stack, ((0,0),(0,0), (extra_left, extra_right), (extratop,extrabot)) )
    ###########################################################################
    #Np.pad example:https://stackoverflow.com/questions/51650325/how-to-pad-an-image-using-np-pad-in-python
    extra_left, extra_right = pad, pad
    extra_top, extra_bottom = pad, pad
    x_pad = np.pad(x, ((0,0),(0,0),(extra_top,extra_bottom),(extra_left,extra_right)),
                 mode='constant', constant_values=0)
    out = np.zeros((N, F, H_p, W_p), dtype=x.dtype)
    # w_ = w.reshape(F, -1) # convert filter to vectors
    for n_ in range(N):
        for f_ in range(F):
            for hp in range(0, H_p):
                for wp in range(0, W_p):
                    cube = x_pad[n_, :, hp*stride:hp*stride+HH, wp*stride:wp*stride+WW].reshape(-1)
                    out[n_, f_, hp, wp] = np.sum(cube * w[f_,:].reshape(-1)) + b[f_]
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives. Shape(F, N, H', W')
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b(Shape: F,)
    good understanding: https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
    Walk through example of backprop conv: 
        https://computing.ece.vt.edu/~f15ece6504/ (Lecture 5)
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache 
    N, C, H, W = x.shape

    F, C, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']  
    H_p = 1 + (H + 2*pad - HH) // stride 
    W_p = 1 + (W + 2*pad - WW) // stride

    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)),
                 mode='constant', constant_values=0)

    dx_pad = np.zeros_like(x_pad)
    dx = np.zeros_like(x) 
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):
        for f in range(F):
            db[f] += np.sum(dout[n,f])
            for h_ in range(H_p):
                for w_ in range(W_p):
                    dw[f] += x_pad[n,:,h_*stride:h_*stride + HH, w_*stride:w_*stride + WW] * dout[n, f, h_, w_]
                    dx_pad[n,:,h_*stride:h_*stride + HH, w_*stride:w_*stride + WW] += w[f] * dout[n, f, h_, w_]

    #extract the pad
    dx = dx_pad[:,:,pad:-pad,pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    pool_h, pool_w, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H_p = 1 + (H - pool_h) // stride
    W_p = 1 + (W - pool_w) // stride
    #Create space 
    out = np.zeros((N, C, H_p, W_p))
    #Naive 
    for n in range(N):
        for c in range(C):
            for h_ in range(H_p):
                for w_ in range(W_p):
                    out[n,c,h_,w_] = np.amax(x[n,c,h_*stride:h_*stride+pool_h, w_*stride:w_*stride+pool_w])


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    #unpack
    x, pool_param = cache 
    pool_h, pool_w, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    N, C, H, W = x.shape
    H_p = 1 + (H - pool_h) // stride
    W_p = 1 + (W - pool_w) // stride
    dx = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for h_ in range(H_p):
                for w_ in range(W_p):
                    max_index = np.unravel_index(x[n,c,h_*stride:h_*stride+pool_h, w_*stride:w_*stride+pool_w].argmax(), x[n,c,h_*stride:h_*stride+pool_h, w_*stride:w_*stride+pool_w].shape);
                    dx[n,c,h_*stride+max_index[0],w_*stride+max_index[1]] = dout[n,c, h_, w_]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, {}
    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H , W = x.shape
    out = np.zeros_like(x)

    #Since we only care about all axis but axis C. Then just combine them together 
    #Using transpose to get shape (N, H, W, C) and then reshape
    out = out.transpose(0, 2, 3, 1).reshape(N*H*W, C)
    x = x.transpose(0, 2, 3, 1).reshape(N*H*W, C)

    #INtermidiate cache will be append in cache variable and become a tuple
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)

    #Reshape and transpose back to original. 
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    #Unpack some variable to retrieve shape
    N, C, H, W = dout.shape

    #
    dgamma = np.zeros(C)
    dbeta = np.zeros(C)
    # dx = np.zeros_like(dout)

    #Same reason as above just grp all axis that we care together
    dout = dout.transpose(0, 2, 3, 1).reshape(N*H*W, C)

    #Note: Cache is a tuple. Each cache intances hold 6 element aof tuple 
    #When computing. The all tuple 
    #Will get compute
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)

    #Reshape and transpose back to original. 
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, {}
    eps = gn_param.get('eps',1e-5)
    N, C, H, W = x.shape
    per_group = C // G



    gamma = gamma.reshape(-1)
    beta = beta.reshape(-1)
    ln_param = {}
    ln_param['eps'] = eps
    out = np.zeros_like(x)

    #Transpose the  shape so that we have x.shape = (N, H, W, C)
    #Since we compute norm between axis=C therefore, other axis can be combine
    x = x.transpose(0, 2, 3, 1)
    out = x.transpose(0, 2, 3, 1).reshape(N*H*W, C)

    #Loop through each grp and sclice each grp out to compute grp norm
    for g in range(G):
        x_reshaped = x[:,:,:,per_group*g:g*per_group+per_group].reshape(N*H*W, per_group)
        out[:,per_group*g:g*per_group+per_group], cache[g] = layernorm_forward(x_reshaped,
                gamma[per_group*g:g*per_group+per_group], 
                beta[per_group*g:g*per_group+per_group], ln_param)
    #Reshape back to original
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    cache['G'] = G    
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    N, C, H, W = dout.shape
    G = cache['G']
    per_group = C // G

    dgamma = np.zeros(C)#WIll later be reshape back to (1,C,1,1)
    dbeta = np.zeros(C)#Will later be reshape back to (1,C,1,1)
    dx = np.zeros_like(dout)

    #Same as in forward case. Just nail all axis down to (?,C) and norm over C axis
    dx = dx.transpose(0, 2, 3, 1).reshape(N*H*W, C)
    dout = dout.transpose(0, 2, 3, 1).reshape(N*H*W, C)

    for g in range(G):
        grp_idx = per_group*g
        (dx[:,grp_idx:grp_idx+per_group],
         dgamma[grp_idx:grp_idx+per_group],
         dbeta[grp_idx:grp_idx+per_group]) = layernorm_backward(
                dout[:,grp_idx:grp_idx+per_group],
                cache[g])

    #Reshape back to original
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    #Reshape gamma and beta grads as well
    dgamma = dgamma.reshape(1,C,1,1)
    dbeta = dbeta.reshape(1,C,1,1)

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y.astype(int)]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
