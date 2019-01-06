import numpy as np
from nettool.layers import * 
from nettool.layer_utils import * 

class NeuralNet(object):
	'''
	Fully connected Neural net with arbitary number of hidden layers, 
	RELU non-linearity and a softmax loss. This allow batchnorm and dropout
	as intermidiate step. 
	affine: Linear Transformation
	{affine - [batch/layer norm] <-> relu <-> [dropout]} x (L-1) <-> affine <-> softmax
	
	batch/layer norm and dropout is optional
	Learnable parameter are stored in the self.params dictionary
	and will be learned using Trainer class
	'''
	def __init__(self,hidden_dims, input_dim=200*200*3, num_classes=10, 
		dropout=1, normalization=None, reg=0.0, 
		weight_scale=1e-2, dtype=np.float32):
		'''
		Init new Neural net model 

		Input:
			- hidden_dims: A list of int giving size of network 
			- input_dims: int giving size of input
			- num_classes: number of class predicting 
			- dropout: Scaler range [0,1] giving dropout strenth. Store probability 
				of keeping neuron. dropout=1 -> no drop at all
			- normalization: option ['batchnorm', 'layernorm', or none]
			- reg: Scalar range [0,1] giving regularization strength
			- weight_scale: how much initial random weight scale by
			- dtype: numpy datatype object
		'''
		self.normalization = normalization
		self.use_dropout = dropout != 1
		self.reg = reg
		self.num_layers = 1 + len(hidden_dims)
		self.dtype = dtype
		self.params = {}

		#initialize the parameter of the network, storing all val
		# in self.params dict
		input_layer_dim = input_dim
		for layer, hid_dim in enumerate(hidden_dims + [num_classes]):
			self.params['W'+str(layer+1)] = weight_scale * np.random.randn(input_layer_dim, hid_dim)
			self.params['b'+str(layer+1)] = np.zeros(hid_dim)
			if(normalization == "batchnorm" and layer < self.num_layers - 1):
				self.params['gamma'+str(layer+1)] = np.ones((1,1))
				self.params['beta'+str(layer+1)] = np.zeros((1,1))
			input_layer_dim = hid_dim
		# When using dropout we need to pass a dropout_param dictionary to each
		# dropout layer so that the layer knows the dropout probability and the mode
		# (train / test). You can pass the same dropout_param to each dropout layer.
		self.dropout_param = {}
		if self.use_dropout:
			self.dropout_param = {'mode': 'train', 'p': dropout}

		# With batch normalization we need to keep track of running means and
		# variances, so we need to pass a special bn_param object to each batch
		# normalization layer. You should pass self.bn_params[0] to the forward pass
		# of the first batch normalization layer, self.bn_params[1] to the forward
		# pass of the second batch normalization layer, etc.
		self.bn_params = []
		if self.normalization=='batchnorm':
			self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
		if self.normalization=='layernorm':
			self.bn_params = [{} for i in range(self.num_layers - 1)]

		# Cast all parameters to the correct datatype
		for k, v in self.params.items():
			self.params[k] = v.astype(dtype)

	def loss(self, X, y=None):
		"""
		Compute loss and gradient for the fully-connected net.
		Input / output: Same as TwoLayerNet above.
		"""
		X = X.astype(self.dtype)
		mode = 'test' if y is None else 'train'

		# Set train/test mode for batchnorm params and dropout param since they
		# behave differently during training and testing.
		if self.use_dropout:
			self.dropout_param['mode'] = mode
		if self.normalization=='batchnorm':
			for bn_param in self.bn_params:
				bn_param['mode'] = mode
		scores = None
		############################################################################
		# TODO: Implement the forward pass for the fully-connected net, computing  #
		# the class scores for X and storing them in the scores variable.          #
		#                                                                          #
		# When using dropout, you'll need to pass self.dropout_param to each       #
		# dropout forward pass.                                                    #
		#                                                                          #
		# When using batch normalization, you'll need to pass self.bn_params[0] to #
		# the forward pass for the first batch normalization layer, pass           #
		# self.bn_params[1] to the forward pass for the second batch normalization #
		# layer, etc.                                                              #
		############################################################################

		##Forward pass
		cache = {}
		out = {} 
		scores = X#intermidiate vairable for each layer
		# output = X 
		for layer in range(self.num_layers):
			
			L = layer + 1
			#Last layer does not have activate funciton
			if L == self.num_layers:
				scores, cache['aff%d'%L] = affine_forward(
					scores, self.params['W%d'%L], self.params['b%d'%L])
				continue
			scores, cache['aff_relu%d'%L] = affine_relu_forward(
					scores, self.params['W%d'%L], self.params['b%d'%L])

			if self.normalization=='batchnorm':
				scores, cache['bn%d'%L] = batchnorm_forward(scores, 
								self.params['gamma%d'%L], self.params['beta%s'%L], 
								self.bn_params[L-1])
			if self.use_dropout:
				scores , cache['drop%d'%L] = dropout_forward(scores, self.dropout_param)
		# If test mode return early
		if mode == 'test':
			return scores
		loss, grads = 0.0, {}
		############################################################################
		# TODO: Implement the backward pass for the fully-connected net. Store the #
		# loss in the loss variable and gradients in the grads dictionary. Compute #
		# data loss using softmax, and make sure that grads[k] holds the gradients #
		# for self.params[k]. Don't forget to add L2 regularization!               #
		#                                                                          #
		# When using batch/layer normalization, you don't need to regularize the scale   #
		# and shift parameters.                                                    #
		#                                                                          #
		# NOTE: To ensure that your implementation matches ours and you pass the   #
		# automated tests, make sure that your L2 regularization includes a factor #
		# of 0.5 to simplify the expression for the gradient.                      #
		############################################################################
		#Backward pass.
		loss, dout = softmax_loss(scores, y)#dout use as itermidiate variable for chaining down stream
		for layer in range(self.num_layers, 0, -1): #decrement 
			#since last layer does not have activation fucntion. 
			if layer == self.num_layers:
				dout, grads['W%d'%layer], grads['b%d'%layer] = affine_backward(dout, cache['aff%d'%layer])
				continue

			if self.use_dropout:
				dout = dropout_backward(dout, cache['drop%d'%layer])
				# dout = grads['W%d'%layer]
			if self.normalization=='batchnorm':
				dout, grads['gamma%d'%layer], grads['beta%d'%layer] = batchnorm_backward_alt(dout, cache['bn%d'%layer])
				# dout = grads['W%d'%layer]
			dout, grads['W%d'%layer], grads['b%d'%layer] = affine_relu_backward(dout, cache['aff_relu%d'%layer])
			
		#Normalize and add regularization term
		for layer in range(self.num_layers):
			L = layer+1 
			loss += 0.5 * self.reg * np.sum(self.params['W%d'%L]**2)
			grads['W%d'%L] += self.reg * self.params['W%d'%L]
		return loss, grads

	def predict(self, X):
		'''
		Use trained weight of the network to predict the 
		label for given data points. for each datapoint 
		we predict the score for each of C classes and 
		assign each datapoint to be class with highest score.
		'''
		scores = self.loss(X)#Shape: (N x C)
		return np.argmax(scores, axis=1) #Stack vertically
