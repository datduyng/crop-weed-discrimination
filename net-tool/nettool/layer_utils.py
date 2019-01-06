'''
Layer_utils module use API from layers module 
to build/stack API gate. Convinience way to
stack module 
'''
import numpy as np 
from nettool.layers import *

def affine_relu_forward(x, w, b):
	out, affine_cache = affine_forward(x, w, b)
	out, relu_cache = relu_forward(out)
	cache = (affine_cache, relu_cache)
	return out, cache

def affine_relu_backward(dout, cache):
	affine_cache, relu_cache = cache
	dz = relu_backward(dout, relu_cache)
	dx, dw, db = affine_backward(dz, affine_cache)
	return dx, dw, db