import os
import sys
import time
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class LogisticRegression(object):
	def __init__(self, input, n_in, n_out):
		self.W = theano.shared(
			value=np.zeros(
				(n_in, n_out), dtype=theano.config.floatX
			), borrow=True
		)
		
		self.b = theano.shared(
			value=np.zeros(
				(n_out,), dtype=theano.config.floatX
			), borrow=True
		)

		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		self.params = [self.W, self.b]
	
	def negative_log_likelihood(self, y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
	
	def errors(self, y):
		return T.mean(T.neq(self.y_pred, y))

class Layer(object):
	def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation_type='tanh', loss_type=None):
		self.input = input
		if W is None:
			W_values = np.asarray(
				rng.uniform(
					low=-np.sqrt(6. / (n_in + n_out)),
					high=np.sqrt(6. / (n_in + n_out)),
					size=(n_in, n_out)
				),dtype=theano.config.floatX
			)

			if activation_type == 'sigmoid':
				W_values *= 4
			W = theano.shared(value=W_values, borrow=True)
		
		if b is None:
			b_values = np.zeros((n_out, ),dtype=theano.config.floatX)
			b = theano.shared(value=b_values, borrow=True)
		self.W = W
		self.b = b
		
		if activation_type == 'sigmoid':
			activation = T.nnet.sigmoid
		elif activation_type == 'tanh':
			activation = T.tanh
		elif activation_type == 'relu':
			activation = lambda x: x * (x > 0)
		else:
			activation = None

		lin_output = T.dot(input, self.W) + self.b

		self.output = lin_output if activation is None else activation(lin_output)

		self.params = [self.W, self.b]

		if loss_type == 'mse':
			self.loss = lambda y: T.mean(((self.output - y) ** 2).sum(axis=1))
		else:
			self.loss = None

class ConvPoolLayer(object):
	def __init__(self, rng, input, filter_shape=None, image_shape=None, poolsize=(2, 2), W=None, b=None):
		self.input = input
		
		if W is None:
			fan_in = np.prod(filter_shape[1:])
			fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
			W_bound = np.sqrt(6. / (fan_in + fan_out))
			self.W = theano.shared(
				np.asarray(
					rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
					dtype=theano.config.floatX
				),
				borrow=True
			)
		else:
			self.W = W
		
		if b is None:
			b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
			self.b = theano.shared(value=b_values, borrow=True)
		else:
			self.b = b

		conv_out = conv.conv2d(
			input =input,
			filters=self.W,
			filter_shape=filter_shape,
			image_shape=image_shape
		)

		pooled_out = downsample.max_pool_2d(
			input=conv_out,
			ds=poolsize,
			ignore_border=True
		)

		self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		self.params = [self.W, self.b]

class DeepNet(object):
	def __init__(self):
		self.layers = []
		self.params = []
		self.errors = None
		self.output = None

	#layer_config : Dictionary
	#rng : Random generator
	#type : conv, hidden, output
	#n_in : number of input
	#n_out : number of output
	#filter_shape
	#image_shape
	#poolsize
	#activation_type
	#loss_type
	#input : if first layer

	def default_settings(self, outsize=18):
		rng = np.random.RandomState(1234)
		inp = T.matrix('inp')
		#inp = T.tensor4('inp')
		target = T.matrix('target')

		#inp = inp.reshape((32, 1, 84, 84))
		layer0_inp = inp.reshape((32, 4, 84, 84))
		layer0 = ConvPoolLayer(rng, layer0_inp,
			filter_shape=(20, 4, 9, 9), image_shape=(32, 4, 84, 84), poolsize=(2,2))

		layer1 = ConvPoolLayer(rng, layer0.output,
			filter_shape=(50, 20, 5, 5), image_shape=(32, 20, 38, 38), poolsize=(2, 2))

		layer2_inp = layer1.output.flatten(2)

		layer2 = Layer(rng, layer2_inp, n_in=50 * 17 * 17, n_out=256)
		layer3 = Layer(rng, layer2.output, n_in=256, n_out=outsize, activation_type='None', loss_type='mse')
		
		self.params = layer3.params + layer2.params + layer1.params + layer0.params
		self.layers = [layer3, layer2, layer1, layer0]
		
		self.output = layer3.output
		self.cost = layer3.loss(target)

		#s_inp = T.tensor4(name='s_inp')
		#s_inp = s_inp.reshape((1, 1, 84, 84))

		#l0 = ConvPoolLayer(rng, s_inp, filter_shape=None, image_shape=(1,1,84, 84), W=layer0.W, b=layer0.b)
		#l1 = ConvPoolLayer(rng, l0.output, filter_shape=None, image_shape=(1, 16, 38, 38), W=layer1.W, b=layer1.b)
		#l2 = Layer(rng, l1.output.flatten(2), n_in=32*17*17, n_out=256, W=layer2.W, b=layer2.b)
		#l3 = Layer(rng, l2.output, 256, 3, loss_type='mse', W=layer3.W, b=layer3.b)

		#self.s_layers = [l0, l1, l2, l3]
		#self.single_predict = theano.function(
		#	inputs=[s_inp],
		#	outputs=l3.output,
		#	allow_input_downcast=True
		#	)

		self.train_x, self.train_y = shared_dataset([
				np.asarray(np.random.random((32, 4 * 84 * 84)), dtype='float32'),
				np.asarray(np.random.random((32, outsize)), dtype='float32')
				])

		gparams = T.grad(self.cost, self.params)
		
		updates = []
		rho = 0.9
		epsilon = 1e-6
		learning_rate = 0.001
		
		#rmsprop
		for p, g in zip(self.params, gparams):
			acc = theano.shared(p.get_value() * 0.)
			acc_new = rho * acc + (1 - rho) * g ** 2
			gradient_scaling = T.sqrt(acc_new + epsilon)
			g = g / gradient_scaling
			updates.append((acc, acc_new))
			updates.append((p, p - learning_rate * g))
		
		#sgd
		#updates = [(p,p - learning_rate * g) for p, g in zip(self.params, gparams)]


		print "... building model"
		self.train_net = theano.function(
				inputs=[],
				outputs=self.cost,
				updates = updates,
				givens = {
					inp: self.train_x,
					target: self.train_y,
					}
				#allow_input_downcast=True,
				#on_unused_input='ignore'
				
			)

		self.compute_q = theano.function(
				inputs=[inp],
				outputs=self.output,
				allow_input_downcast=True
			)

		self.predict = theano.function(
			inputs=[inp],
			outputs=T.argmax(self.output, axis=1),
			allow_input_downcast=True
			)
	
	def add_layer(self, layer_config):
		if len(self.layers) == 0:
			input = layer_config['input']
		else:
			input = self.layers[-1].output

		if layer_config['type'] == 'conv':
			layer = ConvPoolLayer(layer_config['rng'], input,
				layer_config['filter_shape'], layer_config['image_shape'],
				layer_config['poolsize'])
		elif layer_config['type'] == 'hidden':
			layer = Layer(layer_config['rng'], input.flatten(2),
				layer_config['n_in'], layer_config['n_out'],
				activation_type=layer_config['activation_type'])
		else :#layer_config['type'] == 'output':
			layer = Layer(layer_config['rng'], input,
				layer_config['n_in'], layer_config['n_out'],
				activation_type=layer_config['activation_type'], 
				loss_type=layer_config['loss_type'])
			self.errors = layer.loss
			self.output = layer.output
		
		self.layers.append(layer)
		self.params.extend(layer.params)
	
	def build(self, s_img):
		self.predict = theano.function(
			inputs = [s_img],
			outputs = self.output,
			allow_input_downcast=True
		)	

def shared_dataset(data_xy):
	data_x, data_y = data_xy
	
	shared_x = theano.shared(
		np.asarray(data_x, dtype=theano.config.floatX),
		borrow=True
	)

	shared_y = theano.shared(
		np.asarray(data_y, dtype=theano.config.floatX),
		borrow=True
	)

	return shared_x, shared_y

