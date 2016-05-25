import numpy as np 
import theano.tensor as T 
from theano import function, shared 
import time
import h5py
from sklearn import preprocessing

class SOM(object):

	'''
	Properties of the map: 
		- Variables
			- features 
			- dim: dimensionality of the map 
			- units: number of units in the map 
		- theano symbolic shared variables
			- shape: shape of the map
			- grid: coordinates of all the cells in the map
			- codebook: weights of all the cells

	Training parameters: 
		- Variables
			- epochs
			- lrate_i, lrate_f
			- sigma_i, sigma_f 
			- threshold
		- theano symbolic shared variables
			- sigma: contains the current sigma 
			- lrate: contains the current lrate 
			- sigma_factor: the factor by which sigma gets updated at each epoch
			- lrate_factor: the factor by which lrate gets updates at each epoch

	Properties of the datasets:
		- Variables
			- means: means of the datasets
			- stds: standard deviation of the datasets
			- fires: index of the cell that fires (produced in training)
			- test: hits in each cell by different types of data (produced in testing)

	Functions: 
		- __init__(shape, features, filename = None)
			- Initialise the map 
		- set_params(epochs = 5, sigma = (6, 0.001), lrate = (0.2, 0.001), threshold = 4)
			- Set training parameters 
		- train_theano(data): 
			- Rescale the data and train the map
		- test_theano(data, types):
			- Test with data
		- save_map(filename)
			- Save the entire content of the map
		- save_results(filename) 
			- Save the results from testing
	'''
	def __init__(self, shape = (10, 10, 10), features = 8, filename = None):

		if isinstance(filename, str):
			f = h5py.File(filename, 'r')
			grp = f['map'] 
			self.features = grp.attrs['features']
			self.dim = grp.attrs['dim']
			shape = grp.attrs['shape']
			self.units = np.prod(shape)
			self.means = grp.attrs['means']
			self.stds = grp.attrs['stds']

			# Initialise the shared variables
			self.shape = shared(shape, name = 'shape')
			self.grid = shared(grp['grid'], name = 'grid')
			self.codebook = shared(grp['codebook'], name = 'grid')

		else:
			self.units = np.prod(shape)
			self.features = features
			self.dim = len(shape)

			# Initialise the shared variables 
			self.shape = shared(np.array(shape), name = 'shape')
			self.grid = shared(np.vstack(map(np.ravel, np.indices(shape))).T, name = 'grid')
			self.codebook = shared(np.random.random((self.units, features)), name = 'codebook')




	def set_params(self, epochs = 5, sigma = (6, 0.001), lrate = (0.2, 0.001), threshold = 4):
		'''
		epochs: the number of times all the samples get passed
		sigma: neighbourhood 
		lrate: learning rate 
		threshold: stopping condition
		'''
		self.epochs = epochs
		self.threshold = threshold

		self.lrate_i, self.lrate_f = lrate 
		self.sigma_i, self.sigma_f = sigma
		lrate_factor = (self.lrate_f/self.lrate_i) ** (1/float(self.epochs))
		sigma_factor = (self.sigma_f/self.sigma_i) ** (1/float(self.epochs))

		self.sigma = shared(self.sigma_i, name = 'sigma')
		self.lrate = shared(self.lrate_i, name = 'lrate')
		self.sigma_factor = shared(sigma_factor, name = 'sigma_factor')
		self.lrate_factor = shared(lrate_factor, name = 'lrate_factor')

	def _match(self, sample):
		diff = (T.sqr(self.codebook)).sum(axis = 1, keepdims = True) + (T.sqr(sample)).sum(axis = 1, keepdims = True) - 2 * T.dot(self.codebook, sample.T)
		bmu = T.argmin(diff)
		err = T.min(diff)
		return err, bmu 

	def _update_map(self, sample, weight, winner):
		dist = T.sqrt((T.sqr((self.grid - winner)/self.shape)).sum(axis = 1, keepdims = True)/self.shape.ndim)
		gaussian = T.exp(- T.sqr(dist/self.sigma))
		return [[self.codebook, 
						sample + (self.codebook - sample) * (1 - gaussian * self.lrate) ** weight]]

	def _update_params(self):
		return [[self.lrate, self.lrate * self.lrate_factor], 
				[self.sigma, self.sigma * self.sigma_factor]]

	def train_theano(self, data): 

		''' 
		A method that takes an np.array, scales it to have zero mean and unit standard deviation, and train the map on the data
		data: np.array with the last column being weights
		''' 
		# -----
		# Define symbolic variables and compile the functions
		# -----

		broadscalar = T.TensorType('float32', (True, True))
		s = T.frow('s')
		w = broadscalar('w')
		win = T.frow('win')

		match = function(
			inputs = [s], 
			outputs = self._match(s), # return err, bmu
			allow_input_downcast = True
			)

		update_map = function(
			inputs = [s, w, win], 
			outputs = [], 
			updates = self._update_map(s, w, win),
			allow_input_downcast = True
			)

		update_params = function(
			inputs = [],
			outputs = [],
			updates = self._update_params(),
			allow_input_downcast = True
			)

		# ----- 
		# Training starts here 
		# ----- 

		# Preprocess the data 
		scaler = preprocessing.StandardScaler().fit(data[:, 0:self.features])
		data[:, 0:self.features] = scaler.transform(data[:, 0:self.features]) + 0.5
		self.means = scaler.mean_
		self.stds = scaler.std_

		samples = data.shape[0]
		self.fires = np.zeros((samples, 2)) # One for index of the neuron that fired and 

		print 'Training starts....'
		print 'Number of samples:', samples

		for e in range(self.epochs):

			print 'epoch:', e
			print 'sigma:', self.sigma.get_value()
			print 'lrate:', self.lrate.get_value() 

			start = time.mktime(time.localtime())
			ordering = np.random.permutation(samples)
		
			for i in ordering:

				sample = data[i, 0:self.features][None, :]
				weight = np.array([[data[i, -1]]])
			
				error, unit = match(sample)
				
				if self.fires[i, 0] == unit:
					self.fires[i, 1] += 1
			
				else:
					self.fires[i, 0] = unit
					self.fires[i, 1] = 0
			
				if self.fires[i, 1] < self.threshold: 
					winner = self.grid.get_value()[unit][None, :]
					update_map(sample, weight, winner)
			
			update_params() 
			print 'number of stable samples:', np.sum(self.fires[:, 1] >= self.threshold)
			end = time.mktime(time.localtime())
			print 'time stamp:', end - start 

	def test_theano(self, data, types):
		'''
		data: np.array with the last column as what type of data it is 
		types: the number of types
		'''
		self.test = np.zeros((self.units, types))
		self.err = 0

		s = T.frow('s')
		match = function(
			inputs = [s], 
			outputs = self._match(s), # return err, bmu
			allow_input_downcast = True
			)

		# Rescale the test data the same way training data gets rescaled
		samples = data.shape[0] 
		data[:, 0:self.features] = (data[:, 0:self.features] - self.means)/self.stds + 0.5

		for i in range(samples):
			sample = data[i, 0:self.features][None, :]
			index = data[i, -1]

			err, unit = match(sample)
			self.test[unit, index] += 1 
			self.err += err

		self.err = self.err/float(self.units)
		print 'error:', self.err 

	def show_map(self):
		types = self.test.shape[-1]
		[x, y, z] = self.grid.get_value().T
		for t in range(types):
			fig = plt.figure()
			ax = fig.add_subplot(111, projection = '3d')
			ax.scatter(x, y, z, c = self.test[..., t], norm = LogNorm(), lw = 0)
			plt.show() 

	def save_map(self, filename): 
		'''
		A method that saves the map into hdf5 format 
		top group: 'map' 
		datasets: 'codebook' and 'grid' 
		attributes: information on the map and training params 
		'''
		f = h5py.File(filename)
		if 'map' in f.keys():
			del f['map']

		grp = f.create_group('map')	
		grp.create_dataset('codebook', data = self.codebook.get_value())
		grp.create_dataset('grid', data = self.grid.get_value())

		# Information on the map
		grp.attrs['shape'] = self.shape.get_value()
		grp.attrs['dim'] = self.dim 
		grp.attrs['features'] = self.features
		grp.attrs['means'] = self.means
		grp.attrs['stds'] = self.stds

		# Information on training
		grp.attrs['epochs'] = self.epochs
		grp.attrs['sigma'] = (self.sigma_i, self.sigma_f)
		grp.attrs['lrate'] = (self.lrate_i, self.lrate_f)
		grp.attrs['threshold'] = self.threshold
		f.close()

	def save_results(self, filename): 
		'''
		A method that saves the testing results
		'''
		f = h5py.File(filename)
		if 'results' in f.keys():
			del f['results'] 
		grp = f.create_group('results')
		grp.create_dataset('test', data = self.test)
		f.close() 





                 









	




