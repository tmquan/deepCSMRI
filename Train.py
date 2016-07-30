#!/usr/bin/env python



# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 10:12:52 2016

@author: tmquan
"""

from Model 			import *
from Utility		import *
from GeneratePair 	import * # Import the data generation: zerofilling, full recon, mask
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
######################################################################################
	
def get_model():
	devs = [mx.gpu(2)]
	# network = get_unet()
	network = get_res_unet()
	
	model = mx.model.FeedForward(ctx=devs,
		symbol          = network,
		num_epoch       = 1,
		learning_rate	= 0.001,
        wd				= 0.0000000001,
        momentum		= 0.99,
		initializer     = mx.init.Xavier(rnd_type="gaussian", 
							factor_type="in", 
							magnitude=2.34),
		)	
	return model
######################################################################################

def train():
	# X = np.load('X_train.npy')
	# y = np.load('y_train.npy')

	# y  = y.astype('float32')
	# X  = X.astype('float32')
	
	# print "X shape", X.shape
	# print "X dtype", X.dtype
	# print "Y shape", y.shape
	# print "Y dtype", y.dtype
	
	## Load the data 
	print "Load the data"
	images = np.load('images.npy')
	
	# Xz, Xf = generatePair(images)
	X, y, R = generatePair(images)
	np.save("X_train.npy", X)
	np.save("y_train.npy", y)
	np.save("R_train.npy", R)
	##################################################################################
	nb_iter 		= 201
	epochs_per_iter = 1 
	batch_size 		= 500
	
	model = get_model()
	
	
	nb_folds = 3
	kfolds = KFold(len(y), nb_folds)
	for iter in range(nb_iter):
		print('-'*50)
		print('Iteration {0}/{1}'.format(iter, nb_iter))  
		print('-'*50) 
		
		# Shuffle the data
		print('Shuffle data...')
		seed = np.random.randint(1, 10e6)
		np.random.seed(seed)
		np.random.shuffle(X)
		np.random.seed(seed)
		np.random.shuffle(y)
		
		f = 0
		for train, valid in kfolds:
			print('='*50)
			print('Fold', f+1)
			f += 1
			
			# Extract train, validation set
			X_train = X[train]
			X_valid = X[valid]
			y_train = y[train]
			y_valid = y[valid]
			
			print "X_train", X_train.shape
			print "y_train", y_train.shape
			
			
			# prepare data
			data_train = mx.io.NDArrayIter(X_train, 
										   y_train,
										   batch_size			=	batch_size, 
										   shuffle				=	True, 
										   last_batch_handle	=	'roll_over'
										   )
			data_valid = mx.io.NDArrayIter(X_valid, 
			                               y_valid,
										   batch_size			=	batch_size, 
										   shuffle				=	True, 
										   last_batch_handle	=	'roll_over'
										   )
			
			model.fit(X = data_train, 
							eval_metric = mx.metric.RMSE(),
							eval_data   = data_valid)
			del X_train, X_valid, y_train, y_valid
		# if i%1==0:
		model.save('model', iter)
if __name__ == '__main__':
	train()
