from Utility import *
from DeepCSMRI import *
import mxnet as mx
import timeit
def deploy():
	X = np.load('X_train.npy')
	y = np.load('y_train.npy')
	
	# X = np.reshape(X, (30, 1, 512, 512))
	print "X.shape", X.shape
	k=1
	# k=50
	# k=52
	t=np.arange(tempo);
	X_deploy = X[1:2,t,:,:]
	y_full = y[1:2,t,:,:]
	
	# X_deploy = np.expand_dims(X_deploy, axis=0)
	# y_full   = np.expand_dims(y_full, axis=0)
	# X_deploy = X
	print "X_deploy.shape", X_deploy.shape
	# Load model
	iter = 4 
	model 	= mx.model.FeedForward.load('model', iter, ctx=mx.gpu(1))
	# model_recon 	= mx.model.FeedForward.load('model', iter)
	

	
	# Perform prediction
	batch_size = 1
	print('Predicting on data...')
	start = timeit.timeit()
	pred  = model.predict(X_deploy, num_batch=None)
	end = timeit.timeit()
	print end - start
	
	print pred.shape
	pred = np.reshape(pred, (-1, 256, tempo, 256, 256))
	pred = 1.0*(pred>0.5)
	pred = np.cumsum(pred, axis=1)
	# pred = np.argmax(pred, axis=1)
	pred = pred[:,-1,:,:,:]
	print pred.shape
	skimage.io.imsave('y_pred.tif', np.float32(pred))
	skimage.io.imsave('y_zero.tif', np.float32(X_deploy))
	skimage.io.imsave('y_full.tif', np.float32(y_full))
	# pred  = pred[1,:,:]
	
	
	# plt.imshow(((pred)) , cmap = plt.get_cmap('gray'))
	# plt.show()	
	
if __name__ == '__main__':
	deploy()
