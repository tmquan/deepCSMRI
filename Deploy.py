from Utility import *
import mxnet as mx
import timeit
def deploy():
	X = np.load('X_train.npy')
	
	# X = np.reshape(X, (30, 1, 512, 512))
	print "X.shape", X.shape
	k=1
	# k=50
	# k=52
	X_deploy = X[(k-1):k,:,:,:]
	# X_deploy = X
	print "X_deploy.shape", X_deploy.shape
	# Load model
	iter = 0 
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
	pred  = np.reshape(pred, (20, 256, 256))
	print pred.shape
	skimage.io.imsave('y_pred.tif', pred)
	pred  = pred[1,:,:]
	
	
	# plt.imshow(((pred)) , cmap = plt.get_cmap('gray'))
	# plt.show()	
	
if __name__ == '__main__':
	deploy()