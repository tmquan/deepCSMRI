from DeepCSMRI 		import * # All the import were added in there
from GenerateMask 	import * # Import the mask generation for 2D image
##################################################################

def generatePair(images):
	"""
	Input is an 5D tensor images (1,1,dimz,dimy,dimx)
	Call the generateMask to make the single mask (dimy, dimx)
	Expand the mask along z and expand it dimension to (1,1,dimz,dimy,dimx)
	Make undersampled images consist of 3 channels (1,3,dimz,dimy,dimx)
		images[:,0,:,:,:] is the real part of zero filling
		images[:,1,:,:,:] is the imag part of zero filling
		images[:,2,:,:,:] is the mask for undersampling
	Make the prediction contains 256 level of
	"""
	
	# Get the shape of images
	# print images.shape
	dimn, dimz, dimy, dimx = images.shape
	shape = images.shape

	print shape
	srcImage = np.zeros(shape)
	#srcImage = np.zeros((  1,dimz,dimy,dimx))
	dstImage = np.zeros(shape)
	###########################################################
	# Generate the 3 channel input
	# Generate the undersampling pattern
	mask   =  generateMask(dimz, dimy, dimx, sampling_rate=0.125, center_ratio=0.5)
	
	# Perform forward Fourier transform
	kspace = np.fft.fft2(images)
	 
	# Perform undersampling
	under  = mask*kspace
	
	# Perform inverse Fourier transform for zerofilling
	zfill  = np.fft.ifft2(under)
	
	# March through the temporal dimension
	#for z in range(dimz):
	# Assign the channels

	srcImage = np.abs(zfill)  /255
	
	# srcImage = np.abs(zfill) /255
	# srcImage = np.expand_dims(srcImage, axis=0)
	
	#print srcImage.shape
	dstImage = images 
	
	###########################################################
	# Generate the 256 channel output
	# Note that pixel value 128 corresponds to a vector
	# [1 1 1 1 1 ... 1   0 0 0 0 0 ]  < Accumulating vector, not binary
	# [0 1 2 3 4 ....128 ..		   ]  < Corresponding indices
	# Example
	# a 	= np.ones(10)
	# idx 	= np.arange(10)
	# pixel = 3
	# a[idx>pixel] = 0
	# idx = np.arange(256)
	# dstImage = np.ones((256,dimz,dimy,dimx))
	# for z in range(dimz):
		# for y in range(dimy):
			# for x in range(dimx):
				# pixelVal = images[z,y,x]
				# pixelVec = np.ones(256)
				# pixelVec[idx>pixelVal] = 0
				# dstImage[:,z,y,x] = pixelVec
	# valImage = np.repeat(images, 256, axis=0) # Make repeat along z	
	# dstImage = np.arange(256)
	# dstImage[
	# b = np.zeros((a.size, 256))
	# b[np.arange(a.size),a] = 1
	
	# print srcImage.shape
	# print dstImage.shape
 
	return srcImage, dstImage, mask
def test_generatePair(images):
	print "Here"
	print images.shape
	Xz, Xf, R = generatePair(images)
	print "After Generating"
	# print images.shape
	sliceId = 0
	full = Xf[0,sliceId,:,:]
	zero = Xz[0,sliceId,:,:]
	#tmp = 
	#tmp = y[sliceId,:,:]
	print full.shape
	print zero.shape
	# tmp = images
	plt.imshow( full, cmap=cm.gray) 
	plt.axis('off')
	plt.show()
	plt.imshow( zero, cmap=cm.gray) 
	plt.axis('off')
	plt.show()
##################################################################
if __name__ == '__main__':
	images = np.load('images.npy')

	# # Extract a single image
	# image  = images[0,0,:,:]
	# print image.shape
	# plt.imshow( np.squeeze(image) , cmap=cm.gray) 
	# plt.axis('off')
	# plt.show()
	test_generatePair(images[0:1,:,:,:])