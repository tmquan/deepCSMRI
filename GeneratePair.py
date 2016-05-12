from DeepCSMRI 		import * # All the import were added in there
from GenerateMask 	import * # Import the mask generation for 2D image
##################################################################

def generatePair(image):
	"""
	Input is an 5D tensor image (1,1,dimz,dimy,dimx)
	Call the generateMask to make the single mask (dimy, dimx)
	Expand the mask along z and expand it dimension to (1,1,dimz,dimy,dimx)
	Make undersampled image consist of 3 channels (1,3,dimz,dimy,dimx)
		image[:,0,:,:,:] is the real part of zero filling
		image[:,1,:,:,:] is the imag part of zero filling
		image[:,2,:,:,:] is the mask for undersampling
	Make the prediction contains 256 level of
	"""
	
	# Get the shape of image
	print image.shape
	dimz, dimy, dimx = image.shape


	srcImage = np.zeros((1,  3,dimz,dimy,dimx))
	dstImage = np.zeros((1,256,dimz,dimy,dimx))

	# Generate the undersampling pattern
	mask   =  generateMask(dimz, dimy, dimx, sampling_rate=0.25, center_ratio=0.5)

	kspace = np.fft.fft2(image)
	under  = mask*kspace
	# March through the temporal dimension
	#for z in range(dimz):

	#print srcImage.shape
	return srcImage, dstImage # 
def test_generatePair(image):
	X, y = generatePair(image)

##################################################################
if __name__ == '__main__':
	images = np.load('images.npy')

	# Extract a single image
	image  = images[0,0,:,:,:]

	test_generatePair(image)