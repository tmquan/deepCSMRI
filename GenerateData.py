from DeepCSMRI import *
from Directory import * # dataDir

def generateData():
	"""
	Query the all of the tif file and store as images.npy

	"""
	images = []
	for dirName, sibdirList, fileList in os.walk(dataDir):
		# Sort the tif file numerically
		fileList = natsort.natsorted(fileList) 

		#Query the file in list
		for fname in fileList:
			fullPath = dataDir+fname
			print fullPath

			# Read to numpy array
			# image = skimage.io.imread(fullPath)
			image = cv2.imread(fullPath, cv2.IMREAD_GRAYSCALE)
			print image.shape
			
			# Append to the images
			images.append(image)

	# Convert images list to numpy array
	images = np.array(images)

	# Get the current shape of images
	print images.shape

	# Convert to 4D tensor array
	# from (num, dimz, dimy, dimx)
	# to   (num, dimk, dimz, dimy, dimx)
	# by inserting singleton dimension to axis 1
	# images = np.reshape(images, (-1,20,256,256)) #
	images = np.expand_dims(images, axis=1)
	print images.shape


	# Show the first image for testing
	plt.imshow(np.abs(images[0,0,:,:]), cmap=cm.gray)
	plt.axis('off')
	plt.show()
	
	np.save('images.npy', images)
	return images


if __name__ == '__main__':
	images = generateData()