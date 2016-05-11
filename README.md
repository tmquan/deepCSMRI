# deepCSMRI
# This repository contains the source code to reconstruct undersampled dynamic MRI data using deep neural network

Library			: 	keras.io
Tensor shape	:	(num,dimk,dimz,dimy,dimx)

List of directory:

Use mkdir command in Linux
./data 		# Contains the tif file image
./models	# Contains the models of the keras system
./result	# Contains the result of deploy the network


Run python GenerateData.py # Save all of images in 5D tensor format in images.npy file
Run python GenerateMask.py # See the mask is generating, function can be call on the fly
