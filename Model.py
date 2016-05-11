from DeepCSMRI import * # All the import were added in there
##################################################################
def get_model():
	# Return 3D reconstruction model here
	model = Sequential()

	model.add(	Convolution3D(30, 3, 3, 3, border_mode='same', input_shape=(3, 30, 256,256))	)
	model.add(	Activation('relu')	)
	
	#
	model.add(	MaxPooling3D(pool_size=(2, 2, 2))	)

	model.add(	Convolution3D(60, 3, 3, 3, border_mode='same')	)
	model.add(	Convolution3D(60, 3, 3, 3, border_mode='same')	)
	model.add(	Activation('relu')	)
	
	# #
	# model.add(	MaxPooling3D(pool_size=(2, 2, 2))	)

	# model.add(	Convolution3D(60, 3, 3, 3, border_mode='same')
	# model.add(	Convolution3D(60, 3, 3, 3, border_mode='same')
	# model.add(	Activation('relu')	)

	# #
	# model.add(	UpSampling3D(size=(2, 2, 2))	)

	# model.add(	Convolution3D(60, 3, 3, 3, border_mode='same')
	# model.add(	Convolution3D(60, 3, 3, 3, border_mode='same')
	# model.add(	Activation('relu')	)

	#
	model.add(	UpSampling3D(size=(2, 2, 2))	)

	model.add(	Convolution3D(60, 3, 3, 3, border_mode='same')	)
	model.add(	Convolution3D(60, 3, 3, 3, border_mode='same')	)
	model.add(	Activation('relu')	)

	# Softmax to 256 class of prediction
	model.add(	Convolution3D(256, 3, 3, 3, border_mode='same')	)
	#model.add( 	Reshape((30,256,256))	)

	model.add(	Activation('relu')	)
	
	# training phase
	sgd = SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd)

	return model
##################################################################
if __name__ == '__main__':
	model = get_model()
	plot(model, to_file='model.png', show_shapes=True)