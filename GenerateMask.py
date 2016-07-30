from DeepCSMRI import *
##################################################################
def generatePattern(
	dimy,
	dimx,
	sampling_rate=0.25,
	center_ratio=0.5
	):
	"""
	Generate the sampling pdf of kspace
	"""
	ratio = 0
	while ratio != sampling_rate:
		pattern = np.zeros((dimy, dimx))
		print pattern.shape        
		##
		total_ones_column 		= int(np.floor(sampling_rate*dimy))
		set_center_index_len 	= int(np.floor(center_ratio*total_ones_column))
	
		## Deal with even sampling only    
		if set_center_index_len % 2 !=0:
			set_center_index_len += 1
			
		# Low frequency
		set_center_index_start = ((dimy-set_center_index_len)/2)
		set_center_index_end   = set_center_index_start + set_center_index_len 
		pattern[:,set_center_index_start:set_center_index_end] = 1
		
		# High frequency
		available_ones_column = total_ones_column - set_center_index_len
	
		left  = range(0, set_center_index_start-1)
		right = range(set_center_index_end, dimy-1)
		avail = np.hstack((left, right))
	
		#chosen = np.random.choice(avail, available_ones_column)
		chosen = random.sample(avail, available_ones_column)
		pattern[:,chosen]=1; 
	
	
		
		#print pattern.shape
		ratio = (pattern.sum()/(dimy*dimx))
		print "Ratio %4.5f" % (pattern.sum()/(dimy*dimx))
		pattern = np.fft.fftshift(pattern, axes=(0,1))
	return pattern
##################################################################
def test_generatePattern():
    #img = np.zeros(256, 256)
    pat = generatePattern(256, 256, sampling_rate=0.25, center_ratio=0.4)
    plt.imshow(pat, cmap=cm.gray)
    plt.axis('off')
    plt.show()
##################################################################
def generateMask(
	dimz, dimy, dimx,
	sampling_rate=0.25, center_ratio=0.5
	):
	"""
	Return 3D Sampling mask, 
	"""
	mask = np.zeros((dimz, dimy, dimx))
	for k in range(dimz):
		mask[k,:,:] = generatePattern(dimy, dimx, sampling_rate, center_ratio)
	return mask

##################################################################
if __name__ == '__main__':
	test_generatePattern()