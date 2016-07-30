#from __future__ import print_function


import numpy as np
import matplotlib.pyplot as plt
import random
import skimage.io
import skimage.measure


import scipy
import re 
import natsort
import time
import random
import cv2
from sklearn.cross_validation 		import KFold # For cross_validation
from random 						import randint
from skimage 						import exposure
from skimage 						import data, img_as_float
from scipy.misc 					import imresize
from scipy 							import ndimage
from scipy.ndimage.interpolation 	import map_coordinates
from scipy.stats 					import norm
