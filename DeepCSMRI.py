# Import everything that needs

from __future__ import print_function


import sys
import os

import pydot
import cv2
import logging
import dicom
import scipy
import re 
import natsort
import time
import csv
import random
import subprocess

import skimage.io  				# For tif image reading
##################################################################
import numpy 	as np 			# For basic array processing
import pandas 	as pd
##################################################################
import matplotlib.pyplot as plt
##################################################################
from sklearn.cross_validation 	import KFold
# from scikits.statsmodels.tools  import categorical

from matplotlib.pyplot 			import cm
from graphviz 					import Digraph 
from IPython.display import SVG


########################################################
# nb_iter 		= 5
# nb_classes 		= 256
nb_folds 		= 3
epochs_per_iter = 1
batch_size 		= 1