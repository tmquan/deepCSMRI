# Import everything that needs

from __future__ import print_function
import keras
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
from sklearn.cross_validation 	import KFold
from keras.models 				import Sequential
from keras.layers.core 			import Dense, Dropout, Activation
from keras.optimizers 			import SGD, Adam, RMSprop
from keras.utils 				import np_utils
from keras.utils.generic_utils 	import Progbar

from matplotlib.pyplot 			import cm
from graphviz 					import Digraph 