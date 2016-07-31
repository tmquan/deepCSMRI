from DeepCSMRI import * # All the import were added in there
import mxnet as mx
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 21:03:22 2016
@author: tmquan
"""

from Utility import *



# def print_inferred_shape(net):
    # ar, ou, au = net.infer_shape(data=(BATCH_SIZE, 1, INPUT_SIZE, INPUT_SIZE))
    # print ou


# def convolution_module(net, kernel_size, pad_size, filter_count, stride=(1, 1), work_space=2048, batch_norm=True, down_pool=False, up_pool=False, act_type="relu", convolution=True):
    # if up_pool:
        # net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=filter_count, workspace = work_space)
        # net = mx.sym.BatchNorm(net)
        # if act_type != "":
            # net = mx.sym.Activation(net, act_type=act_type)
        # # print_inferred_shape(net)

    # if convolution:
        # conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count, workspace=work_space)
        # net = conv
        # # print_inferred_shape(conv)

    # if batch_norm:
        # net = mx.sym.BatchNorm(net)

    # if act_type != "":
        # net = mx.sym.Activation(net, act_type=act_type)

    # if down_pool:
        # pool = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
        # net = pool
        # # print_inferred_shape(net)

    # return net

def primal_factory(data, num_filter, kernel, stride, pad):
	conv 			= mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
	work_space=4096
	deconv 			= mx.symbol.Deconvolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad, workspace = work_space)
	
	return conv+deconv
def residual_factory(data, num_filter, kernel, stride, pad):	
	identity_data 	= data
	conv1 			= mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
	act1 			= mx.symbol.Activation(data = conv1, act_type='relu')
	# act1 		    = mx.symbol.Dropout(data = act1, p=0.25) 
	
	conv2 			= mx.symbol.Convolution(data = act1, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
	act2 			= mx.symbol.Activation(data = conv2, act_type='relu')
	
	
	conv3 			= mx.symbol.Convolution(data = act2, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
	conv3 		    = mx.symbol.Dropout(data = conv3, p=0.5) 
	new_data 		= conv3 + identity_data
	# new_data		= mx.symbol.Concat(*[conv3, identity_data])
	act3 			= mx.symbol.Activation(data = new_data, act_type='relu')
	
	return act3

def convolution_module(net, kernel_size, pad_size, filter_count, stride=(1, 1), work_space=4096, batch_norm=True, down_pool=False, up_pool=False, act_type="relu", convolution=False, residual=True):
	if up_pool:
		net = mx.symbol.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=filter_count, workspace = work_space)
		net = mx.symbol.BatchNorm(net)
		if act_type != "":
			net = mx.symbol.Activation(net, act_type=act_type)
	
	if convolution:
		net = mx.symbol.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count, workspace=work_space)
		net = mx.symbol.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count, workspace=work_space)
		
	if residual:
		# net = mx.symbol.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count, workspace=work_space)
		# for i in range(3):
			# net = residual_factory(net, filter_count, kernel_size, stride=(1, 1), pad=(1,1))
		# net = mx.symbol.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count, workspace=work_space)
		net = primal_factory(net, filter_count, kernel_size, stride=(1, 1), pad=(1,1))
		net = residual_factory(net, filter_count, kernel_size, stride=(1, 1), pad=(1,1))
		net = primal_factory(net, filter_count, kernel_size, stride=(1, 1), pad=(1,1))
		
	if batch_norm:
		net = mx.symbol.BatchNorm(net)
	
	if act_type != "":
		net = mx.symbol.Activation(net, act_type=act_type)
	
	if down_pool:
		net = mx.symbol.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*2, workspace=work_space)
		net = mx.symbol.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))

	
	return net

def get_res_unet():
	source = mx.sym.Variable("data")
	source = source/255.0
	net = source
	kernel_size = (3, 3)
	pad_size = (1, 1)
	filter_count = 32
		
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*1, down_pool=True)
	pool1	= net
	net		= mx.symbol.Dropout(net)
	
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*2, down_pool=True)
	pool2	= net
	net		= mx.symbol.Dropout(net)
	
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*4, down_pool=True)
	pool3	= net
	net		= mx.symbol.Dropout(net)
	
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*8, down_pool=True)
	pool4	= net
	net		= mx.symbol.Dropout(net)
	
	# net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*16, down_pool=True)
	# pool5	= net
	# net		= mx.symbol.Dropout(net)
	
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*16)
	
	# net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True)
	# net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True)
	
	# net		= mx.symbol.Dropout(net)
	# net		= mx.symbol.Concat(*[pool5, net])
	# net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*16)
	# net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*16, up_pool=True)
	
	net		= mx.symbol.Dropout(net)	
	# net		= mx.symbol.Concat(*[pool4, net])
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*8)
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*8, up_pool=True)

	net		= mx.symbol.Dropout(net)
	# net		= mx.symbol.Concat(*[pool3, net])
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*4)
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*4, up_pool=True)
	
	net		= mx.symbol.Dropout(net)	
	# net		= mx.symbol.Concat(*[pool2, net])
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*2)
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*2, up_pool=True)
	
	net		= mx.symbol.Dropout(net)
	# net		= mx.symbol.Concat(*[pool1, net])
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*1)
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*1, up_pool=True)
	
	net		= mx.symbol.Dropout(net)	
	
	
	net = convolution_module(net, kernel_size, pad_size, filter_count=256*tempo, batch_norm=False, act_type="")
	# print_inferred_shape(net)
	
	# net = mx.symbol.Flatten(net)
	return mx.symbol.LogisticRegressionOutput(data=net, name='softmax')
	
	


	
if __name__ == '__main__':
	# Draw the net
	data 	= mx.symbol.Variable('data')
	network = get_res_unet()
	dot = mx.viz.plot_network(network,
		None,
		# shape={"data" : (30, 1, 512, 512)}
		) 
	dot.graph_attr['rankdir'] = 'BT'
	
	
	
	