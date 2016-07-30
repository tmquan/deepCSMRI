# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 21:03:22 2016

@author: tmquan
"""

from Utility import *

class RMSECustom(mx.metric.EvalMetric):
    """Calculate Root Mean Squred Error loss and allow for some degugging"""
    def __init__(self):
        super(RMSECustom, self).__init__('rmsecustom')
        self.epoch = 0

    def update(self, labels, preds):
        assert len(labels) == len(preds)
        for label, pred in zip(labels, preds):
            assert label.shape == pred.shape
            pred_num = pred.asnumpy()
            label_num = label.asnumpy()
            self.sum_metric += np.sqrt(np.mean((label_num - pred_num.clip(0, 1)) ** 2))

            self.epoch += 1
            if self.epoch % 1000 == 0:
                for i in range(len(pred_num)):
                    p = pred_num[i]
                    p = p.reshape(INPUT_SIZE, INPUT_SIZE)
                    p *= 256
                    # cv2.imwrite("c:\\tmp\\dump_" + str(self.epoch) + "_" + str(i) + "_o.png", p)

                    l = label_num[i]
                    l = l.reshape(INPUT_SIZE, INPUT_SIZE)
                    l *= 256
                    # cv2.imwrite("c:\\tmp\\dump_" + str(self.epoch) + "_" + str(i) + ".png", l)

        self.num_inst += 1




def convolution_module(net, kernel_size, pad_size, filter_count, stride=(1, 1), work_space=2048, batch_norm=True, down_pool=False, up_pool=False, act_type="relu", convolution=True):
    if up_pool:
        net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=filter_count, workspace = work_space)
        net = mx.sym.BatchNorm(net)
        if act_type != "":
            net = mx.sym.Activation(net, act_type=act_type)

    if convolution:
        conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count, workspace=work_space)
        net = conv

    if batch_norm:
        net = mx.sym.BatchNorm(net)

    if act_type != "":
        net = mx.sym.Activation(net, act_type=act_type)

    if down_pool:
        pool = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
        net = pool

    return net


def get_net_180():
    source = mx.sym.Variable("data")
    kernel_size = (3, 3)
    pad_size = (1, 1)
    filter_count = 32
    pool1 = convolution_module(source, kernel_size, pad_size, filter_count=filter_count, down_pool=True)
    net = pool1
    pool2 = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 2, down_pool=True)
    net = pool2
    pool3 = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, down_pool=True)
    net = pool3
    pool4 = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, down_pool=True)
    net = pool4
    net = mx.sym.Dropout(net)
    pool5 = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 8, down_pool=True)
    net = pool5
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True)

    # dirty "CROP" to wanted size... I was on old MxNet branch so user conv instead of crop for cropping
    net = convolution_module(net, (3, 3), (1, 1), filter_count=filter_count * 4)
    net = mx.sym.Concat(*[pool3, net])
    net = mx.sym.Dropout(net)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True)

    net = mx.sym.Concat(*[pool2, net])
    net = mx.sym.Dropout(net)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True)
    convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4)
    net = mx.sym.Concat(*[pool1, net])
    net = mx.sym.Dropout(net)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 2)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 2, up_pool=True)

    net = convolution_module(net, kernel_size, pad_size, filter_count=1, batch_norm=False, act_type="")

    net = mx.symbol.Flatten(net)
    return mx.symbol.LogisticRegressionOutput(data=net, name='softmax')
	
if __name__ == '__main__':
	# Draw the net

	data 	= mx.symbol.Variable('data')
	# network = symmetric_residual()
	network = get_net_180()
	dot = mx.viz.plot_network(network,
		None,
		shape={"data" : (30, 1, 512, 512)}
		) 
	dot.graph_attr['rankdir'] = 'RL'
	
	
	
	