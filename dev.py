# scratch space for dev

import layers.ubhiresdata as datalayer
import caffe
from caffe import layers as L
from caffe import params as P
from larcv import larcv
import numpy as np
import cv2
from time import sleep

#caffe.set_mode_cpu()
gpu_id = 0
caffe.set_mode_gpu()
caffe.set_device(gpu_id)

def dummy():
    net = caffe.NetSpec()
    pydata_params = dict(configfile="config_train.yaml")
    pylayer = 'UBHiResData'
    net.data, net.label = L.Python(module='layers.ubhiresdata', layer=pylayer, ntop=2, param_str=str(pydata_params))
    return net

def ubtri():
    model = "results/ubtri/001/snapshot_rmsprop_iter_17732.caffemodel"
    prototxt = "ub_trimese_resnet_deploy.prototxt"
    net = caffe.Net(prototxt, model, caffe.TEST)
    return net

if __name__ == "__main__":
    #net = dummy()
    net = ubtri()

    for x in range(20,30):
        net.layers[0].evalEntry(x)
        net.forward()
        print net.blobs["label"].data
        print net.blobs["fc2"].data
        print net.blobs["probt"].data

    print "[ENTER] to stop"

    raw_input()
