# scratch space for dev

import layers.ubhiresdata as datalayer
import caffe
from caffe import layers as L
from caffe import params as P
from larcv import larcv
import numpy as np
import cv2
from time import sleep

caffe.set_mode_cpu()


def dummy():
    net = caffe.NetSpec()
    pydata_params = dict(configfile="config_train.yaml")
    pylayer = 'UBHiResData'
    net.data, net.label = L.Python(module='layers.ubhiresdata', layer=pylayer, ntop=2, param_str=str(pydata_params))
                                   
    return net

if __name__ == "__main__":
    netspec = dummy()
    print netspec.to_proto()
    with open('dummy.prototxt','w') as f:
        f.write(str(netspec.to_proto()))

    net = caffe.Net( "dummy.prototxt", caffe.TEST )

    for i in range(0,10):
        print "pass: ",i
        net.forward()
        sleep(1)
    print "[ENTER] to stop"

    raw_input()
