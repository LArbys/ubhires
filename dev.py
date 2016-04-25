# scratch space for dev

import layers.ubhiresdata as datalayer
import caffe
from caffe import layers as L
from caffe import params as P
from larcv import larcv
import numpy as np
import cv2

def dummy():
    net = caffe.NetSpec()
    pydata_params = dict(configfile="config_train.yaml")
    pylayer = 'UBHiResData'
    net.data, net.label = L.Python(module='layers.ubhiresdata', layer=pylayer,
                                   ntop=2, param_str=str(pydata_params))
    return net

if __name__ == "__main__":
    netspec = dummy()
    print netspec.to_proto()
    with open('dummy.prototxt','w') as f:
        f.write(str(netspec.to_proto()))

    net = caffe.Net( "dummy.prototxt", caffe.TEST )
    net.layermap = {}
    for n,name in enumerate(net._layer_names):
        print name,net.layers[n]
        net.layermap[name] = net.layers[n]

    net.layermap["data"].getEntry(0)
    event_img = net.layermap["data"].ioman.get_data( larcv.kProductImage2D, "tpc_hires_crop" )
    planeimgs = {}
    for img2d in event_img.Image2DArray():
        imgnd = larcv.as_ndarray( img2d )
        planeimgs[img2d.meta().plane()] = imgnd
    img = np.zeros( (planeimgs[0].shape[0],planeimgs[0].shape[1],3) )
    print img.shape
    img[...,0] = planeimgs[0]
    img[...,1] = planeimgs[1]
    img[...,2] = planeimgs[2]
    cv2.imwrite( "test_img.png", img )
