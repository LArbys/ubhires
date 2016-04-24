# scratch space for dev

import layers.ubhiresdata as datalayer
import caffe
from caffe import layers as L
from caffe import params as P

def dummy():
    net = caffe.NetSpec()
    pydata_params = dict(mean=(104.00699, 116.66877, 122.67892),seed=1337)
    pydata_params['sbdd_dir'] = '../../data/sbdd/dataset'
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
