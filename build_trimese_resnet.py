import os,sys
import layer_tools as lt
import caffe
from caffe import params as P
from caffe import layers as L

augment_data = True
use_batch_norm = True
use_dropout = False

def buildnet( inputdb, mean_file, batch_size, height, width, nchannels, net_type="train"):
    net = caffe.NetSpec()

    crop_size = -1
    if augment_data:
        crop_size = width

    train = False
    if net_type=="train":
        train = True

    data_layers,label = lt.data_layer_trimese( net, inputdb, mean_file, batch_size, net_type, height, width, nchannels, [2,4], crop_size=768 )

    # First conv  layer
    branch_ends = []
    for n,layer in enumerate(data_layers):
        conv1 = lt.convolution_layer( net, layer, "plane%d_conv1"%(n), "tri_conv1_plane%d"%(n), 64, 2, 5, 3, 0.05, addbatchnorm=True, train=train )
        pool1 = lt.pool_layer( net, conv1, "plane%d_pool1"%(n), 3, 1 )

        conv2 = lt.convolution_layer( net, pool1, "plane%d_conv2"%(n), "tri_conv2_plane%d"%(n), 64, 2, 3, 3, 0.05, addbatchnorm=True, train=train )
        
        conv3 = lt.convolution_layer( net, conv2, "plane%d_conv3"%(n), "tri_conv3_plane%d"%(n), 64, 2, 3, 3, 0.05, addbatchnorm=True, train=train )

        pool3 = lt.pool_layer( net, conv3, "plane%d_pool3"%(n), 3, 1 )

        branch_ends.append( pool3 )
        
    concat = lt.concat_layer( net, "mergeplanes", *branch_ends )


    resnet1  = lt.resnet_module( net, concat,  "resnet1", 64*3, 3, 1, 1,8,32, use_batch_norm, train)
    resnet2  = lt.resnet_module( net, resnet1, "resnet2", 32, 3, 1, 1,8,32, use_batch_norm, train)
    resnet3  = lt.resnet_module( net, resnet2, "resnet3", 32, 3, 1, 1,16,64, use_batch_norm, train)
    
    resnet4  = lt.resnet_module( net, resnet3, "resnet4", 64, 3, 1, 1,16,64, use_batch_norm, train)
    resnet5  = lt.resnet_module( net, resnet4, "resnet5", 64, 3, 1, 1,16,64, use_batch_norm, train)
    resnet6  = lt.resnet_module( net, resnet5, "resnet6", 64, 3, 1, 1,32,128, use_batch_norm, train)

    resnet7  = lt.resnet_module( net, resnet6, "resnet7", 128, 3, 1, 1, 32,128, use_batch_norm, train)
    resnet8  = lt.resnet_module( net, resnet7, "resnet8", 128, 3, 1, 1, 32,128, use_batch_norm, train)
    resnet9  = lt.resnet_module( net, resnet8, "resnet9", 128, 3, 1, 1, 64,256, use_batch_norm, train)
        
    net.lastpool = lt.pool_layer( net, resnet9, "lastpool", 5, 1, P.Pooling.AVE )
    lastpool_layer = net.lastpool
    
    if use_dropout:
        net.lastpool_dropout = L.Dropout(net.lastpool,
                                         in_place=True,
                                         dropout_param=dict(dropout_ratio=0.5))
        lastpool_layer = net.lastpool_dropout
    
    fc1 = lt.final_fully_connect( net, lastpool_layer, nclasses=512 )
    fc2 = lt.final_fully_connect( net, fc1, nclasses=4096 )
    fc3 = lt.final_fully_connect( net, fc2, nclasses=2 )
    
    if train:
        net.loss = L.SoftmaxWithLoss(fc3, net.label )
        net.acc = L.Accuracy(fc3,net.label)
    else:
        net.probt = L.Softmax( fc3 )
        net.acc = L.Accuracy(fc3,net.label)

    return net

def append_rootdata_layer( prototxt ):
    fin = open(prototxt,'r')
    fout = open( prototxt.replace(".prototxt","_rootdata.prototxt"), 'w' )
    lines = fin.readlines()
    lout = []
    found_end_of_data = False
    mean_file = ""
    flist = ""
    for l in lines:
        #l = l.strip()
        if found_end_of_data:
            lout.append(l)
        if l=="}\n":
            found_end_of_data = True
        n = len(l.strip().split(":"))
        if n>=2 and l.strip().split(":")[0]=="mean_file":
            mean_file = l.strip().split(":")[1].strip()
        if n>=2 and l.strip().split(":")[0]=="source":
            flist = l.strip().split(":")[1].strip()
        if n>=2 and l.strip().split(":")[0]=="batch_size":
            batch_size = int(l.strip().split(":")[1].strip())
    print mean_file,flist,batch_size

    rootlayer = """
layer {
  name: "data"
  type: "ROOTData"
  top: "data"
  top: "label"

  root_data_param {
    source: %s
    mean: %s
    mean_producer: "mean"
    image_producer: "6ch_hires_crop"
    roi_producer: "tpc_hires_crop"
    nentries: %d
    batch_size: %d
    imin: "[35,10,30,10,40,10]"
    imax: "[400,400,400,400,400,400]"
    flat_mean: "[0,0,0,0,0,0]"
    random_adc_scale_mean: 1.0
    random_adc_scale_sigma: -1.0
    random_col_pad: 0
    random_row_pad: 0
  }    
}
""" % (flist,mean_file,batch_size,batch_size)

    print >>fout,rootlayer

    for l in lout:
        print>>fout,l,
    fin.close()
    fout.close()

if __name__ == "__main__":
    
    traindb    = "/home/taritree/working/larbys/ubhires/flist_train.txt"
    train_mean = "/home/taritree/working/larbys/staged_data/train6ch_mean.root"
    testdb     = "/home/taritree/working/larbys/ubhires/flist_test.txt"
    test_mean  = "/home/taritree/working/larbys/staged_data/val6ch_mean.root"
    

    train_net   = buildnet( traindb, train_mean, 36, 768, 768, 3, net_type="train"  )
    test_net    = buildnet( testdb,   test_mean, 1, 768, 768, 3, net_type="test"  )
    deploy_net  = buildnet( testdb, test_mean, 1, 768, 768, 3, net_type="deploy"  )

    testout   = open('ub_trimese_resnet_test.prototxt','w')
    trainout  = open('ub_trimese_resnet_train.prototxt','w')
    deployout = open('ub_trimese_resnet_deploy.prototxt','w')
    print >> testout, test_net.to_proto()
    print >> trainout, train_net.to_proto()
    print >> deployout, deploy_net.to_proto()
    testout.close()
    trainout.close()
    deployout.close()

    append_rootdata_layer( 'ub_trimese_resnet_train.prototxt' )
    append_rootdata_layer( 'ub_trimese_resnet_test.prototxt' )

    os.system("rm ub_trimese_resnet_train.prototxt")
    os.system("rm ub_trimese_resnet_test.prototxt")



