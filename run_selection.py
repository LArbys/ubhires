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

def ubtri():
    model = "results/ubtri/001/snapshot_rmsprop_iter_17732.caffemodel"
    prototxt = "ub_trimese_resnet_deploy.prototxt"
    net = caffe.Net(prototxt, model, caffe.TEST)
    return net

def ubtri_6ch():
    model = "/mnt/disk0/kterao/tarinet6ch/snapshot_rmsprop_iter_3000.caffemodel"
    prototxt = "ub_6ch_trimese_resnet_deploy.prototxt"
    net = caffe.Net(prototxt, model, caffe.TEST)
    return net

if __name__ == "__main__":
    #net = dummy()
    net = ubtri_6ch()

    import ROOT as rt
    from array import array

    out = rt.TFile( "out_selection_6ch_val.root", "recreate" )
    run        = array('i',[0])
    subrun     = array('i',[0])
    event      = array('i',[0])
    index      = array('i',[0])
    truthlabel = array('i',[0])
    nuprob     = array('f',[0])
    tree = rt.TTree("net","Network Output")
    tree.Branch( "index", index, "index/I" )
    tree.Branch( "run", run, "run/I" )
    tree.Branch( "subrun", subrun, "subrun/I" )
    tree.Branch( "event", event, "event/I" )
    tree.Branch( "truthlabel", truthlabel, "truthlabel/I" )
    tree.Branch( "nuprob", nuprob, "nuprob/F" )

    entrylist = []
    with open("runlist_6ch_val.txt",'r' ) as frunlist:
        lines = frunlist.readlines()
        for l in lines:
            entrylist.append( int(l.strip().split()[0]) )

    for x in entrylist:
        net.layers[0].evalEntry(x)
        print "ENTRY ",x
        net.forward()
        for (label,probs,eventid) in zip( net.blobs["label"].data, net.blobs["probt"].data, net.blobs["eventid"].data ):
            truthlabel[0] = label
            nuprob[0] = probs[1]
            run[0] = eventid[0]
            subrun[0] = eventid[1]
            event[0] = eventid[2]
            index[0] = eventid[4]
            print "nu prob: ",nuprob[0]
            tree.Fill()

    print "[ENTER] to stop"
    out.Write()

    raw_input()
