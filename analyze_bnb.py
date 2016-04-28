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
    #net = ubtri()
    net = ubtri_6ch()

    import ROOT as rt
    from array import array

    out = rt.TFile( "out_bnb_6ch.root", "recreate" )
    run        = array('i',[0])
    subrun     = array('i',[0])
    event      = array('i',[0])
    index      = array('i',[0])
    truthlabel = array('i',[0])
    nuprob_div = array('f',400*[0.0])
    nuprob     = array('f',[0.0])
    
    tree = rt.TTree("net","Network Output")
    tree.Branch( "index", index, "index/I" )
    tree.Branch( "run", run, "run/I" )
    tree.Branch( "subrun", subrun, "subrun/I" )
    tree.Branch( "event", event, "event/I" )
    tree.Branch( "truthlabel", truthlabel, "truthlabel/I" )
    tree.Branch( "nuprob", nuprob, "nuprob/F" )
    tree.Branch( "nuprob_div", nuprob_div, "nuprob_div[400]/F" )

    current_event = None
    event_entry_count = 0
    max_index = -1
    max_prob = 0.0
    nfilled = 0
    x = 0
    while nfilled<10:

        print "BATCH ",x,current_event
        net.forward()
        for (label,probs,eventid) in zip( net.blobs["label"].data, net.blobs["probt"].data, net.blobs["eventid"].data ):

            truthlabel[0] = label

            event_tag = ( eventid[0], eventid[1], eventid[2] )
            print int(eventid[4]),probs[1]

            if current_event!=event_tag or current_event is None:
                if current_event is not None:
                    # end of event, store stuff
                    nuprob[0] = 0.0
                    for p in nuprob_div:
                        if p>nuprob[0]:
                            nuprob[0] = p
                    if nuprob[0]!=max_prob:
                        print "max probs diagree?: ",nuprob[0],max_prob
                    # set values
                    index[0] = max_index
                    run[0] = current_event[0]
                    subrun[0] = current_event[1]
                    event[0] = current_event[2]
                    tree.Fill() # fill per entry
                    print "Filled Event: max nuprob=",nuprob[0],"index=",index[0]
                    nfilled += 1
                    for i in range(0,400):
                        nuprob_div[i] = 0.0
                # reset counters
                current_event = event_tag
                event_entry_count = 0
                max_prob = 0
                max_index = -1

            nuprob_div[event_entry_count] = probs[1]
            if probs[1]>max_prob:
                max_prob = probs[1]
                max_index = eventid[4]
            event_entry_count+=1
        x += 1
    print "[ENTER] to stop"
    out.Write()

    raw_input()
