import caffe

import ROOT as rt
import numpy as np
from larcv import larcv
import yaml

from Queue import Queue
from threading import Thread

class EventData:
    # place to hold image data for blob
    def __init__( self, iom ):
        pass

def fill_event_queue( iom, mean_images, event_queue, maxqueuesize ):

    while True:
        if event_queue.qsize()<maxqueuesize:
            # get an entry from random
            iom.read_entry( np.random.randint(0,iom.get_n_entries()) )
    
            # get three TPC channel images: later we will be clever and understand how to specify this 
            # via text file (YMAL, json, whatever)
    
            evtimgs = iom.get_data( larcv.kProductImage2D, "6ch_hires_crop" )
            roi     = iom.get_data( larcv.kProductROI, "tpc_hires_crop" )
            nchannels = evtimgs.Image2DArray().size()
            img2d_arr = np.zeros( mean_image, dtype=np.float )
            for n,img2d in enumerate(evtimgs.Image2DArray()):
                img2d_arr[n,...] = larcv.as_ndarray( img2d )
            evtdata = EventData()
            evtdata.img2d_arr = img2d_arr
            if roi.Type()==larcv.kROICosmics:
                evtdata.label     = 0
            else:
                evtdata.label     = 1
            event_queue.put( evtdata )
        else:
            print "queue is full (",maxqueuesize,")"
            sleep(0.1)

class UBHiResData(caffe.Layer):
    """
    Load (input image collection, labeled image (for sem. seg. or truth label)

    Use this to feed UB Hi Res data to the network.
    """
    
    def setup( self, bottom, top):
        """
        seems to be a required method for a PythonDataLayer
        """
        
        # get parameters
        params = eval(self.param_str)
        with open(params['configfile'], 'r') as f:
            self.config = yaml.load(f)

        self._setupBranches( self.config )

        meanio = larcv.IOManager( larcv.IOManager.kREAD, "IOmean" )
        meanio.add_in_file( self.config["meanfile"] )
        meanio.initialize()
        mean_evtimg = meanio.get_data( larcv.kProductImage2D, "mean" )
        w = mean_evtimg.Image2DArray().at(0).meta().width()
        h = mean_evtimg.Image2DArray().at(0).meta().height()
        self.mean_img = np.zeros( (mean_evtimg.Image2DArray().size(), w, h ), dtype=np.float )
        for ch,img2d in enumerate(mean_evtimg.Image2DArray()):
            mean_img[ch,...] = larcv.as_ndarray( img2d )[...]
            

        print dir(self)
        data_params = eval(self.data_param)
        self.batchsize = self.data_param["batch_size"]

        self.event_queue = Queue()
        self.event_thread = Thread( target=fill_event_queue, args=(self.ioman, self.mean_image, self.event_queue, self.batchsize*2 ) )
        self.event_thread.setDaemon(True)
        self.event_thread.start()
        


    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        """ 
        we fill the top blobs.
        """
        nfilled = 0
        while nfilled<self.batchsize:
            while q.empty():
                print "waiting on image queue to be filled"
                sleep( 0.1 )
            eventdata = self.event_queue.get()
            self._fillBlob( nfilled, eventdata )
            nfilled += 1

    def backward(self,top,propagate_down,bottom):
        pass

    def getEntry( self, entry ):
        if self.ioman is not None:
            self.ioman.read_entry( entry )

    def getEventID( self, run, subrun, event ):
        if self.ioman is not None:
            self.ioman.set_id( run, subrun, event )        

    def _setupBranches( self, config ):
        """
        load the larcv iomanager
        """
        print "IOMANAGER/BRANCH SETUP"

        self.ioman = larcv.IOManager( larcv.IOManager.kREAD, "IO")

        with open(self.config["filelist"],'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip()
                self.ioman.add_in_file( l )
        self.ioman.initialize()
        
    def _fillBlob( self, index, eventdata ):
        pass
        
    def _batch_advancer(self):
        """
        this loads enough image for the next batch
        """
        pass
