import caffe

import ROOT as rt
import numpy as np
from larcv import larcv
import yaml

from Queue import Queue
from threading import Thread
from time import sleep

class EventData:
    # place to hold image data for blob
    def __init__( self ):
        pass

def makeEventData( iom, mean_images, index, config ):
    #evtimgs = iom.get_data( larcv.kProductImage2D, "tpc_hires_crop" )
    evtimgs = iom.get_data( larcv.kProductImage2D, "6ch_hires_crop" )
    evtroi  = iom.get_data( larcv.kProductROI, "tpc_hires_crop" )
    roi = evtroi.ROIArray().at(0)
    nchannels = evtimgs.Image2DArray().size()
    img2d_arr = np.zeros( mean_images.shape, dtype=np.float )
    min_thresh = config["imin"]
    max_thresh = config["imax"]
    for n,img2d in enumerate(evtimgs.Image2DArray()):
        # get image as ND array
        imgnd = larcv.as_ndarray( img2d )
        plane = img2d.meta().plane()

        # subtract mean
        imgnd -= mean_images[n,...]

        # subtract imin
        imgnd -= min_thresh[n]

        # zero pixels below threshold
        minmask = imgnd < 0.0
        maxmask = imgnd > max_thresh[n]
        imgnd[minmask] = 0.0
        imgnd[maxmask] = max_thresh[n]

        # copy to event array
        img2d_arr[n,...] = imgnd
        
    evtdata = EventData()
    evtdata.img2d_arr = img2d_arr
    # label
    if roi.Type()==larcv.kROICosmic:
        evtdata.label     = 0
    else:
        evtdata.label     = 1

    # event ID
    evtdata.eventid = np.zeros( (5), dtype=np.int )
    evtdata.eventid[0] = iom.event_id().run()
    evtdata.eventid[1] = iom.event_id().subrun()
    evtdata.eventid[2] = iom.event_id().event()
    evtdata.eventid[3] = -1 # window id goes here eventually
    evtdata.eventid[4] = index

    return evtdata
    

def fill_event_queue( iom, mean_images, event_queue, maxqueuesize, config ):

    nentries = iom.get_n_entries()
    ientry = 0
    while True:
        if event_queue.qsize()<maxqueuesize:
            if config["run_mode"]=="randomize":
                # get an entry from random
                ientry = np.random.randint(0,nentries)
                iom.read_entry( ientry )
            elif config["run_mode"]=="sequential":
                iom.read_entry( ientry )
    
            # get three TPC channel images: later we will be clever and understand how to specify this 

            # via text file (YMAL, json, whatever)

            evtdata = makeEventData( iom, mean_images, ientry, config )
            event_queue.put( evtdata )
            #print "filled queue w/ entry: ",ientry
            
            if config["run_mode"]=="sequential":
                ientry += 1
                if ientry>=nentries:
                    ientry = 0
        else:
            #print "queue is full (",maxqueuesize,")"
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
            
        self.batch_size = self.config["batch_size"]
        self._setupBranches( self.config )

        meanio = larcv.IOManager( larcv.IOManager.kREAD, "IOmean" )
        meanio.add_in_file( self.config["meanfile"] )
        meanio.initialize()
        mean_evtimg = meanio.get_data( larcv.kProductImage2D, "mean" )
        self.nchannels = int(mean_evtimg.Image2DArray().size())
        self.width    = int(mean_evtimg.Image2DArray().at(0).meta().cols())
        self.height   = int(mean_evtimg.Image2DArray().at(0).meta().rows())
        self.mean_img = np.zeros( ( self.nchannels, self.width, self.height), dtype=np.float )
        for ch,img2d in enumerate(mean_evtimg.Image2DArray()):
            self.mean_img[ch,...] = larcv.as_ndarray( img2d )[...]

        # set the blob sizes I guess
        data_shape  = (self.batch_size, self.nchannels, self.width, self.height ) 
        label_shape = (self.batch_size,)
        eventid_shape = (self.batch_size, 5)
        top[0].reshape( *data_shape )
        top[1].reshape( *label_shape )
        top[2].reshape( *eventid_shape )

        # depending on the run mode, we setup the queue
        self.event_queue = Queue()
        if self.config["run_mode"] in ["sequential","randomize"]:
            # setup the queue
            self.event_thread = Thread( target=fill_event_queue, args=(self.ioman, self.mean_img, self.event_queue, self.batch_size*2, self.config ) )
            self.event_thread.setDaemon(True)
            self.event_thread.start()
        elif self.config["run_mode"]=="selection":
            self.batch_size = 1
        else:
            raise ValueError("unrecognized run_mode. either [sequential,randomize,selection]")

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        """ 
        we fill the top blobs.
        """
        nfilled = 0
        maxattempts = 100
        while nfilled<self.batch_size:
            nattempts=0
            while self.event_queue.empty() and nattempts<maxattempts:
                print "waiting on image queue to be filled"
                sleep( 0.1 )
                nattempts += 1
            eventdata = self.event_queue.get()
            top[0].data[nfilled,...] = eventdata.img2d_arr[...]
            top[1].data[nfilled]     = eventdata.label
            top[2].data[nfilled,...] = eventdata.eventid
            nfilled += 1

    def backward(self,top,propagate_down,bottom):
        pass

    def getEntry( self, entry ):
        self.current_entry = entry
        if self.ioman is not None:
            self.ioman.read_entry( entry )

    def getEventID( self, run, subrun, event ):
        if self.ioman is not None:
            self.ioman.set_id( run, subrun, event )        

    def evalEntry( self, entry ):
        self.getEntry(entry)
        evtdata = makeEventData(self.ioman, self.mean_img, entry, self.config )
        self.event_queue.put( evtdata )
        print "evalEntry: ",entry

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
        
    def _batch_advancer(self):
        """
        this loads enough image for the next batch
        """
        pass
