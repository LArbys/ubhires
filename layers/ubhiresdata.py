import caffe

import ROOT as rt
import numpy as np
from larcv import larcv
import yaml

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

        print dir(self)
        data_params = eval(self.data_param)
        self.batchsize = self.data_param["batch_size"]


    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        """ 
        we fill the top blobs.
        """
        pass

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
        
    def _batch_advancer(self):
        """
        this loads enough image for the next batch
        """
        pass
