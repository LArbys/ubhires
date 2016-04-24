import caffe

import numpy as np
import larcv

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
        
        self.ioman = self._loadIOmanager( params )

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        pass

    def backward(self,top,propagate_down,bottom):
        pass

    def _loadIOmanager( self, params ):
        """
        load the larcv iomanager
        """
        print "IOMANAGER SETUP"
        print params

        
        
