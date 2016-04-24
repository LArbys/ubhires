import os,sys
import larcv
import caffe
import numpy as np

class LArCVData(caffe.Layer):
    """
    base class to caffe data layers that will use LArCV root files.

    provides descdents methods for setting up IO manager.
    first we try specific case, then we work on how to generalize.
    general routines go here.
    """
    pass
