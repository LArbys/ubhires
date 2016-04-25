# UB HiRes Caffe Network

This repository contains the code to build, run, analyze a caffe network using high-resolution crops from MicroBooNE event data.

We use caffe and the LArCV package.

## Pieces

### Setup script

setup.sh

* points to the version of caffe we want to us
* setups the python bindings for LArCV
* setups the python bindings for caffe

### layers

This is where the data layer is defined

### Status

![losscurve](https://github.com/LArbys/ubhires/blob/master/training_plot.png)
