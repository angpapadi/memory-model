# to run from jupyter notebook

simply go to a cell and write:

%matplotlib inline

import simulation                 #this calls the simulation script 

import post

filename = 'filename-as-defined-in-the-end-of-simulation.py'

post.processing(filename)        #this generates spike rasters


____________

This will show 5 plots
1. a spike raster of pyramidal cells of MC0
2. a spike raster of pyramidal cells of HC0
3. a spike raster of pyramidal cells of the whole network
4. a spike raster of basket cells of one HC
5. a voltage trace of one pyramidal neuron
