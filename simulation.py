import nest
import nest.voltage_trace
import parameters
import helpers
import pylab as plt
import numpy as np
import timeit, sys, copy, itertools, random, collections
from mpi4py import MPI

nest.ResetKernel()
nest.SetKernelStatus({'resolution': parameters.dt,'min_delay': parameters.dt, 'max_delay': parameters.max_delay})
'''
nest.set_verbosity('M_QUIET')    
#nest.SetKernelStatus({"total_num_virtual_procs": 2})                    # To ensure reproducibility

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

COMM.Barrier() 
starttime = timeit.default_timer()'''                  
network = helpers.createModel()
'''
COMM.Barrier()                                      # synchronize to get the actual model build time
modelfinished=timeit.default_timer()
'''

# Simulate
#nest.Simulate(1000.)
#orthogonal_memories = helpers.LoadMemories(parameters.num_memories, network)
nest.Simulate(8000.)


# cue
#memory_nodes = helpers.GetPatternNodes(orthogonal_memories[0], network)
#stimulus_node = nest.Create('poisson_generator',params={'rate' : 600.,'start':2000. ,'stop':2050.})
#nest.Connect(stimulus_node, memory_nodes, syn_spec=parameters.noise_syn_e)  
#nest.Simulate(500.)


#print(network['population_nodes'][0])
#print('Nodes that belong to learned pattern:')
#print(memory_nodes)

# Gather spikes


# Plot
helpers.PlottingJoe(network) 





############# cv2 of the most active neuron when no memories are loaded
spike_detector = network['device_nodes'][0]
dSD = nest.GetStatus(spike_detector, keys="events")[0]
evs = dSD["senders"]
ts = dSD["times"]

counts = np.bincount(evs)
winner = np.argmax(counts)
print('most frequent is',winner)

# get spike times for that neuron
winnertimes = ts[np.where(evs==winner)]

# compute cv2 for that neuron
print(winnertimes)
print(helpers.isi(winnertimes))
print(helpers.ComputeCV2(helpers.isi(winnertimes)))


'''
### FIND A BG NEURON THAT FIRES THE MOST AND COMPUTE ITS CV2
spike_detector = network['device_nodes'][0]
dSD = nest.GetStatus(spike_detector, keys="events")[0]
evs = dSD["senders"]
ts = dSD["times"]

# get all background neuron spikes
bgevents = evs[np.where(np.isin(evs,memory_nodes)==False)]     
bgtimes = ts[np.where(np.isin(evs,memory_nodes)==False)] 

# find the background neuron that fired the most
counts = np.bincount(bgevents)
winner = np.argmax(counts)
print('most frequent is',winner)

# get spike times for that neuron
winnertimes = bgtimes[np.where(bgevents==winner)]

# compute cv2 for that neuron
print(winnertimes)
print(helpers.isi(winnertimes))
print(helpers.ComputeCV2(helpers.isi(winnertimes)))
'''
