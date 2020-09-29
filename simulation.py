import nest
import parameters
import helpers
import numpy as np
import timeit

nest.ResetKernel()
nest.set_verbosity('M_QUIET')    
nest.SetKernelStatus({"total_num_virtual_procs": 16})       
nest.SetKernelStatus({'resolution': parameters.dt,'min_delay': parameters.dt, 'max_delay': parameters.max_delay})
      
starttime = timeit.default_timer()                
network = helpers.createModel()
modelfinished=timeit.default_timer()

# Simulate
helpers.HCspecificInput(network)                                               #hc specific stimulation
nest.Simulate(1000.)
memories = helpers.LoadMemories(parameters.num_memories, network)              #orthogonal
helpers.CuePattern(network, memories[0], 2500.)                                #cue

nest.Simulate(parameters.phase_duration)

stoptime = timeit.default_timer()    
print('Total Runtime: {0:.2f} min'.format((stoptime-starttime)/60.))
print('Model Construction Runtime: {0:.2f} min'.format((modelfinished-starttime)/60.))

# Save device recordings to file
helpers.SaveSimulationToFile('runs/simresults_'+str(int(stoptime)),network, memories)
print('Simulation complete. Results have been saved to file.')


