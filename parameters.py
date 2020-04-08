import numpy as np
import helpers

### SIZE
n_HC_desired              = 4    #we want 9. represents cortical column
n_MC_per_HC               = 22   #we want ~50. in cortex ~ 100 per HC
L23_per_MC                = 30
L4_per_MC                 = 30
inh_per_MC                = 10     #4


### CONNECTIVITY
prob_pyr2pyr_recurrent    = 0.25     # static local connection within the same MC
prob_L4_to_L23            = 0.25 
prob_inh2pyr              = 0.7
prob_pyr2inh              = 0.7
prob_pyr2pyr_longrange    = 0.3      # long-range plastic connection, pattern specific


### TOPOLOGICAL INFORMATION
delay_dither_relative_sd  =.15 #delays can vary up to 15% from their distance-derived value(Long-range plastic)
conductance_speed         = 0.2  #was 2.[mm/ms] # 2011 paper says 0.2 mm/ms
HC_spatial_extent         = 0.64 # mm
MC_diameter               = 0.03 # mm
local_delay               = 1.0 + ((HC_spatial_extent/2)/conductance_speed)
mc_centers = helpers.CreateNetworkGrid()


### NEURON PARAMETERS
neuron_model = 'aeif_cond_alpha_multisynapse'
syn_ports = {'AMPA': 1, 'NMDA': 2, 'GABA': 3}

neuron_params = {
        'tau_syn'       : [5.,100., 5.0],
        'E_rev'         : [0., 0., -75.0],                                  
        'Delta_T'       :  3.0,                                            
        'E_L'           : -70.0,                                            
        'V_reset'       : -80.0,                                            
        'V_th'          : -55.0,                                            
        'a'             :  0.0,                                            
        'b'             :  86.,                                             
        'g_L'           :  14.0,                                            
        'gsl_error_tol' :  1e-12,                                           
        't_ref'         :  5.0,                                            
        'tau_w'         :  500.,
        'w'             :  90.,                                             
        }

### SYNAPTIC PARAMETERS
plastic_synapse_model='tsodyks2_synapse' 
AMPA_synapse_params = {
        'u'             : 0.18,                      
        'delay'         : 2.5,               ###useless as it is overidden later
        'receptor_type' : syn_ports['AMPA'],                                
        'tau_fac'       : 0.,                #6000.0,                         
        'tau_rec'       : 500. ,                
        'U'             : 0.18,
        'weight'        : 1.,       # florian had 0. later gain value =7.2.weight 0.4 for target epsp 0.15 of 2011 paper
        # florian also mentions that the e2e gain=25 translates to an effective ampa_weight of 4.6, e2e_gain =88 -> effective ampa weight 16.
        'x'             : 0.25
        }

NMDA_synapse_params = {
        'u'             : 0.18,                      
        'delay'         : 0.5,               ###useless as it is overidden later
        'receptor_type' : syn_ports['NMDA'],                                
        'tau_fac'       : 0.,                #6000.0,                         
        'tau_rec'       : 500. ,                
        'U'             : 0.18,
        'weight'        : 0.7,       # florian had 0. later gain value =0.04. weight 0.15 gives target epsp 0.15 of 2011 paper
        'x'             : 0.25
        }


# local static synapses
local_recurrent_pyr_synapse_params = {                          # target EPSP 0.9mV
        'delay'         : 0.5,                                  # Not sure about delay. florian has 1.66 but paper says 0.5 for plastic
        'receptor_type' : syn_ports['AMPA'],                                   
        'weight'        : 0.5
        }
        
inh_to_pyr_synapse_params = {                                   # target IPSP -2.5mV
        'delay'         : local_delay,
        'receptor_type' : syn_ports['GABA'],                                      
        'weight'        : 50.                                   # 50.
        }     
        
pyr_to_inh_synapse_params = {                                   # target EPSP 0.45mV
        'delay'         : local_delay,
        'receptor_type' : syn_ports['AMPA'],                                   
        'weight'        : 2.25                                  # to achieve target EPSP, weight must be 0.25. florian had 3.5 weight here
        }      
        
local_l4_to_l23_synapse_params = {                              # florian has weight of 25. too big i think
        'delay'         : local_delay,
        'receptor_type' : syn_ports['AMPA'],                                   
        'weight'        : 5.
        }      


### POISSON NOISE PARAMETERS
background_zmn_rate = []                      # homogeneous background noise across all HCs (uncorrelated poisson processes)
for hc in range(n_HC_desired):
    background_zmn_rate.append(250.)
    
L23_zmn_rate = [] #[150., 150., 150.]         # HC-specific poisson stimulation
for hc in range(n_HC_desired):
    L23_zmn_rate.append(250.)

zmn_weight = 0.8             # nSiemens
zmn_delay  = 0.1

noise_syn_e = {'model': 'static_synapse', 'weight': zmn_weight, 'delay': zmn_delay,'receptor_type': syn_ports['AMPA']}
noise_syn_i = {'model': 'static_synapse', 'weight': zmn_weight, 'delay': zmn_delay,'receptor_type': syn_ports['GABA']}
noise_conn_dict = {'rule': 'all_to_all', 'autapses': False, 'multapses': True}


### STIMULATION PARAMETERS
stim_length = 50.              # length of each cue stimulus
stim_rate   = 600.  
stim_weight = 1.5  
stim_delay  = 0.1  


### RECORDING PARAMETERS
SD_syn_dict = {'model': 'static_synapse', 'weight': 1.0, 'delay': 0.1}
SD_conn_dict = {'rule': 'all_to_all', 'autapses': False,'multapses': True}  

mm_n_per_MC = 2  
multimeterkeys = ['V_m'] 
multimeter_interval = 1.0     # temporal resolution


### SIMULATION PARAMETERS
dt           = 0.1                       # sim resolution (+smallest time unit)
max_delay    = 10.
num_memories = 1


