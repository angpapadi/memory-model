import numpy as np
import helpers

net_modules = ['STM']                               # list of all network modules that are part of the simulation
net = 'STM'             #TEMPORARY. IN THE FUTURE THIS SCRIPT SHOULD LOOP FOR ALL MODULES
syn_ports = {'AMPA': 1, 'NMDA': 2, 'GABA': 3}
DOWNSCALE_FACTOR = 0.95                          # set 0 for original network size. maximum 95% downscale

#global
global_tau_fac = 0.
STM_stim = 850.
STM_zmn = 575.

GLOBAL = {}
conductance_speed=2.#[mm/ms]
#net_distance=40. #distance PFC-ITC in macaque (used to compute S2L and L2S delays)
HC_spatial_extent=0.64
delay_dither_relative_sd=.15 #delays can vary up to 15% from their distance-derived value (applied to AMPA, i2e, e2i and indirectly also NMDA, they are AMPA copies)
b              =  86.#86pA!, as in Petrovichi'14? #200 earlier, Phil uses 100.
tau_w          = 500.#300. was tested in [Reference3],[ReferenceFastGamma]: more persistent activity #160ms before, nest original 144ms, Ala WM paper 2.7s, Phil 150ms, my MSc 160ms
tau_rec        = 500.#300. was tested in [Reference3],[ReferenceFastGamma]: more persistent activity #STP parameter as in Petrovici'14, Lundqvist.et.al. suggests 700, Phil 800.,Tsodyks_synapse default 800.
U              =  0.33
bias_gain      =  90.
tau_AMPA       =   5.
tau_NMDA       = 100.
tau_p          =5000. #5k good, attractors less explosive+even with 10k
#fmax           =  20.0  #fmax used in BCPNN learning rule #
AMPA_NMDA_ratio=10.#Phil: 5., yields more sequential replay
GLOBAL.update(conductance_speed=conductance_speed,HC_spatial_extent=HC_spatial_extent,delay_dither_relative_sd=delay_dither_relative_sd)
GLOBAL.update(b=b,tau_w=tau_w,tau_rec=tau_rec,U=U,bias_gain=bias_gain,tau_AMPA=tau_AMPA,tau_NMDA=tau_NMDA,AMPA_NMDA_ratio=AMPA_NMDA_ratio,tau_p=tau_p)

    #------------------------------STM---------------------------------------
    # DOWNSCALE_FACTOR is a percentage that helps downscale the network to run the simulation locally. it is multiplied by the original population size and then the final result is casted to integer
    #---STM Size---
SIZE={}
n_HC_desired        = 2#int(25 - 25 * DOWNSCALE_FACTOR + 2) #number of hypercolumns (minimum 2)
n_MC_per_HC         = 4#int(12 - 12*DOWNSCALE_FACTOR + 4) #number of minicolumns per hypercolumn (minimum 4)
L2e_per_MC          = int(20 - 20*DOWNSCALE_FACTOR) # minimum 1         # OBSOLETE AFTER OPERATION COLLAPSIO
L3ae_per_MC         = int(5 - 5*DOWNSCALE_FACTOR + 1) # minimum 1       # OBSOLETE AFTER OPERATION COLLAPSIO
L3be_per_MC         = int(5 - 5*DOWNSCALE_FACTOR + 1) # minimum 1       # OBSOLETE AFTER OPERATION COLLAPSIO
L4e_per_MC          = int(30 - 30*DOWNSCALE_FACTOR) #number of MC-specific input cells from Layer 4 # minimum 1
L2i_per_MC          = int(4 - 4*DOWNSCALE_FACTOR + 1)#number of L2i cells per minicolumn # minimum 1

SIZE.update(n_HC_desired=n_HC_desired, n_MC_per_HC=n_MC_per_HC, L2e_per_MC=L2e_per_MC, L3ae_per_MC=L3ae_per_MC, L3be_per_MC=L3be_per_MC, L4e_per_MC=L4e_per_MC, L2i_per_MC=L2i_per_MC)

#---STM Neurons---
NRN={}
cell_model = 'aeif_cond_alpha_multisynapse' #'aeif_cond_exp_multisynapse' #original
bias_gain  = GLOBAL['bias_gain'] #90. has spurious activity after kappa_modulated learning. Boosting to 95. prevents this. #Phil: 50. #WM model 65.
NRN.update(cell_model=cell_model,bias_gain=bias_gain)


    #---STM Connections/Synapses---
SYN={}
#E2E
synapse_model='tsodyks2_synapse' # 'bcpnn_synapse'
receptors=['AMPA','NMDA']
prob_L23e_to_L23e_intraHC=0.2 #changes since [zmn45] #Physiology: ~10%, Micke: 25% #scale down with size, set in-degree instead?
prob_L23e_to_L23e_interHC=0.2 #tested in LTMa for [zmn45] #donut shaped kernel arround the local HC
gain       =   7.2  #compromise value
gi_gain    =  77.  #dampens the amount of pre-loaded inhibition. (at 135., global inhibition disappears in the PreloadMeans dictionary). Less inhibition than (i.e. gi_gain>80.) will allow for superattractors
#earlier I used 25. when there is also some consolidation to reduce inhibition over time, lately the tuning around that idea has changed.
replacement={'AMPA':0.06, 'NMDA':0.06} #former inter_HC weights will be Preloaded with these values instead (if that flag is set)
SYN.update(synapse_model=synapse_model,receptors=receptors,gain=gain,gi_gain=gi_gain,prob_L23e_to_L23e_intraHC=prob_L23e_to_L23e_intraHC,prob_L23e_to_L23e_interHC=prob_L23e_to_L23e_interHC)

    # NEW -----L4 specifics ---                                                                                     ----!!!!----
L4e_to_L23e_weight=25. #rather big, yet helpful (see v37 and v37Learning). I probably should defend the EPSP
prob_L4e_to_L23e  =0.25 #strong specificity according to data
SYN.update(L4e_to_L23e_weight=L4e_to_L23e_weight,prob_L4e_to_L23e=prob_L4e_to_L23e)
#EIE
e2i_weight  = 3.5
prob_e2i    = 0.7
i2e_weight = -20.0 #this makes strong gamma happen, use this to adjust its power #NEGATIVEWEIGHT
prob_i2e   = 0.7
SYN.update(e2i_weight=e2i_weight,prob_e2i=prob_e2i,i2e_weight=i2e_weight,prob_i2e=prob_i2e)
#DELAYS #based on this idea: 1.5ms + distance at Speed= .2[mm]/[ms]
net_spatial_extent=17.*np.sqrt(2.)#distance see Paper draft calculation based on dlPFC size estimations
SYN.update(net_spatial_extent=net_spatial_extent)

# ---STM Simulation + Noise Drive---
STIM = {}
stim_length = 250.  # length of each stimulus
stim_gap = 0.  # (gap>10*tau_i avoids learning strong links between attractors)
stim_rate = 850.  # [after Refernce 5] #lower values seem to have a delayed activation effect (only 1-2 gamma phases during the cue at this level)
# 1100. earlier not fifferent from 850.
# 625 makes for a weak first pattern (but recency is desirable anyways).
# 450 (even less than 200!) still causes reactivations, BUT will not yield STM attractors that can reactvate with the current tuning (its a bit unclear why - probably delay in z-traces trom LTM) !!!
stim_zmn_rate = 600.
stim_weight = 1.5  # Weight of Stim Connector
stim_delay = 0.1  # Minimal delay (used for all poissson sources)
kappa_modulation = 6.
STIM.update(stim_length=stim_length, stim_gap=stim_gap, stim_rate=stim_rate, stim_zmn_rate=stim_zmn_rate, stim_weight=stim_weight, stim_delay=stim_delay, kappa_modulation=kappa_modulation)

zmn_weight = 1.5
zmn_delay = 0.1
f_desired = 1.0  # noise activity from init_zmn_rate=550 -->0.7Hz init_zmn_rate=750 -->1.2Hz

# DIFFERENT ZMN_RATE FOR EACH HYPERCOLUMN TARGETING L23e, might need to adjust noise now that it is distributed differently
init_zmn_rate = [550., 550.] # destributes initial weights + weak global inhibition without superattractors
L23e_zmn_rate = [850., 850.]

# 400 good for silent STM AFTER learning (however STM is not ever cued directly anyways).
# LTMs need 600 and is assigned further below.
# At 700 there are SOME LTM-driven reactivations.
L4e_zmn_rate = 300.0  # based on v37 BurstTriggered MC ActivityProfiles (lower rate means higher L4e_to_L23e_:weights are needed)
STIM.update(zmn_weight=zmn_weight, zmn_delay=zmn_delay, f_desired=f_desired, init_zmn_rate=init_zmn_rate, L23e_zmn_rate=L23e_zmn_rate, L4e_zmn_rate=L4e_zmn_rate)

# ---STM Recording---
REC = {}
# Spike recording
spike_rec_HCs = [0, 1]  # HCs to record spikes from, this includes all MCs from specified HCs
spike_rec_extra_MCs = []  # Name specific Minicolumns to record from aditionally
L2e_rec_per_MC = L2e_per_MC  # 20   #recorded L2 exitatory cells per MC             # OBSOLETE AFTER OPERATION COLLAPSIO
L3ae_rec_per_MC = L3ae_per_MC  # 5   #recorded L3a exitatory cells per MC           # OBSOLETE AFTER OPERATION COLLAPSIO
L3be_rec_per_MC = L3be_per_MC  # 5   #recorded L3b exitatory cells per MC           # OBSOLETE AFTER OPERATION COLLAPSIO
L23e_rec_per_MC = L2e_rec_per_MC + L3ae_rec_per_MC + L3be_rec_per_MC  # recorded L23e cells per MC
L4e_rec_per_MC = L4e_per_MC  # 30   #recorded L4 exitatory  cells per MC
L2i_rec_per_MC = L2i_per_MC  # 2   #recording one inhibitory cell per MCC
REC.update(spike_rec_HCs=spike_rec_HCs, spike_rec_extra_MCs=spike_rec_extra_MCs, L2e_rec_per_MC=L2e_rec_per_MC,
           L3ae_rec_per_MC=L3ae_rec_per_MC, L3be_rec_per_MC=L3be_rec_per_MC, L23e_rec_per_MC=L23e_rec_per_MC,
           L4e_rec_per_MC=L4e_rec_per_MC, L2i_rec_per_MC=L2i_rec_per_MC)
# ---Weight Recording Parameters---
w_rec_HCs = [0]
w_rec_extra_MCs = []
w_rec_per_MC = (L2e_per_MC + L3ae_per_MC + L3be_per_MC)  # recording all L23 cells
REC.update(w_rec_HCs=w_rec_HCs, w_rec_extra_MCs=w_rec_extra_MCs, w_rec_per_MC=w_rec_per_MC)
# ---Multimeter Recording Parameters---
multimeter_used = True  # No MM recording by default (expensive)
multimeter_MCs = list(range(n_HC_desired*n_MC_per_HC))      # record from all MCs
mm_n_per_MC = int(    30 - 30 * DOWNSCALE_FACTOR)  # record this many neurons from each multimeter_MCs #cHANGEd TO REFLECT DOWNSCALE
multimeterkeys = [
    'V_m']  # ,'I_GABA','I_AMPA','I_AMPA_NEG','I_NMDA','I_NMDA_NEG','bias'] #Angeliki: RAISES KEYERROR IF I DONT COMMENT THE REST OF THE KEYS OUT
# ['V_m', 'I_AMPA', 'I_NMDA', 'I_NMDA_NEG', 'I_AMPA_NEG', 'I_GABA','w','bias']
multimeter_interval = 1.0  # temporal resolution. I.e. fs=1000 '1ms resolution
REC.update(multimeter_used=multimeter_used, multimeter_MCs=multimeter_MCs, mm_n_per_MC=mm_n_per_MC, multimeterkeys=multimeterkeys, multimeter_interval=multimeter_interval)

    # define network dictionary
STM = {}
STM.update(SIZE=SIZE,NRN=NRN,SYN=SYN,STIM=STIM,REC=REC)

# -----------------------------------prog--------------------------------------
prog = {}
noisetime = 1500.0
freetime = 2500.0
constime = 6000.0  # 6000. before zmn45 #Sometimes there are spurious attractors at 8000., because the 25. gi_gain still relieves too much of the inhibition (maybe gi_gain=20. would allow an 8sec consolidation?)
emptytime = 500.0
cooldowntime = 100.0
cuetime = 450.  # brief cues for testing pattern recognition/and triggering activations. For reverberations that can be learned in the kappa-modulated phase, whis needs to be longer than 25ms (see Tuning50)!
prog.update(noisetime=noisetime, freetime=freetime, constime=constime, emptytime=emptytime, cooldowntime=cooldowntime,            cuetime=cuetime)

    # -----------------------------------other--------------------------------------
other = {}
dt = .1  # nest Simulation resolution (+smallest time unit)
noiseweightpreload = False  # one way on initializing the network
Noisepreload_file = 'Preload/Noisepreload_5k'
other.update(dt=dt, noiseweightpreload=noiseweightpreload, Noisepreload_file=Noisepreload_file)
# General Weight Recording and Handling
weight_readout_injection = +15.  # active weight readout injection
wait_for_NMDA = True  # let the synaptic time constant decay after active weight readout
weight_readout_spikerecord = False  # debugging option for "hidden" weight-readout-spikes
dump_weight_data = False  # Dump raw connection weight data after analysis?
execute_slow_weight_analysis = False  # currently not well implemented AND not paralleized: So full mean wight analysis can take LONG!
execute_binned_rate_analysis = False  # replaced with EMA analysis
execute_SheepCount = False  #
dump_model_conn_data = True  # Dump raw connection data after analysis?
other.update(weight_readout_injection=weight_readout_injection, wait_for_NMDA=wait_for_NMDA,
             weight_readout_spikerecord=weight_readout_spikerecord, dump_weight_data=dump_weight_data, execute_slow_weight_analysis=execute_slow_weight_analysis,
             execute_binned_rate_analysis=execute_binned_rate_analysis, execute_SheepCount=execute_SheepCount, dump_model_conn_data=dump_model_conn_data)


params ={}
params.update(GLOBAL=GLOBAL,STM=STM,prog=prog,other=other)

NRN = params[net]['NRN']
SYN = params[net]['SYN']
STIM = params[net]['STIM']
REC = params[net]['REC']
S = params[net]['SIZE']


#FINALIZEMYPARAMS
SYN['delay_IntraHC'] = 1.5 + (GLOBAL['HC_spatial_extent'] / 2) / GLOBAL[    'conductance_speed']  # distance=0.64/2, as there the average HC connection goes half way across?
SYN['delay_eie'] = 1.5 + (GLOBAL['HC_spatial_extent'] / 2) / GLOBAL[    'conductance_speed']  # distance=0.64/2, as there the average HC connection goes half way across?
SYN['delay_InterHC'] = 1.5 + SYN['net_spatial_extent'] / GLOBAL[    'conductance_speed']  # distance see Paper draft calculation based on dlPFC size estimations

SYN['AMPA_gain'] = SYN['gain'] * (GLOBAL['AMPA_NMDA_ratio'] - 1.) / (     GLOBAL['U'] * GLOBAL['tau_AMPA'] * GLOBAL['AMPA_NMDA_ratio'])
SYN['NMDA_gain'] = SYN['gain'] / (GLOBAL['U'] * GLOBAL['tau_NMDA'] * GLOBAL['AMPA_NMDA_ratio'])


params[net]['SIZE']['L23e_per_MC'] = params[net]['SIZE']['L2e_per_MC'] + params[net]['SIZE']['L3ae_per_MC'] + \
                                     params[net]['SIZE']['L3be_per_MC']
params[net]['REC']['L2e_rec_per_MC'] = np.min([params[net]['SIZE']['L2e_per_MC'], params[net]['REC']['L2e_rec_per_MC']])
params[net]['REC']['L3ae_rec_per_MC'] = np.min(    [params[net]['SIZE']['L3ae_per_MC'], params[net]['REC']['L3ae_rec_per_MC']])
params[net]['REC']['L3be_rec_per_MC'] = np.min(    [params[net]['SIZE']['L3be_per_MC'], params[net]['REC']['L3be_rec_per_MC']])
params[net]['REC']['L4e_rec_per_MC'] = np.min([params[net]['SIZE']['L4e_per_MC'], params[net]['REC']['L4e_rec_per_MC']])
params[net]['REC']['L2i_rec_per_MC'] = np.min([params[net]['SIZE']['L2i_per_MC'], params[net]['REC']['L2i_rec_per_MC']])
params[net]['REC']['L23e_rec_per_MC'] = params[net]['REC']['L2e_rec_per_MC'] + params[net]['REC']['L3ae_rec_per_MC'] + \
                                        params[net]['REC']['L3be_rec_per_MC']

NRN['neuron_params']={
        'tau_syn'      : [GLOBAL['tau_AMPA'], GLOBAL['tau_NMDA'], 5.0],
        'E_rev'        : [0., 0., -75.0],                               # CORRECT VALUES. WHEN AT LEAST ONE VALUE IS >-25 THROWS NEST BAD PROPERTY ERROR
        'E_rev'        : [-25.0, -25.0, -75.0],                         # WRONG VALUES NEST DEBUGGING

        'Delta_T'      :   3.0, #AdExp spike rise time
        'E_L'          : -70.0, #nest default is -70.6
        'V_reset'      : -80.0, #nest default is -60
        'V_th'         : -55.0, #nest default is -50.4 # CORRECT VALUE IS -55. Value -40 used for DEBUGGING as a tradeoff to input the correct Erev values
        'a'            :   0.0, #no subthreshold adaptation, nest default is 4.0
        'b'            :GLOBAL['b'], #spike adaptation
        'g_L'          :  14.0,  #nest default is 30.0; 14.0 yields slower tau_m=20ms, due to tau_m=C_m/g_L
        'gsl_error_tol':  1e-12,  #decr.by 6 magnitudes to 1e-12 for more numerical stability #default is 1e-6 i guess
        't_ref'        :   5.0,  #absolute refractory period (implies fmax<200Hz)
        'tau_w'        :GLOBAL['tau_w'],
        'w'            :  NRN['bias_gain'],#0.0, #adaptation variable #changed to old gain value after talking with anders
        }

SYN['AMPA_synapse_params']={
        'u'         :GLOBAL['U'],#only the initial value, not the real parameter, which is Capital U
        'delay'     :params['other']['dt'],
        'receptor_type': syn_ports['AMPA'],#1
        'tau_fac'   :   0.0, #Tsodycks-Markram facilitation
        'tau_rec'   :GLOBAL['tau_rec'], #Tsodycks-Markram depression 800.
        'U'         :GLOBAL['U'],
        'weight'    : SYN['AMPA_gain'], # 0.0,        #POTENTIALLY INITIAL WEIGHT . USE GAIN INSTEAD OF 0. #changed to old gain value after talking with anders
        'x'         :   0.25
        }

SYN['NMDA_synapse_params']={
        'u'         :GLOBAL['U'],#only the initial value, not the real parameter, which is Capital U
        'delay'     :params['other']['dt'],
        'receptor_type':  syn_ports['NMDA'], #2
        'tau_fac'   :   0.0, #Tsodycks-Markram facilitation
        'tau_rec'   :GLOBAL['tau_rec'], #Tsodycks-Markram depression 800.
        'U'         :GLOBAL['U'],
        'weight'    :   SYN['NMDA_gain'], # 0.0, #changed to old gain value after talking with anders
        'x'         :   0.25
        }


for net in ['STM']:  # Grid Topology (adjusting numbers for topological grid placement)
    S = params[net][        'SIZE']  # for better readability of this block (as S is a reference, all operations on S will execute on the actual params dictionary)
    if S['n_HC_desired'] == 0:
        for key in ['HC_rows', 'HC_columns', 'n_HC', 'n_MC_per_HC', 'grid_extent', 'grid_center']:
            S[key] = 0
    else:
        S['HC_rows'] = int(round(np.sqrt(S['n_HC_desired'])))
        S['HC_columns'] = int(round(S['n_HC_desired'] / S['HC_rows']))
        S['n_HC'] = S['HC_rows'] * S['HC_columns']
        if S['n_HC'] != S['n_HC_desired']:
            print(('...Warning: desired number of Hypercolumns per network was' + str(
                S['n_HC_desired']) + ' but due to topological geometry the established number is now ' + str(
                S['n_HC'])))
        S['grid_extent'] = [S['HC_columns'] + 0.0, S['HC_rows'] + 0.0];
        S['grid_center'] = [S['grid_extent'][0] / 2 + 0.5, S['grid_extent'][1] / 2 + 0.5]
        S['n_MC'] = S['n_HC'] * S['n_MC_per_HC']  # total number of minicolumns
        S['MC_HC'] = [range(hc * S['n_MC_per_HC'], (hc + 1) * S['n_MC_per_HC']) for hc in range(S['n_HC'])]
        S['HC_MC'] = [S['MC_HC'].index(sublist) for sublist in S['MC_HC'] for mc in sublist]
        S['n_exc'] = S['n_HC'] * S['n_MC_per_HC'] * (S['L23e_per_MC'] + S['L4e_per_MC'])  # total number of exc neurons
        S['n_inh'] = S['n_HC'] * S['n_MC_per_HC'] * S['L2i_per_MC']
        S['n_neurons'] = S['n_exc'] + S['n_inh']

for net in ['STM']:  # Totaling Recorders
    S = params[net][        'SIZE']  # for better readability of this block (as S is a reference, all operations on S will execute on the actual params dictionary)
    R = params[net][        'REC']  # for better readability of this block (as R is a reference, all operations on R will execute on the actual params dictionary)

    R['spike_rec_MCs'] = []
    for hc in R['spike_rec_HCs']:
        R['spike_rec_MCs'].extend(range(hc * S['n_MC_per_HC'], (hc + 1) * S['n_MC_per_HC']))
    for mc in R['spike_rec_extra_MCs']:
        if mc not in R['spike_rec_MCs']:
            R['spike_rec_MCs'].extend(range(hc * S['n_MC_per_HC'], (hc + 1) * S['n_MC_per_HC']))
    R['spike_rec_MCs'].sort()

    R['w_rec_MCs'] = []
    for hc in R['w_rec_HCs']:
        R['w_rec_MCs'].extend(range(hc * S['n_MC_per_HC'], (hc + 1) * S['n_MC_per_HC']))
    for mc in R['w_rec_extra_MCs']:
        if mc not in R['w_rec_MCs']:
            R['w_rec_MCs'].extend(range(hc * S['n_MC_per_HC'], (hc + 1) * S['n_MC_per_HC']))
    R['w_rec_MCs'].sort()
    R['w_rec_MCs_pre'] = R['w_rec_MCs']
    R['w_rec_MCs_post'] = R['w_rec_MCs']


# STIM MATRICES
for net in ['STM']:
    ST = params[net]['STIM']
    S = params[net]['SIZE']
    ST['stim_matrix_default'] = helpers.stim_matrix_generator(S['n_MC_per_HC'], params, net, 'orthogonal')
    # ST['stim_matrix_random']=stim_matrix_generator(S['n_MC_per_HC'],params,net,'random')


params['prog']['cooldownphase'] = {
    'name': 'cooldown_default',
    'type': 'free',
    'length': params['prog']['cooldowntime'],
    'L23e_cell_params': {
        'STM': {'b': params['STM']['NRN']['neuron_params']['b']}},
    # {'kappa': 0.,'gain': params['LTMb']['NRN']['bias_gain'],'b': params['LTMb']['NRN']['neuron_params']['b']}},
    'AMPA_params': {
        'STM': {},
        },  # backward
    # 'STM':{'K' : 0., 'gain': 0., 'stp_flag': 1.},
    # 'LTMa':{'K' : 0., 'gain': 0., 'stp_flag': 1.},
    # 'LTMb':{'K' : 0., 'gain': 0., 'stp_flag': 1.},
    # 'L2S':{'weight':0.},  #forward
    # 'S2L':{'K' : 0., 'gain': 0., 'stp_flag': 1.}}, #backward

    'NMDA_params': {
        'STM': {},  # 'STM':{'K' : 0., 'gain': 0., 'stp_flag': 1.},
        },  # 'S2L':{'K' : 0., 'gain': 0., 'stp_flag': 1.}}, #backward
    'L4e_to_L23e_params': {
        'STM': {'weight': params['STM']['SYN']['L4e_to_L23e_weight']}
        },
    'i2e_synapse_params': {
        'STM': {'weight': params['STM']['SYN']['i2e_weight']},  # activate WTA
        },
    # activate WTA
    'L23e_zmn_rate': {
        'STM': params['STM']['STIM']['L23e_zmn_rate'],
        },
    'L4e_zmn_rate': {
        'STM': params['STM']['STIM']['L4e_zmn_rate'],
        }
}
params['prog']['noisephase'] = {
    'name': 'noise_default',
    'type': 'noise',
    'length': params['prog']['noisetime'],
    'L23e_cell_params': {
        'STM': {'b': params['STM']['NRN']['neuron_params']['b']},
        },
    # {'kappa': 1.,'gain': 0.,'b': params['LTMb']['NRN']['neuron_params']['b']}},
    'AMPA_params': {
        'STM': {}
        },
    'NMDA_params': {
        'STM': {},  # {'K' : 1., 'gain': 0., 'stp_flag': 1.},
        },  # {'K' : 1., 'gain': 0., 'stp_flag': 1.}}, #backward
    'L4e_to_L23e_params': {
        'STM': {'weight': params['STM']['SYN']['L4e_to_L23e_weight']},
        },
    'i2e_synapse_params': {
        'STM': {'weight': params['STM']['SYN']['i2e_weight']},  # activate WTA
        },  # activate WTA
    'L23e_zmn_rate': {
        'STM': params['STM']['STIM']['init_zmn_rate'],
        },
    'L4e_zmn_rate': {
        'STM': params['STM']['STIM']['L4e_zmn_rate'],
        }
}
params['prog']['stimphase'] = {
    'name': 'stim_default',
    'type': 'stim',
    'length': 1000.,  # cannot actually be set manually, overwritten at runtime depending on the stimulation.
    'L23e_cell_params': {
        'STM': {'b': params['STM']['NRN']['neuron_params']['b']},
        },

    # 'STM':{'kappa': 1.,'gain': 0.,'b': params['STM']['NRN']['neuron_params']['b']},
    # 'LTMa':{'kappa': 1.,'gain': 0.,'b': params['LTMa']['NRN']['neuron_params']['b']},
    # 'LTMb':{'kappa': 1.,'gain': 0.,'b': params['LTMb']['NRN']['neuron_params']['b']}},
    'AMPA_params': {
        'STM': {},
        'LTMa': {},
        'LTMb': {},
        },
    # 'S2L':{'K' : 1., 'gain': 0., 'stp_flag': 1.}}, #assumes that STM and LTMa are usually stimulated in conjunction, so S2L has to learn then)
    'NMDA_params': {
        'STM': {},
        },
    # 'S2L':{'K' : 1., 'gain': 0., 'stp_flag': 1.}}, #assumes that STM and LTMa are usually stimulated in conjunction, so S2L has to learn then)
    'L4e_to_L23e_params': {
        'STM': {'weight': params['STM']['SYN']['L4e_to_L23e_weight']},
        },
    'i2e_synapse_params': {
        'STM': {'weight': params['STM']['SYN']['i2e_weight']},  # activate WTA
        },  # activate WTA
    'stim_matrix': {
        'STM': params['STM']['STIM']['stim_matrix_default'],
        },
    'stim_gap': {
        'STM': params['STM']['STIM']['stim_gap'],
        },
    'stim_length': {
        'STM': params['STM']['STIM']['stim_length'],
        },
    'stim_rate': {
        'STM': params['STM']['STIM']['stim_rate'],
        },
    'stim_weight': {
        'STM': params['STM']['STIM']['stim_weight'],
        },
    'stim_delay': {
        'STM': params['STM']['STIM']['stim_delay'],
        },
    'L23e_zmn_rate': {
        'STM': params['STM']['STIM']['L23e_zmn_rate'],
        },
    'L4e_zmn_rate': {
        'STM': params['STM']['STIM']['L4e_zmn_rate'],
        }
}
params['prog']['cuephase'] = {
    # similar to stimphase, but there is no learning, and the gain is on to allow for immideate attractor activation, stim_matrix is processed differently
    'name': 'cue_default',
    'type': 'cue',
    'length': params['prog']['cuetime'],  # will be overwritten at Runtime with the longest cuetime
    'L23e_cell_params': {
        'STM': {'b': params['STM']['NRN']['neuron_params']['b']},
        },

    'AMPA_params': {
        'STM': {'U': params['STM']['SYN']['AMPA_synapse_params']['U']},
        },
    # 'S2L':{'K' : 0., 'gain': params['S2L']['SYN']['AMPA_gain'], 'stp_flag': 1.}}, #backward
    'NMDA_params': {
        'STM': {'U': params['STM']['SYN']['NMDA_synapse_params']['U']},
        },
    # 'S2L':{'K' : 0., 'gain': params['S2L']['SYN']['NMDA_gain'], 'stp_flag': 1.}}, #backward
    'L4e_to_L23e_params': {
        'STM': {'weight': params['STM']['SYN']['L4e_to_L23e_weight']},
        },
    'i2e_synapse_params': {
        'STM': {'weight': params['STM']['SYN']['i2e_weight']},  # activate WTA
        },
    'stim_matrix': {
        'STM': params['STM']['STIM']['stim_matrix_default'][0],  # cues are only one pattern/vector, this is the first
        },
    'cuetime': {
        'STM': params['prog']['cuetime'],
        },
    'stim_rate': {
        'STM': params['STM']['STIM']['stim_rate'],
        },
    'stim_weight': {
        'STM': params['STM']['STIM']['stim_weight'],
        },
    'stim_delay': {
        'STM': params['STM']['STIM']['stim_delay'],
        },
    'L23e_zmn_rate': {
        'STM': params['STM']['STIM']['stim_zmn_rate'],
        },
    'L4e_zmn_rate': {
        'STM': params['STM']['STIM']['L4e_zmn_rate'],
        }
}
params['prog']['freephase'] = {
    'name': 'free_default',
    'type': 'free',
    'length': params['prog']['freetime'],
    'L23e_cell_params': {
        'STM': {'b': params['STM']['NRN']['neuron_params']['b']},
        },
    'AMPA_params': {
        'STM': {},
        },
    'NMDA_params': {
        'STM': {},
        },
    # 'S2L':{'K' : 0., 'gain': params['S2L']['SYN']['NMDA_gain'], 'stp_flag': 1.}}, #backward
    'L4e_to_L23e_params': {
        'STM': {'weight': params['STM']['SYN']['L4e_to_L23e_weight']},
        },
    'i2e_synapse_params': {
        'STM': {'weight': params['STM']['SYN']['i2e_weight']},  # activate WTA
        },  # activate WTA
    'L23e_zmn_rate': {
        'STM': params['STM']['STIM']['L23e_zmn_rate'],
        },
    'L4e_zmn_rate': {
        'STM': params['STM']['STIM']['L4e_zmn_rate'],
        }
}
params['prog']['modulationphase'] = {
    'name': 'cons_modulated',
    'type': 'cons',
    'length': 165.,
    'L23e_cell_params': {
        'STM': {'gain': params['STM']['NRN']['bias_gain'], 'b': params['STM']['NRN']['neuron_params']['b']},
        },
    'AMPA_params': {
        'STM': {'K': params['STM']['STIM']['kappa_modulation'], 'gain': params['STM']['SYN']['AMPA_gain'],
                'stp_flag': 1.},
        },  # backward
    'NMDA_params': {
        'STM': {'K': params['STM']['STIM']['kappa_modulation'], 'gain': params['STM']['SYN']['NMDA_gain'],
                'stp_flag': 1.},
        },  # backward
    'L4e_to_L23e_params': {
        'STM': {'weight': params['STM']['SYN']['L4e_to_L23e_weight']},
        },
    'i2e_synapse_params': {
        'STM': {'weight': params['STM']['SYN']['i2e_weight']},  # activate WTA
        },  # activate WTA
    'L23e_zmn_rate': {
        'STM': params['STM']['STIM']['L23e_zmn_rate'],
        },
    'L4e_zmn_rate': {
        'STM': params['STM']['STIM']['L4e_zmn_rate'],
        }
}
params['prog']['lightmodulationphase'] = {
    'name': 'cons_modulated',
    'type': 'cons',
    'length': 165.,
    'L23e_cell_params': {
        'STM': {'gain': params['STM']['NRN']['bias_gain'], 'b': params['STM']['NRN']['neuron_params']['b'] / 2},
        }, #/2
    'AMPA_params': {
        'STM': {'K': params['STM']['STIM']['kappa_modulation'], 'gain': params['STM']['SYN']['AMPA_gain'],
                'stp_flag': 1., 'U': params['STM']['SYN']['AMPA_synapse_params']['U'] / 2},
        },  # backward
    'NMDA_params': {
        'STM': {'K': params['STM']['STIM']['kappa_modulation'], 'gain': params['STM']['SYN']['NMDA_gain'],
                'stp_flag': 1., 'U': params['STM']['SYN']['NMDA_synapse_params']['U'] / 2},
        },  # backward
    'L4e_to_L23e_params': {
        'STM': {'weight': params['STM']['SYN']['L4e_to_L23e_weight']},
        },
    'i2e_synapse_params': {
        'STM': {'weight': params['STM']['SYN']['i2e_weight']},  # activate WTA
        },  # activate WTA
    'L23e_zmn_rate': {
        'STM': params['STM']['STIM']['L23e_zmn_rate'],
        },
    'L4e_zmn_rate': {
        'STM': params['STM']['STIM']['L4e_zmn_rate'],
        }
}
params['prog']['lightcuephase'] = {
    # similar to stimphase, but there is no learning, and the gain is on to allow for immideate attractor activation, stim_matrix is processed differently
    'name': 'cue_halfSTP',
    'type': 'cue',
    'length': params['prog']['cuetime'],
    'L23e_cell_params': {
        'STM': {'gain': params['STM']['NRN']['bias_gain'], 'b': params['STM']['NRN']['neuron_params']['b'] / 2},
        },
    'AMPA_params': {
        'STM': {'K': 1.01, 'gain': params['STM']['SYN']['AMPA_gain'], 'stp_flag': 1.,
                'U': params['STM']['SYN']['AMPA_synapse_params']['U'] / 2},
        },  # backward
    'NMDA_params': {
        'STM': {'K': 1.01, 'gain': params['STM']['SYN']['NMDA_gain'], 'stp_flag': 1.,
                'U': params['STM']['SYN']['NMDA_synapse_params']['U'] / 2},
        },  # backward
    'L4e_to_L23e_params': {
        'STM': {'weight': params['STM']['SYN']['L4e_to_L23e_weight']},
        },
    'i2e_synapse_params': {
        'STM': {'weight': params['STM']['SYN']['i2e_weight']},  # activate WTA
        },  # activate WTA
    'stim_matrix': {
        'STM': params['STM']['STIM']['stim_matrix_default'][0],  # cues are only one pattern/vector, this is the first
        },
    'cuetime': {
        'STM': params['prog']['cuetime'],
        },
    'stim_rate': {
        'STM': params['STM']['STIM']['stim_rate'],
        },
    'stim_weight': {
        'STM': params['STM']['STIM']['stim_weight'],
        },
    'stim_delay': {
        'STM': params['STM']['STIM']['stim_delay'],
        },
    'L23e_zmn_rate': {
        'STM': params['STM']['STIM']['stim_zmn_rate'],
        },
    'L4e_zmn_rate': {
        'STM': params['STM']['STIM']['L4e_zmn_rate'],
        }
}
params['prog']['consphase'] = {
    'name': 'cons_default',
    'type': 'cons',
    'length': params['prog']['constime'],  # overwritten at runtime depending on the stimulation.
    'L23e_cell_params': {
        'STM': {'gain': params['STM']['NRN']['bias_gain'], 'b': params['STM']['NRN']['neuron_params']['b']},
        },
    'AMPA_params': {
        'STM': {'K': 1., 'gain': params['STM']['SYN']['AMPA_gain'], 'stp_flag': 1.,
                'U': params['STM']['SYN']['AMPA_synapse_params']['U']},  # until [Reference5], i did dont set or reset U
        },
    'NMDA_params': {
        'STM': {'K': 1., 'gain': params['STM']['SYN']['NMDA_gain'], 'stp_flag': 1.,
                'U': params['STM']['SYN']['NMDA_synapse_params']['U']},  # until [Reference5], i did dont set or reset
        },  # backward
    'L4e_to_L23e_params': {
        'STM': {'weight': params['STM']['SYN']['L4e_to_L23e_weight']},
        },
    'L4e_to_L23e_params': {
        'STM': {'weight': params['STM']['SYN']['L4e_to_L23e_weight']},
        },
    'i2e_synapse_params': {
        'STM': {'weight': params['STM']['SYN']['i2e_weight']},  # activate WTA
        },  # activate WTA
    'L23e_zmn_rate': {
        'STM': params['STM']['STIM']['L23e_zmn_rate'],
        },
    'L4e_zmn_rate': {
        'STM': params['STM']['STIM']['L4e_zmn_rate'],
        }
}