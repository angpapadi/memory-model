import nest
import os
import numpy as np, pylab as plt
import collections

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d

def PreloadMeans(params, model, net, load_MCs=True, load_WTA=True, load_attractors=True, load_globalinhibition=False, MC_gain=1, WTA_gain=1, gi_gain=1):  # add gain factors to control this



    print('Preloading %s', net)
    print('...from Means')
    PreloadMeans = {'Intra_HC': {'Intra_MC': {'AMPA': [0.10, 0.09, 0.26], 'NMDA': [0.10, 0.09, 0.050]},
                                 'Inter_MC': {'AMPA': [0.10, 0.10, 8.10e-05], 'NMDA': [0.10, 0.10, 0.006]}},
                    'Inter_HC': {'Intra_MC': {'AMPA': [0.10, 0.06, 0.13], 'NMDA': [0.10, 0.06, 0.036]},
                                 'Inter_MC': {'AMPA': [0.10, 0.10, 7.72e-05 * gi_gain],
                                              'NMDA': [0.10, 0.10, 0.006]}}}  # means extracted from LTMaSample

    # loading of biases to be integrated with connection initialization
    print('...loading biases')
    # nrn_dict={'p_j':0.1,'bias':np.log(0.1)} #means extracted from LTMaSample
    # nest.SetStatus(model[net]['L23e_pop'],nrn_dict)
    # -----------------------------------------------------------------------------------------------------

    print('...categorizing and collecting connections to be modified')
    AllMCs = range(params[net]['SIZE']['n_MC'])
    conn_MC = {'AMPA': [], 'NMDA': []}
    conn_WTA = {'AMPA': [], 'NMDA': []}
    conn_attractor = {'AMPA': [], 'NMDA': []}
    for mc_pre in AllMCs:
        for mc_post in AllMCs:
            if params[net]['SIZE']['HC_MC'][mc_pre] == params[net]['SIZE']['HC_MC'][mc_post]:
                HCkey = 'Intra_HC'
            else:
                HCkey = 'Inter_HC'
            if mc_pre == mc_post:
                MCkey = 'Intra_MC'  # within exact same MC
            elif ((mc_pre % params[net]['SIZE']['n_MC_per_HC'] == mc_post % params[net]['SIZE']['n_MC_per_HC']) & (
                    mc_pre != mc_post)):  # same RESPECTIVE MC across HCs
                MCkey = 'Intra_MC'
            else:
                MCkey = 'Inter_MC'

                # now use this information to collect on connections
            if (HCkey == 'Intra_HC') & (MCkey == 'Intra_MC') & (load_MCs):
                for receptor in params[net]['SYN']['receptors']:
                    conn_MC[receptor] += nest.GetConnections(source=model[net]['L23e_pop_MC'][mc_pre],
                                                             target=model[net]['L23e_pop_MC'][mc_post],
                                                             synapse_model=receptor + '_synapse_' + net)
            if (HCkey == 'Intra_HC') & (MCkey == 'Inter_MC') & (load_WTA):
                for receptor in params[net]['SYN']['receptors']:
                    conn_WTA[receptor] += nest.GetConnections(source=model[net]['L23e_pop_MC'][mc_pre],
                                                              target=model[net]['L23e_pop_MC'][mc_post],
                                                              synapse_model=receptor + '_synapse_' + net)
            if (HCkey == 'Inter_HC') & (MCkey == 'Intra_MC') & (load_attractors):
                for receptor in params[net]['SYN']['receptors']:
                    conn_attractor[receptor] += nest.GetConnections(source=model[net]['L23e_pop_MC'][mc_pre],
                                                                    target=model[net]['L23e_pop_MC'][mc_post],
                                                                    synapse_model=receptor + '_synapse_' + net)

                    # now act on collected connections accordingly
    """
    if load_globalinhibition:
        logging.info('%s set global inhibition', net)
        for receptor in params[net]['SYN']['receptors']:
            gi_dict={'p_i':PreloadMeans['Inter_HC']['Inter_MC'][receptor][0],
                     'p_j':PreloadMeans['Inter_HC']['Inter_MC'][receptor][1],
                     'p_ij':PreloadMeans['Inter_HC']['Inter_MC'][receptor][2]} #only one dict (saves memory)
            #if receptor=='AMPA':
            #    gi_dict['p_ij']*=gi_gain         
            nest.SetStatus(model[net][receptor+'_connections'],gi_dict)#update all connections with this dictionary
            logging.info('MC %s receptors load %s', receptor, gi_dict)
        logging.info('%s global inhibition completed', net)
    if load_MCs:
        logging.info('%s MC loading started', net)
        for receptor in params[net]['SYN']['receptors']:
            MC_dict={'p_i':PreloadMeans['Intra_HC']['Intra_MC'][receptor][0],
                     'p_j':PreloadMeans['Intra_HC']['Intra_MC'][receptor][1],
                     'p_ij':PreloadMeans['Intra_HC']['Intra_MC'][receptor][2]*MC_gain} 
            nest.SetStatus(conn_MC[receptor],MC_dict)#update MC recurrent connections with this dictionaries
            logging.info('MC %s receptors load %s', receptor, MC_dict)
        logging.info('%s MC loading completed', net)
    if load_WTA:
        logging.info('%s WTA loading started', net)
        for receptor in params[net]['SYN']['receptors']:
            WTA_dict={'p_i':PreloadMeans['Intra_HC']['Inter_MC'][receptor][0],
                      'p_j':PreloadMeans['Intra_HC']['Inter_MC'][receptor][1],
                      'p_ij':PreloadMeans['Intra_HC']['Inter_MC'][receptor][2]/WTA_gain} 
            nest.SetStatus(conn_WTA[receptor],WTA_dict)#update MCs WTA connections with this dictionaries
            logging.info('WTA %s receptors load %s', receptor, WTA_dict)
        logging.info('%s WTA loading completed', net)
    if load_attractors:
        logging.info('%s attractor loading started', net)
        for receptor in params[net]['SYN']['receptors']:
            attractor_dict={'p_i':PreloadMeans['Inter_HC']['Intra_MC'][receptor][0],
                            'p_j':PreloadMeans['Inter_HC']['Intra_MC'][receptor][1],
                            'p_ij':PreloadMeans['Inter_HC']['Intra_MC'][receptor][2]}
            logging.info('attractor %s receptors load %s', receptor, attractor_dict)
            nest.SetStatus(conn_attractor[receptor],attractor_dict)
        logging.info('%s attractor loading completed', net)
    logging.info('%s all preload completed \n', net)
    """
    return

def stim_matrix_generator(n_new_patterns, params, net, pattern_type='orthogonal'):

    S = params[net]['SIZE']
    stim_matrix = [[0 for mc in range(params[net]['SIZE']['n_MC'])] for n in range(n_new_patterns)]

    if pattern_type == 'orthogonal':
        if n_new_patterns <= params[net]['SIZE']['n_MC_per_HC']:
            for idx in range(n_new_patterns):
                MC0_HC = range(0, params[net]['SIZE']['n_MC'], params[net]['SIZE']['n_MC_per_HC'])
                for hc in range(params[net]['SIZE']['n_HC']):
                    stim_matrix[idx][MC0_HC[hc] + idx] = 1
        else:
            print('not possible to create this many orthogonal patterns')
    elif pattern_type == 'random':
        for idx in range(n_new_patterns):
            MC0_HC = range(0, params[net]['SIZE']['n_MC'], params[net]['SIZE']['n_MC_per_HC'])
            randomMC_HC = list(np.random.randint(0, params[net]['SIZE']['n_MC_per_HC'], params[net]['SIZE']['n_HC']))
            for hc in range(params[net]['SIZE']['n_HC']):
                stim_matrix[idx][MC0_HC[hc] + randomMC_HC[hc]] = 1
    else:
        print('pattern type not specified correctly')
    return stim_matrix

def GetPatternNodes(pattern,model,net,returntype):
    if returntype=='pop':
        MC_population=model[net]['L23e_pop_MC']
    elif returntype=='rec':
        MC_population=model[net]['L23e_rec_MC']
    else:
        print('Error. GetPatternNodes expects a valid returntype')
    cells=[]
    for mc in range(len(pattern)):#Parsing the pattern
            if pattern[mc]==1: #small change from !=0
                cells.extend(MC_population[mc])
    return cells

def add_stim_schedule(Phase, offset, model):
    # offset is the timepoint at which the provided stim_matrix is inserted into NEST
    # added_stim_schedule is stored into the Phase and also applied to NEST
    # the passed Phase contains the stim-matrix and other relevant stim parameters
    new_events = []  # to be appended to the main stim_schdule (a list of event dictionaries) in the model dict AFTER full processing
    for net in ['STM', 'LTMa', 'LTMb']:
        for time_idx in range(len(Phase['stim_matrix'][net])):  # Parsing the stim_matrix
            pattern = Phase['stim_matrix'][net][time_idx]
            pattern_no = Phase['trained_patterns'][net].index(pattern)
            new_events.append({
                'rate': Phase['stim_rate'][net],
                # *stim_matrix[time_idx][mc], could be used to generate graded stimulation
                'pattern': pattern_no,
                'target': GetPatternNodes(pattern, model, net, 'pop'),
                'startms': offset + Phase['stim_gap'][net] + time_idx * (
                            Phase['stim_gap'][net] + Phase['stim_length'][net]),
                'stopms': offset + (time_idx + 1) * (Phase['stim_gap'][net] + Phase['stim_length'][net]),
                'weight': Phase['stim_weight'][net],  # stim node connection strength
                'delay': Phase['stim_delay'][net],  # stim node connection delay
                'synapse': 'stim_synapse_' + net  # stim connector model
            })
            # also update the pattern training times tracker with this new event:
            if pattern_no < len(Phase['pattern_training_log'][net]):
                # pattern that has an index in the pattern_training_log list
                Phase['pattern_training_log'][net][pattern_no].append(
                    (new_events[-1]['startms'], new_events[-1]['stopms']))
            else:  # a'new' pattern'
                Phase['pattern_training_log'][net].append([(new_events[-1]['startms'], new_events[-1]['stopms'])])
    Phase['added_stim_schedule'] = new_events  # finally add all new events to the schedule

    # Connect Stimulation according to the new events
    for stim_event in new_events:
        if stim_event['rate'] > 0:
            new_stim_node = nest.Create('poisson_generator', \
                                        params={'rate': stim_event['rate'], \
                                                'start': stim_event['startms'], \
                                                'stop': stim_event['stopms']})
            syn_dict = {'model': stim_event['synapse'], 'weight': stim_event['weight'], 'delay': stim_event['delay']}
            conn_dict = {'rule': 'all_to_all', 'autapses': False, 'multapses': True}  # Nest 2.4+
            nest.Connect(new_stim_node, stim_event['target'], conn_dict, syn_dict)  # Nest 2.4+
            # nest.DivergentConnect(new_stim_node,stim_event['target'], model=syn_dict['model'],weight=syn_dict['weight'],delay=syn_dict['delay'])  #adjusted for nest 2.2.2
    return  # the Phase was passed as a reference and is thus already updated with 'added_stim_schedule'

def add_cue_stimulation(Phase,offset,model):
    #Connect Stimulation according to the cue activation provided
    for net in ['STM']:
        pattern=Phase['stim_matrix'][net]#is a vector, not a matrix
        new_stim_node=nest.Create('poisson_generator',\
        params={'rate' : Phase['stim_rate'][net],\
                'start': offset,\
                'stop' : offset+Phase['cuetime'][net]})
        syn_dict = {'model': 'stim_synapse_'+net,'weight': Phase['stim_weight'][net], 'delay': Phase['stim_delay'][net]}
        conn_dict = {'rule': 'all_to_all',   'autapses': False,  'multapses': True}# Nest 2.4+
        nest.Connect(new_stim_node,GetPatternNodes(pattern,model,net,'pop'), conn_dict, syn_dict) #Nest 2.4+
        #nest.DivergentConnect(new_stim_node,GetPatternNodes(pattern,model,net,'pop'), model=syn_dict['model'],weight=syn_dict['weight'],delay=syn_dict['delay'])  #adjusted for nest 2.2.2

    return

def RunPhase(Phase, model, program=[], OptDict={}):

    ThisPhase = {}
    ThisPhase = update(Phase, OptDict)  # possible updates to the Phase

    while ThisPhase['name'] in [p['name'] for p in program]:  # making sure all Phasenames are unique
        ThisPhase['name'] += 'I'
    print('Current phase name is:', ThisPhase['name'])

    print('Preparing Phase' , ThisPhase['name'],  'of length ', ThisPhase['length'])

    for net in ['STM']:
        ThisPhase = update(ThisPhase, {'trained_patterns': {net: []}, 'pattern_training_log': {net: []}})
        if len(program) > 0:  # keep track of PREVIOUSLY learned patterns
            ThisPhase['trained_patterns'][net] = program[-1]['trained_patterns'][net]
            ThisPhase['pattern_training_log'][net] = program[-1]['pattern_training_log'][net]

        nest.SetStatus(model[net]['L23e_pop'], ThisPhase['L23e_cell_params'][net])
        nest.SetStatus(model[net]['AMPA_connections'], ThisPhase['AMPA_params'][net])
        nest.SetStatus(model[net]['NMDA_connections'], ThisPhase['NMDA_params'][net])
        nest.SetStatus(model[net]['i2e_connections'], ThisPhase['i2e_synapse_params'][net])
        # nest.SetStatus(model[net]['AMPA_connections'],{'t_k': nest.GetKernelStatus('time')})
        # nest.SetStatus(model[net]['NMDA_connections'],{'t_k': nest.GetKernelStatus('time')})
        #nest.SetStatus(model[net]['zmn_nodes_L23e'], {'rate': ThisPhase['L23e_zmn_rate'][net]})
        #nest.SetStatus(model[net]['zmn_nodes_L23i'], {'rate': ThisPhase['L23e_zmn_rate'][net]})
        nest.SetStatus(model[net]['zmn_nodes_L4e'], {'rate': ThisPhase['L4e_zmn_rate'][net]})
        nest.SetStatus(model[net]['zmn_nodes_L4i'], {'rate': ThisPhase['L4e_zmn_rate'][net]})
        nest.SetStatus(model[net]['L4e_to_L23e_connections'], ThisPhase['L4e_to_L23e_params'][net])

        for hypercolumn in range(len(model[net]['L23e_pop_HC'])):
            nest.SetStatus(model[net]['zmn_nodes_L23e'][hypercolumn], {'rate': ThisPhase['L23e_zmn_rate'][net][hypercolumn]})
            nest.SetStatus(model[net]['zmn_nodes_L23i'][hypercolumn], {'rate': ThisPhase['L23e_zmn_rate'][net][hypercolumn]})



    if ThisPhase['type'] == 'stim':
        for net in ['STM']:
            for pattern in ThisPhase['stim_matrix'][net]:  # add the new patterns to the list of trained patterns
                if pattern not in ThisPhase['trained_patterns'][net]:
                    ThisPhase['trained_patterns'][net].append(pattern)
        add_stim_schedule(ThisPhase, nest.GetKernelStatus('time'), model)
        ThisPhase['length'] = np.max(
            [len(ThisPhase['stim_matrix'][net]) * (ThisPhase['stim_gap'][net] + ThisPhase['stim_length'][net]) for net
             in ['STM', 'LTMa', 'LTMb']])  # the Phase length is the duration of the longest net stimulation protocol

    if ThisPhase['type'] == 'cue':
        add_cue_stimulation(ThisPhase, nest.GetKernelStatus('time'), model)
        ThisPhase['length'] = np.max([ThisPhase['cuetime'][net] for net in ['STM']])  # cues are short and simple (no gaps, and just one pattern)

    ThisPhase['start'] = nest.GetKernelStatus('time')
    print('Executing Phase' , ThisPhase['name'],  'of length ', ThisPhase['length'])
    nest.Simulate(ThisPhase['length'])  # <-------------------- Actual Simulation command!
    ThisPhase['stop'] = nest.GetKernelStatus('time')
    print(' ...done. \n')

    # plot spikes per population for debugging
    all_populations = ['L23e_rec_spikes','L2e_rec_spikes', 'L3ae_rec_spikes', 'L3be_rec_spikes', 'L4e_rec_spikes', 'L2i_rec_spikes']
    spikedetective_from_population = all_populations[0]
    dSD = nest.GetStatus(model['STM'][spikedetective_from_population], keys="events")[0]
    evs = dSD["senders"]
    ts = dSD["times"]
    print('In phase:', ThisPhase['name'], len(ts), ' spikes were recorded from population ',
          spikedetective_from_population[:4])
    print()
    plt.figure(ThisPhase['name'])
    plt.plot(ts, evs, ".")
    # save figure to file
    plt.savefig("plots/"+ThisPhase['name']+'_'+ spikedetective_from_population[:3] +'spikeraster'+'.png')

    # ADDITION of plot of spikes per MC
    which_MC = 3
    dSD = nest.GetStatus(model['STM']['spike_recorder_per_MC'][which_MC], keys="events")[0]
    evs = dSD["senders"]
    ts = dSD["times"]
    plt.figure('Spikes per MC')
    plt.plot(ts, evs, ".")
    plt.savefig("plots/"+ThisPhase['name']+'_from'+ str(which_MC) +'spikeraster'+'.png')

    # plot membrane potential per MC
    plotting_minicolumn = 4
    dmm = nest.GetStatus(model['STM']['multimeter_MC'][plotting_minicolumn])[0]
    Vms = dmm["events"]["V_m"]
    ts = dmm["events"]["times"]
    plt.figure('Membrane potential')
    plt.plot(ts, Vms)
    plt.savefig("plots/"+ThisPhase['name']+'_from'+ str(plotting_minicolumn) +'membrane potential'+'.png')
    # """


    plt.show()    

    program.append(ThisPhase)  # Keeps an ordered list of the phases being run
    return program
