import numpy as np
import nest
import nest.topology as tp
import pylab as pltPYC
import timeit, sys, copy, itertools, random, collections
import parameters, helpers
from mpi4py import MPI

def BuildNet(params):

    syn_ports = parameters.syn_ports 
    modules = parameters.net_modules

    for net in modules:
        NRN = params[net]['NRN']
        SYN = params[net]['SYN']
        STIM = params[net]['STIM']
        REC = params[net]['REC']
        S = params[net]['SIZE']

        model = {'STM': {}}
        nest.SetKernelStatus({'resolution': params['other']['dt']})

        # define populations and synapses
        L23e_cell_params = NRN['neuron_params']
        nest.CopyModel(NRN['cell_model'], 'L23e_cell_' + net, L23e_cell_params)

        L4e_cell_params = NRN['neuron_params']
        L4e_cell_params.update(b=0.0)  # L4e_cell_params.update(b = 0.0, gain= 0.0) #L4 cells have no neural plasticity. #hopefully this avoid the numerical instability that has been plaguing FFutilsv36
        nest.CopyModel(NRN['cell_model'], 'L4e_cell_' + net, L4e_cell_params)

        i_cell_params = NRN['neuron_params']
        i_cell_params.update(b=0.0)  # i_cell_params.update(b = 0.0, gain= 0.0) #i cells have no neural plasticity.

        nest.CopyModel(NRN['cell_model'], 'i_cell_' + net, i_cell_params)

        nest.CopyModel('static_synapse', 'stim_synapse_' + net,
                   {'weight': STIM['stim_weight'], 'delay': STIM['stim_delay'], 'receptor_type': syn_ports['AMPA']})
        nest.CopyModel('static_synapse', 'zmn_synapse_' + net,
                   {'weight': STIM['zmn_weight'], 'delay': STIM['zmn_delay'], 'receptor_type': syn_ports['AMPA']})
        nest.CopyModel('static_synapse', 'e2i_synapse_' + net,
                   {'weight': SYN['e2i_weight'], 'delay': SYN['delay_eie'], 'receptor_type': syn_ports['AMPA']})
        nest.CopyModel('static_synapse', 'L4e_to_L23e_synapse_' + net,
                   {'weight': SYN['L4e_to_L23e_weight'], 'delay': SYN['delay_IntraHC'],
                    'receptor_type': syn_ports['AMPA']})
        nest.CopyModel('static_synapse', 'i2e_synapse_' + net,
                   {'weight': SYN['i2e_weight'], 'delay': SYN['delay_eie'], 'receptor_type': syn_ports['GABA']})
        nest.CopyModel(SYN['synapse_model'], 'AMPA_synapse_' + net, SYN['AMPA_synapse_params'])
        nest.CopyModel(SYN['synapse_model'], 'NMDA_synapse_' + net, SYN['NMDA_synapse_params'])

        # layer topology
        S['L23e_per_MC'] = S['L2e_per_MC'] + S['L3ae_per_MC'] + S[
            'L3be_per_MC']  # just to guarantee consistency between L23e and its constituents
        L23e_layer = tp.CreateLayer({'rows': S['HC_rows'],
                                     'columns': S['HC_columns'],
                                     'elements': ['L23e_cell_' + net, S['L23e_per_MC'] * S['n_MC_per_HC']],
                                     # elements now encompass all MCs of each HC
                                     'extent': S['grid_extent'],
                                     'center': S['grid_center'],
                                     'edge_wrap': False})

        L4e_layer = tp.CreateLayer({'rows': S['HC_rows'],
                                    'columns': S['HC_columns'],
                                    'elements': ['L4e_cell_' + net, S['L4e_per_MC'] * S['n_MC_per_HC']],
                                    'extent': S['grid_extent'],
                                    'center': S['grid_center'],
                                    'edge_wrap': False})

        L2i_layer = tp.CreateLayer({'rows': S['HC_rows'], 'columns': S['HC_columns'],
                                    'elements': ['i_cell_' + net, S['L2i_per_MC'] * S['n_MC_per_HC']],
                                    'extent': S['grid_extent'],
                                    'center': S['grid_center'],
                                    'edge_wrap': False})

        # node addresses ???
        # Collect gid adresses
        L23e_pop = nest.GetNodes(L23e_layer)[0]
        L2i_pop = nest.GetNodes(L2i_layer)[0]
        L4e_pop = nest.GetNodes(L4e_layer)[0]

        # Making nodes adressable by MC index and register Grid coordinates for MC and HC
        L23e_pop_HC = []  # now assemble the right minicolumns into each hypercolumn
        L4e_pop_HC = []
        GridPosition_HC = []
        for hc in range(S['n_HC']):
            xgrid = int(hc / S['HC_rows'])
            ygrid = int(hc % S['HC_rows'])
            L23e_pop_HC.append(tp.GetElement(L23e_layer, [xgrid, ygrid]))
            if S['L4e_per_MC'] > 0:
                L4e_pop_HC.append(
                    tp.GetElement(L4e_layer, [xgrid, ygrid]))  # this would crash the kernel if L4 not used
            else:
                L4e_pop_HC.append(())
            GridPosition_HC.append([xgrid, ygrid])
        # Making nodes adressable by MC index
        L23e_pop_MC = []
        L4e_pop_MC = []

        for hc in range(S['n_HC']):
            for i in range(0, len(L23e_pop_HC[hc]), S['L23e_per_MC']):
                L23e_pop_MC.append(L23e_pop_HC[hc][i:i + S['L23e_per_MC']])

        if S['L4e_per_MC'] > 0:
            for hc in range(S['n_HC']):
                for i in range(0, len(L4e_pop_HC[hc]), S['L4e_per_MC']):
                    L4e_pop_MC.append(L4e_pop_HC[hc][i:i + S['L4e_per_MC']])
        else:
            L4e_pop_MC = [[] for mc in range(S['n_MC'])]

        L2i_pop_HC = []  # assemble L2i cells
        for hc in range(S['n_HC']):
            L2i_pop_HC.append(tp.GetElement(L2i_layer, [hc // S['HC_rows'], hc % S['HC_rows']]))

        # Making nodes adressable by MC index (this is a pro-forma attribution of L2i cells towards a local MC, as they are really distributed around the center of each HC, and thus do not belong)
        L2i_pop_MC = []
        for MC in range(S['n_MC']):
            HC = S['HC_MC'][MC]
            index = MC % S['n_MC_per_HC'] * S['L2i_per_MC']
            L2i_pop_MC.append(L2i_pop_HC[HC][index:index + S['L2i_per_MC']])

        # subdividing L23e into its constituent layers. Also calculating its layers population
        # collapsio
        """
        L2e_pop = []
        L3ae_pop = []
        L3be_pop = []
        L2e_pop_MC = []
        L3ae_pop_MC = []
        L3be_pop_MC = []
        for MC in range(S['n_MC']):
            L2e_pop_MC.append(L23e_pop_MC[MC][:S['L2e_per_MC']])  # first 20 normally
            L3ae_pop_MC.append(L23e_pop_MC[MC][S['L2e_per_MC']:S['L2e_per_MC'] + S['L3ae_per_MC']])  # next 5
            L3be_pop_MC.append(L23e_pop_MC[MC][-S['L3be_per_MC']:])  # last 5
            L2e_pop.extend(L2e_pop_MC[MC])
            L3ae_pop.extend(L3ae_pop_MC[MC])
            L3be_pop.extend(L3be_pop_MC[MC])
        """

        # Defining recorded Populations
        L23e_rec = []
        L4e_rec = []
        L2i_rec = []
        for mc in REC['spike_rec_MCs']:  # go through recorded MCs
            for nrn in range(REC['L23e_rec_per_MC']):
                L23e_rec.append(L23e_pop_MC[mc][nrn])
            for nrn in range(REC['L4e_rec_per_MC']):
                L4e_rec.append(L4e_pop_MC[mc][nrn])
            for nrn in range(REC['L2i_rec_per_MC']):
                L2i_rec.append(L2i_pop_MC[mc][nrn])

                # Defining recorded Populations and MC_wise populations
                #L2e_rec = []
                #L3ae_rec = []
                #L3be_rec = []
                #L2e_rec_MC = []
                #L3ae_rec_MC = []
                #L3be_rec_MC = []
                L23e_rec_MC = []
                L4e_rec_MC = []
                L2i_rec_MC = []
                for mc in range(S['n_MC']):
                    L23e_rec_MC.append([nrn for nrn in L23e_pop_MC[mc] if nrn in L23e_rec])
                    L4e_rec_MC.append([nrn for nrn in L4e_pop_MC[mc] if nrn in L4e_rec])
                    L2i_rec_MC.append([nrn for nrn in L2i_pop_MC[mc] if nrn in L2i_rec])

                    # collapsio
                    """
                    L2e_rec_MC.append(L23e_rec_MC[mc][:REC['L2e_rec_per_MC']])
                    L3ae_rec_MC.append(L23e_rec_MC[mc][REC['L2e_rec_per_MC']:REC['L2e_rec_per_MC'] + REC['L3ae_rec_per_MC']])
                    L3be_rec_MC.append(L23e_rec_MC[mc][-REC['L3be_rec_per_MC']:])
                    L2e_rec.extend(L2e_rec_MC[mc])
                    L3ae_rec.extend(L3ae_rec_MC[mc])
                    L3be_rec.extend(L3be_rec_MC[mc])
                    """

                # after operation collapsio, model update no longer inlcudes subpopulations 2e,3a,3b. refer to previous code versions if this needs to be brought back
                model[net].update(L23e_layer=L23e_layer, L4e_layer=L4e_layer, L2i_layer=L2i_layer)  # layer
                model[net].update(L23e_pop=L23e_pop, L4e_pop=L4e_pop, L2i_pop=L2i_pop)  # _pop
                model[net].update(L23e_pop_HC=L23e_pop_HC, L4e_pop_HC=L4e_pop_HC, L2i_pop_HC=L2i_pop_HC)  # _pop_HC

                ##
                model[net].update(L23e_pop_MC=L23e_pop_MC, L4e_pop_MC=L4e_pop_MC, L2i_pop_MC=L2i_pop_MC)  # _pop_MC
                model[net].update(L23e_rec=L23e_rec, L4e_rec=L4e_rec, L2i_rec=L2i_rec)  # _rec
                model[net].update(L23e_rec_MC=L23e_rec_MC, L4e_rec_MC=L4e_rec_MC, L2i_rec_MC=L2i_rec_MC)  # _rec_MC

        # Invert the *_rec_MC mappings as a direct dictionary access to any neurons MC number
        MC_nrn = {}
        populations = ['L23e', 'L4e', 'L2i']
        for population in populations:
            MC_nrn.update(dict.fromkeys(model['STM'][population + '_rec']))  # extend keys with this recorded poulation
            for mc in range(len(model[net][population + '_rec_MC'])):
                for nrn in model[net][population + '_pop_MC'][mc]:
                    MC_nrn[nrn] = mc  # add this neurons mc
        # build a translation table of recorded node_ids to sorted node_ids for plotting
        sorted_nodes = []
        for mc in REC['spike_rec_MCs']:
            for nrn in range(REC['L2i_rec_per_MC']):
                sorted_nodes.append(L2i_pop_MC[mc][nrn])
            for nrn in range(REC['L4e_rec_per_MC']):
                sorted_nodes.append(L4e_rec_MC[mc][nrn])
            for nrn in range(REC['L23e_rec_per_MC']):
                sorted_nodes.append(L23e_rec_MC[mc][nrn])
        # invert the mapping and shift the list to get a plotting_idx for each neuron
        plotting_idx = np.zeros(max(sorted_nodes) + 1, np.int)
        for nrn in sorted_nodes:
            plotting_idx[nrn] = sorted_nodes.index(nrn) - len(L2i_rec)
            # plotting_idx=plotting_idx.tolist() #because json cannot write arrays
        # Randomize intial value of V_m to shorten the initialization transient
        for i in L23e_pop:
            nest.SetStatus([i], {'V_m': L23e_cell_params['E_L'] + .95 * np.random.random() * (
                        L23e_cell_params['V_th'] - L23e_cell_params['E_L'])})
        for i in L4e_pop:
            nest.SetStatus([i], {'V_m': L4e_cell_params['E_L'] + .95 * np.random.random() * (
                        L4e_cell_params['V_th'] - L4e_cell_params['E_L'])})
        for i in L2i_pop:
            nest.SetStatus([i], {'V_m': i_cell_params['E_L'] + .95 * np.random.random() * (
                        i_cell_params['V_th'] - i_cell_params['E_L'])})
        print('All regular nodes completed for network %s', net)
        model[net].update(GridPosition_HC=GridPosition_HC, MC_nrn=MC_nrn, sorted_nodes=sorted_nodes,
                          plotting_idx=plotting_idx)

        # draw connections
        grid_l_max = np.sqrt((S['grid_extent'][0] - 1) ** 2 + (S['grid_extent'][
                                                                   1] - 1) ** 2) + 0.0001  # maximum grid distance. Before DelayReformA, the grid was wrapped around. #+0.0001 avoids division by 0
        # I_to_L23e (I = inhibitory basket cells)
        conn_dict = {'connection_type': 'convergent',
                     'number_of_connections': int(np.round(SYN['prob_i2e'] * len(L2i_pop_HC[0]))),
                     'synapse_model': 'i2e_synapse_' + net, 'allow_autapses': False, 'allow_multapses': False,
                     'mask': {'rectangular': {'lower_left': [-1 / 2., -1 / 2.],
                                              # no longer over an MC grid, so this is just local (half a grid unit in each direction)
                                              'upper_right': [+1 / 2., +1 / 2.]}
                              }
                     }
        # print(nest.GetDefaults('i2e_synapse_'+net)) # just to check that the synaptic weight is indeed negative
        tp.ConnectLayers(L2i_layer, L23e_layer, conn_dict)
        i2e_connections = [list(conn) for conn in
                           list(nest.GetConnections(model[net]['L2i_pop'], target=model[net]['L23e_pop']))]
        # L23e_to_I
        conn_dict = {'connection_type': 'convergent',
                     'number_of_connections': int(np.round(SYN['prob_e2i'] * len(L23e_pop_HC[0]))),
                     'synapse_model': 'e2i_synapse_' + net, 'allow_autapses': False, 'allow_multapses': False,
                     'mask': {'rectangular': {'lower_left': [-1 / 2., -1 / 2.],
                                              'upper_right': [+1 / 2., +1 / 2.]}
                              }
                     }
        tp.ConnectLayers(L23e_layer, L2i_layer, conn_dict)
        e2i_connections = [list(conn) for conn in
                           list(nest.GetConnections(model[net]['L23e_pop'], target=model[net]['L2i_pop']))]
        # L23e_to_L23e
        # Intra HC connections (radial mask)
        conn_dict = {'connection_type': 'convergent',
                     'number_of_connections': int(np.round(SYN['prob_L23e_to_L23e_intraHC'] * len(L23e_pop_HC[0]))),
                     'mask': {'circular': {'radius': 0.5}},  # will include only the local HC?
                     'synapse_model': 'AMPA_synapse_' + net, 'allow_autapses': False, 'allow_multapses': False,
                     'weights': SYN['AMPA_synapse_params']['weight'],
                     'delays': {'linear': {'c': SYN['delay_IntraHC'],
                                           'a': (SYN['delay_InterHC'] - SYN['delay_IntraHC']) / grid_l_max}
                                }
                     }

        tp.ConnectLayers(L23e_layer, L23e_layer, conn_dict)
        # Inter HC connections (Doughnut mask)
        n_connections = int(np.round(SYN['prob_L23e_to_L23e_interHC'] * (len(L23e_pop) - len(L23e_pop_HC[0]))))
        conn_dict = {'connection_type': 'divergent', 'number_of_connections': n_connections,
                     'mask': {'doughnut': {'inner_radius': 0.5, 'outer_radius': 1000.}},
                     # will include only the local HC?
                     'synapse_model': 'AMPA_synapse_' + net, 'allow_autapses': False, 'allow_multapses': True,
                     'weights': SYN['AMPA_synapse_params']['weight'],
                     'delays': {'linear': {'c': SYN['delay_IntraHC'],
                                           'a': (SYN['delay_InterHC'] - SYN['delay_IntraHC']) / grid_l_max}
                                }
                     }
        tp.ConnectLayers(L23e_layer, L23e_layer, conn_dict)
        AMPA_connections = [list(conn) for conn in list(
            nest.GetConnections(model[net]['L23e_pop'], target=model[net]['L23e_pop'],
                                synapse_model='AMPA_synapse_' + net))]

        # L4e_to_L23e, static,local synapse
        conn_number = int(np.round(SYN['prob_L4e_to_L23e'] * len(L23e_pop_MC[0])))
        option_dict = {'allow_autapses': False,
                       'allow_multapses': False}  # obsolete connection dictionary for old connection routines

        condict = {'rule': 'fixed_outdegree', 'outdegree': conn_number, 'autapses': False,
                   'multapses': False}  # new option dict for nest 2.18
        syndict = {'model': 'L4e_to_L23e_synapse_' + net, 'weight': SYN['L4e_to_L23e_weight'],
                   'delay': SYN['delay_IntraHC']}  # new syn dict for nest 2.18
        for mc in range(S['n_MC']):
            # nest.RandomDivergentConnect(L4e_pop_MC[mc],L23e_pop_MC[mc],conn_number,SYN['L4e_to_L23e_weight'],SYN['delay_IntraHC'],'L4e_to_L23e_synapse_'+net,option_dict)
            nest.Connect(L4e_pop_MC[mc], L23e_pop_MC[mc], condict, syndict)
        if S['L4e_per_MC'] > 0:
            L4e_to_L23e_connections = [list(conn) for conn in
                                       list(nest.GetConnections(model[net]['L4e_pop'], target=model[net]['L23e_pop']))]
        else:
            L4e_to_L23e_connections = []

        model[net].update(AMPA_connections=AMPA_connections, i2e_connections=i2e_connections,
                          e2i_connections=e2i_connections,
                          L4e_to_L23e_connections=L4e_to_L23e_connections)  # now lists of list and thus JSON compatible
        print('%s Connections Listed', net)

        # Dither the Connections 15% from their distance derived value
        if parameters.delay_dither_relative_sd > 0.:
            for conn_type in ['AMPA_connections', 'i2e_connections', 'e2i_connections']:
                delays = nest.GetStatus(model[net][conn_type], 'delay')
                dithered_delays = [
                    {'delay': max(1.5, round(np.random.normal(delays[i], delays[i] * parameters.delay_dither_relative_sd), 1))}
                    for i in range(len(model[net][conn_type]))]
                nest.SetStatus(model[net][conn_type], dithered_delays)
            print('%s, Connection Delays Dithered', net)

        # Add NMDA to each existing AMPA connection
        print('%s adding NMDA...', net)
        AMPA_status = nest.GetStatus(AMPA_connections)
        for status in AMPA_status:
            if ('2.2.2' in nest.version()):  # old 2.2.2 syntax aplies
                nest.Connect([status['source']], [status['target']], [status['weight']], [status['delay']],
                             'NMDA_synapse_' + net)
            else:  # NEST 2.4 syntax
                my_syn_dict = {'model': 'NMDA_synapse_' + net, 'weight': status['weight'],
                               'delay': status['delay']}  # not needed in old 2.2.2 connect function
                nest.Connect([status['source']], [status['target']], syn_spec=my_syn_dict)

        NMDA_connections = [list(conn) for conn in list(
            nest.GetConnections(model[net]['L23e_pop'], target=model[net]['L23e_pop'],
                                synapse_model='NMDA_synapse_' + net))]
        model[net].update(NMDA_connections=NMDA_connections)
        print('%s...NMDA completed', net)

        syn_dict_e = {'model': 'zmn_synapse_' + net, 'weight': STIM['zmn_weight'], 'delay': STIM['zmn_delay'],
                      'receptor_type': syn_ports['AMPA']}
        syn_dict_i = {'model': 'zmn_synapse_' + net, 'weight': STIM['zmn_weight'], 'delay': STIM['zmn_delay'],
                      'receptor_type': syn_ports['GABA']}

        conn_dict = {'rule': 'all_to_all', 'autapses': False, 'multapses': True}

        # zmn noise for L23         # one zmn to rule them all. noise targets all l23e population/ is not specific to each HC
        #zmn_nodes_L23e = nest.Create('poisson_generator', params={'rate': STIM['L23e_zmn_rate']})
        #zmn_nodes_L23i = nest.Create('poisson_generator', params={'rate': STIM['L23e_zmn_rate']})
        #nest.Connect(zmn_nodes_L23e, L23e_pop, conn_dict, syn_dict_e)
        #nest.Connect(zmn_nodes_L23i, L23e_pop, conn_dict, syn_dict_i)

        # create independent noise components for each HC
        # L23e_pop_HC: each element corresponds to a HC and is a list of all neuron ids belonging to it (across all MCs)
        zmn_nodes_L23e = []
        zmn_nodes_L23i = []
        for hypercolumn in range(S['n_HC']):
            rate = STIM['L23e_zmn_rate'][hypercolumn]
            zmn_nodes_L23e.append(nest.Create('poisson_generator', params={'rate': rate}))
            zmn_nodes_L23i.append(nest.Create('poisson_generator', params={'rate': rate}))

            nest.Connect(zmn_nodes_L23e[hypercolumn], L23e_pop_HC[hypercolumn], conn_dict, syn_dict_e)
            nest.Connect(zmn_nodes_L23i[hypercolumn], L23e_pop_HC[hypercolumn], conn_dict, syn_dict_i)

        # stimulation of L4 with zmn
        zmn_nodes_L4e = nest.Create('poisson_generator', params={'rate': STIM['L4e_zmn_rate']})
        zmn_nodes_L4i = nest.Create('poisson_generator', params={'rate': STIM['L4e_zmn_rate']})
        nest.Connect(zmn_nodes_L4e, L4e_pop, conn_dict, syn_dict_e)
        nest.Connect(zmn_nodes_L4i, L4e_pop, conn_dict, syn_dict_i)

        model[net].update(zmn_nodes_L23e=zmn_nodes_L23e, zmn_nodes_L23i=zmn_nodes_L23i, zmn_nodes_L4e=zmn_nodes_L4e,
                          zmn_nodes_L4i=zmn_nodes_L4i)

        # devices

        # spike recorders per population
        L23e_rec_spikes = nest.Create('spike_detector')         # collapsio
        #L2e_rec_spikes = nest.Create('spike_detector')
        #L3ae_rec_spikes = nest.Create('spike_detector')
        #L3be_rec_spikes = nest.Create('spike_detector')
        L4e_rec_spikes = nest.Create('spike_detector')
        L2i_rec_spikes = nest.Create('spike_detector')
        syn_dict = {'model': 'static_synapse', 'weight': 1.0, 'delay': 0.1}
        conn_dict = {'rule': 'all_to_all', 'autapses': False,
                     'multapses': True}  # Nest 2.4+ NOT SURE IF ALL_TO_ALL IS CORRECT ??????

        nest.Connect(L23e_rec, L23e_rec_spikes, conn_dict, syn_dict) # collapsio
        #nest.Connect(L2e_rec, L2e_rec_spikes, conn_dict, syn_dict)
        #nest.Connect(L3ae_rec, L3ae_rec_spikes, conn_dict, syn_dict)
        #nest.Connect(L3be_rec, L3be_rec_spikes, conn_dict, syn_dict)
        nest.Connect(L4e_rec, L4e_rec_spikes, conn_dict, syn_dict)
        nest.Connect(L2i_rec, L2i_rec_spikes, conn_dict, syn_dict)

        # ADDITION of spike recorders per MC
        MC_spikes = []
        for minicol in REC['multimeter_MCs']:

            mc_sd = nest.Create('spike_detector')
            # get all neurons in particular MC across all 3 layers
            mc_neurons_alllayers = list(model[net]['L23e_pop_MC'][minicol][:REC['mm_n_per_MC']])+\
                                   list(model[net]['L4e_pop_MC'][minicol][:REC['mm_n_per_MC']])+\
                                   list(model[net]['L2i_pop_MC'][minicol][:REC['mm_n_per_MC']])
            #print(mc_neurons_alllayers)
            nest.Connect(mc_neurons_alllayers, mc_sd, conn_dict, syn_dict) # connect device with neurons
            MC_spikes.append(mc_sd)
        model[net].update(spike_recorder_per_MC = MC_spikes)

        # multimeters per MC
        if REC['multimeter_used'] == True:
            print('Multimeters active')
            multimeter_MC = dict.fromkeys(REC['multimeter_MCs'])
            multimeter_pop_MC = dict.fromkeys(REC['multimeter_MCs'])
            for MC in REC['multimeter_MCs']:
                multimeter_MC[MC] = nest.Create('multimeter', params={'interval': REC['multimeter_interval'],
                                                                      'record_from': REC['multimeterkeys'],
                                                                      'withgid': True,
                                                                      'withtime': True, 'start': 0.})
                multimeter_pop_MC[MC] = []
                multimeter_pop_MC[MC].extend(
                    model[net]['L23e_pop_MC'][MC][:REC['mm_n_per_MC']])  # pick all the neuron of the desired MCs
                # nest.DivergentConnect(multimeter_MC[MC], multimeter_pop_MC[MC])

                nest.Connect(multimeter_MC[MC], multimeter_pop_MC[MC])
            model[net].update(multimeter_MC=multimeter_MC, multimeter_pop_MC=multimeter_pop_MC)

        print('%s Recording Infrastructure completed', net)
        model[net].update(L4e_rec_spikes=L4e_rec_spikes, L2i_rec_spikes=L2i_rec_spikes,
                          L23e_rec_spikes = L23e_rec_spikes) # collapsio
        #model[net].update(L2e_rec_spikes=L2e_rec_spikes, L3ae_rec_spikes=L3ae_rec_spikes, L3be_rec_spikes=L3be_rec_spikes, L4e_rec_spikes=L4e_rec_spikes, L2i_rec_spikes=L2i_rec_spikes)

    return model

def OriginalSimulationProtocol(model,params):
    program = []

    # noise
    Phase = copy.deepcopy(params['prog']['noisephase'])
    program = helpers.RunPhase(Phase, model, program, {})

    # preload attractors
    helpers.PreloadMeans(params, model, 'STM', load_MCs=True, load_WTA=True, load_attractors=True, load_globalinhibition=True,
                 MC_gain=1, WTA_gain=1, gi_gain=1)
    # cooldown
    Phase = copy.deepcopy(params['prog']['cooldownphase'])  # recover STP ?
    program = helpers.RunPhase(Phase, model, program, {'name': 'CooldownPhase', 'length': (200.)})

    # cue - activation for every attractor
    for LTM_memory_id in range(4):  # activate four attractors. Maintain them via facilitation. Cuephase does not have any Hebbian Plasticity (kappa=0)
        Phase = copy.deepcopy(params['prog'][
                                  'cuephase'])  # not lightcuephase, because I would like to use this opportunity to get rid of it, even if its useful for stronger FF response in STM. Problem: LTM activations are only 250ms and STM is indeed too weak now. so i return to lessened STP, also now with Kappa=1,K=1
        Phase['stim_matrix']['STM'] = []
        tem = params['prog']['stimphase']['stim_matrix']['STM']
        print(tem)
        Phase['stim_matrix']['STM'] = params['prog']['stimphase']['stim_matrix']['STM'][LTM_memory_id]
        Phase['AMPA_params']['STM']['tau_fac'] = parameters.global_tau_fac
        Phase['stim_rate']['STM'] = parameters.STM_stim

        program = helpers.RunPhase(Phase, model, program, {'name': 'Cue_' + str(LTM_memory_id), 'length': 350.})

        Phase = copy.deepcopy(params['prog']['freephase'])
        Phase['AMPA_params']['STM']['tau_fac'] = parameters.global_tau_fac
        program = helpers.RunPhase(Phase, model, program,
                           {'name': 'Activation_' + str(LTM_memory_id), 'length': 50.})  # rest of the 500ms

    # maintainance
    Phase = copy.deepcopy(params['prog']['freephase'])
    Phase['L23e_zmn_rate'][
        'STM'] = 0.  # 985.#STM upregulation for strongly WM-driven rehearsal. 1000 works great BUT! at 1000ms, a lot of weak units get triggered, reactivatating unsed LTM attractors
    # Phase['L23e_zmn_rate']['STM']=STM_zmn # Maybe it must have noise??
    Phase['AMPA_params']['STM']['tau_fac'] = parameters.global_tau_fac
    duration = 7000.
    program = helpers.RunPhase(Phase, model, program, {'name': ('Maintainance'), 'length': duration})

    return program

def SimulationProtocol(model,params):
    program = []

    # noise
    Phase = copy.deepcopy(params['prog']['noisephase'])
    program = helpers.RunPhase(Phase, model, program, {'length': (600.)})

    # preload attractors
    helpers.PreloadMeans(params, model, 'STM', load_MCs=True, load_WTA=True, load_attractors=True,
                         load_globalinhibition=True,
                         MC_gain=1, WTA_gain=1, gi_gain=1)
    # cooldown
    Phase = copy.deepcopy(params['prog']['cooldownphase'])  
    program = helpers.RunPhase(Phase, model, program, {'name': 'CooldownPhase', 'length': (200.)})

    return program


nest.ResetKernel()
nest.set_verbosity('M_QUIET')

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

if RANK==0:
    params = parameters.params                      # retrieve parameters
else:
    params = None
params = COMM.bcast(params,root=0)
    
model = BuildNet(params)                            # build and connect network
program = SimulationProtocol(model,params)          # sequentially runs all phases defined in the sim protocol function

# TODO post processing of devices


