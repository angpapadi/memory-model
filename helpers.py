import nest
import os
import numpy as np, pylab as plt
import collections, math, random
import parameters 
import pickle

#__________________Functions involved in network creation________________________#
def CreateMC(basket_nodes):
    # load parameters
    neuron_model = parameters.neuron_model
    neuron_params = parameters.neuron_params
    
    local_recurrent_pyr_synapse_params = parameters.local_recurrent_pyr_synapse_params
    local_l4_to_l23_synapse_params     = parameters.local_l4_to_l23_synapse_params
    pyr_to_inh_synapse_params       = parameters.pyr_to_inh_synapse_params
    
    L23_per_MC = parameters.L23_per_MC
    L4_per_MC = parameters.L4_per_MC
    
    # build a minicolumn
    L23_pyr = nest.Create(neuron_model, L23_per_MC , neuron_params)
    L4_pyr = nest.Create(neuron_model, L4_per_MC, neuron_params) 
    nest.SetStatus(L4_pyr, {'b': 0.,'w':0.})                              # switch off adaptation
    
    # Local static connections
    conn_dict = {'rule':'pairwise_bernoulli', 'p': parameters.prob_pyr2pyr_recurrent, 'autapses': False, 'multapses': False}
    nest.Connect(L23_pyr, L23_pyr, conn_dict, local_recurrent_pyr_synapse_params)

    conn_dict = {'rule':'pairwise_bernoulli', 'p': parameters.prob_L4_to_L23, 'autapses': False, 'multapses': False}
    nest.Connect(L4_pyr, L23_pyr, conn_dict, local_l4_to_l23_synapse_params)
    
    conn_dict = {'rule':'pairwise_bernoulli', 'p': parameters.prob_pyr2inh, 'autapses': False, 'multapses': False}
    nest.Connect(L23_pyr, basket_nodes, conn_dict, pyr_to_inh_synapse_params)
    
    
    MC_nodes = {'L23_pyr':L23_pyr, 'L4_pyr': L4_pyr, 'basket_cells': basket_nodes}
    return MC_nodes


def CreateHC(n_MC_per_HC):
    basket_cells_per_MC = parameters.inh_per_MC
    
    all_MC_populations_in_this_HC = []

    # create basket cells
    basket_cells = nest.Create(parameters.neuron_model, n_MC_per_HC, parameters.neuron_params)
    nest.SetStatus(basket_cells, {'b': 0.,'w':0.}) # switch off adaptation
    
    L23_population_in_this_HC = []
    for mc in range(n_MC_per_HC):
        basket_nodes = random.sample(basket_cells, basket_cells_per_MC)
        MC_nodes = CreateMC(tuple(basket_nodes))
        all_MC_populations_in_this_HC.append(MC_nodes)
        
        L23_population_in_this_HC.append(MC_nodes['L23_pyr'])
        
    L23_population_in_this_HC = tuple(np.array(L23_population_in_this_HC).flatten())
    # connect basket cells
    inh_to_pyr_synapse_params       = parameters.inh_to_pyr_synapse_params
    conn_dict = {'rule':'pairwise_bernoulli', 'p': parameters.prob_inh2pyr, 'autapses': False, 'multapses': False}
    nest.Connect(basket_cells, L23_population_in_this_HC, conn_dict, inh_to_pyr_synapse_params)
    
    #SHARED NEURONS (todo in the future)
    #take an argument that specifies how many neurons to share. (could be a percentage e.g. 10% shared neurons)
    #select a number of nodes and return them to be shared between this and another HC
    
    return all_MC_populations_in_this_HC


def createModel(n_HC_desired = parameters.n_HC_desired, n_MC = parameters.n_MC_per_HC):
    
    # Build network architecture    
    all_HC_populations_in_this_net = []
    for hc in range(n_HC_desired):
        HC_nodes = CreateHC(n_MC)
        all_HC_populations_in_this_net.append(HC_nodes)
        
    # Get population dictionaries at all levels
    MC_lvl_pops, HC_lvl_pops, Global_lvl_pops = GetPopulations_per_lvl(all_HC_populations_in_this_net)

    # Connect noise input
    zmn_nodes_BGe = zmn_nodes_BGi = []
    for hypercolumn in range(len(MC_lvl_pops)):
        target_population = HC_lvl_pops[hypercolumn]['L23_pyr']
                
        # bg noise (homogeneous across all HCs but must be connected separately for the poisson processes to be uncorrelated)
        bg_rate = parameters.background_zmn_rate
        zmn_nodes_BGe.append(nest.Create('poisson_generator',params={'rate': bg_rate})) 
        zmn_nodes_BGi.append(nest.Create('poisson_generator',params={'rate': bg_rate}))
        
        nest.Connect(zmn_nodes_BGe[hypercolumn], target_population, parameters.noise_conn_dict, parameters.noise_syn_e)
        nest.Connect(zmn_nodes_BGi[hypercolumn], target_population, parameters.noise_conn_dict, parameters.noise_syn_i)
        
        
    # Connect recording devices
    ## spike detectors from the whole population
    recordfromL23 = Global_lvl_pops['L23_pyr']
    recordfromL4 = Global_lvl_pops['L4_pyr']
    recordfrominh = Global_lvl_pops['basket_cells']

    L23_rec_spikes = nest.Create('spike_detector')         
    L4_rec_spikes = nest.Create('spike_detector')
    inh_rec_spikes = nest.Create('spike_detector')
    
    nest.Connect(recordfromL23, L23_rec_spikes, parameters.SD_conn_dict, parameters.SD_syn_dict) 
    nest.Connect(recordfromL4, L4_rec_spikes, parameters.SD_conn_dict, parameters.SD_syn_dict)
    nest.Connect(recordfrominh, inh_rec_spikes, parameters.SD_conn_dict, parameters.SD_syn_dict)
    
    ## record membrane potential from some or all neurons per MC 
    multimeters_per_MC = []
    for hc in range(len(MC_lvl_pops)):
        for mc in range(len(MC_lvl_pops[hc])):
            multimeter = nest.Create('multimeter', params={'interval': parameters.multimeter_interval,
                                                           'record_from': parameters.multimeterkeys,
                                                           'withgid': True,
                                                           'withtime': True, 'start': 0.})
        
            multimeter_pop_MC = random.sample(MC_lvl_pops[hc][mc]['L23_pyr'],1)
            #multimeter_pop_MC = MC_lvl_pops[hc][mc]['L23_pyr'][:parameters.mm_n_per_MC]  
            #multimeter_pop_MC = random.sample(MC_lvl_pops[hc][mc]['basket_cells'],1)

            
            nest.Connect(multimeter, multimeter_pop_MC)
            multimeters_per_MC.append(multimeter)
    

    # return nodes, noise, devices
    model = {'population_nodes': [MC_lvl_pops, HC_lvl_pops, Global_lvl_pops], 
             'noise_nodes':      [zmn_nodes_BGe, zmn_nodes_BGi], 
             'device_nodes':     [L23_rec_spikes, L4_rec_spikes, inh_rec_spikes, multimeters_per_MC] }
    
    return model
    

#____________Functions involved in defining network topology_____________#
def GenerateRadialPoints(center_x, center_y, mean_radius, sigma_radius, num_points):
    points = []
    for theta in np.linspace(0, 2*math.pi - (2*math.pi/num_points), num_points):
        radius = random.gauss(mean_radius, sigma_radius)
        x = round(center_x + radius * math.cos(theta),4)
        y = round(center_y + radius * math.sin(theta),4)
        points.append([x,y])
    return np.array(points)


def CreateNetworkGrid():
    # Create a square grid where each square is an HC spanning 0.64 mm
    rows = cols = int(np.sqrt(parameters.n_HC_desired))
    #rows = 2
    #cols = 3
    interHC_grid_X, interHC_grid_Y = np.meshgrid(np.linspace(0.,parameters.HC_spatial_extent,rows),
                                                 np.linspace(0.,parameters.HC_spatial_extent,cols))
    grid = [interHC_grid_X, interHC_grid_Y]
    
    # Organize MC centers radially around each HC center
    mc_centers = []
    for whichHC in range(parameters.n_HC_desired):
        
        inter_grid_row = whichHC // int(np.sqrt(parameters.n_HC_desired))
        inter_grid_col = whichHC % int(np.sqrt(parameters.n_HC_desired))
        HC_center = [grid[0][inter_grid_row][inter_grid_col] + parameters.HC_spatial_extent/2, 
                     grid[1][inter_grid_row][inter_grid_col] + parameters.HC_spatial_extent/2]
        
        #mc_centers.append(OrganizeMCsAround(HC_center)) # organize MCs in different rings around HC center
        mc_centers.append(GenerateRadialPoints(HC_center[0], HC_center[1], parameters.HC_spatial_extent/2, 0., parameters.n_MC_per_HC))   # all mcs isapexoun apo to HC center (one ring only)
       
    return mc_centers   # = [HC0, HC1, ...], HC0 = [MC0center, MC1center, ...], MC0center = [x,y]
    
    



#____________Functions involved in Preloading Memories___________________#
def GenerateMemoryPatterns(n_new_patterns, pattern_type='orthogonal'):
    totalmcs = parameters.n_HC_desired * parameters.n_MC_per_HC
    stim_matrix = [[0 for mc in range(totalmcs)] for n in range(n_new_patterns)]
    
    if pattern_type == 'orthogonal':
        if n_new_patterns <= parameters.n_MC_per_HC:
            for idx in range(n_new_patterns):
                MC0_HC = range(0, totalmcs, parameters.n_MC_per_HC)
                for hc in range(parameters.n_HC_desired):
                    stim_matrix[idx][MC0_HC[hc] + idx] = 1
        else:
            print('Not possible to create this many orthogonal patterns')
    elif pattern_type == 'random':
        for idx in range(n_new_patterns):
            MC0_HC = range(0, totalmcs, parameters.n_MC_per_HC)
            randomMC_HC = list(np.random.randint(0, parameters.n_MC_per_HC, parameters.n_HC_desired))
            for hc in range(parameters.n_HC_desired):
                stim_matrix[idx][MC0_HC[hc] + randomMC_HC[hc]] = 1
    else:
        print('Pattern type not specified correctly')
    
    return stim_matrix


def GetPatternNodes(pattern, mc_lvl_pops, whichpop = 'L23_pyr', cue = False):
    cells = []
    for mc in range(len(pattern)):
        hc_idx = int(mc/parameters.n_MC_per_HC)
        mc_idx = mc % parameters.n_MC_per_HC
        if pattern[mc]==1:
            if cue:    
               # if random.random() <= parameters.completeness_lvl:
               #cells.extend(mc_lvl_pops[hc_idx][mc_idx][whichpop])
               if hc_idx <4:  #target specific HCs during cue activation
                    cells.extend(mc_lvl_pops[hc_idx][mc_idx][whichpop])
            else:
                cells.extend(mc_lvl_pops[hc_idx][mc_idx][whichpop])

    return cells


def ComputeLongRangeDelay(mc_pre_idx, mc_post_idx):
    mc_centers = CreateNetworkGrid()
    delay = 1.0
    
    # compute coordinates for pre and post
    whichHC = mc_pre_idx // parameters.n_MC_per_HC
    whichMCinHC = mc_pre_idx % parameters.n_MC_per_HC
    pre  = mc_centers[whichHC][whichMCinHC]
    
    whichHC = mc_post_idx // parameters.n_MC_per_HC
    whichMCinHC = mc_post_idx % parameters.n_MC_per_HC
    post = mc_centers[whichHC][whichMCinHC]
    
    # compute distance between mc_pre and mc_post
    distance = np.linalg.norm(pre-post)
    #print(distance)
    
    # compute delay
    delay = delay + (distance / parameters.conductance_speed)
    
    return delay



def LoadMemories(number_of_memories, model, pattern_type= 'orthogonal'):
    
    # create orthogonal patterns
    orthogonal_memories = GenerateMemoryPatterns(number_of_memories, pattern_type)
    # get all neurons by MC
    MC_population=model['population_nodes'][0]
    
    # define plastic synapse models
    nest.CopyModel(parameters.plastic_synapse_model, 'AMPA_synapse', parameters.AMPA_synapse_params)
    nest.CopyModel(parameters.plastic_synapse_model, 'NMDA_synapse', parameters.NMDA_synapse_params)
    
    alldelaysforhist = [] # assuming we only have one memory 
    # for each pattern, connect pyr neurons with 0.3 prob (tsodyks) (both AMPA+NMDA connections)
    for pattern in orthogonal_memories:
        # get mc indices that are connected in this pattern (where pattern[i]==1)
        participating_MCs = np.where(np.array(pattern)==1)[0]

        # connect MC pairs with 0.3 prob (l23e pop) double forloop for all mcs from previous step
        for mc_pre in participating_MCs:
            for mc_post in participating_MCs:
                if mc_pre!=mc_post: # to ensure that connections are only drawn between different MCs
 
                    # select neurons belonging to the orthogonal pattern
                    pre_pop = MC_population[int(mc_pre/parameters.n_MC_per_HC)][mc_pre % parameters.n_MC_per_HC]['L23_pyr']
                    post_pop = MC_population[int(mc_post/parameters.n_MC_per_HC)][mc_post % parameters.n_MC_per_HC]['L23_pyr']
                    
                    conn_dict = {'rule':'pairwise_bernoulli', 'p': parameters.prob_pyr2pyr_longrange, 
                                 'autapses': False, 'multapses': False}
                      
                    #if int(mc_pre/parameters.n_MC_per_HC) < 4:  #reduce connections from targeted HCs
                    #     if int(mc_post/parameters.n_MC_per_HC) >=4: #but only to non  targeted HCs
                    #        conn_dict = {'rule':'pairwise_bernoulli', 'p':parameters.prob_pyr2pyr_longrange*0.5,
                    #            'autapses':False,'multapses':False}
                     
                    delay = ComputeLongRangeDelay(mc_pre, mc_post)
                    syn_dict = {'model': 'AMPA_synapse', 'delay': delay}
                    nest.Connect(pre_pop, post_pop,conn_dict, syn_dict)

                    #DRAW LONG RANGE PROJECTIONS TO DISTANT INHIBITORY CELLS
                    #distantinhcells = MC_population[int(mc_post/parameters.n_MC_per_HC)][mc_post % parameters.n_MC_per_HC]['basket_cells']
                    #conn_dict2inh = {'rule':'pairwise_bernoulli', 'p':parameters.prob_pyr2pyr_longrange*.5,
                    #       'autapses':False,'multapses':False}
                    #nest.Connect(pre_pop, distantinhcells, conn_dict2inh,syn_dict)

                    # retrieve the connections we just created
                    AMPA_conns = nest.GetConnections(source=pre_pop,target=post_pop, synapse_model='AMPA_synapse')
                    AMPA_status = nest.GetStatus(AMPA_conns)
                    #print(mc_pre,mc_post,len(AMPA_conns))
                    # Dither 
                    dither = parameters.delay_dither_relative_sd
                    if dither > 0.:
                        delays = nest.GetStatus(AMPA_conns, 'delay')
                        #print(delays[0])
                        dithered_delays = [{'delay': max(1.5, round(np.random.normal(delays[i], delays[i] * dither), 1))}
                                            for i in range(len(AMPA_conns))]
                        nest.SetStatus(AMPA_conns, dithered_delays)
                        alldelaysforhist.extend(dithered_delays)   #### to create histogram of conn delays
                                             
                    # Add NMDA connections
                    for status in AMPA_status:
                        my_syn_dict = {'model': 'NMDA_synapse', 'delay': status['delay']}
                        nest.Connect([status['source']], [status['target']], syn_spec = my_syn_dict)
                            
    
    return orthogonal_memories


#____________Functions involved in computing firing statistics___________#

def ComputeCV2(v, with_nan=False):
    """
    Calculate the measure of CV2 for a sequence of time intervals between 
    events.
    """
    # convert to array, cast to float  
    v = np.asarray(v)

    # ensure the input ia a vector
    if len(v.shape) > 1:
        raise ValueError("Input shape is larger than 1. Please provide "
                             "a vector as an input.")

    # ensure we have enough entries
    if v.size < 2:
        if with_nan:
            warnings.warn("Input size is too small. Please provide"
                          "an input with more than 1 entry. cv2 returns `NaN`"
                          "since the argument `with_nan` is `True`")
            return np.NaN
        else:
            raise ValueError("Input size is too small. Please provide "
                                 "an input with more than 1 entry. cv2 returns any"
                                 "value since the argument `with_nan` is `False`")

    # calculate CV2 and return result
    return 2. * np.mean(np.absolute(np.diff(v)) / (v[:-1] + v[1:]))


def isi(spiketrain, axis=-1):
    """
    Return an array containing the inter-spike intervals of the SpikeTrain.
    """
    if axis is None:
        axis = -1
    
    intervals = np.diff(np.sort(spiketrain), axis=axis)
    return intervals



#________________Functions involved in parsing the network and plotting________________#
def GetPopulations_per_lvl(fullnet):
    # gets as input the full dictionary containing all nodes organized per population in each MC
    # return 3 dictionaries at different levels of organization
    
    # mc level returns the population nodes in that particular mc (trivial, simply return the full dict)
    mc_lvl_pops = fullnet
    
    # hc level returns the population nodes in each HC
    hc_lvl_pops = [{'L23_pyr':[],'L4_pyr':[],'basket_cells':[]} for hc in range(len(fullnet))]
    
    for hc in range(len(fullnet)):
        for mc in fullnet[hc]:
            hc_lvl_pops[hc]['L23_pyr']      += list(mc['L23_pyr'])
            hc_lvl_pops[hc]['L4_pyr']       += list(mc['L4_pyr'])
            hc_lvl_pops[hc]['basket_cells'] += list(mc['basket_cells'])
            
            # remove duplicates because basket cells of MCs overlap
            hc_lvl_pops[hc]['basket_cells'] = list(dict.fromkeys(list(hc_lvl_pops[hc]['basket_cells'])))
          
    # global level returns the population nodes in the whole network
    global_lvl_pops = {'L23_pyr':[],'L4_pyr':[],'basket_cells':[]}
    
    for hc in range(len(fullnet)):
        global_lvl_pops['L23_pyr']          += hc_lvl_pops[hc]['L23_pyr']
        global_lvl_pops['L4_pyr']           += hc_lvl_pops[hc]['L4_pyr']
        global_lvl_pops['basket_cells']     += hc_lvl_pops[hc]['basket_cells']

    return mc_lvl_pops, hc_lvl_pops, global_lvl_pops


def CreateParametersDict():
    params = {
                'background noise rate' : parameters.background_zmn_rate,
                'spatial noise rate' : parameters.L23_zmn_rate,
                'pyr2basket connection probability':parameters.prob_pyr2inh,
                'basket2pyr connection probability':parameters.prob_inh2pyr,
                'pyr2pyr long range connection probability' : parameters.prob_pyr2pyr_longrange,
                'pyr2pyr recurrent connection probability': parameters.prob_pyr2pyr_recurrent,
                'pyr2pyr recurrent connection weight' : parameters.local_recurrent_pyr_synapse_params['weight'],
                'basket2pyr connection weight' : parameters.inh_to_pyr_synapse_params['weight'],
                'pyr2basket connection weight' : parameters.pyr_to_inh_synapse_params['weight'],
                'ampa tau_fac' : parameters.AMPA_synapse_params['tau_fac'],
                'ampa tau_rec' : parameters.AMPA_synapse_params['tau_rec'],
                'ampa weight ' : parameters.AMPA_synapse_params['weight'],
                'nmda tau_fac' : parameters.NMDA_synapse_params['tau_fac'],
                'nmda tau_rec' : parameters.NMDA_synapse_params['tau_rec'],
                'nmda weight ' : parameters.NMDA_synapse_params['weight'],
                'neuron b    ' : parameters.neuron_params['b'],
                'neuron tau_w' : parameters.neuron_params['tau_w'],
                'neuron w    ' : parameters.neuron_params['w']
              }
    
    return params
        
    
    
def SaveSimulationToFile(filename, network, memories):
    network_nodes = network['population_nodes']
    
    memory_nodes = []
    for memory in memories:
        memory_nodes += GetPatternNodes(memory, network_nodes[0])
            
    L23_sd = nest.GetStatus(network['device_nodes'][0], keys='events')[0]
    L4_sd = nest.GetStatus(network['device_nodes'][1], keys='events')[0]
    basket_sd = nest.GetStatus(network['device_nodes'][2], keys='events')[0]
    
    multimeters_per_MC =[]
    for mm in network['device_nodes'][3]:
        multimeters_per_MC.append(nest.GetStatus(mm)[0])

    alldata = {'population_nodes': network_nodes, 
               'spike_detectors': [L23_sd, L4_sd, basket_sd], 
               'multimeters': multimeters_per_MC,
               'memories': memory_nodes,
               'parameters': CreateParametersDict(),
               'simtime': nest.GetKernelStatus()['time']
              }
    
    
    with open(filename, "wb") as fp:   
        pickle.dump(alldata, fp)

        
        

#_________________Functions enabling simulation protocol___________________#
def CuePattern(network, memory, start):
    # this is a function that given a pattern it finds the l23 pattern nodes and stimulates its respective l4
    
    target_nodes = GetPatternNodes(memory, network['population_nodes'][0], 'L4_pyr', cue = True)
        
    stimulus_node = nest.Create('poisson_generator',
                                params={'rate' : parameters.stim_rate,'start': start, 'stop': start + parameters.stim_length})
    
    nest.Connect(stimulus_node, target_nodes, syn_spec=parameters.noise_syn_e)  
    

def HCspecificInput(network):
    MC_lvl_pops = network['population_nodes'][0]
    HC_lvl_pops = network['population_nodes'][1]

    
    #zmn_nodes_L23e = zmn_nodes_L23i = []
    for hypercolumn in range(len(MC_lvl_pops)):
        target_population = HC_lvl_pops[hypercolumn]['L23_pyr']
        #target_population = MC_lvl_pops[hypercolumn][0]['L23_pyr']   # mc specific input
         
        # HC specific noise input
        HCspecific_rate = parameters.L23_zmn_rate[hypercolumn]
        if HCspecific_rate>0.:

            zmn_nodes_L23e = nest.Create('poisson_generator',params={'rate': HCspecific_rate, 'start': 2000.,'stop':2600.}) 
            zmn_nodes_L23i = nest.Create('poisson_generator',params={'rate': HCspecific_rate, 'start': 2000.,'stop':2600.})
                 
            nest.Connect(zmn_nodes_L23e, target_population, parameters.noise_conn_dict, parameters.noise_syn_e)
            nest.Connect(zmn_nodes_L23i, target_population, parameters.noise_conn_dict, parameters.noise_syn_i)
        if HCspecific_rate<0:

            only_i_nodes = nest.Create('poisson_generator',params={'rate':-1*HCspecific_rate,'start':2000., 'stop':2600.})
            nest.Connect(only_i_nodes, target_population, parameters.noise_conn_dict, parameters.noise_syn_i)
    




