import nest
import os
import numpy as np, pylab as plt
import collections,math, random
import parameters 
from mpi4py import MPI
import nest.topology as tp

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
    nest.SetStatus(L4_pyr, {'b': 0.})                              # switch off adaptation
    
    # Local static connections
    conn_dict = {'rule':'pairwise_bernoulli', 'p': parameters.prob_pyr2pyr_recurrent, 'autapses': False, 'multapses': False}
    nest.Connect(L23_pyr, L23_pyr, conn_dict, local_recurrent_pyr_synapse_params)

    conn_dict = {'rule':'pairwise_bernoulli', 'p': parameters.prob_L4_to_L23, 'autapses': False, 'multapses': False}
    nest.Connect(L4_pyr, L23_pyr, conn_dict, local_l4_to_l23_synapse_params)
    
    conn_dict = {'rule':'pairwise_bernoulli', 'p': parameters.prob_pyr2inh, 'autapses': False, 'multapses': False}
    nest.Connect(L23_pyr, basket_nodes, conn_dict, pyr_to_inh_synapse_params)
    
    
    MC_nodes = {'L23_pyr':L23_pyr, 'L4_pyr': L4_pyr, 'basket_cells': basket_nodes}
    return MC_nodes


def CreateHC():
    n_MC_per_HC = parameters.n_MC_per_HC
    basket_cells_per_MC = parameters.inh_per_MC
    
    all_MC_populations_in_this_HC = []

    # create basket cells
    basket_cells = nest.Create(parameters.neuron_model, n_MC_per_HC, parameters.neuron_params)
    nest.SetStatus(basket_cells, {'b': 0.}) # switch off adaptation
    
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


def createModel():
    
    # Build network architecture
    n_HC_desired = parameters.n_HC_desired
    
    all_HC_populations_in_this_net = []
    for mc in range(n_HC_desired):
        HC_nodes = CreateHC()
        all_HC_populations_in_this_net.append(HC_nodes)
        
    # Get population dictionaries at all levels
    MC_lvl_pops, HC_lvl_pops, Global_lvl_pops = GetPopulations_per_lvl(all_HC_populations_in_this_net)

    # Connect noise input
    zmn_nodes_L23e = []
    zmn_nodes_L23i = []
    for hypercolumn in range(len(MC_lvl_pops)):
        rate = parameters.L23_zmn_rate[hypercolumn]
        zmn_nodes_L23e.append(nest.Create('poisson_generator',params={'rate': rate})) 
        zmn_nodes_L23i.append(nest.Create('poisson_generator',params={'rate': rate}))
        
        target_population = HC_lvl_pops[hypercolumn]['L23_pyr']
        nest.Connect(zmn_nodes_L23e[hypercolumn], target_population, parameters.noise_conn_dict, parameters.noise_syn_e)
        nest.Connect(zmn_nodes_L23i[hypercolumn], target_population, parameters.noise_conn_dict, parameters.noise_syn_i)
        
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
    
    ### Extra spike detector to compute cv2 from an arbitrary background neuron
    extrasd = nest.Create('spike_detector')
    #nest.Connect([2000],extrasd)
        
    ## record membrane potential from 2 L23_pyr neurons per MC 
    multimeters_per_MC = []
    for hc in range(len(MC_lvl_pops)):
        for mc in range(len(MC_lvl_pops[hc])):
            multimeter = nest.Create('multimeter', params={'interval': parameters.multimeter_interval,
                                                           'record_from': parameters.multimeterkeys,
                                                           'withgid': True,
                                                           'withtime': True, 'start': 0.})
        
            #multimeter_pop_MC = random.sample(MC_lvl_pops[hc][mc]['L23_pyr'],2)
            multimeter_pop_MC = MC_lvl_pops[hc][mc]['L23_pyr'][:parameters.mm_n_per_MC]  
            
            nest.Connect(multimeter, multimeter_pop_MC)
            multimeters_per_MC.append(multimeter)

    # return nodes, noise, devices
    model = {'population_nodes': [MC_lvl_pops, HC_lvl_pops, Global_lvl_pops], 
             'noise_nodes':      [zmn_nodes_L23e, zmn_nodes_L23i], 
             'device_nodes':     [L23_rec_spikes, L4_rec_spikes, inh_rec_spikes, multimeters_per_MC, extrasd] }
    
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


def GetPatternNodes(pattern, model):
    cells = []
    for mc in range(len(pattern)):
        hc_idx = int(mc/parameters.n_MC_per_HC)
        mc_idx = mc % parameters.n_MC_per_HC
        if pattern[mc]==1:
            cells.extend(model['population_nodes'][0][hc_idx][mc_idx]['L23_pyr'])
    return cells


def ComputeLongRangeDelay(mc_pre_idx, mc_post_idx):
    delay = 1.0
    mc_centers = parameters.mc_centers
    
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



def LoadMemories(number_of_memories, model):
    
    # create orthogonal patterns
    orthogonal_memories = GenerateMemoryPatterns(number_of_memories)
    # get all neurons by MC
    MC_population=model['population_nodes'][0]
    
    # define plastic synapse models
    nest.CopyModel(parameters.plastic_synapse_model, 'AMPA_synapse', parameters.AMPA_synapse_params)
    nest.CopyModel(parameters.plastic_synapse_model, 'NMDA_synapse', parameters.NMDA_synapse_params)
    
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

                    delay = ComputeLongRangeDelay(mc_pre, mc_post)
                    syn_dict = {'model': 'AMPA_synapse', 'delay': delay}
                    nest.Connect(pre_pop, post_pop,conn_dict, syn_dict)

                    # retrieve the connections we just created
                    AMPA_conns = nest.GetConnections(source=pre_pop,target=post_pop, synapse_model='AMPA_synapse')
                    AMPA_status = nest.GetStatus(AMPA_conns)

                    # Dither 
                    dither = parameters.delay_dither_relative_sd
                    if dither > 0.:
                        delays = nest.GetStatus(AMPA_conns, 'delay')
                        #print(delays[0])
                        dithered_delays = [{'delay': max(1.5, round(np.random.normal(delays[i], delays[i] * dither), 1))}
                                            for i in range(len(AMPA_conns))]
                        nest.SetStatus(AMPA_conns, dithered_delays)
                        #print(dithered_delays[0])
                    
                    # Add NMDA 
                    for status in AMPA_status:
                        my_syn_dict = {'model': 'NMDA_synapse', 'weight': status['weight'], 'delay': status['delay']}  
                        nest.Connect([status['source']], [status['target']], syn_spec = my_syn_dict)
                        
                           
    return orthogonal_memories


#____________Functions involved in computing firing statistics___________#

def ComputeCV2(v, with_nan=False):
    """
    Calculate the measure of CV2 for a sequence of time intervals between 
    events.

    Given a vector v containing a sequence of intervals, the CV2 is
    defined as:

    .math $$ CV2 := \\frac{1}{N}\\sum_{i=1}^{N-1}

                   \\frac{2|isi_{i+1}-isi_i|}
                          {|isi_{i+1}+isi_i|} $$

    The CV2 is typically computed as a substitute for the classical
    coefficient of variation (CV) for sequences of events which include
    some (relatively slow) rate fluctuation.  As with the CV, CV2=1 for
    a sequence of intervals generated by a Poisson process.

    Parameters
    ----------

    v : quantity array, numpy array or list
        Vector of consecutive time intervals

    with_nans : bool, optional
        If `True`, cv2 with less than two spikes results in a `NaN` value 
        and a warning is raised. 
        If `False`, an attribute error is raised. 
        Default: `True`

    Returns
    -------
    cv2 : float
       The CV2 of the inter-spike interval of the input sequence.

    Raises
    ------
    ValueError :
       If an empty list is specified, or if the sequence has less
       than two entries, an AttributeError will be raised.
    ValueError :
        Only vector inputs are supported.  If a matrix is passed to the
        function an AttributeError will be raised.

    References
    ----------
    ..[1] Holt, G. R., Softky, W. R., Koch, C., & Douglas, R. J. (1996). 
    Comparison of discharge variability in vitro and in vivo in cat visual 
    cortex neurons. Journal of neurophysiology, 75(5), 1806-1814.
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

    Parameters
    ----------

    spiketrain : NumPy ndarray
                 The spike times.
    axis : int, optional
           The axis along which the difference is taken.
           Default is the last axis.

    Returns
    -------

    NumPy array or quantities array.

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

    
    '''
    for hc in fullnet:
        for mc in hc:
            global_lvl_pops['L23_pyr']      += list(mc['L23_pyr'])
            global_lvl_pops['L4_pyr']       +=list(mc['L4_pyr'])
            global_lvl_pops['basket_cells'] +=list(mc['basket_cells'])
    
    
    # flatten the resulting list of lists of nodes (DO IT BETTER WITHOUT FOR LOOP?)
    for pop in global_lvl_pops:
        global_lvl_pops[pop] = tuple(np.concatenate(global_lvl_pops[pop]))
    
    for hc in range(len(fullnet)):
        for pop in global_lvl_pops:
            hc_lvl_pops[hc][pop] = tuple(np.concatenate(hc_lvl_pops[hc][pop]))
    '''
    return mc_lvl_pops, hc_lvl_pops, global_lvl_pops


def PlottingJoe(network):
    # a function used to plot various stuff
    spike_detector = network['device_nodes'][0]
    dSD = nest.GetStatus(spike_detector, keys="events")[0]
    
    # PLOT ONLY ONE HC
    nodes = network['population_nodes'][1][0]['L23_pyr'] # get the nodes we want to plot, here the first HC
    evs = dSD["senders"]
    ts = dSD["times"]
    events = evs[np.where(np.isin(evs,nodes))]     # get indices of senders that are withing the range of nodes we want
    times = ts[np.where(np.isin(evs,nodes))]       # using those indices get the corresponding times
    print(len(times),'spikes')
    plt.figure('l23 population spike raster one hc')
    plt.title('l23 population spike raster one hc')
    axes = plt.gca()
    axes.set_xlim([0,2500])
    #plt.plot(times, events, ".")
    plt.scatter(times,events,s=1)

    
    
    # PLOT THE WHOLE POPULATION
    evs = dSD["senders"]
    ts = dSD["times"]
    print(len(ts),'spikes')
    plt.figure('l23 population spike raster')
    plt.title('l23 pop spike raster of the whole net')
    plt.scatter(ts,evs,s=1)
    
    # PLOT THE WHOLE BASKET CELL POPULATION
    spike_detector_bc = network['device_nodes'][2]
    dSD = nest.GetStatus(spike_detector, keys="events")[0]
    
    ievs = dSD["senders"]
    its = dSD["times"]
    print(len(its),'spikes')
    plt.figure('basket population spike raster')
    plt.title('basket pop spike raster of the whole net')
    plt.scatter(its,ievs,s=1)

    
    
    # HISTOGRAM OF FIRING RATES of nodes
    spikes = []         # spikes
    for node in nodes:
        spikes.append(np.count_nonzero(events == node))
    #print(spikes)
    rates = []  # spikes/sec or spikes/sim_time
    cv2s = []   # measures how irregular is the spiking
    for node in nodes:
        spiketimes = times[np.where(events==node)]
        if len(spiketimes)>2:
            cv2s.append(ComputeCV2(isi(spiketimes)))
    num_bins = 20
    plt.figure('hist of spikes')
    plt.title('hist of spikes of HC0 across the whole sim time')

    n, bins, patches = plt.hist(spikes, num_bins)
    
    plt.figure('hist of cv2s')
    plt.title('hist of cv2s of HC0')
    n, bins, patches = plt.hist(cv2s, num_bins)
    
    
    plt.show()



#_________________Functions enabling simulation protocol___________________#
def CuePattern():
    # this is a function that given a pattern it finds the l23 pattern nodes and stimulates its respective l4
    return




