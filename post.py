import nest
import os
import numpy as np, pylab as plt
import collections, math, random
import pickle
import helpers
from scipy import ndimage, signal
plt.rcParams['figure.figsize'] = [20, 7]
plt.rcParams['figure.dpi'] = 200

def plots(simresults):
    
    #______________plots______________#
    # isolate data from 1 HC
    dSD = simresults['spike_detectors'][0]
    nodes =simresults['population_nodes'][1][0]['L23_pyr'] # get the nodes we want to plot, here the first HC
    evs = dSD["senders"]
    ts = dSD["times"]
    events = evs[np.where(np.isin(evs,nodes))]     # get indices of senders that are withing the range of nodes we want
    times = ts[np.where(np.isin(evs,nodes))]       # using those indices get the corresponding times
    print(len(times),'pyr spikes in 1 hc')
    
    # plot MC0
    plt.figure('l23 population spike raster one mc')
    plt.title('l23 population spike raster MC0')
    axes = plt.gca()
    axes.set_xlim([2000,3000])
    axes.set_ylim([0,120])
    plt.scatter(times,events,s=1)
    
    # plot HC0
    plt.figure('l23 population spike raster one hc')
    plt.title('l23 population spike raster HC0')
    axes = plt.gca()
    axes.set_xlim([2000,3500])
    #axes.set_ylim([0,120])
    axes.set_xticks(np.arange(2000, 3000., 100.))
    plt.scatter(times,events,s=1)
    plt.grid()
    
    # plot all L23pyr
    evs = dSD["senders"]
    ts = dSD["times"]
    print(len(ts),'pyr spikes')
    plt.figure('l23 population spike raster')
    plt.title('l23 pop spike raster of the whole net')
    axes = plt.gca()
    axes.set_xlim([0, 6000])
    axes.axvline(x=2500,c='r')  # cue onset
    #axes.set_ylim([0,12000])
    plt.scatter(ts,evs,s=1)
    
    # plot basket cells in one HC
    dSD = simresults['spike_detectors'][2]
    ievs = dSD["senders"]
    its = dSD["times"]
    print(len(its),'bc spikes')
    plt.figure('basket population spike raster')
    plt.title('basket pop spike raster of 1 HC')
    axes = plt.gca()
    axes.set_xlim([2500,4000])
    axes.set_ylim([9100,9200])
    plt.scatter(its,ievs,s=1)
    
    # plot 1 multimeter recording
    plotting_minicolumn = 0
    dmm = simresults['multimeters'][plotting_minicolumn]
    Vms = dmm["events"]["V_m"]
    ts = dmm["events"]["times"]
    plt.figure('Membrane potential')
    #plt.title('membrane potential of one basket cell (mc0)')
    plt.title('membrane potential of one pyr cell')
    plt.plot(ts, Vms)
    
    plt.show()
    
    
    
def showparams(simresults):
    for key in simresults['parameters'].keys():
        print(key,':  ',simresults['parameters'][key])


    
    
def get_histogram(spikes, window=None, dt=1, bins=None):
    spks = spikes
        
    if window is None:
        window = (spks[0], spks[-1])
    if bins is None:
        bins = np.arange(window[0], window[1], dt)
        bins = np.append(bins, window[1])
        
    plt.title('histogram of spikes with bins of 10 ms')
    plt.hist(spks,bins)
    plt.show()
    return np.histogram(spks, bins)

def spectral(alltimes):
    (discrete, bin_edges) = get_histogram(alltimes, window=(1000.,5000.), dt=10.)
    result = ndimage.gaussian_filter1d(np.asfarray(discrete), sigma=.5, mode='constant')

    # plot histogram convolved with gaussian kernel
    plt.plot(result)
    plt.title('after filtering with gaussian kernel')
    #plt.xlim([0,1300])

    # power spectral estimation
    f, Pxx_den = signal.welch(result, fs =100, nperseg=100, noverlap=50)
    #f, Pxx_den = signal.periodogram(result)
    plt.figure('power spectral density')
    plt.title('power spectral density')
    plt.semilogy(f, Pxx_den)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()



#_______wrapper function_______#
def processing(filename):
    
    with open(filename, 'rb') as fp:
        simresults = pickle.load(fp)
        
    plots(simresults)
    showparams(simresults)
    
