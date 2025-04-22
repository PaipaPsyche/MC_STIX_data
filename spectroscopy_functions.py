
#package imports
# UTILS
import datetime as dt
from datetime import datetime,time,timedelta
import os
## PLOT
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter
## MATH
from scipy.optimize import curve_fit as curvefit
import numpy as np


## ASTROPY
from astropy.io import fits
from astropy.time.core import Time, TimeDelta
from astropy.table import Table
import astropy.units as u


def get_tables(sci_file_path):
    hdulist = fits.open(sci_file_path)
    header = hdulist[0].header
    
    data = Table(hdulist[2].data)
    energies = Table(hdulist[4].data)
    
    
    energies["mean_energy"] = np.array([np.mean([e["e_low"],e["e_high"]]) if e["e_high"]!=np.inf else e["e_low"] for e in energies ])
    
    data["time"]= Time(header['date-obs']) + TimeDelta(data['time'] * u.cs)
    timedel = data["timedel"]/100
    data["cts_per_sec"]=data["counts"]/timedel.reshape(-1,1)

    return data,energies,header




def remove_bkg(sci_file_path,bkg_file_path):
    sci_data_,sci_ene_,sci_head_=get_tables(sci_file_path)
    n_energies = len(sci_ene_)
    #bkg_data_,bkg_ene_,bkg_head_=get_tables(bkg_file_path)
    
    
    hdulist_bkg = fits.open(bkg_file_path)
    header_bkg = hdulist_bkg[0].header
    data_bkg = Table(hdulist_bkg[2].data)

    energies_bkg = Table(hdulist_bkg[4].data)
    n_energies_bkg = len(energies_bkg)
    mean_e_bkg = np.array([np.mean([e["e_low"],e["e_high"]]) if e["e_high"]!=np.inf else e["e_low"] for e in energies_bkg ])
    
    
    cts_sec = np.sum(data_bkg["counts"],axis=(1,2))
    timedel = data_bkg['timedel']/100
    data_bkg["cts_per_sec"] = [(cts_sec/timedel)[0].reshape(n_energies_bkg)]

    #we'll store here the 32 bkg count rates for each energy bin 
    bkg_countrate= []
    
    # we iterate over the energy bins of the science file 
    for e in sci_ene_:
        # we look for bins with the same bounds
        bkg_energy_bin = energies_bkg[energies_bkg[["e_low","e_high"]]==e[["e_low","e_high"]]]
        # if found, use the count rate value for that energy bin that we already estimated
        # if not, use 0 instead
        if len(bkg_energy_bin)==1:
            bkg_countrate.append(data_bkg["cts_per_sec"][0][bkg_energy_bin["channel"]][0])
        else:     
            bkg_countrate.append(0)

    # reshape the bkg spectrum (vector of len n_energies) into a 2D array of the same shape as the science cts/sec  
    # in order to remove the bkg counts per energy bin from every time bin in the sci file 
    
    # repet the cts/sec for each time bin to make a counts "matrix" for 
    # the bkg in the same shape as cts/sec for SCI file
    bkg_array = np.repeat(np.array([bkg_countrate])[None, :], len(sci_data_["time"]), axis=1)
    # subtract and limit lower boundary to 0
    corr_array =sci_data_["cts_per_sec"]-bkg_array
    corr_array = np.clip(corr_array,0,np.inf)
    corr_array = corr_array.reshape( ( len(sci_data_["time"]) , n_energies ) )
    
    # save BKG and corrected cts/Sec
    sci_data_["bkg_cts_per_sec"] =data_bkg["cts_per_sec"]
    sci_data_["corrected_cts_per_sec"] = corr_array




    

    # bkg_array = np.repeat(np.array(data_bkg["cts_per_sec"])[None, :], len(sci_data_["time"]), axis=1)
    # corr_array =sci_data_["cts_per_sec"]-bkg_array
    # corr_array = np.clip(corr_array,0,np.inf)
    # corr_array = corr_array.reshape( ( len(sci_data_["time"]) , n_energies_bkg ) )

    # sci_data_["bkg_cts_per_sec"] =data_bkg["cts_per_sec"]
    # sci_data_["corrected_cts_per_sec"] = corr_array
    
    
    sci_data_["corrected_counts"]= np.zeros(np.shape(corr_array))
    for i in range(np.shape(corr_array)[0]):
        sci_data_["corrected_counts"][i,:] =  corr_array[i,:]*sci_data_["timedel"][i]/100
    
    return sci_data_,sci_ene_,sci_head_


#paint timeline markers
def paint_markers(ax,markers):
    if(markers):
        for mk in markers:
            m = dt.datetime.strptime(mk,"%Y-%m-%d %H:%M:%S")
            ax.axvline(m,c="r",ls="--")
def plot_quicklooks(data,energies,header,plot_channels=[1,10,16,22,26,30],time_range = None, smooth_pts=None,markers=None):

    # define time range as the data limit, but define a new one if provided
    time_range_dt = [np.min(data["time"].datetime),np.max(data["time"].datetime)]
    if time_range:
        time_range_dt = [datetime.strptime(x,"%Y-%m-%d %H:%M:%S") for x  in time_range]
    
    
    # PLOT
    plt.figure(figsize=(12,8))

    # paint cts/sec for each seleceted channel
    ax=plt.subplot(311)
    for i in plot_channels:
         # energy bin boundaries
        _energy_lbl = f"{round(energies['e_low'][i])} - {round(energies['e_high'][i])} keV"
        # plot counts corresponding to channel i
        ebin_toplot = data["cts_per_sec"][:,i].tolist()
        #smooth if needed
        if(smooth_pts):
            box = np.ones(smooth_pts)/smooth_pts
            ebin_toplot = np.convolve(ebin_toplot,box,mode="same")

        ax.plot(data["time"].datetime,ebin_toplot,label=_energy_lbl)
    # format the date axis (%H: hour,%M: minute,%H: second)
    # you might use your prefered format
    myFmt = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(myFmt)
    # add elegend, plot in logscale,do title
    ax.legend()
    ax.set_yscale("log")
    ax.set_ylabel("Counts/sec")
    ax.set_title("Observation time [@ SolO]: "+Time(header["date-obs"]).datetime.strftime("%d-%b-%Y %H:%M:%S"))

    paint_markers(ax,markers)
    
    ax.set_xlim(*time_range_dt)
    ax.grid(color='lightgrey',lw=0.5)

    #paint variation of time bin lengths 
    ax2=plt.subplot(312)

    # add the cts of all channels
    summed_data = np.sum(data["cts_per_sec"],axis=1)
    #smooth if needed
    if(smooth_pts):
        box = np.ones(smooth_pts)/smooth_pts
        summed_data = np.convolve(summed_data,box,mode="same")

    ax2.plot(data["time"].datetime,summed_data,label="All energy bins",c="k")
    # fortmat date
    myFmt = DateFormatter("%H:%M")
    ax2.xaxis.set_major_formatter(myFmt)

    ax2.legend()
    ax2.set_yscale("log")
    ax2.set_ylabel("Counts/sec")
    paint_markers(ax2,markers)
    ax2.set_xlim(*time_range_dt)
    ax2.grid(color='lightgrey',lw=0.5)


    #paint variation of time bin lengths 
    ax3=plt.subplot(313)

    timedel_plot = data['timedel']
    #smooth if needed
    if(smooth_pts):
        box = np.ones(smooth_pts)/smooth_pts
        timedel_plot = np.convolve(timedel_plot,box,mode="same")

    ax3.plot(data["time"].datetime,timedel_plot)
    # fortmat date
    myFmt = DateFormatter("%H:%M")
    ax3.xaxis.set_major_formatter(myFmt)

    ax3.set_xlabel("Time [@SolO]")
    ax3.set_ylabel("Time bin length (seconds)")
    
    paint_markers(ax3,markers)

    ax3.set_xlim(*time_range_dt)
    ax3.grid(color='lightgrey',lw=0.5)
    
    
def plot_spectrogram(data,energies,header,energy_range=[4,28],corrected=False,markers=None,time_range=None):
    
    # define time range as the data limit, but define a new one if provided
    time_range_dt = [np.min(data["time"].datetime),np.max(data["time"].datetime)]
    if time_range:
        time_range_dt = [datetime.strptime(x,"%Y-%m-%d %H:%M:%S") for x  in time_range]

    
    plt.figure(figsize=(12,4))
    ax = plt.subplot(111)


    myFmt = mdates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(myFmt)


    cts_data = np.array(data['cts_per_sec']).T if not corrected else np.array(data['corrected_cts_per_sec']).T 
    #to plot we replace empty data and 0 counts with a 1
    # do NOT do this for spectroscopy analysis , just for visualization
    cts_data=np.nan_to_num(cts_data,nan=1)
    cts_data[cts_data<=0]=0

    ii = np.logical_and(energies["e_low"]>=energy_range[0],energies["e_high"]<=energy_range[1])
    jj= np.where(ii)[0].astype(int)
    cts_data = cts_data[jj,:]
    mean_e = energies["mean_energy"][jj]


    cm=None
    try:
        cm= plt.pcolormesh(data["time"].datetime,mean_e,cts_data,shading="auto",cmap="nipy_spectral",vmin=1,norm="log")
    except:
        cm= plt.pcolormesh(data["time"].datetime,mean_e,np.log10(cts_data),shading="auto",cmap="nipy_spectral",vmin=1)
    
    
    cblabel = "$Log_{10}$ Counts $s^{-1}$" 
    plt.colorbar(cm,label=cblabel)

    plt.xlabel("Observation time [@ SolO]: "+Time(header["date-obs"]).datetime.strftime("%d-%b-%Y %H:%M:%S"),fontsize=14)
    plt.ylabel('Energy bins [KeV]',fontsize=14)
    plt.ylim(*energy_range)
    
    title_txt = "Background-subtracted" if corrected else ""
    plt.title(title_txt+" counts spectrogram",fontsize=15)

    paint_markers(ax,markers)

    ax.set_xlim(*time_range_dt)
    
    
    


def plot_bkg_spectrum(data,energies,header):
    energy_edges =  energies["e_low"].tolist() + [200]

    plt.figure(figsize=(9,4))
    #plt.plot(energies["mean_energy"],data["bkg_cts_per_sec"][0]/energies["mean_energy"],c="k")
    plt.stairs(data["bkg_cts_per_sec"][0]/energies["mean_energy"][1:],energy_edges[1:])
    plt.axvline(8,c="orange",ls="--",label="CdTe escape peaks")
    plt.axvline(31,c="r",ls="--",label="callibration lines\n31 and 81 kev")
    plt.axvline(81,c="r",ls="--")

    plt.xlabel("Energy (kev)")
    plt.ylabel("Counts / sec / kev")
    plt.title("Background counts spectrum")
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(4,150)
    plt.grid(color='lightgrey',lw=0.5)
    
    
def plot_integrated_bins(data,energies,header,energy_bins=None,corrected=False,smooth_pts=None,markers=None,time_range=None):
    
    # define time range as the data limit, but define a new one if provided
    time_range_dt = [np.min(data["time"].datetime),np.max(data["time"].datetime)]
    if time_range:
        time_range_dt = [datetime.strptime(x,"%Y-%m-%d %H:%M:%S") for x  in time_range]

    energy_bins= energy_bins if energy_bins else [ [4,8] , [12,16],[22,28],[32,50] ]
    #energy_range = [4,50]

    # for each energy range 
    ranges_to_plot = []
    for er in energy_bins:
        # select the energy bin indexes that fall into the desired energy interval
        ii = np.logical_and(energies["e_low"]>=er[0],energies["e_high"]<=er[1])
        jj= np.where(ii)[0].astype(int)

        # select these energies
        selected_eb = energies[jj]
        # from the counts, take only the ones associated wtih the selected energy range
        sel_counts = data["cts_per_sec"][:,jj]
        # if "corrected" and corrected counts available, use instead
        if(corrected):
            sel_counts = data["corrected_cts_per_sec"][:,jj]
        # add the contribution of the energy bins within the selected energy range
        sel_counts =np.sum(sel_counts,axis=1)
        
        #smooth if needed
        if(smooth_pts):
            box = np.ones(smooth_pts)/smooth_pts
            sel_counts = np.convolve(sel_counts,box,mode="same")

            
        # create a directory with the energy boundaries, associated counts and indexes 
        to_plot = {"e_low":selected_eb[0]["e_low"],
                   "e_high":selected_eb[-1]["e_high"],
                   "idx":jj,
                   "cts_per_sec":sel_counts
                  }
        # append this to the "to plot" list
        ranges_to_plot.append(to_plot)
    # PLOT
    plt.figure(figsize=(12,3))
    ax= plt.subplot(111)
    # plot each "to_plot" element
    for  tp  in ranges_to_plot:
        ax.plot(data['time'].datetime,tp["cts_per_sec"],label=f"{round(tp['e_low'])} - {round(tp['e_high'])} kev")

    myFmt = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(myFmt)

    plt.yscale("log")
    plt.xlabel("Observation time [@ SolO]: "+Time(header["date-obs"]).datetime.strftime("%d-%b-%Y %H:%M:%S"),fontsize=14)
    plt.ylabel("Counts / Sec ")
    plt.ylim(bottom=1)
    paint_markers(ax,markers)
    ax.set_xlim(*time_range_dt)
    plt.legend()
    plt.grid(color='lightgrey',lw=0.5)


def plot_spectrum(data,energies,header,selected_interval, e_range = [4,40]):
    # convert marker boundaries to datetime objects 
    dt_markers = [Time(x).datetime for x in selected_interval]
    # estimate the difference btween the boundaries in seconds
    dur_sec = (dt_markers[1]-dt_markers[0]).seconds
    print("Interval duration:",dur_sec," seconds")

    # indexes where time is within the markers interval
    ii = np.logical_and( data["time"].datetime>=dt_markers[0] , data["time"].datetime<=dt_markers[1] )
    jj = np.where(ii)[0]

    # correcting the counts
    # create a target array of the same shape as corrected cts/sec array
    corr_cts = np.zeros(np.shape(data["corrected_cts_per_sec"]))
    
    #for each time bin , multiply by the timedel (to pass from cts/sec to counts) 
    for i in range(np.shape(data["corrected_cts_per_sec"])[0]):
        corr_cts[i,:] =  data["corrected_cts_per_sec"][i,:]*data["timedel"][i]/100

    # add all counts within interval (along time axis)
    summed_counts = np.sum(data["corrected_counts"][jj,:],axis=0)
    # counts/sec spectrum
    spectrum_interval = summed_counts/energies["mean_energy"]/dur_sec
    spectrum_bkg = data["bkg_cts_per_sec"][0]/energies["mean_energy"][1:]


    #using  same process for the raw counts to compare 
    uncorr_counts = data["counts"]/(data["timedel"]/100).reshape(-1,1)
    # add all counts within interval (along time axis)
    summed_raw_counts = np.sum(data["counts"][jj,:],axis=0)
    # counts/sec spectrum
    spectrum_raw_interval = summed_raw_counts/energies["mean_energy"]/dur_sec

    plt.figure(figsize=(12,3))
    plt.loglog(energies["mean_energy"],spectrum_interval,c="k",label="Counts spectrum (w/o bkg)")
    plt.loglog(energies["mean_energy"],spectrum_raw_interval,c="gray",ls="--",label="Counts spectrum (with bkg)")
    plt.loglog(energies["mean_energy"][1:],spectrum_bkg,c="lightgray",label="Background spectrum")
    plt.xlim(*e_range)
    plt.ylim(bottom=0.1)
    
    plt.legend()
    plt.xlabel("Energy (kev)")
    plt.ylabel("Counts / sec / kev")
    plt.title("Spectrum comparison")
    plt.grid(color='lightgrey',lw=0.5)
        



# define the linear fit for the log space (x=log(energy), output=log(spec) ) 
def nonthermalindex(x, a, b):
    return a * x + b
# for the slope m and constant b that are found, we need a function to go from energy to spec (the powerlaw)
def inv_nonthermalindex(x,a,b):
    return np.exp(b)*(x**a)
    

def do_powerlaw_fit(data,energies,header,selected_interval, e_range_fit ,e_range = [4,40]):
    # convert marker boundaries to datetime objects 
    dt_markers = [Time(x).datetime for x in selected_interval]
    # estimate the difference btween the boundaries in seconds
    dur_sec = (dt_markers[1]-dt_markers[0]).seconds
    print("Interval duration:",dur_sec," seconds")

    # indexes where time is within the markers interval
    ii = np.logical_and( data["time"].datetime>=dt_markers[0] , data["time"].datetime<=dt_markers[1] )
    jj = np.where(ii)[0]

    # correcting the counts
    # create a target array of the same shape as corrected cts/sec array
    corr_cts = np.zeros(np.shape(data["corrected_cts_per_sec"]))
    
    #for each time bin , multiply by the timedel (to pass from cts/sec to counts) 
    for i in range(np.shape(data["corrected_cts_per_sec"])[0]):
        corr_cts[i,:] =  data["corrected_cts_per_sec"][i,:]*data["timedel"][i]/100

    # add all counts within interval (along time axis)
    summed_counts = np.sum(data["corrected_counts"][jj,:],axis=0)
    # counts/sec spectrum
    spectrum_interval = summed_counts/energies["mean_energy"]/dur_sec
    spectrum_bkg = data["bkg_cts_per_sec"][0]/energies["mean_energy"][1:]


    #using  same process for the raw counts to compare 
    uncorr_counts = data["counts"]/(data["timedel"]/100).reshape(-1,1)
    # add all counts within interval (along time axis)
    summed_raw_counts = np.sum(data["counts"][jj,:],axis=0)
    # counts/sec spectrum
    spectrum_raw_interval = summed_raw_counts/energies["mean_energy"]/dur_sec

    e_edges = list(energies["e_high"][:-1]) # every e_high value except the last one
    e_edges = np.array([e_edges[0]] + e_edges + [e_edges[-1]]) #repeat the las edge value


    
    # select the energy indexes in the energy fit range
    energy_idx = np.where((energies['mean_energy']>=e_range_fit[0]) & (energies['mean_energy']<=e_range_fit[1]))[0]
    # indexes for the bin edges
    energy_idx_edge = np.insert(energy_idx,energy_idx.size,energy_idx[-1]+1)
    
    # perform the curve fit
    # the linear fit is faster (done in the log-log space)
    popt, pcov = curvefit(nonthermalindex, np.log(energies['mean_energy'][energy_idx]),np.log(spectrum_interval)[energy_idx])
    # associated deviaitons are diagonal of covariance matrix
    err= pcov.diagonal()
    
    # once obtained the coefficents m and b, create the curve corresponding to the fit using the powerlaw
    fit_result = inv_nonthermalindex(energies["mean_energy"][energy_idx],*popt)
    
    print('Spectral index = ',round(popt[0],3),'+-',round(err[0],3))


    plt.figure(figsize=(12,3))
    plt.stairs(spectrum_bkg,edges=e_edges[1:],color="lightgray",label="Background spectrum")
    plt.stairs(spectrum_interval,edges=e_edges,color="k",label="Counts spectrum")
    plt.stairs(spectrum_raw_interval,edges=e_edges,color="gray",ls="--",label="Counts spectrum (Raw)")
    
    plt.stairs(fit_result,edges=e_edges[energy_idx_edge],color="r",
               label=f"Fit from {e_range_fit[0]} - {e_range_fit[1]} keV (index = {round(popt[0],2)})",baseline=None,lw=1.5)
    plt.xlim(*e_range)
    plt.ylim(bottom=0.1)
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Energy (kev)")
    plt.ylabel("Counts / sec / kev")
    plt.title("Spectrum comparison")
    plt.grid(color='lightgrey',lw=0.5)


    
