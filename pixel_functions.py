
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

import numpy as np


## ASTROPY
from astropy.io import fits
from astropy.time.core import Time, TimeDelta
from astropy.table import Table
import astropy.units as u


def get_tables_pixel(sci_file_path):
    hdulist = fits.open(sci_file_path)
    header = hdulist[0].header
    
    data = Table(hdulist[2].data)
    energies = Table(hdulist[4].data)
    
    
    energies["mean_energy"] = np.array([np.mean([e["e_low"],e["e_high"]]) if e["e_high"]!=np.inf else e["e_low"] for e in energies ])
    
    data["time"]= Time(header['date-obs']) + TimeDelta(data['time'] * u.cs)
    # we want an array with the same shape as the counts array
    cts_sec_array = np.zeros(np.shape(data["counts"]))

    # for each timestep (axis 0)
    for t in range(len(data["time"])):
        # the 3 dimesional counts matrix associated with the
        # timestep is divided by the timestep length in seconds
        time_del =data["timedel"][t]/100
        cts_sec_array[t,:,:,:] = data["counts"][t,:,:,:]/time_del
    # append count rate to the data Table
    data["cts_per_sec"] = cts_sec_array
    
    

    data["cts_per_sec_summed"] = np.sum(cts_sec_array,axis=(1,2))
    
    
    
    
    
    
    return data,energies,header

def remove_bkg_pixel(sci_file_path,bkg_file_path):
    sci_data_,sci_ene_,sci_head_=get_tables_pixel(sci_file_path)
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

    bkg_array = np.repeat(np.array(data_bkg["cts_per_sec"])[None, :], len(sci_data_["time"]), axis=1)
    corr_array =sci_data_["cts_per_sec_summed"]-bkg_array
    corr_array = np.clip(corr_array,0,np.inf)
    corr_array = corr_array.reshape( ( len(sci_data_["time"]) , n_energies_bkg ) )

    sci_data_["bkg_cts_per_sec"] =data_bkg["cts_per_sec"]
    sci_data_["corrected_cts_per_sec_summed"] = corr_array
    
    
    sci_data_["corrected_counts_summed"]= np.zeros(np.shape(corr_array))
    for i in range(np.shape(corr_array)[0]):
        sci_data_["corrected_counts_summed"][i,:] =  corr_array[i,:]*sci_data_["timedel"][i]
    
    return sci_data_,sci_ene_,sci_head_


#paint timeline markers
def paint_markers(ax,markers):
    if(markers):
        for mk in markers:
            m = dt.datetime.strptime(mk,"%Y-%m-%d %H:%M:%S")
            ax.axvline(m,c="r",ls="--")

    
def plot_spectrogram_pixel(data,energies,header,energy_range=[4,28],corrected=False,markers=None):
    # ACTIVITY using the quicklooks, deduce the energy interval for which this file has counts and define the energy range
    # The low range ususally is 4kev, the at high energies are the ones cropped out.
    plt.figure(figsize=(12,4))
    ax = plt.subplot(111)


    myFmt = mdates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(myFmt)


    cts_data = np.array(data['cts_per_sec_summed']).T if not corrected else np.array(data['corrected_cts_per_sec_summed']).T 
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
    
    
    



    
def plot_integrated_bins_pixel(data,energies,header,energy_bins=None,corrected=False,smooth_pts=None,markers=None):

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
        sel_counts = data["cts_per_sec_summed"][:,jj]
        # if "corrected" and corrected counts available, use instead
        if(corrected):
            sel_counts = data["corrected_cts_per_sec_summed"][:,jj]
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
    plt.legend()