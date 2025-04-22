#package imports
# UTILS
import datetime as dt
from datetime import datetime,time,timedelta
import os
## PLOT
import matplotlib.pyplot as plt

import numpy as np


## ASTROPY
from astropy.io import fits
from astropy.time.core import Time, TimeDelta
from astropy.coordinates import SkyCoord
from sunpy.coordinates import get_body_heliographic_stonyhurst,HeliographicStonyhurst
import astropy.units as u



#Ancillary
def print_obs_info(header):
    # observation invterval
    try:
        obs_start=header['DATE_BEG']
        obs_end=header['DATE_END']


    except:
        obs_start=header['DATE-BEG']
        obs_end=header['DATE-END']

    # distance to sun and time delay w.r.t. to sapcecraft (s/c)
    dist_sun_sc=(header["DSUN_OBS"]*u.m).to(u.au)
    time_delay = header["SUN_TIME"]*u.s
    
    instrument = header["INSTRUME"]
    observatory = header["OBSRVTRY"]

    print(f"Observatory: {observatory}")
    print(f"Instrument: {instrument}")
    print(f"Observation time:\n from: {obs_start}\n to: {obs_end}")
    print(f"Distance s/c - sun: {dist_sun_sc}")
    print(f"Time delay s/c - sun: {time_delay}")
    
    
def plot_sc_position(header):
    try:
        obs_start=header['DATE_BEG']
    except:
        obs_start=header['DATE-BEG']
    observatory = header["OBSRVTRY"]
    obstime =Time(obs_start, format='fits') #fits time format = "%Y-%m-%dT%H:%M:%S"
    dist_sun_sc=(header["DSUN_OBS"]*u.m).to(u.au)

    #getting heliographic stonyhurst coordinates - longitude and latitude
    sc_lon,sc_lat=header["HGLN_OBS"]*u.deg, header["HGLT_OBS"]*u.deg
    solo_coords=HeliographicStonyhurst(sc_lon,sc_lat, dist_sun_sc, obstime=obs_start)
    #planets to display
    planet_list = ['earth',  'mercury','venus','sun']
    planet_coord = [get_body_heliographic_stonyhurst(this_planet, time=obstime) for this_planet in planet_list]
    #append satellite
    planet_list.append(observatory)
    planet_coord.append(solo_coords)
    
    # personalized color list
    color_list= ["green","brown","orange","yellow","magenta"]

    # PLOT - projection of positions over ecliptic plane
    fig = plt.figure(figsize=(5,5))
    ax1 = plt.subplot(1, 1, 1, projection='polar')
    # plot positions over the ecliptic plane in polar coordiantes
    for this_planet, this_coord,col in zip(planet_list, planet_coord,color_list):
        plt.plot(np.deg2rad(this_coord.lon), this_coord.radius, 'o', label=this_planet,c=col,markersize=10 if this_planet=="sun" else 3)
        plt.text(np.deg2rad(this_coord.lon.value), this_coord.radius.value,this_planet+"  ",
                 horizontalalignment='right',verticalalignment='center',fontweight="demibold")

    #Formatting observation time for title 
    obstime_formatted = (obstime.to_datetime()).strftime("%d-%b-%Y %H:%M")

    plt.title(observatory+" position for "+obstime_formatted,fontweight="demibold")
    plt.show()
