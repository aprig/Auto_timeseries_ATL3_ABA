
# Author: Arthur Prigent
# Email: aprigent@geomar.de

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import scipy.stats as stats
from datetime import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import cartopy.crs as ccrs
import cartopy
import matplotlib.patches as mpatches
now = datetime.now()
date_time = now.strftime("%d/%m/%Y")
import matplotlib

def nandetrend(y):
    ''' Remove the linear trend from the data '''
    
    x = np.arange(0,y.shape[0],1)
    m, b, r_val, p_val, std_err = stats.linregress(x,np.array(y))
    y_detrended= np.array(y) - m*x -b
    return y_detrended


def ano_norm_t(ds):
    
    '''Compute the anomalies by removing the monthly means. 
    The anomalies are normalized by their corresponding month.
    
    Parameters
    ----------
    
    ds : xarray_like
    Timeserie or 3d field.
    
    Returns
    -----------
    
    ano : xarray_like
    Returns the anomalies of var relative the climatology.
    
    ano_norm : xarray_like
    Returns the anomalies of var relative the climatology normalized by the standard deviation.
    
    '''    

    clim     = ds.groupby('time.month').mean('time')
    clim_std = ds.groupby('time.month').std('time')
    ano      = ds.groupby('time.month') - clim
    ano_norm = xr.apply_ufunc(lambda x, m, s: (x - m) / s,
                                    ds.groupby('time.month'),
                                    clim, clim_std)
    
    return ano, ano_norm 



def ano_norm_t_wk(ds):
    
    '''Compute the anomalies by removing the monthly means. 
    The anomalies are normalized by their corresponding month.
    
    Parameters
    ----------
    
    ds : xarray_like
    Timeserie or 3d field.
    
    Returns
    -----------
    
    ano : xarray_like
    Returns the anomalies of var relative the climatology.
    
    ano_norm : xarray_like
    Returns the anomalies of var relative the climatology normalized by the standard deviation.
    
    '''    

    clim     = ds.groupby('time.week').mean('time')
    clim_std = ds.groupby('time.week').std('time')
    ano      = ds.groupby('time.week') - clim
    ano_norm = xr.apply_ufunc(lambda x, m, s: (x - m) / s,
                                    ds.groupby('time.week'),
                                    clim, clim_std)
    
    return ano, ano_norm 


def read_data_compute_anomalies_week(path_data):
    
    ds = xr.open_dataset(path_data+'sst.wkmean.1990-present.nc',engine='pydap')
    mask = xr.open_dataset(path_data+'lsmask.nc',engine='pydap')
    ds = ds.sst.where(mask.mask[0,:,:]==1)
    sst= ds.sel(time=slice(datetime(1990, 1, 1), now))
    sst = xr.concat([sst[:, :, 180:], sst[:, :, :180]], dim='lon')
    sst.coords['lon'] = (sst.coords['lon'] + 180) % 360 - 180  
    
    ## Make sub areas ##
    sst_atl3 = sst.where((  sst.lon>=-20) & (sst.lon<=0) &
                           (sst.lat<=3) & (sst.lat>=-3),drop=True).mean(dim='lon').mean(dim='lat')
    sst_nino34 = sst.where(( sst.lon>=-170) & (sst.lon<=-120) &
                           (sst.lat<=5) & (sst.lat>=-5),drop=True).mean(dim='lon').mean(dim='lat')
    sst_aba = sst.where((  sst.lon>=8) & (sst.lon<=16) &
                           (sst.lat<=-10) & (sst.lat>=-20),drop=True).mean(dim='lon').mean(dim='lat')
    sst_dni = sst.where((  sst.lon>=-21) & (sst.lon<=-17) &
                           (sst.lat<=17) & (sst.lat>=9),drop=True).mean(dim='lon').mean(dim='lat')
    sst_cni = sst.where((  sst.lon>=-120) & (sst.lon<=-110) &
                           (sst.lat<=30) & (sst.lat>=20),drop=True).mean(dim='lon').mean(dim='lat')
    sst_nni = sst.where((  sst.lon>=108) & (sst.lon<=115) &
                           (sst.lat<=-22) & (sst.lat>=-28),drop=True).mean(dim='lon').mean(dim='lat')
    
   
    ## Linearly detrend the data ## 
    sst_atl3 = sst_atl3.assign_coords(sst_dtd=('time',  nandetrend(sst_atl3.values)))
    sst_nino34 = sst_nino34.assign_coords(sst_dtd=('time',  nandetrend(sst_nino34.values)))
    sst_aba = sst_aba.assign_coords(sst_dtd=('time',  nandetrend(sst_aba.values)))
    sst_dni = sst_dni.assign_coords(sst_dtd=('time',  nandetrend(sst_dni.values)))
    sst_cni = sst_cni.assign_coords(sst_dtd=('time',  nandetrend(sst_cni.values)))
    sst_nni = sst_nni.assign_coords(sst_dtd=('time',  nandetrend(sst_nni.values)))

    ## Compute the SST anomalies ## 

    
    ssta_atl3,ssta_atl3_norm = ano_norm_t_wk(sst_atl3.sst_dtd.load())
    ssta_nino34,ssta_nino34_norm = ano_norm_t_wk(sst_nino34.sst_dtd.load())
    ssta_aba,ssta_aba_norm = ano_norm_t_wk(sst_aba.sst_dtd.load())
    ssta_dni,ssta_dni_norm = ano_norm_t_wk(sst_dni.sst_dtd.load())
    ssta_cni,ssta_cni_norm = ano_norm_t_wk(sst_cni.sst_dtd.load())
    ssta_nni,ssta_nni_norm = ano_norm_t_wk(sst_nni.sst_dtd.load())
    
    
    return ssta_atl3_norm,ssta_aba_norm,ssta_nino34_norm,ssta_dni_norm,ssta_cni_norm,ssta_nni_norm


def read_data_compute_anomalies_oi(path_data):
    
    ds = xr.open_dataset(path_data+'sst.mnmean.nc',engine='pydap')
    mask = xr.open_dataset(path_data+'lsmask.nc',engine='pydap')
    ds = ds.sst.where(mask.mask[0,:,:]==1)
    sst= ds.sel(time=slice(datetime(1982, 1, 1), now))
    sst = xr.concat([sst[:, :, 180:], sst[:, :, :180]], dim='lon')
    sst.coords['lon'] = (sst.coords['lon'] + 180) % 360 - 180  
    
    
    ## Make sub areas ##
    sst_atl3 = sst.where((  sst.lon>=-20) & (sst.lon<=0) &
                           (sst.lat<=3) & (sst.lat>=-3),drop=True).mean(dim='lon').mean(dim='lat')
    sst_nino34 = sst.where(( sst.lon>=-170) & (sst.lon<=-120) &
                           (sst.lat<=5) & (sst.lat>=-5),drop=True).mean(dim='lon').mean(dim='lat')
    sst_aba = sst.where((  sst.lon>=8) & (sst.lon<=16) &
                           (sst.lat<=-10) & (sst.lat>=-20),drop=True).mean(dim='lon').mean(dim='lat')
    sst_dni = sst.where((  sst.lon>=-21) & (sst.lon<=-17) &
                           (sst.lat<=17) & (sst.lat>=9),drop=True).mean(dim='lon').mean(dim='lat')
    sst_cni = sst.where((  sst.lon>=-120) & (sst.lon<=-110) &
                           (sst.lat<=30) & (sst.lat>=20),drop=True).mean(dim='lon').mean(dim='lat')
    sst_nni = sst.where((  sst.lon>=108) & (sst.lon<=115) &
                           (sst.lat<=-22) & (sst.lat>=-28),drop=True).mean(dim='lon').mean(dim='lat')
    
   
    ## Linearly detrend the data ## 
    sst_atl3 = sst_atl3.assign_coords(sst_dtd=('time',  nandetrend(sst_atl3.values)))
    sst_nino34 = sst_nino34.assign_coords(sst_dtd=('time',  nandetrend(sst_nino34.values)))
    sst_aba = sst_aba.assign_coords(sst_dtd=('time',  nandetrend(sst_aba.values)))
    sst_dni = sst_dni.assign_coords(sst_dtd=('time',  nandetrend(sst_dni.values)))
    sst_cni = sst_cni.assign_coords(sst_dtd=('time',  nandetrend(sst_cni.values)))
    sst_nni = sst_nni.assign_coords(sst_dtd=('time',  nandetrend(sst_nni.values)))

    ## Compute the SST anomalies ## 

    
    ssta_atl3,ssta_atl3_norm = ano_norm_t(sst_atl3.sst_dtd.load())
    ssta_nino34,ssta_nino34_norm = ano_norm_t(sst_nino34.sst_dtd.load())
    ssta_aba,ssta_aba_norm = ano_norm_t(sst_aba.sst_dtd.load())
    ssta_dni,ssta_dni_norm = ano_norm_t(sst_dni.sst_dtd.load())
    ssta_cni,ssta_cni_norm = ano_norm_t(sst_cni.sst_dtd.load())
    ssta_nni,ssta_nni_norm = ano_norm_t(sst_nni.sst_dtd.load())
    
    
    return ssta_atl3_norm,ssta_aba_norm,ssta_nino34_norm,ssta_dni_norm,ssta_cni_norm,ssta_nni_norm


def read_data_compute_anomalies(path_data):
    
    ds = xr.open_dataset(path_data,engine='pydap')
    sst= ds.sst.sel(time=slice(datetime(1982, 1, 1), now))
    sst = xr.concat([sst[:, :, 180:], sst[:, :, :180]], dim='lon')
    sst.coords['lon'] = (sst.coords['lon'] + 180) % 360 - 180  
    
    
    ## Make sub areas ##
    sst_atl3 = sst.where((  sst.lon>=-20) & (sst.lon<=0) &
                           (sst.lat<=3) & (sst.lat>=-3),drop=True).mean(dim='lon').mean(dim='lat')
    sst_nino34 = sst.where(( sst.lon>=-170) & (sst.lon<=-120) &
                           (sst.lat<=5) & (sst.lat>=-5),drop=True).mean(dim='lon').mean(dim='lat')
    sst_aba = sst.where((  sst.lon>=8) & (sst.lon<=16) &
                           (sst.lat<=-10) & (sst.lat>=-20),drop=True).mean(dim='lon').mean(dim='lat')
    sst_dni = sst.where((  sst.lon>=-21) & (sst.lon<=-17) &
                           (sst.lat<=17) & (sst.lat>=9),drop=True).mean(dim='lon').mean(dim='lat')
    sst_cni = sst.where((  sst.lon>=-120) & (sst.lon<=-110) &
                           (sst.lat<=30) & (sst.lat>=20),drop=True).mean(dim='lon').mean(dim='lat')
    sst_nni = sst.where((  sst.lon>=108) & (sst.lon<=115) &
                           (sst.lat<=-22) & (sst.lat>=-28),drop=True).mean(dim='lon').mean(dim='lat')
    
   
    ## Linearly detrend the data ## 
    sst_atl3 = sst_atl3.assign_coords(sst_dtd=('time',  nandetrend(sst_atl3.values)))
    sst_nino34 = sst_nino34.assign_coords(sst_dtd=('time',  nandetrend(sst_nino34.values)))
    sst_aba = sst_aba.assign_coords(sst_dtd=('time',  nandetrend(sst_aba.values)))
    sst_dni = sst_dni.assign_coords(sst_dtd=('time',  nandetrend(sst_dni.values)))
    sst_cni = sst_cni.assign_coords(sst_dtd=('time',  nandetrend(sst_cni.values)))
    sst_nni = sst_nni.assign_coords(sst_dtd=('time',  nandetrend(sst_nni.values)))

    ## Compute the SST anomalies ## 

    
    ssta_atl3,ssta_atl3_norm = ano_norm_t(sst_atl3.sst_dtd.load())
    ssta_nino34,ssta_nino34_norm = ano_norm_t(sst_nino34.sst_dtd.load())
    ssta_aba,ssta_aba_norm = ano_norm_t(sst_aba.sst_dtd.load())
    ssta_dni,ssta_dni_norm = ano_norm_t(sst_dni.sst_dtd.load())
    ssta_cni,ssta_cni_norm = ano_norm_t(sst_cni.sst_dtd.load())
    ssta_nni,ssta_nni_norm = ano_norm_t(sst_nni.sst_dtd.load())
    
    
    return ssta_atl3_norm,ssta_aba_norm,ssta_nino34_norm,ssta_dni_norm,ssta_cni_norm,ssta_nni_norm


def read_data_compute_anomalies_ersstv5(path_data):
    
    ds = xr.open_dataset(path_data,engine='pydap')
    sst= ds.sst.sel(time=slice(datetime(1982, 1, 1), now))
    sst = xr.concat([sst[:, :, 90:], sst[:, :, :90]], dim='lon')
    sst.coords['lon'] = (sst.coords['lon'] + 180) % 360 - 180  
    
    ## Make sub areas ##
    sst_atl3 = sst.where((  sst.lon>=-20) & (sst.lon<=0) &
                           (sst.lat<=3) & (sst.lat>=-3),drop=True).mean(dim='lon').mean(dim='lat')
    sst_nino34 = sst.where(( sst.lon>=-170) & (sst.lon<=-120) &
                           (sst.lat<=5) & (sst.lat>=-5),drop=True).mean(dim='lon').mean(dim='lat')
    sst_aba = sst.where((  sst.lon>=8) & (sst.lon<=16) &
                           (sst.lat<=-10) & (sst.lat>=-20),drop=True).mean(dim='lon').mean(dim='lat')
    sst_dni = sst.where((  sst.lon>=-21) & (sst.lon<=-17) &
                           (sst.lat<=17) & (sst.lat>=9),drop=True).mean(dim='lon').mean(dim='lat')
    sst_cni = sst.where((  sst.lon>=-120) & (sst.lon<=-110) &
                           (sst.lat<=30) & (sst.lat>=20),drop=True).mean(dim='lon').mean(dim='lat')
    sst_nni = sst.where((  sst.lon>=108) & (sst.lon<=115) &
                           (sst.lat<=-22) & (sst.lat>=-28),drop=True).mean(dim='lon').mean(dim='lat')
    
   
    ## Linearly detrend the data ## 
    sst_atl3 = sst_atl3.assign_coords(sst_dtd=('time',  nandetrend(sst_atl3.values)))
    sst_nino34 = sst_nino34.assign_coords(sst_dtd=('time',  nandetrend(sst_nino34.values)))
    sst_aba = sst_aba.assign_coords(sst_dtd=('time',  nandetrend(sst_aba.values)))
    sst_dni = sst_dni.assign_coords(sst_dtd=('time',  nandetrend(sst_dni.values)))
    sst_cni = sst_cni.assign_coords(sst_dtd=('time',  nandetrend(sst_cni.values)))
    sst_nni = sst_nni.assign_coords(sst_dtd=('time',  nandetrend(sst_nni.values)))

    ## Compute the SST anomalies ## 

    
    ssta_atl3,ssta_atl3_norm = ano_norm_t(sst_atl3.sst_dtd.load())
    ssta_nino34,ssta_nino34_norm = ano_norm_t(sst_nino34.sst_dtd.load())
    ssta_aba,ssta_aba_norm = ano_norm_t(sst_aba.sst_dtd.load())
    ssta_dni,ssta_dni_norm = ano_norm_t(sst_dni.sst_dtd.load())
    ssta_cni,ssta_cni_norm = ano_norm_t(sst_cni.sst_dtd.load())
    ssta_nni,ssta_nni_norm = ano_norm_t(sst_nni.sst_dtd.load())
    
    
    return ssta_atl3_norm,ssta_aba_norm,ssta_nino34_norm,ssta_dni_norm,ssta_cni_norm,ssta_nni_norm

def plot_anomalies(ssta_atl3,ssta_aba,ssta_nino34,ssta_dni,ssta_cni,ssta_nni):
    
    f,ax = plt.subplots(6,1,figsize=[15,30])
    color_lines='grey'
    ftz=15
    ax=ax.ravel()

    
    ### ATL3 ###
    ax[0].axhline(0,color=color_lines)
    ax[0].axhline(1,color=color_lines,linestyle='--')
    ax[0].axhline(-1,color=color_lines,linestyle='--')
    ax[0].plot(ssta_atl3.time.values,ssta_atl3,color='black')
    years = mdates.YearLocator(5)   # every year
    years_minor = mdates.YearLocator(1)  # every month
    ax[0].xaxis.set_major_locator(years)
    ax[0].xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax[0].xaxis.set_major_formatter(myFmt)
    ax[0].tick_params(labelsize=ftz)
    ax[0].fill_between(ssta_atl3.time.values,ssta_atl3,1,ssta_atl3>1,color='red')
    ax[0].fill_between(ssta_atl3.time.values,ssta_atl3,-1,ssta_atl3<-1,color='blue')
    ax[0].set_title('Normalized SST anomalies ATL3 [20$^{\circ}$W-0; 3$^{\circ}$S-3$^{\circ}$N] | Baseline '+
                    str(ssta_atl3.time.values[0])[:7] +' --> '+
                    str(ssta_atl3.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[0].text(0.01,0.04,'Updated '+date_time,transform=ax[0].transAxes,
           size=ftz,
           weight='bold')
    ax[0].set_ylim([-3,3])
    
    
    ### ABA ###
    ax[1].set_title(
        'Normalized SST anomalies ABA [8$^{\circ}$E-16$^{\circ}$E; 20$^{\circ}$S-10$^{\circ}$S] | Baseline '+
                    str(ssta_aba.time.values[0])[:7] +' --> '+
                    str(ssta_aba.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[1].plot(ssta_aba.time,ssta_aba,color='black')
    ax[1].axhline(0,color=color_lines)
    ax[1].axhline(1,color=color_lines,linestyle='--')
    ax[1].axhline(-1,color=color_lines,linestyle='--')
    ax[1].text(0.01,0.04,'Updated '+date_time,transform=ax[1].transAxes,
           size=ftz,
           weight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax[1].xaxis.set_major_locator(years)
    ax[1].xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax[1].xaxis.set_major_formatter(myFmt)
    ax[1].tick_params(labelsize=ftz)
    ax[1].fill_between(ssta_aba.time.values,ssta_aba,1,ssta_aba>1,color='red')
    ax[1].fill_between(ssta_aba.time.values,ssta_aba,-1,ssta_aba<-1,color='blue')
    ax[1].set_ylim([-3,3])
    
    ### NINO 3.4 ###
    ax[3].set_title(
        'Normalized SST anomalies NINO3.4 [170$^{\circ}$W-120$^{\circ}$W; 5$^{\circ}$S-5$^{\circ}$N] | Baseline '+
                    str(ssta_aba.time.values[0])[:7] +' --> '+
                    str(ssta_aba.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[3].plot(ssta_nino34.time,ssta_nino34,color='black')
    ax[3].axhline(0,color=color_lines)
    ax[3].axhline(1,color=color_lines,linestyle='--')
    ax[3].axhline(-1,color=color_lines,linestyle='--')
    ax[3].text(0.01,0.04,'Updated '+date_time,transform=ax[3].transAxes,
           size=ftz,
           weight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax[3].xaxis.set_major_locator(years)
    ax[3].xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax[3].xaxis.set_major_formatter(myFmt)
    ax[3].tick_params(labelsize=ftz)
    ax[3].fill_between(ssta_nino34.time.values,ssta_nino34,1,ssta_nino34>1,color='red')
    ax[3].fill_between(ssta_nino34.time.values,ssta_nino34,-1,ssta_nino34<-1,color='blue')
    ax[3].set_ylim([-3,3])
    
    ### DNI ###
    ax[2].set_title(
        'Normalized SST anomalies DNI [17$^{\circ}$W-21$^{\circ}$W; 9$^{\circ}$N-14$^{\circ}$N] | Baseline '+
                    str(ssta_dni.time.values[0])[:7] +' --> '+
                    str(ssta_dni.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[2].plot(ssta_dni.time,ssta_dni,color='black')
    ax[2].axhline(0,color=color_lines)
    ax[2].axhline(1,color=color_lines,linestyle='--')
    ax[2].axhline(-1,color=color_lines,linestyle='--')
    ax[2].text(0.01,0.04,'Updated '+date_time,transform=ax[2].transAxes,
           size=ftz,
           weight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax[2].xaxis.set_major_locator(years)
    ax[2].xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax[2].xaxis.set_major_formatter(myFmt)
    ax[2].tick_params(labelsize=ftz)
    ax[2].fill_between(ssta_dni.time.values,ssta_dni,1,ssta_dni>1,color='red')
    ax[2].fill_between(ssta_dni.time.values,ssta_dni,-1,ssta_dni<-1,color='blue')
    ax[2].set_ylim([-3,3])
    
    ### CNI ###
    ax[4].set_title(
        'Normalized SST anomalies CNI [110$^{\circ}$W-120$^{\circ}$W; 20$^{\circ}$N-30$^{\circ}$N] | Baseline '+
                    str(ssta_cni.time.values[0])[:7] +' --> '+
                    str(ssta_cni.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[4].plot(ssta_cni.time,ssta_cni,color='black')
    ax[4].axhline(0,color=color_lines)
    ax[4].axhline(1,color=color_lines,linestyle='--')
    ax[4].axhline(-1,color=color_lines,linestyle='--')
    ax[4].text(0.01,0.04,'Updated '+date_time,transform=ax[4].transAxes,
           size=ftz,
           weight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax[4].xaxis.set_major_locator(years)
    ax[4].xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax[4].xaxis.set_major_formatter(myFmt)
    ax[4].tick_params(labelsize=ftz)
    ax[4].fill_between(ssta_cni.time.values,ssta_cni,1,ssta_cni>1,color='red')
    ax[4].fill_between(ssta_cni.time.values,ssta_cni,-1,ssta_cni<-1,color='blue')
    ax[4].set_ylim([-3,3])
    
    ### NNI ###
    ax[5].set_title(
        'Normalized SST anomalies NNI [108$^{\circ}$E-115$^{\circ}$E; 28$^{\circ}$S-22$^{\circ}$N] | Baseline '+
                    str(ssta_nni.time.values[0])[:7] +' --> '+
                    str(ssta_nni.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[5].plot(ssta_nni.time,ssta_nni,color='black')
    ax[5].axhline(0,color=color_lines)
    ax[5].axhline(1,color=color_lines,linestyle='--')
    ax[5].axhline(-1,color=color_lines,linestyle='--')
    ax[5].text(0.01,0.04,'Updated '+date_time,transform=ax[5].transAxes,
           size=ftz,
           weight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax[5].xaxis.set_major_locator(years)
    ax[5].xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax[5].xaxis.set_major_formatter(myFmt)
    ax[5].tick_params(labelsize=ftz)
    ax[5].fill_between(ssta_nni.time.values,ssta_nni,1,ssta_nni>1,color='red')
    ax[5].fill_between(ssta_nni.time.values,ssta_nni,-1,ssta_nni<-1,color='blue')
    ax[5].set_ylim([-3,3])
    
def plot_anomalies_wk_aba(ssta_aba):
    
    f,ax = plt.subplots(2,1,figsize=[15,10])
    color_lines='grey'
    ftz=15
    ax=ax.ravel()
    ssta_aba_1 = ssta_aba.sel(time=slice(datetime(1990, 1, 1), datetime(2004, 12, 1)))
    ssta_aba_2 = ssta_aba.sel(time=slice(datetime(2005, 1, 1), now))
    ### ABA ###
    ax[0].set_title(
        'Normalized SST anomalies ABA [8$^{\circ}$E-16$^{\circ}$E; 20$^{\circ}$S-10$^{\circ}$S] | Baseline '+
                    str(ssta_aba.time.values[0])[:7] +' --> '+
                    str(ssta_aba.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[0].plot(ssta_aba_1.time,ssta_aba_1,color='black')
    ax[0].axhline(0,color=color_lines)
    ax[0].axhline(1,color=color_lines,linestyle='--')
    ax[0].axhline(-1,color=color_lines,linestyle='--')
    ax[0].text(0.01,0.04,'Updated '+date_time,transform=ax[0].transAxes,
           size=ftz,
           weight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax[0].xaxis.set_major_locator(years)
    ax[0].xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax[0].xaxis.set_major_formatter(myFmt)
    ax[0].tick_params(labelsize=ftz)
    ax[0].fill_between(ssta_aba_1.time.values,ssta_aba_1,1,ssta_aba_1>1,color='red')
    ax[0].fill_between(ssta_aba_1.time.values,ssta_aba_1,-1,ssta_aba_1<-1,color='blue')
    ax[0].set_ylim([-3,3])
    
    

    ### ABA ###
    ax[1].set_title(
        'Normalized SST anomalies ABA [8$^{\circ}$E-16$^{\circ}$E; 20$^{\circ}$S-10$^{\circ}$S] | Baseline '+
                    str(ssta_aba.time.values[0])[:7] +
                    ' --> '+str(ssta_aba.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[1].plot(ssta_aba_2.time,ssta_aba_2,color='black')
    ax[1].axhline(0,color=color_lines)
    ax[1].axhline(1,color=color_lines,linestyle='--')
    ax[1].axhline(-1,color=color_lines,linestyle='--')
    ax[1].text(0.01,0.04,'Updated '+date_time,transform=ax[1].transAxes,
           size=ftz,
           weight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax[1].xaxis.set_major_locator(years)
    ax[1].xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax[1].xaxis.set_major_formatter(myFmt)
    ax[1].tick_params(labelsize=ftz)
    ax[1].fill_between(ssta_aba_2.time.values,ssta_aba_2,1,ssta_aba_2>1,color='red')
    ax[1].fill_between(ssta_aba_2.time.values,ssta_aba_2,-1,ssta_aba_2<-1,color='blue')
    ax[1].set_ylim([-3,3])
    
    
    
    
    
def plot_anomalies_wk_atl3(ssta_atl3):
    
    f,ax = plt.subplots(2,1,figsize=[15,10])
    color_lines='grey'
    ftz=15
    ax=ax.ravel()
    ssta_atl3_1 = ssta_atl3.sel(time=slice(datetime(1990, 1, 1), datetime(2004, 12, 1)))
    ssta_atl3_2 = ssta_atl3.sel(time=slice(datetime(2005, 1, 1), now))
    ### ABA ###
    ax[0].set_title('Normalized SST anomalies ATL3 [20$^{\circ}$W-0; 3$^{\circ}$S-3$^{\circ}$N] | Baseline '+
                    str(ssta_atl3.time.values[0])[:7] +' --> '+
                    str(ssta_atl3.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[0].plot(ssta_atl3_1.time,ssta_atl3_1,color='black')
    ax[0].axhline(0,color=color_lines)
    ax[0].axhline(1,color=color_lines,linestyle='--')
    ax[0].axhline(-1,color=color_lines,linestyle='--')
    ax[0].text(0.01,0.04,'Updated '+date_time,transform=ax[0].transAxes,
           size=ftz,
           weight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax[0].xaxis.set_major_locator(years)
    ax[0].xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax[0].xaxis.set_major_formatter(myFmt)
    ax[0].tick_params(labelsize=ftz)
    ax[0].fill_between(ssta_atl3_1.time.values,ssta_atl3_1,1,ssta_atl3_1>1,color='red')
    ax[0].fill_between(ssta_atl3_1.time.values,ssta_atl3_1,-1,ssta_atl3_1<-1,color='blue')
    ax[0].set_ylim([-3,3])
    
    

    ### ABA ###
    ax[1].set_title('Normalized SST anomalies ATL3 [20$^{\circ}$W-0; 3$^{\circ}$S-3$^{\circ}$N] | Baseline '+
                    str(ssta_atl3.time.values[0])[:7] +' --> '+
                    str(ssta_atl3.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[1].plot(ssta_atl3_2.time,ssta_atl3_2,color='black')
    ax[1].axhline(0,color=color_lines)
    ax[1].axhline(1,color=color_lines,linestyle='--')
    ax[1].axhline(-1,color=color_lines,linestyle='--')
    ax[1].text(0.01,0.04,'Updated '+date_time,transform=ax[1].transAxes,
           size=ftz,
           weight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax[1].xaxis.set_major_locator(years)
    ax[1].xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax[1].xaxis.set_major_formatter(myFmt)
    ax[1].tick_params(labelsize=ftz)
    ax[1].fill_between(ssta_atl3_2.time.values,ssta_atl3_2,1,ssta_atl3_2>1,color='red')
    ax[1].fill_between(ssta_atl3_2.time.values,ssta_atl3_2,-1,ssta_atl3_2<-1,color='blue')
    ax[1].set_ylim([-3,3])
    
    
def plot_anomalies_wk_nino34(ssta_nino34):
    
    f,ax = plt.subplots(2,1,figsize=[15,10])
    color_lines='grey'
    ftz=15
    ax=ax.ravel()
    ssta_nino34_1 = ssta_nino34.sel(time=slice(datetime(1990, 1, 1), datetime(2004, 12, 1)))
    ssta_nino34_2 = ssta_nino34.sel(time=slice(datetime(2005, 1, 1), now))
    ### NINO3.4 ###
    ax[0].set_title('Normalized SST anomalies NINO3.4 [170$^{\circ}$W-120$^{\circ}$W; 5$^{\circ}$S-5$^{\circ}$N] | Baseline '+
                    str(ssta_nino34.time.values[0])[:7] +' --> '+
                    str(ssta_nino34.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[0].plot(ssta_nino34_1.time,ssta_nino34_1,color='black')
    ax[0].axhline(0,color=color_lines)
    ax[0].axhline(1,color=color_lines,linestyle='--')
    ax[0].axhline(-1,color=color_lines,linestyle='--')
    ax[0].text(0.01,0.04,'Updated '+date_time,transform=ax[0].transAxes,
           size=ftz,
           weight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax[0].xaxis.set_major_locator(years)
    ax[0].xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax[0].xaxis.set_major_formatter(myFmt)
    ax[0].tick_params(labelsize=ftz)
    ax[0].fill_between(ssta_nino34_1.time.values,ssta_nino34_1,1,ssta_nino34_1>1,color='red')
    ax[0].fill_between(ssta_nino34_1.time.values,ssta_nino34_1,-1,ssta_nino34_1<-1,color='blue')
    ax[0].set_ylim([-3,3])
    
    

    ### NINO3.4 ###
    ax[1].set_title('Normalized SST anomalies NINO3.4 [170$^{\circ}$W-120$^{\circ}$W; 5$^{\circ}$S-5$^{\circ}$N] | Baseline '+
                    str(ssta_nino34.time.values[0])[:7] +' --> '+
                    str(ssta_nino34.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[1].plot(ssta_nino34_2.time,ssta_nino34_2,color='black')
    ax[1].axhline(0,color=color_lines)
    ax[1].axhline(1,color=color_lines,linestyle='--')
    ax[1].axhline(-1,color=color_lines,linestyle='--')
    ax[1].text(0.01,0.04,'Updated '+date_time,transform=ax[1].transAxes,
           size=ftz,
           weight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax[1].xaxis.set_major_locator(years)
    ax[1].xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax[1].xaxis.set_major_formatter(myFmt)
    ax[1].tick_params(labelsize=ftz)
    ax[1].fill_between(ssta_nino34_2.time.values,ssta_nino34_2,1,ssta_nino34_2>1,color='red')
    ax[1].fill_between(ssta_nino34_2.time.values,ssta_nino34_2,-1,ssta_nino34_2<-1,color='blue')
    ax[1].set_ylim([-3,3])
    
def plot_anomalies_wk_dni(ssta_dni):
    
    f,ax = plt.subplots(2,1,figsize=[15,10])
    color_lines='grey'
    ftz=15
    ax=ax.ravel()
    ssta_dni_1 = ssta_dni.sel(time=slice(datetime(1990, 1, 1), datetime(2004, 12, 1)))
    ssta_dni_2 = ssta_dni.sel(time=slice(datetime(2005, 1, 1), now))
    ### NINO3.4 ###
    ax[0].set_title('Normalized SST anomalies DNI [17$^{\circ}$W-21$^{\circ}$W; 9$^{\circ}$N-14$^{\circ}$N] | Baseline '+
                    str(ssta_dni.time.values[0])[:7] +' --> '+
                    str(ssta_dni.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[0].plot(ssta_dni_1.time,ssta_dni_1,color='black')
    ax[0].axhline(0,color=color_lines)
    ax[0].axhline(1,color=color_lines,linestyle='--')
    ax[0].axhline(-1,color=color_lines,linestyle='--')
    ax[0].text(0.01,0.04,'Updated '+date_time,transform=ax[0].transAxes,
           size=ftz,
           weight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax[0].xaxis.set_major_locator(years)
    ax[0].xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax[0].xaxis.set_major_formatter(myFmt)
    ax[0].tick_params(labelsize=ftz)
    ax[0].fill_between(ssta_dni_1.time.values,ssta_dni_1,1,ssta_dni_1>1,color='red')
    ax[0].fill_between(ssta_dni_1.time.values,ssta_dni_1,-1,ssta_dni_1<-1,color='blue')
    ax[0].set_ylim([-3,3])
    
    

    ### NINO3.4 ###
    ax[1].set_title('Normalized SST anomalies DNI [17$^{\circ}$W-21$^{\circ}$W; 9$^{\circ}$N-14$^{\circ}$N] | Baseline '+
                    str(ssta_dni.time.values[0])[:7] +' --> '+
                    str(ssta_dni.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[1].plot(ssta_dni_2.time,ssta_dni_2,color='black')
    ax[1].axhline(0,color=color_lines)
    ax[1].axhline(1,color=color_lines,linestyle='--')
    ax[1].axhline(-1,color=color_lines,linestyle='--')
    ax[1].text(0.01,0.04,'Updated '+date_time,transform=ax[1].transAxes,
           size=ftz,
           weight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax[1].xaxis.set_major_locator(years)
    ax[1].xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax[1].xaxis.set_major_formatter(myFmt)
    ax[1].tick_params(labelsize=ftz)
    ax[1].fill_between(ssta_dni_2.time.values,ssta_dni_2,1,ssta_dni_2>1,color='red')
    ax[1].fill_between(ssta_dni_2.time.values,ssta_dni_2,-1,ssta_dni_2<-1,color='blue')
    ax[1].set_ylim([-3,3]) 
    
    
def plot_anomalies_wk_nni(ssta_nni):
    
    f,ax = plt.subplots(2,1,figsize=[15,10])
    color_lines='grey'
    ftz=15
    ax=ax.ravel()
    ssta_nni_1 = ssta_nni.sel(time=slice(datetime(1990, 1, 1), datetime(2004, 12, 1)))
    ssta_nni_2 = ssta_nni.sel(time=slice(datetime(2005, 1, 1), now))
    ### NINO3.4 ###
    ax[0].set_title('Normalized SST anomalies NNI [108$^{\circ}$E-115$^{\circ}$E; 28$^{\circ}$S-22$^{\circ}$N] | Baseline '+
                    str(ssta_nni.time.values[0])[:7] +' --> '+
                    str(ssta_nni.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[0].plot(ssta_nni_1.time,ssta_nni_1,color='black')
    ax[0].axhline(0,color=color_lines)
    ax[0].axhline(1,color=color_lines,linestyle='--')
    ax[0].axhline(-1,color=color_lines,linestyle='--')
    ax[0].text(0.01,0.04,'Updated '+date_time,transform=ax[0].transAxes,
           size=ftz,
           weight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax[0].xaxis.set_major_locator(years)
    ax[0].xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax[0].xaxis.set_major_formatter(myFmt)
    ax[0].tick_params(labelsize=ftz)
    ax[0].fill_between(ssta_nni_1.time.values,ssta_nni_1,1,ssta_nni_1>1,color='red')
    ax[0].fill_between(ssta_nni_1.time.values,ssta_nni_1,-1,ssta_nni_1<-1,color='blue')
    ax[0].set_ylim([-3,3])
    
    

    ### NINO3.4 ###
    ax[1].set_title('Normalized SST anomalies NNI [108$^{\circ}$E-115$^{\circ}$E; 28$^{\circ}$S-22$^{\circ}$N] | Baseline '+
                    str(ssta_nni.time.values[0])[:7] +' --> '+
                    str(ssta_nni.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[1].plot(ssta_nni_2.time,ssta_nni_2,color='black')
    ax[1].axhline(0,color=color_lines)
    ax[1].axhline(1,color=color_lines,linestyle='--')
    ax[1].axhline(-1,color=color_lines,linestyle='--')
    ax[1].text(0.01,0.04,'Updated '+date_time,transform=ax[1].transAxes,
           size=ftz,
           weight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax[1].xaxis.set_major_locator(years)
    ax[1].xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax[1].xaxis.set_major_formatter(myFmt)
    ax[1].tick_params(labelsize=ftz)
    ax[1].fill_between(ssta_nni_2.time.values,ssta_nni_2,1,ssta_nni_2>1,color='red')
    ax[1].fill_between(ssta_nni_2.time.values,ssta_nni_2,-1,ssta_nni_2<-1,color='blue')
    ax[1].set_ylim([-3,3]) 
    
def plot_anomalies_wk_cni(ssta_cni):
    
    f,ax = plt.subplots(2,1,figsize=[15,10])
    color_lines='grey'
    ftz=15
    ax=ax.ravel()
    ssta_cni_1 = ssta_cni.sel(time=slice(datetime(1990, 1, 1), datetime(2004, 12, 1)))
    ssta_cni_2 = ssta_cni.sel(time=slice(datetime(2005, 1, 1), now))
    ### NINO3.4 ###
    ax[0].set_title('Normalized SST anomalies CNI [110$^{\circ}$W-120$^{\circ}$W; 20$^{\circ}$N-30$^{\circ}$N] | Baseline '+
                    str(ssta_cni.time.values[0])[:7] +' --> '+
                    str(ssta_cni.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[0].plot(ssta_cni_1.time,ssta_cni_1,color='black')
    ax[0].axhline(0,color=color_lines)
    ax[0].axhline(1,color=color_lines,linestyle='--')
    ax[0].axhline(-1,color=color_lines,linestyle='--')
    ax[0].text(0.01,0.04,'Updated '+date_time,transform=ax[0].transAxes,
           size=ftz,
           weight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax[0].xaxis.set_major_locator(years)
    ax[0].xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax[0].xaxis.set_major_formatter(myFmt)
    ax[0].tick_params(labelsize=ftz)
    ax[0].fill_between(ssta_cni_1.time.values,ssta_cni_1,1,ssta_cni_1>1,color='red')
    ax[0].fill_between(ssta_cni_1.time.values,ssta_cni_1,-1,ssta_cni_1<-1,color='blue')
    ax[0].set_ylim([-3,3])
    
    

    ### NINO3.4 ###
    ax[1].set_title('Normalized SST anomalies CNI [110$^{\circ}$W-120$^{\circ}$W; 20$^{\circ}$N-30$^{\circ}$N] | Baseline '+
                    str(ssta_cni.time.values[0])[:7] +' --> '+
                    str(ssta_cni.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[1].plot(ssta_cni_2.time,ssta_cni_2,color='black')
    ax[1].axhline(0,color=color_lines)
    ax[1].axhline(1,color=color_lines,linestyle='--')
    ax[1].axhline(-1,color=color_lines,linestyle='--')
    ax[1].text(0.01,0.04,'Updated '+date_time,transform=ax[1].transAxes,
           size=ftz,
           weight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax[1].xaxis.set_major_locator(years)
    ax[1].xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax[1].xaxis.set_major_formatter(myFmt)
    ax[1].tick_params(labelsize=ftz)
    ax[1].fill_between(ssta_cni_2.time.values,ssta_cni_2,1,ssta_cni_2>1,color='red')
    ax[1].fill_between(ssta_cni_2.time.values,ssta_cni_2,-1,ssta_cni_2<-1,color='blue')
    ax[1].set_ylim([-3,3]) 
    
import cartopy.crs as ccrs
import cartopy    
import matplotlib.patches as mpatches
def plot_regions_of_interest():
    f = plt.figure(figsize=[20,20])
    ftz=15
    minlon = -175
    maxlon = 120
    minlat = -35
    maxlat = 35
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND, edgecolor='black',color='lightgrey')
    ax.coastlines()
    ax.set_extent([minlon,maxlon,minlat,maxlat],ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='black', alpha=0.5, linestyle='-')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 15, 'color': 'black'}
    gl.ylabel_style = {'size': 15, 'color': 'black'}
    gl.xlocator = mticker.FixedLocator([-160,-140,-120,-100,-80,-60,-40,-20, 0])
    gl.ylocator = mticker.FixedLocator([-20, 0,20])
    ax.coastlines(linewidth=1)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black',color='lightgrey')
    ax.coastlines(resolution='50m', color='black', linewidth=1)

    ax.add_patch(mpatches.Rectangle(xy=[-20, -3], width=20, height=6,
                                        facecolor='blue',
                                        alpha=0.5,
                                        edgecolor='black',
                                        transform=ccrs.PlateCarree()))

    ax.text(-12.5, 0.8, 'ATL3',
             horizontalalignment='left',fontsize=ftz,fontweight='bold',
             transform=ccrs.PlateCarree())

    ax.add_patch(mpatches.Rectangle(xy=[8, -20], width=8, height=10,
                                        facecolor='red',
                                        edgecolor='black',
                                        alpha=0.5,
                                        transform=ccrs.PlateCarree()))

    ax.text(2, -15, 'ABA',
             horizontalalignment='left',fontsize=ftz,fontweight='bold',
             transform=ccrs.PlateCarree())
    
    ax.add_patch(mpatches.Rectangle(xy=[-170, -5], width=50, height=10,
                                        facecolor='green',
                                        edgecolor='black',
                                        alpha=0.5,
                                        transform=ccrs.PlateCarree()))

    ax.text(-150, 1, 'NINO3.4',
             horizontalalignment='left',fontsize=ftz,fontweight='bold',
             transform=ccrs.PlateCarree())
    
    
    ax.add_patch(mpatches.Rectangle(xy=[-21, 9], width=4, height=5,
                                        facecolor='orange',
                                        edgecolor='black',
                                        alpha=0.5,
                                        transform=ccrs.PlateCarree()))

    ax.text(-27, 10, 'DNI',
             horizontalalignment='left',fontsize=ftz,fontweight='bold',
             transform=ccrs.PlateCarree())
    
    ax.add_patch(mpatches.Rectangle(xy=[-120, 20], width=10, height=10,
                                        facecolor='grey',
                                        edgecolor='black',
                                        alpha=0.5,
                                        transform=ccrs.PlateCarree()))

    ax.text(-120, 15, 'CNI',
             horizontalalignment='left',fontsize=ftz,fontweight='bold',
             transform=ccrs.PlateCarree()) 
    
    
    ax.add_patch(mpatches.Rectangle(xy=[108, -28], width=7, height=6,
                                        facecolor='pink',
                                        edgecolor='black',
                                        alpha=0.5,
                                        transform=ccrs.PlateCarree()))

    ax.text(108, -32, 'NNI',
             horizontalalignment='left',fontsize=ftz,fontweight='bold',
             transform=ccrs.PlateCarree()) 
    

def read_data_compute_anomalies_week_map(path_data):
    
    ds = xr.open_dataset(path_data+'sst.wkmean.1990-present.nc',engine='pydap')
    mask = xr.open_dataset(path_data+'lsmask.nc',engine='pydap')
    ds = ds.sst.where(mask.mask[0,:,:]==1)
    sst= ds.sel(time=slice(datetime(1990, 1, 1), now))
    sst = xr.concat([sst[:, :, 180:], sst[:, :, :180]], dim='lon')
    sst.coords['lon'] = (sst.coords['lon'] + 180) % 360 - 180  
    
    ## Make sub areas ##
    sst_atl = sst.where((  sst.lon>=-45) & (sst.lon<=20) &
                           (sst.lat<=30) & (sst.lat>=-30),drop=True)
    sst_dtd_tmp = np.ones(sst_atl.shape)*np.nan
    for i in range(sst_dtd_tmp.shape[1]):
        for j in range(sst_dtd_tmp.shape[2]):
            sst_dtd_tmp[:,i,j] = nandetrend(sst_atl[:,i,j].values)
            
   
    ## Linearly detrend the data ## 
    sst_atl = sst_atl.assign_coords(sst_dtd=(['time','lat','lon'],  sst_dtd_tmp))

    ## Compute the SST anomalies ## 

    
    ssta_atl,ssta_atl_norm = ano_norm_t_wk(sst_atl.sst_dtd.load())
    return ssta_atl_norm


    
def plot_map_ssta_atl(ssta_atl_wk):
    f = plt.figure(figsize=[15,15])
    n=45
    x = 0.9
    ftz=15
    lower = plt.cm.Blues_r(np.linspace(0, x, n))
    white = np.ones((100-2*n,4))
    upper = plt.cm.Reds(np.linspace(1-x, 1, n))
    colors = np.vstack((lower, white, upper))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('terrain_map_white', colors)
    bounds= np.arange(-2.5,2.75,0.25)
    ftz=15
    minlon = -45
    maxlon = 30
    minlat = -30
    maxlat = 30
    ax = plt.axes(projection=ccrs.PlateCarree())
    cax = inset_axes(ax,
                   width="100%",  # width = 5% of parent_bbox width
                   height="5%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(0, -0.2, 1, 1),
                   bbox_transform=ax.transAxes,
                   borderpad=0,
                   )
    ax.add_feature(cartopy.feature.LAND, edgecolor='black',color='lightgrey')
    ax.coastlines()
    ax.set_extent([minlon,maxlon,minlat,maxlat],ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='black', alpha=0.5, linestyle='-')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 15, 'color': 'black'}
    gl.ylabel_style = {'size': 15, 'color': 'black'}
    gl.xlocator = mticker.FixedLocator([-40,-20, 0])
    gl.ylocator = mticker.FixedLocator([-20, 0,20])
    ax.coastlines(linewidth=1)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black',color='lightgrey')
    ax.coastlines(resolution='50m', color='black', linewidth=1)
    p0=ax.contourf(ssta_atl_wk.lon,
                ssta_atl_wk.lat,
                ssta_atl_wk.sst_dtd[-1,:,:],transform=ccrs.PlateCarree(),cmap=cmap,levels=bounds,extend='both')
    cbar = plt.colorbar(p0,cax,orientation='horizontal')
    cbar.ax.tick_params(labelsize=ftz)
    ax.set_title(str(ssta_atl_wk.time.values[-1])[:10],fontsize=ftz,fontweight='bold')
    
    ax.add_patch(mpatches.Rectangle(xy=[-20, -3], width=20, height=6,
                                        edgecolor='red',fill=None,alpha=1,linewidth=3,
                                        transform=ccrs.PlateCarree()))

    ax.text(-12.5, 0.8, 'ATL3',
             horizontalalignment='left',fontsize=ftz,fontweight='bold',
             transform=ccrs.PlateCarree())

    ax.add_patch(mpatches.Rectangle(xy=[8, -20], width=8, height=10,
                                        fill=None,
                                        edgecolor='green',alpha=1,linewidth=3,
                                        transform=ccrs.PlateCarree()))

    ax.text(2, -15, 'ABA',
             horizontalalignment='left',fontsize=ftz,fontweight='bold',
             transform=ccrs.PlateCarree())
    
    ax.add_patch(mpatches.Rectangle(xy=[-21, 9], width=4, height=5,
                                        edgecolor='grey',fill=None,alpha=1,linewidth=3,
                                        transform=ccrs.PlateCarree()))

    ax.text(-27, 10, 'DNI',
             horizontalalignment='left',fontsize=ftz,fontweight='bold',
             transform=ccrs.PlateCarree())