
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
import hvplot.pandas  # noqa
import hvplot.xarray  # noqa
def nandetrend(y):
    ''' Remove the linear trend from the data '''
    
    x = np.arange(0,y.shape[0],1)
    m, b, r_val, p_val, std_err = stats.linregress(x,np.array(y))
    y_detrended= np.array(y) - m*x -b
    return y_detrended

def find_event(timeserie_ano_ABA_file,treshold):
    
    index_nino_sst = xr.full_like(timeserie_ano_ABA_file,np.nan)
    index_nino_sst[timeserie_ano_ABA_file >= treshold] = 1
    index_nino_sst[timeserie_ano_ABA_file <  treshold] = 0
    index_nino_sst_tmp = np.array(index_nino_sst)
    #print(index_nino_sst_tmp)
    index_nina_sst = xr.full_like(timeserie_ano_ABA_file,np.nan)
    index_nina_sst[timeserie_ano_ABA_file <= -treshold] = 1
    index_nina_sst[timeserie_ano_ABA_file > - treshold] = 0
    index_nina_sst_tmp = np.array(index_nina_sst)
    
    
    diff_indexes_nino = np.diff(index_nino_sst_tmp[:])
    diff_indexes_nina = np.diff(index_nina_sst_tmp[:])

    id_str_nino = []
    id_end_nino = []
    for i in range(len(diff_indexes_nino)):
        if diff_indexes_nino[i]==1.0:
            id_str_nino.append(i+1)
        elif diff_indexes_nino[i] == -1.0:
            id_end_nino.append(i+1)
            
    if id_str_nino[0]>id_end_nino[0]:
        id_end_nino.pop(0)

    if len(id_str_nino)> len(id_end_nino):
        id_str_nino.pop(-1)
    elif len(id_str_nino)< len(id_end_nino):
        id_end_nino.pop(-1)
    

        
    id_str_nina = []
    id_end_nina = []
    for i in range(len(diff_indexes_nina)):
        if diff_indexes_nina[i]==1.0:
            id_str_nina.append(i+1)
        elif diff_indexes_nina[i] == -1.0:
            id_end_nina.append(i+1)
    #print(id_str_nina)
    #print(id_end_nina)
    
    if id_str_nina[0]>id_end_nina[0]:
        id_end_nina.pop(0)

        
    if len(id_str_nina)> len(id_end_nina):
        id_str_nina.pop(-1)
    elif len(id_str_nina)< len(id_end_nina):
        id_end_nina.pop(-1)
        
    
        
    try:
        nino_indexes_tmp= np.vstack((id_str_nino,id_end_nino))
        length_events_nino = nino_indexes_tmp[1,:] - nino_indexes_tmp[0,:]
        nino_indexes = np.vstack((nino_indexes_tmp,length_events_nino))
    except ValueError:
        nino_indexes_tmp= np.vstack((id_str_nino,id_end_nino[:-1]))
        length_events_nino = nino_indexes_tmp[1,:] - nino_indexes_tmp[0,:]
        nino_indexes = np.vstack((nino_indexes_tmp,length_events_nino))
        
    try:
        nina_indexes_tmp= np.vstack((id_str_nina,id_end_nina[:]))
        length_events_nina = nina_indexes_tmp[1,:] - nina_indexes_tmp[0,:]
        nina_indexes = np.vstack((nina_indexes_tmp,length_events_nina))
    except ValueError:
        nina_indexes_tmp= np.vstack((id_str_nina,id_end_nina[1:]))
        length_events_nina = nina_indexes_tmp[1,:] - nina_indexes_tmp[0,:]
        nina_indexes = np.vstack((nina_indexes_tmp,length_events_nina))
    
    
    return np.array(nino_indexes), np.array(nina_indexes)


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

def create_table_event(ssta):
    warm,cold=find_event(ssta,1)
    data_table_warm = np.vstack((ssta[warm[0,warm[2,:]>3]].time,
                           ssta[warm[1,warm[2,:]>3]].time))
    df_warm = pd.DataFrame(data_table_warm.T, columns = ['Start date','End date'], 
                      )
    data_table_cold = np.vstack((ssta[cold[0,cold[2,:]>3]].time,
                           ssta[cold[1,cold[2,:]>3]].time))
    df_cold = pd.DataFrame(data_table_cold.T, columns = ['Start date','End date'])
        
    return warm,cold,df_warm,df_cold

def plot_anomalies(ssta_atl3,ssta_aba,ssta_nino34,ssta_dni,ssta_cni,ssta_nni):
    
    f,ax = plt.subplots(6,1,figsize=[15,30])
    color_lines='grey'
    ftz=15
    ax=ax.ravel()

    
    ### ATL3 ###
    index_warm,index_cold,_,_ = create_table_event(ssta_atl3)
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
    for i in range(index_warm.shape[1]):
        if index_warm[2,i]>=3:
            ax[0].fill_between(ssta_atl3.time.values[index_warm[0,i]:index_warm[1,i]],
                     ssta_atl3[index_warm[0,i]:index_warm[1,i]],1,
                             ssta_atl3[index_warm[0,i]:index_warm[1,i]]>1,color='red')


    for i in range(index_cold.shape[1]):
        if index_cold[2,i]>=3:
            ax[0].fill_between(ssta_atl3.time.values[index_cold[0,i]:index_cold[1,i]],
                     ssta_atl3[index_cold[0,i]:index_cold[1,i]],
                             -1,ssta_atl3[index_cold[0,i]:index_cold[1,i]]<-1,color='blue')


    
    ax[0].set_title('Normalized SST anomalies ATL3 [20$^{\circ}$W-0; 3$^{\circ}$S-3$^{\circ}$N] | Baseline '+
                    str(ssta_atl3.time.values[0])[:7] +' --> '+
                    str(ssta_atl3.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[0].text(0.01,0.04,'Updated '+date_time,transform=ax[0].transAxes,
           size=ftz,
           weight='bold')
    ax[0].set_ylim([-3,3])
    
    
    ### ABA ###
    index_warm,index_cold,_,_ = create_table_event(ssta_aba)
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
    for i in range(index_warm.shape[1]):
        if index_warm[2,i]>=3:
            ax[1].fill_between(ssta_aba.time.values[index_warm[0,i]:index_warm[1,i]],
                     ssta_aba[index_warm[0,i]:index_warm[1,i]],1,
                             ssta_aba[index_warm[0,i]:index_warm[1,i]]>1,color='red')


    for i in range(index_cold.shape[1]):
        if index_cold[2,i]>=3:
            ax[1].fill_between(ssta_aba.time.values[index_cold[0,i]:index_cold[1,i]],
                     ssta_aba[index_cold[0,i]:index_cold[1,i]],-1,
                               ssta_aba[index_cold[0,i]:index_cold[1,i]]<-1,color='blue')


    ax[1].set_ylim([-3,3])
    
    ### NINO 3.4 ###
    index_warm,index_cold,_,_ = create_table_event(ssta_nino34)
    ax[3].set_title(
        'Normalized SST anomalies NINO3.4 [170$^{\circ}$W-120$^{\circ}$W; 5$^{\circ}$S-5$^{\circ}$N] | Baseline '+
                    str(ssta_nino34.time.values[0])[:7] +' --> '+
                    str(ssta_nino34.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
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
    for i in range(index_warm.shape[1]):
        if index_warm[2,i]>=3:
            ax[3].fill_between(ssta_nino34.time.values[index_warm[0,i]:index_warm[1,i]],
                     ssta_nino34[index_warm[0,i]:index_warm[1,i]],1,
                             ssta_nino34[index_warm[0,i]:index_warm[1,i]]>1,color='red')


    for i in range(index_cold.shape[1]):
        if index_cold[2,i]>=3:
            ax[3].fill_between(ssta_nino34.time.values[index_cold[0,i]:index_cold[1,i]],
                     ssta_nino34[index_cold[0,i]:index_cold[1,i]],
                             -1,ssta_nino34[index_cold[0,i]:index_cold[1,i]]<-1,color='blue')
    ax[3].set_ylim([-3,3])
    
    ### DNI ###
    index_warm,index_cold,_,_ = create_table_event(ssta_dni)
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
    for i in range(index_warm.shape[1]):
        if index_warm[2,i]>=3:
            ax[2].fill_between(ssta_dni.time.values[index_warm[0,i]:index_warm[1,i]],
                     ssta_dni[index_warm[0,i]:index_warm[1,i]],1,
                             ssta_dni[index_warm[0,i]:index_warm[1,i]]>1,color='red')


    for i in range(index_cold.shape[1]):
        if index_cold[2,i]>=3:
            ax[2].fill_between(ssta_dni.time.values[index_cold[0,i]:index_cold[1,i]],
                     ssta_dni[index_cold[0,i]:index_cold[1,i]],
                             -1,ssta_dni[index_cold[0,i]:index_cold[1,i]]<-1,color='blue')
    ax[2].set_ylim([-3,3])
    
    ### CNI ###
    index_warm,index_cold,_,_ = create_table_event(ssta_cni)
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
    for i in range(index_warm.shape[1]):
        if index_warm[2,i]>=3:
            ax[4].fill_between(ssta_cni.time.values[index_warm[0,i]:index_warm[1,i]],
                     ssta_cni[index_warm[0,i]:index_warm[1,i]],1,
                             ssta_cni[index_warm[0,i]:index_warm[1,i]]>1,color='red')


    for i in range(index_cold.shape[1]):
        if index_cold[2,i]>=3:
            ax[4].fill_between(ssta_cni.time.values[index_cold[0,i]:index_cold[1,i]],
                     ssta_cni[index_cold[0,i]:index_cold[1,i]],
                             -1,ssta_cni[index_cold[0,i]:index_cold[1,i]]<-1,color='blue')
    ax[4].set_ylim([-3,3])
    
    ### NNI ###
    index_warm,index_cold,_,_ = create_table_event(ssta_nni)
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
    for i in range(index_warm.shape[1]):
        if index_warm[2,i]>=3:
            ax[5].fill_between(ssta_nni.time.values[index_warm[0,i]:index_warm[1,i]],
                     ssta_nni[index_warm[0,i]:index_warm[1,i]],1,
                             ssta_nni[index_warm[0,i]:index_warm[1,i]]>1,color='red')


    for i in range(index_cold.shape[1]):
        if index_cold[2,i]>=3:
            ax[5].fill_between(ssta_nni.time.values[index_cold[0,i]:index_cold[1,i]],
                     ssta_nni[index_cold[0,i]:index_cold[1,i]],
                             -1,ssta_nni[index_cold[0,i]:index_cold[1,i]]<-1,color='blue')
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


def read_data_compute_anomalies_map_atl(path_data):
    
    ds = xr.open_dataset(path_data+'sst.mnmean.nc',engine='pydap')
    mask = xr.open_dataset(path_data+'lsmask.nc',engine='pydap')
    ds = ds.sst.where(mask.mask[0,:,:]==1)
    sst= ds.sel(time=slice(datetime(1982, 1, 1), now))
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

    
    ssta_atl,ssta_atl_norm = ano_norm_t(sst_atl.sst_dtd.load())
    
    ssta_data = xr.Dataset({'ssta': (['time','lat','lon'], ssta_atl),
                           
                          },
                      coords={ 'time':(np.array(ssta_atl.time)),
                          'lat':(np.array(ssta_atl.lat)),
                                'lon':(np.array(ssta_atl.lon))})
    return ssta_data

#def plot_map_ssta(ssta_data):
#    plot = ssta_data.hvplot.contourf(
#        'lon', 'lat', 'ssta', projection=ccrs.PlateCarree(),levels=np.arange(-3,3.3,0.3),clabel='SST anomalies [K]', cmap='RdYlBu_r',widget_location='bottom',fontsize=15,xlabel='Longitude',ylabel='Latitude',
#        label='Time:'+ str(ssta_data.time.values)[:7],
#        coastline=True
#    )
#    return plot
def plot_map_ssta(ssta_data):
    
    
    f = plt.figure(figsize=[15,15])
    n=45
    x = 0.9
    ftz=15
    lower = plt.cm.Blues_r(np.linspace(0, x, n))
    white = np.ones((100-2*n,4))
    upper = plt.cm.Reds(np.linspace(1-x, 1, n))
    colors = np.vstack((lower, white, upper))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('terrain_map_white', colors)
    bounds= np.arange(-3,3.3,0.3)
    ftz=15
    minlon = ssta_data.lon.min()
    maxlon = ssta_data.lon.max()
    minlat = ssta_data.lat.min()
    maxlat = ssta_data.lat.max()
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    cax = inset_axes(ax,
                   width="100%",  # width = 5% of parent_bbox width
                   height="5%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(0, -0.1, 1, 1),
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
    gl.xlocator = mticker.FixedLocator([-160,-140,-120,-100,-80,-60,-40,-20, 0,20,40,60,80,100,120,140,160])
    gl.ylocator = mticker.FixedLocator([-20, 0,20])
    ax.coastlines(linewidth=1)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black',color='lightgrey')
    ax.coastlines(resolution='50m', color='black', linewidth=1)
    c0=ax.contour(ssta_data.lon,
                ssta_data.lat,
                ssta_data,transform=ccrs.PlateCarree(),colors='black',levels=bounds)
    ax.clabel(c0, inline=True, fontsize=10)
    p0=ax.contourf(ssta_data.lon,
                ssta_data.lat,
                ssta_data,transform=ccrs.PlateCarree(),cmap=cmap,levels=bounds,extend='both')
    cbar = plt.colorbar(p0,cax,orientation='horizontal')
    cbar.ax.tick_params(labelsize=ftz)
    cbar.set_label(r' [$^{\circ}$C]', size=ftz,weight='bold')
    ax.set_title('SST anomalies '+str(ssta_data.time.values)[:10],fontsize=ftz,fontweight='bold')


def read_data_compute_anomalies_map_pac(path_data):
    
    ds = xr.open_dataset(path_data+'sst.mnmean.nc',engine='pydap')
    mask = xr.open_dataset(path_data+'lsmask.nc',engine='pydap')
    ds = ds.sst.where(mask.mask[0,:,:]==1)
    sst= ds.sel(time=slice(datetime(1982, 1, 1), now))
    sst = xr.concat([sst[:, :, 180:], sst[:, :, :180]], dim='lon')
    sst.coords['lon'] = (sst.coords['lon'] + 180) % 360 - 180   
    
    ## Make sub areas ##
    sst_pac = sst.where((  sst.lon>=-180) & (sst.lon<=-60) &
                           (sst.lat<=35) & (sst.lat>=-35),drop=True)
    sst_dtd_tmp = np.ones(sst_pac.shape)*np.nan
    for i in range(sst_dtd_tmp.shape[1]):
        for j in range(sst_dtd_tmp.shape[2]):
            sst_dtd_tmp[:,i,j] = nandetrend(sst_pac[:,i,j].values)
            
   
    ## Linearly detrend the data ## 
    sst_pac = sst_pac.assign_coords(sst_dtd=(['time','lat','lon'],  sst_dtd_tmp))

    ## Compute the SST anomalies ## 

    
    ssta_pac,ssta_pac_norm = ano_norm_t(sst_pac.sst_dtd.load())
    ssta_data = xr.Dataset({'ssta': (['time','lat','lon'], ssta_pac),
                           
                          },
                      coords={ 'time':(np.array(ssta_pac.time)),
                          'lat':(np.array(ssta_pac.lat)),
                                'lon':(np.array(ssta_pac.lon))})
    return ssta_data

    
    
#def plot_map_ssta_pac(ssta_pac):
#    f = plt.figure(figsize=[15,15])
#    n=45
#    x = 0.9
#    ftz=15
#    lower = plt.cm.Blues_r(np.linspace(0, x, n))
#    white = np.ones((100-2*n,4))
#    upper = plt.cm.Reds(np.linspace(1-x, 1, n))
#    colors = np.vstack((lower, white, upper))
#    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('terrain_map_white', colors)
#    bounds= np.arange(-3,3.5,0.5)
#    ftz=15
#    minlon = -180
#    maxlon = -60
#    minlat = -35
#    maxlat = 35
#    ax = plt.axes(projection=ccrs.PlateCarree())
#    cax = inset_axes(ax,
#                   width="100%",  # width = 5% of parent_bbox width
#                   height="5%",  # height : 50%
#                   loc='lower left',
#                   bbox_to_anchor=(0, -0.1, 1, 1),
#                   bbox_transform=ax.transAxes,
#                   borderpad=0,
#                   )
#    ax.add_feature(cartopy.feature.LAND, edgecolor='black',color='lightgrey')
#    ax.coastlines()
#    ax.set_extent([minlon,maxlon,minlat,maxlat],ccrs.PlateCarree())
#    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                      linewidth=2, color='black', alpha=0.5, linestyle='-')
#    gl.xlabels_top = False
#    gl.ylabels_right = False
#    gl.xlabel_style = {'size': 15, 'color': 'black'}
#    gl.ylabel_style = {'size': 15, 'color': 'black'}
#    gl.xlocator = mticker.FixedLocator([-160,-140,-120,-100,-80])
#    gl.ylocator = mticker.FixedLocator([-20, 0,20])
#    ax.coastlines(linewidth=1)
#    ax.add_feature(cartopy.feature.LAND, edgecolor='black',color='lightgrey')
#    ax.coastlines(resolution='50m', color='black', linewidth=1)
#    p0=ax.contourf(ssta_pac.lon,
#                ssta_pac.lat,
#                ssta_pac[-1,:,:],transform=ccrs.PlateCarree(),cmap=cmap,levels=bounds,extend='both')
#    cbar = plt.colorbar(p0,cax,orientation='horizontal')
#    cbar.ax.tick_params(labelsize=ftz)
#    ax.set_title('Normalized SST anomalies '+str(ssta_pac.time.values[-1])[:7]+' | Baseline '+
#                    str(ssta_pac.time.values[0])[:7] +' --> '+
#                    str(ssta_pac.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
#    
#    ax.add_patch(mpatches.Rectangle(xy=[-170, -5], width=50, height=10,
#                                        fill=None,
#                                        edgecolor='black',
#                                        alpha=1,linewidth=4,
#                                        transform=ccrs.PlateCarree()))
#
#    ax.text(-150, 1, 'NINO3.4',
#             horizontalalignment='left',fontsize=ftz,fontweight='bold',
#             transform=ccrs.PlateCarree())
#    
#
#    ax.add_patch(mpatches.Rectangle(xy=[-120, 20], width=10, height=10,
#                                        fill=None,
#                                        edgecolor='grey',
#                                        alpha=1,linewidth=4,
#                                        transform=ccrs.PlateCarree()))
#
#    ax.text(-120, 15, 'CNI',
#             horizontalalignment='left',fontsize=ftz,fontweight='bold',
#             transform=ccrs.PlateCarree()) 
#
#    
    
    
def read_data_compute_anomalies_map_ind(path_data):
    
    ds = xr.open_dataset(path_data+'sst.mnmean.nc',engine='pydap')
    mask = xr.open_dataset(path_data+'lsmask.nc',engine='pydap')
    ds = ds.sst.where(mask.mask[0,:,:]==1)
    sst= ds.sel(time=slice(datetime(1982, 1, 1), now))
    sst = xr.concat([sst[:, :, 180:], sst[:, :, :180]], dim='lon')
    sst.coords['lon'] = (sst.coords['lon'] + 180) % 360 - 180   
      
    
    ## Make sub areas ##
    sst_ind = sst.where((  sst.lon>=30) & (sst.lon<=180) &
                           (sst.lat<=25) & (sst.lat>=-45),drop=True)
    sst_dtd_tmp = np.ones(sst_ind.shape)*np.nan
    for i in range(sst_dtd_tmp.shape[1]):
        for j in range(sst_dtd_tmp.shape[2]):
            sst_dtd_tmp[:,i,j] = nandetrend(sst_ind[:,i,j].values)
            
   
    ## Linearly detrend the data ## 
    sst_ind = sst_ind.assign_coords(sst_dtd=(['time','lat','lon'],  sst_dtd_tmp))

    ## Compute the SST anomalies ## 

    
    ssta_ind,ssta_ind_norm = ano_norm_t(sst_ind.sst_dtd.load())
    ssta_data = xr.Dataset({'ssta': (['time','lat','lon'], ssta_ind),
                           
                          },
                      coords={ 'time':(np.array(ssta_ind.time)),
                          'lat':(np.array(ssta_ind.lat)),
                                'lon':(np.array(ssta_ind.lon))})
    return ssta_data

    
    
#def plot_map_ssta_ind(ssta_ind):
#    f = plt.figure(figsize=[15,15])
#    n=45
#    x = 0.9
#    ftz=15
#    lower = plt.cm.Blues_r(np.linspace(0, x, n))
#    white = np.ones((100-2*n,4))
#    upper = plt.cm.Reds(np.linspace(1-x, 1, n))
#    colors = np.vstack((lower, white, upper))
#    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('terrain_map_white', colors)
#    bounds= np.arange(-3,3.5,0.5)
#    ftz=15
#    minlon = 30
#    maxlon = 170
#    minlat = -35
#    maxlat = 35
#    ax = plt.axes(projection=ccrs.PlateCarree())
#    cax = inset_axes(ax,
#                   width="100%",  # width = 5% of parent_bbox width
#                   height="5%",  # height : 50%
#                   loc='lower left',
#                   bbox_to_anchor=(0, -0.1, 1, 1),
#                   bbox_transform=ax.transAxes,
#                   borderpad=0,
#                   )
#    ax.add_feature(cartopy.feature.LAND, edgecolor='black',color='lightgrey')
#    ax.coastlines()
#    ax.set_extent([minlon,maxlon,minlat,maxlat],ccrs.PlateCarree())
#    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                      linewidth=2, color='black', alpha=0.5, linestyle='-')
#    gl.xlabels_top = False
#    gl.ylabels_right = False
#    gl.xlabel_style = {'size': 15, 'color': 'black'}
#    gl.ylabel_style = {'size': 15, 'color': 'black'}
#    gl.xlocator = mticker.FixedLocator([30,50,70,90,110,130,150])
#    gl.ylocator = mticker.FixedLocator([-20, 0,20])
#    ax.coastlines(linewidth=1)
#    ax.add_feature(cartopy.feature.LAND, edgecolor='black',color='lightgrey')
#    ax.coastlines(resolution='50m', color='black', linewidth=1)
#    p0=ax.contourf(ssta_ind.lon,
#                ssta_ind.lat,
#                ssta_ind[-1,:,:],transform=ccrs.PlateCarree(),cmap=cmap,levels=bounds,extend='both')
#    cbar = plt.colorbar(p0,cax,orientation='horizontal')
#    cbar.ax.tick_params(labelsize=ftz)
#    ax.set_title('Normalized SST anomalies '+str(ssta_ind.time.values[-1])[:7]+' | Baseline '+
#                    str(ssta_ind.time.values[0])[:7] +' --> '+
#                    str(ssta_ind.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
#    
#    ax.add_patch(mpatches.Rectangle(xy=[108, -28], width=7, height=6,edgecolor='pink',fill=None,
#                                        alpha=1,linewidth=4,
#                                        transform=ccrs.PlateCarree()))
#
#    ax.text(108, -32, 'NNI',
#             horizontalalignment='left',fontsize=ftz,fontweight='bold',
#             transform=ccrs.PlateCarree()) 
#

def plot_wamoi(cmap_data):

    ds = xr.open_dataset(cmap_data,engine='pydap')
    
    ds= ds.sel(time=slice(datetime(1982, 1, 1), now))
    precip = xr.concat([ds.precip[:, :, 72:], ds.precip[:, :, :72]], dim='lon')
    precip.coords['lon'] = (precip.coords['lon'] + 180) % 360 - 180
    
    
    
    precip_NI = precip.where((  precip.lon>=-10) & (precip.lon<=10) &
                               (precip.lat<=20) & (precip.lat>=7.5),drop=True).mean(dim='lon').mean(dim='lat')
    
    precip_SI = precip.where((  precip.lon>=-10) & (precip.lon<=10) &
                               (precip.lat<=7.5) & (precip.lat>=0),drop=True).mean(dim='lon').mean(dim='lat')
    
    # standardize
    precip_NI_std = precip_NI/precip_NI.std(dim='time')
    precip_SI_std = precip_SI/precip_SI.std(dim='time')
    
    wamoi = precip_NI_std - precip_SI_std

    wamoi_clim = np.ones((now.year+1-2000,73))*np.nan
    onset_date = np.ones((now.year+1-2000))*np.nan
    k=0

    for i in range(2000,now.year+1,1):
        try:
            
            wamoi_clim[k,:] = wamoi.sel(time=slice(datetime(i, 1, 1),datetime(i, 12, 31) ))
            tmp = wamoi.sel(time=slice(datetime(i, 1, 1),datetime(i, 12, 31) ))
            index_WAM = xr.full_like(tmp,0)
            index_WAM[tmp >= 0] = 1
            index_tmp = []
            for j in range(index_WAM.shape[0]-4):
                if (index_WAM[j] ==1) & (index_WAM[j+1] ==1) & (index_WAM[j+2] ==1)& (index_WAM[j+3] ==1):
                    index_tmp.append(j)
            try:       
                onset_date[k]=index_tmp[0]
            except IndexError:
                pass

            k+=1
        except ValueError:
            test = wamoi.sel(time=slice(datetime(i, 1, 1),datetime(i, 12, 31) ))
            new = np.ones((73))*np.nan
            new[:test.shape[0]] = test
            wamoi_clim[k,:]=new


            
            
            
    time = wamoi.sel(time=slice(datetime(2020, 1, 1),datetime(2020, 12, 31) ))   
    
    
    f,ax = plt.subplots(2,1,figsize=[15,10])
    ftz=20
    ax=ax.ravel()
    
    ax[0].plot(time.time,np.nanmean(wamoi_clim,0),color='black',linewidth=3,label='Mean 2000-2021')
    
    for i in range(wamoi_clim.shape[0]):
        ax[0].plot(time.time,wamoi_clim[i,:],color='grey',linewidth=1,alpha=0.3)
    ax[0].plot(time.time,wamoi_clim[-1,:],color='red',linewidth=3,label=str(now.year))
    ax[0].legend(fontsize=ftz)
    ax[0].tick_params(labelsize=ftz)
    #
    locator = mdates.MonthLocator()  # every month
    # Specify the format - %b gives us Jan, Feb...
    fmt = mdates.DateFormatter('%b')
    ax[0].xaxis.set_major_locator(locator)
    # Specify formatter
    ax[0].xaxis.set_major_formatter(fmt)
    ax[0].set_ylabel('WAMOI',fontsize=ftz,fontweight='bold')
    ax[0].set_title(str(wamoi.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax[0].axhline(0,color='black',linestyle='--')
    
    xtime = pd.date_range(start='1/1/2000', periods=now.year+1-2000, freq='Y')
    ax[1].plot(xtime,onset_date,color='black',linewidth=3)
    ax[1].set_ylabel('Onset date [Pentad num]',fontsize=ftz,fontweight='bold')
    ax[1].set_xlabel('Year',fontsize=ftz,fontweight='bold')
    ax[1].tick_params(labelsize=ftz)
    ax[1].axhline(36,color='grey',label='1st July')
    ax[1].legend(fontsize=ftz)
    
    
    
def plot_amm(data_amm):
    df = pd.read_csv(data_amm) 
    AMM = np.ones((df.shape[0]))*np.nan
    for i in range(AMM.shape[0]):
        AMM[i] = float(np.array(df.values[i,0].split())[2])

    time = pd.date_range(start='1/1/1948', periods=AMM.shape[0], freq='M')

    f,ax = plt.subplots(1,1,figsize=[15,5])
    ftz=15
    ax.plot(time,AMM,color='black',linewidth=2)
    ax.axhline(0,color='black')
    ax.tick_params(labelsize=ftz)
    ax.set_title('AMM (SST based) | '+str(time[-1])[:7]+' | Chiang and Vimont (2004)',fontsize=ftz,fontweight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(myFmt)
    ax.text(0.01,0.04,'Updated '+date_time,transform=ax.transAxes,
           size=ftz,
           weight='bold')
    
    
    
    
import haversine
def read_data_ACT_week_plot(path_data):

    ds = xr.open_dataset(path_data+'sst.wkmean.1990-present.nc',engine='pydap')
    mask = xr.open_dataset(path_data+'lsmask.nc',engine='pydap')
    ds = ds.sst.where(mask.mask[0,:,:]==1)
    sst= ds.sel(time=slice(datetime(1990, 1, 1), now))
    sst = xr.concat([sst[:, :, 180:], sst[:, :, :180]], dim='lon')
    sst.coords['lon'] = (sst.coords['lon'] + 180) % 360 - 180  
    
    ## Make sub areas ##
    sst_act = sst.where((  sst.lon>=-30) & (sst.lon<=12) &
                           (sst.lat<=5) & (sst.lat>=-5),drop=True)
    
    
    
    
    
    sst_index = np.array(25-sst_act)
    Sact = np.zeros((sst_act.shape[0]))
    Tact = np.zeros((sst_act.shape[0]))
    X = np.ones((sst_act.shape[2]-1,sst_act.shape[1]-1))
    Y = np.ones((sst_act.shape[2]-1,sst_act.shape[1]-1))
    lon = np.array(sst_act.lon)
    lat = np.array(sst_act.lat)
    for t in range(Sact.shape[0]):
        tmp_sact=0
        tmp_tact=0
        for i in range(sst_act.shape[2]-1):
            for j in range(sst_act.shape[1]-1):



                if sst_index[t,j,i]>0:

                    X = haversine.haversine((lon[i], lat[j]),
                                        (lon[i+1], lat[j]))

                    Y = haversine.haversine((lon[i], lat[j]),
                                        (lon[i], lat[j+1]))
                    tmp_sact += X*Y
                    tmp_tact += X*Y*sst_index[t,j,i]

        Sact[t] =tmp_sact
        Tact[t] = tmp_tact/Sact[t]

    Sact_dataset = xr.Dataset({'sact': (['time'], Sact)},
                          coords={ 'time':(np.array(sst_act.time)),
                              })






    onset_date=[]
    for i in range(1990,now.year+1,1):
        index_tmp = []
        try:

            sact_clim = Sact_dataset.sact.sel(time=slice(datetime(i, 1, 1),datetime(i, 12, 31) ))


            for j in range(sact_clim.shape[0]):
                if sact_clim[j]>0.4*1e6:
                    index_tmp.append(j)
            onset_date.append(index_tmp[0])

        except IndexError:

            onset_date.append(np.nan)



    f,ax = plt.subplots(2,1,figsize=[15,10],sharex=True)
    ax=ax.ravel()
    ftz=15
    ax[0].plot(Sact_dataset.time,Sact_dataset.sact*1e-6,color='grey')
    ax[0].tick_params(labelsize=ftz)
    ax[0].axhline(0.4,color='red',label='treshold = 0.4 1e6 km$^{2}$')
    ax[0].legend(fontsize=ftz)
    ax[0].set_ylabel('S$_{act}$ [10$^{6}$ km$^{2}$]',fontsize=ftz,fontweight='bold')
    ax[0].set_title('Atlantic Cold Tongue Onset',fontsize=ftz,fontweight='bold')
    ax[1].set_xlabel('Year', fontsize=ftz)

    locator = mdates.YearLocator(2)  # every month
    # Specify the format - %b gives us Jan, Feb...
    fmt = mdates.DateFormatter('%Y')
    ax[0].xaxis.set_major_locator(locator)
    # Specify formatter
    ax[0].xaxis.set_major_formatter(fmt)

    xtime = pd.date_range(start='1/1/1989', periods=now.year+1-1990, freq='Y')
    ax[1].plot(xtime,np.array(onset_date)*7,color='black')
    ax[1].tick_params(labelsize=ftz)

    locator = mdates.YearLocator(2)  # every month
    # Specify the format - %b gives us Jan, Feb...
    fmt = mdates.DateFormatter('%Y')
    ax[1].xaxis.set_major_locator(locator)
    # Specify formatter
    ax[1].xaxis.set_major_formatter(fmt)
    ax[1].set_ylabel('Onset date [Day of year]',fontsize=ftz,fontweight='bold')
    ax[1].axhline(21*7,label='1st June',color='grey')

    ax[1].legend(fontsize=ftz)
    ax[1].text(0.01,0.04,'Updated '+date_time,transform=ax[1].transAxes,
               size=ftz,
               weight='bold')
    
    return sst_act


def plot_amo(data_amo):
    ftz=15
    df = pd.read_csv(data_amo) 
    AMO_tmp = []
    k=0
    while k< (now.year+1-1856):
        for i in range(1,13):
            AMO_tmp.append(float(np.array(df.values[k][0].split())[i]))
        k+=1
    #float(np.array(df.values[i,0].split())[2])

    AMO = np.array(AMO_tmp)

    time = pd.date_range(start='1/1/1856', periods=AMO.shape[0], freq='M')
    AMO_ds  = xr.Dataset({'amo': (['time'],AMO[:])}

                                    ,coords={'time':np.array(time)}
                                    ,attrs={'standard_name': 'AMO',
                                'long_name': 'Atlantic Multi-decadal Oscillation',

                                'Creation_date':date_time,   
                                'author': 'Arthur Prigent',
                                'email': 'aprigent@geomar.de'}
                                      ) 
    AMO_ds_nonan = AMO_ds.where(AMO_ds.amo>-90)
    AMO_ds_nonan_roll = AMO_ds_nonan.amo.rolling(
        time=120,min_periods =120, center = True).mean()
    f,ax = plt.subplots(1,1,figsize=[15,5])


    ax.plot(AMO_ds_nonan.time,AMO_ds_nonan.amo,color='grey',alpha=0.30)
    ax.plot(AMO_ds_nonan_roll.time,AMO_ds_nonan_roll,color='black')
    ax.fill_between(AMO_ds_nonan_roll.time.values,AMO_ds_nonan_roll,0,AMO_ds_nonan_roll>0,color='red')
    ax.fill_between(AMO_ds_nonan_roll.time.values,AMO_ds_nonan_roll,0,AMO_ds_nonan_roll<0,color='blue')
    ax.axhline(0,color='black')
    ax.tick_params(labelsize=ftz)
    ax.tick_params(labelsize=ftz)
    ax.set_title('AMO (SST based) | Enfield et al., (2001)',fontsize=ftz,fontweight='bold')
    years = mdates.YearLocator(20)   # every 20 years
    years_minor = mdates.YearLocator(10)  # every 10 years
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(myFmt)
    ax.text(0.01,0.04,'Updated '+date_time,transform=ax.transAxes,
               size=ftz,
               weight='bold')
    
    
def read_compute_anomalies_uwind_plot(data):

    ds = xr.open_dataset(data,engine='pydap')
    ds= ds.sel(time=slice(datetime(1982, 1, 1), now))
    uwnd = xr.concat([ds.uwnd[:, :, 72:], ds.uwnd[:, :, :72]], dim='lon')
    uwnd.coords['lon'] = (uwnd.coords['lon'] + 180) % 360 - 180



    ## Make sub areas ##
    uwnd_atl4 = uwnd.where((  uwnd.lon>=-40) & (uwnd.lon<=-20) &
                           (uwnd.lat<=3) & (uwnd.lat>=-3),drop=True).mean(dim='lon').mean(dim='lat')


    ## Linearly detrend the data ## 
    uwnd_atl4 = uwnd_atl4.assign_coords(uwnd_dtd=('time',  nandetrend(uwnd_atl4.values)))




    ## Compute the uwnd anomalies ## 


    uwnda_atl4,uwnda_atl4_norm = ano_norm_t_wk(uwnd_atl4.uwnd_dtd.load())
    
    
    f,ax = plt.subplots(1,1,figsize=[15,5])
    color_lines='grey'
    ftz=15
    ax.plot(uwnda_atl4_norm.time,uwnda_atl4_norm)
    ax.axhline(0,color=color_lines)
    ax.axhline(1,color=color_lines,linestyle='--')
    ax.axhline(-1,color=color_lines,linestyle='--')
    ax.plot(uwnda_atl4_norm.time.values,uwnda_atl4_norm,color='black')
    years = mdates.YearLocator(5)   # every year
    years_minor = mdates.YearLocator(1)  # every month
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(myFmt)
    ax.tick_params(labelsize=ftz)
    ax.fill_between(uwnda_atl4_norm.time.values,uwnda_atl4_norm,1,uwnda_atl4_norm>1,color='red')
    ax.fill_between(uwnda_atl4_norm.time.values,uwnda_atl4_norm,-1,uwnda_atl4_norm<-1,color='blue')
    ax.set_title('Normalized UWND anomalies ATL4 [40$^{\circ}$W-20$^{\circ}$W; 3$^{\circ}$S-3$^{\circ}$N] | Baseline '+
                 str(uwnda_atl4_norm.time.values[0])[:7] +' --> '+
                 str(uwnda_atl4_norm.time.values[-1])[:7],fontsize=ftz,fontweight='bold')
    ax.text(0.01,0.04,'Updated '+date_time,transform=ax.transAxes,
           size=ftz,
           weight='bold')
    ax.set_ylim([-3,3])
    
    
def plot_slp(ncep_data_slp):
    ds = xr.open_dataset(ncep_data_slp,engine='pydap')
    ds= ds.sel(time=slice(datetime(1982, 1, 1), now))
    slp = xr.concat([ds.slp[:, :, 72:], ds.slp[:, :, :72]], dim='lon')
    slp.coords['lon'] = (slp.coords['lon'] + 180) % 360 - 180
    slp_atl = slp.where((  slp.lon>=-30) & (slp.lon<=-10) &
                           (slp.lat<=-20) & (slp.lat>=-40),drop=True).mean(dim='lon').mean(dim='lat')


    ## Linearly detrend the data ## 
    slp_atl = slp_atl.assign_coords(slp_dtd=('time',  nandetrend(slp_atl.values)))




    ## Compute the slp anomalies ## 
    slpa_atl,slpa_atl_norm = ano_norm_t_wk(slp_atl.slp_dtd.load())
    slpa_atl_norm_3mean = slpa_atl_norm.rolling(time=3,center='mean').mean()
    
    
    f,ax = plt.subplots(1,1,figsize=[15,5])
    ftz=15
    ax.plot(slpa_atl_norm_3mean.time,slpa_atl_norm_3mean,color='black',linewidth=2)
    ax.axhline(0,color='black')
    ax.axhline(1,color='black',linestyle='--')
    ax.axhline(-1,color='black',linestyle='--')
    ax.fill_between(slpa_atl_norm_3mean.time.values,slpa_atl_norm_3mean,1,slpa_atl_norm_3mean>1,color='red')
    ax.fill_between(slpa_atl_norm_3mean.time.values,slpa_atl_norm_3mean,-1,slpa_atl_norm_3mean<-1,color='blue')
    ax.tick_params(labelsize=ftz)
    ax.set_title('St. Helena index (SLP based) | '+
                 str(slpa_atl_norm.time.values[-1])[:7]+
                 ' | 40$^{\circ}$S-20$^{\circ}$S; 30$^{\circ}$W-10$^{\circ}$W | Lübbecke et al., (2010)',
                 fontsize=ftz,fontweight='bold')
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(myFmt)

