####################################################################
# Mass Balance and Ice Dynamics Calibrations for a set of glaciers #
####################################################################

# Tal Shutkin, 1/31/2022
# Input Data:
#     1] DEM
#     2] Glacier outlines/intersects
#     3] Mass balance observations
#     4] GPR data

import os
import numpy as np
from math import sqrt
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import salem
import datetime
import oggm
from oggm.core import climate
from oggm import workflow, tasks, utils, cfg

def calibrate_MB_from_gMB(gdir,mbdf_path,gmb_path=None,period=None,half_window=5):
#     Inputs: 
#         gdir : OGGM glacier directory with all files up to climate written
#         mbdf_path : path to annually indexed mass balance data (.csv) (index='Year')
#         gmb_path : path to geodetic mass balance epochs dataframe (.csv) (indexed using RGIId following OGGM match_geodetic_mb_for_selection convention) 
#         period : the mass balance epoch for final residual correction
#         half_window : The half perdiod where OGGM will search for t-star 
#                       (default 5 years for 11 year window, given short duration of mbdf)
#     Output:
#         df_diag : pandas dataframe containing calibration diagnostics: mu-star, t-star, and bias for each climate product
    mbdf=pd.read_csv(mbdf_path,index_col='Year')
    cfg.PARAMS['tstar_search_window']=[mbdf.index.values.min(),2019] # Duration of MB timeseries and climate data (2019)
    cfg.PARAMS['mu_star_halfperiod']=half_window # 5-year mu-star half period

    gdir.set_ref_mb_data(mbdf) # Set ref data
    utils.get_ref_mb_glaciers_candidates(rgi_version='6');
    cfg.DATA['RGI60_ref_ids'].append(gdir.rgi_id) 
    cfg.PARAMS['run_mb_calibration'] = True
    
    # Run Mass Balance Calibration
    df_diag=pd.DataFrame(columns=['model','mu_star','t_star','bias'])
    i=1
    for clim in ['CRU','ERA5']:
        cfg.PARAMS['baseline_climate']=clim
        tasks.process_climate_data(gdir); # No way around running this again...?
        climate.compute_ref_t_stars([gdir])
        print('Made it to local t-star')
        tasks.local_t_star(gdir);
        tasks.mu_star_calibration(gdir);
        if period is not None:
            workflow.match_geodetic_mb_for_selection(gdir, period=period,
                                                     file_path=gmb_path, fail_safe=False)
#             except:
#                 print('You must provide path to geodetic mass balance dataframe!')

        # Diagnostics
        bias=gdir.read_json('local_mustar')['bias']
        mu=gdir.read_json('local_mustar')['mu_star_glacierwide']
        t_star=gdir.read_json('local_mustar')['t_star']
        df_diag.loc[i]=f'{clim}_cal',mu,t_star,bias
        i+=1
        
    return df_diag

###########################################################

def aggregate_gpr_data_clip(gdir,gpr,plot=False): 
    """
    :param gdir: an OGGM glacier directory
    :param gpr: path to GPR data or dataframe containing GPR data 
    :param plot: Set to true to view plotted aggregated GPR data
    """
    #  Import GPR Observations
    if type(gpr)==str:
        try:
            df = salem.read_shapefile(gpr_path,cached=True)
        except:
            print('WARNING: We do not have GPR data for this glacier and/or year')
            return ''
    else:
        try:
            df = gpr
            safe = df.iloc[0] # see if dataframe
        except:
            print('WARNING: No valid GPR provided')
            return ''
    coords = np.array([p.xy for p in df.geometry]).squeeze()
    df['lon'] = coords[:, 0]
    df['lat'] = coords[:, 1]
    try:
        df = df[['lon', 'lat', 'profundida','altura_sup','altura_Bed','geometry']]
        mask = gdir.read_shapefile('outlines').to_crs('epsg:4326')
        df=gpd.clip(df,mask) ##Clipping
    except:
        df = df[['lon', 'lat', 'profundida','geometry']]
        mask = gdir.read_shapefile('outlines').to_crs('epsg:4326')
        df=gpd.clip(df,mask) ##Clipping
    xx, yy = salem.transform_proj(salem.wgs84, gdir.grid.proj, df['lon'].values, df['lat'].values)
    df['x'] = xx
    df['y'] = yy
    # Create aggregate GPR dataframe based on gdir grid, (and clip to outline geometry)
    df_agg = df.copy()
    ii, jj = gdir.grid.transform(df['lon'], df['lat'], crs=salem.wgs84, nearest=True)
    df_agg['i'] = [int(i) for i in ii]
    df_agg['j'] = [int(j) for j in jj]
    # We trick by creating an index of similar i's and j's
    df_agg['ij'] = ['{:04d}_{:04d}'.format(i, j) for i, j in zip(ii, jj)]
    df_agg = df_agg.groupby('ij').mean()
    
    # Check for Null Values
    if df_agg.isnull().values.any()==True:
        print('WARNING: This dataframe contains Null Values!!')
    else:
        print('Dataframe looks good!')
        
    # Plotting
    if plot==True:
        geom = gdir.read_shapefile('outlines')
        f, ax = plt.subplots(1,1)
        ax.scatter(df_agg.x,df_agg.y,c=df_agg.profundida,cmap='Blues',s=10);
        #df_agg.plot.scatter(x='x', y='y',c='profundida', cmap='Blues', s=10, ax=ax); #cmap='Blues',
        geom.plot(ax=ax, facecolor='none', edgecolor='k');
        
    return df_agg

##################################################################3

def inversion_model_run(gdir, glen_a=None,fs=None, best=False, file_suffix=None, kcalving=False, k=None,water_level=None):
    """
    :param gdir: glacier directory object
    :param name: name of the parameter to be modified
    :param parameter: value of the parameter to be modified
    :param best: if TRUE, an extra save with prefix 'best' will be created
    :param kcalving: if TRUE, will use calving for inversion
    :param water_level: water level (m asl) for lake terminating glaciers
    :param k: calving constant to use for inversion
    """
#     try:
#         gdir.read_pickle('model_flowlines')[0].thick
#     except:
#         print('Be sure to run tasks.init_present_time_glacier() on gdirs prior to run!')
    
    glacier=gdir.name
    if kcalving==True:
        cfg.PARAMS['use_kcalving_for_inversion']=True
        gdir.is_tidewater = True 
    else:
        cfg.PARAMS['use_kcalving_for_inversion']=False
        gdir.is_tidewater = False 
    # Inversion tasks
    workflow.inversion_tasks([gdir],glen_a=glen_a, fs=fs, filter_inversion_output=True);
    workflow.execute_entity_task(tasks.distribute_thickness_per_altitude, [gdir]); # dis_from_border_exp=DFB
    workflow.execute_entity_task(tasks.init_present_time_glacier, [gdir]); ##This updates the mode_flowlines file and creates a stand-alone numerical glacier ready to run.
    
    ########### gather more detailed information about the glacier ##############
    df  = utils.compile_glacier_statistics([gdir], inversion_only=True)
    volume=df['inv_volume_km3']

    mfl=gdir.read_pickle('model_flowlines')
    thickness= mfl[-1].surface_h - mfl[-1].bed_h
    thickness=thickness[thickness>0].mean()

    ############## write it into txt file: #################

    if(best):
        file= open(os.getcwd()+'/best_inversion_model_run_of_'+glacier+'_'+ file_suffix + '.txt', mode='w')
        print('parameter value: '+str(glen_a)+'\n')
        print('glacier total volume: '+str(volume)+' km^3 \n \n \n')
        print('mean thickness value: '+str(thickness)+' m \n')
    else:
        file= open(os.getcwd()+'/inversion_model_run_of_'+glacier+'.txt', mode='w')

    file.writelines(['Run of the model on ' +glacier+' on ', str(datetime.datetime.now()), '\n'])
    file.write('Inversion Glen A: '+str(gdir.get_diagnostics()['inversion_glen_a'])+'\n')
    file.write('Inversion Sliding Factor: '+str(gdir.get_diagnostics()['inversion_fs'])+'\n')
    file.write('glacier total volume: '+str(volume)+' km^3 \n \n \n')
    file.write('mean thickness value: '+str(thickness)+' m \n')
#     file.write('Minimal value is '+str(np.min(to_plot[1]))+ ' at ' + parameter_name+' = ' +str(np.round(to_plot[0][x],decimals=28)))
    file.close()
    ##########################################################
    #print(gdirs[0])
    return volume, thickness

####################################################################################################3

def error_eval(obs, var, gdir, return_df=False):
    
    """
    :param obs: observational point thickness data
    :param var: a variable in obs represented ice thickness (str) 
    :param gdir: OGGM gdir with modeled distributed thickness file
    :type obs: pandas dataframe indexed using aggregate_gpr_data() function
    :type model: glacier directory (model_run() output)
    :returns: RMSE error
    :rtype: float
    """
    
    ds = xr.open_dataset(gdir.get_filepath('gridded_data')).load() 
    df=obs.copy()
    df['i']=df.i.values.astype(int) # ensure integer indeces
    df['j']=df.j.values.astype(int)
    df['model_thickness'] = ds.distributed_thickness.isel(x=('z', df['i']), y=('z', df['j'])) # take raster points with GPR data
    ds.close()
    df=df.dropna() # Fixes bug with Arteson and others....Not suree of consequenses yet...
    ERROR = df['model_thickness']-df[var] # Find deviation from observation
    AE = abs(ERROR)
    ME = ERROR.mean()
    MAE = AE.mean()
    RMSE = sqrt((ERROR**2).mean())
    
    df['error']=ERROR
    df['abs_error']=AE
    
   # assert (isnan(error)==True), 'There is a problem in obs/model allignment. Check for null values'
    if return_df==True:
        return ME, MAE, RMSE, df
    else:
        return ME, MAE, RMSE

###################################################################################################

    
    
    













    
    
                              

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
