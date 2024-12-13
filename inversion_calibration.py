# Improved function for calibrating OGGM inversion using GPR

# Citation:
# Shutkin, T. Y. et al. (under review). Modeling the impacts of climate trends and lake formation on the
#   retreat of a tropical Andean glacier (1962-2020). The Cryosphere.

# Tal Shutkin: 4/24/2024

import os
import numpy as np
import time
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
from sklearn.model_selection import train_test_split

def custom_distribute_thickness_per_altitude(gdir, add_slope=True,
                                      smooth_radius=None,
                                      dis_from_border_exp=0.25,
                                            varname_suffix=''):
    """Compute a thickness map by redistributing mass along altitudinal bands.

    This is a rather cosmetic task, not relevant for OGGM but for ITMIX.
    Unlike OGGM function, does not write file.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    add_slope : bool
        whether a corrective slope factor should be used or not
    smooth_radius : int
        pixel size of the gaussian smoothing. Default is to use
        cfg.PARAMS['smooth_window'] (i.e. a size in meters). Set to zero to
        suppress smoothing.
    dis_from_border_exp : float
        the exponent of the distance from border mask
    """
    from oggm.core.gis import gaussian_blur
    
    # Variables
    grids_file = gdir.get_filepath('gridded_data')
    # See if we have the masks, else compute them
    with utils.ncDataset(grids_file) as nc:
        has_masks = 'glacier_ext_erosion' in nc.variables
    if not has_masks:
        from oggm.core.gis import gridded_attributes
        gridded_attributes(gdir)

    with utils.ncDataset(grids_file) as nc:
        topo_smoothed = nc.variables['topo_smoothed'][:]
        glacier_mask = nc.variables['glacier_mask'][:]
        dis_from_border = nc.variables['dis_from_border'][:]
        if add_slope:
            slope_factor = nc.variables['slope_factor'][:]
        else:
            slope_factor = 1.

    # Along the lines
    cls = gdir.read_pickle('inversion_output')
    fls = gdir.read_pickle('inversion_flowlines')
    hs, ts, vs, xs, ys = [], [], [], [], []
    for cl, fl in zip(cls, fls):
        hs = np.append(hs, fl.surface_h)
        ts = np.append(ts, cl['thick'])
        vs = np.append(vs, cl['volume'])
        try:
            x, y = fl.line.xy
        except AttributeError:
            # Squeezed flowlines, dummy coords
            x = fl.surface_h * 0 - 1
            y = fl.surface_h * 0 - 1
        xs = np.append(xs, x)
        ys = np.append(ys, y)

    init_vol = np.sum(vs)

    # Assign a first order thickness to the points
    # very inefficient inverse distance stuff
    thick = glacier_mask * np.NaN
    for y in range(thick.shape[0]):
        for x in range(thick.shape[1]):
            phgt = topo_smoothed[y, x]
            # take the ones in a 100m range
            starth = 100.
            while True:
                starth += 10
                pok = np.nonzero(np.abs(phgt - hs) <= starth)[0]
                if len(pok) != 0:
                    break
            sqr = np.sqrt((xs[pok]-x)**2 + (ys[pok]-y)**2)
            pzero = np.where(sqr == 0)
            if len(pzero[0]) == 0:
                thick[y, x] = np.average(ts[pok], weights=1 / sqr)
            elif len(pzero[0]) == 1:
                thick[y, x] = ts[pzero]
            else:
                raise RuntimeError('We should not be there')

    # Distance from border (normalized)
    dis_from_border = dis_from_border**dis_from_border_exp
    dis_from_border /= np.mean(dis_from_border[glacier_mask == 1])
    thick *= dis_from_border

    # Slope
    thick *= slope_factor

    # Smooth
    dx = gdir.grid.dx
    if smooth_radius != 0:
        if smooth_radius is None:
            smooth_radius = np.rint(cfg.PARAMS['smooth_window'] / dx)
        thick = gaussian_blur(thick, int(smooth_radius))
        thick = np.where(glacier_mask, thick, 0.)

    # Re-mask
    utils.clip_min(thick, 0, out=thick)
    thick[glacier_mask == 0] = np.NaN
    assert np.all(np.isfinite(thick[glacier_mask == 1]))

    # Conserve volume
    tmp_vol = np.nansum(thick * dx**2)
    thick *= init_vol / tmp_vol
    
    return thick

def map_dist_thk(gdir, ds=None, glen_a=None, fs=None, varname_suffix=''):
    
    # OGGM stuff
    #workflow.inversion_tasks([gdir], glen_a=glen_a, fs=fs, filter_inversion_output=True)
    workflow.execute_entity_task(tasks.prepare_for_inversion, [gdir])
    workflow.execute_entity_task(tasks.mass_conservation_inversion, [gdir], glen_a=glen_a, fs=fs, write=True)
    workflow.execute_entity_task(tasks.filter_inversion_output, [gdir])
    
    volume= tasks.get_inversion_volume(gdir)
    thick = custom_distribute_thickness_per_altitude(gdir,varname_suffix=varname_suffix) # calculate dist thicknesss without writing to file
    
    return_ds=False
    if ds is None:
        return_ds=True # remember to spit out for later use
        # Get dimensions from glacier directory
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            x_vals= ds.x.values
            y_vals= ds.y.values
    else:
        x_vals=ds.x.values
        y_vals=ds.y.values
    
    da = xr.DataArray(
            data=thick, # need to transpose array for some reason, I think due to the order that I looped through the dataset
            dims=["y", "x"],
            coords=dict(
                y=y_vals,
                x=x_vals,

            ),
            attrs=dict(
                name=varname_suffix,
                Creep = glen_a,
                Volume = volume,
                description=f"Distributed thickness predicted by OGGM (Model run suffix: {varname_suffix}.",
                units="meters",
            ),
        )
    
    ds.update({f'param_{varname_suffix}_dist_thk':da}) # store thickness map

    if return_ds:
        return da, volume, ds
    else:
        return da, volume


def error_eval(model, data, data_var, return_df=False):
    """
    Evaluate gridded model output data (xarray dataarray) of variable
    against point observations (geopandas dataframe) of variable (data_var)
    Returns RMSE, MAE, with option of point by point error table output.
    """
    from math import sqrt 
    
    df=data.copy()
    df['i']=df.i.values.astype(int) # ensure integer indeces
    df['j']=df.j.values.astype(int)
    df['model_thk'] = model.isel(x=('z', df['i']), y=('z', df['j'])) # take raster points with GPR data
    df=df.dropna() # Fixes bug with Arteson and others....Not suree of consequenses yet...
    ERROR = df['model_thk']-df[data_var] # Find deviation from observation
    AE = abs(ERROR)
    ME = ERROR.mean()
    MAE = AE.mean()
    RMSE = sqrt((ERROR**2).mean())
    
    df['error']=ERROR
    df['abs_error']=AE
    
   # assert (isnan(error)==True), 'There is a problem in obs/model allignment. Check for null values'
    if return_df==True:
        return RMSE, MAE, ME, df
    else:
        return RMSE, MAE, ME
    

def Creep_Cal(glacier, param_values, obs, model, model_varname,varname_suffix=''):
    # Function to randomly divide calibration data and minimize glacier thickness MAE against GPR data
    # Outputs: Candidate Creep Param, "A-factor", RMSE, MAE, ME, glacier volume for the given TRAIN/TEST split
    # This function is to be run iteratively producing a candidate param PDF
    default_glen_a = 2.4e-24
    all_coords = obs.index.to_list()
    TRAIN_coords, TEST_coords = train_test_split(all_coords,test_size=0.5)
    TRAIN = obs.loc[TRAIN_coords]
    TEST = obs.loc[TEST_coords]
    
    training_MAE = []

    for A in param_values:
        try:
            time.sleep(0.002)
            da,v = map_dist_thk(glacier, ds=model, glen_a=A, fs=None, varname_suffix='')
            MAE = error_eval(da, TRAIN, model_varname)[1]
            if np.isnan(MAE)==True:
                break
            training_MAE.append(MAE)
        except ValueError:
            pass
    candidate = param_values[training_MAE.index(min(training_MAE))] # Take parameter value for lowest error across training dataset
    da,volume = map_dist_thk(glacier, ds=model, glen_a=candidate, fs=None, varname_suffix=varname_suffix)
    model.update({f'cand_{varname_suffix}_dist_thk':da}) # store thickness map
    RMSE,MAE,ME = error_eval(da, TEST, model_varname) # evaluate error against testing dataset        
    
    return candidate,candidate/default_glen_a,RMSE,MAE,ME,volume
    
