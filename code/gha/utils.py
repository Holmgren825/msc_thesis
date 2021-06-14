'''Utility functions for hydro analysis of glaciated basins with the OGGM.'''

import xarray as xr
import numpy as np
import geopandas as gpd
import shapely.geometry as shpg
from oggm import utils, workflow, tasks, cfg
from oggm.shop import gcm_climate
import zipfile
import tempfile
import os
from pathos.multiprocessing import ProcessPool
import itertools


def download_proj_data(rcp):
    '''
    Small helper function to download data.

    Args
    rcp: List of rcp scenarios.

    Returns
    ft: Path to temperature file
    fp: Path to precipitation files
    '''
    # Base file paths
    # Note that these dont contain any scenario data.
    bp = 'https://cluster.klima.uni-bremen.de/~oggm/cmip5-' + \
         'ng/pr/pr_mon_CCSM4_{}_r1i1p1_g025.nc'
    bt = 'https://cluster.klima.uni-bremen.de/~oggm/cmip5-' + \
         'ng/tas/tas_mon_CCSM4_{}_r1i1p1_g025.nc'
    # Download the data
    ft = utils.file_downloader(bt.format(rcp))
    fp = utils.file_downloader(bp.format(rcp))

    return ft, fp


def download_clim_data():
    '''Helper function to download climate projection data'''
    f = 'https://cluster.klima.uni-bremen.de/~oggm/climate/cru/cru_cl2.nc.zip'
    fp = utils.file_downloader(f)
    with zipfile.ZipFile(fp, 'r') as zip_ref:
        zip_ref.extractall(fp[:-7])
    return fp[:-7] + '/cru_cl2.nc'


def select_basin_data(df, basin):
    '''Makes a selection of an xarray dataset from a basin (shapefile)'''
    # Region of interest.
    df_sel = df.salem.roi(geometry=basin.geometry)

    # If selection didn't return anything, expect the basin to be too small
    #  last_var = list(df_sel.variables.items())[0][0]
    #  if np.all(np.isnan(df_sel[last_var].values)):
    #      lon = basin.geometry.centroid.x
    #      lat = basin.geometry.centroid.y
    #      # We then select the point closest to the center of the basin.
    #      df_sel = df.sel(lon=lon, lat=lat, method='nearest')
    # Magic
    # With a pandas dataframe we can drop any nans without loosing valid data
    # since the multi index stacks the coordinates.
    df_sel = df_sel.to_dataframe().dropna()
    # Remove the last time steps (NaT). Couldnt figure out how to index the
    # multi index...
    df_sel = df_sel[~df_sel.index.duplicated()].to_xarray()
    # Coming from pandas we have to sort the coordinates.
    df_sel = df_sel.sortby(['lat', 'lon'])

    return df_sel


def process_clim_data(basin, rcp):
    '''Downscales climate projection data for a rcp scenario for a
    basin.

    Args:
    rcp: Strings - rcp scenario
    basin: Geodataseries with basin.

    Return: Bias corrected data selection
    '''

    # Get the projection data for the scenarios
    ft, fp = download_proj_data(rcp)
    # And get the climate data
    fclim = download_clim_data()

    # Open the datasets
    # Temperature, precipitation and climate
    with xr.open_dataset(ft, use_cftime=True) as ds_t,\
            xr.open_dataset(fp, use_cftime=True) as ds_p,\
            xr.open_dataset(fclim, use_cftime=True) as ds_clim:
        # Let's do a coarse selection of the data first.
        minlon, minlat, maxlon, maxlat = basin.geometry.bounds
        # Pad the box
        pad = 1.5
        minlon -= pad
        maxlon += pad
        minlat -= pad
        maxlat += pad
        # Select the data within this box
        ds_t_sel = ds_t.sel(lon=slice(minlon, maxlon),
                            lat=slice(minlat, maxlat))
        ds_p_sel = ds_p.sel(lon=slice(minlon, maxlon),
                            lat=slice(minlat, maxlat))
        ds_clim_sel = ds_clim.sel(lon=slice(minlon, maxlon),
                                  lat=slice(maxlat, minlat))

        # Convert the precipitaion to mm per month
        ny, r = divmod(len(ds_p_sel.time), 12)
        assert r == 0, r
        dimo = [cfg.DAYS_IN_MONTH[m - 1] for m in
                ds_p_sel['time.month']]
        # We need some numpy functionality.
        dimo = np.array(dimo)
        # Reshape for broadcasting
        dimo = dimo.reshape(-1, 1, 1)
        # Multiply with seconds per month.
        ds_p_sel['pr'] = ds_p_sel.pr * dimo * (60 * 60 * 24)
        # Convert clim temp to Kelvin
        ds_clim_sel['temp'] = ds_clim_sel.temp + 273.15

        # Can now compute the anomalies, and do the bias correction.
        # Temperature
        ds_t_avg = ds_t_sel.sel(time=slice('1961', '1990'))
        ds_t_avg = ds_t_avg.groupby('time.month').mean(dim='time')
        # Anomaly
        ds_t_ano = ds_t_sel.groupby('time.month') - ds_t_avg

        # Precipitation
        ds_p_avg = ds_p_sel.sel(time=slice('1961', '1990'))
        ds_p_avg = ds_p_avg.groupby('time.month').mean(dim='time')
        # Anomaly
        ds_p_ano = ds_p_sel.groupby('time.month') - ds_p_avg
        # Scaled anomaly?
        ds_p_scaled = ds_p_sel.groupby('time.month') / ds_p_avg

        # Now we can do the downscaling.
        # Temperature
        ds_t_sel_remap = ds_clim_sel.salem.grid.map_gridded_data(
                            ds_t_ano.tas.values, ds_t_ano.salem.grid,
                            interp='linear')

        # Precipitation
        ds_p_sel_remap = ds_clim_sel.salem.grid.map_gridded_data(
                            ds_p_ano.pr.values, ds_p_ano.salem.grid,
                            interp='linear')

        time = ds_t_sel.time
        ds_t_sel = xr.DataArray(ds_t_sel_remap, dims=['time', 'lat', 'lon'],
                                coords={'time': time, 'lat': ds_clim_sel.lat,
                                        'lon': ds_clim_sel.lon}, name='temp')

        ds_p_sel = xr.DataArray(ds_p_sel_remap, dims=['time', 'lat', 'lon'],
                                coords={'time': time, 'lat': ds_clim_sel.lat,
                                        'lon': ds_clim_sel.lon}, name='prcp')
        # Put it all in a dataset.
        ds_selection = xr.Dataset({'temp': ds_t_sel, 'prcp': ds_p_sel})

        # Final selection of data for the basin, i.e. finer selection.
        ds_selection = select_basin_data(ds_selection, basin)
        # Climatology
        ds_clim_sel = select_basin_data(ds_clim_sel, basin)

        # Add the bias to the climatology.
        ds_selection['temp'] = ds_selection.temp.groupby('time.month') +\
            ds_clim_sel.temp
        ds_selection['prcp'] = ds_selection.prcp.groupby('time.month') +\
            ds_clim_sel.prcp
        # Add some attributes to the data.
        ds_selection.attrs = {'basin': basin.RIVER_BASI, 'MRBID': basin.MRBID}
        ds_selection.temp.attrs = {'unit': 'K'}
        ds_selection.prcp.attrs = {'unit': 'mm month-1'}

    return ds_clim_sel, ds_selection


def select_glaciers(basin, gdf):
    '''Function to select the glaciers within a basin.
    -----
    arguments:
    basin: geopandas dataframe of the basin (one shapefile)
    gdf: geopandas dataframe containing the glaciers of the
    region.

    returns:
    geopandas dataframe of the glaciers within the basin.

    '''
    in_bas = [basin.geometry.contains(shpg.Point(x, y))[0]
              for (x, y) in zip(gdf.CenLon, gdf.CenLat)]
    return gdf.loc[in_bas]


def run_hydro_projections(gdirs, rcps):
    '''Small wrapper for running hydro simulations
    arguments:
    gdirs: glacier directories.
    rcps: list of rcp scenarios to run.
    '''

    for rcp in rcps:
        # Download the files
        ft, fp = download_proj_data(rcp)
        # bias correct them
        workflow.execute_entity_task(gcm_climate.process_cmip_data, gdirs,
                                     # recognize the climate file for later
                                     filesuffix='_CCSM4_{}'.format(rcp),
                                     # temperature projections
                                     fpath_temp=ft,
                                     # precip projections
                                     fpath_precip=fp,
                                     )

    for rcp in rcps:
        rid = f'_CCSM4_{rcp}'
        workflow.execute_entity_task(
                             tasks.run_with_hydro,  gdirs,
                             run_task=tasks.run_from_climate_data,
                             ys=2020,
                             # Use gcm_data
                             climate_filename='gcm_data',
                             # Use the scenario
                             climate_input_filesuffix=rid,
                             # When to start?
                             init_model_filesuffix='_historical',
                             # Good naming for recognizing later
                             output_filesuffix=rid,
                             # Store monthyl?
                             store_monthly_hydro=True,
                            )


def process_basins(basins, rcps, data_dir=None):
    '''Process the climate data for one or more basin(s). Takes a list of basins
    (geoseries) and downloads projection data, selects the region of interest
    and downscales it. Saves the basin to disk. If data_dir is not provided,
    data is saved to a temporary folder.

    Args:
    basins : Geopandas geoseries with basin spatial data. One or more basins.
    rcps : list of strings with rcp scenarios.
    data_dir : str : where to store the data. Provide an absolute path.

    Returns:
    Nothing.
    '''

    # Where are we saving the data?
    if data_dir is None:
        tmp_dir = tempfile.gettempdir()
        data_dir = mkdir(tmp_dir, 'basin_data')
    # Check the path.
    else:
        if not os.path.exists(data_dir):
            raise ValueError('data_dir is not a valid path.')
    # After we checked the path we can begin processing the basins. Want to use
    # multi processing to speed things up.

    # Subfunction for multiprocessing.
    def processing(basin, rcp, data_dir=data_dir):
        # Create the basin dir
        bid = str(basin.MRBID)
        basin_dir = mkdir(data_dir, bid)
        # Get the data
        _, ds_selection = process_clim_data(basin, rcp)
        # Save it to file.
        path = os.path.join(basin_dir, f'{bid}_{rcp}.nc')
        ds_selection.to_netcdf(path=path)

    # Create all the combinations that should be processed.
    iter_prod = list(itertools.product(basins.itertuples(), rcps))
    basins_prod, rcps_prod = zip(*iter_prod)
    # Map with the mp pool
    with ProcessPool() as p:
        p.map(processing, basins_prod, rcps_prod)
        # Have to close the pool.
        p.close()
        p.join()


def mkdir(path, folder):
    '''Create directory'''
    path = os.path.join(path, folder)
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    return path
