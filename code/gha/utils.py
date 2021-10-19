'''Utility functions for hydro analysis of glaciated basins with the OGGM.'''

import xarray as xr
import numpy as np
import shapely.geometry as shpg
from oggm import utils, workflow, tasks, cfg
from oggm.shop import gcm_climate
import zipfile
import tempfile
import itertools
import os
import json
from gha import hydro
from gha.hydro import basin_hydro_analysis


def download_proj_data(rcp):
    '''
    Small helper function to download CCSM4 climate projection data
    from OGGMs archive. Makes use of the OGGM file downloader.

    Args:
    -----
    rcp: list(strings)
        List of rcp scenarios to download.

    Returns:
    --------
    ft: string
        Path to temperature file
    fp: string
        Path to precipitation files
    '''
    # Base file paths
    # Note that these dont contain any scenario specification.
    bp = 'https://cluster.klima.uni-bremen.de/~oggm/cmip5-' + \
         'ng/pr/pr_mon_CCSM4_{}_r1i1p1_g025.nc'
    bt = 'https://cluster.klima.uni-bremen.de/~oggm/cmip5-' + \
         'ng/tas/tas_mon_CCSM4_{}_r1i1p1_g025.nc'
    # Download the data
    ft = utils.file_downloader(bt.format(rcp))
    fp = utils.file_downloader(bp.format(rcp))

    return ft, fp


def get_cmip6_data(gcm, ssp):
    '''
    Helper function to return filepaths of temperature and precipitation for
    the gcm, located on the cluster. Also gives back the rid. This does not
    download anyting. Files need to be available.

    Args:
    -----
    gcm: pandas dataframe
        Contains metadata about the gcm and the filepaths etc.
    ssp: string
        Specifies the ssp scenario.

    Returns:
    --------
    rid: string
        Gcm file identifier.
    ft: string
        Path to the temperature dataset on the cluster.
    fp: string
        Path to the precipitation dataset on the cluster.
    '''

    # Select the entires relevant for our ssp.
    gcm_ssp = gcm.loc[gcm.ssp == ssp]
    # Temperature entry
    ft = gcm_ssp.loc[gcm_ssp['var'] == 'tas'].iloc[0]
    # Precipitation path.
    fp = gcm_ssp.loc[gcm_ssp['var'] == 'pr'].iloc[0].path
    # rid
    rid = ft.fname.replace('_r1i1p1f1_tas.nc', '')
    # Temp. path
    ft = ft.path

    return rid, ft, fp


def download_clim_data():
    '''Helper function to download reference climate data (CRU).

    Returns:
    --------
    Path (string) to the climate data.
    '''

    f = 'https://cluster.klima.uni-bremen.de/~oggm/climate/cru/cru_cl2.nc.zip'
    fp = utils.file_downloader(f)
    # Unzip it.
    with zipfile.ZipFile(fp, 'r') as zip_ref:
        zip_ref.extractall(fp[:-7])

    return fp[:-7] + '/cru_cl2.nc'


def select_basin_data(df, basin):
    '''Makes a selection in an xarray dataset based on the provided basin
    (shapefile).

    Args:
    -----
    df: xarray dataset.
        Can contain anything basically. But should be
        climate data.  Should cover the coordinates of the basin.
    basin: geopandas series
        Geopandas series of a basin. Has to contain a basin
        shapefile (geometry).

    Returns:
    df_sel: xarray dataset
        Contains only the data within the basin.
    '''
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


def process_clim_data(basin, gcm, ssp):
    '''Downscales climate projection data to the basin for a rcp scenario.

    Args:
    -----
    basin: geopandas series
        Geopandas series of the basin.
    gcm: pandas dataframe
        Contains metadata about the gcm and the filepaths etc.
    ssp: string
        String decaling the ssp scenario.

    Returns:
    --------
    ds_clim_sel: xarray dataset
        Dataset with the selected climate data.
    ds_selection: xarray dataset
        Contains the downscaled (bias corrected) temperature and precipitation
        data for the given basin and rcp scenario.
    '''

    # Extract the pandas series.
    basin = basin.iloc[0]
    # Get the projection data for the scenarios
    rid, ft, fp = get_cmip6_data(gcm, ssp)
    # And get the climate data
    fclim = download_clim_data()

    # Open the datasets
    # Temperature, precipitation and climate
    with xr.open_dataset(ft, use_cftime=True) as ds_t,\
            xr.open_dataset(fp, use_cftime=True) as ds_p,\
            xr.open_dataset(fclim, use_cftime=True) as ds_clim:
        # Longitude correcction of ds.
        for ds in [ds_t, ds_p, ds_clim]:
            if ds.lon.max() >= 181.0:
                ds['lon'] = ds.lon - 180.0
        # Let's do a coarse selection of the data first.
        minlon, minlat, maxlon, maxlat = basin.geometry.bounds
        # Pad the box
        pad = 2.5
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
        assert r == 0
        dimo = [cfg.DAYS_IN_MONTH[m - 1] for m in
                ds_p_sel['time.month']]
        # We need some numpy functionality.
        dimo = np.array(dimo)
        # Reshape for broadcasting. We want to multiply all coordinates.
        dimo = dimo.reshape(-1, 1, 1)
        # Multiply with seconds per month.
        ds_p_sel['pr'] = ds_p_sel.pr * dimo * (60 * 60 * 24)
        # Convert clim temp to Kelvin
        ds_clim_sel['temp'] = ds_clim_sel.temp + 273.15

        # Can now compute the anomalies, and do the bias correction.
        # Temperature
        ds_t_avg = ds_t_sel.sel(time=slice('1981', '2018'))
        ds_t_avg = ds_t_avg.groupby('time.month').mean(dim='time')
        # Anomaly
        ds_t_ano = ds_t_sel.groupby('time.month') - ds_t_avg

        # Precipitation
        ds_p_avg = ds_p_sel.sel(time=slice('1981', '2018'))
        ds_p_avg = ds_p_avg.groupby('time.month').mean(dim='time')
        # Anomaly
        ds_p_ano = ds_p_sel.groupby('time.month') - ds_p_avg
        # Scaled anomaly? Not used for now.
        # ds_p_scaled = ds_p_sel.groupby('time.month') / ds_p_avg

        # Now we can do the downscaling/interpolation.
        # Temperature
        ds_t_sel_remap = ds_clim_sel.salem.grid.map_gridded_data(
                            ds_t_ano.tas.values, ds_t_ano.salem.grid,
                            interp='linear')

        # Precipitation
        ds_p_sel_remap = ds_clim_sel.salem.grid.map_gridded_data(
                            ds_p_ano.pr.values, ds_p_ano.salem.grid,
                            interp='linear')

        # Need the time series.
        time = ds_t_sel.time
        # Create new DataArrays
        # Temperature
        ds_t_sel = xr.DataArray(ds_t_sel_remap, dims=['time', 'lat', 'lon'],
                                coords={'time': time, 'lat': ds_clim_sel.lat,
                                        'lon': ds_clim_sel.lon}, name='temp')
        # Precipitation
        ds_p_sel = xr.DataArray(ds_p_sel_remap, dims=['time', 'lat', 'lon'],
                                coords={'time': time, 'lat': ds_clim_sel.lat,
                                        'lon': ds_clim_sel.lon}, name='prcp')
        # Put it all in a dataset.
        ds_selection = xr.Dataset({'temp': ds_t_sel, 'prcp': ds_p_sel})

        # Final selection of data for the basin, i.e. finer selection.
        # Maybe more efficient to this earlier, interpolation on less data.
        ds_selection = select_basin_data(ds_selection, basin)
        # Climatology
        ds_clim_sel = select_basin_data(ds_clim_sel, basin)

        # Add the climatology (each month) to the projected anomaly.
        ds_selection['temp'] = ds_selection.temp.groupby('time.month') +\
            ds_clim_sel.temp
        ds_selection['prcp'] = ds_selection.prcp.groupby('time.month') +\
            ds_clim_sel.prcp
        ds_selection['prcp'] = ds_selection['prcp'].clip(0)
        # Add some attributes to the data.
        ds_selection.attrs = {'basin': basin.RIVER_BASI, 'MRBID': basin.MRBID,
                              'gcm': rid}
        ds_selection.temp.attrs = {'unit': 'K'}
        ds_selection.prcp.attrs = {'unit': 'mm month-1'}

    return ds_clim_sel, ds_selection


def select_glaciers(basin, gdf):
    '''Function to select the glaciers within a basin.

    Args:
    -----
    basin: geopandas series
        Series of the the basin (one shapefile).
    gdf: geopandas dataframe
        Containing the glaciers of the region.

    Returns:
    --------
    geopandas dataframe of the glaciers within the basin.
    '''
    in_bas = [basin.geometry.contains(shpg.Point(x, y))[0]
              for (x, y) in zip(gdf.CenLon, gdf.CenLat)]
    return gdf.loc[in_bas]


def run_hydro_projections(gdirs, gcm, ssps, data_dir, basin):
    '''Small wrapper for running hydro simulations with the OGGM.

    Args:
    -----
    gdirs: list
        List of glacier directories.
    gcm: pandas dataframe
        Contains metadata about the gcm and the filepaths etc.
    ssps: list (strings)
        List of ssps scenarios to run.
    data_dir: string
        Path to the directory where to store the data.
    basin: string
        String of the basin MRBID. I.e. '3209'.
    '''

    for ssp in ssps:
        # Download the files
        rid, ft, fp = get_cmip6_data(gcm, ssp)
        # bias correct them
        workflow.execute_entity_task(gcm_climate.process_cmip_data, gdirs,
                                     # recognize the climate file for later
                                     filesuffix='_' + rid,
                                     # temperature projections
                                     fpath_temp=ft,
                                     # precip projections
                                     fpath_precip=fp,
                                     year_range=('1981', '2018'),
                                     )

    for ssp in ssps:
        rid, _, _ = get_cmip6_data(gcm, ssp)
        workflow.execute_entity_task(
                             tasks.run_with_hydro, gdirs,
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

        # Where do we store the data?
        data_dir = init_data_dir(data_dir)
        basin_dir = mkdir(data_dir, basin)
        # File suffixes
        output_suffix = f'oggm_compiled_{basin}_{rid}.nc'
        # If we do a custom path, we have to supply the whole path to OGGM.
        path = os.path.join(basin_dir, output_suffix)
        # Compile the output
        utils.compile_run_output(gdirs,
                                 path=path,
                                 input_filesuffix=rid,
                                 )


def process_basin(basin, gcm, ssps, data_dir=None,
                  parametric=False, window=15):
    '''Process the climate data for one basin. Takes a basin
    (geoseries) and downloads projection data, selects the region of interest
    and downscales it for the specified rcp scenarios. Saves the basin to disk.
    If data_dir is not provided, data is saved to a temporary folder.

    Args:
    -----
    basin: geopandas dataframe
        Dataframe with basin spatial data.
    ssps: list
        List of strings with rcp scenarios.
    gcm: pandas dataframe
        Contains metadata about the gcm and the filepaths etc.
    data_dir: str
        Path to where to store the data. Provide an absolute path.
    parametric: bool
        Calculate paramteric SPEI or not. Default is  False.
    window: int
        Size of the window used for calculating SPEI. Default is 15.
    '''

    # Where are we saving the data?
    data_dir = init_data_dir(data_dir)
    # After we checked the path we can begin processing the basins.
    # Loop climate processing over the rcps scenarios.
    # Create the basin dir
    bid = str(basin.iloc[0].MRBID)
    basin_dir = mkdir(data_dir, bid)
    # Loop the scenarios.
    for ssp in ssps:
        # Get the rid
        rid, _, _ = get_cmip6_data(gcm,)
        # Get the data
        _, ds_selection = process_clim_data(basin, gcm, ssp)
        # Calc PET. This is inserting PET into the dataset.
        ds_selection = hydro.calc_PET(ds_selection)
        # Save it to file.
        path = os.path.join(basin_dir, f'{bid}_{rid}.nc')
        ds_selection.to_netcdf(path=path)
        # Hydro analysis.
        # This opens the glacier run-off file and the climate projection data
        # and compiles it to the discharge dataframes. After this the SPEI
        # dataframe is created and and saved to disk.
        basin_hydro_analysis(basin, ssp, window, parametric, data_dir)


def init_data_dir(data_dir):
    '''Initialize the data directory. Takes care of make the directory
    if it doesn't exist. Otherwise it returns the correct path. If none,
    it creates a directory in the temp dir.

    Args:
    -----
    data_dir: str
        Path to data directory.

    Returns:
    --------
    data_dir: str
        Path to data directory.
    '''
    # Where are we saving the data?
    if data_dir is None:
        tmp_dir = tempfile.gettempdir()
        data_dir = mkdir(tmp_dir, 'basin_data')
    # Check the path.
    else:
        if not os.path.exists(data_dir):
            raise ValueError('data_dir is not a valid path.')

    return data_dir


def mkdir(path, folder):
    '''Create a new folder at specified path.'''
    path = os.path.join(path, folder)
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    return path


def select_glaciers_json(basin='all'):
    '''Select glaciers within a basin by MRBID from a json-file,
    which is stoored in the data directory.

    Args:
    -----
    basin: str
        String of MRBID or 'all'.

    Returns:
    --------
    If basin is 'all' a list of all relevant glaciers is returned, for
    initiating glacier simulations. If basin is a MRBID the list of glaciers
    within that basin is returned.
    '''

    fpath = './data/rgi_ids_per_basin.json'
    with open(fpath) as f:
        basin_dict = json.load(f)

    if basin.lower() != 'all':
        glacier_list = basin_dict[basin]
    else:
        glacier_list = list(itertools.chain.from_iterable(basin_dict.values()))

    return glacier_list
