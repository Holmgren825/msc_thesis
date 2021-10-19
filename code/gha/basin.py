'''Module containing basin relevant functions.'''
from gha.utils import get_cmip6_data, download_clim_data, select_basin_data
from gha.utils import init_data_dir, mkdir
from gha import hydro
from oggm import cfg
import xarray as xr
import numpy as np
import os


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

        # Subselection of projection data
        ds_t = ds_t.sel(time=slice('1950', '2100'))
        ds_p = ds_p.sel(time=slice('1950', '2100'))

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
        rid, _, _ = get_cmip6_data(gcm, ssp)
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
        hydro.basin_hydro_analysis(basin, gcm, ssp, window,
                                   parametric, data_dir)
