'''Hydro related functions'''
import numpy as np
import os
import xarray as xr
from scipy.stats import fisk


def calc_PET(ds):
    '''Function used to calculate PET. PET has the unit of mm/month.

    Args:
    ----
    ds: xarray dataset
        Dataset with the temperature and precipitation projection data.

    Returns:
    --------
    ds: xarray dataset
        Updated data with projected temperature, precipitation and PET.
    '''
    # For the correction.
    # Average Julian day, is this then the average of all
    # the julian dates of the month?
    julian = ds.indexes['time'].to_datetimeindex().to_julian_date().values
    # Get the latitude from the ds
    # Convert the lat to radians
    latitude = np.deg2rad(ds.lat).values
    latitude = latitude.reshape(-1, 1, 1)
    latitude = np.broadcast_to(latitude, (len(ds.lat), len(ds.lon),
                                          len(ds.time)))
    # Calculate the days in each month
    NDM = ds.indexes['time'].to_datetimeindex().days_in_month.values
    NDM = NDM.reshape(1, 1, -1)
    # Calc solar declination
    solar_declination = 0.4093 * np.sin((2 * np.pi * julian / 365) - 1.405)
    # Reshape for broadcasting.
    solar_declination = solar_declination.reshape(1, 1, -1)
    # Calculate the hourly angle of sun rising
    h_sun_angle = np.arccos(-np.tan(latitude) * np.tan(solar_declination))
    # Maximum number of sun hours
    N = (24 / np.pi) * h_sun_angle
    # Correction coefficient
    K = (N / 12) * (NDM / 30)
    # Uncorrected PET.
    # The annual heat index
    temp = ds.temp - 273.15
    t_index = ((temp / 5)**1.514).groupby('time.year').sum()

    # We repeat each value 12 times.
    # Get m
    m = ((6.75 * 1e-7) * t_index**3) - ((7.71 * 1e-5) * t_index**2) +\
        ((1.79 * 1e-2) * t_index) + 0.49239
    # Final PET
    PET_u = (temp.groupby('time.year') / t_index)
    PET_u = PET_u.groupby('time.year') ** m
    PET_u = PET_u * 10

    # Add the corrected PET to the dataset.
    ds['PET'] = PET_u * 16 * K
    ds['PET'] = ds['PET'].fillna(0)

    ds.PET.attrs = {'unit': 'mm month-1'}
    return ds


def get_discharge_df(basin, data_dir, rcp):
    '''Generate a xarray dataframe containing the compiled runoff from the oggm,
    the temperature, precipitation and PET from the projection data netcdf.

    Args:
    -----
    basin: geopandas dataframe
        Dataseries of the basin. Should contain the geometry.
    data_dir: str
        Path of the data directory.
    rcp: str
         String declaring the rcp scenario to process.
    '''
    bid = str(basin.iloc[0].MRBID)
    basin_dir = os.path.join(data_dir, bid)
    # Paths for the files. Leave the naming structure hard coded for now.
    # The compiled glacier output.
    glacier_path = os.path.join(basin_dir,
                                f'oggm_compiled_{bid}_CCSM4_{rcp}.nc')
    # Projection data path
    proj_path = os.path.join(basin_dir, f'{bid}_{rcp}.nc')

    with xr.open_dataset(glacier_path, use_cftime=True) as ds_gl,\
            xr.open_dataset(proj_path, use_cftime=True) as ds_proj:
        # Sum the glaciers in the compiled run.
        ds_gl = ds_gl.sum(dim='rgi_id')
        # Get the largest glaciated area.
        glaciated_area = ds_gl.area.max()
        # Get the basin area.
        basin_area = basin.to_crs({'proj': 'cea'}).area.iloc[0]

        # Lets get the monthly runoff from the glaciers
        # This should work in both hemispheres maybe?
        ds_roll = ds_gl.roll(month_2d=ds_gl['calendar_month_2d'].data[0] - 1,
                             roll_coords=True)
        ds_roll['month_2d'] = ds_roll['calendar_month_2d']

        # Select only the runoff variables
        monthly_runoff = (ds_roll['melt_off_glacier_monthly'] +
                          ds_roll['melt_on_glacier_monthly'] +
                          ds_roll['liq_prcp_off_glacier_monthly'] +
                          ds_roll['liq_prcp_on_glacier_monthly'])
        # Convert to kg m^-2
        monthly_runoff = monthly_runoff / glaciated_area
        # Clip runoff to 0
        monthly_runoff = monthly_runoff.clip(0)

        # Get the total precipitation and PET
        hydro_ds = ds_proj[['prcp', 'PET']].mean(dim=['lat', 'lon'],
                                                 keep_attrs=True)
        # Projection hydro subset
        hydro_proj_ds = hydro_ds.sel(time=slice('2019', '2100'))
        # We add the glacier projections to this dataset.
        time = hydro_proj_ds.time
        runoff = xr.DataArray(monthly_runoff.values.flatten(),
                              dims=['time'],
                              coords={'time': time})
        hydro_proj_ds = hydro_proj_ds.assign(glacier_runoff=runoff)
        # Attributes
        hydro_proj_ds.glacier_runoff.attrs = {'unit': 'mm month-1'}
        # Area adjustments for precipitation and glacier runoff
        glacier_runoff_adj = hydro_proj_ds['glacier_runoff'] *\
            (glaciated_area / basin_area)
        hydro_proj_ds = hydro_proj_ds.assign(
            glacier_runoff_adj=glacier_runoff_adj
        )
        # Attributes
        hydro_proj_ds.glacier_runoff_adj.attrs = {'unit': 'mm month-1'}

        # Adjusted precipitation
        prcp_adj = get_adjusted_precipitation(hydro_proj_ds,
                                              basin_area, glaciated_area)
        hydro_proj_ds = hydro_proj_ds.assign(prcp_adj=prcp_adj)
        # Attributes
        hydro_proj_ds.prcp_adj.attrs = {'unit': 'mm month-1'}

        # And finally calculate the moisture availability
        D = hydro_proj_ds['prcp'] - hydro_proj_ds['PET']
        hydro_proj_ds = hydro_proj_ds.assign(D=D)
        hydro_proj_ds.D.attrs = {'unit': 'mm month-1'}
        # Adjusted
        D_adj = hydro_proj_ds['prcp_adj']\
            + hydro_proj_ds['glacier_runoff_adj'] - hydro_proj_ds['PET']
        hydro_proj_ds = hydro_proj_ds.assign(D_adj=D_adj)
        hydro_proj_ds.D_adj.attrs = {'unit': 'mm month-1'}
        # Reference
        D = hydro_ds['prcp'] - hydro_ds['PET']
        hydro_ds = hydro_ds.assign(D=D)
        hydro_ds.D.attrs = {'unit': 'mm month-1'}

        return hydro_proj_ds, hydro_ds


def get_adjusted_precipitation(ds, basin_area, glaciated_area):
    '''Calculate the adjusted precipitation. This is simple version for now
    but could possbile be done by masking the glaciated area in the
    downscaled data. Not sure if necessary but would be nice.'''

    ice_free = basin_area - glaciated_area
    prcp_adj = ds['prcp'] * ice_free / basin_area

    return prcp_adj


def calc_SPEI(ds, ds_hist, window, parametric=True):
    '''Calculate SPEI for the given dataset.

    Args:
    -----
    ds: xarray dataset
       Containing data (moisture) from which to do the SPEI calculation.
       D (prcp - PET) is needed.
    ds_hist: xarray dataset
        Containing the reference period. Discharge.
    window: int
        Size of the window (months) for the accumulated moisture.

    Returns:
    --------
    SPEI: xarray dataArray
    '''
    # Start with calculating the rolling sum of desired length.
    reference = ds_hist.rolling(time=window).sum().dropna(dim='time')

    # Constants, standardizing the percentiles,
    C0 = 2.515517
    C1 = 0.802853
    C2 = 0.010328
    d1 = 1.432788
    d2 = 0.1819269
    d3 = 0.001308
    # Calculate the rolling sum of the investigated runoff.
    D = ds.rolling(time=window).sum().dropna(dim='time')
    if parametric:
        # Fit the fisk distribution.
        fit = fisk.fit(reference)
        # Calc P, use the cdf, paper is wrong saying the pdf.
        P = 1 - fisk.cdf(D, *fit)
        # Calc W for P <= 0.5 and P>0.5.
        W = np.where(P < 0.5, np.sqrt(-2 * np.log(P)),
                     -np.sqrt(-2 * np.log(1 - P)))
        # Calc SPEI
        SPEI = W - (C0 + C1 * W + C2 * W**2) /\
            (1 + d1 * W + d2 * W**2 + d3 * W**3)
    # Non-parametric
    else:
        # Get the rank
        rank = D.rank(dim='time')
        # Empirical probability
        P = (rank - 0.44) / (len(rank) + 0.12)
        # Standardize it
        W = np.where(P <= 0.5, -np.sqrt(np.log(1 / P**2)),
                     np.sqrt(1 / np.log(1 / (1 - P)**2)))
        # Calc SPEI
        SPEI = W - (C0 + C1 * W + C2 * W**2) /\
            (1 + d1 * W + d2 * W**2 + d3 * W**3)
    # Put into dataframe
    SPEI = xr.DataArray(SPEI, dims=['time'], coords={'time': D.time})
    return SPEI
