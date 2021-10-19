'''Utility functions for hydro analysis of glaciated basins with the OGGM.'''

import shapely.geometry as shpg
import oggm.utils
import zipfile
import tempfile
import itertools
import os
import json


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
    ft = oggm.utils.file_downloader(bt.format(rcp))
    fp = oggm.utils.file_downloader(bp.format(rcp))

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
    fp = oggm.utils.file_downloader(f)
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
