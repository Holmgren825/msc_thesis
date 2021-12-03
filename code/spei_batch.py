import argparse
import os
from gha.hydro import calc_SPEI
import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np


def get_spei_files(base_path, window=15, parametric=False, reference=False):
    '''Create netcdf files with SPEI data in batch. Useful on
    a cluster which cant run the SPEI calulation because of missing
    dependencies. Saves the files to disk.

    Args:
    -----
    window: int
        Size of the window (months) for accumulating moisture in SPEI.
    parametric: bool
        Calculate a parametric SPEI of not.
    '''
    # First we need a loop to go over all the basins.
    # Easy way to get all the mrbids.
    basins_df = gpd.read_file('./data/glacier_basins.shp')
    # Get the gcms
    gcm_df = pd.read_csv('/home/www/oggm/cmip6/all_gcm_list.csv', index_col=0)
    gcms = gcm_df.gcm.unique()
    gcms = np.delete(gcms, 10)

    # Rcp scenarios
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    # What about the reference period?
    hist = None
    hist_str = 'w.o.'
    hist_pth = 'nref'
    # Main loop.
    for mrbid in basins_df.MRBID:
        # Set it to a string.
        mrbid = str(mrbid)
        print(f'Basin {mrbid}')
        # Base path to file.
        # This is where the data is stored.
        basin_path = os.path.join(base_path, mrbid)

        for gcm in gcms:
            spei_list = []
            for ssp in ssps:
                # Some gcms don't have all the ssps
                try:
                    print(f'SSP {ssp}')
                    # Filenames
                    hist_file = f'{mrbid}_discharge_hist_{gcm}_{ssp}.nc'
                    proj_file = f'{mrbid}_discharge_proj_{gcm}_{ssp}.nc'
                    hist_path = os.path.join(basin_path, hist_file)
                    proj_path = os.path.join(basin_path, proj_file)
                    print(proj_path)
                    # Open the files.
                    with xr.open_dataset(hist_path) as ds_hist,\
                            xr.open_dataset(proj_path) as ds_proj:

                        # Subselection
                        if reference:
                            hist = ds_hist.sel(time=slice('1918', '2018'))
                            hist_str = 'w.'
                            hist_pth = 'wref'
                        # Calc SPEI
                        SPEI = calc_SPEI(ds_proj.D, hist.D, window, parametric)
                        # Adjusted SPEI
                        SPEI_adj = calc_SPEI(ds_proj.D_adj, hist.D, window, parametric)
                        # Add some attributes
                        SPEI.attrs = {'unit': 'mm month-1', 'description':
                                      f'SPEI calculated w.o. glacier runoff, {hist_str} reference'}
                        SPEI_adj.attrs = {'unit': 'mm month-1', 'description':
                                          f'SPEI calculated w. glacier runoff, {hist_str} reference'}
                        # Create the SPEI dataset.
                        SPEI_ds = xr.Dataset({f'SPEI_{ssp}': SPEI, f'SPEI_adj_{ssp}': SPEI_adj})
                        spei_list.append(SPEI_ds)
                except KeyError:
                    pass
                except FileNotFoundError:
                    pass

                # Merge the list.
                ds = xr.merge(spei_list)
                ds.attrs = {'GCM': gcm}

                # Save this.
                path = os.path.join(basin_path,
                                    f'{mrbid}_SPEI_{gcm}_{parametric}_{hist_pth}.nc')
                ds.to_netcdf(path=path)


def main():
    # Get the args
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument('--path', help='Path to where the basins are stored.',
                        type=str)
    parser.add_argument('--window',
                        help='Size of the window to accumulate moisture.',
                        type=int)
    parser.add_argument('--parametric', dest='parametric',
                        help='Parametric SPEI',
                        action='store_true')
    parser.add_argument('--non-parametric', dest='parametric',
                        help='Non parametric SPEI',
                        action='store_false')
    parser.add_argument('--reference', dest='reference',
                        help='With reference period',
                        action='store_true')
    parser.add_argument('--no-reference', dest='reference',
                        help='No reference period',
                        action='store_false')
    parser.set_defaults(parametric=False)
    args = parser.parse_args()
    get_spei_files(base_path=args.path, window=args.window,
                   parametric=args.parametric, reference=args.reference)


if __name__ == '__main__':
    main()
