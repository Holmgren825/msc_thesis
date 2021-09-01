import argparse
import os
from gha.hydro import calc_SPEI
import xarray as xr
import geopandas as gpd


def get_spei_files(base_path, window=15, parametric=False):
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
    # Rcp scenarios
    rcps = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
    # Main loop.
    for mrbid in basins_df.MRBID:
        # Set it to a string.
        mrbid = str(mrbid)
        # Base path to file.
        # This is where the data is stored.
        basin_path = os.path.join(base_path, mrbid)

        for rcp in rcps:
            # Filenames
            hist_file = f'{mrbid}_discharge_hist_{rcp}.nc'
            proj_file = f'{mrbid}_discharge_proj_{rcp}.nc'
            hist_path = os.path.join(basin_path, hist_file)
            proj_path = os.path.join(basin_path, proj_file)
            # Open the files.
            with xr.open_dataset(hist_path) as ds_hist,\
                    xr.open_dataset(proj_path) as ds_proj:
                # Subselection
                hist = ds_hist.sel(time=slice('1960', '2010'))
                # Calc SPEI
                SPEI = calc_SPEI(ds_proj.D, hist.D, window, parametric)
                # Adjusted SPEI
                SPEI_adj = calc_SPEI(ds_proj.D_adj, hist.D, window, parametric)
                # Create the SPEI dataset.
                SPEI_ds = xr.Dataset({'SPEI': SPEI, 'SPEI_adj': SPEI_adj})
                # Add some attributes
                SPEI_ds.SPEI.attrs = {'unit': 'mm month-1'}
                SPEI_ds.SPEI_adj.attrs = {'unit': 'mm month-1'}
                SPEI_ds.SPEI.attrs = {'description':
                                      'SPEI calculated without glacier runoff.'}
                SPEI_ds.SPEI_adj.attrs = {'description':
                                          'SPEI calculated with glacier runoff.'}

                # Save this as well.
                path = os.path.join(basin_path,
                                    f'{mrbid}_SPEI_{parametric}_{rcp}.nc')
                SPEI_ds.to_netcdf(path=path)


def main():
    # Get the args
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument('--path', help='Path to where the basins are stored.',
                        type=str)
    parser.add_argument('--window',
                        help='Size of the window to accumulate moisture.',
                        type=int)
    parser.add_argument('--parametric', help='Parametric SPEI or not',
                        type=bool)
    args = parser.parse_args()

    get_spei_files(args.base_path, args.window, args.parametric)


if __name__ == '__main__':
    main()
