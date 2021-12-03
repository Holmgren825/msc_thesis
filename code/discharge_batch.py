import argparse
import os
from gha.hydro import basin_hydro_analysis
import xarray as xr
import geopandas as gpd


def get_discharge_files(base_path, window=15, parametric=False):
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
    for i in range(75):
        basin = basins_df.iloc[[i]]
        # Set it to a string.
        mrbid = str(basin.iloc[0].MRBID)
        print(f'Basin {mrbid}')

        for rcp in rcps:
            print(f'Rcp {rcp}')
            # Run the basin hydro analysis
            basin_hydro_analysis(basin, rcp, window=window, parametric=parametric,
                                 data_dir = base_path)
            

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
    parser.set_defaults(parametric=False)
    args = parser.parse_args()
    get_discharge_files(base_path=args.path, window=args.window,
                   parametric=args.parametric)


if __name__ == '__main__':
    main()
