import os
import logging
import sys

# oggm
import oggm.cfg as cfg
import oggm.utils as utils

# gha
from gha.glacier import glacier_simulations, compile_basin_output
from gha.utils import process_basin

import geopandas as gpd
import pandas as pd

# OGGM init.
cfg.initialize(logging_level='WARNING')
cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['use_multiprocessing'] = True
cfg.PARAMS['border'] = 80

# Set the working dir.
WORKING_DIR = os.environ.get('OGGM_WORKDIR', '')
if not WORKING_DIR:
    raise RuntimeError('Need a working dir')
utils.mkdir(WORKING_DIR)
cfg.PATHS['working_dir'] = WORKING_DIR

# Output dir
OUTPUT_DIR = os.environ.get('OGGM_OUTDIR', '')
if not OUTPUT_DIR:
    raise RuntimeError('Need an output dir')
utils.mkdir(OUTPUT_DIR)

# Logging
log = logging.getLogger(__name__)
log.workflow('Starting run for basins')

# Read in the basin file.
gdf = gpd.read_file('./data/glacier_basins.shp')
# Read in the gcm info.
gcm_df = pd.read_csv('/home/www/oggm/cmip6/all_gcm_list.csv', index_col=0)

# Run the glacier simulations and basin processing.
for basin_idx, gcm_idx in zip(sys.argv[1], sys.argv[2]):
    # Get the basin and the MRBID.
    basin = gdf.iloc[[basin_idx]]
    mrbid = str(basin.iloc[0].MRBID)
    # Get the gcm
    gcm = gcm_df.loc[gcm_df.gcm == gcm_df.gcm.unique()[gcm_idx]]
    # ssp scenarios we want to run.
    # These are the ssp scenarios we want to do, but not all gcm have them.
    # So have to check what it has.
    ssps_wanted = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    # Save the ones we get.
    ssps = []
    # Loop over wanter ssps.
    for ssp in ssps_wanted:
        df2 = gcm.loc[gcm['ssp'] == ssp]
        # If it contains tas and pr, keep it.
        if len(df2) == 2:
            ssps.append(ssp)
    # log
    log.workflow(f'Starting run for {mrbid}')
    # Glacier run
    glacier_simulations(mrbid, gcm, ssps, OUTPUT_DIR, restart=True)

    log.workflow('OGGM done')
    log.workflow('Begin basin climate processing and hydro analysis')
    # Process the basin climate data.
    process_basin(basin, gcm, ssps, OUTPUT_DIR)
    log.workflow('Processing done')

log.workflow('Basin completed')
