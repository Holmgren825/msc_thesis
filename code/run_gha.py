import os
import logging
import sys

# oggm
import oggm.cfg as cfg
import oggm.utils as utils

# gha
from gha.glacier import glacier_simulations, compile_basin_output
from gha.utils import process_basins
from gha.hydro import get_discharge_df

import geopandas as gpd

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

gdf = gpd.read_file('./data/glacier_basins.shp')
# Run the glacier simulations and basin processing.
for basin_idx in sys.argv[1:]:
    # Get the basin and the MRBID.
    basin = gdf.iloc[[basin_idx]]
    mrbid = str(basin.iloc[0].MRBID)
    # log
    log.workflow(f'Starting run for {mrbid}')
    # Glacier run
    rcps = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
    glacier_simulations(mrbid, rcps, restart=True)

    # Compile the basin glaciers.
    for rcp in rcps:
        compile_basin_output(mrbid, rcp, OUTPUT_DIR)
    log.workflow('OGGM done')
    log.workflow('Begin basin climate processing')
    # Process the basin climate data.
    process_basins(basin, rcps, OUTPUT_DIR)
    log.workflow('Climate processing done')

log.workflow('Basin completed')
