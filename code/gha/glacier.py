'''Module for hydro calculations.'''
import os
from oggm import utils, workflow, tasks
from oggm.shop import gcm_climate
import gha.utils


def glacier_simulations(basin, gcm, ssps, data_dir, restart=False, test=False):
    '''Run glacier simulations for all glaciers within a basin.
    Compiles the output.

    Args:
    -----
    basin: str
        String of the basin MBRID. I.e. '3209'. To simulate glaciers in
        all the basins simply pass 'all'.
    gcm: pandas dataframe
        Contains metadata about the gcm and the filepaths etc.
    rcps: list(strings)
        List of strings decalring the ssp scenarios to run.
    data_dir: string
        Path to the directory where to store the data.
    restart: bool
        If to restart the the gdirs. If this is true, glacier directories (all)
        model data will be reset.
    test: bool
        If to run just a subset of the glaciers. For testing.
    '''
    # Begin with fetching the ids for all the glaciers.
    rgiids = gha.utils.select_glaciers_json(basin=basin)
    # For testing we only simulate one glacier.
    if test:
        rgiids = rgiids[:2]

    # # OGGM init.
    # cfg.initialize(logging_level='WARNING')
    # cfg.PARAMS['continue_on_error'] = True
    # cfg.PARAMS['use_multiprocessing'] = True
    # cfg.PARAMS['border'] = 80
    # # Set the path where to save tha data.
    # cfg.PATHS['working_dir'] = '/home/erik/data/oggm_output/'

    # If we want to restart, wipe everything and simulate glaciers again.
    if restart:
        base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/' +\
            'oggm_v1.4/L3-L5_files/CRU/elev_bands/qc3/pcp2.5/no_match'
        # Initialize the gdirs
        gdirs = workflow.init_glacier_directories(rgiids,
                                                  from_prepro_level=5,
                                                  prepro_border=80,
                                                  prepro_base_url=base_url)
        # Run glacier projections.
        run_hydro_projections(gdirs, gcm, ssps, data_dir, basin)
    # If we already have all the data, load the gdirs.
    else:
        gdirs = workflow.init_glacier_directories(rgiids)


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
        rid, ft, fp = gha.utils.get_cmip6_data(gcm, ssp)
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
        rid, _, _ = gha.utils.get_cmip6_data(gcm, ssp)
        workflow.execute_entity_task(
                             tasks.run_with_hydro, gdirs,
                             run_task=tasks.run_from_climate_data,
                             ys=2020,
                             # Use gcm_data
                             climate_filename='gcm_data',
                             # Use the scenario
                             climate_input_filesuffix='_' + rid,
                             # When to start?
                             init_model_filesuffix='_historical',
                             # Good naming for recognizing later
                             output_filesuffix=rid,
                             # Store monthyl?
                             store_monthly_hydro=True,
                            )

        # Where do we store the data?
        data_dir = gha.utils.init_data_dir(data_dir)
        basin_dir = gha.utils.mkdir(data_dir, basin)
        # File suffixes
        output_suffix = f'oggm_compiled_{basin}_{rid}.nc'
        # If we do a custom path, we have to supply the whole path to OGGM.
        path = os.path.join(basin_dir, output_suffix)
        # Compile the output
        utils.compile_run_output(gdirs,
                                 path=path,
                                 input_filesuffix=rid,
                                 )


def compile_basin_output(basin, rcp, data_dir, test=False):
    '''Compile the model output for glaciers within a basin.

    Args:
    -----
    basin: string
        String of the basin MRBID. I.e. '3209'.
    rcp: string
        String declaring the rcp scenario. I.e. 'rcp26'
    data_dir: string
        Path to the directory where to store the data.
    test: bool
        If to run on only a subset of the glaciers. For testing.

    Returns
    -------
    ds: xarray dataset
        Dataset with the compiled run outputs.
    '''
    # Get the ids for the glaciers within the basin.
    rgiids = gha.utils.select_glaciers_json(basin)
    # Testing
    if test:
        rgiids = rgiids[:2]
    #  We need the gdirs to compile.
    gdirs = workflow.init_glacier_directories(rgiids)

    # Where do we store the data?
    data_dir = gha.utils.init_data_dir(data_dir)
    basin_dir = gha.utils.mkdir(data_dir, basin)
    # File suffixes
    input_suffix = f'_CCSM4_{rcp}'
    output_suffix = f'oggm_compiled_{basin}_CCSM4_{rcp}.nc'
    # If we do a custom path, we have to supply the whole path to OGGM.
    path = os.path.join(basin_dir, output_suffix)
    # Compile the output
    ds = utils.compile_run_output(gdirs,
                                  path=path,
                                  input_filesuffix=input_suffix,
                                  )
    return ds
