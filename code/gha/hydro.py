'''Hydro related functions'''
import numpy as np


def calc_PET(ds):
    '''Function used to calculate PET. PET has the unit of mm/month.'''
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
# # Correction coefficient
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

    ds.PET.attrs = {'unit': 'mm month-1'}
    return ds
