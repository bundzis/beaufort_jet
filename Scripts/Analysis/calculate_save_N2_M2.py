############## M2 and N2 in Idealized ROMS-Budgell Beaufort Jet ###############
# The purpose of this script is to process the model output so that a different
# script can plot time series of M2 and N2 from output
# for the idealized ROMS-Budgell model based on the Beaufort shelf break jet.
# So the purpose of this script is to act as a function that can be called 
# by a bash script to process model output and save it to a netcdf. 

# 
# Notes: 
# - This script runs in xroms
# - This script heavily uses code from Dylan Schlichting 
# - For M2 and N2, use the whole domain (and a little bit south of the 
#   northern boundary)
# - For energetics, slice toooooo the region the mixing is for? or keep the 
#   whole domain...? I think the whole lateral domain... which makes senseeeee
# - N2 (and maybe M2) have negative values so doing the log is a little messed
#   up for the PDF versions. However I think we want regions with unstable 
#   stratification (negative N2) sooo think on this...
# - JK I think we are going to just do the same box as the mixing stats...and we will
#   take the absolute value of N2 for now but do a signed version later
#
#################################################################################

# Load in the packages
import numpy as np
import xarray as xr
import xroms
import xarray as xr
from netCDF4 import Dataset
# from mpas_analysis.shared.io.utility import decode_strings
from glob import glob
from xhistogram.xarray import histogram

import warnings 
warnings.filterwarnings("ignore") #turns off annoying warnings



# Make a function to load ROMS output
def open_mfroms(path):
    '''
Opens multiple netcdf files with xroms
    '''
    # chunk = {"xi": -1, "eta": -1, "ocean_time": -1} 
    # chunks = {}
    # for sub in ["rho", "u", "v", "psi"]:
        # for k, v in chunk.items():
            # chunks[f"{k}_{sub}"] = v
    # chunks["ocean_time"] = chunk["ocean_time"]

    ds = xroms.open_mfnetcdf(path)
    ds,grid = xroms.roms_dataset(ds,include_cell_volume=True)
    ds.xroms.set_grid(grid)
    return ds,grid


# Make a function that does the analysis for N2 and M2, then saves it to 
# a netcdf based on a user-given input file and user-given output file
def calc_save_N2_M2_roms(avg_file, min_xi_idx, max_xi_idx, min_eta_idx, max_eta_idx, z_slice, output_file):
    """
    This function loads a given model output, then calculates N2 and M2 for 
    a user-specified region in the domain and saves the post-processed data to a 
    user-specified output file.

    Inputs:
    avg_file: path to roms_avg output file(s) ('/path/to/roms_avg*.nc')
    min_xi_idx: minimum xi index for box for calculations
    max_xi_idx: maximum xi index for box for calculations
    min_eta_idx: minimum eta index for box for calculations
    max_eta_idx: maximum eta index for box for calculations
    z_slice: depth above which to calculate stats (meters) (ex: 200)
    output_file: path to place the output of the post-processed data ('/path/to/processed_data.nc')

    Outputs:
    N2 and M2 in netCDF (see attributes of netCDF for details)

    """

    # Load in the data
    #ds, grid = xroms.open_mfnetcdf(glob(avg_file))
    ds, grid = open_mfroms(glob(avg_file))

    # Make the xi and eta slices
    xi_slice = slice(min_xi_idx, max_xi_idx)
    eta_slice = slice(min_eta_idx, max_eta_idx)

    # Set the constants for the math
    # Density
    R0    = 1027.0     # kg/m^3
    # Temperature 
    T0    = 0.5       # Celsius
    # Salinity 
    S0    = 31       # nondimensional
    # Coefficient for temperature for... (nonlinear equation of state?)
    TCOEF = 1.7e-4     # 1/Celsius
    # Coefficient for salinity for... (nonlinear equation of state?)
    SCOEF = 7.6e-4     # 1/nondimensional

    # Calculate density 
    rho = R0*(1 - (TCOEF * (ds.temp - T0)) + (SCOEF * (ds.salt - S0)))
    rho.name = 'rho'

    # Calculate N2 and put it on the rho grid
    N2 = xroms.N2(rho, grid, rho0=1025.0)
    print('N2 shape: ', np.shape(N2), flush=True)
    print('calculated N2', flush=True)

    # Interpolate to rho points
    N2_rho = grid.interp(N2, 'Z', boundary = 'extend')

    # Set the bins and normalize
    # Compute PDF, normalize by max value so the max is one
    n2_bins = np.arange(-5.5,-2.5,0.05)
    
    print('started N2 pdf', flush=True)
    n2_pdf = histogram(np.log10(abs(N2_rho.isel(eta_rho=eta_slice, xi_rho=xi_slice).where(ds.z_rho>-200))), #.where(ds.z_rho>-200), 
                    bins = n2_bins, 
                    block_size=N2_rho.ocean_time.size,
                    dim = ['s_rho','xi_rho', 'eta_rho'],
                    density=True).compute()
    print('finished N2 pdf', flush=True)

    #n2_bins_log = np.logspace(-5.5, -2.5, 60)
    # n2_pdf = histogram(N2_rho.where(ds.z_rho>-z_slice), 
    #                    bins = n2_bins_log, 
    #                    dim = ['s_rho','xi_rho', 'eta_rho'],
    #                    density=True).compute()

    # Normalize by the math
    n2_pdf = n2_pdf / n2_pdf.max()
    print('normalized N2 pdf', flush=True)
    print('n2_pdf shape: ', np.shape(n2_pdf), flush=True)


    # Calculate M2
    # Set the gravitational constant 
    g = 9.807

    # Calculate buoyancy
    b = -g * ( (rho-R0) / R0)

    # Calculate horizontal gradients accounting for s-coordinate
    # NOT needed if looking at surface only, then a simple grid.diff will do!
    dbdx, dbdy = xroms.hgrad(b.where(ds.z_rho>-z_slice),grid)
    dbdx = xroms.to_rho(dbdx,grid)
    dbdx = grid.interp(dbdx,'Z')
    dbdy = xroms.to_rho(dbdy,grid)
    dbdy = grid.interp(dbdy,'Z')

    # Calculate M2 from these gradients 
    M2 = np.sqrt(dbdx**2+dbdy**2)
    print('M2 shape: ', np.shape(M2), flush=True)
    print('calculated M2', flush=True)

    # Compute PDF, normalize by max value so the max is one
    m2_bins = np.arange(-8,-5.,0.05)
    print('started M2 pdf', flush=True)
    m2_pdf = histogram(np.log10(M2.isel(eta_rho=eta_slice, xi_rho=xi_slice)), 
                    bins = m2_bins, 
                    block_size=M2.ocean_time.size,
                    dim = ['xi_rho', 'eta_rho','s_rho'],
                    density=True).compute()
    print('finished M2 pdf', flush=True)
    m2_pdf = m2_pdf / m2_pdf.max()
    print('normalized M2 pdf', flush=True)
    print('m2_pdf shape: ', np.shape(m2_pdf), flush=True)

    print('started making time', flush=True)
    # Make a time to use for saving the data
    time_tmp = ds['ocean_time']
    t0 = time_tmp.values[0]
    time = np.array([(t - t0).total_seconds() / 86400 for t in time_tmp.values])
    #time_short = time[0:2]

    # Write to netcdf the simple way
    #print('started writing to netcdf simple way')
    #N2[:980,1:,:,:].to_netcdf()

    print('started making xarray dataset for netcdf', flush=True)
    # Save these to a netcdf
    roms_M2_N2 = xr.Dataset(
    data_vars=dict(
        N2=(['ocean_time', 's_rho', 'eta_rho', 'xi_rho'], N2[:980,1:,:,:].values),
        n2_pdf=(['ocean_time', 'n2_bins'], n2_pdf[:980,:].values),
        M2=(['ocean_time', 's_rho', 'eta_rho', 'xi_rho'], M2[:980,:,:,:].values),
        m2_pdf=(['ocean_time', 'm2_bins'], m2_pdf[:980,:].values),
        #n2_bins=(['n2_bin_len'], n2_bins),
        #m2_bins=(['m2_bin_len'], m2_bins),
    ),
    coords=dict(
        ocean_time=('ocean_time', time[:980]),
        n2_bins=('n2_bins', n2_bins[:-1]), 
        m2_bins=('m2_bins', m2_bins[:-1]),
        s_rho=('s_rho', ds.s_rho.values),
        eta_rho=('eta_rho', ds.eta_rho.values),
        xi_rho=('xi_rho', ds.xi_rho.values),
        #ocean_time_short=('ocean_time_short', time_short),
        #n2_bin_len=('n2_bin_len', len(n2_bins)),
        #m2_bin_len=('m2_bin_len', len(m2_bins)),
    ),
    attrs=dict(description='Time-series of parameters of ROMS output including N2, M2, and their normalized PDFs and bins from 2-hourly model output')
)
 
    print('started adding attributes to netcdf', flush=True)
    # Add attributes to the 'temperature' data variable
    roms_M2_N2['N2'].attrs['long_name'] = 'N2, vertical stratification'
    roms_M2_N2['N2'].attrs['units'] = '($s^{-2}$)'
    roms_M2_N2['n2_pdf'].attrs['long_name'] = 'normalized PDF of N2'
    roms_M2_N2['n2_pdf'].attrs['units'] = '(fraction)'
    #roms_M2_N2['n2_bins'].attrs['long_name'] = 'bins used for N2 PDF'
    #roms_M2_N2['n2_bins'].attrs['units'] = '($s^{-2}$)'

    roms_M2_N2['M2'].attrs['long_name'] = 'M2, magnitude of lateral buoyancy gradients'
    roms_M2_N2['M2'].attrs['units'] = '($s^{-2}$)'
    roms_M2_N2['m2_pdf'].attrs['long_name'] = 'normalized PDF of M2'
    roms_M2_N2['m2_pdf'].attrs['units'] = '(fraction)'
    #roms_M2_N2['m2_bins'].attrs['long_name'] = 'bins used for M2 PDF'
    #roms_M2_N2['m2_bins'].attrs['units'] = '($s^{-2}$)'
    print('done adding attributes to netcdf', flush=True)

    # Define compression settings for the data variables
    # zlib=True enables compression, complevel=1 is fast and efficient
    encoding = {
        'N2': {'zlib': True, 'complevel': 1},
        'M2': {'zlib': True, 'complevel': 1},
        'n2_pdf': {'zlib': True, 'complevel': 1},
        'm2_pdf': {'zlib': True, 'complevel': 1}
    }

    # Save to a netcdf
    roms_M2_N2.to_netcdf(output_file, encoding=encoding, engine='netcdf4')
    print('saved to netcdf', flush=True)


# Make these functions callable from bash

import argparse

if __name__ == "__main__":
    # 1. Initialize the parser
    parser = argparse.ArgumentParser(description='Process ROMS output for N2 and M2 analysis.')

    # 2. Add arguments (matching your function parameters)
    parser.add_argument('--avg_file', type=str, required=True, help='Path to input ROMS avg file')
    parser.add_argument('--min_xi', type=int, required=True)
    parser.add_argument('--max_xi', type=int, required=True)
    parser.add_argument('--min_eta', type=int, required=True)
    parser.add_argument('--max_eta', type=int, required=True)
    parser.add_argument('--z_slice', type=float, required=True, help='Depth slice value (e.g. 200)')
    parser.add_argument('--output', type=str, required=True, help='Path to save the NetCDF')

    # 3. Parse the arguments from the command line
    args = parser.parse_args()

    # 4. Call your function using the parsed arguments
    calc_save_N2_M2_roms(
        avg_file=args.avg_file,
        min_xi_idx=args.min_xi,
        max_xi_idx=args.max_xi,
        min_eta_idx=args.min_eta,
        max_eta_idx=args.max_eta,
        z_slice=args.z_slice,
        output_file=args.output
    )








