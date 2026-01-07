############## Energetics in Idealized ROMS-Budgell Beaufort Jet ###############
# The purpose of this script is to process the model output so that a different
# script can plot time series of eddy energy (eddy
# kinetic energy- EKE; mean kinetic energy - MKE; total kinetic energy - TKE) 
# from output
# for the idealized ROMS-Budgell model based on the Beaufort shelf break jet.
# So the purpose of this script is to act as a function that can be called 
# by a bash script to process model output and save it to a netcdf. 

# 
# Notes: 
# - This script runs in xroms
# - This script heavily uses code from Dylan Schlichting 
# - For energetics, slice toooooo the region the mixing is for? or keep the 
#   whole domain...? I think the whole lateral domain... which makes senseeeee
# - JK I think we are going to just do the same box as the mixing stats...
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


# Define a function to calculate the different 
# eddy energies (eddy kinetic (EKE), mean kinetic (MKE),
# and total kinetic (TKE) energy)
def energy_vint(ds,grid,etaslice,xislice,zslice):
    '''
Modifies volume-integrated eddy, mean, and total kinetic energy modified 
from Hetland (2017) and Schlichting et al 2024. For the beaufort jet, 
we will look only in the top 200 m 

Notes:
------
EKE = 1/2(uprime^2 + vprime^2). 
MKE = 1/2(ubar^2+vbar^2)
TKE = 1/2(u^2+v^2)
u = ubar+uprime, ubar = 1/L int_0^L u dx, aka alongshore mean
v = vbar+vprime
Velocities interpolated to rho points
    '''
    u = xroms.to_rho(ds.u, grid)
    urho = u.isel(eta_rho = etaslice, xi_rho = xislice).where(ds.z_rho>-zslice) 
    v = xroms.to_rho(ds.v, grid)
    vrho = v.isel(eta_rho = etaslice, xi_rho = xislice).where(ds.z_rho>-zslice)
    
    ubar = urho.mean('xi_rho')
    uprime = urho-ubar
    
    vbar = vrho.mean('xi_rho')
    vprime = vrho-vbar
    
    dV = ds.dV.isel(eta_rho = etaslice, xi_rho = xislice).where(ds.z_rho>-zslice) 
    #Mean kinetic energy
    mke = 0.5*(ubar**2+vbar**2)
    mke_int = (mke*dV).sum(['eta_rho', 'xi_rho', 's_rho'])
    mke_initial = (mke*dV).sum(['eta_rho', 'xi_rho', 's_rho'])[0] # Initial value for normalization
    mke_int.attrs = ''
    mke_int.name = 'mke'
    mke_initial.attrs = ''
    mke_initial.name = 'mke_initial'

    #Eddy kinetic energy
    eke = 0.5*(uprime**2 + vprime**2)
    eke_int = (eke*dV).sum(['eta_rho', 'xi_rho', 's_rho'])
    eke_int.attrs = ''
    eke_int.name = 'eke'

    #Total kinetic energy 
    tke = 0.5*(urho**2+vrho**2)
    tke_int = (tke*dV).sum(['eta_rho', 'xi_rho', 's_rho'])
    tke_int.attrs = ''
    tke_int.name = 'tke'     
    
    ds_energy = xr.merge([eke_int, mke_int, tke_int])
    return ds_energy


# Make a function that does calculations for energetics 
# and saves them to a netcdf
def calc_save_energetics_roms(avg_file, min_xi_idx, max_xi_idx, min_eta_idx, max_eta_idx, z_slice, output_file):
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

    print('started calculating energy', flush=True)
    # Call the function to calculate energy things
    dse = energy_vint(ds,grid,eta_slice,xi_slice,z_slice).compute()
    print('done calculating energy', flush=True)

    # Make a time to use for saving the data
    time_tmp = ds['ocean_time']
    t0 = time_tmp.values[0]
    time = np.array([(t - t0).total_seconds() / 86400 for t in time_tmp.values])
    #time=time[:2]

    # Save this to a netcdf
    roms_tke_mke_eke = xr.Dataset(
        data_vars=dict(
            tke=(['ocean_time'], dse.tke.values),
            mke=(['ocean_time'], dse.mke.values),
            eke=(['ocean_time'], dse.eke.values),
        ),
        coords=dict(
            ocean_time=('ocean_time', time)
        ),
        attrs=dict(description='Time-series of energetics of ROMS output including EKE, MKE, and TKE from 2-hourly model output')
    )
 
    print('started adding attributes to netcdf', flush=True)
    # Add attributes to the 'temperature' data variable
    roms_tke_mke_eke['tke'].attrs['long_name'] = 'Total kinetic energy at 2-hourly intervals'
    roms_tke_mke_eke['tke'].attrs['units'] = '(J/kg)'
    roms_tke_mke_eke['mke'].attrs['long_name'] = 'Mean kinetic energy at 2-hourly intervals'
    roms_tke_mke_eke['mke'].attrs['units'] = '(J/kg)'
    roms_tke_mke_eke['eke'].attrs['long_name'] = 'Eddy kinetic energy at 2-hourly intervals'
    roms_tke_mke_eke['eke'].attrs['units'] = '(J/kg)'
    print('done adding attributes to netcdf', flush=True)

    # Save to a netcdf
    # Run with ice
    roms_tke_mke_eke.to_netcdf(output_file)
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
    calc_save_energetics_roms(
        avg_file=args.avg_file,
        min_xi_idx=args.min_xi,
        max_xi_idx=args.max_xi,
        min_eta_idx=args.min_eta,
        max_eta_idx=args.max_eta,
        z_slice=args.z_slice,
        output_file=args.output
    )
