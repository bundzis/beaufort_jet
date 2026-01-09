#################### Numerical & Physical Mixing in Ideal ROMS-Budgell Beaufort Jet ########################
# The purpose of this script is to plot time series of volume-averaged numerical and physical 
# mixing for salinity and temperature. Other processing for time series might be added in here, too,
# but all post-processed data will be save to netCDFs so the analysis does not need to be redone 
# to replot things. This script has been edited to be called as a python function in a bash script 
# for a user-specified input and written to a user-specific output. 
#
# Notes:
# - This script runs in xroms 
# - This script heavily uses code from Dylan Schlichting 
# - Other things that will maybe be added into here: energy analysis of eddy things, 
#   M2 and N2, ice volume/concentration/whatever is standard in papers
# - Choose an area that always has eddies, vertical slices constrained to top 200 m, 
# - I think I am going to add things for ice variables into here...
# - This is the same as the .ipynb version of this script but ran for just one model output
#   so that I can run them all at the same time since they take so long -- the output for this one 
#   is using the kpp vertical mixing scheme 

##############################################################################################################


# Load in the packages
import numpy as np
import glob
import warnings
import xarray as xr
import xroms
warnings.filterwarnings("ignore") #turns off annoying warnings

# The core of the analysis is done with xhistogram
from xhistogram.xarray import histogram


# ---------------------------------------------------------------------------
# -------------------------- Define Functions -------------------------------
# ---------------------------------------------------------------------------

# Make a function to open the model output
def open_roms(path):
    '''
Opens multiple netcdf files with xroms
    '''
    # chunk = {"xi": -1, "eta": -1, "ocean_time": -1} 
    # chunks = {}
    # for sub in ["rho", "u", "v", "psi"]:
        # for k, v in chunk.items():
            # chunks[f"{k}_{sub}"] = v
    # chunks["ocean_time"] = chunk["ocean_time"]

    ds = xroms.open_netcdf(path)
    ds,grid = xroms.roms_dataset(ds,include_cell_volume=True)
    ds.xroms.set_grid(grid)
    return ds,grid


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


# Make a function to calculate useful derivatives and add them back into 
# the dataset
def add_derivatives(ds, grid, q='salt'):
    '''
Analysis of velocity gradient tensor and frontogenesis function
following Hetland et al. (2025) JPO
    '''
    qs = ds[q]
    
    #############################
    # Flow and property gradients at the ocean surface
    
    ds['dqdx'] = xroms.to_rho(grid.derivative(qs.isel(s_rho=-1), 'X'), grid)    # defined at rho-points
    ds['dqdy'] = xroms.to_rho(grid.derivative(qs.isel(s_rho=-1), 'Y'), grid)    # defined at rho-points
    
    ds['dudx'] = grid.derivative(ds.isel(s_rho=-1).u, 'X', boundary='extend')  # defined at rho-points
    ds['dvdy'] = grid.derivative(ds.isel(s_rho=-1).v, 'Y', boundary='extend')  # defined at rho-points
    ds['dvdx'] = xroms.to_rho(grid.derivative(ds.isel(s_rho=-1).v, 'X', boundary='extend'), grid)  # defined at rho-points
    ds['dudy'] = xroms.to_rho(grid.derivative(ds.isel(s_rho=-1).u, 'Y', boundary='extend'), grid)  # defined at rho-points
    
    ###########################
    # Invariant flow properties
    
    # Vorticity:  v_x - u_y
    ds['zeta'] = (ds.dvdx - ds.dudy)/ds.f
    ds['zeta'].name = 'Normalized vorticity'

    # Divergence: u_x + v_y
    ds['delta'] = (ds.dudx + ds.dvdy)/ds.f
    ds['delta'].name = 'Normalized divergence'

    # Major axis of deformation
    ds['alpha'] = ( np.sqrt( (ds.dudx-ds.dvdy)**2 + (ds.dvdx+ds.dudy)**2 ) )/ds.f
    ds['alpha'].name = 'Normalized total strain'

    ##################################
    # Principle deformation components

    ds['lminor'] = 0.5 * (ds.delta - ds.alpha)
    ds['lminor'].name = 'lambda minor'

    ds['lmajor'] = 0.5 * (ds.delta + ds.alpha)
    ds['lmajor'].name = 'lambda major'
    
    #############################################
    # Along- and cross-frontal velocity gradients
    
    # angle is wrt x, so need to do arctan2(y, x)
    ds['phi_cf'] = np.arctan2(ds.dqdy, ds.dqdx)
    ds['phi_af'] = ds.phi_cf + np.pi/2.0

    ds['du_cf'] = ( ds.dudx*np.cos(ds.phi_cf)**2 + ds.dvdy*np.sin(ds.phi_cf)**2 
               + (ds.dudy + ds.dvdx)*np.sin(ds.phi_cf)*np.cos(ds.phi_cf) )/ds.f

    ds['du_af'] = ( ds.dudx*np.cos(ds.phi_af)**2 + ds.dvdy*np.sin(ds.phi_af)**2
              + (ds.dudy + ds.dvdx)*np.sin(ds.phi_af)*np.cos(ds.phi_af) )/ds.f
    
    ############################
    # The frontogenesis function
    
    # Dimensional frontogenesis function
    Dgradq_i = - ds.dudx*ds.dqdx - ds.dvdx*ds.dqdy
    Dgradq_j = - ds.dudy*ds.dqdx - ds.dvdy*ds.dqdy
    ds['Ddelq2'] = (ds.dqdx*Dgradq_i + ds.dqdy*Dgradq_j)
    ds['Ddelq2'].name = 'Frontogenesis function'

    # Density gradients squared
    ds['gradq2'] = ds.dqdx**2 + ds.dqdy**2
    ds['gradq2'].name = r'$(\nabla q)^2$'

    # Normalized frontogenesis function
    ds['nFGF'] = 0.5 * ds.Ddelq2 / (ds.gradq2 * ds.f)
    ds['nFGF'].name = r'Normalized Frontogenesis Function'
    
    return ds


# ---------------------------------------------------------------------------
# -------------------------- Process Output -------------------------------
# ---------------------------------------------------------------------------

# Make a function that loads in the output, calculates the mixing, and saves
# the results to a netcdf
def calc_save_phy_num_mixing_roms(avg_file, min_xi_idx, max_xi_idx, min_eta_idx, max_eta_idx, z_slice, output_file):
        """
        This function loads a given model output, then calculates numerical and physical
        mixing of salinity and temperature for 
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
        path = glob.glob(avg_file)
        ds, grid = open_mfroms(path)
        add_derivatives(ds, grid)

        # Make the xi and eta slices
        xi_slice = slice(min_xi_idx, max_xi_idx)
        eta_slice = slice(min_eta_idx, max_eta_idx)
 
        # OG Way
        # # Get physical mixing from the output
        # # Salinity (2-hourly)
        # print('started calculating physical salt mixing', flush=True)
        # Akr_rho = grid.interp(ds.AKr, 'Z')
        # mphys = (Akr_rho*ds.dV).isel(eta_rho = eta_slice, xi_rho = xi_slice).where(ds.z_rho>-z_slice).sum(['eta_rho', 'xi_rho', 's_rho']).compute()
        # mphys.attrs = ''

        # # Temperature (2-hourly)
        # print('started calculating physical temp mixing', flush=True)
        # Akrt_rho = grid.interp(ds.AKrt, 'Z')
        # mphyt = (Akrt_rho*ds.dV).isel(eta_rho = eta_slice, xi_rho = xi_slice).where(ds.z_rho>-z_slice).sum(['eta_rho', 'xi_rho', 's_rho']).compute()
        # mphyt.attrs = ''

        # # Get numerical mixing from the output
        # # Numerical mixing (2-hourly)
        # print('started calculating numerical salt mixing', flush=True)
        # mnum_salt_dv = (ds.dye_03*ds.dV).isel(eta_rho = eta_slice, xi_rho=xi_slice).where(ds.z_rho>-z_slice).sum(['eta_rho', 'xi_rho', 's_rho']).compute()
        # print('started calculating numerical temp mixing', flush=True)
        # mnum_temp_dv = (ds.dye_06*ds.dV).isel(eta_rho = eta_slice, xi_rho=xi_slice).where(ds.z_rho>-z_slice).sum(['eta_rho', 'xi_rho', 's_rho']).compute()

        # Faster?
        # 1. Define the calculations (Lazy/Dask)
        Akr_rho = grid.interp(ds.AKr, 'Z')
        Akrt_rho = grid.interp(ds.AKrt, 'Z')
        
        mphys_lazy = (Akr_rho * ds.dV).isel(eta_rho=eta_slice, xi_rho=xi_slice).where(ds.z_rho > -z_slice).sum(['eta_rho', 'xi_rho', 's_rho'])
        mphyt_lazy = (Akrt_rho * ds.dV).isel(eta_rho=eta_slice, xi_rho=xi_slice).where(ds.z_rho > -z_slice).sum(['eta_rho', 'xi_rho', 's_rho'])
        mnum_salt_lazy = (ds.dye_03 * ds.dV).isel(eta_rho=eta_slice, xi_rho=xi_slice).where(ds.z_rho > -z_slice).sum(['eta_rho', 'xi_rho', 's_rho'])
        mnum_temp_lazy = (ds.dye_06 * ds.dV).isel(eta_rho=eta_slice, xi_rho=xi_slice).where(ds.z_rho > -z_slice).sum(['eta_rho', 'xi_rho', 's_rho'])

        # 2. Compute everything in ONE go to save memory and time
        print('Starting simultaneous computation of all mixing terms...', flush=True)
        results = xr.compute(mphys_lazy, mphyt_lazy, mnum_salt_lazy, mnum_temp_lazy)
        mphys, mphyt, mnum_salt_dv, mnum_temp_dv = results
        print('Computation complete.', flush=True)

        
        # Save to a netcdf
        # Variables to save: mphys, mphyt, mnum_salt_dv, mnum_temp_dv

        roms_phy_num_mix_salt_temp = xr.Dataset(
            data_vars=dict(
                mphys=(['ocean_time'], mphys.values),
                mphyt=(['ocean_time'], mphyt.values),
                mnum_salt_dv=(['ocean_time'], mnum_salt_dv.values),
                mnum_temp_dv=(['ocean_time'], mnum_temp_dv.values)
            ),
            coords=dict(
                ocean_time=('ocean_time', ds.ocean_time.values)
            ),
            attrs=dict(description='Time-series ROMS output including physical and numerical mixing of salinity and temperature from 2-hourly model output; uses /global/cfs/cdirs/m4572/dylan617/beaufort_jet/runs/roms_avg_ice_500m_w_dvd_constant_forcing_u3c4_kpp_*.nc')
        )

        print('started adding attributes to netcdf', flush=True)
        # Add attributes to the 'temperature' data variable
        roms_phy_num_mix_salt_temp['mphys'].attrs['long_name'] = 'Physical mixing of salinity at 2-hourly intervals'
        roms_phy_num_mix_salt_temp['mphys'].attrs['units'] = '(g/kg)\u00b2(m\u00b3/s))'
        roms_phy_num_mix_salt_temp['mphyt'].attrs['long_name'] = 'Physical mixing of temperature at 2-hourly intervals'
        roms_phy_num_mix_salt_temp['mphyt'].attrs['units'] = '(C)\u00b2(m\u00b3/s))'

        roms_phy_num_mix_salt_temp['mnum_salt_dv'].attrs['long_name'] = 'Numerical mixing of salinity at 2-hourly interval'
        roms_phy_num_mix_salt_temp['mnum_salt_dv'].attrs['units'] = '(g/kg)\u00b2(m\u00b3/s))'
        roms_phy_num_mix_salt_temp['mnum_temp_dv'].attrs['long_name'] = 'Numerical mixing of temperature at 2-hourly interval'
        roms_phy_num_mix_salt_temp['mnum_temp_dv'].attrs['units'] = '(C)\u00b2(m\u00b3/s))'
        print('done adding attributes to netcdf', flush=True)

        # Use encoding to keep file size down
        encoding = {var: {'zlib': True, 'complevel': 1} for var in roms_phy_num_mix_salt_temp.data_vars}

        # Save to a netcdf
        # Run with ice (change name based on the run being analyzed)
        roms_phy_num_mix_salt_temp.to_netcdf(output_file, encoding=encoding)
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
    calc_save_phy_num_mixing_roms(
        avg_file=args.avg_file,
        min_xi_idx=args.min_xi,
        max_xi_idx=args.max_xi,
        min_eta_idx=args.min_eta,
        max_eta_idx=args.max_eta,
        z_slice=args.z_slice,
        output_file=args.output
    )

