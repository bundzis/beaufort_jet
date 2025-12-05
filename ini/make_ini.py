'''
Create an initialization file for an idealized Beaufort shelf simulation. Based on the shelfstrat model originally developed 
by Rob Hetland in the 2010s! The default parameters MUST be adjusted based on a water mass analysis of the beaufort 
shelf. These parameters produce baroclinic instabilities, but the deformation radius in an unforced simulation is too large.
The flow is configured in thermal wind balance with added noise to bathmetry to encourage instabilities. 

To do, so this is more like a jet than a plume: 
- Rework make_ini so we calculate dρ/dy instead of giving a constant M2
This way, a jet can be prescribed. The plume edge looks like a jet with this bathymetric configuration, but we MUST 
do better!

Horizontal stratification is controlled by **salinity** (S), and vertical stratification is controlled by **temperature** (T). 
The density is computed using a **linear equation of state**:

   R0    : 1025.0 kg/m³      # Reference density
   T0    : 2.0 °C             # Reference temperature for linear stratification
   S0    : 34.0 PSU           # Reference salinity for linear stratification
   TCOEF : 1.7e-4 1/°C        # Thermal expansion coefficient
   SCOEF : 7.6e-4 1/PSU       # Saline contraction coefficient

Vertical grid parameters (S-coordinate):
   zlevs   : 40                # Number of vertical layers
   theta_s : 5.0               # Surface stretching parameter (increases vertical resolution near the surface)
   theta_b : 0.4               # Bottom stretching parameter (increases vertical resolution near the bottom)
   hc      : 5.0               # Critical depth for stretching functions (m)

Lateral buoyancy parameters (for horizontal salinity gradient):
   M20   : 5e-7                # Maximum horizontal density gradient amplitude
   M2_yo : 120e3               # Y-location of plume edge
   M2_r  : 5e3                 # E-folding scale for horizontal stratification (m)

Buoyancy frequency / vertical stratification (controls deformation radius):
   N20    : 1e-4 s⁻²           # Maximum N² (squared buoyancy frequency)
   N2_zo  : 50.0 m             # Reference depth for the stratified layer
   N2_r   : 50.0 m             # E-folding depth scale of the pycnocline

Notes on deformation radius:
   L_d ~ N * H / f
   H  : vertical pycnocline thickness (You can plot the initial conditions to figure this out)
   f  : Coriolis parameter (depends on latitude)
   N  : characteristic buoyancy frequency (depends on N20 and pycnocline profile)
   => Adjusting N20, N2_zo, N2_r changes the first-mode baroclinic deformation radius.

balanced_run : True/False     # If True, computes a balanced initial velocity from the density field

References: 
- Hetland, R. D. (2017). Suppression of baroclinic instabilities in buoyancy-driven flow over sloping bathymetry. 
  Journal of Physical Oceanography, 47(1), 49-68.
- Schlichting, D., Hetland, R., & Jones, C. S. (2024). Numerical mixing suppresses submesoscale baroclinic instabilities 
  over sloping bathymetry. Journal of Advances in Modeling Earth Systems, 16(12), e2024MS004321.
'''

import numpy as np
import xarray as xr
from datetime import datetime

def get_depths(h, hc, s, Cs, dim):
    """SEE Eq. (2) or (3) on https://www.myroms.org/wiki/Vertical_S-coordinate"""
    return (hc * s + h * Cs) / (hc + h) * h


def C(theta_s, theta_b, s):
    C = (1.0 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1.0)
    if theta_b > 0.0:
        return (np.exp(theta_b * C) - 1.0) / (1.0 - np.exp(-theta_b))
    else:
        return -C

def make_ini_no_ice(output='/global/homes/b/bundzis/Projects/Beaufort_ROMS_idealized_jet/Include/ini.nc', 
                    grd_path='/global/homes/b/bundzis/Projects/Beaufort_ROMS_idealized_jet/Include/grd.nc',
                    zlevs=40, theta_s=5.0, theta_b=0.4, hc=5.0,
                    T0=2.0, S0=34.0, TCOEF=1.7e-4, SCOEF=7.6e-4,
                    M20=5e-7, M2_yo=120e3, M2_r=5e3,
                    N20=1e-4, N2_zo=50.0, N2_r=50.0,
                    balanced_run=True):


    grd = xr.open_dataset(grd_path)
    g = 9.81
    dy = 1 / grd.pn
    # Vertical grid information
    s_w = xr.DataArray(np.linspace(-1., 0., zlevs + 1), dims=['s_w'])
    s_rho = np.linspace(-1., 0., zlevs + 1)
    s_rho = s_rho[:-1] + np.diff(s_rho) / 2
    s_rho = xr.DataArray(s_rho, dims=['s_rho'])
    Cs_r = C(theta_s, theta_b, s_rho)
    Cs_w = C(theta_s, theta_b, s_w)

    M2 = M20 * np.exp((M2_yo - grd.y_rho) / M2_r)
    M2 = M2.where(grd.y_rho > M2_yo, M20)
    salt = (M2 * dy / g / SCOEF).cumsum(axis=0)
    salt -= salt[-1] - S0
    salt = salt.expand_dims('s_rho') * np.ones((zlevs, 1, 1), 'd')
    salt.coords['s_rho'] = s_rho
    # (h, hc, s, Cs)
    z = get_depths(grd.h, hc, s_rho, Cs_r, 's_rho')
    Hz = get_depths(grd.h, hc, s_w, Cs_w, 's_w').diff('s_w').rename({'s_w': 's_rho'})
    Hz.coords['s_rho'] = s_rho

    # -------------------------------------------------------------
    # Compute vertical stratification (N^2) and initialize temperature profile
    # -------------------------------------------------------------
    # N2 = N20 * exp(-(N2_zo - z) / N2_r)
    #   - Define the vertical buoyancy frequency profile (N^2) as an exponential decay
    #   - N20: maximum stratification near the surface
    #   - N2_zo: reference depth for decay (depth of pycnocline onset)
    #   - N2_r: e-folding depth scale over which N^2 decreases
    #   - z: depth of rho points (positive downward)
    #
    # N2 = N2.where(z > N2_zo, N20)
    #   - Clamp N2 above the reference depth to N20 (ensure strong surface stratification)
    #
    # temp = xr.zeros_like(salt)
    #   - Initialize a temperature array of same shape as salinity
    #
    # for n in range(zlevs):
    #     temp[n] = T0 - np.trapz(N2[n:] / (g * TCOEF), x=z[n:], axis=0)
    #   - Integrate N^2 downward to get temperature profile using a linear equation of state
    #   - T0: reference temperature at surface
    #   - TCOEF: thermal expansion coefficient (relates density changes to temperature)
    #   - g: gravity
    #   - np.trapz(...) integrates the buoyancy frequency contribution to temperature
    #   - Result: vertical temperature profile in approximate hydrostatic balance with N^2
    # -------------------------------------------------------------

    N2 = N20 * np.exp(-(N2_zo - z) / N2_r)
    N2 = N2.where(z > N2_zo, N20)

    temp = xr.zeros_like(salt)
    for n in range(zlevs):
        temp[n] = T0 - np.trapz(N2[n:] / (g * TCOEF), x=z[n:], axis=0)
    
    temp = xr.where(temp < -1.8, -1.8, temp)

    #########################################
    # Create dataset

    ds = xr.Dataset({'temp': temp, 'salt': salt,
                     's_rho': s_rho, 'xi_rho': grd.xi_rho, 'eta_rho': grd.eta_rho})

    # -------------------------------------------------------------
    # Compute initial geostrophically balanced u-velocity (thermal wind)
    # If balanced_run is True:
    # 1. Use the vertical density gradients (here encoded as M2) to 
    #    compute the vertical shear of the zonal velocity using
    #       du/dz ≈ - (g / (f * rho0)) * dρ/dy
    #    This is the thermal wind equation in the Arctic shelf context.
    # 2. rhs = Hz * M2 / grd.f approximates the layer-integrated shear
    # 3. u_z = 0.5 * (rhs[:, :, 1:] + rhs[:, :, :-1]) averages shear to midpoints
    # 4. u = np.cumsum(u_z, axis=0) integrates shear from bottom to top
    # 5. ubottom = zeros imposes no-slip at the bottom
    # 6. Final averaging u = 0.5*(u[1:]+u[:-1]) interpolates velocity to rho points
    # If balanced_run is False, u is initialized to zero everywhere
    # -------------------------------------------------------------
    if balanced_run:
        rhs = Hz * M2 / grd.f
        u_z = 0.5 * (rhs[:, :, 1:] + rhs[:, :, :-1])
        u = np.cumsum(u_z, axis=0)
        ubottom = np.zeros((1, u.shape[1], u.shape[2]))
        u = np.concatenate((ubottom, u))
        u = 0.5 * (u[1:] + u[:-1])
    else:
        u = 0.

    ds['ocean_time'] = xr.DataArray([0.0], dims=['ocean_time'])
    ds['ocean_time'].attrs['units'] = 'days'

    ds['u'] = xr.DataArray(u[np.newaxis, :, :, :],
                           dims=['ocean_time', 's_rho', 'eta_u', 'xi_u'],
                           attrs={'units': 'm s-1'})
    ds['v'] = xr.DataArray(np.zeros((1, int(zlevs), grd.dims['eta_v'], grd.dims['xi_v'])),
                           dims=['ocean_time', 's_rho', 'eta_v', 'xi_v'],
                           attrs={'units': 'm s-1'})
    ds['zeta'] = xr.DataArray(np.zeros((1, grd.dims['eta_rho'], grd.dims['xi_rho'])),
                              dims=['ocean_time', 'eta_rho', 'xi_rho'],
                              attrs={'units': 'm'})
    ds['ubar'] = xr.DataArray(np.zeros((1, grd.dims['eta_u'], grd.dims['xi_u'])),
                              dims=['ocean_time', 'eta_u', 'xi_u'],
                              attrs={'units': 'm s-1'})
    ds['vbar'] = xr.DataArray(np.zeros((1, grd.dims['eta_v'], grd.dims['xi_v'])),
                              dims=['ocean_time', 'eta_v', 'xi_v'],
                              attrs={'units': 'm s-1'})
    # Add z-rho for plotting sections of the initial condition
    z_rho = xr.DataArray(z.astype(float).expand_dims('ocean_time', axis=0),
                         dims=['ocean_time', 's_rho', 'eta_rho', 'xi_rho'],
                         attrs={'units': 'm', 'long_name': 'depth of rho points'})
    ds['z_rho'] = z_rho

    ds.attrs['type'] = 'ROMS Ini file'
    ds.attrs['Description'] = 'Initial conditions for ideal shelf'
    ds.attrs['Author'] = 'Dylan Schlichting'
    ds.attrs['Created'] = datetime.now().isoformat()
    print('Writing netcdf INI file: '+output)
    ds.to_netcdf(output)

def add_ice_to_ic(ini_path = '/global/homes/b/bundzis/Projects/Beaufort_ROMS_idealized_jet/Include/ini.nc',
                  ini_modified_path = '/global/homes/b/bundzis/Projects/Beaufort_ROMS_idealized_jet/Include/ini_ice.nc'):
    '''
    Adds ice variables to initial condition files. Currently, the model will start from an ice-free state,
    so all values are set to zero! 
    
    '''
    
    ds = xr.open_dataset(ini_path)
    ds['Aice'] = xr.DataArray(np.zeros((1, ds.dims['eta_rho'], ds.dims['xi_rho'])),
                              dims=['ocean_time', 'eta_rho', 'xi_rho'],
                              attrs={'units': ''})
    ds['ice_thickness'] = xr.DataArray(np.zeros((1, ds.dims['eta_rho'], ds.dims['xi_rho'])),
                              dims=['ocean_time', 'eta_rho', 'xi_rho'],
                              attrs={'units': 'meter'})
    ds['meltpond_thickness'] = xr.DataArray(np.zeros((1, ds.dims['eta_rho'], ds.dims['xi_rho'])),
                              dims=['ocean_time', 'eta_rho', 'xi_rho'],
                              attrs={'units': 'meter'})
    ds['ice_age'] = xr.DataArray(np.zeros((1, ds.dims['eta_rho'], ds.dims['xi_rho'])),
                              dims=['ocean_time', 'eta_rho', 'xi_rho'],
                              attrs={'units': 'second'})
    ds['snow_thickness'] = xr.DataArray(np.zeros((1, ds.dims['eta_rho'], ds.dims['xi_rho'])),
                              dims=['ocean_time', 'eta_rho', 'xi_rho'],
                              attrs={'units': 'meter'})
    ds['Tice'] = xr.DataArray(np.zeros((1, ds.dims['eta_rho'], ds.dims['xi_rho'])),
                              dims=['ocean_time', 'eta_rho', 'xi_rho'],
                              attrs={'units': 'Celcius'})
    ds['under_ice_temp'] = xr.DataArray(np.zeros((1, ds.dims['eta_rho'], ds.dims['xi_rho'])),
                              dims=['ocean_time', 'eta_rho', 'xi_rho'],
                              attrs={'units': 'Celcius'})
    ds['under_ice_salt'] = xr.DataArray(np.zeros((1, ds.dims['eta_rho'], ds.dims['xi_rho'])),
                              dims=['ocean_time', 'eta_rho', 'xi_rho'],
                              attrs={'units': 'psu'})
    ds['ice_sst'] = xr.DataArray(np.zeros((1, ds.dims['eta_rho'], ds.dims['xi_rho'])),
                              dims=['ocean_time', 'eta_rho', 'xi_rho'],
                              attrs={'units': 'Celcius'})
    ds['ice_Sxx'] = xr.DataArray(np.zeros((1, ds.dims['eta_rho'], ds.dims['xi_rho'])),
                              dims=['ocean_time', 'eta_rho', 'xi_rho'],
                              attrs={'units': 'Newton meter-1'})
    ds['ice_Sxy'] = xr.DataArray(np.zeros((1, ds.dims['eta_rho'], ds.dims['xi_rho'])),
                              dims=['ocean_time', 'eta_rho', 'xi_rho'],
                              attrs={'units': 'Newton meter-1'})
    ds['ice_Syy'] = xr.DataArray(np.zeros((1, ds.dims['eta_rho'], ds.dims['xi_rho'])),
                              dims=['ocean_time', 'eta_rho', 'xi_rho'],
                              attrs={'units': 'Newton meter-1'})
    ds['Uice'] = xr.DataArray(np.zeros((1, ds.dims['eta_u'], ds.dims['xi_u'])),
                              dims=['ocean_time', 'eta_u', 'xi_u'],
                              attrs={'units': 'Newton meter-1'})
    ds['Vice'] = xr.DataArray(np.zeros((1, ds.dims['eta_v'], ds.dims['xi_v'])),
                              dims=['ocean_time', 'eta_v', 'xi_v'],
                              attrs={'units': 'Newton meter-1'})
    ds.to_netcdf(ini_modified_path)

if __name__ == '__main__':
    make_ini_no_ice()
    add_ice_to_ic()