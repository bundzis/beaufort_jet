'''
Create an initialization file for an idealized Beaufort shelf simulation. Based on the shelfstrat model originally developed 
by Rob Hetland in the 2010s! The flow is configured in partial thermal wind balance with added noise to bathmetry to encourage 
instabilities. Note that the thermal wind balance is approximate since we only use the lateral salinity gradient to compute 
the balanced velocity.

Horizontal stratification is controlled by salinity (S), and vertical stratification is controlled by temperature (T). 
The density is computed using a linear equation of state:

   R0    : 1025.0 kg/m³      # Reference density
   T0    : 1.0 °C             # Reference temperature for linear stratification
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
   M2_yo : 120e3               # Y-location of plume/jet edge
   M2_r  : 5e3                 # E-folding scale for horizontal stratification (m)

Temperature vertical stratification parameters:
    T0    : 0.5 °C              # Surface temperature  
    Tbot  : -1.6 °C             # Bottom temperature
    H_pyc : -50.0 m             # Depth of pycnocline center
    delta : 15.0 m              # Thickness (e-folding scale) of the pycnocline


Notes on deformation radius:
   L_d ~ N * H / f
   H  : vertical pycnocline thickness (You can plot the initial conditions to figure this out)
   f  : Coriolis parameter (depends on latitude)
   N  : characteristic buoyancy frequency (depends on N20 and pycnocline profile)

balanced_run : True/False     # If True, computes a balanced initial velocity from the density field
lateral_strat_type:    # Sets whether the lateral stratification is a buoyant plume or jet, PLUME no longer supported
    - Valid arguments: "jet"

References: 
- Hetland, R. D. (2017). Suppression of baroclinic instabilities in buoyancy-driven flow over sloping bathymetry. 
  Journal of Physical Oceanography, 47(1), 49-68.
- Schlichting, D., Hetland, R., & Jones, C. S. (2024). Numerical mixing suppresses submesoscale baroclinic instabilities 
  over sloping bathymetry. Journal of Advances in Modeling Earth Systems, 16(12), e2024MS004321.
'''

import numpy as np
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import warnings
warnings.filterwarnings("ignore") #turns off annoying warnings

def get_depths(h, hc, s, Cs, dim):
    """SEE Eq. (2) or (3) on https://www.myroms.org/wiki/Vertical_S-coordinate"""
    return (hc * s + h * Cs) / (hc + h) * h


def C(theta_s, theta_b, s):
    C = (1.0 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1.0)
    if theta_b > 0.0:
        return (np.exp(theta_b * C) - 1.0) / (1.0 - np.exp(-theta_b))
    else:
        return -C

def make_ini_no_ice(output='/pscratch/sd/d/dylan617/beaufort_roms/runs_idealized/inputs/ini_500_m.nc', 
                    grd_path='/pscratch/sd/d/dylan617/beaufort_roms/runs_idealized/inputs/grd_500_m.nc',
                    zlevs=40, theta_s=5.0, theta_b=0.4, hc=5.0,
                    T0=0.5, Tbot = -1.6, delta = 15.0, H_pyc = -50.0,
                    S0=30.0, TCOEF=1.7e-4, SCOEF=7.6e-4,
                    M2_yo=120e3, M2_r=5e3,
                    balanced_run=True, lateral_strat_type = 'jet',
                    y0_jet=115e3, L_jet = 15e3, M2_jet_amp=5e-7):


    grd = xr.open_dataset(grd_path)
    # Ld =  (np.sqrt(N20)*np.abs(N2_r)) / (grd.f.values[0,0])
    # print('Deformation radius in km, ', Ld/1000)

    g = 9.81
    dy = 1 / grd.pn
    # Vertical grid information
    s_w = xr.DataArray(np.linspace(-1., 0., zlevs + 1), dims=['s_w'])
    s_rho = np.linspace(-1., 0., zlevs + 1)
    s_rho = s_rho[:-1] + np.diff(s_rho) / 2
    s_rho = xr.DataArray(s_rho, dims=['s_rho'])
    Cs_r = C(theta_s, theta_b, s_rho)
    Cs_w = C(theta_s, theta_b, s_w)

    z = get_depths(grd.h, hc, s_rho, Cs_r, 's_rho')  # depth at rho points
    Hz = get_depths(grd.h, hc, s_w, Cs_w, 's_w').diff('s_w').rename({'s_w': 's_rho'})
    Hz.coords['s_rho'] = s_rho

    if lateral_strat_type == "jet":
        # -----------------------------
        # 1. Background salinity
        # -----------------------------
        salt_1D = xr.full_like(grd.y_rho, S0)

        # ---------------------------------------------------
        # 2. Compute lateral salinity gradient from M2_jet_amp
        #    Gaussian-shaped jet in y
        # ---------------------------------------------------
        
        M2_jet_y = M2_jet_amp * np.exp(-((grd.y_rho - y0_jet) / L_jet) ** 2)

        # Convert to salinity gradient: dS/dy = M2 / (g * SCOEF)
        dS_dy = M2_jet_y / (9.81 * SCOEF)

        # Cumulative integral in y to get salinity anomaly
        salt_anomaly = dS_dy.cumsum(dim='eta_rho') * (grd.y_rho[1] - grd.y_rho[0])

        # Shift to keep background salinity
        salt_1D += (salt_anomaly - salt_anomaly.min())

        # Add option for vertical salinity stratification, not needed but nice to have!
        dSdz = 0e-3
        z_abs = np.abs(z)
        salt = salt_1D + dSdz * z_abs

        # For reference or plotting: total M2 used in thermal wind
        M2 = M2_jet_y

    # Construct temperature field. 
    # -------------------------------------------------------------

    # --- Reorder arrays so vertical dimension is first ---
    # This ensures that operations over the vertical (s_rho) axis are consistent
    z = z.transpose('s_rho', 'eta_rho', 'xi_rho')
    salt = salt.transpose('s_rho', 'eta_rho', 'xi_rho')
    zlevs = z.s_rho.size  # Number of vertical levels

    # --- Define surface and bottom temperatures ---
    T_surface = T0        # Surface temperature [°C]
    T_bottom  = Tbot      # Bottom temperature [°C]

    # --- Compute smooth vertical temperature profile using a hyperbolic tangent ---
    # Produces a continuous transition from bottom to surface temperature around the pycnocline
    # temp_profile = T_bottom + 0.5*(T_surface - T_bottom)*(1 - tanh((H_pyc - z)/delta))
    temp_profile = T_bottom + 0.5*(T_surface - T_bottom)*(1 - np.tanh((H_pyc - z)/delta))

    # --- Broadcast profile to full 3D grid ---
    # Ensures temperature array has same shape as salinity array for ROMS
    temp = temp_profile * np.ones_like(salt)

    # --- Set surface temperature to horizontal mean at the top layer ---
    # This avoids small horizontal variations at the surface
    T_surf_val = temp.isel(s_rho=-1).mean(dim=('eta_rho','xi_rho'))
    temp.loc[dict(s_rho=temp.s_rho[-1])] = T_surf_val

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
    # 2. rhs = Hz * M2 / grd.f approximates the layer-integrated shear. We IGNORE 
    #    temperature's contributions to density otherwise the profile will look 
    #    extremely noisy seaward of the jet. This is only a problem because of the 
    #    steep bathymetry! 
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

def add_ice_to_ic(ini_path = '/pscratch/sd/d/dylan617/beaufort_roms/runs_idealized/inputs/ini_500_m.nc',
                  ini_modified_path = '/pscratch/sd/d/dylan617/beaufort_roms/runs_idealized/inputs/ini_ice_500_m.nc'):
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

def plot_ic(grd_path = '/pscratch/sd/d/dylan617/beaufort_roms/runs_idealized/inputs/grd_500_m.nc',
            ini_path = '/pscratch/sd/d/dylan617/beaufort_roms/runs_idealized/inputs/ini_ice_500_m.nc',
            fig_path_plan = '/pscratch/sd/d/dylan617/beaufort_roms/runs_idealized/inputs/ini_conditions.png',
            fig_path_cs = '/pscratch/sd/d/dylan617/beaufort_roms/runs_idealized/inputs/ini_conditions_cross_section.png'):
    '''
    Plots ROMS initial conditions. 
    '''
    dsg = xr.open_dataset(grd_path)
    ds = xr.open_dataset(ini_path)

    # Convert to km
    xrho = dsg.x_rho / 1000
    xu = dsg.x_u / 1000
    yrho = dsg.y_rho / 1000 
    yu = dsg.y_u / 1000

    temp = ds.temp[-1]
    salt = ds.salt[-1]
    u = ds.u[0,-1]

    fig, ax = plt.subplots(1, 3, figsize=(9, 2.75),
                       constrained_layout=True, 
                       sharex=True, sharey=True, dpi=200)

    # Temperature
    m = ax[0].pcolormesh(xrho, yrho, temp, cmap=cmo.thermal)
    fig.colorbar(m, ax=ax[0], label='[C]')

    # Salinity
    m1 = ax[1].pcolormesh(xrho, yrho, salt, cmap=cmo.haline)
    fig.colorbar(m1, ax=ax[1], label='[g/kg]')

    # U-velocity
    m2 = ax[2].pcolormesh(xu, yu, u, cmap=cmo.amp)
    fig.colorbar(m2, ax=ax[2], label='[m/s]')

    # Set consistent axis limits
    for a in ax:
        a.set_xlabel('Along-shelf distance [km]')
        a.set_aspect(1)

    ax[0].set_ylabel('Across-shelf distance [km]')
    ax[0].set_title('Surface temperature')
    ax[1].set_title('Surface salinity')
    ax[2].set_title('Surface alongshore velocity')

    # Contours for bathymetry
    positions = [(100, 10),(100,60),(100,90),(100,140),(100,160),(100,210)]
    cs = ax[1].contour(xrho, yrho, dsg.h, levels=[10, 20, 50, 100, 500, 1000],
                    colors='grey', linewidths=0.8)
    ax[1].clabel(cs, fmt='%d m', manual=positions, inline=True, fontsize=8, colors='black')

    plt.savefig(fig_path_plan, dpi = 300, bbox_inches = 'tight')

    print('Temperature min: ', ds.temp.min().values)
    print('Temperature max: ', ds.temp.max().values)
    print('Salinity min: ',   ds.salt.min().values)
    print('Salinity max: ',   ds.salt.max().values)
    print('U velocity min: ', ds.u.min().values)
    print('U velocity max: ', ds.u.max().values)

    # Plot cross section of initial conditions
    tidx = 0

    # Pick a fixed along-shelf location
    xi_idx = 100  # pick a reasonable index along xi_rho

    # --- Extract Data at the Final Time Step ---
    temp_section = ds.temp[:, :, xi_idx]        # shape: (s_rho, eta_rho)
    salt_section = ds.salt[:, :, xi_idx]        # shape: (s_rho, eta_rho)
    z_rho_section = ds.z_rho[tidx, :, :, xi_idx]       # shape: (s_rho, eta_rho)
    u_section = ds.u[tidx, :, :, xi_idx]  # shape: (s_rho, eta_u or eta_rho)

    # --- Plotting 3x1 Subplots ---

    # 1. Create the figure and a 3x1 grid of axes
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex=True, 
                            sharey=True, dpi = 200, constrained_layout=True)

    # 2. Define the horizontal coordinate (assuming it's y_rho for the across-shelf direction)
    # Use dsg.y_rho for the coordinate.
    y_coord = dsg.y_rho[:, 0] / 1000  # Across-shelf distance in km

    # Define common plotting parameters
    common_ylim = (-200, 0)
    xlabel = 'Across Shelf Distance [km]'

    # --- Plot 1: Temperature ---
    ax = axes[0]
    pcm_temp = ax.pcolormesh(y_coord, z_rho_section, temp_section,
                            cmap='RdYlBu_r')
    fig.colorbar(pcm_temp, ax=ax, label='Temperature [°C]')
    # ax.set_title(f'Vertical Section at $\\xi={xi_idx}$')
    ax.set_ylabel('Depth [m]')
    ax.set_ylim(common_ylim)

    # --- Plot 2: Salinity ---
    ax = axes[1]
    # Using typical values for demonstration:
    pcm_salt = ax.pcolormesh(y_coord, z_rho_section, salt_section,
                            cmap=cmo.haline)
    fig.colorbar(pcm_salt, ax=ax, label='Salinity [PSU]')
    ax.set_ylim(common_ylim)

    # --- Plot 3: U-Velocity ---
    ax = axes[2]
    # U-velocity is cross-shelf/along-shelf. Use vmin/vmax for flow speed.
    # Assuming u is the along-shelf velocity and 0 is the interface:
    pcm_u = ax.pcolormesh(y_coord, z_rho_section, u_section,
                        cmap=cmo.amp) # Example range
    fig.colorbar(pcm_u, ax=ax, label='$u$ [m/s]')
    ax.set_ylim(common_ylim)
    ax.set_xlabel(xlabel) # Only set x-label on the bottom plot (shared x-axis)

    for i in range(3):
        axes[i].set_facecolor('lightgray')
        axes[i].set_xlabel('Across-shelf distance [km]')

    plt.savefig(fig_path_cs, dpi = 300, bbox_inches = 'tight')

    print('plan view of initial conditions saved to '+fig_path_plan)
    print('cross sections of inital conditions saved to '+fig_path_cs)

if __name__ == '__main__':
    make_ini_no_ice()
    add_ice_to_ic()
    plot_ic()