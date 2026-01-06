############################# Cross Sections #############################
# The purpose of this notebook is to plot various cross-sections 
# of model output to characterize the domain and what is going 
# on.This notebook will look at a few corss sections in the 
# domine, plot averages, and potentially make movies of the 
# variables on interest. This notebook will include cross-
# sections of (1) salinity and temperature with density contours,
# (2) velocities, and (3) mixing and eddy viscosity/diffusivity.
#
# Notes:
# - This script runs in xroms 
# - This script heavily uses code from Dylan Schlichting 
# - This is the same as parts of plot_cross_sections_salt_temp_vel_mix.ipynb
#   but written as .py so it can be submitted as a job to make things faster
#
########################################################################


# Load in the packages
import numpy as np
import cartopy
import glob
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cmocean.cm as cmo
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import warnings
import xarray as xr
import xroms
from matplotlib import ticker
import matplotlib.colors as colors
crs = ccrs.PlateCarree()
from joblib import Parallel, delayed
warnings.filterwarnings("ignore") #turns off annoying warnings
#Cartopy
land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                edgecolor='face',
                                facecolor=cfeature.COLORS['land'])

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


def to_rho(var):
    if var.dims[-1] != 'xi_rho':
        var = grid.interp(var, 'X', to='center', boundary='extend')
    if var.dims[-2] != 'eta_rho':
        var = grid.interp(var, 'Y', to='center', boundary='extend')
    return var


def velgrad(ds,grid):
    '''
    Calculates flow invariants vorticity, divergence, strain
    '''
    us = ds.u 
    vs = ds.v

    dudx = grid.derivative(us, 'X', boundary='extend')
    dvdy = grid.derivative(vs, 'Y', boundary='extend')
    dudy = to_rho(grid.derivative(us, 'Y', boundary='extend'))
    dvdx = to_rho(grid.derivative(vs, 'X', boundary='extend'))
    # Vorticity:  v_x - u_y
    rv = (dvdx - dudy)/ds.f
    rv.name = 'Normalized vorticity'

    # Divergence: u_x + v_y
    delta = (dudx + dvdy)/ds.f
    delta.name = 'Normalized divergence'

    # Major axis of deformation
    alpha = ( np.sqrt( (dudx-dvdy)**2 + (dvdx+dudy)**2 ) )/ds.f
    alpha.name = 'Normalized strain'
    
    return rv,delta,alpha


# ---------------------------------------------------------------------------
# -------------------------- Process Output -------------------------------
# ---------------------------------------------------------------------------

# Set the path to the desired model output

# Constant forcing (time series), ice, U3C4, K-eps
#path = glob.glob('/global/cfs/cdirs/m4572/dylan617/beaufort_jet/runs/roms_avg_ice_500m_w_dvd_constant_forcing_u3c4_keps*.nc')

# Constant forcing, U3C4, k-epsilon
grd_file = '/global/homes/b/bundzis/Projects/Beaufort_ROMS_idealized_jet/Include/grd_1000_m.nc'
#path = glob.glob('/global/cfs/cdirs/m4572/dylan617/beaufort_jet/runs/roms_avg_ice_500m_w_dvd_constant_forcing_u3c4_keps_*.nc')
# Constant forcing, U3C4, gen
# grd_file = '/global/homes/b/bundzis/Projects/Beaufort_ROMS_idealized_jet/Include/grd_1000_m.nc'
path = glob.glob('/global/cfs/cdirs/m4572/dylan617/beaufort_jet/runs/roms_avg_ice_500m_w_dvd_constant_forcing_u3c4_gen_*.nc')
# Constant forcing, U3C4, kkl
# grd_file = '/global/homes/b/bundzis/Projects/Beaufort_ROMS_idealized_jet/Include/grd_1000_m.nc'
# path = glob.glob('/global/cfs/cdirs/m4572/dylan617/beaufort_jet/runs/roms_avg_ice_500m_w_dvd_constant_forcing_u3c4_kkl_*.nc')
# Constant forcing, U3C4, k-omega
# grd_file = '/global/homes/b/bundzis/Projects/Beaufort_ROMS_idealized_jet/Include/grd_1000_m.nc'
# path = glob.glob('/global/cfs/cdirs/m4572/dylan617/beaufort_jet/runs/roms_avg_ice_500m_w_dvd_constant_forcing_u3c4_komegaaltkmin_*.nc')
# Constant forcing, U3C4, kpp
# grd_file = '/global/homes/b/bundzis/Projects/Beaufort_ROMS_idealized_jet/Include/grd_1000_m.nc'
# path = glob.glob('/global/cfs/cdirs/m4572/dylan617/beaufort_jet/runs/roms_avg_ice_500m_w_dvd_constant_forcing_u3c4_kpp_*.nc')


# Load in the output from above with xroms
ds, grid = open_mfroms(path)
add_derivatives(ds, grid)

# Load in the grid
grid_nc = xr.open_dataset(grd_file)

# Calculate some submesoscale things 
# (relative vorticity (rv), gradient normalized by Coriolos (delta), 
# and ...axis of deformation? (alpha))
rv, delta, alpha = velgrad(ds, grid)

# Calculate distance in km
# to use for plotting 
# Can use x_rho, y_rho, x_u, y_u, etc. but those are a 
# little wonky...sooooo calculate it yourself I guess
# (I think if we read in the grid directly then they will
# be fixed? but are weird in the output? No clue but do manually
# for now)
dx = ds.x_rho[0,2].values - ds.x_rho[0,1].values
print('dx: ', dx)
dy = ds.y_rho[1,0].values - ds.y_rho[0,0].values
print('dy: ', dy)

len_xi_rho = len(ds.xi_rho.values)
print('len_xi_rho: ' , len_xi_rho)
len_eta_rho = len(ds.eta_rho)
print('len_eta_rho: ', len_eta_rho)

x_rho = np.arange(0, len_xi_rho*dx, dx)
print('x_rho: ', x_rho[-10:-1])
print('ds.x_rho: ', ds.x_rho[0,-10:-1].values)

y_rho = np.arange(0, len_eta_rho*dy, dy)
print('y_rho: ', y_rho[-10:-1])
print('ds.y_rho: ', ds.y_rho[-10:-1,0].values)

x_u = x_rho[:-1]
y_u = np.copy(y_rho)
x_v = np.copy(x_rho)
y_v = y_rho[:-1]

# Set the transect locations (across-shelf)
x_tran1 = np.where(x_rho == 50000)
x_tran1 = int(x_tran1[0])
print(x_tran1)
x_tran2 = np.where(x_rho == 100000)
x_tran2 = int(x_tran2[0])
print(x_tran2)
x_tran3 =  np.where(x_rho == 150000)
x_tran3 = int(x_tran3[0])
print(x_tran3)


# Calculate density using linear equation of state 
# Set the coefficients/reference values from ROMS output
R0 = ds.R0.values # kg/m3
T0 = 0.5 # Celsius
S0 = 31  # nondimensional
T_coeff = ds.Tcoef.values # 1/Celsius
S_coeff = ds.Scoef.values # 1/nondimensional

# Now calculate density
roms_density = R0 * (1 - (T_coeff*(ds.temp-T0)) + (S_coeff*(ds.salt-S0)))


# Pull things from other plots to have for big plot
# --- Salt & Temp ---
# Set the levels
lev_salt = np.arange(29, 32, 0.1)
lev_temp = np.arange(-2.5, -0.5, 0.1)
#lev_dens = np.arange(1015, 1035, 0.5)

# Set the colormaps 
cmap_salt = cmo.haline
cmap_temp = cmo.thermal

# Define the normalization instance
# ncolors specifies how many color intervals the cmap should be split into
norm_salt = BoundaryNorm(lev_salt, ncolors=cmap_salt.N, clip=True)
norm_temp = BoundaryNorm(lev_temp, ncolors=cmap_temp.N, clip=True)

# --- U & V ---
# Put u and v on the rho grid first
u_rho = to_rho(ds.u)
v_rho = to_rho(ds.v)

# Set the levels
lev_cur = np.arange(-0.2, 0.2, 0.01)

# Set the colormaps
cmap_u = cmo.curl
cmap_v = cmo.tarn

# Define the normalization instance
# ncolors specifies how many color intervals the cmap should be split into
norm_cur = BoundaryNorm(lev_cur, ncolors=cmap_u.N, clip=True)


# Pick the time to look at (one month in)
time_1mon = ds.ocean_time[168].values
print(time_1mon)
time_half_month = ds.ocean_time[372].values
print(time_half_month)

time_idx_half_month = 168
time_idx_1mon = 372

# --- Mixing ---
# Get physical mixing from the output
# Salinity (2-hourly)
Akr_rho = grid.interp(ds.AKr, 'Z')
# Temperature (2-hourly)
Akrt_rho = grid.interp(ds.AKrt, 'Z')

# Set the levels and colorbar for mixing 
cmap_mix_phy = cmo.algae
cmap_mix_num = cmo.balance
#lev_mix = np.logspace(-13, -6, num=50)
#norm_mix = BoundaryNorm(lev_mix, ncolors=cmap_mix.N, clip=True)
norm_mix_phy = colors.LogNorm(vmin=1e-10, vmax=1e-6)
#norm_mix_num = colors.LogNorm(vmin=-1e-5, vmax=1e-5)

# linthresh defines the range within which the plot is linear.
# It should be set to the smallest value you care about before it "counts" as zero.
norm_mix_num = colors.SymLogNorm(
    linthresh=1e-8, 
    vmin=-1e-3, 
    vmax=1e-3, 
    base=10
)

# Set density levels
lev_dens = np.arange(1024, 1030, 0.5)


# ---------------------------------------------------------------------------
# ----------------------- Make Function to Plot -------------------------------
# ---------------------------------------------------------------------------


# Now define function to plot the movie
def plot_salt_temp_uv_mix(time_idx, plot_dir, plot_name):
    i = time_idx // 1
    #print('i: ', i)
    #print('time_idx: ', time_idx)

    # Get all the things for this time and transect
    # Mixed layer depth
    # Find where the math equals 0.0
    iso_array = (roms_density[time_idx,:,:,x_tran1] - 
                roms_density[time_idx,-1,:,x_tran1] - 0.03)
    # Use the grid object to find the Z value at the 0.0 crossing
    # This replicates isoslice but gives you a "clean" DataArray back
    mld = grid.transform(
        ds.z_rho[time_idx,:,:,x_tran1], # The values we want (depth)
        'Z',                                               # The axis to search along
        np.array([0.0]),                                             # The target value in iso_array
        target_data=iso_array                              # The array to search through
    )
    # The transform returns a dimension for the target value (0.0), so we squeeze it out
    mld = abs(mld.squeeze())

    # Physical mixing
    # Salinity (2-hourly)
    mphys_tran1_at_time_idx = (Akr_rho).isel(xi_rho=x_tran1, ocean_time=time_idx).compute()
    mphys_tran1_at_time_idx.attrs = ''
    # Temperature (2-hourly)
    mphyt_tran1_at_time_idx = (Akrt_rho).isel(xi_rho=x_tran1, ocean_time=time_idx).compute()
    mphyt_tran1_at_time_idx.attrs = ''

    # Numerical mixing
    # Salinity 
    mnum_salt_dv_tran1_at_time_idx = (ds.dye_03).isel(xi_rho=x_tran1, ocean_time=time_idx).compute()
    # Temperature
    mnum_temp_dv_tran1_at_time_idx = (ds.dye_06).isel(xi_rho=x_tran1, ocean_time=time_idx).compute()


    # Make the figure 
    fig, ax = plt.subplots(2, 4, figsize=(12,5))

    # Set limits to focus on the jet
    for r in range(4):
        for j in range(2):
            # Set the limits
            ax[j,r].set_xlim(0,200)
            ax[j,r].set_ylim(-175,0)
            ax[j,r].set_facecolor('darkgray')
            #ax[j,r].set_aspect(1.0) # don't think this makes sense here since depth is m and across-shore is km

            # Plot density contours
            if r == 0:
                y_2d = np.tile(y_rho, (ds.z_rho[time_idx,:,:,x_tran1].shape[0], 1))
                c1 = ax[j,r].contour(y_2d/1000, ds.z_rho[time_idx,:,:,x_tran1], roms_density[time_idx,:,:,x_tran1],
                        colors='black', linewidths=0.5, levels=lev_dens)
                ax[j,r].clabel(c1, inline=True, fmt='%1.2f')


    # Plot Salt
    p1 = ax[0,0].pcolormesh(y_rho/1000, ds.z_rho[time_idx,:,:,x_tran1], ds.salt[time_idx,:,:,x_tran1],
                            cmap=cmap_salt, norm=norm_salt)
    # Add colorbar
    #plt.colorbar(p3, ax=[ax[0,0], ax[0,1], ax[0,2]], extend='both').set_label('Salinity (PSU)')
    cbar_ax1 = fig.add_axes([0.13,0.62,0.09,0.015])
    fig.colorbar(p1,ax=ax,extend='both', ticks=[29, 30, 31, 32], # label='Salinity (PSU)', 
                pad=0.03, orientation='horizontal', cax=cbar_ax1)
    # Plot mixed layer depth
    c1b = ax[0,0].plot(y_rho/1000,  mld*(-1),
                        color='aqua', linewidth=0.8)

    # Plot temp
    p2 = ax[1,0].pcolormesh(y_rho/1000, ds.z_rho[time_idx,:,:,x_tran1], ds.temp[time_idx,:,:,x_tran1],
                            cmap=cmap_temp, norm=norm_temp)
    cbar_ax2 = fig.add_axes([0.13,0.18,0.09,0.015])
    fig.colorbar(p2,ax=ax,extend='both', ticks=[-1, -2, -3, -4], # label='Temperature ($\degree$C)', 
                pad=0.03, orientation='horizontal', cax=cbar_ax2)
    # Plot mixed layer depth
    c2b = ax[1,0].plot(y_rho/1000,  mld*(-1),
                        color='aqua', linewidth=0.8)

    # Plot u
    p3 = ax[0,1].pcolormesh(y_rho/1000, ds.z_rho[time_idx,:,:,x_tran1], u_rho[time_idx,:,:,x_tran1],
                            cmap=cmap_u, norm=norm_cur)
    # Add colorbar
    #plt.colorbar(p3, ax=[ax[0,0], ax[0,1], ax[0,2]], extend='both').set_label('Salinity (PSU)')
    cbar_ax3 = fig.add_axes([0.34,0.62,0.09,0.015])
    fig.colorbar(p3,ax=ax,extend='both', ticks=[-0.15, 0, 0.15], # label='Along-Shore \nCurrent Velocity (m/s)',
                pad=0.03, orientation='horizontal', cax=cbar_ax3)
    # Plot mixed layer depth
    c3b = ax[0,1].plot(y_rho/1000,  mld*(-1),
                        color='aqua', linewidth=0.8)

    # Plot v
    p4 = ax[1,1].pcolormesh(y_rho/1000, ds.z_rho[time_idx,:,:,x_tran1], v_rho[time_idx,:,:,x_tran1],
                            cmap=cmap_v, norm=norm_cur)
    #plt.colorbar(p3, ax=[ax[0,0], ax[0,1], ax[0,2]], extend='both').set_label('Salinity (PSU)')
    cbar_ax4 = fig.add_axes([0.34,0.18,0.09,0.015])
    fig.colorbar(p4,ax=ax,extend='both', ticks=[-0.15, 0, 0.15], # label='Across-Shore \nCurrent Velocity (m/s)', 
                pad=0.03, orientation='horizontal', cax=cbar_ax4)
    # Plot mixed layer depth
    c4b = ax[1,1].plot(y_rho/1000,  mld*(-1),
                        color='aqua', linewidth=0.8)

    # Plot salt mixing - physical
    p5 = ax[0,2].pcolormesh(y_rho/1000, ds.z_rho[time_idx,:,:,x_tran1], mphys_tran1_at_time_idx[:,:],
                            cmap=cmap_mix_phy, norm=norm_mix_phy)
    #plt.colorbar(p3, ax=[ax[0,0], ax[0,1], ax[0,2]], extend='both').set_label('Salinity (PSU)')
    cbar_ax5 = fig.add_axes([0.54,0.62,0.09,0.015])
    fig.colorbar(p5,ax=ax,extend='both', ticks=[1e-9, 1e-7],
                pad=0.03, orientation='horizontal', cax=cbar_ax5)
    # Plot mixed layer depth
    c5b = ax[0,2].plot(y_rho/1000,  mld*(-1),
                        color='aqua', linewidth=0.8)

    # Plot salt mixing - numerical 
    p6 = ax[1,2].pcolormesh(y_rho/1000, ds.z_rho[time_idx,:,:,x_tran1], mnum_salt_dv_tran1_at_time_idx[:,:],
                            cmap=cmap_mix_num, norm=norm_mix_num)
    cbar_ax6 = fig.add_axes([0.54,0.18,0.09,0.015])
    fig.colorbar(p6,ax=ax,extend='both', ticks=[-1e-3, 0, 1e-3],
                pad=0.03, orientation='horizontal', cax=cbar_ax6)
    # Plot mixed layer depth
    c6b = ax[1,2].plot(y_rho/1000,  mld*(-1),
                        color='aqua', linewidth=0.8)

    # Plot temp mixing - physical 
    p7 = ax[0,3].pcolormesh(y_rho/1000, ds.z_rho[time_idx,:,:,x_tran1], mphyt_tran1_at_time_idx[:,:],
                            cmap=cmap_mix_phy, norm=norm_mix_phy)
    cbar_ax7 = fig.add_axes([0.745,0.62,0.09,0.015])
    fig.colorbar(p7,ax=ax,extend='both', ticks=[1e-9, 1e-7],
                pad=0.03, orientation='horizontal', cax=cbar_ax7)
    # Plot mixed layer depth
    c7b = ax[0,3].plot(y_rho/1000,  mld*(-1),
                        color='aqua', linewidth=0.8)

    # Plot temp mixing - numerical 
    p8 = ax[1,3].pcolormesh(y_rho/1000, ds.z_rho[time_idx,:,:,x_tran1], mnum_temp_dv_tran1_at_time_idx[:,:],
                            cmap=cmap_mix_num, norm=norm_mix_num)
    cbar_ax8 = fig.add_axes([0.745,0.18,0.09,0.015])
    fig.colorbar(p8,ax=ax,extend='both', ticks=[-1e-3, 0, 1e-3],
                pad=0.03, orientation='horizontal', cax=cbar_ax8)
    # Plot mixed layer depth
    c8b = ax[1,3].plot(y_rho/1000,  mld*(-1),
                        color='aqua', linewidth=0.8)


    # Label the plots/axes
    ax[0,0].set_title('Salinity (PSU)') #, weight='bold')
    ax[1,0].set_title('Temperature ($\degree$C)') #, weight='bold')
    ax[0,1].set_title('Along-Shore Velocity (m/s)') #, weight='bold')
    ax[1,1].set_title('Across-Shore Velocity (m/s)') #, weight='bold')
    ax[0,2].set_title('$\mathcal{M}_{phys,salt}$ (g/kg)\u00b2(m\u00b3/s)') #, weight='bold')
    ax[1,2].set_title('$\mathcal{M}_{num,salt}$ (g/kg)\u00b2(m\u00b3/s)') #, weight='bold')
    ax[0,3].set_title('$M_{phys,temp}$ ($\degree$C)\u00b2(m\u00b3/s)') #, weight='bold')
    ax[1,3].set_title('$M_{num,temp}$ ($\degree$C)\u00b2(m\u00b3/s)') #, weight='bold')

    fig.suptitle('Time: ' + str(ds.ocean_time[time_idx].values)[:10], weight='bold')
    fig.text(0.45, 0.001, 'Across-Shore Distance (km)')
    fig.text(0.06, 0.4, 'Depth (m)', rotation=90)

    # Set the spacing
    plt.subplots_adjust(hspace=0.35)

    # Hide some axis labels 
    plt.setp(ax[0,1].get_xticklabels(), visible=False)
    plt.setp(ax[0,1].get_yticklabels(), visible=False)
    plt.setp(ax[0,2].get_xticklabels(), visible=False)
    plt.setp(ax[0,2].get_yticklabels(), visible=False)
    plt.setp(ax[0,3].get_xticklabels(), visible=False)
    plt.setp(ax[0,3].get_yticklabels(), visible=False)
    plt.setp(ax[0,0].get_xticklabels(), visible=False)
    plt.setp(ax[1,1].get_yticklabels(), visible=False)
    plt.setp(ax[1,2].get_yticklabels(), visible=False)
    plt.setp(ax[1,3].get_yticklabels(), visible=False)

    # Comment this out if you are printing a lot of frames... jupyter will be unhappy
    #print(f'/pscratch/sd/b/bundzis/Beaufort_ROMS_idealized_jet_scratch/Movies/Surf_temp_salt_vort_ice/Plots01/surf_temp_salt_aice_icethick_rvort_tnum_{i}.png')
   
    plt.savefig(
        f'/pscratch/sd/b/bundzis/Beaufort_ROMS_idealized_jet_scratch/Movies/Salt_temp_uv_mix/'+plot_dir+'/'+plot_name+f'{i}.png',
        dpi=200, bbox_inches='tight'
    )
    plt.close(fig)  # Close to avoid memory leaks


# Call the function to plot - gen
Parallel(n_jobs=128)(delayed(plot_salt_temp_uv_mix)(time_idx, 'Plots02', 'salt_temp_uv_mix_const_u3c4_gen_') for time_idx in range(0, len(ds.ocean_time), 1))

# --- kkl ---
# Now delete the output and load in new output, call function again
del(ds)
# Constant forcing, U3C4, kkl
path = glob.glob('/global/cfs/cdirs/m4572/dylan617/beaufort_jet/runs/roms_avg_ice_500m_w_dvd_constant_forcing_u3c4_kkl_*.nc')
# Load in the output from above with xroms
ds, grid = open_mfroms(path)
add_derivatives(ds, grid)

# Calculate density
roms_density = R0 * (1 - (T_coeff*(ds.temp-T0)) + (S_coeff*(ds.salt-S0)))

# Interpolate onto rho points
# Put u and v on the rho grid first
u_rho = to_rho(ds.u)
v_rho = to_rho(ds.v)
# Get physical mixing from the output
# Salinity (2-hourly)
Akr_rho = grid.interp(ds.AKr, 'Z')
# Temperature (2-hourly)
Akrt_rho = grid.interp(ds.AKrt, 'Z')

# Call the function to plot - kkl
Parallel(n_jobs=128)(delayed(plot_salt_temp_uv_mix)(time_idx, 'Plots03', 'salt_temp_uv_mix_const_u3c4_kkl_') for time_idx in range(0, len(ds.ocean_time), 1))


# --- k-omega ---
# Now delete the output and load in new output, call function again
del(ds)
# Constant forcing, U3C4, k-omega
path = glob.glob('/global/cfs/cdirs/m4572/dylan617/beaufort_jet/runs/roms_avg_ice_500m_w_dvd_constant_forcing_u3c4_komega_*.nc')
# Load in the output from above with xroms
ds, grid = open_mfroms(path)
add_derivatives(ds, grid)

# Calculate density
roms_density = R0 * (1 - (T_coeff*(ds.temp-T0)) + (S_coeff*(ds.salt-S0)))

# Interpolate onto rho points
# Put u and v on the rho grid first
u_rho = to_rho(ds.u)
v_rho = to_rho(ds.v)
# Get physical mixing from the output
# Salinity (2-hourly)
Akr_rho = grid.interp(ds.AKr, 'Z')
# Temperature (2-hourly)
Akrt_rho = grid.interp(ds.AKrt, 'Z')

# Call the function to plot - kkl
Parallel(n_jobs=128)(delayed(plot_salt_temp_uv_mix)(time_idx, 'Plots04', 'salt_temp_uv_mix_const_u3c4_komega_') for time_idx in range(0, len(ds.ocean_time), 1))


# --- kpp ---
# Now delete the output and load in new output, call function again
del(ds)
# Constant forcing, U3C4, kpp
path = glob.glob('/global/cfs/cdirs/m4572/dylan617/beaufort_jet/runs/roms_avg_ice_500m_w_dvd_constant_forcing_u3c4_kpp_*.nc')
# Load in the output from above with xroms
ds, grid = open_mfroms(path)
add_derivatives(ds, grid)

# Calculate density
roms_density = R0 * (1 - (T_coeff*(ds.temp-T0)) + (S_coeff*(ds.salt-S0)))

# Interpolate onto rho points
# Put u and v on the rho grid first
u_rho = to_rho(ds.u)
v_rho = to_rho(ds.v)
# Get physical mixing from the output
# Salinity (2-hourly)
Akr_rho = grid.interp(ds.AKr, 'Z')
# Temperature (2-hourly)
Akrt_rho = grid.interp(ds.AKrt, 'Z')

# Call the function to plot - kkl
Parallel(n_jobs=128)(delayed(plot_salt_temp_uv_mix)(time_idx, 'Plots05', 'salt_temp_uv_mix_const_u3c4_kpp_') for time_idx in range(0, len(ds.ocean_time), 1))

