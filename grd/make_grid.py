"""
ROMS Grid Generation from GEBCO Bathymetry

This script extracts a cross-shelf mean bathymetry from a GEBCO NetCDF file,
fits a tanh function to represent the shelf slope, and generates a ROMS C-grid
with optional random bathymetry noise. The grid and bathymetry are saved to a 
NetCDF file for use in ROMS simulations.

Command-line arguments:

--dx       : float, optional, default=1000
             Grid spacing in the x-direction (meters).

--dy       : float, optional, default=1000
             Grid spacing in the y-direction (meters).

--Lx_km    : float, optional, default=201
             Domain length in the x-direction (km). 

--Ly_km    : float, optional, default=251
             Domain length in the y-direction (km).

--ncfile   : str, default="/pscratch/sd/d/dylan617/beaufort_roms/generate_inputs/gebco_2025_n75.0_s68.0_w-154.0_e-138.0.nc"
             Path to the input GEBCO bathymetry NetCDF file. Change this to where you have it stored!

Note: the resulting roms.in file will have these properties

          Lm == Lx_km-3            ! Number of I-direction INTERIOR RHO-points
          Mm == Ly_km-3            ! Number of J-direction INTERIOR RHO-points

Bathymetry formula:

    h(x) = | H_min + 0.5*(H_offshore - H_min) * (1 + tanh((x - x_mid)/L)) - 13 |

where:
    H_min      : minimum coastal depth (m)
    H_offshore : offshore depth (m)
    x_mid      : midpoint of the shelf slope (km)
    L          : slope width scale (km)
    13         : artificial shoaling (m)
    | ... |    : ensures depth is positive
    h[0]      : enforced minimum depth at coast = 5 m
"""

import numpy as np
import xarray as xr
import os
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

def make_CGrid(x, y):
    """
    Construct a ROMS C-grid (rho, u, v, psi points, and grid metrics pm/pn).
    
    Inputs:
        x, y: 2D arrays of vertex coordinates (meters)
    Outputs:
        xr.Dataset containing:
            - x_rho, y_rho: RHO-point coordinates
            - x_u, y_u: U-point coordinates
            - x_v, y_v: V-point coordinates
            - x_psi, y_psi: PSI-point coordinates
            - pm, pn: inverse grid spacing (1/dx, 1/dy)
    """
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        x = np.ma.masked_where((np.isnan(x)) | (np.isnan(y)), x)
        y = np.ma.masked_where((np.isnan(x)) | (np.isnan(y)), y)

    ds = xr.Dataset({'x_vert': (['eta_vert', 'xi_vert'], x),
                     'y_vert': (['eta_vert', 'xi_vert'], y)})

    # RHO, U, V, PSI points
    ds['x_rho'] = (['eta_rho', 'xi_rho'], 0.25 * (x[1:, 1:] + x[1:, :-1] + x[:-1, 1:] + x[:-1, :-1]))
    ds['y_rho'] = (['eta_rho', 'xi_rho'], 0.25 * (y[1:, 1:] + y[1:, :-1] + y[:-1, 1:] + y[:-1, :-1]))
    ds['x_u'] = (['eta_u', 'xi_u'], 0.5 * (x[:-1, 1:-1] + x[1:, 1:-1]))
    ds['y_u'] = (['eta_u', 'xi_u'], 0.5 * (y[:-1, 1:-1] + y[1:, 1:-1]))
    ds['x_v'] = (['eta_v', 'xi_v'], 0.5 * (x[1:-1, :-1] + x[1:-1, 1:]))
    ds['y_v'] = (['eta_v', 'xi_v'], 0.5 * (y[1:-1, :-1] + y[1:-1, 1:]))
    ds['x_psi'] = (['eta_psi', 'xi_psi'], x[1:-1, 1:-1])
    ds['y_psi'] = (['eta_psi', 'xi_psi'], y[1:-1, 1:-1])

    # Grid metrics
    x_temp = 0.5 * (ds.x_vert[1:, :] + ds.x_vert[:-1, :])
    y_temp = 0.5 * (ds.y_vert[1:, :] + ds.y_vert[:-1, :])
    dx = np.sqrt(np.diff(x_temp, axis=1)**2 + np.diff(y_temp, axis=1)**2)
    x_temp = 0.5 * (ds.x_vert[:, 1:] + ds.x_vert[:, :-1])
    y_temp = 0.5 * (ds.y_vert[:, 1:] + ds.y_vert[:, :-1])
    dy = np.sqrt(np.diff(x_temp, axis=0)**2 + np.diff(y_temp, axis=0)**2)

    ds['pm'] = (['eta_rho', 'xi_rho'], 1. / dx)
    ds['pn'] = (['eta_rho', 'xi_rho'], 1. / dy)

    return ds

def make_grd_from_bathymetry(bfit, x_km, dx=1000, dy=1000,
                             Lx_km=201, Ly_km=251,
                             output='/global/homes/b/bundzis/Projects/Beaufort_ROMS_idealized_jet/Include/grd.nc',
                             spherical=False, angle=0.0):
    """
    Generate a ROMS C-grid using a 1D bathymetry profile.

    Inputs:
        bfit: fitted bathymetry values (meters, positive downward)
        x_km: cross-shelf distance (km)
        dx, dy: horizontal grid spacing (meters)
        Lx_km, Ly_km: domain dimensions (km)
        output: path to write netCDF grid file
    Outputs:
        xr.Dataset containing ROMS C-grid
    """

    # --- Grid coordinates (vertices) ---
    nx_vert = int(Lx_km * 1000 / dx)
    ny_vert = int(Ly_km * 1000 / dy)
    x = np.arange(nx_vert) * dx
    y = np.arange(ny_vert) * dy
    x_vert, y_vert = np.meshgrid(x, y)

    # --- Build C-grid ---
    grd = make_CGrid(x_vert, y_vert)

    # --- Interpolate bathymetry ---
    y_rho_km = grd['y_rho'].values[:, 0] / 1000.0
    b_eta = np.interp(y_rho_km, x_km, bfit)
    h_grid = np.tile(b_eta[:, None], (1, grd.dims['xi_rho']))


    # --- Add random noise equal to 0.5% of local depth ---
    noise_amplitude = 0.02 * h_grid
    rng = np.random.default_rng(seed=42)  # deterministic for reproducibility
    h_grid_noisy = h_grid + rng.uniform(-1, 1, size=h_grid.shape) * noise_amplitude

    grd['h'] = (['eta_rho', 'xi_rho'], np.abs(h_grid_noisy))

    # grd['h'] = (['eta_rho', 'xi_rho'], np.abs(h_grid))

    # --- Add Coriolis, angle, etc. ---
    f_value = 1.367e-4  
    grd['f'] = f_value * xr.ones_like(grd.pm)
    grd.f.attrs.update({
        'long_name': 'Coriolis parameter at RHO-points',
        'units': 'second-1',
        'field': 'Coriolis, scalar'
    })
    grd['angle'] = angle * xr.ones_like(grd.pm)
    grd.angle.attrs.update({
        'long_name': 'angle between xi axis and east',
        'units': 'degree'
    })
    grd['spherical'] = spherical
    grd['xl'] = x_vert.max()
    grd['el'] = y_vert.max()

    visc_factor = xr.ones_like(grd.pm)
    visc_factor.attrs.update({
        'long_name': 'Horizontal viscosity factor at RHO-points',
        'units': 'nondimensional',
        'field': 'VISC_FACTOR, scalar'
    })
    grd['visc_factor'] = visc_factor

    diff_factor = xr.ones_like(grd.pm)
    diff_factor.attrs.update({
        'long_name': 'Horizontal diffusivity factor at RHO-points',
        'units': 'nondimensional',
        'field': 'DIFF_FACTOR, scalar'
    })
    grd['diff_factor'] = diff_factor
    # --- Write file ---
    if os.path.exists(output):
        os.remove(output)
        print(f"Existing grid file '{output}' deleted.")

    grd.to_netcdf(output)
    print(f"✅ Grid file successfully written to {output}")

    return grd

def extract_mean_bathymetry(ncfile, lon_max=152, lat_max=72, smooth_sigma=2):
    """
    Extract and smooth a longitudinal mean bathymetry profile from a GEBCO NetCDF file.
    """
    ds = xr.open_dataset(ncfile)
    bathy = ds.elevation.where(ds.elevation < 0).where(ds.lon < lon_max).where(ds.lat < lat_max)
    b = bathy.mean('lon')

    b_clean = b.dropna('lat')
    depth = b_clean.values
    lat = b_clean.lat.values

    # Smooth the depth profile
    depth_smooth = gaussian_filter1d(depth, sigma=smooth_sigma)

    # Convert latitude to cross-shelf distance (km)
    km_per_deg = 111
    x_km = (lat - lat[0]) * km_per_deg

    return x_km, depth_smooth


def fit_tanh_bathymetry(x_km, depth_smooth):
    """
    Fit a tanh function to the cross-shelf bathymetry:

        h(x) = | H_min + 0.5 * (H_offshore - H_min) * (1 + tanh((x - x_mid)/L)) - 13 |

    where:
        H_min      = minimum depth (coastal shallowest point)
        H_offshore = offshore depth
        x_mid      = midpoint of the shelf slope
        L          = slope width scale
        -13        = artificial shoaling
        | ... |    = ensure positive depth
        h[0]      = 5 m  (enforce shallowest coastal depth)

    Returns:
        b_fit: bathymetry values (m)
        popt: fitted parameters
    """

    def tanh_bathymetry(x, H_min, H_offshore, x_mid, L):
        return H_min + 0.5 * (H_offshore - H_min) * (1 + np.tanh((x - x_mid) / L))

    # Initial guesses
    H_min_guess = depth_smooth[0]
    H_offshore_guess = depth_smooth[-1]
    x_mid_guess = x_km[np.argmax(np.gradient(depth_smooth, x_km))]
    L_guess = 60  # typical slope width (km)
    p0 = [H_min_guess, H_offshore_guess, x_mid_guess - 10, L_guess]

    popt, _ = curve_fit(tanh_bathymetry, x_km, depth_smooth, p0=p0)
    b_fit = np.abs(tanh_bathymetry(x_km, *popt))

    # Artificially shoal the shelf to match better with observations. Set minimum H 
    # to 5 m to be more realistic. This should be improved, but is good enough for 
    # a starting point. 
    b_fit = b_fit - 13
    b_fit[0] = 5

    H_min, H_offshore, x_mid, L = popt
    print(f"Fitted tanh parameters:\n H_min={H_min:.2f}, H_offshore={H_offshore:.2f}, x_mid={x_mid:.2f}, L={L:.2f}")

    return b_fit, popt


# === COMBINED PIPELINE ===

def prepare_bathymetry_for_grid(ncfile):
    """
    Full pipeline: extract mean bathymetry, smooth, fit tanh curve, and prepare for ROMS grid.
    """
    x_km, depth_smooth = extract_mean_bathymetry(ncfile)
    b_fit, params = fit_tanh_bathymetry(x_km, depth_smooth)
    return b_fit, x_km


if __name__ == "__main__":
    import argparse

    # --- Define command-line arguments ---
    parser = argparse.ArgumentParser(description="Generate a ROMS grid from GEBCO bathymetry.")
    parser.add_argument(
        "--dx", type=float, default=1000,
        help="Grid spacing in the x-direction (m). Default = 1000"
    )
    parser.add_argument(
        "--dy", type=float, default=1000,
        help="Grid spacing in the y-direction (m). Default = 1000"
    )
    parser.add_argument(
        "--Lx_km", type=float, default=201,
        help="Domain length in x (km). Default = 200"
    )
    parser.add_argument(
        "--Ly_km", type=float, default=251,
        help="Domain length in y (km). Default = 250"
    )
    parser.add_argument(
        "--ncfile", type=str, default="/pscratch/sd/b/bundzis/Beaufort_ROMS_idealized_jet_scratch/gebco_2025_n75.0_s68.0_w-154.0_e-138.0.nc",
        help="Path to input GEBCO bathymetry NetCDF file."
    )

    args = parser.parse_args()

    # --- Run the workflow ---
    b_fit, x_km = prepare_bathymetry_for_grid(args.ncfile)

    grd = make_grd_from_bathymetry(
        b_fit,
        x_km,
        dx=args.dx,
        dy=args.dy,
        Lx_km=args.Lx_km,
        Ly_km=args.Ly_km
    )
