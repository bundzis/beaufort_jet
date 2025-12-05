# beaufort_jet
Idealized ROMS model of the Alaskan Beafort shelf. Based on a fork of Rob Hetland's and Dylan Schlichting's ```shelfstrat``` repo. The model is currently setup as a buoyant plume that undergoes baroclinic instability. It is run as an initial value problem and evolves unforced. 

To generate the grid and initial conditions, load a python environment with xarray support, and edit paths in these files
```
python grd/make_grid.py

python ini/make_ini.py
```
Bathymetry is based on GEBCO and fit with a hyperbolic tangent function that is artificially shoaled to enforce a minimum water depth of 5 m. This could be modified to be more like the observations, see ```grd/gebco_roms_bathymetry.ipynb``` for a comparison. 

### Compiling and running 
Clone my ROMS branch, which is based on the myroms develop branch v4.1 (I think). Modified to include discrete variance decay (DVD) analysis of temperature mixing and open boundary condition support for sea ice. DVD added by Brianna Undzis, and sea ice modifications added by Tale Bakken Ulfsby, which you can see in the commit history. 
```
git clone -b dylanschlichting/roms-seaice-dvd git@github.com:dylanschlichting/roms.git
```
I suggest you have two directories for executables, one with/without ice so the executable isn't overwritten when switching. Edit ```build_roms_no_ice.sh``` or ```build_roms_ice.sh``` for the correct paths and application name. The relevant / required analytical functions are stored in ```project/Functionals```. 

Then
```
./perlmutter_env.sh
./build_roms_no_ice.sh -j 4 
# or 
./build_roms_ice.sh -j 4
```
That should place an executable ```romsM``` in your project directory. To run, go to your project directory and 
```
# No ice 
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint cpu --account=m4304

srun -n 128 ./romsM ocean_beaufort_jet_unforced_no_ice_dx_1km_dz_40_layers.in >log_no_ice.out 2>&1
srun -n 128 ./romsM ocean_beaufort_jet_bulk_fluxes_ice_dx_1km_dz_40_layers.in >log_ice.out 2>&1
```
### Model properties and notes
Edit as you see fit. Both the ice and ice-free models share the following properties (for 1km):
- 60 sec DT
- U3HC4 tracer advection scheme
- k-epsilon vertical mixing
- No nudging
- No lateral mixing for momentum or tracers
- DVD header flags because it will slow the model down. This can be turned on. 

Ice-free model is UNFORCED, so it runs purely as an initial value problem. Ice model is stable with ```nEVP=60``` but requires ```BULK_FLUXES``` to run and form sea ice from the current initial conditions. If you change the with ice application name, you must grep and replace it in all relevant analyticals or compiling/running will break. Also, I did not do a complete comparison of the ice / no ice header options (excluding bulk fluxes). This needs to be checked and corrected as necessary.