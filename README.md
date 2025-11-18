# beaufort_jet
Idealized ROMS model of the Alaskan Beafort shelf. Based on a fork of Rob Hetland's and Dylan Schlichting's ```shelfstrat``` repo. The model is currently setup as a buoyant plume that undergoes baroclinic instability. It is run as an initial value problem and evolves unforced. Idealized forcing and ice model will be added in the near future. 

To generate the grid, load a python environment with xarray support, and edit paths to these files
```
python grd/make_grid.py

python ini/make_ini.py
```
### Compiling and running
First, clone this repository and my roms branch 
```
git clone git@github.com:dylanschlichting/beaufort_jet.git

# ROMS with DVD + sea ice
git clone -b dylanschlichting/roms-seaice-dvd git@github.com:dylanschlichting/roms.git
```
Edit ```build_roms_no_ice.sh``` for the correct paths and application name, then 
```
./perlmutter_env.sh
./build_roms_no_ice.sh -j 4
``
That should place an executable ```romsM``` in your project directory. To run, go to your project directory and 
```
salloc --nodes 2 --qos interactive --time 04:00:00 --constraint cpu --account=m4304
srun -n 128 ./romsM ocean_beaufort_jet_unforced_no_ice_dx_1km_dz_40_layers.in >log_ice.out
```
