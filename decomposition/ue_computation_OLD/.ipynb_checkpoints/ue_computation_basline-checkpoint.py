import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def compute_ue(qgpv_DA,qref_DA,ucomp_DA,uref_DA):

    #initialize
    
    qgpv = qgpv_DA.values
    qref = qref_DA.values
    ucomp = ucomp_DA.values
    uref = uref_DA.values
    times = ucomp_DA['time'].values
    lons = ucomp_DA['lon'].values
    lats = ucomp_DA['lat'].values
    ue = np.zeros(qgpv.shape)

    #compute
    
    for i,time in tqdm(enumerate(times)):
        for j,lat in enumerate(lats):
            for k,lon in enumerate(lons):
                diff = qref[i,:] - qgpv[i,j,k]
                min_index = np.argmin(abs(diff))
                ue[i,j,k] = ucomp[i,j,k] - uref[i,min_index]
                #print(j,min_index)

    #prepare output data array
    ue_DA = ucomp_DA.copy()
    ue_DA.values = ue
    return ue_DA

file_path = "/emmy-noether/home/mf671/DATA/isca_data/exeter_project_mf_baseline_KS/tot/filtered_decomposition_data_flipped_2-51.nc"

#import dataset with decomposition data (output of previous cell)
ds = xr.open_dataset(file_path)
ucomp = ds['ucomp']
qgpv = ds['dw_qgpv']
qref = ds['dw_qref']
uref = ds['dw_uref']
lon = ds['lon']
lat = ds['lat']
cosphi = np.cos(np.deg2rad(lat)).broadcast_like(ucomp)

print("time axis length: " + str(len(qref['time'])))

#computin ue
ue = compute_ue(qgpv,qref,ucomp,uref)

ds_ue = ue.to_dataset(name='ue')

ds_ue.to_netcdf("/emmy-noether/home/mf671/DATA/isca_data/exeter_project_mf_baseline_KS/tot/filtered_decomposition_ue_flipped_2-51.nc")

ds.close()