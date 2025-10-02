import datetime as dt
from math import pi
import numpy as np
from numpy import dtype
import xarray as xr

#my own classes
import sys
sys.path.append("/emmy-noether/home/mf671/github/hn2016_falwa/src/falwa/")
from wrapper import qgpv_eqlat_lwa
from oopinterface import QGFieldNH18
import utilities as utilities

for run in np.arange(13,613,1):
  str_run = str(run).zfill(4) 
  print('processing run: ' + str_run)

  ds = xr.open_dataset('/home/links/mf671/DATA/isca_data/exeter_project_mf_baseline_st_KS/falwa_vars/atm_daily_uvt_'+str_run+'.nc')
  
  ntimes = ds.time.size
  time_array = ds.time
  
  #adjust coordinates
  xlon = ds.lon.values
  
  # lat has to be in ascending order
  ylat_flipped = False
  ylat = ds.lat.values  
  if np.diff(ylat)[0]<0:
      print('Flip ylat.')
      ds = ds.sortby('lat', ascending=True)
      ylat = ds.lat.values
      ylat_flipped = True
    
  # pfull level has to be in descending order (ascending height)
  pfull_flipped = False
  pfull = ds.pfull.values
  if np.diff(pfull)[0]>0:
      print('Flip pfull.')    
      ds = ds.sortby('pfull', ascending=False)
      pfull = ds.lat.values
      pfull_flipped = True
  
  nlon = xlon.size
  nlat = ylat.size
  nlev = pfull.size
  
  #set falwa parameter
  
  clat = np.cos(np.deg2rad(ylat))     # cosine lat
  p0 = 1000.                          # surface pressure [hPa]
  kmax = 41                           # number of grid points for vertical extrapolation (dimension of height)
  dz = 500.                          # differential height element
  height = np.arange(0,kmax)*dz       # pseudoheight [m]
  dphi = np.diff(ylat)[0]*pi/180.     # differential latitudinal element
  dlambda = np.diff(xlon)[0]*pi/180.  # differential latitudinal element
  hh = 6745.                          # scale height
  cp = 1004.                          # heat capacity of dry air
  rr = 287.                           # gas constant
  omega = 7.29e-5                     # rotation rate of the earth
  aa = 6.378e+6                       # earth radius
  prefactor = np.array([np.exp(-z/hh) for z in height[1:]]).sum() # integrated sum of density from the level
                                                                  #just above the ground (z=1km) to aloft
  coeff = np.array([np.exp(-z/hh) for z in height[1:]])
  npart = nlat*5                        # number of partitions to construct the equivalent lat grids
  maxits = 100000                     # maximum number of iteration in the SOR solver to solve for reference state
  tol = 1.e-5                         # tolerance that define convergence of solution
  rjac = 0.95                        # spectral radius of the Jacobi iteration in the SOR solver.              
  jd = nlat//2+1                      # (one plus) index of lat grid point with value 0 deg
                                            # This is to be input to fortran code. The index convention is different.

  print(height[-1])
  print(pfull[-1])
  print(-hh*np.log(pfull[-1]/p0))
  
  #initialize lists for the integration
  
  qgpv = []
  qref = []
  lwa = []
  u_baro = []
  adv_flux_f1 = []
  adv_flux_f2 = []
  adv_flux_f3 = []
  lon = ds['lon'].values
  lat = ylat
  pfull = pfull
  time = ds['time']
  
  #integration devided per time steps
  for tstep in range(ntimes):
    uu = ds.ucomp.isel(time=tstep).fillna(0).values
    vv = ds.vcomp.isel(time=tstep).fillna(0).values
    tt = ds.temp.isel(time=tstep).fillna(0).values
    
    qgfield_object = QGFieldNH18(xlon, ylat, pfull, uu, vv, tt, kmax=kmax,dz=dz,rjac=rjac,scale_height=hh,northern_hemisphere_results_only=False)
    
    qgfield_object.interpolate_fields(return_named_tuple=False)
    
    qgfield_object.compute_reference_states(return_named_tuple=False)
  
    qgfield_object.compute_lwa_and_barotropic_fluxes(return_named_tuple=False)

    print(qgfield_object.qref.shape)
    
    qgpv.append(xr.DataArray(qgfield_object.qgpv,dims=['height','lat','lon'],coords={'height':height,'lat':lat,'lon':lon})) #'Qusigeostrophic potential vorticity (QGPV)')
    qref.append(xr.DataArray(qgfield_object.qref,dims=['height','lat'],coords={'height':height,'lat':lat}))
    lwa.append(xr.DataArray(qgfield_object.lwa,dims=['height','lat','lon'],coords={'height':height,'lat':lat,'lon':lon})) #'Local wave activity (LWA)'
    u_baro.append(xr.DataArray(qgfield_object.u_baro,dims=['lat','lon'],coords={'lat':lat,'lon':lon})) #'Advective flux F1'
    adv_flux_f1.append(xr.DataArray(qgfield_object.adv_flux_f1,dims=['lat','lon'],coords={'lat':lat,'lon':lon})) #'Advective flux F1'
    adv_flux_f2.append(xr.DataArray(qgfield_object.adv_flux_f2,dims=['lat','lon'],coords={'lat':lat,'lon':lon})) #'Advective flux F2'
    adv_flux_f3.append(xr.DataArray(qgfield_object.adv_flux_f3,dims=['lat','lon'],coords={'lat':lat,'lon':lon})) #'Advective flux F3'
  
  #convert lists to data arrays
  qgpv = xr.concat(qgpv,dim='time').assign_coords({'time':time})
  qref = xr.concat(qref,dim='time').assign_coords({'time':time})
  lwa = xr.concat(lwa,dim='time').assign_coords({'time':time})
  u_baro = xr.concat(u_baro,dim='time').assign_coords({'time':time})
  adv_flux_f1 = xr.concat(adv_flux_f1,dim='time').assign_coords({'time':time})
  adv_flux_f2 = xr.concat(adv_flux_f2,dim='time').assign_coords({'time':time})
  adv_flux_f3 = xr.concat(adv_flux_f3,dim='time').assign_coords({'time':time})

  density_weighted_qgpv_NPA = np.sum(coeff[np.newaxis,:,np.newaxis,np.newaxis]*qgpv.values[:,1:,:,:],axis=1)/prefactor
  density_weighted_qgpv = xr.DataArray(density_weighted_qgpv_NPA,dims=['time','lat','lon'],coords={'time':time,'lat':lat,'lon':lon}) #'Advective flux F1'    
  density_weighted_lwa_NPA = np.sum(coeff[np.newaxis,:,np.newaxis,np.newaxis]*lwa.values[:,1:,:,:],axis=1)/prefactor
  density_weighted_lwa = xr.DataArray(density_weighted_lwa_NPA,dims=['time','lat','lon'],coords={'time':time,'lat':lat,'lon':lon}) #'Advective flux F1'
  density_weighted_qref_NPA = np.sum(coeff[np.newaxis,:,np.newaxis]*qref.values[:,1:,:],axis=1)/prefactor
  density_weighted_qref = xr.DataArray(density_weighted_qref_NPA,dims=['time','lat'],coords={'time':time,'lat':lat})
  
  qgpv.name = 'qgpv'
  ds_new = qgpv.to_dataset()
  ds_new['dw_qgpv'] = density_weighted_qgpv
  ds_new['dw_lwa'] = density_weighted_lwa
  ds_new['dw_qref'] = density_weighted_qref
  ds_new['u_baro'] = u_baro
  ds_new['adv_flux_f1'] = adv_flux_f1
  ds_new['adv_flux_f2'] = adv_flux_f2
  ds_new['adv_flux_f3'] = adv_flux_f3
  
  ds_new.to_netcdf('/home/links/mf671/DATA/isca_data/exeter_project_mf_baseline_st_KS/falwa_output/falwa_output_1.5x1.5_'+str_run+'.nc')
