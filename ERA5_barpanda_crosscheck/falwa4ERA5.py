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

years = np.arange(1980,2000,1)
months = [1,2,12]

for year in years:
  for month in months:
    str_year = str(year)
    str_month = str(month).zfill(2) 

    print('processing month: ' + str_month + '/' + str_year)
    
    ds = xr.open_dataset('/home/links/mf671/DATA/ERA5/falwa_data/per_month/falwa_vars_daily_1.5x1.5_'+str_year+'-'+str_month+'.nc')
    
    ntimes = ds.valid_time.size
    time_array = ds.valid_time
    
    #adjust coordinates
    xlongitude = ds.longitude.values
    
    # latitude has to be in ascending order
    ylatitude_flipped = False
    ylatitude = ds.latitude.values
    
    if np.diff(ylatitude)[0]<0:
        print('Flip ylatitude.')
        ds = ds.sortby('latitude', ascending=True)
        ylatitude = ds.latitude.values
        ylatitude_flipped = True
      
    # pressure_level level has to be in descending order (ascending height)
    pressure_level_flipped = False
    pressure_level = ds.pressure_level.values
    
    if np.diff(pressure_level)[0]>0:
        print('Flip pressure_level.')    
        ds = ds.sortby('pressure_level', ascending=False)
        #ds['pressure_level'] = ds['pressure_level'][::-1]
        pressure_level = ds.latitude.values
        pressure_level_flipped = True
    
    nlongitude = xlongitude.size
    nlatitude = ylatitude.size
    nlev = pressure_level.size
    
    #set falwa parameter
    
    clatitude = np.cos(np.deg2rad(ylatitude))     # cosine latitude
    p0 = 1000.                          # surface pressure [hPa]
    kmax = 96                           # number of grid points for vertical extrapolatitudeion (dimension of height)
    dz = 500.                          # differential height element
    height = np.arange(0,kmax)*dz       # pseudoheight [m]
    dphi = np.diff(ylatitude)[0]*pi/180.     # differential latitudeitudinal element
    dlambda = np.diff(xlongitude)[0]*pi/180.  # differential latitudeitudinal element
    hh = 6745.                          # scale height
    cp = 1004.                          # heat capacity of dry air
    rr = 287.                           # gas constant
    omega = 7.29e-5                     # rotation rate of the earth
    aa = 6.378e+6                       # earth radius
    prefactor = np.array([np.exp(-z/hh) for z in height[1:]]).sum() # integrated sum of density from the level
                                                                    #just above the ground (z=1km) to aloft
    coeff = np.array([np.exp(-z/hh) for z in height[1:]])
    npart = nlatitude                        # number of partitions to construct the equivalent latitude grids
    maxits = 100000                     # maximum number of iteration in the SOR solver to solve for reference state
    tol = 1.e-5                         # tolerance that define convergence of solution
    rjac = 0.95                        # spectral radius of the Jacobi iteration in the SOR solver.              
    jd = nlatitude//2+1                      # (one plus) index of latitude grid point with value 0 deg
                                              # This is to be input to fortran code. The index convention is different.

    print(height[-1])
    print(pressure_level[-1])
    print(-hh*np.log(pressure_level[-1]/p0))
    
    #initialize lists for the integration
    
    qgpv = []
    qref = []
    uref = []
    lwa = []
    u_baro = []
    lwa_baro = []
    adv_flux_f1 = []
    adv_flux_f2 = []
    adv_flux_f3 = []
    longitude = ds['longitude'].values
    latitude = ylatitude
    pressure_level = pressure_level
    valid_time = ds['valid_time']
    
    #integration devided per time steps
    for tstep in range(ntimes):
      uu = ds.u.isel(valid_time=tstep).fillna(0).values
      vv = ds.v.isel(valid_time=tstep).fillna(0).values
      tt = ds.t.isel(valid_time=tstep).fillna(0).values
      
      qgfield_object = QGFieldNH18(xlongitude, ylatitude, pressure_level, uu, vv, tt, kmax=kmax,dz=dz,rjac=rjac,northern_hemisphere_results_only=False)
      
      qgfield_object.interpolate_fields(return_named_tuple=False)
      
      qgfield_object.compute_reference_states(return_named_tuple=False)
    
      qgfield_object.compute_lwa_and_barotropic_fluxes(return_named_tuple=False)

      print(qgfield_object.qref.shape)
      
      qgpv.append(xr.DataArray(qgfield_object.qgpv,dims=['height','latitude','longitude'],coords={'height':height,'latitude':latitude,'longitude':longitude})) #'Qusigeostrophic potential vorticity (QGPV)')
      qref.append(xr.DataArray(qgfield_object.qref,dims=['height','latitude'],coords={'height':height,'latitude':latitude}))
      uref.append(xr.DataArray(qgfield_object.uref,dims=['height','latitude'],coords={'height':height,'latitude':latitude}))
      lwa.append(xr.DataArray(qgfield_object.lwa,dims=['height','latitude','longitude'],coords={'height':height,'latitude':latitude,'longitude':longitude})) #'Local wave activity (LWA)'
      u_baro.append(xr.DataArray(qgfield_object.u_baro,dims=['latitude','longitude'],coords={'latitude':latitude,'longitude':longitude})) #'Advective flux F1'
      lwa_baro.append(xr.DataArray(qgfield_object.lwa_baro,dims=['latitude','longitude'],coords={'latitude':latitude,'longitude':longitude})) #'Advective flux F1'
      adv_flux_f1.append(xr.DataArray(qgfield_object.adv_flux_f1,dims=['latitude','longitude'],coords={'latitude':latitude,'longitude':longitude})) #'Advective flux F1'
      adv_flux_f2.append(xr.DataArray(qgfield_object.adv_flux_f2,dims=['latitude','longitude'],coords={'latitude':latitude,'longitude':longitude})) #'Advective flux F2'
      adv_flux_f3.append(xr.DataArray(qgfield_object.adv_flux_f3,dims=['latitude','longitude'],coords={'latitude':latitude,'longitude':longitude})) #'Advective flux F3'
    
    #convert lists to data arrays
    qgpv = xr.concat(qgpv,dim='valid_time').assign_coords({'valid_time':valid_time})
    qref = xr.concat(qref,dim='valid_time').assign_coords({'valid_time':valid_time})
    uref = xr.concat(uref,dim='valid_time').assign_coords({'valid_time':valid_time})
    lwa = xr.concat(lwa,dim='valid_time').assign_coords({'valid_time':valid_time})
    u_baro = xr.concat(u_baro,dim='valid_time').assign_coords({'valid_time':valid_time})
    lwa_baro = xr.concat(lwa_baro,dim='valid_time').assign_coords({'valid_time':valid_time})
    adv_flux_f1 = xr.concat(adv_flux_f1,dim='valid_time').assign_coords({'valid_time':valid_time})
    adv_flux_f2 = xr.concat(adv_flux_f2,dim='valid_time').assign_coords({'valid_time':valid_time})
    adv_flux_f3 = xr.concat(adv_flux_f3,dim='valid_time').assign_coords({'valid_time':valid_time})
  
    density_weighted_qgpv_NPA = np.sum(coeff[np.newaxis,:,np.newaxis,np.newaxis]*qgpv.values[:,1:,:,:],axis=1)/prefactor
    density_weighted_qgpv = xr.DataArray(density_weighted_qgpv_NPA,dims=['valid_time','latitude','longitude'],coords={'valid_time':valid_time,'latitude':latitude,'longitude':longitude}) #'Advective flux F1'    
    density_weighted_lwa_NPA = np.sum(coeff[np.newaxis,:,np.newaxis,np.newaxis]*lwa.values[:,1:,:,:],axis=1)/prefactor
    density_weighted_lwa = xr.DataArray(density_weighted_lwa_NPA,dims=['valid_time','latitude','longitude'],coords={'valid_time':valid_time,'latitude':latitude,'longitude':longitude}) #'Advective flux F1'
    density_weighted_qref_NPA = np.sum(coeff[np.newaxis,:,np.newaxis]*qref.values[:,1:,:],axis=1)/prefactor
    density_weighted_qref = xr.DataArray(density_weighted_qref_NPA,dims=['valid_time','latitude'],coords={'valid_time':valid_time,'latitude':latitude})
    density_weighted_uref_NPA = np.sum(coeff[np.newaxis,:,np.newaxis]*uref.values[:,1:,:],axis=1)/prefactor
    density_weighted_uref = xr.DataArray(density_weighted_uref_NPA,dims=['valid_time','latitude'],coords={'valid_time':valid_time,'latitude':latitude})
    
    qgpv.name = 'qgpv'
    ds_new = qgpv.to_dataset()
    ds_new['dw_qgpv'] = density_weighted_qgpv
    ds_new['dw_lwa'] = density_weighted_lwa
    ds_new['dw_qref'] = density_weighted_qref
    ds_new['dw_uref'] = density_weighted_uref
    ds_new['u_baro'] = u_baro
    ds_new['lwa_baro'] = lwa_baro
    ds_new['adv_flux_f1'] = adv_flux_f1
    ds_new['adv_flux_f2'] = adv_flux_f2
    ds_new['adv_flux_f3'] = adv_flux_f3
    
    #save dataset  
    ds_new.to_netcdf('/home/links/mf671/DATA/ERA5/falwa_data/falwa_output_per_month/falwa_output_daily_1.5x1.5_'+str_year+'-'+str_month+'.nc')
    
