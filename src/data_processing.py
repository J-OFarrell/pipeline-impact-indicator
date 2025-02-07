# For export_ee
import ee
import os
import numpy as np
import rioxarray as rxr
import xarray as xr
import sys
sys.path.append('/home/jemima/Data/gee_s1_ard/python-api/')
import wrapper as wp

sys.path.append('../src/')

def GLCM(img):
    img = img.int32()
    glcmVV = img.select('VV').glcmTexture()
    glcmVH = img.select('VH').glcmTexture()
    return img.addBands(glcmVV).addBands(glcmVH)

def getS1_imcol(param_grid, bounds, start_year, end_year, orbit_nums):
    startDate = str(int(start_year))+'-01-01'
    endDate = str(int(end_year))+'-01-01'

# Update parameter grid
    param_grid['START_DATE'] = startDate
    param_grid['STOP_DATE'] = endDate
    param_grid['ROI'] = bounds
    param_grid['ORBIT_NUM'] = orbit_nums

# Get processed Sentinel-1 ImageCollection
    s1_processed = wp.s1_preproc(param_grid)

# Apply GLCM texture analysis
    glcm_features = ['VV', 'VH', 'angle', 'VV_contrast', 'VV_corr', 'VV_diss', 'VV_ent' ,'VV_var', 'VV_idm', 'VV_savg', 'VV_asm', 'VH_contrast', 'VH_corr', 'VH_diss', 'VH_ent' ,'VH_var', 'VH_idm', 'VH_savg', 'VH_asm']
    s1_processed = s1_processed.map(GLCM).select(glcm_features)
    return s1_processed


def getS1_composites(param_grid, bounds, start_year, end_year, orbit_nums):

  imcol = getS1_imcol(param_grid, bounds, start_year, end_year, orbit_nums)
  bandNames = imcol.first().bandNames().getInfo()

  statsImg = imcol.median()
  medBandNames = [s + '_median' for s in bandNames]
  statsImg = statsImg.rename(medBandNames)

  statsImg_min = imcol.min()
  minBandNames = [s + '_min' for s in bandNames]
  statsImg_min = statsImg_min.rename(minBandNames)
  statsImg = statsImg.addBands(statsImg_min)

  statsImg_max = imcol.max()
  maxBandNames = [s + '_max' for s in bandNames]
  statsImg_max = statsImg_max.rename(maxBandNames)
  statsImg = statsImg.addBands(statsImg_max)

  return statsImg


def stackbytile(dirpath,outpath,year_start,year_end,save_file=True):
    """
    This works if your exports have a filename extension (eg. 2020_0000000000_0000004352.nc) as a result of your AOI spanning several 
    Sentinel-1 tiles or a larger area than EE export will allow in one tiff. This function will create .nc xarray.Dataarray stacks 
    shaped (time, y, x, bands) and save them by file extension name.
    """

    yearList = np.linspace(year_start,year_end,(year_end-year_start+1))
    f_list = sorted([filename for filename in os.listdir(dirpath)])
    if len(f_list[0]) > 7: 
        f_extensions = np.unique([str(fname)[4:] for fname in f_list]) 
    else:
        print('No f_name extension found - '+str(f_list[0]))

    print('File Extensions Found - ',f_extensions)
    for f_ext in f_extensions:
        print('Stacking - ',f_ext)
        da_list = []
        for year in yearList:
            f_path = dirpath+str(int(year))+str(f_ext)
            raster = rxr.open_rasterio(f_path, masked=True).transpose('y','x','band')
            da = raster.expand_dims(dim={'time':[str(int(year))]},axis=0)
            da_list.append(da)
        da_merge = xr.concat(da_list, dim='time')
        if save_file == True:
            da_merge.to_netcdf(outpath+str(f_ext)[1:-3]+".nc")


def stackbytime(dirpath,outpath,year_start,year_end,save_file=True):
    """
    This is for stacking multiple tiles into one mosaiced image grouped by time.
    """

    yearList = np.linspace(year_start,year_end,(year_end-year_start+1))
    f_list = sorted([filename for filename in os.listdir(dirpath)])

    for year in yearList.astype(int):
        print('Stacking - ',int(year))
        da_list = []
        f_list_sub = [str(f_name) for f_name in f_list if str(f_name)[:4] == str(year)]

        for f_name in f_list_sub:
            f_path = dirpath+f_name
            raster = rxr.open_rasterio(f_path, masked=True)
            da = raster.transpose('y','x','band')
            da['band'] = [bandname for bandname in raster.long_name]
            da_list.append(da)

        da_merge = xr.combine_by_coords(da_list, join="outer", combine_attrs="override")
        da_merge = da_merge.expand_dims(dim={'time':[str(int(year))]},axis=0)
        if save_file == True:
            da_merge.to_netcdf(outpath+str(year)+".nc")



def StackIms(dirpath):
    im_list = sorted([filename for filename in os.listdir(dirpath)])
    ref_tiff = rxr.open_rasterio(dirpath+im_list[0], masked=True).to_numpy()
    ref_shape = ref_tiff.shape
    stacksize = len(im_list)
    Stack = np.zeros((int(stacksize),ref_shape[1],ref_shape[2],ref_shape[0]), dtype = np.float32) #time, y, x, bands

    for i in range(int(stacksize)):
        Img = rxr.open_rasterio(dirpath+im_list[i], masked=True).to_numpy()
        Img = np.transpose(Img,[1,2,0])
        Stack[i,:,:,:] = Img

    return Stack



