
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from pickle import dump , load
import xgboost as xgb

def clip_nan_bounds(da):
    """Crops an xarray DataArray to the smallest bounding box that contains non-NaN values."""
    valid_mask = ~np.isnan(da)
    if not valid_mask.any():
        raise ValueError("The entire DataArray contains NaNs.")

    # Get min/max indices along each axis
    non_nan_idx = np.where(valid_mask)
    min_x, max_x = non_nan_idx[-2].min(), non_nan_idx[-2].max()  # Lat/Row axis
    min_y, max_y = non_nan_idx[-1].min(), non_nan_idx[-1].max()  # Lon/Column axis

    # Slice the DataArray
    return da.isel({da.dims[-2]: slice(min_x, max_x + 1), da.dims[-1]: slice(min_y, max_y + 1)})

def run_model(f_path, out_path, var_cols, scaler_path, model_path, save_results=True, plot_results=True):

    scaler = load(open(scaler_path, 'rb'))
    model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3)

    model.load_model(model_path)

    print('Opening DataArray...')
    da = xr.open_dataset(f_path,mask_and_scale=True)['__xarray_dataarray_variable__']
    mask = da.to_masked_array(copy=False).mask[0,:,:,0]
    da = da.sel(band=var_cols) # Select features from feature sel
    da = da.stack(locs=['y','x'])
    arr = da.transpose('time','locs','band').to_numpy().squeeze()
    print('Scaling DataArray...')
    arr = scaler.transform(arr)

    print('Running Model...')
    predictions = model.predict(arr)

    da_pred = xr.DataArray(predictions,coords={'locs':da.locs}).unstack()
    da_pred = da_pred.rio.write_crs("EPSG:32632", inplace=True)
    da_pred = da_pred.where(mask == False)
    da_clipped = clip_nan_bounds(da_pred)


    if save_results == True:
        print('Saving Results...')
        da_clipped.to_netcdf(out_path)

    if plot_results == True:
        clist = ['red','yellowgreen','forestgreen'] #['dodgerblue','saddlebrown','red','yellowgreen','forestgreen']
        cmap=colors.ListedColormap(list(clist))

        labels  = {0:'Bare',1:'Standard Mangroves',2:'Tall Mangroves/Forest'} #{0:'Water',1:'Built',2:'Bare',3:'Standard Mangroves',4:'Tall Mangroves/Forest'}
        cmapList =  {0:'red',1:'yellowgreen',2:'forestgreen'} #{0:'dodgerblue',1:'saddlebrown',2:'red',3:'yellowgreen',4:'forestgreen'}
        patches =[mpatches.Patch(color=cmapList[i],label=labels[i]) for i in cmapList]

        fig,ax = plt.subplots(figsize=(8,10))
        da_clipped.plot(ax=ax,cmap=cmap,add_colorbar=False)
        ax.grid(False)
        legend = ax.legend(handles=patches,loc="upper right",frameon=1)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        ax.set_facecolor('white')
        ax.patch.set_edgecolor('black')
        fig.tight_layout()
        ax.set_title(str(out_path).split('/')[-1][:-3])
        plt.show()

