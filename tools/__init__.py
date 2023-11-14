import os
import h5py

import numpy as np

import umap
import umap.plot
from umap import UMAP
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import rasterio

from tqdm.auto import tqdm
from glob import glob

# set up working directory

if __name__ == '__main__':
    os.chdir('..')

# import parameters
from PARAMETER import PATH_HDF, SAMPLE_PTS_PATH, TIF_SAVE_PATH, \
                      GEO_META_PATH, REGION, YEAR_RANGE

# function to plot the smaple points in a 2d space using TSEN
def plot_sample_pts():
    ''' plot the smaple points in a 2d space using TSEN
    INPUT:  None
    OUTPUT: None, but save the plot to {TIF_SAVE_PATH}
    '''
    # get the sample points
    sample_pts = np.load(f'{SAMPLE_PTS_PATH}/sample_values_{REGION}.npy', allow_pickle=True)
    sample_X, sample_y = sample_pts[:,:-1], sample_pts[:,-1]

    # plot the sample points
    X_embedded = UMAP(n_components=2, init='random')\
                    .fit_transform(sample_X)
    # plot the sample points
    plt.figure(figsize=(10,10))
    plt.scatter(X_embedded[:,0], 
                X_embedded[:,1], 
                color=['red' if i==1 else 'blue' for i in sample_y],
                alpha=0.5,
                s=0.5)
    plt.colorbar()
    plt.title(f'Sample points of {REGION}')


# get hdf input files
def get_hdf_files():
    ''' get the hdf files from the PATH_HDF
    INPUT:  None
    OUTPUT: hdf_files: list of hdf files
    '''
    hdf_files = glob(f'{PATH_HDF}/{REGION}*.hdf')
    hdf_files = [i for i in hdf_files if (('terrain' in i)|(YEAR_RANGE in i))]
    
    # spectral unmixing hdf files are not used for classification
    hdf_files = [i for i in hdf_files if 'Spectral_Unmixing' not in i]

    if len(hdf_files) == 0:
        raise ValueError(f'No hdf file of {REGION} found in {PATH_HDF}!')

    return sorted(hdf_files)


# get the value from hdf given row/col index
def extract_val_to_pt(sample_pts,f):
    ''' extract the value from hdf given row/col index
    INPUT:  sample_pts:      dataframe with columns ['row','col']
            f:               hdf file
    OUTPUT: sample_values:   array with shape (n_sample, C)
    '''
    # read the hdf dataset
    ds = h5py.File(f, 'r')
    ds_name = list(ds.keys())[0]
    ds_arr = ds[ds_name]

    # report the progress
    print(f"Extracting values from {ds_name}...")

    # get the sample values
    sample_values = []
    for row,col in tqdm(sample_pts[['row','col']].values):
        sample_values.append(ds_arr[:,row,col])
    
    # close the dataset
    ds.close()
    
    return np.array(sample_values).astype(np.float32)

# function to get geo_transformation
def get_geo_meta():
    ''' get the geo_transformation from the tif_meta
    INPUT:  None
    OUTPUT: tif_meta: dictionary of geo_transformation'''
    # get the geo_transformation
    tif_meta = pd.read_pickle(f'{GEO_META_PATH}/tif_df.pkl')
    tif_meta = tif_meta[tif_meta['region'] == REGION]['trans'].values[0]
    return tif_meta 

# save hdf to tif

def arr_to_TIFF(ds_arr):
    ''' save the array to tif file
    INPUT:  ds_arr: array with shape (C, H, W)
    OUTPUT: None,   but export the tif file to the {TIF_SAVE_PATH}
    '''
    # open the HDF file, read the chunks
    hdf_classified =  h5py.File(f'{TIF_SAVE_PATH}/classification_{REGION}.hdf', 'r')
    hdf_chunks = [chunck for chunck in hdf_classified['classification'].iter_chunks()]

    # get the geo_transformation
    tif_meta = get_geo_meta()   
    tif_meta.update({'count': 1, 
                     'dtype': 'int8', 
                     'driver': 'GTiff',
                     'compress': 'lzw'})

    # write the array to tif, chunk by chunk
    with rasterio.open(f'{TIF_SAVE_PATH}/classification_{REGION}.tif', 
                       'w',
                       **tif_meta) as dst:
        for chunk in tqdm(hdf_chunks):
            arr = hdf_classified['pred'][chunk].astype(np.int8)
            dst.write(arr, window=rasterio.windows.Window(chunk[1], chunk[0], arr.shape[1], arr.shape[0]))


    # close the hdf file
    hdf_classified.close()