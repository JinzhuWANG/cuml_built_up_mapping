import os
import geopandas as gpd
import numpy as np
import rasterio
import h5py
from rasterio import Affine
from tqdm.auto import tqdm


from PARAMETER import SAMPLE_PTS_PATH, REGION, SUBSET_PATH, TIF_SAVE_PATH
from tools import get_hdf_files, get_geo_meta, extract_val_to_pt


def extract_img_val_to_sample():
    ''' extract the image values to sample points
    INPUT:  None, but read the sample points from {SAMPLE_PTS_PATH}
    OUTPUT: None, but save the sample values to {SAMPLE_PTS_PATH}
    '''
    # check if the sample values already exist
    if os.path.exists(f'{SAMPLE_PTS_PATH}/sample_values_{REGION}.npy'):
        print('The sample values already exist!')

    else:

        print('Extracting image values to sample points...')

        # read the sample points
        sample_pts = gpd.read_file(f'{SAMPLE_PTS_PATH}/merge_pts_{REGION}.shp')

        # get the hdf files
        hdf_files = get_hdf_files()

        # convert the xy coordinate to col/row index
        geo_trans = get_geo_meta()['transform']
        sample_pts[['col','row']] = sample_pts['geometry'].apply(lambda geo:~geo_trans*(geo.x,geo.y)).tolist()
        sample_pts[['col','row']] = sample_pts[['col','row']].astype(int)

        # get the sample values from hdf files
        sample_values = np.hstack([extract_val_to_pt(sample_pts,f) for f in hdf_files])

        # save the sample values
        np.save(f'{SAMPLE_PTS_PATH}/sample_values_{REGION}.npy',sample_values)


def arr_to_TIFF():
    ''' save the array to tif file
    INPUT:  ds_arr: array with shape (C, H, W)
    OUTPUT: None, but export the tif file to the {TIF_SAVE_PATH}
    '''
    # open the HDF file, read the chunks
    hdf_classified =  h5py.File(f'{TIF_SAVE_PATH}/classification_{REGION}.hdf', 'r')
    hdf_classified_arr = hdf_classified['classification']
    hdf_chunks = [chunck for chunck in hdf_classified['classification'].iter_chunks()]


    # get the geo_transformation
    tif_meta = get_geo_meta() 
    tif_meta.update({'count': 1, 
                     'dtype': 'int8', 
                     'driver': 'GTiff',
                     'compress': 'lzw',
                     'height': hdf_classified_arr.shape[1],
                     'width': hdf_classified_arr.shape[2]})

    # update the geo_transformation if subset is used
    if os.path.exists(SUBSET_PATH):
        # get the xy coordinates of the bounds of the shp
        bounds = gpd.read_file(SUBSET_PATH).bounds
        trans = list(tif_meta['transform'])
        trans[2] = bounds['minx'].values[0]
        trans[5] = bounds['maxy'].values[0]
        tif_meta['transform'] = Affine(*trans)
    

    # write the array to tif, chunk by chunk
    with rasterio.open(f'{TIF_SAVE_PATH}/classification_{REGION}.tif', 
                       'w',
                       **tif_meta) as dst:
        for chunk in tqdm(hdf_chunks):
            arr = hdf_classified['classification'][chunk].astype(np.int8)
            dst.write(arr, window=rasterio.windows.Window.from_slices(chunk[1], chunk[2]))


    # close the hdf file
    hdf_classified.close()