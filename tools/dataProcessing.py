import os
from glob import glob

import geopandas as gpd
import numpy as np
import pandas as pd

import rasterio
import h5py
from rasterio import Affine
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import re

# setting up working directory
if __name__ == '__main__':
    os.chdir('..')


from PARAMETER import BASE_PATH, OVERLAY_THRESHOLD, SAMPLE_PTS_PATH, PATH_HDF, \
                      REGION, SUBSET_PATH, TIF_SAVE_PATH, INDICES_CAL_EXPRESSION

from tools import get_hdf_files, get_geo_meta, extract_val_to_pt
from dataprep.dataprep_tools import get_str_info



def compute_indices_for_all_landsat():
     # get the hdf files
    hdf_files = get_hdf_files()
    hdf_landsat = [f for f in hdf_files if 'Landsat' in f]

    # loop through each landsat hdf file
    for path in hdf_landsat:

        # loop through each index
        indices = INDICES_CAL_EXPRESSION.keys()

        for index in indices:
            # check if the index already exist
            year_range = get_str_info(path)[-1]
            index_path = f"{PATH_HDF}/{REGION}_{index}_{year_range}.hdf"

            if os.path.exists(index_path):
                print(f'{index_path} already exists!\n')
            else:
                # compute the index
                print(f'Computing {index} for {os.path.basename(path)}...')
                compute_normalized_indices(path, index)



def compute_indices(arr_slice, year, index:str):
    """
    Computes the index for a given Landsat type and array slice.

    Args:
        arr_slice (numpy.ndarray): The array slice to compute the index for.
        year (int): The year of the Landsat type.
        index (str): The index to compute.

    Returns:
        numpy.ndarray: The computed index array.
    """
    # convert the array to float32
    arr_slice = arr_slice.astype(np.float32)

    # get the landsat type
    if year<=2010:
        landsat_type = 'Landsat5'
    elif year<=2013:
        landsat_type = 'Landsat7'
    else:
        landsat_type = 'Landsat8'

    # get the expression
    expression = INDICES_CAL_EXPRESSION[index][landsat_type]
    # use re to replace the band name with the array
    for band in range(arr_slice.shape[0]):
        expression = re.sub(f'Band_{band}',f'arr_slice[{band-1}]',expression)

    # compute the index
    index_arr = eval(expression)
    # rescale and change the dtype
    index_arr = (index_arr*127).astype(np.int8)

    return index_arr




def compute_normalized_indices(landsat_img_path:str,index:str):
    """
    Computes a list of normalized indices for a Landsat image and saves them to an HDF5 file.

    Args:
    - landsat_img_path (str): the path to the Landsat image file in HDF format.
    - indices: a string with a single index name.

    Returns:
    None
    """

    # get the year of the landsat image
    region,img_type,year_range = get_str_info(landsat_img_path)
    year = int(year_range[-4:])

    # create the name for output hdf file
    out_name = f"{PATH_HDF}/{region}_{index}_{year_range}.hdf"

    # get the shape, blocks of the landsat image
    landsat_hdf = h5py.File(landsat_img_path, 'r')
    landsat_arr = landsat_hdf[list(landsat_hdf.keys())[0]]
    ds_shape = landsat_arr.shape
    block_size = landsat_arr.chunks[1]

    # create an empty hdf file for holding the computed indices
    with h5py.File(out_name,mode='w') as hdf_file:
        # Create a dataset and save the NumPy array to it
        hdf_file.create_dataset(index, 
                                shape=ds_shape,
                                dtype=np.int8, 
                                fillvalue=np.nan, 
                                compression="gzip", 
                                compression_opts=9,
                                chunks=(ds_shape[0],block_size,block_size))

    # loop through each block to compute the indices
    slice_rows = range(0,ds_shape[1],block_size)
    slice_cols = range(0,ds_shape[2],block_size)
    slices = [(slice(None),slice(row,row+block_size),slice(col,col+block_size))
                    for row in slice_rows for col in slice_cols]
    
    # loop through each slice and compute the index
    for slice_ in tqdm(slices):
        # get the array_slice
        arr_slice = landsat_arr[slice_]
        # compute the index
        index_arr = compute_indices(arr_slice, year, index)
        # save the index to hdf
        with h5py.File(out_name,mode='r+') as hdf_file:
            hdf_file[index][slice_] = index_arr






def extract_img_val_to_sample(force_resample=False):
    ''' extract the image values to sample points
    INPUT:  None, but read the sample points from {SAMPLE_PTS_PATH}
    OUTPUT: None, but save the sample values to {SAMPLE_PTS_PATH}
    '''
    # check if the sample values already exist
    if os.path.exists(f'{SAMPLE_PTS_PATH}/sample_values_{REGION}.npy') and not force_resample:
        print('The sample values already exist!\n')

    else:

        print('Extracting image values to sample points...\n')

        # get the sample points
        sample_path = f'{SAMPLE_PTS_PATH}/merge_pts_{REGION}.shp'
        if not os.path.exists(sample_path):
            raise ValueError(f'{sample_path} does not exist!')
        
        sample_pts = gpd.read_file(sample_path)

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
    print('Saving the classified hdf to tif...')

    # open the HDF file, read the chunks
    hdf_classified_paths =  glob(f'{TIF_SAVE_PATH}/classification_{REGION}_*.hdf')

    for path in hdf_classified_paths:
        # get the sample_type
        sample_type = re.compile(rf'{REGION}_(.*).hdf').findall(path)[0]
        print(f'Processing the classified_hdf_{sample_type}...')

        hdf_classified =  h5py.File(path, 'r')
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

            # update the meta
            tif_meta['transform'] = Affine(*trans)
        

        # write the array to tif, chunk by chunk
        with rasterio.open(f'{TIF_SAVE_PATH}/classification_{REGION}_{sample_type}.tif', 
                        'w',
                        **tif_meta) as dst:
            for chunk in tqdm(hdf_chunks):
                arr = hdf_classified['classification'][chunk].astype(np.int8)
                dst.write(arr, window=rasterio.windows.Window.from_slices(chunk[1], chunk[2]))


        # close the hdf file
        hdf_classified.close()

# overlay the classified tif files, get the pixels with value > {OVERLAY_THRESHOLD}
def overlay_classified_tif():

    # get the number of classification tifs
    num_classified_tifs = len(glob(f'{TIF_SAVE_PATH}/classification_{REGION}_*.tif'))

    # if the number of classified tifs is 1, then no need to overlay
    if num_classified_tifs == 1:
        print('Only one classified tif file, no need to overlay!\n')
        return
    
    else:  

        # report to console
        print(f'Overlaying the classiffication and filter pixels with value > {OVERLAY_THRESHOLD} \n')

        # get the classified tif files
        classified_tifs = glob(f'{TIF_SAVE_PATH}/classification_{REGION}_*.tif')
        classified_tifs = [tif for tif in classified_tifs if '0'  in tif]

        # open the first tif file
        with rasterio.open(classified_tifs[0]) as src:
            # get the meta data
            meta = src.meta
            # get the array shape
            array_shape = src.height, src.width
            # get the windows
            windows_rio = [window for _, window in src.block_windows()]

        # update the meta
        meta.update({'count': 1, 
                    'dtype': 'int8', 
                    'driver': 'GTiff',
                    'compress': 'lzw'})

        # create an empty array to store the overlayed result
        overlayed_arr = np.zeros(array_shape, dtype=np.int8)

        # save the empty array to disk
        with rasterio.open(f'{TIF_SAVE_PATH}/classification_{REGION}_overlayed.tif','w',
                            **meta) as dst:
            dst.write(overlayed_arr,1)



        # open the overlayed tif, and write sum_arry with window
        with rasterio.open(f'{TIF_SAVE_PATH}/classification_{REGION}_overlayed.tif',
                           'r+') as dst:

            # read each tif file by iterating the window from the meta
            for win in tqdm(windows_rio):

                arrs = []
                for tif in classified_tifs: 
                    with rasterio.open(tif) as src:
                        # read the tif file
                        arr = src.read(1, window=win)
                        # record the array
                        arrs.append(arr)

                # sum the arrays
                arrs_sum = np.array(arrs).sum(axis=0)

                # filter the array
                arrs_sum = np.where(arrs_sum >= OVERLAY_THRESHOLD, 1, 0)

                # write the overlayed result to disk
                dst.write(arrs_sum,1,window=win)

# calculate the accuracy of the overlayed result
def calculate_overlay_accuracy():

    # get the number of classification tifs
    num_classified_tifs = len(glob(f'{TIF_SAVE_PATH}/classification_{REGION}_*.tif'))

    # if the number of classified tifs is 1, then no need to overlay
    if num_classified_tifs == 1:
        print('Only one classified tif file, no need to overlay!\n')
        return
    
    else:
        
        print(f'Calculating the accuracy and save to {BASE_PATH}/accuracy_{REGION}.csv')

        # open the overlayed tif
        with rasterio.open(f'{TIF_SAVE_PATH}/classification_{REGION}_overlayed.tif') as src:
            # read the array
            arr = src.read(1)
            # get the transform
            trans = src.transform

        # read the test sample points
        test_pts = gpd.read_file(f'{SAMPLE_PTS_PATH}/y_test_{REGION}.shp')

        # convert the xy coordinate to col/row index
        test_pts[['col','row']] = test_pts['geometry']\
                                    .apply(lambda geo:~trans*(geo.x,geo.y)).tolist()
        test_pts[['col','row']] = test_pts[['col','row']].astype(int)

        # get the test sample values
        test_values = test_pts['Built'].values

        # pred arry
        pred_values = []
        for row,col in zip(test_pts['row'], test_pts['col']):
            try:
                pred_values.append(arr[row,col])
            except:
                pred_values.append(np.nan)

        # convert to np array
        pred_values = np.array(pred_values)
        
        # remove the nan
        test_values = test_values[~np.isnan(pred_values)]
        pred_values = pred_values[~np.isnan(pred_values)]

        # calculate the accuracy
        accuracy = classification_report(test_values,
                                        pred_values,
                                        target_names=['Non-Built', 'Built'],
                                        output_dict=True)
            
        # formatting the accuracy
        accuracy = pd.DataFrame(accuracy).T.reset_index()
        accuracy.columns = ['Indicator','precision','recall','f1-score','Sample Size']
        accuracy.insert(0, 'Non-Urban ratio', 'overlayed')

        # append the accuracy to {BASE_PATH}/accuracy_{REGION}.csv
        accuracy.to_csv(f'{BASE_PATH}/accuracy_{REGION}.csv', 
                        mode='a', 
                        index=False,
                        header=False)
