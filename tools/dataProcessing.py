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

from PARAMETER import BASE_PATH, OVERLAY_THRESHOLD, SAMPLE_PTS_PATH, \
                      REGION, SUBSET_PATH, TIF_SAVE_PATH
from tools import get_hdf_files, get_geo_meta, extract_val_to_pt


def extract_img_val_to_sample():
    ''' extract the image values to sample points
    INPUT:  None, but read the sample points from {SAMPLE_PTS_PATH}
    OUTPUT: None, but save the sample values to {SAMPLE_PTS_PATH}
    '''
    # check if the sample values already exist
    if os.path.exists(f'{SAMPLE_PTS_PATH}/sample_values_{REGION}.npy'):
        print('The sample values already exist!\n')

    else:

        print('Extracting image values to sample points...\n')

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

        # create an empty array to store the overlayed result
        overlayed_arr = np.zeros(array_shape, dtype=np.int8)

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
        
            # get the overlayed result
            overlayed_arr[win.toslices()] = arrs_sum

        # save the overlayed result to tif
        meta.update({'count': 1, 
                    'dtype': 'int8', 
                    'driver': 'GTiff',
                    'compress': 'lzw'})
        
        with rasterio.open(f'{TIF_SAVE_PATH}/classification_{REGION}_overlayed.tif', 
                            'w',
                            **meta) as dst:
            dst.write(overlayed_arr,1)

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
