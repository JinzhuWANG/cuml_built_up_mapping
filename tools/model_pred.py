import os
import h5py
from itertools import product
from tqdm.auto import tqdm
import numpy as np
import geopandas as gpd
from rasterio import Affine

# set upworking directory
if __name__ == '__main__':
    os.chdir('..')

from PARAMETER import CHUNK_DILATE, TIF_SAVE_PATH, REGION, SUBSET_PATH
from tools import get_geo_meta

# function to calculate the start/end row/col index with a shp file
def subset_bounds(shp:str=SUBSET_PATH):
    # get the transform
    geo_trans = get_geo_meta()['transform']
    # get the xy coordinates of the bounds of the shp
    bounds = gpd.read_file(shp).bounds
    # get the top left and bottom right coordinates
    top_left = ~geo_trans*(bounds['minx'].values[0], bounds['maxy'].values[0])
    bottom_right = ~geo_trans*(bounds['maxx'].values[0], bounds['miny'].values[0])

    # get the start/end row/col index
    start_row = int(top_left[1])
    end_row = int(bottom_right[1])
    start_col = int(top_left[0])
    end_col = int(bottom_right[0])
    
    return (start_row, end_row), (start_col, end_col)




def get_hdf_chunks(hdf_path,subset=SUBSET_PATH):
    ''' get the chunks from the hdf file
    INPUT:  hdf_path: path to the hdf file
    OUTPUT: hdf_arr_shape: shape of the hdf array,
            chunk_dilate_size: the input array size to feed to the model,
            chunks: list of tuples of chunk indices'''
    # get the chunks
    with  h5py.File(hdf_path, 'r') as hdf_ds:
        hdf_arr = hdf_ds[list(hdf_ds.keys())[0]]
        hdf_arr_shape = hdf_arr.shape

        # get the chunks
        hdf_chunk_size = hdf_arr.chunks[-1]
        chunk_dilate_size = int(hdf_chunk_size * CHUNK_DILATE)

        if os.path.exists(subset):
            # get the bounds of the subset
            (start_row, end_row), (start_col, end_col) = subset_bounds()

            # make sure the end row/col index is dividable by chunk_dilate_size
            end_row = end_row - (end_row - start_row) % chunk_dilate_size + chunk_dilate_size
            end_col = end_col - (end_col - start_col) % chunk_dilate_size + chunk_dilate_size 
               
        else:
            start_row, end_row = 0, hdf_arr_shape[1]
            start_col, end_col = 0, hdf_arr_shape[2]

        # make sure the start/end row/col index is within the hdf array
        start_row = max(0, start_row)
        start_row = min(hdf_arr_shape[1], start_row)
        end_row = max(0, end_row)
        end_row = min(hdf_arr_shape[1], end_row)

        start_col = max(0, start_col)
        start_col = min(hdf_arr_shape[2], start_col)
        end_col = max(0, end_col)
        end_col = min(hdf_arr_shape[2], end_col)

        

        # get the chunks
        chunk_split_row = list(range(start_row, end_row, chunk_dilate_size))
        chunk_split_col = list(range(start_col, end_col, chunk_dilate_size))

        chunks = list(product(chunk_split_row, chunk_split_col))

    return ((hdf_arr_shape[0],(end_row-start_row),(end_col-start_col)), 
            chunk_dilate_size, 
            chunks)



# loop through hdf files and make prediction on each chunk
def pred_hdf(hdf_paths, model):
    ''' make prediction on the hdf file
    INPUT: hdf_path: path to the hdf file
           model: trained model
    OUTPUT: pred: prediction array'''

    # get the array sizes and chunks
    hdf_arr_shape, chunk_dilate_size, chunks = get_hdf_chunks(list(hdf_paths)[0])

    # create an empty hdf to store the prediction
    with h5py.File(f'{TIF_SAVE_PATH}/classification_{REGION}.hdf', 'w') as f:

        # create the dataset
        dataset = f.create_dataset('classification', 
                                shape= (1, *hdf_arr_shape[1:]), 
                                dtype='int8', 
                                chunks=(1, chunk_dilate_size, chunk_dilate_size),
                                compression=7,
                                fillvalue=0)

    # open all the hdf file
    hdf_arrs = []
    for hdf_path in hdf_paths:
        hdf_ds = h5py.File(hdf_path, 'r')
        hdf_arr = hdf_ds[list(hdf_ds.keys())[0]]

        hdf_arrs.append(hdf_arr)

    for chunk in tqdm(chunks):
        # get the array from the chunk (C, H, W)
        input_arr = []
        for hdf_arr in hdf_arrs:
            arr_slice = hdf_arr[:,
                                chunk[0]:chunk[0] + chunk_dilate_size,
                                chunk[1]:chunk[1] + chunk_dilate_size]
            # get the shape of the array_slice
            arr_slice_shape = arr_slice.shape
            
            # reshape the array into (H*W, C)
            arr_slice = np.moveaxis(arr_slice, 0, -1) # (H, W, C)
            arr_slice = arr_slice.reshape(-1, arr_slice.shape[-1]) # (H*W, C)
            input_arr.append(arr_slice)

        # reshape the input array into correct shape
        input_arr = np.hstack(input_arr) # (n*H*W, C)
  
        # make prediction
        pred = model.predict(input_arr.astype(np.float32)) # (H*W, 1)

        # reshape pred into (H, W)
        pred = pred.reshape(arr_slice_shape[-2], arr_slice_shape[-1])

        # get the row/col index of the 
        if os.path.exists(SUBSET_PATH):
            row,col = corespond_index(chunk)
        else:
            row,col = chunk[0], chunk[1]

        # save to hdf
        with h5py.File(f'{TIF_SAVE_PATH}/classification_{REGION}.hdf', 'r+') as ds:
            ds_arr = ds[list(ds.keys())[0]]
            # save the pred to the hdf file
            ds_arr[:,
                   row:row + chunk_dilate_size,
                   col:col + chunk_dilate_size] = pred



def corespond_index(row_col_from):
    ''' get the coresponding index of the xy coordinates
    INPUT: row_col_from: tuple of row/col index
    geo_trans_to: geo_transformation of the target array'''


    # get the geo_trans_from
    geo_trans_from = get_geo_meta()['transform']

    # get the xy coordinates from the row/col index
    yx_coord_from = geo_trans_from*(row_col_from[1],row_col_from[0])

    # get the geo_trans_to
    if os.path.exists(SUBSET_PATH):
        # get the xy coordinates of the bounds of the shp
        bounds = gpd.read_file(SUBSET_PATH).bounds
        trans = list(geo_trans_from)
        trans[2] = bounds['minx'].values[0]
        trans[5] = bounds['maxy'].values[0]
        geo_trans_to = Affine(*trans)

    # get the row/col index from the xy coordinates
    row_col_to = ~geo_trans_to*yx_coord_from

    return (int(row_col_to[1]), int(row_col_to[0]))