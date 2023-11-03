import os
import h5py
from itertools import product
from tqdm.auto import tqdm
from glob import glob
import re

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio import Affine

from cuml.ensemble import RandomForestClassifier as cuRF
from sklearn.metrics import classification_report

# set upworking directory
if __name__ == '__main__':
    os.chdir('..')

from tools import get_geo_meta, get_hdf_files
from dataprep.dataprep_tools import get_str_info
from PARAMETER import BASE_PATH, CHUNK_DILATE, MAX_DEPTH, N_ESTIMATROS, SAMPLE_PTS_PATH, SUBSET,\
                      TIF_SAVE_PATH, REGION, SUBSET_PATH

import warnings
warnings.filterwarnings("ignore")

def get_models():
    ''' get the trained models
    INPUT:  None
    OUTPUT: trained models'''
    # import the sample points
    sample_paths = glob(f'{SAMPLE_PTS_PATH}/sample_pts_{REGION}*.csv')

    print(f'Training the models using sample_pts_{REGION}_X.csv ...')
    print(f'The accuracy report will be saved to {BASE_PATH}/accuracy_{REGION}.csv \n')

    # read the sample points, 
    #   sample_X is the image values (without the Built columns), 
    #   sample_y is the built type (only the Built columns)
    accuracies = []
    models = {}

    # loop through the sample paths
    for path in sample_paths:
        # get the built_nonurban ratio from path
        sample_type = re.compile(rf'{REGION}_(.*).csv').findall(path)[0]

        # get the X and y
        sample_X = pd.read_csv(path).drop(columns=['Built']).values
        sample_y = pd.read_csv(path)['Built'].values

        # train model with the sample points
        model = cuRF( max_depth = MAX_DEPTH,
                      n_estimators = N_ESTIMATROS )

        trained_RF = model.fit( sample_X, sample_y )

        # append the model to the dict
        models[sample_type] = trained_RF
        

        # report the accuracy using the test data
        test_X = np.load(f'{SAMPLE_PTS_PATH}/X_test_{REGION}.npy')
        test_y = gpd.read_file(f'{SAMPLE_PTS_PATH}/y_test_{REGION}.shp')['Built'].values
        accuracy = classification_report(test_y,
                                        trained_RF.predict(test_X),
                                        target_names=['Non-Built', 'Built'],
                                        output_dict=True)
        
        # formatting the accuracy
        accuracy = pd.DataFrame(accuracy).T.reset_index()
        accuracy.columns = ['Indicator','precision','recall','f1-score','Sample Size']
        accuracy.insert(0, 'Sample type', sample_type)

        # append the accuracy to the list
        accuracies.append(accuracy)

    # concat the accuracy and save to disk
    accuracies = pd.concat(accuracies)
    accuracies.to_csv(f'{BASE_PATH}/accuracy_{REGION}.csv', index=False)

    # return models
    return models


# function to calculate the start/end row/col index with a shp file
def subset_bounds(shp:str=SUBSET_PATH):
    ''' get the start/end row/col index with a shp file
    INPUT:  shp: path to the shp file
    OUTPUT: (start_row, end_row), (start_col, end_col)
    '''
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
    ''' 
    Get the chunks from the hdf file.

    Args:
    - hdf_path (str): path to the hdf file
    - subset (str): path to the subset file (default: SUBSET_PATH)

    Returns:
    - tuple: a tuple containing:
        - hdf_arr_shape (tuple): shape of the hdf array
        - chunk_dilate_size (int): the input array size to feed to the model
        - chunks (list): list of tuples of chunk indices
    '''
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
def pred_hdf(models, force_use_nonurban_subset=False):
    '''Loop through hdf files and make prediction using trained models.

    Args:
        models (dict): A dictionary of trained models of {sample_type: model}.
        force_use_nonurban_subset (bool, optional): Whether to force the use of non-urban subset for classification. Defaults to False.

    Returns:
        None, but saves the prediction to a hdf file.
    '''

    # remove previously saved classification files if to classify the whole dataset,
    if not SUBSET:
        print(f'Removing previously saved classification_{REGION} files...')
        for path in glob(f'{TIF_SAVE_PATH}/classification_{REGION}*'):
            os.remove(path)

    # check the sample_type of the highest accuracy,
    # if it comes from the whole training samples, then perform the classification just use the whole dataset
    # otherwise, perform the classification using the nonurban_increase_sample subset
    accuracy_df = pd.read_csv(f'{BASE_PATH}/accuracy_{REGION}.csv')
    highest_acc = accuracy_df[accuracy_df['Indicator'] == 'Built']\
                    .sort_values(by='f1-score', ascending=False).iloc[0]
    
    # report the highest accuracy
    print(f'''The higest accuracy is {highest_acc["f1-score"]:.2%} comming from {highest_acc["Sample type"]} subset\n''')
    
    # perform the classification
    if not force_use_nonurban_subset:
        classify_hdf(highest_acc["Sample type"], models[highest_acc["Sample type"]])
    else:
        print('''=======================================================
              The classification will use [10 non-urban built-samples]
                                    and [1 whole traning samples]
              =======================================================\n''')
        for sample_type, model in models.items():
            print(f'Classification using the model trained with {sample_type} non-urban subset...')
            classify_hdf(sample_type, model)
            


# function to make prediction given a trained model
def classify_hdf(sample_type,model):
    # get the hdf files
    hdf_paths = get_hdf_files()

    # get the array sizes and chunks
    hdf_arr_shape, chunk_dilate_size, chunks = get_hdf_chunks(list(hdf_paths)[0])

    # create an empty hdf to store the prediction
    with h5py.File(f'{TIF_SAVE_PATH}/classification_{REGION}_{sample_type}.hdf', 'w') as f:

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
        with h5py.File(f'{TIF_SAVE_PATH}/classification_{REGION}_{sample_type}.hdf', 'r+') as ds:
            ds_arr = ds[list(ds.keys())[0]]
            # save the pred to the hdf file
            ds_arr[:,
                row:row + chunk_dilate_size,
                col:col + chunk_dilate_size] = pred



def corespond_index(row_col_from):
    ''' get the coresponding index of the xy coordinates
    INPUT:  row_col_from:   tuple of row/col index from original coordinating system
            geo_trans_to:   geo_transformation of the target array
    OUTPUT: tuple of row/col index from the target coordinating system'''


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


def get_bands_index():
    """
    Returns a dataframe containing the index for each band type in the HDF files.
    The dataframe has two columns: 'band_type' and 'indices'.
    'band_type' is a string representing the type of band (e.g. 'Fourier', 'NDWI', ...).
    'indices' is a list of integers representing the index for each band type.
    """

    hdfs = get_hdf_files()

    bands_info = []
    for hdf in hdfs:
        with h5py.File(hdf, 'r') as ds:
            hdf_arr = ds[list(ds.keys())[0]]
            band_type = get_str_info(hdf)[1]
            band_count = hdf_arr.shape[0]
        
        bands_info.append({'band_type':band_type, 
                           'band_count':band_count})
    # put bands_info into a dataframe
    bands_info = pd.DataFrame(bands_info)

    # get the index for each band type
    bands_info['cum_sum'] = bands_info['band_count'].cumsum()
    bands_info['start_index'] = bands_info['cum_sum'] - bands_info['band_count']
    bands_info['end_index'] = bands_info['cum_sum'] 
    bands_info['indices'] = bands_info.apply(lambda x: list(range(x['start_index'], x['end_index'])), axis=1)

    return bands_info[['band_type','indices']]



def split_sample_to_bands():
    """
    Splits the sample points into bands and returns a DataFrame 
    with the band type and sample array for each band.

    Returns:
    pandas.DataFrame: A DataFrame with the band type and sample array for each band.
    """
    # read the sample points
    sample = pd.read_csv(f'{SAMPLE_PTS_PATH}/sample_pts_{REGION}_ALL_Train_Sample.csv')

    # get the bands
    bands = get_bands_index()

    # check if all bands are in the sample
    assert (bands['indices'].apply(len).sum() == len(sample.columns) -1), 'Not all bands are in the sample'

    bands['sample_arr'] = bands.apply(lambda x: sample.iloc[:,x['indices']].values, axis=1)

    return bands[['band_type','indices','sample_arr']]


def get_bands_accuracy():

    # get bands and its coreponding sample array
    bands = split_sample_to_bands()

    # train model for each band
    train_y = pd.read_csv(f'{SAMPLE_PTS_PATH}/sample_pts_{REGION}_ALL_Train_Sample.csv')
    train_y = train_y[train_y.columns[-1]].values

    bands['model'] = bands.apply(lambda x: cuRF( max_depth = MAX_DEPTH,n_estimators = N_ESTIMATROS )\
                                            .fit(x['sample_arr'], train_y), axis=1)

    # get the accuracy for each band
    test_X = np.load(f'{SAMPLE_PTS_PATH}/X_test_{REGION}.npy')
    test_y = gpd.read_file(f'{SAMPLE_PTS_PATH}/y_test_{REGION}.shp')['Built'].values

    # get the accuracy for each band
    accuracies = []
    for _, row in bands.iterrows():
        pred = row['model'].predict(test_X[:,row['indices']])
        accuracy = classification_report(test_y, 
                                        pred, 
                                        target_names=['Non-Built', 'Built'], 
                                        output_dict=True)

        # formatting the accuracy
        accuracy = pd.DataFrame(accuracy).T.reset_index()
        
        accuracy.columns = ['Indicator','precision','recall','f1-score','Sample Size']
        accuracy.insert(0, 'Band type', row['band_type'])
        accuracies.append(accuracy)

    # format and save to disk
    accurices = pd.concat(accuracies)
    accurices.to_csv(f'{BASE_PATH}/accuracy_{REGION}_bands.csv', index=False)