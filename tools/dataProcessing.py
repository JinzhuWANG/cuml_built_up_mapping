import os
from glob import glob
import re
from functools import partial

import geopandas as gpd
import numpy as np
import pandas as pd

import rasterio
import h5py
from rasterio import Affine
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from numpy.linalg import lstsq




# setting up working directory
if __name__ == '__main__':
    os.chdir('..')


from PARAMETER import BASE_PATH, OVERLAY_THRESHOLD, SAMPLE_PTS_PATH, PATH_HDF, \
                      REGION, SUBSET_PATH, TIF_SAVE_PATH, INDICES_CAL_EXPRESSION, YEAR_RANGE

from tools.model_pred import get_bands_index
from tools import get_hdf_files, get_geo_meta, extract_val_to_pt
from dataprep.dataprep_tools import get_str_info

class compute_custom_indices_by_chunk():

    def __init__(self, img_path:str, 
                 compute_type:str, 
                 dtype:np.dtype=np.int8,
                 bands_count:int=1,
                 *args,
                 **kwargs):

        self.img_path = img_path
        self.compute_type = compute_type
        self.dtype = dtype
        self.bands_count = bands_count
        self.args = args
        self.kwargs = kwargs


    def __call__(self, compute_func:callable,):
        def wrapper():
            # get the year of the landsat image
            region,_,year_range = get_str_info(self.img_path)
            year = int(year_range[-4:])

            # create the name for output hdf file
            out_name = f"{PATH_HDF}/{region}_{self.compute_type}_{year_range}.hdf"

            # get the shape, blocks of the landsat image
            landsat_hdf = h5py.File(self.img_path, 'r')
            landsat_arr = landsat_hdf[list(landsat_hdf.keys())[0]]
            ds_shape = landsat_arr.shape
            block_size = landsat_arr.chunks[1]

            # create an empty hdf file for holding the computed indices
            with h5py.File(out_name,mode='w') as hdf_file:
                # Create a dataset and save the NumPy array to it
                hdf_file.create_dataset(self.compute_type, 
                                        shape=(self.bands_count,ds_shape[1],ds_shape[2]),
                                        dtype=self.dtype,
                                        fillvalue=np.nan, 
                                        compression="gzip", 
                                        compression_opts=9,
                                        chunks=(self.bands_count,block_size,block_size))

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
                index_arr = compute_func(arr_slice, *self.args, **self.kwargs)
                # save the index to hdf
                with h5py.File(out_name,mode='r+') as hdf_file:
                    hdf_file[self.compute_type][slice_] = index_arr

        return wrapper



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
    if int(year)<=2010:
        landsat_type = 'Landsat5'
    elif int(year)<=2013:
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




def get_custom_indices():
    
    # loop through each index
    Landsat_img_path = f"{PATH_HDF}/{REGION}_Landsat_cloud_free_{YEAR_RANGE}.hdf"
    indices = INDICES_CAL_EXPRESSION.keys()

    for index in indices:
        # check if the index already exist
        year_range = get_str_info(Landsat_img_path)[-1]
        index_path = f"{PATH_HDF}/{REGION}_{index}_{year_range}.hdf"

        if os.path.exists(index_path):
            print(f'{index_path} already exists!\n')
        else:
            # compute the index
            print(f'Computing {index} for {os.path.basename(Landsat_img_path)}...')

            # Decorate the compute_indices function
            compute_indices_decorated = compute_custom_indices_by_chunk(
                                    img_path=Landsat_img_path,
                                    compute_type=index,
                                    dtype=np.int8)(partial(compute_indices,
                                                           year=int(year_range[-4:]),
                                                           index=index))
            
            # Invoke the docorated function
            compute_indices_decorated()



def spectral_unmixing(arr:np.ndarray,end_numbers:np.ndarray):
    ''' spectral unmixing for the arr
    INPUT:  arr: array with shape (C, H, W)
            end_numbers: array with shape (C, N)
    OUTPUT: unmixing_arr: array with shape (N, H, W)
    '''
    # get the shape of the array
    C, H, W = arr.shape
    # get the number of endmembers
    N = end_numbers.shape[0] # in_shapge -> (N, C)

    # reshape the array
    arr = arr.reshape(C,H*W)

    # solve the linear equation
    unmixing_arr_col, residuals, rank, s = lstsq(end_numbers.T, arr)
    unmixing_arr_col = unmixing_arr_col.reshape(N,H,W)
    # apply softmax
    unmixing_arr_col = np.exp(unmixing_arr_col)/np.sum(np.exp(unmixing_arr_col),axis=0)
    # change the dtype
    unmixing_arr_col = (unmixing_arr_col*127).astype(np.uint8)

    return unmixing_arr_col

    

def get_spectral_unmixing(unmixing_from:str='Landsat_cloud_free'):
    # check if the spectral unmixing already exist
    year_range = get_str_info(f"{PATH_HDF}/{REGION}_Landsat_cloud_free_{YEAR_RANGE}.hdf")[-1]
    spectral_unmixing_path = f"{PATH_HDF}/{REGION}_Spectral_Unmixing_{year_range}.hdf"

    if os.path.exists(spectral_unmixing_path):
        print(f'{spectral_unmixing_path} already exists!\n')
    else:
        # get the end_numbers
        end_numbers = np.load(f'{SAMPLE_PTS_PATH}/sample_values_{REGION}_unmixing.npy')

        # get the index of the bands used for unmixing
        unmixing_sample_colunms = get_bands_index()
        if not unmixing_from in unmixing_sample_colunms['band_type'].tolist():
            raise ValueError(f'{unmixing_from} should be one of {unmixing_sample_colunms["band_type"].tolist()}!')
        band_idx = unmixing_sample_colunms.query(f'band_type=="{unmixing_from}"')['indices'].tolist()[0]

        end_numbers = end_numbers[:,band_idx]

        # compute the spectral unmixing
        spectral_unmixing_decorated = compute_custom_indices_by_chunk(
                                        img_path=f"{PATH_HDF}/{REGION}_Landsat_cloud_free_{YEAR_RANGE}.hdf",
                                        compute_type='Spectral_Unmixing',
                                        dtype=np.uint8,
                                        bands_count=end_numbers.shape[0])(partial(spectral_unmixing,
                                                                end_numbers=end_numbers))
        # invoke the decorated function
        spectral_unmixing_decorated()




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
        unmixing_sample_path = f'{SAMPLE_PTS_PATH}/Spectral_unmixing_samples_{REGION}.shp'

        for sample_type,path in zip(['sample_classification', 'sample_unmixing'],
                                    [sample_path, unmixing_sample_path]):
                                    
            # check if the path exists
            if not os.path.exists(path):
                raise ValueError(f'{path} does not exist!')    
            # extract the image values to sample points
            img_val_to_point(sample_type,path)
        
def img_val_to_point(sample_type,sample_path:str):
    sample_pts = gpd.read_file(sample_path)
    # explode the sample points
    sample_pts = sample_pts.explode().reset_index(drop=True)

    # get the hdf files
    hdf_files = get_hdf_files()

    # convert the xy coordinate to col/row index
    geo_trans = get_geo_meta()['transform']
    sample_pts[['col','row']] = sample_pts['geometry'].apply(lambda geo:~geo_trans*(geo.x,geo.y)).tolist()
    sample_pts[['col','row']] = sample_pts[['col','row']].astype(int)

    # get the sample values from hdf files
    sample_values = np.hstack([extract_val_to_pt(sample_pts,f) for f in hdf_files])

    # save the sample values
    if sample_type == 'sample_classification':
        np.save(f'{SAMPLE_PTS_PATH}/sample_values_{REGION}.npy',sample_values)
    else:
        sample_num = sample_values.shape[1]
        sample_df = pd.concat([sample_pts,pd.DataFrame(sample_values)],1)
        sample_values = sample_df.groupby(['unmixing_t']).mean()[range(sample_num)].values
        np.save(f'{SAMPLE_PTS_PATH}/sample_values_{REGION}_unmixing.npy',sample_values)



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
