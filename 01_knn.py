# import libraries to perform K-Mean using cuML
import cudf
import cuml
import h5py
from glob import glob
from tqdm import tqdm
import numpy as np
from itertools import product


from cuml.cluster import KMeans


from tools import sample_from_hdf



# define the path to the dataset
path_hdf = '/mnt/d/HDF'
hdf_files = glob(path_hdf + '/*.hdf')


# sample from the dataset and create a 1-D sample vector
sample_arr = sample_from_hdf(hdf_files[1])

# fit the KNN model
n_neighbors = 10
knn = KMeans(n_neighbors=n_neighbors)
knn.fit(sample_arr)

# predict the labels
hdf = h5py.File(hdf_files[1], 'r')
hdf_arr = hdf[list(hdf.keys())[0]]

hdf_chunk_size = hdf_arr.chunks[-1]

# get the chunks
chunk_dilate = 3
chunk_dilate_size = int(hdf_chunk_size * chunk_dilate)

chunk_split_row = range(0, hdf_arr.shape[1] , chunk_dilate_size)
chunk_split_col = range(0, hdf_arr.shape[2] , chunk_dilate_size)

chunks = list(product(chunk_split_row, chunk_split_col))


with h5py.File('pred.hdf', 'w') as f:

    # create the dataset
    dataset = f.create_dataset('pred', 
                               shape= (1, *hdf_arr.shape[1:]), 
                               dtype='int8', 
                               chunks=(1, chunk_dilate_size, chunk_dilate_size),
                               compression=7,
                               fillvalue=0)
    
    for chunk in tqdm(chunks):
        # get the array from the chunk
        arr = hdf_arr[:,
                      chunk[0]:chunk[0] + chunk_dilate_size,
                      chunk[1]:chunk[1] + chunk_dilate_size] 
        
        arr_shape = arr.shape  # (C, H, W)

        # reshape the array to 1-D vector
        arr = np.swapaxes(arr, 0, 2).reshape(-1, arr.shape[0])

        # pred
        pred = knn.predict(arr.astype(np.float32))

        # reshape the pred to the original shape
        pred = pred.reshape(arr_shape[1], arr_shape[2])
        pred = np.expand_dims(pred, axis=0) # extra dimension for the channel

        # save the pred to the hdf file
        dataset[:,
                chunk[0]:chunk[0] + chunk_dilate_size,
                chunk[1]:chunk[1] + chunk_dilate_size] = pred

