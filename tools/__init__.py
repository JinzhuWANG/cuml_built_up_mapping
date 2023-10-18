import h5py
import numpy as np

# import parameters
from PARAMETER import SAMPLE_PROP

# read the hdf dataset, get the total chunks
def sample_from_hdf(path):
    ds = h5py.File(path, 'r')
    ds_arr = ds[list(ds.keys())[0]]

    # get the sample indices
    chunks = [chunck for chunck in ds_arr.iter_chunks()]

    # random choice of the sample indices from the total chunks
    sample_indices = np.random.choice(len(chunks), 
                                    int(len(chunks)*SAMPLE_PROP), 
                                    replace=False)
    sample_chunks = [chunks[i] for i in sample_indices]

    # get the array from sample indices
    sample_arrs = []
    for chunk in sample_chunks:
        arr = ds_arr[chunk]
        if (arr.shape != ds_arr.chunks) or (arr.sum() == 0):
            continue
        else:
            sample_arrs.append(arr)

    # concatenate the sample arrays, flatten the array, and close the dataset
    sample_arr = np.array(sample_arrs) # (n_sample, C, H, W)
    sample_arr = np.swapaxes(sample_arr, 1, 3).reshape(-1, sample_arr.shape[1])
    ds.close()

    return sample_arr.astype(np.float32)
