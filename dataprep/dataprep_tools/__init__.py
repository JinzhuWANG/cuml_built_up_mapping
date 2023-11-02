import os
import re
import rasterio
import h5py
import numpy as np
from tqdm.auto import tqdm


# function to extract names from tif_path
def get_str_info(path):
    '''Function to extract [Region,Type,Year] from a tif path
    '''
    # get the base name
    base = os.path.basename(path)
    
    # get the registrations
    regs = [re.compile(r'^(\w+?)_'), 
            re.compile(r'_(\w+?)_\d{4}'),
            re.compile(r'\d{4}_\d{4}')]
    
    # find the matchs
    matchs = [reg.findall(base)[0] for reg in regs]
    
    return matchs

# function to save an vrt to hdf
def vrt2hdf(vrt,save_path):
    '''
    Function to save vrt raster file to hdf
    
    Argument:
        vrt: the {path} to vrt file
        save_path: the directory to save hdf
     
    Return:
        vrt: same as the input vrt, which can be used as a flag to show the vrt2hdf
             transformation is successed'''

    vrt_name = os.path.basename(vrt)

    # open the dataset
    ds = rasterio.open(vrt)

    # get the shape the vrt
    ds_shape = ds.shape

    # get the block size
    block_size = ds.block_shapes[0][0]

    # compute the total number of blocks
    block_num = (int(ds_shape[0]/block_size) + 1) * (int(ds_shape[1]/block_size) + 1)

    # define the path/name for save
    hdf_name = vrt_name.replace('vrt','hdf')
    out_hdf_path = os.path.abspath(f'{save_path}/{hdf_name}')

    # create an hdf file for writing
    with h5py.File(out_hdf_path,mode='w') as hdf_file:
        # Create a dataset and save the NumPy array to it
        hdf_file.create_dataset(vrt_name, 
                                shape=(ds.count,*ds_shape),
                                dtype=np.int16, 
                                fillvalue=0, 
                                compression='lzf', 
                                # compression_opts=7,
                                chunks=(ds.count,block_size,block_size))

        # loop throught each block to save the block array to hdf
        for _,window in tqdm(ds.block_windows(),total=block_num,leave=True,desc=f'Saving {vrt_name} to HDF \n'):
            arr = ds.read(window=window)

            # write the block arry to hdf
            hdf_file[vrt_name][:,
                     window.row_off:window.row_off+block_size,
                     window.col_off:window.col_off+block_size] = arr
                        
    return vrt