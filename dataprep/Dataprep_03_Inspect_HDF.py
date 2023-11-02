
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# set up working directory
if __name__ == '__main__':
    os.chdir('..')

from PARAMETER import  PATH_HDF

path = f"{PATH_HDF}/huadong_Sentinel_SAR_2020_2022.hdf"


dataset = h5py.File(path,'r')
arr = dataset[list(dataset.keys())[0]]




delta = 5000
row_start=np.random.randint(0,arr.shape[1]-delta)
col_start=np.random.randint(0,arr.shape[2]-delta)

tt = arr[:,row_start:row_start+delta,col_start:col_start+delta]
tt_t = np.transpose(tt[(2,1,0),:,:],(1,2,0))

plt.imshow(tt_t,vmin=0,vmax=80)









































