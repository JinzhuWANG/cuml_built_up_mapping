import os
import re
import pandas as pd
from glob import glob

from tqdm.auto import tqdm
import pandas as pd


# set up working directory
if __name__ == '__main__':
    os.chdir('..')

from dataprep.dataprep_tools import vrt2hdf, get_str_info
from PARAMETER import  PATH_HDF,TIF_PATH


# get all VRT files
VRT_files = glob(f'{TIF_PATH}/*.vrt')
VRT_bases = ['_'.join(get_str_info(i)) for i in VRT_files]


# exclude the VRT files that alread processed
completed_convertions = f'{TIF_PATH}/meta/vrt2hdf_complete.csv'

# create a csv file to record the completed convertions
if not os.path.exists(completed_convertions):
    hdf_exist = []
    with open(completed_convertions,'w') as f: 
         f.write('Completed\n')  
else:
    hdf_exist = pd.read_csv(completed_convertions)['Completed'].tolist()
    hdf_exist = ['_'.join(get_str_info(i)) for i in hdf_exist]

# get the VRT files that has not been processed
VRT_to_process = set(VRT_bases) - set(hdf_exist)

# sort files from latest to oldest
reg_year = re.compile(r'\d{4}_\d{4}')
VRT_files = [f"{TIF_PATH}/{i}.vrt" for i in VRT_to_process]
VRT_files = sorted(VRT_files,key=lambda x:reg_year.findall(x)[0],reverse=True)


# convert VRT to HDF
for vrt in VRT_files:
    successed = vrt2hdf(vrt,PATH_HDF)


