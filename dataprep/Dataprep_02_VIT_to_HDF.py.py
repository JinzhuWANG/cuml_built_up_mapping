import os
import re
import pandas as pd
from glob import glob

from dfply import *
from tqdm.auto import tqdm
import pandas as pd

from functools import partial

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

if not os.path.exists(completed_convertions):
    with open(completed_convertions,'w'): pass
    hdf_exist = []
else:
    hdf_exist = (pd.read_csv(completed_convertions, header=None)
                >> mask(X[0] != 'None'))[0].tolist()
    hdf_exist = ['_'.join(get_str_info(i)) for i in hdf_exist]

# get the VRT files that has not been processed
VRT_to_process = set(VRT_bases) - set(hdf_exist)

# sort files from latest to oldest
reg_year = re.compile(r'\d{4}_\d{4}')
VRT_files = [f"{TIF_PATH}/{i}.vrt" for i in VRT_to_process]
VRT_files = sorted(VRT_files,key=lambda x:reg_year.findall(x)[0],reverse=True)





def retry_convertion(func, max_retries = 3):
    def wraper(vrt,PATH_HDF):
        attempts = 0
        while attempts < max_retries:
            try:
                # convert vrt to hdf, the function return nothing
                out = func(vrt,PATH_HDF)
                
                # write the sccessefuly transfered files        
                with open('./vrt2hdf_complete.csv','a',encoding='utf-8') as f:
                    f.write(f"{vrt}\n")   
                return out
            
            except:
                attempts = attempts + 1
                print(f"Convert of {vrt} failed! Retrying {attempts}/{max_retries}")
                
        # if passed the max_retries, then write the failed files
        with open('./vrt2hdf_incomplete.csv','a',encoding='utf-8') as f:
                    f.write(f"{vrt}\n")
                
    return wraper      


# apply the decorator
vrt2hdf = retry_convertion(vrt2hdf)


for vrt in tqdm(VRT_files,total=len(VRT_files)):
    successed = vrt2hdf(vrt,PATH_HDF)












































