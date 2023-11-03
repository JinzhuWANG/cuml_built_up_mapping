
import os
import pandas as pd
from glob import glob

from osgeo import gdal
import rasterio
from tqdm.auto import tqdm

# set up working directory
if __name__ == '__main__':
    os.chdir('..')

from dataprep.dataprep_tools import get_str_info
from PARAMETER import  TIF_PATH


# get all tif files
tifs = glob(f'{TIF_PATH}/*.tif')

# convert TIF_PATHs to a df, append name columns
files_df = pd.DataFrame({'path':tifs})
files_df[['region','type','year']] = files_df['path'].apply(lambda x:get_str_info(x)).tolist()

# append the TIF_PATHs of same {region|type|year} as a list
grouped_df = files_df.groupby(['region','type','year']).apply(lambda x:x['path'].tolist()).reset_index()



# save the tifs of of same {region|type|year} as a virtual raster
for _,row in tqdm(grouped_df.iterrows(), total=len(grouped_df)):
    
    vir = gdal.BuildVRT(f"{TIF_PATH}/{row['region']}_{row['type']}_{row['year']}.vrt", row[0])
    vir.FlushCache()



# save the transformation info
VRT_files = glob(f'{TIF_PATH}/*.vrt')

# convert VRT files into df
tif_info = []
for i in tqdm(VRT_files,total=len(VRT_files)):
    region,tif_type,year = get_str_info(i) 
    tif_meta = rasterio.open(i).meta
    tif_info.append({'region':region,'type':tif_type,'year':year,'trans':tif_meta})


# put all record into a df
tif_df = pd.DataFrame(tif_info)

# save tif_df to disk
if not os.path.exists(f"{TIF_PATH}/meta"):
    os.makedirs(f"{TIF_PATH}/meta")

tif_df.to_pickle(f'{TIF_PATH}/meta/tif_df.pkl')

































