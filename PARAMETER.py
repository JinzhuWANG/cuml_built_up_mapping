# Description: This file contains all the parameters used in the project

############################################
#               Data PATH                  #
############################################

# Note the backslash, !!!DO NOT use windows style path
BASE_PATH = '/mnt/e'                                                   # change this to the base folder 

PATH_HDF = f'{BASE_PATH}/HDF'
TIF_SAVE_PATH = f'{BASE_PATH}/classification'
GEO_META_PATH = f'{BASE_PATH}/tif_meta'
SAMPLE_PTS_PATH = f'{BASE_PATH}/sample_points'
REFERENCE_DATA_PATH = f'{BASE_PATH}/reference_data'



############################################
#           REGEION & NAME                 #
############################################
# one of 'xinan,xibei,zhongnan,dongbei,huabei,huadong'
REGION = 'xinan'                                                     # change this to fit the region
YEAR_RANGE = '2020_2022'

# the ShpFile to subset the input HDF data,
# so that we can get partial result fast
SUBSET = True


if SUBSET == True:
    
    SUBSET_PATH = f'{BASE_PATH}/subset_SHP/subset_shp_{REGION}.shp'        
else:
    SUBSET_PATH = 'None'    


############################################
#           MODEL PARAMETERS               #
############################################
N_ESTIMATROS = 25
MAX_DEPTH = 10
TRAIN_SAMPLE_RATIO = 0.8
OVERLAY_THRESHOLD = 7

# the hdf file are stored as chunks of 128*128
# if the chunk_dilate = 2, then the input array 
# to Random Forest model will be 256*256
CHUNK_DILATE = 2