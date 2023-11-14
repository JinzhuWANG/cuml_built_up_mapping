# Description: This file contains all the parameters used in the project



############################################
#            Data Preparation              #
############################################

# the unmixing dictionary between lucc and its code,
# here we use 'bare' to mask the prediction, meaning
# that only if a pixel has the highest probability of being
# bare land (among ['bare','vegetation', 'water']), then 
# it will be passed to the Random Forest model. Otherwise,
# it will be assigned with the value of 0.

UNMIXING_LU_ID = {'bare':0,
                  'vegetation':1,
                  'water':2} 

UNMIXING_LU = 'bare'




# define the expression to calculate the Normalized Index
INDICES_CAL_EXPRESSION = {
                            'NDWI':{ 'Landsat5':'(Band_4 - Band_2) / (Band_4 + Band_2)',
                                     'Landsat7':'(Band_4 - Band_5) / (Band_4 + Band_5)',
                                     'Landsat8':'(Band_5 - Band_6) / (Band_5 + Band_6)'
                                    }
                        }



# define the path to RAW_TIF files
# These tif files were downloaded from Google Earth Engine             # change this if you want to use your own data
# and will be converted to hdf files
TIF_PATH = '/mnt/d/TIF'


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
REGION = 'zhongnan'                                                     # change this to fit the region
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