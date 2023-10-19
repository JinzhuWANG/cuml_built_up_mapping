# Description: This file contains all the parameters used in the project

############################################
#               Data PATH                  #
############################################
PATH_HDF = '/mnt/d/HDF'
TIF_SAVE_PATH = '/mnt/d/classification'
GEO_META_PATH = '/mnt/d/tif_meta'
SAMPLE_PTS_PATH = '/mnt/d/sample_points'



############################################
#           REGEION & NAME                 #
############################################
REGION = 'xinan'
YEAR_RANGE = '2020_2022'

# the ShpFile to subset the input HDF data,
# so that we can get partial result fast

SUBSET_PATH = '/mnt/d/subset_SHP/subset_shp.shp'
# SUBSET_PATH = 'None' # this will use the whole HDF file

############################################
#           MODEL PARAMETERS               #
############################################
N_ESTIMATROS = 25
MAX_DEPTH = 10

# the hdf file are stored as chunks of 128*128
# if the chunk_dilate = 2, then the input array 
# to Random Forest model will be 256*256
CHUNK_DILATE = 2