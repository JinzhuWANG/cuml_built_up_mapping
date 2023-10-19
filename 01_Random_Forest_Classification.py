import numpy as np
import geopandas as gpd

from cuml.ensemble import RandomForestClassifier as cuRF
from tools import get_hdf_files 
from PARAMETER import SAMPLE_PTS_PATH, REGION, N_ESTIMATROS, MAX_DEPTH, SUBSET_PATH
from tools.model_pred import pred_hdf
from tools.dataProcessing import arr_to_TIFF, extract_img_val_to_sample

# get the hdf files
hdf_files = get_hdf_files()

# extract image values to sample points
extract_img_val_to_sample()

# read the sample values
sample_X = np.load(f'{SAMPLE_PTS_PATH}/sample_values_{REGION}.npy')
sample_y = gpd.read_file(f'{SAMPLE_PTS_PATH}/merge_pts_{REGION}.shp')['Built'].values


# initiate the RF classifier
model = cuRF( max_depth = MAX_DEPTH,
              n_estimators = N_ESTIMATROS,
              random_state  = 0 )

trained_RF = model.fit( sample_X, sample_y )


# apply the model to the hdf files, and then save the prediction to a hdf
print('Applying the model to the hdf files...')
pred_hdf(hdf_files, trained_RF)

# save classified hdf to tif
print('Saving the classified hdf to tif...')
arr_to_TIFF()

