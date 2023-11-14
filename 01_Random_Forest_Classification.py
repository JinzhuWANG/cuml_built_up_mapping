

from tools import plot_sample_pts
from tools.model_pred import get_bands_accuracy, get_models, mask_classification_arr, pred_hdf
from tools.training_spliting import increase_nonurban_pts, train_test_split_sample
from tools.dataProcessing import classified_HDF_to_TIFF, calculate_overlay_accuracy,\
                                 extract_img_val_to_sample, overlay_classified_tif,\
                                 get_custom_indices, get_spectral_unmixing


############################################
#             Data preparing               #
############################################

# 1) compute the Normalized Index for all the Landsat images
# NOTE the values are rescaled to -127 ~ 127 as int8

# !!! to_tif=True will save the result to tif in the {TIF_SAVE_PATH}
get_custom_indices(to_tif=False)                 



############################################
#           Sample preparing               #
############################################

# 1) extract image values to sample points
extract_img_val_to_sample(force_resample=True)
plot_sample_pts()

# 2) split sample into train and test subset
train_test_split_sample()
# 3) gradually increse the size of non-urban built points
increase_nonurban_pts()

# 4) get the classification accuracy by only use each band as input
get_bands_accuracy()

# 5) get the spectral unmixing result
# !!! to_tif=True will save the result to tif in the {TIF_SAVE_PATH}
get_spectral_unmixing(to_tif=True)              




############################################
#             Model training               #
############################################

# the keys are the non-urban ratio used in trainning the model
# i.e., if there are 5k non-urban built-points,  ratio = 0.1 means
#       we will use 500 non-urban built-points to train the model

# Also, each model will be evaluated by the test subset, and the 
# accuracy will be saved to the root directory
trained_models = get_models()



############################################
#        Perform classification            #
############################################

# apply the model to the hdf files, and then save the prediction to a hdf
pred_hdf(trained_models,force_use_nonurban_subset=False)
mask_classification_arr()

# save classified hdf to tif
classified_HDF_to_TIFF()



############################################
#        Ovelay classifications            #
############################################

# Overlay all the classified tif files (num=10) to one tif file
# and finding the pixles with value > 5 as the "final urban built-up area"
overlay_classified_tif()
calculate_overlay_accuracy()