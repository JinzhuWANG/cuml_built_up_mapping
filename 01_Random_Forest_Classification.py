from PARAMETER import REGION, SUBSET_PATH
from tools.model_pred import get_models, pred_hdf
from tools.dataProcessing import arr_to_TIFF, calculate_overlay_accuracy,\
                                 extract_img_val_to_sample, overlay_classified_tif
from tools.training_spliting import increase_nonurban_pts, train_test_split_sample


############################################
#           Sample preparing               #
############################################

# 1) extract image values to sample points
extract_img_val_to_sample(force_resample=True)
# 2) split sample into train and test subset
train_test_split_sample()
# 3) gradually increse the size of non-urban built points
increase_nonurban_pts()




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

# save classified hdf to tif
arr_to_TIFF()



############################################
#        Ovelay classifications            #
############################################

# Overlay all the classified tif files (num=10) to one tif file
# and finding the pixles with value > 5 as the "final urban built-up area"
overlay_classified_tif()
calculate_overlay_accuracy()