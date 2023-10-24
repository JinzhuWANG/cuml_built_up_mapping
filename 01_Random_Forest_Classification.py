from tools import get_hdf_files 
from tools.model_pred import get_models, pred_hdf
from tools.dataProcessing import arr_to_TIFF, extract_img_val_to_sample
from tools.training_spliting import increase_nonurban_pts, train_test_split_sample


############################################
#           Sample preparing               #
############################################

# 1) extract image values to sample points
extract_img_val_to_sample()
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

trained_models = get_models()



############################################
#        Perform classification            #
############################################

# apply the model to the hdf files, and then save the prediction to a hdf
pred_hdf(trained_models)

# save classified hdf to tif
print('Saving the classified hdf to tif...')
arr_to_TIFF()