import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from tqdm.auto import tqdm

# setting up working directory
if __name__ == '__main__':
    os.chdir('..')

from PARAMETER import REGION, SAMPLE_PTS_PATH,TRAIN_SAMPLE_RATIO,REFERENCE_DATA_PATH



# randomly train_test_split the sample points, and save them to {SAMPLE_PTS_PATH}
def train_test_split_sample(train_sample_ratio=TRAIN_SAMPLE_RATIO, 
                            rand_state=0):
    '''get the sample points from the hdf files, 
    and then split them into training and testing data.

    Input: train_sample_ratio: the ratio of training data.
           rand_state: the random state for random selection.
    Output: None, but save the training and testing data to {SAMPLE_PTS_PATH}.
    '''

    # read the sample values
    sample_X = np.load(f'{SAMPLE_PTS_PATH}/sample_values_{REGION}.npy')
    sample_y = gpd.read_file(f'{SAMPLE_PTS_PATH}/merge_pts_{REGION}.shp')

    # random select {train_sample_ratio} of sample points as training data
    np.random.seed(rand_state)
    train_idx = np.random.choice(len(sample_X), 
                                 int(len(sample_X)*train_sample_ratio), 
                                 replace=False)
    
    # split the sample points into training and testing
    X_train = sample_X[train_idx]
    y_train = sample_y.iloc[train_idx]

    X_test = np.delete(sample_X, train_idx, axis=0)
    y_test = sample_y.drop(train_idx)

    # save the training and testing data with spatial information
    np.save(f'{SAMPLE_PTS_PATH}/X_train_{REGION}.npy', X_train)
    y_train.to_file(f'{SAMPLE_PTS_PATH}/y_train_{REGION}.shp')

    np.save(f'{SAMPLE_PTS_PATH}/X_test_{REGION}.npy', X_test)
    y_test.to_file(f'{SAMPLE_PTS_PATH}/y_test_{REGION}.shp')

    # save the training data without spatial information
    X_train_df = pd.DataFrame(X_train)
    X_train_df['Built'] = y_train['Built'].values
    X_train_df.to_csv(f'{SAMPLE_PTS_PATH}/sample_pts_{REGION}_ALL_Train_Sample.csv', index=False)

    # report the size of training and testing data
    print(f'The size of training data is {len(X_train)}')
    print(f'The size of testing data is {len(X_test)}\n')


# function to split the built points into urban and non-urban
# and gradually increase the size of non-urban sample points
def increase_nonurban_pts():
    '''split the built points into urban and non-urban,
    and gradually increase the size of non-urban sample points.
    '''
    # get the urban_mask. 
    # See data_info of the {REFERENCE_DATA_PATH} to know the producing of urban_mask
    urban_mask = rasterio.open(f'{REFERENCE_DATA_PATH}/raster/urban_mask.tif')
    urban_mask_arr = urban_mask.read(1)
    urban_mask_trans = urban_mask.transform

    # read the sample pts
    sample_X_pts = pd.DataFrame(np.load(f'{SAMPLE_PTS_PATH}/X_train_{REGION}.npy'))
    sample_y_pts = gpd.read_file(f'{SAMPLE_PTS_PATH}/y_train_{REGION}.shp')

    sample_pts = pd.concat([sample_X_pts, sample_y_pts], axis=1)

    # extract the value of urban_mask to sample points
    sample_pts[['col','row']] = sample_pts['geometry'].apply(lambda geo:~urban_mask_trans*(geo.x,geo.y)).tolist()
    sample_pts[['col','row']] = sample_pts[['col','row']].astype(int)
    sample_pts['urban_mask'] = urban_mask_arr[sample_pts['row'],sample_pts['col']]

    # split the sample points into urban and non-urban
    sample_non_built = sample_pts[sample_pts['Built']==0]

    sample_built_urban = sample_pts[(sample_pts['Built']==1) & (sample_pts['urban_mask']==1)]
    sample_built_nonurban = sample_pts[(sample_pts['Built']==1) & (sample_pts['urban_mask']==0)]

    # gradualy increase the size of non-urban sample points
    nonurban_sample_ratio = np.arange(0.1,1.1,0.1)
    
    print(f'Increase non-urban sample points ratio from 0.1 to 1.0...')
    for ratio in tqdm(nonurban_sample_ratio):
        # randomly select non-urban sample points
        nonurban_idx = np.random.choice(len(sample_built_nonurban), 
                                        int(len(sample_built_nonurban)*ratio), 
                                        replace=False)
        sample_built_nonurban_sub = sample_built_nonurban.iloc[nonurban_idx]

        # merge the built_pts (urban and non-urban sample points) 
        #   and non_built_pts (randomly sampled to the same size of built_pts)
        sample_built = pd.concat([sample_built_urban, sample_built_nonurban_sub])
        sample_non_built = sample_non_built.sample(min(len(sample_built),len(sample_non_built)),
                                                   replace=False)

        sample_pts = pd.concat([sample_built,sample_non_built])\
                       .drop(columns=['OBJECTID','geometry','urban_mask','col','row'])
        

        # save the sample points
        sample_pts.to_csv(f'{SAMPLE_PTS_PATH}/sample_pts_{REGION}_{np.round(ratio,1)}.csv', index=False)
    
    # report Done!
    print('Sample preparation done!\n')
       