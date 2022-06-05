import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def datetime_to_epoch(datetime):
    """
        Converts a datetime to a number
        args:
            datatime: is a pandas column
    """
    return datetime.astype('int64') // 1e9


def create_grid_from_coords(R, t, R_scaler, N_pixels=20):
    '''
    Function tha creates a grid from the coordinates of the systems
    R - numpy array of the system coordinates, dimensions = [N_systems, 2] where 2 corresponds to lat,lon
    t - time vector
    R_scaler - used to define the minimum value that it can take on the grid
    N_pixels - number of pixels per dimension of the 2d grid
    '''
    min_value = R_scaler.transform([[0]]).item() #this is the minimum value in the transformed space that these coordinate can take
    X1range = max(R[:, 0]) - min(R[:, 0])
    X2range = max(R[:, 1]) - min(R[:, 1])
    r1 = np.linspace(max(min(R[:, 0]) - 0.05 * X1range, min_value), max(R[:, 0]) + 0.05 * X1range, num=N_pixels)
    r2 = np.linspace(max(min(R[:, 1]) - 0.05 * X2range, min_value), max(R[:, 1]) + 0.05 * X2range, num=N_pixels)
    rA, rB = np.meshgrid(r1, r2)
    r = np.hstack((rA.reshape(-1, 1), rB.reshape(-1, 1)))  # Flattening grid for use in kernel functions
    Rplot = np.tile(r, [t.shape[0], 1, 1])

    return r1, r2, Rplot

def create_ind_point_grid(R, n_points = None):
    '''
    Function that creates the grid of inducing points given the array of coordinates of the systems,
    and the number of inducing points per dimension (essentially in 2d, this gives n_points^2 inducing points)
    :param R: numpy array of the system coordinates, dimensions = [N_systems, 2] where 2 corresponds to lat,lon
    :param n_points: number of inducing points per dimension
    :return: array of inducing points coordinatesm dimensions = [N_systems ^ 2, 2]
    '''
    if n_points is None:
        n_points = int(np.sqrt(len(R)))
    z1 = np.linspace(np.min(R[:, 0]), np.max(R[:, 0]), num=n_points)
    z2 = np.linspace(np.min(R[:, 1]), np.max(R[:, 1]), num=n_points)
    zA, zB = np.meshgrid(z1, z2)  # Adding additional dimension to inducing points grid
    z = np.hstack((zA.reshape(-1, 1), zB.reshape(-1, 1)))  # Flattening grid for use in kernel functions
    return z

def stack_dataframe(pve_df, lats_map, longs_map):
    '''
    Input a dataframe with a column for the datetime, and columns for each system corresponding to the time-series
    of PVE outputs, and returns a stacked dataframes with lat, lon, datetime, epoch, and PV output.
    It requires also a mapping from column names (system IDs) to latitude and longitudes
    :param pve_df: dataframe with columns for datetime and each system, values correspond to PV output
                    shappe: [n_timestamps, N_systems + 1]
    :param lats_map: dictionary with mapping from system_id to latitude
    :param lons_map: dictionary with mapping from system_id to longitude
    :return: dataframe, shape: [n_timestamps * N_systems, 6]
    '''

    stacked = pd.DataFrame()
    for column in pve_df.drop(columns=['datetime']).columns:
        df = pve_df[['datetime', column]]
        df = df.assign(farm = int(df.columns[-1])).rename(columns = {column:'PV'})
        stacked = pd.concat([stacked, df])
    stacked['latitude'] = stacked['farm'].map(lats_map)
    stacked['longitude'] = stacked['farm'].map(longs_map)
    stacked['datetime'] = pd.to_datetime(stacked['datetime'])
    # stacked['epoch'] = datetime_to_epoch(stacked['datetime'])
    stacked['epoch'] = stacked.index

    return stacked

def train_split_3d(t, R, Y,  train_frac = 0.9):
    '''
    Function that splits the space, time, label data in tran and test sets
    :param t: time data
    :param R: space data
    :param Y: labels
    :param train_frac: training set fraction to keep
    :return: train-test splits
    '''

    # Train and Test split
    test_ix = np.sort(np.random.choice(t.shape[0], int(train_frac * len(t)), replace=False))

    t_train = t[test_ix]
    t_test = np.delete(t, test_ix, axis=0)

    R_train = R[test_ix]
    R_test = np.delete(R, test_ix, axis=0)

    Y_train = Y[test_ix]
    Y_test = np.delete(Y, test_ix, axis=0)

    return t_train, t_test, R_train, R_test, Y_train, Y_test


def scale_2d_train_test_data(R, Y, R_train, R_test, Y_train, Y_test ):
    '''
    Get the space data and labels, and scales them according to the train set only. Then performs the transformation
    on the test set and the total set as well, and returns all scaled dataframes as well as the scaler functions
    :param R:
    :param Y:
    :param R_train:
    :param R_test:
    :param Y_train:
    :param Y_test:
    :return:
    '''
    # HERE I AM SCALING TRAINING AND TEST SET (USING THE SCALER FROM TRAINING TO SCALE THE TEST)
    # I AM FLATTENING THE ARRAYS AND THEN RESHAPING INTO GRIDS FOR THE SCALER

    # create scalers from train data
    R_scaler = StandardScaler().fit(R_train.flatten()[:, np.newaxis])
    Y_scaler = StandardScaler().fit(Y_train.flatten()[:, np.newaxis])

    Y_scaled = Y_scaler.transform(Y.flatten()[:, np.newaxis]).reshape(Y.shape)
    R_scaled = R_scaler.transform(R.flatten()[:, np.newaxis]).reshape(R.shape)

    # Apply scaler on Train Data
    R_train_scaled = R_scaler.transform(R_train.flatten()[:, np.newaxis]).reshape(R_train.shape)
    Y_train_scaled = Y_scaler.transform(Y_train.flatten()[:, np.newaxis]).reshape(Y_train.shape)

    # Apply scaler on Test Data
    R_test_scaled = R_scaler.transform(R_test.flatten()[:, np.newaxis]).reshape(R_test.shape)
    Y_test_scaled = Y_scaler.transform(Y_test.flatten()[:, np.newaxis]).reshape(Y_test.shape)

    return R_scaler, R_scaled, R_train_scaled, R_test_scaled, Y_scaler, Y_scaled, Y_train_scaled, Y_test_scaled

