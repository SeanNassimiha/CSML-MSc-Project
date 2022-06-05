import bayesnewton
import jax
import objax
import numpy as np
import pandas as pd
from convertbng.util import convert_bng, convert_lonlat

import sys
sys.path.append('../Utils')
import kernels_definitions as kerns
import model_utils as mutils


#DATA VARIABLES
SYSTEMS_NUM = 50 #len(data.columns)
TIMESTEPS_NUM = 1000 #len(data.index)
TRAIN_FRAC = 0.9
GRID_PIXELS = 25

#OPTIMISATION VARIABLES
LR_ADAM = 0.05
LR_NEWTON = 0.5
ITERS = 15

#GP Variables
VAR_Y = 1.
VAR_F = 1.
LEN_TIME = 1  # step size = 1 (hour)
LEN_SPACE = 1

#Want to use a sparse approximation
SPARSE = True
#Should we optimise the inducing points
OPT_Z = True  # will be set to False if SPARSE=SPARSE

#use a mean field approximation?
MEAN_FIELD = False

data =  pd.read_csv('../../Data/pv_power_df_5day.csv', index_col='datetime').drop(columns=['2657', '2828']) #DROPPING FAULTY SYSTEMS
uk_pv = pd.read_csv('../../Data/system_metadata_location_rounded.csv')

data_multiple = data.iloc[:, :SYSTEMS_NUM][:TIMESTEPS_NUM]
#data_multiple.plot(legend=False)
lats = dict(uk_pv.set_index('ss_id')['latitude_rounded'])
longs = dict(uk_pv.set_index('ss_id')['longitude_rounded'])
a = data_multiple.reset_index()
stacked = mutils.stack_dataframe(a, lats, longs)

X = np.array(stacked[['epoch', 'longitude', 'latitude']])
Y = np.array(stacked[['PV']])

# convert to easting and northings
british_national_grid_coords = convert_bng(X[:, 1], X[:, 2])
X = np.vstack([X[:, 0],
              np.array(british_national_grid_coords[0]),
              np.array(british_national_grid_coords[1])]).T

#Create a space-time grid from X and Y
t, R, Y = bayesnewton.utils.create_spatiotemporal_grid(X, Y)

#train test split for 3 dimensional data
t_train, t_test, R_train, R_test, Y_train, Y_test = mutils.train_split_3d(t, R, Y, train_frac = TRAIN_FRAC)

#get the mask of the test points
test_mask = np.in1d(t.squeeze(), t_test.squeeze())

#Scale the data
scaled_values = mutils.scale_2d_train_test_data(R, Y, R_train, R_test, Y_train, Y_test )
R_scaler, R_scaled, R_train_scaled, R_test_scaled, Y_scaler, Y_scaled, Y_train_scaled, Y_test_scaled = scaled_values

#here get a list of scaled coordinates (frozen because at some point in time)
R_scaled_frozen = R_scaled[0]

#Create a grid to perform prediction/interpolation on
r1, r2, Rplot = mutils.create_grid_from_coords(R = R_scaled_frozen, t = t, R_scaler = R_scaler, N_pixels = GRID_PIXELS)

if SPARSE:
    z = mutils.create_ind_point_grid(R_scaled_frozen, n_points = None)
else:
    z = R[0, ...]



kern = kerns.get_SpatioTemporal_combined(variance=VAR_F,
                                           lengthscale_time=LEN_TIME,
                                           lengthscale_space=[LEN_SPACE, LEN_SPACE/ 5],
                                           z=z,
                                           sparse=SPARSE,
                                           opt_z=OPT_Z,
                                           matern_order = '32',
                                           conditional='Full')

lik = bayesnewton.likelihoods.Gaussian(variance=VAR_Y)

if MEAN_FIELD:
    model_loaded = bayesnewton.models.MarkovVariationalMeanFieldGP(kernel=kern, likelihood=lik, X=t_train, R=R_train_scaled, Y=Y_train_scaled, parallel = True)
else:
    model_loaded = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=t_train, R=R_train_scaled, Y=Y_train_scaled, parallel = True)

objax.io.load_var_collection('model_trial_1.npz', model_loaded.vars())
posterior_mean_ts_loaded, posterior_var_ts_loaded = model_loaded.predict(X=t, R=R_scaled)
