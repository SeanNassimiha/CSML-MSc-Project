import bayesnewton
import objax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from convertbng.util import convert_bng, convert_lonlat
import time

import logging
import sys
sys.path.append('../Utils')
import model_utils as mutils
import kernels_definitions as kerns

logging.basicConfig(format='%(asctime)s %(message)s', level = logging.INFO)

######################### GLOBAL VARIABLES

#DATA VARIABLES
SYSTEMS_NUM = 20 #len(data.columns)
TIMESTEPS_NUM = 5000 #len(data.index)
TRAIN_FRAC = 0.9
GRID_PIXELS = 20

#OPTIMISATION VARIABLES
LR_ADAM = 0.05
LR_NEWTON = 0.5
ITERS = 30

#GP Variables
VAR_Y = 1.
VAR_F = 1.
LEN_TIME = 5  # step size = 1 (hour)
LEN_SPACE = 1

#Want to use a sparse approximation
SPARSE = True
#Should we optimise the inducing points
OPT_Z = True  # will be set to False if SPARSE=SPARSE

######################### IMPORTS
logging.info('Importing the data')
data =  pd.read_csv('../Data/pv_power_df_5day.csv', index_col='datetime').drop(columns=['2657', '2828']) #DROPPING FAULTY SYSTEMS
uk_pv = pd.read_csv('../Data/system_metadata_location_rounded.csv')


######################### DATASET CREATION
logging.info('create the X,Y datasets')
data_multiple = data.iloc[:, :SYSTEMS_NUM][:TIMESTEPS_NUM]
lats = dict(uk_pv.set_index('ss_id')['latitude_rounded'])
longs = dict(uk_pv.set_index('ss_id')['longitude_rounded'])
a = data_multiple.reset_index()
stacked = mutils.stack_dataframe(a, lats, longs)

X = np.array(stacked[['epoch', 'longitude', 'latitude']])
Y = np.array(stacked[['PV']])

logging.info('Create the spatio-temporale grid')
# convert to easting and northings
british_national_grid_coords = convert_bng(X[:, 1], X[:, 2])
X = np.vstack([X[:, 0],
              np.array(british_national_grid_coords[0]),
              np.array(british_national_grid_coords[1])]).T

#Create a space-time grid from X and Y
t, R, Y = bayesnewton.utils.create_spatiotemporal_grid(X, Y)

logging.info('Scale the data')
#performing the scaling for time dimension
t_scaled = (t - min(t)) / (60 * 5) # convert from seconds to 5 mins intervals

#train test split for 3 dimensional data
t_train_scaled, t_test_scaled, R_train, R_test, Y_train, Y_test = mutils.train_split_3d(t_scaled, R, Y, train_frac = TRAIN_FRAC)

#get the mask of the test points
test_mask = np.in1d(t_scaled.squeeze(), t_test_scaled.squeeze())

#Scale the data
scaled_values = mutils.scale_2d_train_test_data(R, Y, R_train, R_test, Y_train, Y_test )
R_scaler, R_scaled, R_train_scaled, R_test_scaled, Y_scaler, Y_scaled, Y_train_scaled, Y_test_scaled = scaled_values

#here get a list of scaled coordinates (frozen because at some point in time)
R_scaled_frozen = R_scaled[0]

#Create a grid to perform prediction/interpolation on
r1, r2, Rplot = mutils.create_grid_from_coords(R = R_scaled_frozen, t = t,  N_pixels = GRID_PIXELS)

if SPARSE:
    z = mutils.create_ind_point_grid(R_scaled_frozen, n_points = None)
else:
    z = R[0, ...]


######################### KERNEL DEFINING
logging.info('Get the Kernel')
#period in year and in day?
#A day is 96 5min time-steps, therefore the period in days is given by
number_of_days = len(t) / 96
total_length = (t[-1] - t[0]).item()
length_of_one_day = total_length / number_of_days
length_of_one_year = length_of_one_day * 365.25

kern = kerns.get_SpatioTemporal_combined(variance=VAR_F,
                                           lengthscale_time=LEN_TIME,
                                           lengthscale_space=[LEN_SPACE*3, LEN_SPACE],
                                           z=z,
                                           sparse=SPARSE,
                                           opt_z=OPT_Z,
                                           matern_order = '32',
                                           conditional='Full')


######################### MODEL TRAINING
logging.info('Define likelihood, model, target function and parameters')
lik = bayesnewton.likelihoods.Gaussian(variance=VAR_Y)
# model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=t_train_scaled, R=R_train_scaled, Y=Y_train_scaled)
model = bayesnewton.models.MarkovVariationalMeanFieldGP(kernel=kern, likelihood=lik, X=t_train_scaled, R=R_train_scaled, Y=Y_train_scaled)
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())

logging.info('START TRAINING!')
@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    model.inference(lr=LR_NEWTON)  # perform inference and update variational params
    dE, E = energy()  # compute energy and its gradients w.r.t. hypers
    opt_hypers(LR_ADAM, dE)
    return E
train_op = objax.Jit(train_op)

t0 = time.time()
for i in range(1, ITERS + 1):
    loss = train_op()
    print('iter %2d, energy: %1.4f' % (i, loss[0]))
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

######################### METRICS
logging.info('Calculate predictive distributions, and NLPD')
# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
t0 = time.time()
print('calculating the posterior predictive distribution ...')
posterior_mean, posterior_var = model.predict(X=t_scaled, R=Rplot)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))

t2 = time.time()
print('calculating the negative log predictive density ...')
nlpd = model.negative_log_predictive_density(X=t_test_scaled, R=R_test_scaled, Y=Y_test_scaled)
t3 = time.time()
print('nlpd calculation time: %2.2f secs' % (t3-t2))
print('nlpd: %2.3f' % nlpd)

########################## PLOT THE GRID PREDICTIONS
logging.info('Calculate the mean prediction for the grid')

z_opt = model.kernel.z.value
mu = bayesnewton.utils.transpose(posterior_mean)
mu = Y_scaler.inverse_transform(mu).reshape(-1, GRID_PIXELS, GRID_PIXELS)
Y = Y_scaler.inverse_transform(Y_scaled[:,:,0])

logging.info('Get the lat-lon coordinates')
#get lat-lon coordinates
longitude_grid, latitude_grid = convert_lonlat(R_scaler.inverse_transform(r1[:, np.newaxis]), R_scaler.inverse_transform(r2[:, np.newaxis]))
longitude_grid, latitude_grid = [x for x in longitude_grid if str(x) != 'nan'], [x for x in latitude_grid if str(x) != 'nan']
longitude_sys_train, latitude_sys_train = convert_lonlat(R_train[:,:,0][0], R_train[:,:,1][0])
longitude_z, latitude_z = convert_lonlat(R_scaler.inverse_transform(z_opt)[:,0], R_scaler.inverse_transform(z_opt)[:,1])

save_result = False
# del model, kern, Rplot  # , var

logging.info('Plot the time sequence of grid predictions')
print('plotting ...')
cmap = cm.viridis
vmin = np.nanpercentile(Y, 1)
vmax = np.nanpercentile(Y, 99)
# get the labels for the dates
dates = pd.to_datetime(a.datetime).dt.date
days_index = max(97, int(((len(t_scaled) / 5) // 97) * 97))  # number of time intervals to match 5 beginnings of days

for time_step in range(t.shape[0])[:50]:
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [20, 1]})
    f.set_figheight(8)
    # f.set_figwidth(8)
    im = a0.imshow(mu[time_step].T, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[longitude_grid[0], longitude_grid[-1], latitude_grid[0], latitude_grid[-1]], origin='lower')
    a0.scatter(longitude_sys_train, latitude_sys_train, cmapmodel=cmap, vmin=vmin, vmax=vmax,
               c=np.squeeze(Y[time_step]), s=50, edgecolors='black')
    plt.colorbar(im, fraction=0.0348, pad=0.03, aspect=30, ax=a0)
    if SPARSE:
        a0.scatter(longitude_z, latitude_z, c='r', s=20, alpha=0.5)  # plot inducing inputs
    a0.set_xlim(longitude_grid[0], longitude_grid[-1])
    a0.set_ylim(latitude_grid[0], latitude_grid[-1])
    a0.set_title(f'PVE at {a.datetime.unique()[time_step]}')
    a0.set_ylabel('Latitude')
    a0.set_xlabel('Longitude')
    a1.vlines(t_scaled[time_step].item(), -1, 1, 'r')
    a1.set_xlabel('time (days)')
    a1.set_xlim(t_scaled[0], t_scaled[-1])

    a1.set_xticks(np.asarray(t_scaled[1:-1:days_index][:, 0].tolist()),
                  labels=dates[0:-1:days_index].values,
                  fontsize=10)
    plt.show()
    plt.close(f)


logging.info('Get the system specific predictions')
#GET THE SYSTEM SPECIFIC PREDICTIONS (NOT THE TOTAL INTERPOLATION)
posterior_mean_ts, posterior_var_ts = model.predict(X=t_scaled, R=R_scaled)
posterior_mean_rescaled = Y_scaler.inverse_transform(posterior_mean_ts)
posterior_pos_twostd_rescaled = Y_scaler.inverse_transform(posterior_mean_ts + 1.96 * np.sqrt(posterior_var_ts))
posterior_neg_twostd_rescaled = Y_scaler.inverse_transform(posterior_mean_ts - 1.96 * np.sqrt(posterior_var_ts))

#adjust this for the correct quantities
rmse = np.sqrt(np.nanmean((np.squeeze(Y) - np.squeeze(posterior_mean_rescaled))**2))
print(f'The RMSE is {rmse.round(3)}')

rmse_train = np.sqrt(np.nanmean((np.squeeze(Y[~test_mask]) - np.squeeze(posterior_mean_rescaled[~test_mask]))**2))
print(f'The train RMSE is {rmse_train.round(3)}')

rmse_test = np.sqrt(np.nanmean((np.squeeze(Y[test_mask]) - np.squeeze(posterior_mean_rescaled[test_mask]))**2))
print(f'The test RMSE is {rmse_test.round(3)}')

logging.info('Plot the time series individually')
for i in range(SYSTEMS_NUM):
    plt.show()
    plt.figure(figsize=(10, 7))
    plt.title(f'Prediction for system {i}')
    plt.plot(np.arange(len(Y)), Y[:, i], "xk")
    plt.plot(np.arange(len(Y)), posterior_mean_rescaled[:, i], c="C0", lw=2, zorder=2)
    plt.fill_between(
        np.arange(len(Y)),
        posterior_neg_twostd_rescaled[:, i],
        posterior_pos_twostd_rescaled[:, i],
        color="C0",
        alpha=0.2)
    plt.xticks(ticks=np.arange(len(Y))[0:-1:days_index], labels=a.datetime[0:-1:days_index].values, size=8)
