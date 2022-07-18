import bayesnewton
import objax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from convertbng.util import convert_bng, convert_lonlat
import time
import math
import logging
import cv2
import sys, os
sys.path.append('../Utils')
import model_utils as mutils
import kernels_definitions as kerns

logging.basicConfig(format='%(asctime)s %(message)s', level = logging.INFO)

######################### GLOBAL VARIABLES

#DATA VARIABLES
SYSTEMS_NUM = 15 #len(data.columns)
TIMESTEPS_NUM = 227 #len(data.index)
TRAIN_FRAC = 0.9
GRID_PIXELS = 5

#OPTIMISATION VARIABLES
LR_ADAM = 0.05
LR_NEWTON = 0.5
ITERS = 5

#GP Variables
VAR_Y = 1.
VAR_F = 1.
LEN_TIME = 20  # step size = 1 (hour)
LEN_SPACE = 1.5

#Want to use a sparse approximation
SPARSE = True
#Should we optimise the inducing points
OPT_Z = False  # will be set to False if SPARSE=False

#use a mean field approximation?
MEAN_FIELD = False
MINI_BATCH_SIZE = None #none if you don't want them
SPLIT_BY_DAY = False
TEST_BATCHES = True

#Number of FPS for the video of the time evolution of predictions
FPS_VIDEO = 50
VIDEO_LIMIT = 1000


#PATH TO SAVE OUTPUTS
# model_string = str(int(MEAN_FIELD)) + "_" + str(int(SYSTEMS_NUM)) + "_" + str(int(TIMESTEPS_NUM))+ "_" + str(int(LEN_TIME)) + "_" + str(int(LEN_SPACE)) + "_" + str(int(ITERS)) + '/'
model_string = 'ATTEMPT/'
folder = 'output/'+model_string
# try:
    # os.mkdir(folder)
# except:
#     folder = 'output/'+'New_'+model_string
    # os.mkdir(folder)

######################### IMPORTS
logging.info('Importing the data')
data =  pd.read_csv('../../Data/pv_power_df_5day.csv', index_col='datetime').drop(columns=['2657', '2828']) #DROPPING FAULTY SYSTEMS
uk_pv = pd.read_csv('../../Data/system_metadata_location_rounded.csv')
uk_pv['ss_id_string'] = uk_pv['ss_id'].astype('str')

######################### DATASET CREATION
logging.info('create the X,Y datasets')
data_multiple = data.iloc[:, :SYSTEMS_NUM][:TIMESTEPS_NUM]
lats = dict(uk_pv.set_index('ss_id')['latitude_noisy'])
longs = dict(uk_pv.set_index('ss_id')['longitude_noisy'])
capacities = uk_pv[uk_pv.ss_id_string.isin(data_multiple.columns)].set_index('ss_id_string')['kwp'].values * 1000
a = data_multiple.reset_index()
stacked = mutils.stack_dataframe(a, lats, longs)

X = np.array(stacked[['epoch', 'longitude', 'latitude']])
Y = np.array(stacked[['PV']])

# convert to easting and northings
british_national_grid_coords = convert_bng(X[:, 1], X[:, 2])
X = np.vstack([X[:, 0],
              np.array(british_national_grid_coords[0]),
              np.array(british_national_grid_coords[1])]).T

# Create a space-time grid from X and Y
t, R, Y = bayesnewton.utils.create_spatiotemporal_grid(X, Y)

# train test split for 3 dimensional data
t_train, t_test, R_train, R_test, Y_train, Y_test = mutils.train_split_3d(t, R, Y, train_frac=TRAIN_FRAC, split_type = 'Cutoff')

# get the mask of the test points
test_mask = np.in1d(t.squeeze(), t_test.squeeze())

# Scale the data
scaled_values = mutils.scale_2d_train_test_data(R, Y, R_train, R_test, Y_train, Y_test)
R_scaler, R_scaled, R_train_scaled, R_test_scaled, Y_scaler, Y_scaled, Y_train_scaled, Y_test_scaled = scaled_values

# here get a list of scaled coordinates (frozen because at some point in time)
R_scaled_frozen = R_scaled[0]

# #Create a grid to perform prediction/interpolation on
r1, r2, Rplot = mutils.create_grid_from_coords(R=R_scaled_frozen, t=t, R_scaler=R_scaler, N_pixels=GRID_PIXELS)

if SPARSE:
    # z = mutils.create_ind_point_grid(R_scaled_frozen, n_points=None)
    z = R_scaled[0, ...]
    z = z[:11] #this is just for debugging
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

# kern = kerns.get_SpatioTemporal_combined(variance=VAR_F,
#                                            lengthscale_time=LEN_TIME,
#                                            lengthscale_space=[LEN_SPACE, LEN_SPACE],
#                                            z=z,
#                                            sparse=SPARSE,
#                                            opt_z=OPT_Z,
#                                            matern_order = '32',
#                                            conditional='Full')

# kern = kerns.get_separate_kernel(variance=VAR_F,
#                                            lengthscale_time=LEN_TIME,
#                                            lengthscale_space=LEN_SPACE,
#                                            z=z,
#                                            sparse=SPARSE,
#                                            opt_z=OPT_Z,
#                                            conditional='Full')

kern = kerns.get_periodic_kernel(variance=VAR_F,
                                           lengthscale_time=LEN_TIME,
                                           lengthscale_space=LEN_SPACE,
                                           z=z,
                                           sparse=SPARSE,
                                           opt_z=OPT_Z,
                                           conditional='FIC')

######################### MODEL TRAINING
logging.info('Define likelilikelihood, model, target function and parameters')
# lik = bayesnewton.likelihoods.Gaussian(variance=VAR_Y)
R_total_train = np.concatenate((R_train_scaled, Rplot[:R_train_scaled.shape[0]]), axis=1)
if MEAN_FIELD:
    # model = bayesnewton.models.MarkovVariationalMeanFieldGP(kernel=kern, likelihood=lik, X=t_train, R=R_train_scaled, Y=Y_train_scaled, parallel = True)
    inf = bayesnewton.inference.Taylor
    mod = bayesnewton.basemodels.MarkovMeanFieldGP
    Mod = bayesnewton.build_model(mod, inf)
    model = Mod(kernel=kern, likelihood=lik, X=t_train, Y=Y_train_scaled, R=R_train_scaled)
else:
    # model = bayesnewton.models.MarkovGaussianProcess(kernel=kern, likelihood=lik, X=t_train, R=R_train_scaled, Y=Y_train_scaled, parallel = True)
    # inf = bayesnewton.inference.Taylor
    # mod = bayesnewton.basemodels.MarkovGP
    # Mod = bayesnewton.build_model(mod, inf)
    # model = Mod(kernel=kern, likelihood=lik, X=t_train, Y=Y_train_scaled, R=R_total_train)

    lik = bayesnewton.likelihoods.Beta(scale=30, fix_scale=False, link='probit')
    model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=t_train, Y=Y_train, R=R_train_scaled)

opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())

@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op(batch_ind = None):
    model.inference(lr=LR_NEWTON, batch_ind = batch_ind)  # perform inference and update variational params
    dE, E = energy()  # compute energy and its gradients w.r.t. hypers
    opt_hypers(LR_ADAM, dE)
    return E

# train_op = objax.Jit(train_op)

############# Define minibatches
if MINI_BATCH_SIZE == None:
    number_of_minibatches = 1
    mini_batches_indices = [None] * number_of_minibatches
else:
    number_of_minibatches = int(len(t_train) / MINI_BATCH_SIZE)
    idx_set = np.arange(len(t_train))
    np.random.shuffle(idx_set)
    mini_batches_indices = np.array_split(idx_set, number_of_minibatches)

############# Begin training
try:
    logging.info('Begin training')
    t0 = time.time()
    for i in range(1, ITERS + 1):
        for mini_batch in range(number_of_minibatches):
            if number_of_minibatches > 1:
                print(f'Doing minibatch {mini_batch}')
            loss = train_op(mini_batches_indices[mini_batch])
        print('iter %2d, energy: %1.4f' % (i, loss[0]))
    t1 = time.time()
    print('optimisation time: %2.2f secs' % (t1-t0))
    avg_time_taken = (t1-t0)/ITERS
except:
    # os.rmdir(folder)
    raise Exception("Training Failed!")


#################### SAVE MODEL
logging.info('Save the model weights in a numpy zipped file')
#CAN SAVE THE MODEL THIS WAY
model_name = folder+'model.npz'
objax.io.save_var_collection(model_name, model.vars())

######################### METRICS
logging.info('Calculate predictive distributions, and NLPD')
# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
if TEST_BATCHES is True:
    n_batches = int(Rplot.shape[1] / 20) + 1
    test_split = np.array_split(Rplot, n_batches, axis=1)

    # calculate posterior predictive distribution via filtering and smoothing at train & test locations:
    print('calculating the posterior predictive distribution ...')

    posterior_mean = np.zeros((TIMESTEPS_NUM, 1))
    posterior_variance = np.zeros((TIMESTEPS_NUM, 1))

    t0 = time.time()
    for mini_batch in range(n_batches):
        post_mean, post_var = model.predict(X=t, R=test_split[mini_batch])
        posterior_mean = np.append(posterior_mean, post_mean, axis=1)
        posterior_variance = np.append(posterior_variance, post_var, axis=1)
    posterior_mean = posterior_mean[:, 1:]
    posterior_variance = posterior_variance[:, 1:]
    t1 = time.time()
    print('prediction time: %2.2f secs' % (t1 - t0))
else:
    t0 = time.time()
    print('calculating the posterior predictive distribution ...')
    posterior_mean, posterior_var = model.predict(X=t, R=Rplot)
    t1 = time.time()
    print('prediction time: %2.2f secs' % (t1-t0))

t2 = time.time()
print('calculating the negative log predictive density ...')
nlpd = model.negative_log_predictive_density(X=t_test, R=R_test_scaled, Y=Y_test_scaled)
t3 = time.time()
print('nlpd calculation time: %2.2f secs' % (t3-t2))
print('nlpd: %2.3f' % nlpd)

########################## PLOT THE GRID PREDICTIONS
logging.info('Calculate the mean prediction for the grid')

z_opt = model.kernel.z.value
mu = Y_scaler.inverse_transform(posterior_mean.flatten()[:, np.newaxis]).reshape(-1, GRID_PIXELS, GRID_PIXELS)
Y = Y_scaler.inverse_transform(Y_scaled[:,:,0])

#get lat-lon coordinates
grid_coord = R_scaler.inverse_transform(np.array(np.c_[r1,r2]))
longitude_grid, latitude_grid =  convert_lonlat(grid_coord[:, 0], grid_coord[:, 1])
longitude_sys_train, latitude_sys_train = convert_lonlat(R_train[:,:,0][0], R_train[:,:,1][0])
longitude_z, latitude_z = convert_lonlat(R_scaler.inverse_transform(z_opt)[:,0], R_scaler.inverse_transform(z_opt)[:,1])

logging.info('Plot the time sequence of grid predictions')
print('plotting ...')
cmap = cm.viridis
vmin = np.nanpercentile(Y, 1)
vmax = np.nanpercentile(Y, 99)
#get the labels for the dates
dates = pd.to_datetime(a.datetime).dt.date
days_index = max(97, int(((len(t) / 5) // 97) * 97)) #number of time intervals to match 5 beginnings of days

for time_step in range(t.shape[0])[:VIDEO_LIMIT]:
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [20, 1]})
    f.set_figheight(8)
    # f.set_figwidth(8)
    im = a0.imshow(mu[time_step], cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[longitude_grid[0], longitude_grid[-1], latitude_grid[0], latitude_grid[-1]], origin='lower')
    a0.scatter(longitude_sys_train, latitude_sys_train, cmap=cmap, vmin=vmin, vmax=vmax,
               c=np.squeeze(Y[time_step]), s=50, edgecolors='black')
    plt.colorbar(im, fraction=0.0348, pad=0.03, aspect=30, ax=a0)
    if SPARSE:
        a0.scatter(longitude_z, latitude_z, c='r', s=20, alpha=0.5)  # plot inducing inputs
    a0.set_xlim(longitude_grid[0], longitude_grid[-1])
    a0.set_ylim(latitude_grid[0], latitude_grid[-1])
    a0.set_title(f'PVE at {a.datetime.unique()[time_step]}')
    a0.set_ylabel('Latitude')
    a0.set_xlabel('Longitude')
    a1.vlines(t[time_step].item(), -1, 1, 'r')
    a1.set_xlabel('time (days)')
    a1.set_xlim(t[0], t[-1])

    a1.set_xticks(np.asarray(t[1:-1:days_index][:, 0].tolist()),
                  labels=dates[0:-1:days_index].values,
                  fontsize=10)

    direction = f'images/fig_'+str(time_step).zfill(len(str(TIMESTEPS_NUM)))+'.png'
    f.savefig(direction)
    # plt.show()
    plt.close(f)

logging.info('Save the images evolution as a video')
#CODE THAT GETS THE IMAGES FROM THE IMAGES FOLDER, CONVERTS THEM INTO A VIDEO, AND SAVES THE VIDEO IN THE OUTPUT FOLDER, THEN DELETES THE IMAGES
image_folder = 'images'
video_name = folder+'video.mp4'
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MP4V'), FPS_VIDEO, (width,height))
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
cv2.destroyAllWindows()
video.release()
#DELETE THE IMAGES FROM THE IMAGE FOLDER
[os.remove(image_folder+'/'+img) for img in os.listdir(image_folder) if img.endswith(".png")]

logging.info('Get the system specific predictions')
#GET THE SYSTEM SPECIFIC PREDICTIONS (NOT THE TOTAL INTERPOLATION)
posterior_mean_ts, posterior_var_ts = model.predict(X=t, R=R_scaled)
posterior_mean_rescaled = Y_scaler.inverse_transform(posterior_mean_ts)
posterior_pos_twostd_rescaled = Y_scaler.inverse_transform(posterior_mean_ts + 1.96 * np.sqrt(posterior_var_ts))
posterior_neg_twostd_rescaled = Y_scaler.inverse_transform(posterior_mean_ts - 1.96 * np.sqrt(posterior_var_ts))

rescaled_Y = (Y * capacities)
doubly_rescaled_posterior = posterior_mean_rescaled * capacities

#adjust this for the correct quantities
mae = np.nanmean(abs(np.squeeze(rescaled_Y) - np.squeeze(rescaled_posterior)))
print(f'The MAE is {mae.round(3)}')

mae_train = np.nanmean(abs(np.squeeze(rescaled_Y[~test_mask]) - np.squeeze(rescaled_posterior[~test_mask])))
print(f'The train MAE is {mae_train.round(3)}')

mae_test = np.nanmean(abs(np.squeeze(rescaled_Y[test_mask]) - np.squeeze(rescaled_posterior[test_mask])))
print(f'The test MAE is {mae_test.round(3)}')

logging.info('Plot the time series individually')
fig, axs = plt.subplots(math.ceil(SYSTEMS_NUM / 6), 6, figsize=(15, 40))
fig.subplots_adjust(hspace=.5, wspace=.001)
axs = axs.ravel()

for i in range(SYSTEMS_NUM):
#     plt.figure(figsize=(10,7))
    axs[i].set_title(f'Prediction for system {i}')
    axs[i].plot(np.arange(len(Y)), Y[:, i], "xk")
    axs[i].plot(np.arange(len(Y)), posterior_mean_rescaled[:, i], c="C0", lw=2, zorder=2)
    axs[i].fill_between(
    np.arange(len(Y)),
    posterior_neg_twostd_rescaled[:, i],
    posterior_pos_twostd_rescaled[:, i],
    color="C0",
    alpha=0.2)
axs[i].set_xticks(ticks=np.arange(len(Y))[0:-1:days_index], labels=a.datetime[0:-1:days_index].values, size=8)
fig.savefig(folder+'time_series_pred.png')

logging.info('Save the model results in a csv table')
#SAVING THE MODEL RESULTS
results = pd.DataFrame([avg_time_taken, nlpd, rmse, rmse_train, rmse_test],
             index=['avg_time_taken', 'nlpd', 'rmse', 'rmse_train', 'rmse_test'], columns = ['results'])
results.to_csv(folder+'results.csv')

