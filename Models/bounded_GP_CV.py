import bayesnewton
import jax
import objax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from convertbng.util import convert_bng
import time
from jax import vmap
from scipy.stats import beta
import sys, os
sys.path.append('../Utils')
import model_utils as mutils
import kernels_definitions as kerns


##############################################

#DATA VARIABLES
SYSTEMS_NUM = 100
TIMESTEPS_NUM = 50000
TRAIN_FRAC = 24  #IF TRAIN_FRAC > 1 THEN IT BECOMES THE LENGTH OF THE TEST SET
GRID_PIXELS = 10

#OPTIMISATION VARIABLES
LR_ADAM = 0.01
LR_NEWTON = 0.5
ITERS = 25

#GP Variables
VAR_Y = 0.8
LEN_SPACE = 0.5
LEN_ALTITUDE = 0.3

#PERIODIC KERNEL
VAR_PERIOD = 0.8
VAR_MATERN = 0.8
LEN_MATERN = 24 /  (TIMESTEPS_NUM / 100) #48
LEN_PERIOD = 400 /  (TIMESTEPS_NUM / 100)#24

#Want to use a sparse approximation
SPARSE = True
#Should we optimise the inducing points
OPT_Z = True  # will be set to False if SPARSE=SPARSE

#use a mean field approximation?
MEAN_FIELD = True
MINI_BATCH_SIZE = None #none if you don't want them
TEST_STATIONS = 10

def run_main():
    ##############################################

    data = pd.read_csv('../../Data/pv_power_df_5day_capacity_scaled.csv', index_col='datetime')
    uk_pv = pd.read_csv('../../Data/system_metadata_location_rounded.csv')
    uk_pv['ss_id_string'] = uk_pv['ss_id'].astype('str')
    # data_multiple.plot(legend=False)
    lats = dict(uk_pv.set_index('ss_id')['latitude_noisy'])
    longs = dict(uk_pv.set_index('ss_id')['longitude_noisy'])
    data_multiple = data.iloc[:, :SYSTEMS_NUM][:TIMESTEPS_NUM].reset_index()
    stacked = mutils.stack_dataframe(data_multiple, lats, longs)
    ##############################################

    X = np.array(stacked[['epoch', 'longitude', 'latitude']])
    Y = np.array(stacked[['PV']])

    del data, uk_pv, lats, longs, stacked

    # convert to easting and northings
    british_national_grid_coords = convert_bng(X[:, 1], X[:, 2])
    X = np.vstack([X[:, 0],
                   np.array(british_national_grid_coords[0]),
                   np.array(british_national_grid_coords[1])]).T

    # Create a space-time grid from X and Y
    t, R, Y = bayesnewton.utils.create_spatiotemporal_grid(X, Y)
    # SCALING THE t HERE
    t = t / (TIMESTEPS_NUM / 100)

    ##############################################
    # train test split for 3 dimensional data
    t_train, t_test, R_train, R_test, Y_train, Y_test = mutils.train_split_3d(t, R, Y, train_frac=TRAIN_FRAC,
                                                                              split_type='Cutoff')
    Y = Y[:, :, 0]

    # Scale the data
    scaled_values = mutils.scale_2d_train_test_data(R, Y, R_train, R_test, Y_train, Y_test)
    R_scaler, R_scaled, R_train_scaled, R_test_scaled, _, _, _, _ = scaled_values

    # here get a list of scaled coordinates (frozen because at some point in time)
    R_scaled_frozen = R_scaled[0]

    # z = R_scaled[2, ...]
    z = R_scaled[2, ...]


    ##############################################

    # FIXED WINDOW OF 5000 train and 24 test, the 5000 train slide forward
    length_window = 96 * 50
    max_t = len(data_multiple) - length_window - 24 #14000
    iter_step = 50
    # HERE BUILDING ARRAY OF STARTING ts
    data_multiple = data_multiple.set_index('datetime')
    data_multiple.index = pd.to_datetime(data_multiple.index)
    array_of_indices = data_multiple.reset_index()[(data_multiple.reset_index().datetime.dt.hour > 9) & (
                data_multiple.reset_index().datetime.dt.hour < 14)].index.values
    data_multiple = data_multiple.reset_index()
    array_of_indices = array_of_indices[array_of_indices < max_t]
    range_idx = array_of_indices[length_window + 96:max_t:iter_step][-80:]

    # range_idx = np.array([20248, 20347, 20446, 20545, 20644, 20743, 20842, 20941, 21040,
    #                       21139, 21238, 21337, 21436, 21535, 21634, 21782, 21881, 21980,
    #                       22079, 22178, 22277, 22376, 22475, 22574, 22673, 22772, 22871,
    #                       22970, 23069, 23168, 23267, 23366, 23465, 23564, 23663, 23762,
    #                       23861, 23960, 24059, 24207, 24306, 24405, 24504, 24603, 24702,
    #                       24801, 24900, 24999, 25098, 25197, 25296, 25395, 25494, 25593,
    #                       25692, 25791, 25890, 25989, 26088, 26187, 26335, 26434, 26533,
    #                       26632, 26731, 26830, 26929, 27028, 27127, 27226, 27325, 27424,
    #                       27523, 27622, 27721, 27820, 27919, 28018, 28166, 28241])
    ##############################################

    t1 = time.time()
    print(f'Getting results for Gaussian Process')

    errors = np.zeros((24, 1))
    NNLs = np.zeros((24, 1))
    predictions = np.zeros((24, SYSTEMS_NUM, 1))
    variances = np.zeros((24, SYSTEMS_NUM, 1))

    for t_idx in range_idx:
        print('NEW ITERATION WITH t:', t_idx)

        t_iter, R_iter, Y_iter = t[t_idx:t_idx + length_window + 24], R_scaled[t_idx:t_idx + length_window + 24], Y[
                                                                                                                  t_idx:t_idx + length_window + 24]
        t_train_CV, R_train_scaled_CV, Y_train_CV = t_iter[:length_window], R_iter[:length_window], Y_iter[:length_window]
        t_test_CV, R_test_scaled_CV, Y_test_CV = t_iter[length_window:], R_iter[length_window:], Y_iter[length_window:]
        print(
            f'TRAIN SIZE IS: t_train_CV: {t_train_CV.shape}, R_train_scaled_CV: {R_train_scaled_CV.shape}, Y_train_CV :{Y_train_CV.shape}')
        print(
            f'TEST SIZE IS: t_test_CV: {t_test_CV.shape}, R_test_scaled_CV: {R_test_scaled_CV.shape}, Y_test_CV :{Y_test_CV.shape}')

        # IF WE ARE IN THE FIRST ITERATION
        if t_idx == range_idx[0]:
            kern = kerns.get_periodic_kernel(variance_period=VAR_PERIOD,
                                             variance_matern=VAR_MATERN,
                                             lengthscale_time_period=LEN_PERIOD,
                                             lengthscale_time_matern=LEN_MATERN,
                                             lengthscale_space=[LEN_SPACE, LEN_SPACE],
                                             z=z,
                                             sparse=SPARSE,
                                             opt_z=OPT_Z,
                                             conditional='Full',
                                             matern_order='32',
                                             order=2)

            lik = bayesnewton.likelihoods.Beta(scale=30, fix_scale=False, link='probit')

        model = bayesnewton.models.MarkovVariationalMeanFieldGP(kernel=kern, likelihood=lik, X=t_train_CV, Y=Y_train_CV,
                                                                R=R_train_scaled_CV)

        # IF WE ARE NOT IN THE FIRST ITERATION, WARM-START THE TRAINING
        if t_idx != range_idx[0]:
            print('Warm starting the training')
            # HERE I AM SUBSTITUTING THE PREDICTIONS ETC FROM THE MODEL IN THE TRAINING LOCATIONS
            for key in model.vars().keys():
                if model.vars()[key].shape == ():
                    continue
                else:
                    if model.vars()[key].shape[0] == len(t_train_CV):
                        shared_var = model.vars()[key]
                        init_array = jax.numpy.pad(previous_model.vars()[key][iter_step:], ((0, iter_step), (0, 0), (0, 0)))
                        shared_var.assign(init_array)

        opt_hypers = objax.optimizer.Adam(model.vars())
        energy = objax.GradValues(model.energy, model.vars())

        @objax.Function.with_vars(model.vars() + opt_hypers.vars())
        def train_op(batch_ind=None):
            model.inference(lr=LR_NEWTON, batch_ind=batch_ind)  # perform inference and update variational params
            dE, E = energy()  # compute energy and its gradients w.r.t. hypers
            opt_hypers(LR_ADAM, dE)

        train_op = objax.Jit(train_op)

        @objax.Function.with_vars(model.vars())
        def reduced_train_op(batch_ind=None):
            model.inference(lr=LR_NEWTON, batch_ind=batch_ind)  # perform inference and update variational params

        reduced_train_op = objax.Jit(reduced_train_op)

        print('BEGIN TRAINING')
        t0 = time.time()
        loss = []
        # DOING HALF THE ITERATIONS WHEN UPDATING THE MODEL
        iterations_n = ITERS if t_idx == range_idx[0] else int(ITERS / 2)
        for i in range(1, iterations_n + 1):
            if t_idx == range_idx[0]:
                train_op(None)
            else:
                reduced_train_op(None)
            loss.append(model.compute_kl().item())
            print('iter %2d, energy: %1.4f' % (i, loss[i - 1]))
        t1 = time.time()
        print('optimisation time: %2.2f secs' % (t1 - t0))

        print('Performing the predictions')
        # GET THE SYSTEM SPECIFIC PREDICTIONS (NOT THE TOTAL INTERPOLATION)

        f_mean, f_var = model.predict(X=t_test_CV, R=R_test_scaled_CV)

        #################GET THE Y PREDICTIONS FROM THE F VALUES
        f_mean = f_mean.reshape(f_mean.shape[0], -1, 1)
        f_var = f_var.reshape(f_var.shape[0], -1, 1)

        mean_y, var_y = vmap(model.likelihood.predict, (0, 0, None))(f_mean, f_var, None)
        posterior_mean_ts, posterior_var_ts = np.squeeze(mean_y), np.squeeze(var_y)

        predictions = np.concatenate((predictions, posterior_mean_ts[:, :, np.newaxis]), axis=2)
        variances = np.concatenate((variances, posterior_var_ts[:, :, np.newaxis]), axis=2)

        ##################GET THE ERRORS
        error = np.nanmean(abs(np.squeeze(Y_test_CV) - np.squeeze(posterior_mean_ts)), axis=1)[:, np.newaxis]
        print(f'mae is {error.mean()} \n')

        errors = np.concatenate((errors, error), axis=1)

        #################### GET THE NNL
        # SAMPLE THE LATENT VARIABLE AND GET THE SAMPLED DISTRIBUTIONS
        N_samples = 1000
        # Sample values of f at each point
        sampled_f = np.random.normal(f_mean[:, :, 0], f_var[:, :, 0], size=(N_samples, f_var.shape[0], f_var.shape[1]))

        alpha_sampled = model.likelihood.link_fn(sampled_f) * model.likelihood.scale
        beta_sampled = model.likelihood.scale - alpha_sampled

        # GET THE NEGATIVE LOG LIKELIHOOD GIVEN THE SAMPLED DISTRIBUTION AND THE OBSERVED Y VALUES
        observed_repeated = np.repeat(Y_test_CV[np.newaxis, :, :], N_samples, axis=0)
        observed_repeated = observed_repeated.at[observed_repeated == 0].set(10e-6)
        likelihoods = beta.pdf(observed_repeated, alpha_sampled, beta_sampled)
        NNL_hsteps = -np.sum(np.log(likelihoods.mean(axis=0)), axis=1)[:, np.newaxis]

        NNLs = np.concatenate((NNLs, NNL_hsteps), axis=1)

        np.save('NNLs_temporary', NNLs)
        np.save('errors_temporary', errors)
        np.save('predictions_temporary', predictions)
        np.save('variances_temporary', variances)

        #####################

        previous_model = model

        del t_iter, R_iter, Y_iter, t_train_CV, R_train_scaled_CV, Y_train_CV, t_test_CV, R_test_scaled_CV, Y_test_CV
        del NNL_hsteps, observed_repeated, beta_sampled, alpha_sampled, sampled_f, error, posterior_mean_ts, posterior_var_ts
        del mean_y, var_y, f_mean, f_var,  model

    error_evolution = errors[:, 1:].mean(axis=0)
    MAE_hsteps = errors[:, 1:].mean(axis=1)
    NNLs_hsteps = np.quantile(NNLs[:, 1:], 0.5, axis=1)
    NNLs_hsteps_upper = np.quantile(NNLs[:, 1:], 0.975, axis=1)
    NNLs_hsteps_lower = np.quantile(NNLs[:, 1:], 0.025, axis=1)

    error_evolution = pd.DataFrame(error_evolution).rename(columns={0: 'error_evolution'})
    MAE_hsteps = pd.DataFrame(MAE_hsteps).rename(columns={0: 'MAE_hsteps'})
    NNLs_hsteps = pd.DataFrame(NNLs_hsteps).rename(columns={0: 'NNLs_hsteps'})
    NNLs_hsteps_upper = pd.DataFrame(NNLs_hsteps_upper).rename(columns={0: 'NNLs_hsteps_upper'})
    NNLs_hsteps_lower = pd.DataFrame(NNLs_hsteps_lower).rename(columns={0: 'NNLs_hsteps_lower'})

    return error_evolution, MAE_hsteps, NNLs_hsteps, NNLs_hsteps_upper, NNLs_hsteps_lower


if __name__ == '__main__':

    error_evolution, MAE_hsteps, NNLs_hsteps, NNLs_hsteps_upper, NNLs_hsteps_lower = run_main()

    print(f'error_evolution is {error_evolution}')
    print(f'MAE_hsteps is {MAE_hsteps}')
    print(f'NNLs_hsteps is {NNLs_hsteps}')
    print(f'NNLs_hsteps_upper is {NNLs_hsteps_upper}')
    print(f'NNLs_hsteps_lower is {NNLs_hsteps_lower}')


    # error_evolution.to_csv('error_evolution')
    # MAE_hsteps.to_csv('error_evolution')
    # NNLs_hsteps.to_csv('NNLs_hsteps')
    # NNLs_hsteps_upper.to_csv('NNLs_hsteps_upper')
    # NNLs_hsteps_lower.to_csv('NNLs_hsteps_lower')

