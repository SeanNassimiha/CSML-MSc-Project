import bayesnewton
import objax
import numpy as np
import pandas as pd
from convertbng.util import convert_bng, convert_lonlat
import time
import logging
import sys
sys.path.append('../Utils')
import model_utils as mutils
import kernels_definitions as kerns
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours

logging.basicConfig(format='%(asctime)s %(message)s', level = logging.INFO)


'''
PARAMETERS TO OPTIMIZE:
- LR_ADAM:  (0.01, 0.1)
- LR_NEWTON : (0.01, 1)
- VAR_Y: (0.1, 1)
- VAR_F : (0.1, 1)
- LEN_TIME: (1, 24)
- LEN_SPACE: (0.01, 1)
- MEAN_FIELD: (0, 1.99) with 0 = False, 1 = True
- matern_order: (1, 3.99) with 1 = 1/2, 2 = 3/2, 3 = 5/2
'''

def preprocessing(system_num, timesteps_num, train_frac, sparse):

    ######################### IMPORTS
    logging.info('Importing the data')
    data =  pd.read_csv('../../Data/pv_power_df_5day.csv', index_col='datetime').drop(columns=['2657', '2828']) #DROPPING FAULTY SYSTEMS
    uk_pv = pd.read_csv('../../Data/system_metadata_location_rounded.csv')
    uk_pv['ss_id_string'] = uk_pv['ss_id'].astype('str')
    ######################### DATASET CREATION
    logging.info('create the X,Y datasets')
    data_multiple = data.iloc[:, :system_num][:timesteps_num]
    lats = dict(uk_pv.set_index('ss_id')['latitude_rounded'])
    longs = dict(uk_pv.set_index('ss_id')['longitude_rounded'])
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
    t_train, t_test, R_train, R_test, Y_train, Y_test = mutils.train_split_3d(t, R, Y, train_frac=train_frac, split_by_day = True)

    # get the mask of the test points
    test_mask = np.in1d(t.squeeze(), t_test.squeeze())

    # Scale the data
    scaled_values = mutils.scale_2d_train_test_data(R, Y, R_train, R_test, Y_train, Y_test)
    R_scaler, R_scaled, R_train_scaled, R_test_scaled, Y_scaler, Y_scaled, Y_train_scaled, Y_test_scaled = scaled_values

    # here get a list of scaled coordinates (frozen because at some point in time)
    R_scaled_frozen = R_scaled[0]

    if sparse:
        z = mutils.create_ind_point_grid(R_scaled_frozen, n_points=None)
    else:
        z = R[0, ...]

    return z, t, R_scaled, Y_scaled,  Y_scaler, t_train, R_train_scaled, Y_train_scaled, test_mask, capacities


def evaluate_test_mae(z, t, R_scaled, Y_scaler, t_train, R_train_scaled, Y_train_scaled, test_mask, capacities, iters, #this is data col
                lr_adam, lr_newton, var_y, var_f, len_time, len_space, mean_field, matern_order): #this is params col

    ######################### KERNEL DEFINING
    # logging.info('Get the Kernel')
    kern = kerns.get_SpatioTemporal_combined(variance=var_f,
                                             lengthscale_time=len_time,
                                             lengthscale_space=[len_space, len_space],
                                             z=z,
                                             sparse=SPARSE,
                                             opt_z=True,
                                             matern_order=matern_order,
                                             conditional='Full')

    ######################### MODEL TRAINING
    # logging.info('Define likelihood, model, target function and parameters')
    lik = bayesnewton.likelihoods.Gaussian(variance=var_y)

    if mean_field:
        model = bayesnewton.models.MarkovVariationalMeanFieldGP(kernel=kern, likelihood=lik, X=t_train,
                                                                R=R_train_scaled, Y=Y_train_scaled, parallel=True)
    else:
        model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=t_train, R=R_train_scaled,
                                                       Y=Y_train_scaled, parallel=True)

    opt_hypers = objax.optimizer.Adam(model.vars())
    energy = objax.GradValues(model.energy, model.vars())

    @objax.Function.with_vars(model.vars() + opt_hypers.vars())
    def train_op(batch_ind=None):
        model.inference(lr=lr_newton, batch_ind=batch_ind)  # perform inference and update variational params
        dE, E = energy()  # compute energy and its gradients w.r.t. hypers
        opt_hypers(lr_adam, dE)
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

    # logging.info('Begin training')
    t0 = time.time()
    for i in range(1, iters + 1):
        for mini_batch in range(number_of_minibatches):
            # if number_of_minibatches > 1:
                # print(f'Doing minibatch {mini_batch}')
            loss = train_op(mini_batches_indices[mini_batch])
        # print('iter %2d, energy: %1.4f' % (i, loss[0]))
    t1 = time.time()
    # print('optimisation time: %2.2f secs' % (t1 - t0))

    Y = Y_scaler.inverse_transform(Y_scaled[:, :, 0]) * capacities
    # logging.info('Get the system specific predictions')
    # GET THE SYSTEM SPECIFIC PREDICTIONS (NOT THE TOTAL INTERPOLATION)
    posterior_mean_ts, posterior_var_ts = model.predict(X=t, R=R_scaled)
    posterior_mean_rescaled = Y_scaler.inverse_transform(posterior_mean_ts) * capacities

    neg_mae_test = - np.nanmean(abs(np.squeeze(Y[test_mask] ) - np.squeeze(posterior_mean_rescaled[test_mask])))
    # print(f'The test MAE is {mae_test.round(3)}')

    return neg_mae_test

def optimise_GP(z, t, R_scaled, Y_scaled, Y_scaler, t_train, R_train_scaled, Y_train_scaled, test_mask, capacities,  iters):
    ''' Apply Bayesian Optimisation to GP hyperparameters'''
    def GP_evaluation(lr_adam, lr_newton, var_y, var_f, len_time, len_space, cont_mean_field, cont_matern_order):
        ''' Wrapper of evaluate_test_mae function'''

        matern_dict = {1: '12', 2: '32', 3: '52'}
        mean_field_dict = {0: False, 1: True}

        #THIS WRAPPER IS NEEDED TO DEAL WITH DISCRETE VARIABLES
        mean_field = mean_field_dict[int(cont_mean_field)]
        matern_order = matern_dict[int(cont_matern_order)]

        return evaluate_test_mae(z, t, R_scaled, Y_scaler, t_train, R_train_scaled, Y_train_scaled, test_mask, capacities, iters, #this is data col
                            lr_adam = lr_adam, lr_newton = lr_newton,
                            var_y = var_y, var_f = var_f,
                            len_time =  len_time, len_space = len_space,
                            mean_field = mean_field, matern_order = matern_order)

    optimiser = BayesianOptimization(
        f = GP_evaluation,
        pbounds = {'lr_adam' : (0.01, 0.5), 'lr_newton': (0.01, 1), 'var_y': (0.1, 1), 'var_f': (0.1, 1),
            'len_time': (1, 24), 'len_space': (0.01, 1.5), 'cont_mean_field': (0, 1.99), 'cont_matern_order': (1, 3.99)},
        verbose = 2
        )
    optimiser.maximize(n_iter = 21, init_points = 3)
    print('final result',optimiser.max)

if __name__ == "__main__":

    SYSTEMS_NUM = 15  # 883 is the max
    TIMESTEPS_NUM = 500  # 70571 is the max
    TRAIN_FRAC = 0.9
    SPARSE = True # Want to use a sparse approximation
    MINI_BATCH_SIZE = 16  # None if you don't want them. Yann LeCun suggests <= 32
    ITERS = 5

    z, t, R_scaled, Y_scaled, Y_scaler, t_train, R_train_scaled, Y_train_scaled, test_mask, capacities = preprocessing(system_num=SYSTEMS_NUM,
                                                                                                 timesteps_num=TIMESTEPS_NUM,
                                                                                                 train_frac=TRAIN_FRAC,
                                                                                                 sparse=SPARSE)
    print(Colours.yellow("--- Optimizing GP ---"))
    optimise_GP(z, t, R_scaled, Y_scaled, Y_scaler, t_train, R_train_scaled, Y_train_scaled, test_mask, capacities, iters= ITERS)