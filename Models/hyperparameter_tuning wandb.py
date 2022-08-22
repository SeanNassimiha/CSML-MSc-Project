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
from jax import vmap
import wandb
wandb.login()


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

def preprocessing(system_num, timesteps_num, train_frac, test_system_num = 100, london = True):

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

    a = data_multiple.reset_index()
    stacked = mutils.stack_dataframe(a, lats, longs)
    if london:
        stacked = stacked[
            (stacked.latitude < 52.5) & (stacked.latitude > 50.5) & (stacked.longitude > -1) & (stacked.longitude < 1)]

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
    t_train, t_test, R_train, R_test, Y_train, Y_test = mutils.train_split_3d(t, R, Y, train_frac=train_frac, split_type = 'Cutoff')

    # Scale the data
    scaled_values = mutils.scale_2d_train_test_data(R, Y, R_train, R_test, Y_train, Y_test)
    R_scaler, R_scaled, R_train_scaled, R_test_scaled, _, _, _, _ = scaled_values

    # here get a list of scaled coordinates (frozen because at some point in time)
    R_scaled_frozen = R_scaled[0]

    z = R_scaled[2, ::7]

    #GET UNDEEN LOCS
    data_unseen = data.iloc[:, system_num:system_num + test_system_num][:timesteps_num].reset_index()

    stacked_unseen = mutils.stack_dataframe(data_unseen, lats, longs)
    if london:
        stacked_unseen = stacked_unseen[
            (stacked_unseen.latitude < 52.5) & (stacked_unseen.latitude > 50.5) & (stacked_unseen.longitude > -1) & (
                        stacked_unseen.longitude < 1)]

    X_unseen = np.array(stacked_unseen[['epoch', 'longitude', 'latitude']])
    Y_unseen = np.array(stacked_unseen[['PV']])

    # convert to easting and northings
    british_national_grid_coords_unseen = convert_bng(X_unseen[:, 1], X_unseen[:, 2])
    X_unseen = np.vstack([X_unseen[:, 0],
                          np.array(british_national_grid_coords_unseen[0]),
                          np.array(british_national_grid_coords_unseen[1])]).T

    # Create a space-time grid from X and Y
    t, R_unseen, Y_unseen = bayesnewton.utils.create_spatiotemporal_grid(X_unseen, Y_unseen)
    R_unseen_scaled = np.tile(R_scaler.transform(R_unseen[0]), (R_unseen.shape[0], 1, 1))

    return z, t, t_train, R_train_scaled, Y_train, R_unseen_scaled, Y_unseen

def evaluate_test_mae(z, t, t_train, R_train_scaled, Y_train, iters, R_unseen_scaled, Y_unseen, #this is data col
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
    lik = bayesnewton.likelihoods.Beta(scale = 30, fix_scale=False, link='probit')

    if mean_field:
        model = bayesnewton.models.MarkovVariationalMeanFieldGP(kernel=kern, likelihood=lik, X=t_train,
                                                                R=R_train_scaled, Y=Y_train)
    else:
        model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=t_train, R=R_train_scaled,
                                                       Y=Y_train)

    print(f't_train shape is {t_train.shape}, R_train shape is {R_train_scaled.shape}')
    opt_hypers = objax.optimizer.Adam(model.vars())
    energy = objax.GradValues(model.energy, model.vars())

    @objax.Function.with_vars(model.vars() + opt_hypers.vars())
    def train_op(batch_ind=None):
        model.inference(lr=lr_newton, batch_ind=batch_ind)  # perform inference and update variational params
        dE, E = energy()  # compute energy and its gradients w.r.t. hypers
        opt_hypers(lr_adam, dE)
        return E
    train_op = objax.Jit(train_op)

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
            wandb.log({'loss':loss[0]})
        print('iter %2d, energy: %1.4f' % (i, loss[0]))
    t1 = time.time()
    print('optimisation time: %2.2f secs' % (t1 - t0))
    f_mean_unseen, f_var_unseen = model.predict(X=t, R=R_unseen_scaled)

    # GET THE Y PREDICTIONS FROM THE F VALUES
    f_mean_unseen = f_mean_unseen.reshape(f_mean_unseen.shape[0], -1, 1)
    f_var_unseen = f_var_unseen.reshape(f_var_unseen.shape[0], -1, 1)

    mean_y_unseen, var_y_unseen = vmap(model.likelihood.predict, (0, 0, None))(f_mean_unseen, f_var_unseen, None)
    posterior_mean_unseen, posterior_var_unseen = np.squeeze(mean_y_unseen), np.squeeze(var_y_unseen)

    Y_unseen = Y_unseen[:, :, 0]

    # adjust this for the correct quantities
    mae_test =np.nanmean(abs(np.squeeze(Y_unseen) - np.squeeze(posterior_mean_unseen)))

    wandb.run.summary['mae_test'] = mae_test
    wandb.run.summary['posterior_mean_unseen'] = posterior_mean_unseen
    wandb.run.summary['posterior_var_unseen'] = posterior_var_unseen

    wandb.finish()

if __name__ == "__main__":

    SYSTEMS_NUM = 120  # 883 is the max
    TIMESTEPS_NUM =  35295  # 70571 is the max
    TRAIN_FRAC = 2
    TEST_STATIONS = 271 - 120
    SPARSE = True  # Want to use a sparse approximation
    MINI_BATCH_SIZE = None  # None if you don't want them. Yann LeCun suggests <= 32
    ITERS = 20  # 20
    z, t, t_train, R_train_scaled, Y_train, R_unseen_scaled, Y_unseen = preprocessing(system_num=SYSTEMS_NUM,
                                                                                      timesteps_num=TIMESTEPS_NUM,
                                                                                      train_frac=TRAIN_FRAC,
                                                                                      test_system_num = TEST_STATIONS)

    def wandb_sweep():
        wandb.init()  # project="Nowcasting", entity="snassimiha"
        config = wandb.config
        evaluate_test_mae(z, t, t_train, R_train_scaled, Y_train, ITERS, R_unseen_scaled, Y_unseen,  # this is data col
                          lr_adam=config.lr_adam, lr_newton=config.lr_newton,
                          var_y=config.var_y, var_f=config.var_f,
                          len_time=config.len_time, len_space=config.len_space,
                          mean_field=config.mean_field, matern_order=config.matern_order)


    sweep_config = {'name': 'Interpolation_Study',
                        'method': 'bayes',
                        'parameters':
                            {'lr_adam': {'min': 0.01, 'max': 0.5},
                             'lr_newton': {'min': 0.01, 'max': 1.},
                             'var_y': {'min': 0.1, 'max': 1.},
                             'var_f': {'min': 0.1, 'max': 1.},
                             'len_time': {'min': 1, 'max': 48},
                             'len_space': {'min': 0.01, 'max': 1.5},
                             'mean_field': {"values": [True, False]},
                             'matern_order': {"values": ['12', '32', '52']}}}

    metric = {
        'name': 'mae_test',
        'goal': 'minimize'
    }

    sweep_config['metric'] = metric
    sweep_id = wandb.sweep(sweep_config, project="Nowcasting")

    wandb.agent(sweep_id, function=wandb_sweep, project="Nowcasting")