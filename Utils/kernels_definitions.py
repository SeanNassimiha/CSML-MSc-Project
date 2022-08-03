import bayesnewton


def get_SpatioTemporal_combined(variance, lengthscale_time, lengthscale_space, z, sparse,
                                opt_z, matern_order = '52', conditional='Full'):

    if matern_order == '52':
        kern = bayesnewton.kernels.SpatioTemporalMatern52(variance=variance,
                                                   lengthscale_time=lengthscale_time,
                                                   lengthscale_space=lengthscale_space,
                                                   z=z,
                                                   sparse=sparse,
                                                   opt_z=opt_z,
                                                   conditional=conditional)
    elif matern_order == '32':
        kern = bayesnewton.kernels.SpatioTemporalMatern32(variance=variance,
                                                          lengthscale_time=lengthscale_time,
                                                          lengthscale_space=lengthscale_space,
                                                          z=z,
                                                          sparse=sparse,
                                                          opt_z=opt_z,
                                                          conditional=conditional)

    elif matern_order == '12':
        kern = bayesnewton.kernels.SpatioTemporalMatern12(variance=variance,
                                                          lengthscale_time=lengthscale_time,
                                                          lengthscale_space=lengthscale_space,
                                                          z=z,
                                                          sparse=sparse,
                                                          opt_z=opt_z,
                                                          conditional=conditional)

    return kern

def get_subbandKernel(variance, lengthscale_time, lengthscale_space, z, sparse,
                                opt_z, matern_order = '52', conditional='Full'):
    kern_time = bayesnewton.kernels.Cosine(variance=variance, lengthscale=lengthscale_time)
    kern_space0 = bayesnewton.kernels.Matern32(variance=variance, lengthscale=lengthscale_space)
    kern_space1 = bayesnewton.kernels.Matern32(variance=variance, lengthscale=lengthscale_space)
    kern_space = bayesnewton.kernels.Separable([kern_space0, kern_space1])

    kern = bayesnewton.kernels.SpatioTemporalKernel(temporal_kernel=kern_time,
                                                    spatial_kernel=kern_space,
                                                    z=z,
                                                    sparse=sparse,
                                                    opt_z=opt_z,
                                                    conditional=conditional)

    return kern


def get_separate_kernel(variance, lengthscale_time, lengthscale_space, z, sparse, opt_z, conditional='Full'):
    kern_time = bayesnewton.kernels.Matern32(variance=variance, lengthscale=lengthscale_time)
    kern_space0 = bayesnewton.kernels.Matern32(variance=variance, lengthscale=lengthscale_space)
    kern_space1 = bayesnewton.kernels.Matern32(variance=variance, lengthscale=lengthscale_space)
    kern_space = bayesnewton.kernels.Separable([kern_space0, kern_space1])

    kern = bayesnewton.kernels.SpatioTemporalKernel(temporal_kernel=kern_time,
                                                    spatial_kernel=kern_space,
                                                    z=z,
                                                    sparse=sparse,
                                                    opt_z=opt_z,
                                                    conditional=conditional)
    return kern

def get_periodic_kernel(variance_period, variance_matern, lengthscale_time_period, lengthscale_time_matern, lengthscale_space, z, sparse, opt_z, conditional='Full', matern_order='32', order=6,
                        ):

    # kern_time_year = bayesnewton.kernels.QuasiPeriodicMatern32(variance=variance,
    #                                                     lengthscale_periodic = lengthscale_time,
    #                                                     period = 97 * 365,
    #                                                     lengthscale_matern= lengthscale_time * 100)

    if matern_order == '12':
        kern_time_period = bayesnewton.kernels.QuasiPeriodicMatern12(variance= variance_period,
                                                                  lengthscale_periodic=lengthscale_time_period,
                                                                  period=97,
                                                                  lengthscale_matern=lengthscale_time_period * 30,
                                                                  order=order)

        kern_time_matern = bayesnewton.kernels.Matern32(variance=variance_matern, lengthscale=lengthscale_time_matern)

        kern_time_day = bayesnewton.kernels.Sum([kern_time_period, kern_time_matern])

    elif matern_order == '32':

        kern_time_period = bayesnewton.kernels.QuasiPeriodicMatern32(variance= variance_period,
                                                                  lengthscale_periodic= lengthscale_time_period,
                                                                  period=97/ 103,
                                                                  lengthscale_matern= lengthscale_time_period * 20,
                                                                  order=order)

        kern_time_matern = bayesnewton.kernels.Matern32(variance= variance_matern, lengthscale=lengthscale_time_matern)


        kern_time_day = bayesnewton.kernels.Sum([kern_time_period, kern_time_matern])


    # kern_time = bayesnewton.kernels.Sum([kern_time_day, kern_time_year])

    kern_space0 = bayesnewton.kernels.Matern32(variance=variance_period , lengthscale=lengthscale_space)
    kern_space1 = bayesnewton.kernels.Matern32(variance=variance_period , lengthscale=lengthscale_space)
    kern_space = bayesnewton.kernels.Separable([kern_space0, kern_space1])

    kern = bayesnewton.kernels.SpatioTemporalKernel(temporal_kernel=kern_time_day,
                                                    spatial_kernel=kern_space,
                                                    z=z,
                                                    sparse=sparse,
                                                    opt_z=opt_z,
                                                    conditional=conditional)
    return kern

# kern_time_day = bayesnewton.kernels.QuasiPeriodicMatern32(variance=VAR_F,
#                                                     lengthscale_periodic=(length_of_one_day*5),
#                                                     period=length_of_one_day,
#                                                     lengthscale_matern=(length_of_one_day/3),
#                                                     order=6) #ORDER IS THE ORDER OF THE APPROXIMATION OF THE PERIODIC KERNEL TO TRANSFORM INTO STATE SPACE
# kern_time_year = bayesnewton.kernels.QuasiPeriodicMatern32(variance=VAR_F,
#                                                     lengthscale_periodic=length_of_one_year,
#                                                     period=length_of_one_year,
#                                                     lengthscale_matern=(length_of_one_year/5),
#                                                     order=6) #ORDER IS THE ORDER OF THE APPROXIMATION OF THE PERIODIC KERNEL TO TRANSFORM INTO STATE SPACE

# kern_time = bayesnewton.kernels.Sum([kern_time_day, kern_time_year])

# kern_space0 = bayesnewton.kernels.Matern32(variance=VAR_F, lengthscale=LEN_SPACE)

# kern_space1 = bayesnewton.kernels.Matern32(variance=VAR_F, lengthscale=LEN_SPACE)

# kern_space = bayesnewton.kernels.Separable([kern_space0, kern_space1])

# kern = bayesnewton.kernels.SpatioTemporalKernel(temporal_kernel=kern_time_day,
#                                                 spatial_kernel=kern_space,
#                                                 z=z,
#                                                 sparse=SPARSE,
#                                                 opt_z=OPT_Z,
#                                                 conditional='Full')

#################################################################################################################################

# kernel_day = bayesnewton.kernels.Matern32(variance=var_f, lengthscale=(length_of_one_day/3))

# kernel_periodic_day = bayesnewton.kernels.Periodic(variance=VAR_F, lengthscale = (length_of_one_day*5),
#                                                         period = length_of_one_day, order = 6)

# kernel_year = bayesnewton.kernels.Matern32(variance=var_f, lengthscale=(length_of_one_year/5))

# kernel_periodic_year = bayesnewton.kernels.Periodic(variance=VAR_F, lengthscale = length_of_one_year,
#                                                         period = length_of_one_year, order = 6)

# kern_quasiperiodic_day = bayesnewton.kernels.Separable([kernel_day, kernel_periodic_day])
# kern_quasiperiodic_year = bayesnewton.kernels.Separable([kernel_year, kernel_periodic_year])
# kern_time = bayesnewton.kernels.Separable([kern_quasiperiodic_day, kern_quasiperiodic_year])

# kern_space0 = bayesnewton.kernels.Matern32(variance=VAR_F, lengthscale=LEN_SPACE)
# kern_space1 = bayesnewton.kernels.Matern32(variance=VAR_F, lengthscale=LEN_SPACE)
# kern_space = bayesnewton.kernels.Separable([kern_space0, kern_space1])

# kern = bayesnewton.kernels.SpatioTemporalKernel(temporal_kernel=kern_time,
#                                                 spatial_kernel=kern_space,
#                                                 z=z,
#                                                 sparse=SPARSE,
#                                                 opt_z=OPT_Z,
#                                                 conditional='Full')

