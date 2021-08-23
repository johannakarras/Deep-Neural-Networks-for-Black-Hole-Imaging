''' Import libraries and packages. '''
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomUniform, Constant
import keras.models
import keras.layers
import keras.initializers
import keras.regularizers
import keras.callbacks
from keras import backend as K
from keras import losses
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.layers import Layer

import ehtim as eh # eht imaging package

from ehtim.observing.obs_helpers import *
from scipy.ndimage import gaussian_filter
import csv
import sys
import datetime
import warnings
import helpers_posci as hp
from losses_posci import Lambda_similarity
from layers_posci import _unet_from_tensor

import gc

# mute the verbose warnings
warnings.filterwarnings("ignore")

# initialize GPU
K.clear_session()
gpu_id = 1
gpu = '/gpu:' + str(gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
#set_session(tf.compat.v1.Session(config=config))

plt.ion()


'''Define observation parameters.'''
eht_array='EHT2019'
target='sgrA'

nsamp = 10000
npix = 32 
fov_param = 100.0
flux_label = 1
sefd_param = 1

tint_sec = 5    # integration time in seconds
tadv_sec = 600  # advance time between scans
tstart_hr = 0   # GMST time of the start of the observation
tstop_hr = 24   # GMST time of the end
bw_hz = 4e9     # bandwidth in Hz

stabilize_scan_phase = False # if true then add a single phase error for each scan to act similar to adhoc phasing
stabilize_scan_amp = False # if true then add a single gain error at each scan
jones = False # apply jones matrix for including noise in the measurements (including leakage)
inv_jones = False # no not invert the jones matrix
frcal = True # True if you do not include effects of field rotation
dcal = True # True if you do not include the effects of leakage
dterm_offset = 0 # a random offset of the D terms is given at each site with this standard deviation away from 1
dtermp = 0

array = 'arrays/EHT2019.txt'
eht = eh.array.load_txt(array)

# Define observation field of view
fov = fov_param * eh.RADPERUAS

# define scientific target
if target == 'm87':
    ra = 12.513728717168174
    dec = 12.39112323919932
elif target == 'sgrA':
    ra = 19.414182210498385
    dec = -29.24170032236311

rf = 230e9
mjd = 57853 # day of observation
nsamp = 10000 # number of samples in dataset

'''
	Prepare the EHT training data
	
	-----------------------------------------------------------------------------------------------------
	Parameters:
		- fov_param: field of view
		- flux_label: 0 represents varying flux, 1 represents constant flux
		- blur_param: fraction of nominal resolution
		- sefd_param: type of site-wide standard deviation
		- eht_array: name of EHT telescope array to use
		- target: imaging target, 'm89' or 'sgrA'
		- data_augmentation: if True, augments training images
		- npix: image dimension (square, pixels)
	-----------------------------------------------------------------------------------------------------
'''
def Prepare_EHT_Data(fov_param, flux_label, blur_param, sefd_param=1, eht_array='eht2017', target='m87', data_augmentation=False, npix=32):
    add_th_noise = True # False if you *don't* want to add thermal error. If there are no sefds in obs_orig it will use the sigma for each data point
    phasecal = True # True if you don't want to add atmospheric phase error. if False then it adds random phases to simulate atmosphere
    ampcal = True # True if you don't want to add atmospheric amplitude error. if False then add random gain errors 
    stabilize_scan_phase = False # if true then add a single phase error for each scan to act similar to adhoc phasing
    stabilize_scan_amp = False # if true then add a single gain error at each scan
    jones = False # apply jones matrix for including noise in the measurements (including leakage)
    inv_jones = False # no not invert the jones matrix
    frcal = True # True if you do not include effects of field rotation
    dcal = True # True if you do not include the effects of leakage
    dterm_offset = 0.05 # a random offset of the D terms is given at each site with this standard deviation away from 1
    dtermp = 0

    tint_sec = 10
    tadv_sec = 600
    tstart_hr = 0
    tstop_hr = 24
    bw_hz = 4e9

	array = 'arrays/' + eht_array + '.txt'
	
    eht = eh.array.load_txt(array)
    fov = fov_param * eh.RADPERUAS
    if target == 'm87':
        ra = 12.513728717168174
        dec = 12.39112323919932
    elif target == 'sgrA':
        ra = 19.414182210498385
        dec = -29.24170032236311
    rf = 230e9
    # npix = 32
    mjd = 57853 # day of observation
    simim = eh.image.make_empty(npix, fov, ra, dec, rf=rf, source='random', mjd=mjd)
    simim.imvec = np.zeros((npix, npix, 1)).reshape((-1, 1))#xdata[0, :, :, :].reshape((-1, 1))
    obs = simim.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, add_th_noise=add_th_noise, ampcal=ampcal, phasecal=phasecal, 
                    stabilize_scan_phase=stabilize_scan_phase, stabilize_scan_amp=stabilize_scan_amp,
                    jones=jones,inv_jones=inv_jones,dcal=dcal, frcal=frcal, dterm_offset=dterm_offset)

    obs_data = obs.unpack(['u', 'v', 'vis', 'sigma'])
    uv = np.hstack((obs_data['u'].reshape(-1,1), obs_data['v'].reshape(-1,1)))
    F = ftmatrix(simim.psize, simim.xdim, simim.ydim, uv, pulse=simim.pulse)

    t1 = obs.data['t1']
    t2 = obs.data['t2']
    vis = obs.data['vis']
    n_sites = np.unique(np.concatenate([t1, t2])).shape[0] + 1

    ###############################################################################
    # generate the discrete Fourier transform matrices for closure phases
    ###############################################################################

    # obs.add_cphase(count='min')
    obs.add_cphase(count='max')
    tc1 = obs.cphase['t1']
    tc2 = obs.cphase['t2']
    tc3 = obs.cphase['t3']

    cphase_map = np.zeros((len(obs.cphase['time']), 3))

    zero_symbol = 10000
    for k1 in range(cphase_map.shape[0]):
        for k2 in list(np.where(obs.data['time']==obs.cphase['time'][k1])[0]):
            if obs.data['t1'][k2] == obs.cphase['t1'][k1] and obs.data['t2'][k2] == obs.cphase['t2'][k1]:
                cphase_map[k1, 0] = k2
                if k2 == 0:
                    cphase_map[k1, 0] = zero_symbol
            elif obs.data['t2'][k2] == obs.cphase['t1'][k1] and obs.data['t1'][k2] == obs.cphase['t2'][k1]:
                cphase_map[k1, 0] = -k2
                if k2 == 0:
                    cphase_map[k1, 0] = -zero_symbol
            elif obs.data['t1'][k2] == obs.cphase['t2'][k1] and obs.data['t2'][k2] == obs.cphase['t3'][k1]:
                cphase_map[k1, 1] = k2
                if k2 == 0:
                    cphase_map[k1, 1] = zero_symbol
            elif obs.data['t2'][k2] == obs.cphase['t2'][k1] and obs.data['t1'][k2] == obs.cphase['t3'][k1]:
                cphase_map[k1, 1] = -k2
                if k2 == 0:
                    cphase_map[k1, 1] = -zero_symbol
            elif obs.data['t1'][k2] == obs.cphase['t3'][k1] and obs.data['t2'][k2] == obs.cphase['t1'][k1]:
                cphase_map[k1, 2] = k2
                if k2 == 0:
                    cphase_map[k1, 2] = zero_symbol
            elif obs.data['t2'][k2] == obs.cphase['t3'][k1] and obs.data['t1'][k2] == obs.cphase['t1'][k1]:
                cphase_map[k1, 2] = -k2
                if k2 == 0:
                    cphase_map[k1, 2] = -zero_symbol


    F_cphase = np.zeros((cphase_map.shape[0], npix*npix, 3), dtype=np.complex64)
    cphase_proj = np.zeros((cphase_map.shape[0], F.shape[0]), dtype=np.float32)
    for k in range(cphase_map.shape[0]):
        for j in range(cphase_map.shape[1]):
            if cphase_map[k][j] > 0:
                if int(cphase_map[k][j]) == zero_symbol:
                    cphase_map[k][j] = 0
                F_cphase[k, :, j] = F[int(cphase_map[k][j]), :]
                cphase_proj[k, int(cphase_map[k][j])] = 1
            else:
                if np.abs(int(cphase_map[k][j])) == zero_symbol:
                    cphase_map[k][j] = 0
                F_cphase[k, :, j] = np.conj(F[int(-cphase_map[k][j]), :])
                cphase_proj[k, int(-cphase_map[k][j])] = -1

    ###############################################################################
    # load the data
    ###############################################################################
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

    
    xdata_train = 1.0*x_train[[k%60000 for k in range(int(nsamp*0.7))]]
    xdata_train = np.pad(xdata_train, ((0,0), (2,2), (2,2)), 'constant')  # get to 32x32

    xdata_train = xdata_train[..., np.newaxis]/255

    # xdata_test = 1.0*x_train_mnist[0:int(nsamp*0.3)]
    xdata_test = 1.0*x_train_mnist[[k%60000 for k in range(int(nsamp*0.3))]]
    xdata_test = np.pad(xdata_test, ((0,0), (2,2), (2,2)), 'constant')  # get to 32x32
    xdata_test = xdata_test[..., np.newaxis]/255
    for k in range(int(0.3*nsamp)):
        xdata_test[k] = 2.2 * gaussian_filter(xdata_test[k], 2)

    xdata = np.concatenate([xdata_train, xdata_test], 0)
    
    ###############################################################################
    # blur training data by 0.3*fwhm
    ###############################################################################
    res = obs.res()
    simim = eh.image.make_empty(32, fov, ra, dec, rf=rf, source='random', mjd=mjd)
    for k in range(xdata.shape[0]):
        simim.imvec = xdata[k, :, :, :].reshape((-1, 1))
        im_out = simim.blur_circ(0.3*res)
        xdata[k, :, :, 0] = im_out.imvec.reshape((32, 32))
            
    ###############################################################################
    # data augmentation
    ###############################################################################
    if data_augmentation:
        print("Adding augmented data to training set....")
        xdata_augmented = np.load('precomputed_xdata/fashion_xdata_data_augmentation2.npy')
        xdata_augmented = xdata.reshape((xdata_augmented.shape[0], 32, 32, 1))
        xdata = np.concatenate([xdata, xdata_augmented], 0)
    
    ###############################################################################
    # define uniform flux = 224.46
    ###############################################################################
    for k in range(len(xdata)):
        xdata[k] = 224.46*xdata[k] / np.sum(xdata[k])

        
    ###############################################################################
    # define additional blurry effect: 
    ###############################################################################
    xdata_blur = np.zeros(xdata.shape)
    res = obs.res()
    for k in range(xdata.shape[0]):
        simim.imvec = xdata[k, :, :, :].reshape((-1, 1))
        im_out = simim.blur_circ(blur_param*res)
        xdata_blur[k, :, :, 0] = im_out.imvec.reshape((32, 32))

    ###############################################################################
    # thermal noises: 0 represents no thermal noises, 1 represents site-varying thermal noises, 2 represents site-equivalent thermal noises
    ###############################################################################
    if sefd_param == 1:
        sigma = 224.46 * np.concatenate([np.expand_dims(obs.data['sigma'], -1), np.expand_dims(obs.data['sigma'], -1)], -1)
    elif sefd_param == 2:
        sigma = 224.46 * np.concatenate([np.expand_dims(obs.data['sigma'], -1), np.expand_dims(obs.data['sigma'], -1)], -1)
        sigma = np.mean(sigma.reshape((-1, ))) * np.ones(sigma.shape)
    else:
        sigma = 224.46 * np.concatenate([np.expand_dims(obs.data['sigma'], -1), np.expand_dims(obs.data['sigma'], -1)], -1)
    
        
    return xdata, xdata_blur, t1, t2, F, tc1, tc2, tc3, F_cphase, cphase_proj, sigma, obs.res(), obs

''' Return images and blurred images from dataset. '''
def get_data(dataset, fwhm, blur_param=0.0):
    xdata = []
    pad_width = 2
    if (dataset == 'fashion' or dataset == 'all'):
        npix = 32
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        xdata = 1.0*x_train[[k%60000 for k in range(int(nsamp))]]
        xdata = np.pad(xdata, ((0,0), (pad_width,pad_width), (pad_width,pad_width)), 'constant')
        xdata = xdata[..., np.newaxis]/255
        
        xdata = xdata.reshape((-1, npix*npix))
        xdata = xdata.reshape((xdata.shape[0], 32, 32, 1))
        
        res = fwhm
        simim = eh.image.make_empty(32, fov, ra, dec, rf=rf, source='random', mjd=mjd)
        for k in range(xdata.shape[0]):
            xdata[k] = 224.46 * xdata[k] / np.sum(xdata[k])
            simim.imvec = xdata[k, :, :, :].reshape((-1, 1))
            im_out = simim.blur_circ(0.3*res)
            xdata[k, :, :, 0] = im_out.imvec.reshape((32, 32))
            
    if (dataset == 'mnist' or dataset == 'all'):
        npix = 32
        (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
        
        xdata_train = 1.0*x_train_mnist[[k%60000 for k in range(int(nsamp))]]
        xdata_train = np.pad(xdata_train, ((0,0), (pad_width,pad_width), (pad_width,pad_width)), 'constant')  # get to 160x160
        xdata_train = xdata_train[..., np.newaxis]/255
        
        xdata = xdata_train.reshape((-1, npix*npix))
        xdata = xdata.reshape((xdata.shape[0], 32, 32, 1))
        
        res = fwhm
        simim = eh.image.make_empty(32, fov, ra, dec, rf=rf, source='random', mjd=mjd)
        for k in range(xdata.shape[0]):
            xdata[k] = 224.46 * xdata[k] / np.sum(xdata[k])
            simim.imvec = xdata[k, :, :, :].reshape((-1, 1))
            im_out = simim.blur_circ(0.3*res)
            xdata[k, :, :, 0] = im_out.imvec.reshape((32, 32))
        
    if (dataset == 'bh_data'):
        bh_sim_data = np.load('bh_sim_data.npy', allow_pickle=True).item()
        bh_data = bh_sim_data['image']
        
        # resize images to 32 x 32 and fov = 100
        bh_data = np.array(bh_data)
        bh_data_reshape = []
        for i in range(len(bh_data)):
            bh_img = eh.image.make_empty(160, 160, ra, dec, rf=rf, source='random', mjd=mjd)
            bh_img.imvec = bh_data[i].flatten()
            bh_img_reshape = bh_img.regrid_image(100, 32)
            bh_data_reshape.append(bh_img_reshape.imvec)
        xdata = np.array(bh_data_reshape).reshape((-1, 32*32))
        for k in range(xdata.shape[0]):
            xdata[k] = 224.46 * xdata[k] / np.sum(xdata[k])
        xdata = np.concatenate([xdata, xdata, xdata, xdata])[:10000] # Make 10,000 images long 
 
    xdata = xdata.reshape((xdata.shape[0], 32, 32, 1))
    
    # Adjust image flux
    xdata_blur = np.zeros(xdata.shape)
    res = fwhm
    simim = eh.image.make_empty(32, fov, ra, dec, rf=rf, source='random', mjd=mjd)
    for k in range(xdata.shape[0]):
        simim.imvec = xdata[k, :, :, :].reshape((-1, 1))
        im_out = simim.blur_circ(blur_param*res)
        xdata_blur[k, :, :, 0] = im_out.imvec.reshape((32, 32))
    
    return xdata, xdata_blur

##############################################################################
# Loss Functions
##############################################################################
# Compute chi-squared loss between true and predicted image
def chisq_loss(x_true, pred_vis):
    # compute true visibility
    true_vis = keras.layers.Lambda(hp.Lambda_dft(global_F))(x_true)
    #tf.print(true_vis, output_stream=sys.stderr)
    #tf.print(global_S, output_stream=sys.stderr)

    # compute chisq loss
    num = tf.reduce_mean(tf.square(tf.divide(tf.abs(tf.subtract(pred_vis, true_vis)), global_S)), axis=0)
    chisq = tf.divide(num, tf.cast(tf.multiply(2, 1691), tf.float32))
    chisq = tf.reduce_mean(chisq)
    return chisq

def Lambda_cross_correlation(x):
    x_true0, x_pred0 = x
    x_true = tf.transpose(x_true0, [1, 2, 0, 3])
    x_pred = tf.transpose(x_pred0, [3, 1, 2, 0])
    cross_correlation = tf.nn.depthwise_conv2d(x_pred, x_true, strides=[1, 1, 1, 1], padding='SAME')
    cross_correlation = tf.transpose(cross_correlation, [3, 1, 2, 0])
    norm_prod = ((tf.sqrt(tf.reduce_sum(tf.square(x_pred0), [1, 2])) + 1e-5) * (tf.sqrt(tf.reduce_sum(tf.square(x_true0), [1, 2])) + 1e-5))
    norm_prod = tf.tensordot(norm_prod, tf.ones((1, 1, 1, 1)), [-1, 0])
    return  cross_correlation / norm_prod

def Lambda_similarity(y_true, y_pred):
    cross_correlation = keras.layers.Lambda(Lambda_cross_correlation)([y_true, y_pred])
    max_cross_corr = keras.layers.MaxPool2D((32, 32))(cross_correlation)
    return 1-K.mean(max_cross_corr)

##############################################################################
# Define custom callback to compute loss on all 3 datasets, without 
# adding to the loss function used for training
##############################################################################
class AdditionalValidationSets(tf.keras.callbacks.Callback):
    def __init__(self, validation_sets, verbose=0, batch_size=None):
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size
        
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}
    
    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            true_img, true_vis, validation_targets, validation_set_name = validation_set
            sample_weights = None
            
            [pred_img, pred_vis] = self.model.predict(true_img.reshape((10000, 32, 32, 1)))
            
            # evaluate MAE
            mae = np.mean(np.abs(np.subtract(true_img, pred_img)))
            valuename = validation_set_name + '_mae'
            self.history.setdefault(valuename, []).append(mae)
            #print(validation_set_name, "_mae: ", mae)
            
            # evaluate chi^2
            true_vis = np.concatenate([np.array(true_vis).real, np.array(true_vis).imag], axis=-1)
            true_vis = np.reshape(true_vis, (10000, 1691, 2))
            chisq = np.mean(np.abs((pred_vis-true_vis)/global_S)**2)/(2*1691)
            #chisq = chisq_loss(tf.constant(true_img), tf.constant(pred_vis))
            valuename = validation_set_name + '_chisq'
            self.history.setdefault(valuename, []).append(chisq)
            
            print(validation_set_name, "_mae: ", mae, ' ', validation_set_name, "_chisq: ", chisq)

##############################################################################
# Define Neural Network Model
##############################################################################
def VisNet(t1, t2, F, n_ising_layers=5, slope_const=1e2, sigma=None, binary_slope=10, obs_prob=None):
    filt = 64
    kern = 3
    acti = None

    input_shape = (32, 32, 1)
    input_xdata = keras.layers.Input(shape=input_shape, name='input')
    
    input_vis = keras.layers.Lambda(hp.Lambda_dft(F))(input_xdata)
    
    '''
    input_shape = (32, 32, 1)
    input_xdata = keras.layers.Input(shape=input_shape, name='input')
    
    input_vis = keras.layers.Lambda(hp.Lambda_dft(F))(input_xdata)
    '''
    #tf.print(input_vis, output_stream=sys.stderr)
   # tf.print(sigma, output_stream=sys.stderr)

    if sigma is not None:
        print("Adding random gaussian noise..")
        input_vis = keras.layers.GaussianNoise(sigma)(input_vis)

    # Define Layers
    vis_reshape = keras.layers.Reshape((2*F.shape[0], ))(input_vis)
    dirty_im = keras.layers.Dense(32*32*1, activation=acti, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dirtyim_fashion')(vis_reshape)
    dirty_im_reshape = keras.layers.Reshape((32, 32, 1))(dirty_im)
    pred_img = _unet_from_tensor(dirty_im_reshape, filt, kern, acti)
    pred_img = keras.layers.ReLU(name='xc')(pred_img)
    pred_vis = keras.layers.Lambda(hp.Lambda_dft(F), name='pred_vis')(pred_img)
                                                              
    # Model returns predicted image and predicted visibilities
    model = keras.models.Model(inputs=[input_xdata], outputs=[pred_img, pred_vis]) 
    return model

##############################################################################
# Training Function
##############################################################################
def Train_VisNet(eht_array, target, fov_param, flux_label, blur_param, sefd_param, lr, nb_epochs_train, batch_size = 32, n_ising_layers = 5, models_dir='', savefile_name='nn_params', data_augmentation=False, weather=False):
    # prepare the training data
    xdata, xdata_blur, t1, t2, F, tc1, tc2, tc3, F_cphase, cphase_proj, sigma, fwhm, obs = Prepare_EHT_Data(fov_param, flux_label, blur_param, sefd_param, eht_array, target, data_augmentation)
    
    n_sites = np.unique(np.concatenate([t1, t2])).shape[0] + 1
    global global_F
    global_F = F
    global global_S
    global_S = sigma
        
    # define the model
    if sefd_param == 0:
        model = VisNet(t1, t2, F, sigma=None, n_ising_layers=n_ising_layers, slope_const=3)
    else:
        model = VisNet(t1, t2, F, sigma=sigma, n_ising_layers=n_ising_layers, slope_const=3)
       
    # Define model optimizer and loss functions (MAE and chi^2)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=['mae', chisq_loss], loss_weights=[1, 1])
    
    # define training call backs
    checkpoint = ModelCheckpoint(models_dir+savefile_name+'best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    saveWeights = keras.callbacks.ModelCheckpoint(savefile_name, save_weights_only=True, period=20)

	# load model from checkpoint
    # model.load_weights('/path/to/model/checkpoint')
    
    init_epoch= 0 

    # Define additional datasets to monitor during training 
    fashion_xdata, fashion_blur = get_data("fashion", fwhm, 0.3)
    fashion_vis = np.matmul(np.reshape(fashion_blur, (10000, 1024)), np.transpose(F)).astype(np.complex64)
    
    mnist_xdata, mnist_blur = get_data("mnist", fwhm, 0.3)
    mnist_vis = np.matmul(np.reshape(mnist_blur, (10000, 1024)), np.transpose(F)).astype(np.complex64)
    
    bh_xdata, bh_blur = get_data("bh_data", fwhm, 0.0)
    bh_vis = np.matmul(np.reshape(bh_blur, (10000, 1024)), np.transpose(F)).astype(np.complex64)
    
    # Define validation set with MNIST digits and fashion MNIST
    validation_set = np.concatenate([mnist_xdata[:3000],xdata[7000:1000]], 0) 
    validation_set_blur = np.concatenate([mnist_blur[:3000],xdata_blur[7000:1000]], 0)

    valid_vis = np.matmul(np.reshape(validation_set, (validation_set.shape[0], 1024)), np.transpose(F)).astype(np.complex64)
    valid_vis = np.concatenate([valid_vis[:3000],valid_vis[7000:1000]], 0)
    
    xdata_vis = np.matmul(np.reshape(xdata, (xdata.shape[0], 1024)), np.transpose(F)).astype(np.complex64)
    xdata = np.concatenate([xdata[:7000], xdata[10000:17000]], 0)
    xdata_blur = np.concatenate([xdata_blur[:7000], xdata_blur[10000:17000]], 0)
    xdata_vis = np.concatenate([xdata_vis[:7000], xdata_vis[10000:17000]], 0)
    
    
    # create a callback that displays the loss of the model during training on each dataset
    losses_callback = AdditionalValidationSets([(fashion_xdata, fashion_vis, fashion_blur, 'fashion_data'),
                                        (mnist_xdata, mnist_vis, mnist_blur, 'mnist_data'),
                                        (bh_xdata, bh_vis, bh_vis, 'bh_data')])
                          
	# train model                         
    model.fit({'input': xdata}, 
                    {'xc': xdata_blur, 'pred_vis': xdata},
                    validation_data = (validation_set, [validation_set_blur, validation_set]),
                    initial_epoch=init_epoch,
                    epochs=1 + nb_epochs_train,
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=[saveWeights, checkpoint, losses_callback]) 

    modelname = os.path.join(models_dir, savefile_name+'.h5')
    model.save_weights(modelname)

    return model

if __name__ == '__main__':    
    eht_array = 'eht2017'
    target = 'm87' #'m87', 'sgrA', 'both'
    lr = 0.001 
    nb_epochs_train = 500
    blur_param = 0.3
    fov_param = 100.0
    sefd_param = 1
    flux_label = 1
    file_index = 'vis' 
    batch_size = 256
    data_augmentation = True
    weather = False
    
    # Define model name and directory to save to
    models_dir = 'Models/nonoise_models/'
    savefile_name = 'nonoise_nnet'

    model = Train_VisNet(eht_array, target, fov_param, flux_label, blur_param, sefd_param, lr, 
                         nb_epochs_train, batch_size = batch_size, n_ising_layers = 5, 
                         models_dir=models_dir, savefile_name=savefile_name, 
                         data_augmentation= data_augmentation,
                         weather=weather)

  
