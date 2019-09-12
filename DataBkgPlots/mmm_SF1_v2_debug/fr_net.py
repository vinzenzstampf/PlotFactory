'''
Resources:
https://en.wikipedia.org/wiki/Universal_approximation_theorem
http://neuralnetworksanddeeplearning.com/chap4.html
https://github.com/thomberg1/UniversalFunctionApproximation
'''
from pdb import set_trace
from modules.Samples import createSampleLists, setSumWeights
import modules.Selections as Selections 
from modules.Selections import getSelection
from ROOT import ROOT,TChain, RDataFrame, TCanvas, TH2F
import os
from socket import gethostname
from shutil import copyfile
import uproot as ur
from modules.DataMCPlot import DataMCPlot 
import plotfactory as pf
import pickle

import root_pandas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product

from root_numpy import root2array, tree2array
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback
from keras import backend as K
from keras.activations import softmax

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
       

ROOT.EnableImplicitMT()
# fix random seed for reproducibility (FIXME! not really used by Keras)
np.random.seed(1986)

def createArrays(features, branches, path_to_NeuralNet, faketype = 'DoubleFake'):
    #define basic environmental parameters
    hostname        = gethostname()
    analysis_dir    = '/home/dehuazhu/SESSD/4_production/'
    channel         = 'mmm'
    sample_dict     = {}

# call samples
    samples_all, samples_singlefake, samples_doublefake = createSampleLists(analysis_dir=analysis_dir, server = hostname, channel=channel)
    if faketype == 'SingleFake':
        working_samples = samples_singlefake
    else:
        working_samples = samples_doublefake
    

# necessary if you want to compare data with MC
    working_samples = setSumWeights(working_samples)

# make a TChain object by combining all necessary data samples
    print('###########################################################')
    if faketype == 'DoubleFake': print'# measuring doublefakerate...'
    if faketype == 'SingleFake1': print'# measuring singlefakerate for lepton 1...'
    if faketype == 'SingleFake2': print'# measuring singlefakerate for lepton 2...'
    print'# %d samples to be used:'%(len(working_samples))
    print('###########################################################')
    for w in working_samples: print('{:<20}{:<20}'.format(*[w.name,('path: '+w.ana_dir)]))
    chain = TChain('tree') #TChain'ing all data samples together
    for i,s in enumerate(working_samples):
        # sample = working_samples[0] #super stupid mistake, I'm keeping it here as a painful reminder
        sample = working_samples[i]
        file_name = '/'.join([sample.ana_dir, sample.dir_name, sample.tree_prod_name, 'tree.root'])
        chain.Add(file_name)

# define the selections
    if faketype == 'SingleFake1':
        region = Selections.Region('MR_SF1','mmm','MR_SF1')
        selection_passing = region.data
        selection_failing = region.SF_LT

    if faketype == 'SingleFake2':
        region = Selections.Region('MR_SF2','mmm','MR_SF2')
        selection_passing = region.data
        selection_failing = region.SF_TL

    if faketype == 'DoubleFake':
        region = Selections.Region('MR_DF','mmm','MR_DF')
        selection_passing = region.data
        selection_failing = region.DF

    # now redefine the cutstrings by hand: Skype session with Riccardo on 2019.07.12 morning
    baseline_l1 = '&'.join([
    'l0_pt > 25 '              ,
    'abs(l0_eta) < 2.4 '       ,
    'abs(l0_dz) < 0.2 '        ,
    'abs(l0_dxy) < 0.05 '      ,
    'l0_reliso_rho_03 < 0.2 '  ,
    'l0_id_m == 1  '           ,

    'l2_pt > 10 '              ,
    'abs(l2_eta) < 2.4 '       ,
    'abs(l2_dz) < 0.2 '        ,
    'abs(l2_dxy) < 0.05 '      ,
    'l2_reliso_rho_03 < 0.2 '  ,
    'l2_id_m == 1  '           ,

    'l1_pt > 5 '               ,
    'abs(l1_eta) < 2.4 '       ,

    'nbj == 0'                 ,
    'hnl_q_02 == 0 '           ,
    ])

    tight_l1 = '(l1_reliso_rho_03 < 0.2 && l1_Medium == 1)'
    loose_l1 = '((l1_reliso_rho_03 < 0.38 && abs(l1_eta) < 1.2) || (l1_reliso_rho_03 < 0.29 && abs(l1_eta) > 1.2 && abs(l1_eta) < 2.1) || (l1_reliso_rho_03 < 0.20 && abs(l1_eta) > 2.1))'
            
    selection_passing_SF1 = '&'.join([baseline_l1, tight_l1])
    selection_failing_SF1 = '&'.join([baseline_l1, '!'+tight_l1, loose_l1])

    baseline_l2 = '&'.join([
        'l0_pt > 25 '              , 
        'abs(l0_eta) < 2.4 '       ,
        'abs(l0_dz) < 0.2 '        ,
        'abs(l0_dxy) < 0.05 '      ,
        'l0_reliso_rho_03 < 0.2 '  ,
        'l0_id_m == 1  '           ,

        'l1_pt > 10 '              ,
        'abs(l1_eta) < 2.4 '       ,
        'abs(l1_dz) < 0.2 '        ,
        'abs(l1_dxy) < 0.05 '      ,
        'l1_reliso_rho_03 < 0.2 '  ,
        'l1_id_m == 1  '           ,

        'l2_pt > 5 '               ,
        'abs(l2_eta) < 2.4 '       ,

        'nbj == 0'                 ,
        'hnl_q_01 == 0 '           ,
    ])


    tight_l2 = '(l2_reliso_rho_03 < 0.2 && l2_Medium == 1)'
    loose_l2 = '((l2_reliso_rho_03 < 0.38 && abs(l2_eta) < 1.2) || (l2_reliso_rho_03 < 0.29 && abs(l2_eta) > 1.2 && abs(l2_eta) < 2.1) || (l2_reliso_rho_03 < 0.20 && abs(l2_eta) > 2.1))'
        
    selection_passing_SF2 = '&'.join([baseline_l2, tight_l2])
    selection_failing_SF2 = '&'.join([baseline_l2, '!'+tight_l2, loose_l2])

    #assign the selection strings to the hard coded strings above
    selection_passing = selection_passing_SF1
    selection_failing = selection_failing_SF1

# convert TChain object into numpy arrays for the training
    print 'converting .root ntuples to numpy arrays... (passed events)'
    array_pass = tree2array(
                    chain,
                    branches = branches,
                    selection = selection_passing
                    )
    print 'nevents from array_pass: '+ str(array_pass.size)


    print 'converting .root ntuples to numpy arrays... (failed events)'
    array_fail = tree2array(
                    chain,
                    branches = branches,
                    selection = selection_failing
                    )
    print 'nevents from array_fail: '+ str(array_fail.size)

    df_pass = pd.DataFrame(array_pass)
    df_fail = pd.DataFrame(array_fail)

    df_pass.to_pickle(path_to_NeuralNet + 'training_data_pass.pkl')
    df_fail.to_pickle(path_to_NeuralNet + 'training_data_fail.pkl')

def train(features,branches,path_to_NeuralNet,newArrays = False, faketype = 'DoubleFake'):
    hostname        = gethostname()
    if not os.path.exists(path_to_NeuralNet):
        os.mkdir(path_to_NeuralNet)
        print "Output directory created. "
        print "Output directory: %s"%(path_to_NeuralNet)
    else:
        # print "Output directory: ", path_to_NeuralNet, "already exists, overwriting it!"
        print "Output directory already exists, overwriting it! "
        print "Output directory: %s"%(path_to_NeuralNet)
        os.system("rm -rf %s"%(path_to_NeuralNet))
        os.system("mkdir %s"%(path_to_NeuralNet))
    
    copyfile('modules/fr_net.py', path_to_NeuralNet+'/fr_net.py')
    copyfile('modules/Selections.py', path_to_NeuralNet+'/Selections.py')
    copyfile('modules/Samples.py', path_to_NeuralNet+'/Samples.py')

    print 'cfg files stored in ' + path_to_NeuralNet

    if newArrays == True:
        createArrays(features, branches, path_to_NeuralNet, faketype)

    passing = pd.read_pickle(path_to_NeuralNet + 'training_data_pass.pkl')
    failing = pd.read_pickle(path_to_NeuralNet + 'training_data_fail.pkl')

    # add the target column
    passing['target'] = np.ones (passing.shape[0]).astype(np.int)
    failing['target'] = np.zeros(failing.shape[0]).astype(np.int)

    # concatenate the events and shuffle
    data = pd.concat([passing, failing])
    data = data.sample(frac=1, replace=False, random_state=1986) # shuffle (and DON'T replace the sample)

    # #define ptcone
    data['ptcone'] = (( data.hnl_hn_vis_pt * (data.hnl_iso03_rel_rhoArea < 0.2) ) + ( (data.hnl_iso03_rel_rhoArea >= 0.2) * ( data.hnl_hn_vis_pt * (1. + data.hnl_iso03_rel_rhoArea - 0.2))))
    # features += ['ptcone']
    # branches += ['ptcone']

    # #define abs_eta
    data['abs_eta'] = abs(data.hnl_hn_vis_eta)
    # features += ['abs_eta']
    # branches += ['abs_eta']
    
    # define X and Y
    X = pd.DataFrame(data, columns=branches)
    Y = pd.DataFrame(data, columns=['target'])

    # define the net
    input  = Input((len(features),))
    # dense1 = Dense(64, activation='tanh'   , name='dense1')(input )
    dense1 = Dense(64, activation='relu'   , name='dense1')(input )
    output = Dense( 1, activation='sigmoid', name='output')(dense1)
    # output = Dense( 1, activation='softmax', name='output')(dense1)

    # Define outputs of your model
    model = Model(input, output)

    # compile and choose your loss function (binary cross entropy for a 1-0 classification problem)
    model.compile('Adam', loss='binary_crossentropy')

    # print net summary
    print model.summary()

    # plot the models
    # https://keras.io/visualization/
    plot_model(model, show_shapes=True, show_layer_names=True, to_file=path_to_NeuralNet + 'model.png')

    # normalize inputs FIXME! do it, but do it wisely
    # https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
    from sklearn.preprocessing import QuantileTransformer

    # xx = QuantileTransformer(output_distribution='normal').fit_transform(X[features])

    #improved version of the quantile transformer, saving the parameters later for the evaluation
    qt = QuantileTransformer(output_distribution='normal', random_state=1986)
    qt.fit(X[features])
    xx = qt.transform(X[features])
    pickle.dump( qt, open( path_to_NeuralNet + 'quantile_tranformation.pck', 'w' ) )
    
    # xx = X[features] # use this to bypass the quantile transformer
    # alternative way to scale the inputs
    # https://datascienceplus.com/keras-regression-based-neural-networks/

    # train, give the classifier a head start
    # early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, restore_best_weights=True)

    # train only the classifier. beta is set at 0 and the discriminator is not trained
    # history = model.fit(X[features], Y, epochs=500, validation_split=0.5, callbacks=[es])  
    # history = model.fit(xx, Y, epochs=1000, validation_split=0.5, callbacks=[es])  
    data.to_root(path_to_NeuralNet + 'output_ntuple.root', key='tree', store_index=False)
    history = model.fit(xx, Y, batch_size = 1000, epochs=1000, validation_split=0.5, callbacks=[es])  

    # plot loss function trends for train and validation sample
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(path_to_NeuralNet + 'loss_function_history.pdf')
    plt.clf()

    # calculate predictions on the data sample
    print 'predicting on', data.shape[0], 'events'
    x = pd.DataFrame(data, columns=features)
    # y = model.predict(x)
    # xx = QuantileTransformer(output_distribution='normal').fit_transform(x[features])
    # xx = x[features]# use this to bypass the quantile transformer

    # apply the Quantile Transformer also for the evaluation process
    qt = pickle.load(open( path_to_NeuralNet + 'quantile_tranformation.pck', 'r' ))
    xx = qt.transform(x[features])

    y = model.predict(xx)

    # impose norm conservation if you want probabilities
    # compute the overall rescaling factor scale
    scale = 1.
    # scale = np.sum(passing['target']) / np.sum(y)

    # add the score to the data sample
    data.insert(len(data.columns), 'weight', scale * y)

    # let sklearn do the heavy lifting and compute the ROC curves for you
    fpr, tpr, wps = roc_curve(data.target, data.weight) 
    plt.plot(fpr, tpr)
    plt.savefig(path_to_NeuralNet + 'roc.pdf')

    # save model and weights
    # model.save('modules/net_model_DF.h5')
    model.save(path_to_NeuralNet + 'net.h5')
    print 'Neural Net saved as ' + path_to_NeuralNet + 'net.h5'
    # model.save_weights('net_model_weights.h5')

    # rename branches, if you want
    # data.rename(
    #     index=str, 
    #     columns={'cand_refit_mass12': 'mass12',}, 
    #     inplace=True)

    # save ntuple
    data.to_root(path_to_NeuralNet + 'output_ntuple.root', key='tree', store_index=False)

def makeFriendtree(tree_file_name,sample_name,net_name,path_to_NeuralNet,branches,features,overwrite):
    path_to_tree = path_to_NeuralNet + 'friendtree_fr_%s.root'%sample_name
    if not overwrite:
        if os.path.isfile(path_to_tree): 
            print 'Using existing friendtree at %s'%path_to_tree
            return path_to_tree
        else:
            print 'making friendtree for %s'%sample_name

    f = ur.open(tree_file_name)
    t = f['tree']

    df = t.pandas.df(branches)
    x = pd.DataFrame(df, columns=features)

    from sklearn.preprocessing import QuantileTransformer
    # xx = QuantileTransformer(output_distribution='normal').fit_transform(X[features])
    qt = pickle.load(open( path_to_NeuralNet + 'quantile_tranformation.pck', 'r' ))
    xx = qt.transform(x[features])
    # xx = X[features] # use this to bypass the quantile transformer

    classifier = load_model(net_name)
    print 'predicting on' + tree_file_name
    Y = classifier.predict(xx)
    scale = 1.0
    # add the score to the data_train_l sample
    df.insert(len(df.columns), 'ml_fr', scale * Y)
    df.to_root(path_to_tree, key = 'tree')
    print 'friend tree stored in %s'%path_to_tree
    return path_to_tree

def ptCone():
    # PTCONE = '((l%s_pt*(l%s_reliso_rho_03<0.2))+((l%s_reliso_rho_03>=0.2)*(l%s_pt*(1. + l%s_reliso_rho_03 - 0.2))))'%(lepton,lepton,lepton,lepton,lepton)
    PTCONE   = '(  ( hnl_hn_vis_pt * (hnl_iso03_rel_rhoArea<0.2) ) + ( (hnl_iso03_rel_rhoArea>=0.2) * ( hnl_hn_vis_pt * (1. + hnl_iso03_rel_rhoArea - 0.2) ) )  )'
    return PTCONE


def features_DF():
    features = [
        'l1_eta',
        'l1_phi',
        'l1_pt',
        'l1_reliso_rho_03',
        'l2_eta',
        'l2_phi',
        'l2_pt',
        'l2_reliso_rho_03',
        'hnl_2d_disp',
        'hnl_dr_12',
        'hnl_m_12',
        # 'hnl_hn_vis_eta',
        'hnl_hn_vis_pt',
        'hnl_iso03_rel_rhoArea',
    ]
    return features

def branches_DF(features):
    branches = features + [
        'run',
        'lumi',
        'event',
        'hnl_iso04_rel_rhoArea',
        'l0_pt',
        'l0_eta',
        'l0_dz',
        'l0_dxy',
        'l0_reliso_rho_03',
        'l0_id_m',
        # 'l1_pt',
        # 'l1_eta',
        # 'l2_pt',
        # 'l2_eta',
        # 'l1_reliso_rho_03',
        # 'l2_reliso_rho_03',
        'hnl_q_12',
        'hnl_w_vis_m',
        'l1_Medium',
        'l2_Medium',
        'l1_jet_pt',
        'l2_jet_pt',
        # 'hnl_dr_12',
        # 'hnl_m_12',
        'hnl_hn_vis_eta',
        # 'hnl_iso03_rel_rhoArea',
        # 'hnl_hn_vis_pt',
    ]
    return branches

def features_SF1():
    features = [
        'l1_eta',
        'l1_phi',
        'l1_pt',
        'l1_dxy',
        'l1_dz',
    ]
    return features

def features_SF2():
    features = [
        'l2_eta',
        'l2_phi',
        'l2_pt',
        'l2_dxy',
        'l2_dz',
    ]
    return features

def branches_SF1(features):
    branches = features + [
        'run',
        'lumi',
        'event',
        'l0_pt',
        'l0_eta',
        'l0_dz',
        'l0_dxy',
        'l0_reliso_rho_03',
        'l0_id_m',
        'l2_pt',
        'l2_eta',
        'l2_Medium',
        'l2_jet_pt',
        'l2_reliso_rho_03',
        'l1_Medium',
        'l1_jet_pt',
        'hnl_q_12',
        'hnl_w_vis_m',
        'hnl_dr_12',
        'hnl_m_12',
        'hnl_hn_vis_eta',
        'hnl_hn_vis_pt',
        'hnl_iso03_rel_rhoArea',
        'hnl_iso04_rel_rhoArea',
    ]
    return branches

def branches_SF2(features):
    branches = features + [
        'run',
        'lumi',
        'event',
        'l0_pt',
        'l0_eta',
        'l0_dz',
        'l0_dxy',
        'l0_reliso_rho_03',
        'l0_id_m',
        'l1_pt',
        'l1_eta',
        'l1_Medium',
        'l1_jet_pt',
        'l1_reliso_rho_03',
        'l2_Medium',
        'l2_jet_pt',
        'hnl_q_12',
        'hnl_w_vis_m',
        'hnl_dr_12',
        'hnl_m_12',
        'hnl_hn_vis_eta',
        'hnl_hn_vis_pt',
        'hnl_iso03_rel_rhoArea',
        'hnl_iso04_rel_rhoArea',
    ]
    return branches

def path_to_NeuralNet(faketype ='DoubleFake'):
    if faketype == 'SingleFake1':
        # path_to_NeuralNet = 'NN/mmm_SF1_v1/'
        path_to_NeuralNet = 'NN/mmm_SF1_v2_debug/'

    if faketype == 'SingleFake2':
        # path_to_NeuralNet = 'NN/mmm_SF2_v1/'
        # path_to_NeuralNet = 'NN/mmm_SF2_v2_SingleVariable/'
        path_to_NeuralNet = 'NN/mmm_SF2_v3_AllVariable/'

    if faketype == 'DoubleFake':
        # path_to_NeuralNet = 'NN/mmm_DF_v4/'
        path_to_NeuralNet = 'NN/mmm_DF_v5_etaTraining/'
        # path_to_NeuralNet = 'NN/mmm_DF_v6_CheckNormalization/'
    return path_to_NeuralNet 
#################################################################################

if __name__ == '__main__':
    pf.setpfstyle()
# define input parameters
    #The features are the variable the out should depend on
    features_DF  = features_DF()
    features_SF1 = features_SF1()
    features_SF2 = features_SF2()

#The branches are the variables you want to write in your trees
    branches_DF  = branches_DF(features_DF)
    branches_SF1 = branches_SF1(features_SF1)
    branches_SF2 = branches_SF2(features_SF2)

    # run train() if you want to train the NeuralNet
    faketype = 'SingleFake1'
    # faketype = 'SingleFake2'
    # faketype = 'DoubleFake'


    if faketype == 'SingleFake1':
        features = features_SF1
        branches = branches_SF1
    if faketype == 'SingleFake2':
        features = features_SF2
        branches = branches_SF2
    if faketype == 'DoubleFake':
        features = features_DF
        branches = branches_DF

    path_to_NeuralNet = path_to_NeuralNet(faketype) 

    train(
            features,
            branches,
            path_to_NeuralNet,
            newArrays = True,
            faketype = faketype
            )

