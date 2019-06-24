# https://github.com/bstriner/keras-adversarial/blob/master/examples/example_gan_convolutional.py
# https://github.com/dangeng/Simple_Adversarial_Examples
# https://github.com/bstriner/keras-adversarial
# https://medium.com/data1driveninvestor/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3
# https://arxiv.org/pdf/1703.03507.pdf

import ROOT as rt
from ROOT import RDataFrame as rdf

import root_pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product

from root_numpy import root2array

from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

from pdb import set_trace

# fix random seed for reproducibility
np.random.seed(1986)


# define input features, e.g. pt cone, eta, dxy etc etc.
# Ideally you want to list here all the observables that are correlated with FR
features = [
#     'event',        
#     'lumi',        
#     'run',        
#     'TIGHT',        
     'l2_abs_eta',        
     'l2_ptcone',        
     'l2_abs_dxy',        
#     'l2_dz',        
]

# more branches, useful to apply selection, e.g. mass02 to require on-shell Z
branches = features #+ [
#    'cand_refit_tau_mass',
#    'cand_refit_charge',
#]

# preselections
# here you define your measurement region with Loose lepton selection (Tight is a subset)
#sig_selection = 'abs(cand_refit_tau_mass-1.8)<0.2 & abs(cand_refit_charge)==1'
#bkg_selection = 'abs(cand_refit_tau_mass-1.8)<0.2 & abs(cand_refit_tau_mass-1.78)>0.06 & abs(cand_refit_charge)==1'
data1_selection = '1 == 1'

# load data1set from root ntuples. In your case, first order approx, you only need data1. 
# To be precise you should also load MCs to subtract, but one thing at a time
try: 
    data1 = pd.DataFrame( root2array('data_6_24_training_half.root') )
    data2 = pd.DataFrame( root2array('data_6_24_untouched_half.root') )
    #data2 = pd.DataFrame( root2array('data_eem_6_19.root') )

except:
    print ('\n\tNOT SPLIT FOUND!\n')
    file_name='data_6_24'

    tfile = rt.TFile(file_name+'.root')
    tree = tfile.Get('tree')
    df = rdf(tree)
    n = tree.GetEntries()
    df1 = df.Range(0,int(n/2))
    df2 = df.Range(int(n/2),0)
    df1.Snapshot('tree', '%s_training_half.root'%file_name)
    df2.Snapshot('tree', '%s_untouched_half.root'%file_name)
    data1 = pd.DataFrame( root2array('%s_training_half.root'%file_name) )
    data2 = pd.DataFrame( root2array('%s_untouched_half.root'%file_name) )

# targets
# here you need to define what is Tight and what is LooseNotTight, 1 and 0.
# this is going to be based on lepton isolation and ID
data1['l2_ptcone']  = data1.ptcone #data1.l2_pt * max(1, 1 + data1.l2_reliso_rho_03 - 0.2)
data1['l2_abs_eta'] = np.abs(data1.l2_eta) 
data1['l2_abs_dxy'] = np.abs(data1.l2_dxy)

data2['l2_ptcone']  = data2.ptcone #data1.l2_pt * max(1, 1 + data1.l2_reliso_rho_03 - 0.2)
data2['l2_abs_eta'] = np.abs(data2.l2_eta) 
data2['l2_abs_dxy'] = np.abs(data2.l2_dxy)

# concatenate the events and shuffle
#data1 = pd.concat([sig, bkg])
data1 = data1.sample(frac=1, replace=True, random_state=1986)

# X and Y
# X is a (number of input features) x (number of events) tensor
# Y is a vector of size (number of events) containing the truth (target)
X = pd.DataFrame(data1, columns=branches)
#Y = pd.DataFrame(data1, columns=['target'])
Y = pd.DataFrame(data1, columns=['TIGHT'])

# define the classifier net
classifier_input  = Input((len(features),))
#classifier_dense1 = Dense(64, activation='tanh'   )(classifier_input )
#classifier_dense2 = Dense(64, activation='relu'   )(classifier_dense1)
classifier_dense1 = Dense(128, activation='tanh'  )(classifier_input)
classifier_output = Dense( 1, activation='sigmoid')(classifier_dense1)

# Define outputs of your model
classifier = Model(classifier_input, classifier_output)

# compile
classifier.compile('Adam', loss='binary_crossentropy', loss_weights=[1])        

# plot the models

# https://keras.io/visualization/
plot_model(classifier, show_shapes=True, show_layer_names=True, to_file='classifier.png')

# train
# notice that we only use a subset of X columns as input (feature < branches) 
def train():
    print ('training classifier')
    classifier.fit(X[features], Y, epochs=100, validation_split=0.3)  

# save model
    classifier.save('net.h5')


def predict():
# calculate predictions on the data1 sample
    print ('predicting on', data1.shape[0], 'events')
    x  = pd.DataFrame(data1, columns=features)
    x2 = pd.DataFrame(data2, columns=features)
    y = classifier.predict(x)
    y2 = classifier.predict(x2)

# add the score to the data1 sample
    data1.insert(len(data1.columns), 'score', y)
    k = np.sum(data1.score)
    T = np.count_nonzero(data1.TIGHT)
    K = T/k 
    print(k, T, K)

    data1.insert(len(data1.columns), 'ml_fr_weight', K*y)
    data2.insert(len(data2.columns), 'ml_fr_weight', K*y2)



# let sklearn do the heavy lifting and compute the ROC curves for you
    fpr, tpr, wps = roc_curve(data1.TIGHT, data1.score)

# plot
    plt.xscale('log')
    plt.plot(fpr, tpr, color='m', label=r'Z=$\mathcal{N}(0, 1)$')

# plot the also the diagonal, that corresponds to no random picks, no discrimination power
    xy = [i*j for i,j in product([10.**i for i in range(-2, 0)], [1,2,4,8])]+[1]
    plt.plot(xy, xy, color='grey', linestyle='--')

# cosmetics
    plt.xlabel(r'$\epsilon(\tau)$')
    plt.ylabel(r'$\epsilon(\mu)$')

# axis range
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

# grid
    plt.grid(True)

# legend
    plt.legend(loc='lower right')

# save figure and then clean it
    plt.savefig('roc.pdf')
    plt.clf()

# save ntuple
#    data1.rename(index=str, columns={'cand_refit_tau_mass': 'mass', 'cand_refit_charge': 'charge'}, inplace=True)
    data1.to_root('data_6_18_training_half_output.root', key = 'tree')
    #data2.to_root('%s_untouched_half_output.root'%file_name, key = 'tree')
    data2.to_root('data_eem_6_19_output.root', key = 'tree')

def checkFakeRate(file_name='data_6_18'):

    tfile1 = rt.TFile('data_6_18_training_half_output.root')
    #tfile2 = rt.TFile('%s_untouched_half_output.root'%file_name)
    tfile2 = rt.TFile('data_eem_6_19_output.root')

    tree1 = tfile1.Get('tree')
    tree2 = tfile2.Get('tree')

    tree1.Draw('score>>SCORE_T_trained(100,0,1)',   'TIGHT==1')
    tree1.Draw('score>>SCORE_LNT_trained(100,0,1)', 'TIGHT==0')
    tree2.Draw('score>>SCORE_T_free(100,0,1)',   'TIGHT==1')
    tree2.Draw('score>>SCORE_LNT_free(100,0,1)', 'TIGHT==0')

    h_T_trained   = rt.gDirectory.Get('SCORE_T_trained')
    h_LNT_trained = rt.gDirectory.Get('SCORE_LNT_trained')
    h_T_free   = rt.gDirectory.Get('SCORE_T_free')
    h_LNT_free = rt.gDirectory.Get('SCORE_LNT_free')

    h_T_trained.SetLineColor(rt.kRed+2)
    h_LNT_trained.SetLineColor(rt.kGreen+2)
    h_T_free.SetLineColor(rt.kRed+2)
    h_LNT_free.SetLineColor(rt.kGreen+2)

    c1 = rt.TCanvas('training_half','training_half'); c1.cd()
    h_LNT_trained.SetAxisRange(0.0,0.3,'X')
    h_LNT_trained.SetTitle('')
    h_T_trained.SetTitle('')
    h_LNT_trained.DrawNormalized()
    h_T_trained.DrawNormalized('same')
    c1.BuildLegend(0.5,0.5,0.9,0.75)
    c1.SaveAs('training_half.root')
    c1.SaveAs('training_half.pdf')

    c2 = rt.TCanvas('untouched_half','untouched_half'); c2.cd()
    h_LNT_free.SetAxisRange(0.0,0.3,'X')
    h_LNT_free.SetTitle('')
    h_T_free.SetTitle('')
    h_LNT_free.DrawNormalized()
    h_T_free.DrawNormalized('same')
    c2.BuildLegend(0.5,0.5,0.9,0.75)
    c2.SaveAs('untouched_half.root')
    c2.SaveAs('untouched_half.pdf')
