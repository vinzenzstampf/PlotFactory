from __future__ import division
from ROOT import gROOT as gr
from ROOT import ROOT
from ROOT import RDataFrame as rdf
import os, platform
import ROOT as rt
import numpy as np
from shutil import copyfile
from glob import glob
import pickle
import re, sys
from datetime import datetime
from pdb import set_trace
from copy import deepcopy
from os.path import normpath, basename, split
from collections import OrderedDict
from multiprocessing import Pool
from multiprocessing.dummy import Pool
from itertools import product


#idk why but this doesn't work
#ROOT.EnableImplicitMT(8)

mmm_17B = 'root://cms-xrd-transit.cern.ch//store/user/dezhu/2_ntuples/HN3Lv2.2/mmm/ntuples/Single_mu_2017B/HNLTreeProducer/tree.root'
mmm_17C = 'root://cms-xrd-transit.cern.ch//store/user/dezhu/2_ntuples/HN3Lv2.2/mmm/ntuples/Single_mu_2017C/HNLTreeProducer/tree.root'
mmm_17D = 'root://cms-xrd-transit.cern.ch//store/user/dezhu/2_ntuples/HN3Lv2.2/mmm/ntuples/Single_mu_2017D/HNLTreeProducer/tree.root'
mmm_17E = 'root://cms-xrd-transit.cern.ch//store/user/dezhu/2_ntuples/HN3Lv2.2/mmm/ntuples/Single_mu_2017E/HNLTreeProducer/tree.root'
mmm_17F = 'root://cms-xrd-transit.cern.ch//store/user/dezhu/2_ntuples/HN3Lv2.2/mmm/ntuples/Single_mu_2017F/HNLTreeProducer/tree.root'

eem_17B = '/work/dezhu/4_production/production_20190511_Data_eem/ntuples/Single_ele_2017B/HNLTreeProducer/tree.root'
eem_17C = '/work/dezhu/4_production/production_20190511_Data_eem/ntuples/Single_ele_2017C/HNLTreeProducer/tree.root'
eem_17D = '/work/dezhu/4_production/production_20190511_Data_eem/ntuples/Single_ele_2017D/HNLTreeProducer/tree.root'
eem_17E = '/work/dezhu/4_production/production_20190511_Data_eem/ntuples/Single_ele_2017E/HNLTreeProducer/tree.root'
eem_17F = '/work/dezhu/4_production/production_20190511_Data_eem/ntuples/Single_ele_2017F/HNLTreeProducer/tree.root'

mmm_DY      = '/work/dezhu/4_production/production_20190411_Bkg_mmm/ntuples/DYJetsToLL_M50/HNLTreeProducer/tree.root'
mmm_DY_ext  = '/work/dezhu/4_production/production_20190411_Bkg_mmm/ntuples/DYJetsToLL_M50_ext/HNLTreeProducer/tree.root'

eem_DY      = '/work/dezhu/4_production/production_20190511_Bkg_eem/ntuples/DYJetsToLL_M50/HNLTreeProducer/tree.root'
eem_DY_ext  = '/work/dezhu/4_production/production_20190511_Bkg_eem/ntuples/DYJetsToLL_M50_ext/HNLTreeProducer/tree.root'


###########################################################################################################################################################################################
### FAKEABLE OBJECTS AND PROMPT LEPTON DEFINITIONS
###########################################################################################################################################################################################
PTCONE   = '(  ( hnl_hn_vis_pt * (hnl_iso03_rel_rhoArea<0.2) ) + ( (hnl_iso03_rel_rhoArea>=0.2) * ( hnl_hn_vis_pt * (1. + hnl_iso03_rel_rhoArea - 0.2) ) )  )'
PTCONEL1 = '(  ( l1_pt         * (l1_reliso_rho_03<0.2) )      + ( (l1_reliso_rho_03>=0.2)      * ( l1_pt         * (1. + l1_reliso_rho_03 - 0.2) ) )  )'
PTCONEL2 = '(  ( l2_pt         * (l2_reliso_rho_03<0.2) )      + ( (l2_reliso_rho_03>=0.2)      * ( l2_pt         * (1. + l2_reliso_rho_03 - 0.2) ) )  )'

### PROMPT LEPTONS
l0_m = 'l0_pt > 25 && abs(l0_eta) < 2.4 && abs(l0_dz) < 0.2 && abs(l0_dxy) < 0.05 && l0_reliso_rho_03 < 0.2 && l0_id_m == 1'                  # l0 genuine muon
l1_m = 'l1_pt > 10 && abs(l1_eta) < 2.4 && abs(l1_dz) < 0.2 && abs(l1_dxy) < 0.05 && l1_reliso_rho_03 < 0.2 && l1_id_m == 1'                  # l1 genuine muon 
l2_m = 'l2_pt > 10 && abs(l2_eta) < 2.4 && abs(l2_dz) < 0.2 && abs(l2_dxy) < 0.05 && l2_reliso_rho_03 < 0.2 && l2_id_m == 1'                  # l2 genuine muon 

l0_e = 'l0_pt > 25 && abs(l0_eta) < 2.5 && abs(l0_dz) < 0.2 && abs(l0_dxy) < 0.05 && l0_reliso_rho_03 < 0.2 && l0_eid_mva_iso_wp90 == 1'      # l0 genuine electron
l1_e = 'l1_pt > 10 && abs(l1_eta) < 2.5 && abs(l1_dz) < 0.2 && abs(l1_dxy) < 0.05 && l1_reliso_rho_03 < 0.2 && l1_eid_mva_iso_wp90 == 1'      # l1 genuine electron 
l2_e = 'l2_pt > 10 && abs(l2_eta) < 2.5 && abs(l2_dz) < 0.2 && abs(l2_dxy) < 0.05 && l2_reliso_rho_03 < 0.2 && l2_eid_mva_iso_wp90 == 1'      # l2 genuine electron 

### FAKEABLE OBJECTS
l1_m_loose  = 'l1_pt > 5 && abs(l1_eta) < 2.4 && abs(l1_dz) < 2 && abs(l1_dxy) > 0.05'                                              # l1 kinematics and impact parameter
l1_m_loose  = 'l1_pt > 5 && abs(l1_eta) < 2.4 && abs(l1_dz) < 2 && abs(l1_dxy) > 0.01'#  >0.01 for non-DY!                         # l1 kinematics and impact parameter
l1_m_tight  = l1_m_loose + ' &&  l1_Medium == 1 && l1_reliso_rho_03 < 0.2'
l1_m_lnt    = l1_m_loose + ' && (l1_Medium == 0 || l1_reliso_rho_03 > 0.2)'

l2_m_loose  = 'l2_pt > 5 && abs(l2_eta) < 2.4 && abs(l2_dz) < 2 && abs(l2_dxy) > 0.05'                                              # l2 kinematics and impact parameter
l2_m_loose  = 'l2_pt > 5 && abs(l2_eta) < 2.4 && abs(l2_dz) < 2 && abs(l2_dxy) > 0.01'#  >0.01 for non-DY!                         # l2 kinematics and impact parameter
l2_m_tight  = l2_m_loose + ' &&  l2_Medium == 1 && l2_reliso_rho_03 < 0.2'
l2_m_lnt = l2_m_loose + ' && (l2_Medium == 0 || l2_reliso_rho_03 > 0.2)'

l1_e_loose  = 'l1_pt > 5 && abs(l1_eta) < 2.5 && abs(l1_dz) < 2 && abs(l1_dxy) > 0.05'                                              # l1 kinematics and impact parameter
l1_e_loose  = 'l1_pt > 5 && abs(l1_eta) < 2.5 && abs(l1_dz) < 2 && abs(l1_dxy) > 0.01'                                              # l1 kinematics and impact parameter
l1_e_tight  = l1_e_loose + ' &&  l1_LooseNoIso == 1 && l1_reliso_rho_03 < 0.2'
l1_e_lnt    = l1_e_loose + ' && (l1_LooseNoIso == 0 || l1_reliso_rho_03 > 0.2)'

l2_e_loose  = 'l2_pt > 5 && abs(l2_eta) < 2.5 && abs(l2_dz) < 2 && abs(l2_dxy) > 0.05'                                              # l2 kinematics and impact parameter
l2_e_loose  = 'l2_pt > 5 && abs(l2_eta) < 2.5 && abs(l2_dz) < 2 && abs(l2_dxy) > 0.01'                                              # l2 kinematics and impact parameter
l2_e_tight  = l2_e_loose + ' &&  l2_LooseNoIso == 1 && l2_reliso_rho_03 < 0.2'
l2_e_lnt    = l2_e_loose + ' && (l2_LooseNoIso == 0 || l2_reliso_rho_03 > 0.2)'
###########################################################################################################################################################################################
              ##                 SINGLE FAKE RATE                   ##  
###########################################################################################################################################################################################
### SFR:: LOOSE CUTS OBTAINED THROUGH CDF HEAVY/LIGHT COMPARISON 
SFR_MMM_L_CUT = ' && ( (l1_reliso_rho_03 < 0.6 && abs(l1_eta) < 1.2) || (l1_reliso_rho_03 < 0.95 && abs(l1_eta) > 1.2 && abs(l1_eta) < 2.1) || (l1_reliso_rho_03 < 0.4 && abs(l1_eta) > 2.1) )'  # dR 03 (29.4.19)

### DY - SELECTION
### SFR::MMM 
SFR_MMM_021_L   =  l0_m + ' && ' + l2_m + ' && ' + l1_m_loose 
SFR_MMM_021_L   += ' && hnl_q_02 == 0'                                  # opposite charge 
SFR_MMM_021_L   += SFR_MMM_L_CUT                                    # reliso bound for LOOSE cf. checkIso_mmm_220319 
SFR_MMM_021_LNT =  SFR_MMM_021_L + ' && ' + l1_m_lnt
SFR_MMM_021_T   =  SFR_MMM_021_L + ' && ' + l1_m_tight 

SFR_MMM_012_L   =  l0_m + ' && ' + l1_m + ' && ' + l2_m_loose 
SFR_MMM_012_L   += ' && hnl_q_01 == 0'                                  # opposite charge 
SFR_MMM_012_L   += re.sub('l1', 'l2', SFR_MMM_L_CUT)                # reliso bound for LOOSE cf. checkIso_mmm_220319 
SFR_MMM_012_LNT =  SFR_MMM_012_L + ' && ' + l2_m_lnt
SFR_MMM_012_T   =  SFR_MMM_012_L + ' && ' + l2_m_tight 

SFR_EEM_012_L   =  l0_e + ' && ' + l1_e + ' && ' + l2_m_loose 
SFR_EEM_012_L   += ' && hnl_q_01 == 0'                                  # opposite charge 
#SFR_EEM_012_L   += re.sub('l1', 'l2', SFR_MMM_L_CUT)                # reliso bound for LOOSE cf. checkIso_mmm_220319 
SFR_EEM_012_LNT =  SFR_EEM_012_L + ' && ' + l2_m_lnt
SFR_EEM_012_T   =  SFR_EEM_012_L + ' && ' + l2_m_tight 
###########################################################################################################################################################################################

###########################################################################################################################################################################################
def skim(sample='DY',ch='mmm'):

    if ch == 'mmm':
        d17B = mmm_17B; d17C = mmm_17C; d17D = mmm_17D; d17E = mmm_17E; d17F = mmm_17F; 
        DY = mmm_DY; DY_ext = mmm_DY_ext
        SFR_012_L = SFR_MMM_012_L
        l2_tight = l2_m_tight

    if ch == 'eem':
        d17B = eem_17B; d17C = eem_17C; d17D = eem_17D; d17E = eem_17E; d17F = eem_17F; 
        DY = eem_DY; DY_ext = eem_DY_ext
        SFR_012_L = SFR_EEM_012_L
        l2_tight = l2_m_tight

    t = rt.TChain('tree')

    if sample == 'DY':
        t.Add(DY)
        t.Add(DY_ext)

    if sample == 'data':
        t.Add(d17B)
        t.Add(d17C)
        t.Add(d17D)
        t.Add(d17E)
        t.Add(d17F)

    print '\n\ttotal entries:', t.GetEntries()

    df = rdf(t)

    df1 = df.Filter(SFR_012_L + ' && hnl_dr_12 > 0.3 && hnl_dr_02 > 0.3 && abs(hnl_m_01 - 91.19) < 10 && hnl_q_01 == 0')

    df2 = df1.Define('TIGHT', '1 * (' + l2_tight + ')')

    print '\n\tloose entries in MR:', df2.Count().GetValue()

    num_T = df2.Filter('TIGHT == 1').Count().GetValue()

    print '\n\ttight entries in MR:', num_T

    df2 = df2.Define('ptcone', PTCONEL2)

    branchList = rt.vector('string')()
    for br in ['event', 'lumi', 'run', 'TIGHT', 'l2_reliso_rho_03', 'l2_Medium', 'l2_eta', 'l2_pt', 'l2_dxy', 'l2_dz', 'ptcone']:
        branchList.push_back(br)
 
    df2.Snapshot('tree', '%s_%s_6_19.root'%(sample,ch), branchList)
