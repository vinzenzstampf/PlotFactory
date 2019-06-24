import ROOT as rt
from ROOT import RDataFrame as RDF
import root_pandas
import root_numpy as rnp
import uproot as ur
import pandas as pd
import numpy as np


tf_WJ = rt.TFile('/work/dezhu/4_production/production_20190411_Bkg_mmm/ntuples/WJetsToLNu/HNLTreeProducer/tree.root')
tf_DY = rt.TFile('/work/dezhu/4_production/production_20190411_Bkg_mmm/ntuples/DYBB/HNLTreeProducer/tree.root')

t_WJ = tf_WJ.Get('tree')
t_DY = tf_DY.Get('tree')

rdf_WJ = RDF(t_WJ)
rdf_DY = RDF(t_DY)

uf_WJ = ur.open('/work/dezhu/4_production/production_20190411_Bkg_mmm/ntuples/WJetsToLNu/HNLTreeProducer/tree.root')
uf_DY = ur.open('/work/dezhu/4_production/production_20190411_Bkg_mmm/ntuples/DYBB/HNLTreeProducer/tree.root')

ut_WJ = uf_WJ['tree']
ut_DY = uf_DY['tree']

pdf_WJ_out = ut_WJ.pandas.df(['event','lumi','run','l2_pt','l2_dxy','l2_eta'])
pdf_DY_out = ut_DY.pandas.df(['event','lumi','run','l2_pt','l2_dxy','l2_eta'])

# RUN CLASSIFIER HERE

pdf_WJ_out['NEW'] = np.abs(pdf_WJ_out.l2_eta)
pdf_DY_out['NEW'] = np.abs(pdf_DY_out.l2_eta)

pdf_WJ_out.to_root('WJ_weight.root','tree')
pdf_DY_out.to_root('DY_weight.root','tree')

#friend_DY = ...



#https://stackoverflow.com/questions/15815854/how-to-add-column-to-numpy-array
#
#npdf_WJ_out = pdf_WJ_out.to_records() #here make sure that event has type long int and so on
#
#def makeTree(pdf):
#    d_lst = {}
#    d_lst['d_event' ]  = np.array(pdf.event) 
#    d_lst['d_lumi'  ]  = np.array(pdf.lumi)  
#    d_lst['d_run'   ]  = np.array(pdf.run)   
#    d_lst['d_l2_pt' ]  = np.array(pdf.l2_pt)  
#    d_lst['d_l2_dxy']  = np.array(pdf.l2_dxy) 
#    d_lst['d_l2_eta']  = np.array(pdf.l2_eta) 
#
#    
#    return tree
