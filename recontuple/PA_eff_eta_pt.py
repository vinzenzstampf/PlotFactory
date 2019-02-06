import ROOT as rt
import numpy as np
import plotfactory as pf
import ntup_dir as nt
from glob import glob
import sys
from pdb import set_trace
from os.path import normpath, basename
####################################################################################################
outdir = '/afs/cern.ch/work/v/vstampf/plots/recontuple/'
indir = nt.getntupdir()
####################################################################################################
ntdr = basename(normpath(indir))
####################################################################################################
t = pf.makechain(True)
####################################################################################################
####################################################################################################
pf.setpfstyle()
####################################################################################################
####################################################################################################
####################################################################################################
b_eta = np.arange(-2.4,2.4,0.2)
b_pt = np.arange(30.,100,3)
####################################################################################################
####################################################################################################
####################################################################################################
h_eff_eta_d = rt.TH1F('eff_eta_d','eff_eta_d',len(b_eta)-1,b_eta)
h_eff_eta_n = rt.TH1F('eff_eta_n','eff_eta_n',len(b_eta)-1,b_eta)
####################################################################################################
h_eff_pt_d = rt.TH1F('eff_pt_d','eff_pt_d',len(b_pt)-1,b_pt)
h_eff_pt_n = rt.TH1F('eff_pt_n','eff_pt_n',len(b_pt)-1,b_pt)
####################################################################################################
####################################################################################################
####################################################################################################
t.Draw('prompt_ele_eta >> eff_eta_n','prompt_ana_success == 1 & l0_bestmatchtype == 11 & sqrt((l0_bestmatch_eta - l0_eta)^2 + (l0_bestmatch_phi - l0_phi)^2) < 0.2')
t.Draw('prompt_ele_eta >> eff_eta_d','l0_bestmatchtype == 11')
####################################################################################################
####################################################################################################
t.Draw('prompt_ele_pt >> eff_pt_n','prompt_ana_success == 1 & l0_bestmatchtype == 11 & sqrt((l0_bestmatch_eta - l0_eta)^2 + (l0_bestmatch_phi - l0_phi)^2) < 0.2')
t.Draw('prompt_ele_pt >> eff_pt_d','l0_bestmatchtype == 11')
####################################################################################################
####################################################################################################
####################################################################################################
h_eff_eta = rt.TEfficiency(h_eff_eta_n,h_eff_eta_d)
h_eff_eta.SetTitle(';#eta; #varepsilon') 
h_eff_eta.SetMarkerColor(rt.kBlue+2)
h_eff_pt = rt.TEfficiency(h_eff_pt_n,h_eff_pt_d)
h_eff_pt.SetTitle(';p_{T} [GeV]; #varepsilon') 
h_eff_pt.SetMarkerColor(rt.kBlue+2)
####################################################################################################
####################################################################################################
#h_m_maxcos_pur_n.Divide(h_m_maxcos_pur_d)
#h_m_maxcos_eff_n.Divide(h_m_maxcos_eff_d)
#h_m_maxcos_pur_n.SetMarkerColor(rt.kBlue+2)
#h_m_maxcos_eff_n.SetTitle('MaxCosBPA')
#h_m_maxcos_pur_n.SetTitle('MaxCosBPA')
#####################################################################################################
#####################################################################################################
#lst_pur = [h_m_maxpt_pur_n,h_m_maxdxy_pur_n,h_m_maxcos_pur_n,h_m_mindr_pur_n,h_m_minchi2_pur_n]
#lst_eff = [h_m_maxpt_eff_n,h_m_maxdxy_eff_n,h_m_maxcos_eff_n,h_m_mindr_eff_n,h_m_minchi2_eff_n]
#####################################################################################################
#####################################################################################################
#for h in lst_pur+lst_eff:
#    h.GetXaxis().SetTitle('HNL Mass [GeV]')
#    h.SetMarkerSize(0.3)
#for h in lst_eff:
#    h.SetAxisRange(0.92,1.005,'Y')
#    h.GetYaxis().SetTitle('Efficiency')
#for h in lst_pur:
#    h.GetYaxis().SetTitle('Purity')
#    h.SetAxisRange(0.3,0.905,'Y')
#####################################################################################################
####################################################################################################
####################################################################################################
c_eff_eta = rt.TCanvas('eff_eta','eff_eta')
c_eff_pt = rt.TCanvas('eff_pt','eff_pt')
####################################################################################################
####################################################################################################
clist = [c_eff_pt,c_eff_eta]
####################################################################################################
####################################################################################################
c_eff_eta.cd()
#for h in lst_eff: h.Draw('epsame')
h_eff_eta.Draw()
####################################################################################################
####################################################################################################
c_eff_pt.cd()
#for h in lst_pur: h.Draw('epsame')
h_eff_pt.Draw()
#c_m_pur.BuildLegend()
####################################################################################################
####################################################################################################
for c in clist:
    c.cd()
    pf.showlogoprelimsim('CMS')
#    rt.gStyle.SetOptStat()
    c.Modified()
    c.Update()
#    c.SaveAs(outdir+c.GetTitle()+'_'+ntdr+'.root')
#    c.SaveAs(outdir+c.GetTitle()+'_'+ntdr+'.png')
