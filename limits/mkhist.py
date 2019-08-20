import ROOT as rt
import plotfactory as pf
from re import sub
from collections import OrderedDict as od

pf.setpfstyle()

# def makeHist():
fin = rt.TFile('/work/dezhu/3_figures/1_DataMC/FinalStates/mmm/SR_v9_Disp1_0p5/root/linear/hnl_m_12_money.root')

can = fin.Get('can')

pad = can.GetPrimitive('can_1')

h_dict = od()

h_list = pad.GetListOfPrimitives()

j = 0
LOI = ['00244948974278', 'M2_Vp022360679775', 'M5_Vp00836660026534', 'M8_Vp00547722557505']
linecolor = [rt.kBlue+1, rt.kRed+1, rt.kCyan+1, rt.kGreen+1, rt.kMagenta+1, rt.kGreen+1]

for h in h_list[:len(h_list)/2+1]:
    h_name = h.GetName()
    if 'HN3L' in h_name:
        h_name = sub('.*HN3L_M_', 'M', h_name)
        h_name = sub('_V_0', '_V', h_name)
        h_name = sub('_mu_massiveAndCKM_LO', '', h_name)
        h.SetName(h_name)
        # for n in LOI:
        if j < 6:
            if 'M5' in h_name:
                h_dict[h_name] = h
                print j
                h_dict[h_name].SetLineColor(linecolor[j])
                j += 1 
    elif 'data' in h_name:
        h.SetName('data_obs')
        h_dict['data_obs'] = h

stack = pad.GetPrimitive('hnl_m_12_money_stack')

for h in stack.GetHists():
    h_name = h.GetName()
    if 'Conversions_DY' in h_name: h_name = 'conversions' #
    if 'WW'             in h_name: h_name = 'VV'
    if 'nonprompt'      in h_name: h_name = 'non-prompt'
    h.SetName(h_name)
    h_dict[h_name] = h
    h_dict[h_name].SetTitle('; di-muon mass [GeV]; Counts')

stk = rt.THStack('stk','stk')
stk.Add(h_dict['conversions'])
stk.Add(h_dict['non-prompt'])

knvs = rt.TCanvas('di_mu_mass_disp2', 'di_mu_mass_disp2')
knvs.cd()
stk.Draw()
for k in h_dict.keys(): 
    if '_V' in k:
        h_dict[k].Draw('samehist')
pf.showlumi('41.5 fb^{-1} (13 TeV)')
pf.showlogopreliminary()
knvs.Modified(); knvs.Update()

