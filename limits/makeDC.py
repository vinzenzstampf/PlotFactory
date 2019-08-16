import ROOT as rt
from collections import OrderedDict
from re import sub
from pdb import set_trace
from copy import deepcopy as dc
from glob import glob
from os import system


def printDataCard(signal_name, signal_rate, conversion_rate, nonprompt_rate, diboson_rate, observed_rate, disp_bin='incl'): 

    with open('hnl_%s_%s_mu_dc.txt' %(signal_name, disp_bin), 'w') as f:

        signal_name = sub('_e',   '', signal_name)
        signal_name = sub('_mu',  '', signal_name)
        signal_name = sub('_tau', '', signal_name)

        name_tab = '\t'; xs_tab = '\t'
        if len(signal_name) < len('M2_Vp00244948974278'):   name_tab = '\t\t'
        if len(signal_name) < len('M2_Vp00244948974278')-2:
            name_tab = '\t\t\t\t'
            xs_tab   = name_tab

        n_samples = 3
        if diboson_rate == 0: n_samples = 2 

        print >> f, 'max     1     number of categories'
        print >> f, 'jmax    %s     number of samples minus one' %n_samples
        print >> f, 'kmax    *     number of nuisance parameters'
        print >> f, '---------------------------------------------------------'
        print >> f, 'shapes *   * hnl_mll_combine_input_aug_15.root $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC' 
        print >> f, '---------------------------------------------------------'
        print >> f, 'bin                              %s' %disp_bin
        print >> f, 'observation                      %.2f' %observed_rate
        print >> f, '-----------------------------------------------------------------'
        if not diboson_rate == 0:
            print >> f, 'bin                              %s                     %s                     %s                     %s       ' %(disp_bin, disp_bin, disp_bin, disp_bin)    
            print >> f, 'process                          0                       1                       2                       3         '
            print >> f, 'process                          %s%s conversions             non-prompt              VV        ' %(signal_name, name_tab)
            print >> f, 'rate                             %.2f\t\t\t\t\t %.2f                  %.2f                  %.2f      ' %(signal_rate, conversion_rate, nonprompt_rate, diboson_rate)
            print >> f, '-------------------------------------------------------------------------------------------------------------------'
            print >> f, 'lumi                     lnN     1.026                   1.026                   -                       1.026     '
            print >> f, 'norm_conv                lnN     -                       1.1                     -                       -         '
            print >> f, 'norm_fr_%s              lnN     -                       -                       1.2                     -         ' %disb_bin
            print >> f, 'norm_vv                  lnN     -                       -                       -                       1.1       '
            print >> f, 'norm_sig                 lnN     1.2                     -                       -                       -         '
            print >> f, 'xs_%s%s lnN     1.05                    -                       -                       -         ' % (signal_name, xs_tab)
        if diboson_rate == 0:
            print >> f, 'bin                              %s                     %s                     %s                     ' %(disp_bin, disp_bin, disp_bin)
            print >> f, 'process                          0                       1                       2                       '
            print >> f, 'process                          %s%s conversions             non-prompt              ' %(signal_name, name_tab)
            print >> f, 'rate                             %.2f\t\t\t\t\t %.2f                  %.2f                  ' %(signal_rate, conversion_rate, nonprompt_rate)
            print >> f, '-------------------------------------------------------------------------------------------------------------------'
            print >> f, 'lumi                     lnN     1.026                   1.026                   -                       '
            print >> f, 'norm_conv                lnN     -                       1.1                     -                       '
            print >> f, 'norm_fr_%s              lnN     -                       -                       1.2                     ' %disp_bin
            print >> f, 'norm_vv                  lnN     -                       -                       -                       '
            print >> f, 'norm_sig                 lnN     1.2                     -                       -                       '
            print >> f, 'xs_%s%s lnN     1.05                    -                       -                       ' % (signal_name, xs_tab)
        print >> f, '%s autoMCStats 0                                                                                          ' %disp_bin 
 
    f.close()

def combine_datacards(signals):
    print 'combining datacards'
    for s in signals: 
        system( 'combineCards.py Disp1_0p5=hnl_%s_Disp1_0p5_mu_dc.txt Disp2_0p5_10=hnl_%s_Disp2_0p5_10_mu_dc.txt Disp3_10=hnl_%s_Disp3_10_mu_dc.txt > hnl_%s_mu_dc.txt' %(s, s, s, s) )
    system( 'mkdir split; mv *Disp* split/' )

def make_inputs(verbose=False, has_signals=True):

    out_folder = '/t3home/vstampf/eos/plots/limits/outputs/sr_2d_binned/'
    fout = rt.TFile.Open(out_folder + 'hnl_mll_combine_input_aug_15.root', 'recreate')
    # in_folder = '/t3home/vstampf/eos/plots/limits/inputs/'
    in_folders = glob('/work/dezhu/3_figures/1_DataMC/FinalStates/mmm/SR_v9_*/root/linear/')
    # fin = rt.TFile(in_folder + 'hnl_m_12_sr_jul_30.root')
    # fin = rt.TFile(in_folder + 'hnl_m_12_sr_sidebands_aug_14.root')
    disp_bins = []
    for in_folder in in_folders:
        disp_bin = sub('.*SR_v9_', '', sub('/root.*','', in_folder))
        disp_bins.append(disp_bin)
        if verbose: print disp_bin
        fin = rt.TFile(in_folder + 'hnl_m_12_money.root')

        can = fin.Get('can')
        pad = can.GetPrimitive('can_1')

        h_list = pad.GetListOfPrimitives()

        h_dict = OrderedDict()

        for h in h_list[:len(h_list)/2+1]:
            h_name = h.GetName()
            if 'HN3L' in h_name: 
                h_name = sub('.*HN3L_M_', 'M', h_name)
                h_name = sub('_V_0', '_V', h_name)
                h_name = sub('_mu_massiveAndCKM_LO', '', h_name)
                h.SetName(h_name)
                h_dict[h_name] = h
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

        if not has_signals: 
            h_dict['dummy_sig'] = dc(h_dict['conversions'])
            h_dict['dummy_sig'].SetName('dummy_sig')
            # h_dict['dummy_sig'].Scale(0.01/h_dict['dummy_sig'].Integral())

        if not h_dict.has_key('data_obs'):
            h_dict['data_obs'] = dc(h_dict['conversions'])
            h_dict['data_obs'].SetName('data_obs')
            
        fout.cd()
        rt.gDirectory.mkdir('%s' %disp_bin)
        fout.cd('%s' %disp_bin)

        for h in h_dict.keys():
            if verbose: print 'writing', h
            h_dict[h].Write()

        if has_signals:     signals = [sig for sig in h_dict if 'Vp' in sig]
        if not has_signals:  signals = ['dummy_sig']
        # clone bkg to data for blind limits
        j =1
        for s in signals:
            if verbose: print s, j
            j+=1
            printDataCard(signal_name     = s,
                          signal_rate     = h_dict[s            ].Integral() if h_dict.has_key(s            ) else 0,
                          conversion_rate = h_dict['conversions'].Integral() if h_dict.has_key('conversions') else 0,
                          nonprompt_rate  = h_dict['non-prompt' ].Integral() if h_dict.has_key('non-prompt' ) else 0,
                          diboson_rate    = h_dict['VV'         ].Integral() if h_dict.has_key('VV'         ) else 0,
                          observed_rate   = h_dict['data_obs'   ].Integral() if h_dict.has_key('data_obs'   ) else 0,
                          disp_bin        = disp_bin)

    fout.Close()
    
    combine_datacards(signals)

    

signals_mu = [
'M2_Vp00244948974278',
'M2_Vp00282842712475',
'M2_Vp00316227766017',
'M2_Vp004472135955',
'M2_Vp00547722557505',
'M2_Vp00707106781187',
'M2_Vp00836660026534',
'M2_Vp0141421356237',
'M2_Vp0173205080757',
'M2_Vp01',
'M2_Vp022360679775',
'M5_Vp00244948974278',
'M5_Vp00282842712475',
'M5_Vp00316227766017',
'M5_Vp004472135955',
'M5_Vp00547722557505',
'M5_Vp00707106781187',
'M5_Vp00836660026534',
'M5_Vp01',
'M8_Vp00244948974278',
'M8_Vp00282842712475',
'M8_Vp00316227766017',
'M8_Vp004472135955',
'M8_Vp00547722557505',]

'''
M2_0.00244948974278_e
M2_0.00282842712475_e
M2_0.00316227766017_e
M2_0.004472135955_e
M2_0.00547722557505_e
M2_0.00707106781187_e
M2_0.00836660026534_e
M2_0.0141421356237_e
M2_0.0173205080757_e
M2_0.01_e
M2_0.022360679775_e
M5_0.00244948974278_e
M5_0.00282842712475_e
M5_0.00316227766017_e
M5_0.004472135955_e
M5_0.00547722557505_e
M5_0.00707106781187_e
M5_0.00836660026534_e
M5_0.01_e
M8_0.00244948974278_e
M8_0.00282842712475_e
M8_0.00316227766017_e
M8_0.004472135955_e
M8_0.00547722557505_e

M2_0.00244948974278_tau
M2_0.00282842712475_tau
M2_0.00316227766017_tau
M2_0.004472135955_tau
M2_0.00547722557505_tau
M2_0.00707106781187_tau
M2_0.00836660026534_tau
M2_0.0141421356237_tau
M2_0.0173205080757_tau
M2_0.01_tau
M2_0.022360679775_tau
M3_0.00244948974278_tau
M3_0.00282842712475_tau
M3_0.00316227766017_tau
M3_0.004472135955_tau
M3_0.00547722557505_tau
M3_0.00707106781187_tau
M3_0.00836660026534_tau
M3_0.0141421356237_tau
M3_0.0173205080757_tau
M3_0.01_tau
M5_0.00244948974278_tau
M5_0.00547722557505_tau
M5_0.00836660026534_tau
M5_0.01_tau
M8_0.00244948974278_tau
M8_0.00282842712475_tau
M8_0.00316227766017_tau
M8_0.004472135955_tau
M8_0.00547722557505_tau
'''
