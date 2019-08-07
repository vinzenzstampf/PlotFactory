import ROOT as rt
from collections import OrderedDict
from re import sub


def printDataCard(signal_name, signal_rate, conversion_rate, nonprompt_rate, diboson_rate, observed_rate): 

    with open('hnl_%s_mu_dc.txt' %signal_name, 'w') as f:

        signal_name = sub('_e',   '', signal_name)
        signal_name = sub('_mu',  '', signal_name)
        signal_name = sub('_tau', '', signal_name)

        name_tab = '\t'; xs_tab = '\t'
        if len(signal_name) < len('M2_Vp00244948974278'):   name_tab = '\t\t'
        if len(signal_name) < len('M2_Vp00244948974278')-2:
            name_tab = '\t\t\t\t'
            xs_tab   = name_tab


        print >> f, 'max     1     number of categories'
        print >> f, 'jmax    3     number of samples minus one'
        print >> f, 'kmax    *     number of nuisance parameters'
        print >> f, '---------------------------------------------------------'
        print >> f, 'shapes *   * hnl_mll_combine_input.root $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC'
        print >> f, '---------------------------------------------------------'
        print >> f, 'bin                              mll'
        print >> f, 'observation                      %d' %observed_rate
        print >> f, '-----------------------------------------------------------------'
        print >> f, 'bin                              mll                     mll                     mll                     mll       '
        print >> f, 'process                          0                       1                       2                       3         '
        print >> f, 'process                          %s%s conversions             non-prompt              VV' %(signal_name, name_tab)
        print >> f, 'rate                             %.2f\t\t\t\t\t %.2f                  %.2f                  %.2f      ' %(signal_rate, conversion_rate, nonprompt_rate, diboson_rate)
        print >> f, '-------------------------------------------------------------------------------------------------------------------'
        print >> f, 'lumi                     lnN     1.026                   1.026                   1.026                   1.026     '
        print >> f, 'norm                     lnN     1.2                     1.2                     1.2                     1.2       '
        print >> f, 'fakerate                 lnN     -                       -                       1.1                     -         '
        print >> f, 'xs_VV                    lnN     -                       -                       1.10                    1.10      '
        print >> f, 'xs_conversions           lnN     -                       1.010                   -                       -         '
        print >> f, 'xs_%s%s lnN     1.05                    -                       -                       -         ' % (signal_name, xs_tab)
        print >> f, 'mll autoMCStats 0                                                                                          ' 
 
    f.close()


def make_inputs(verbose=False):

    in_folder = '/t3home/vstampf/eos/plots/limits/inputs/'
    fin = rt.TFile(in_folder + 'hnl_m_12_sr_jul_30.root')

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
        if 'Conversions' in h_name: h_name = 'conversions'
        if 'WW' in h_name:          h_name = 'VV'
        if 'nonprompt' in h_name:   h_name = 'non-prompt'
        h.SetName(h_name)
        h_dict[h_name] = h

    fout = rt.TFile.Open(in_folder + 'hnl_mll_combine_input.root', 'recreate')
    fout.cd()
    rt.gDirectory.mkdir('mll')
    fout.cd('mll')

    for h in h_dict.keys():
        if verbose: print 'writing', h
        h_dict[h].Write()

    fout.Close()

    signals = [sig for sig in h_dict if 'Vp' in sig]
    j =1
    for s in signals:
        if verbose: print s, j
        j+=1
        try:    observed_rate = h_dict['data_obs'].Integral()
        except: observed_rate = 100
        printDataCard(signal_name     = s,
                      signal_rate     = h_dict[s            ].Integral(),
                      conversion_rate = h_dict['conversions'].Integral(),
                      nonprompt_rate  = h_dict['non-prompt' ].Integral(),
                      diboson_rate    = h_dict['VV'         ].Integral(),
                      observed_rate   = observed_rate)
    

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
