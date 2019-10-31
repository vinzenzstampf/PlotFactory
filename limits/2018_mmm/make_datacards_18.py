'''
#########################################################################
## FIRST SOURCE THE COMPATIBLE ROOT VERSION:                           ##
## (FOR RUNNING THE DIFFERENT SCRIPTS ALWAYS START A NEW BASH SESSION) ##
## cd /t3home/vstampf/CMSSW_9_4_6_patch1/src/; cmsenv; cd -            ##
## IN ORDER TO RUN THIS CODE EXECUTE:                                  ## 
##    ipython -i make_datacards.py; DC = dataCards(CH)                 ##
## WHERE CH = 'mmm', eem_OS', ...                                      ##
## THEN EXCUTE DC.make_inputs() TO GENERATE THE DC                     ##
#########################################################################
'''
import ROOT as rt
from collections import OrderedDict
from re import sub
from pdb import set_trace
from copy import deepcopy as dc
from glob import glob
from os import system


class dataCards(object):

    def __init__(self, ch):
        self.ch = ch
        self.out_folder = '/t3home/vstampf/eos/plots/limits/inputs/mmm18_29Oct/'
        # self.in_folder  = '/work/dezhu/3_figures/1_DataMC/FinalStates/2018/%s/datacard/root/linear/' %self.ch
        self.in_folder  = '/work/dezhu/3_figures/1_DataMC/FinalStates/2018/%s/datacard/datacards/' %self.ch
    

    def printDataCards(self, signal_name):
        '''
            #############################
            ## writes the DC for       ##
            ## combine for all signals ##
            #############################
        '''

        with open(self.out_folder + 'hnl_%s_dc_%s.txt' %(signal_name, self.ch), 'w') as f:

            if self.verbose: print self.out_folder+ 'hnl_%s_dc_%s.txt' %(signal_name, self.ch)

            disp_bins = self.disp_bins
            ch = self.ch

            rate_obs_d1,  rate_obs_d2,  rate_obs_d3  = self.rates[disp_bins[0]]['obs' ].Integral(), self.rates[disp_bins[1]]['obs' ].Integral(), self.rates[disp_bins[2]]['obs' ].Integral()
            rate_conv_d1, rate_conv_d2, rate_conv_d3 = self.rates[disp_bins[0]]['conv'].Integral(), self.rates[disp_bins[1]]['conv'].Integral(), self.rates[disp_bins[2]]['conv'].Integral()
            rate_fake_d1, rate_fake_d2, rate_fake_d3 = self.rates[disp_bins[0]]['fake'].Integral(), self.rates[disp_bins[1]]['fake'].Integral(), self.rates[disp_bins[2]]['fake'].Integral()
            rate_sig_d1,  rate_sig_d2,  rate_sig_d3  = self.rates[disp_bins[0]][signal_name ].Integral(), self.rates[disp_bins[1]][signal_name ].Integral(), self.rates[disp_bins[2]][signal_name ].Integral()

            if self.vv: rate_vv_d1,   rate_vv_d2,   rate_vv_d3  = self.rates[disp_bins[0]]['vv' ].Integral(), self.rates[disp_bins[2]]['vv' ].Integral(), self.rates[disp_bins[2]]['vv' ].Integral()

            name_tab = '\t'; xs_tab = '\t'
            if len(signal_name) < len('M2_Vp00244948974278'):   name_tab = '\t\t'
            if len(signal_name) < len('M2_Vp00244948974278')-2:
                name_tab = '\t\t\t\t'
                xs_tab   = name_tab

            n_samples = 3
            if not self.vv: n_samples = 2 

            print >> f, 'max     3     number of categories'
            print >> f, 'jmax    %s     number of samples minus one' %n_samples
            print >> f, 'kmax    *     number of nuisance parameters'
            print >> f, '---------------------------------------------------------'
            print >> f, 'shapes *             %s   \thnl_mll_combine_input_%s.root %s/$PROCESS %s/$PROCESS_$SYSTEMATIC' %(disp_bins[0], ch, disp_bins[0], disp_bins[0])  
            print >> f, 'shapes *             %s   \t\thnl_mll_combine_input_%s.root %s/$PROCESS %s/$PROCESS_$SYSTEMATIC' %(disp_bins[1], ch, disp_bins[1], disp_bins[1])  
            print >> f, 'shapes *             %s   hnl_mll_combine_input_%s.root %s/$PROCESS %s/$PROCESS_$SYSTEMATIC' %(disp_bins[2], ch, disp_bins[2], disp_bins[2])  
            print >> f, '---------------------------------------------------------'
            print >> f, 'bin                  %s\t\t\t %s\t\t\t %s\t\t\t' %(disp_bins[0], disp_bins[1], disp_bins[2])
            print >> f, 'observation          %.2f \t\t\t%.2f \t\t\t     %.2f '%(rate_obs_d1, rate_obs_d2, rate_obs_d3)
            print >> f, '-----------------------------------------------------------------'
            if self.vv:
                print >> f, 'bin                              %s\t\t\t %s\t\t\t %s\t\t\t %s\t\t\t '\
                                                             '%s\t\t\t %s\t\t\t %s\t\t\t %s\t\t\t '\
                                                             '%s\t\t\t %s\t\t\t %s\t\t\t %s\t\t\t' %(disp_bins[0], disp_bins[0], disp_bins[0], disp_bins[0], disp_bins[1], disp_bins[1], disp_bins[1], disp_bins[1], 
                                                                                                     disp_bins[2], disp_bins[2], disp_bins[2], disp_bins[2])
                print >> f, 'process                          0                       1                       2                       3                       '\
                                                             '0                       1                       2                       3                       '\
                                                             '0                       1                       2                       3                       '
                print >> f, 'process                          %s%s conversions             non-prompt              VV                      '\
                                                             '%s%s conversions             non-prompt              VV                      '\
                                                             '%s%s conversions             non-prompt              VV' %(signal_name, name_tab, signal_name, name_tab, signal_name, name_tab)
                print >> f, 'rate                             %.2f\t\t\t\t\t %.2f\t\t\t\t\t %.2f\t\t\t\t\t  %.2f\t\t\t\t\t '\
                                                             '%.2f\t\t\t\t\t %.2f\t\t\t\t\t %.2f\t\t\t\t\t  %.2f\t\t\t\t\t '\
                                                             '%.2f\t\t\t\t\t %.2f\t\t\t\t\t %.2f\t\t\t\t\t %.2f' %(rate_sig_d1, rate_conv_d1, rate_fake_d1, rate_vv_d1, 
                                                                                                                   rate_sig_d2, rate_conv_d2, rate_fake_d2, rate_vv_d2, rate_sig_d3, rate_conv_d3, rate_fake_d3, rate_vv_d3)
                print >> f, '-------------------------------------------------------------------------------------------------------------------'
                print >> f, 'lumi                     lnN     1.026                   1.026                   -                       1.026                   '\
                                                             '1.026                   1.026                   -                       1.026                   '\
                                                             '1.026                   1.026                   -                       1.026                   '
                print >> f, 'norm_conv                lnN     -                       1.1                     -                       -                       '\
                                                             '-                       1.1                     -                       -                       '\
                                                             '-                       1.1                     -                       -                       '
                print >> f, 'norm_vv                  lnN     -                       -                       -                       1.1                     '\
                                                             '-                       -                       -                       1.1                     '\
                                                             '-                       -                       -                       1.1                     '
                print >> f, 'norm_fr_d1               lnN     -                       -                       1.2                     -                       '\
                                                             '-                       -                       -                       -                       '\
                                                             '-                       -                       -                       -                       '
                print >> f, 'norm_fr_d2               lnN     -                       -                       -                       -                       '\
                                                             '-                       -                       1.2                     -                       '\
                                                             '-                       -                       -                       -                       '
                print >> f, 'norm_fr_d3               lnN     -                       -                       -                       -                       '\
                                                             '-                       -                       -                       -                       '\
                                                             '-                       -                       1.2                     -                       '
                print >> f, 'norm_sig                 lnN     1.2                     -                       -                       -                       '\
                                                             '1.2                     -                       -                       -                       '\
                                                             '1.2                     -                       -                       -                       '
            if not self.vv:
                print >> f, 'bin                              %s\t\t\t %s\t\t\t %s\t\t\t '\
                                                             '%s\t\t\t %s\t\t\t %s\t\t\t '\
                                                             '%s\t\t\t %s\t\t\t %s\t\t\t' %(disp_bins[0], disp_bins[0], disp_bins[0], disp_bins[1], disp_bins[1], disp_bins[1], disp_bins[2], disp_bins[2], disp_bins[2])  
                print >> f, 'process                          0                       1                       2                       '\
                                                             '0                       1                       2                       '\
                                                             '0                       1                       2       '
                print >> f, 'process                          %s%s conversions             non-prompt              '\
                                                             '%s%s conversions             non-prompt              '\
                                                             '%s%s conversions             non-prompt' %(signal_name, name_tab, signal_name, name_tab, signal_name, name_tab)
                print >> f, 'rate                             %.2f\t\t\t\t\t %.2f\t\t\t\t\t %.2f\t\t\t\t\t '\
                                                             '%.2f\t\t\t\t\t %.2f\t\t\t\t\t %.2f\t\t\t\t\t '\
                                                             '%.2f\t\t\t\t\t %.2f\t\t\t\t\t %.2f' %(rate_sig_d1, rate_conv_d1, rate_fake_d1, rate_sig_d2, rate_conv_d2, rate_fake_d2, rate_sig_d3, rate_conv_d3, rate_fake_d3)
                print >> f, '-------------------------------------------------------------------------------------------------------------------'
                print >> f, 'lumi                     lnN     1.026                   1.026                   -                       '\
                                                             '1.026                   1.026                   -                       '\
                                                             '1.026                   1.026                   -                       '
                print >> f, 'norm_conv                lnN     -                       1.1                     -                       '\
                                                             '-                       1.1                     -                       '\
                                                             '-                       1.1                     -                       '
                print >> f, 'norm_fr_d1               lnN     -                       -                       1.2                     '\
                                                             '-                       -                       -                       '\
                                                             '-                       -                       -                       '
                print >> f, 'norm_fr_d2               lnN     -                       -                       -                       '\
                                                             '-                       -                       1.2                     '\
                                                             '-                       -                       -                       '
                print >> f, 'norm_fr_d3               lnN     -                       -                       -                       '\
                                                             '-                       -                       -                       '\
                                                             '-                       -                       1.2                     '
                print >> f, 'norm_sig                 lnN     1.2                     -                       -                       '\
                                                             '1.2                     -                       -                       '\
                                                             '1.2                     -                       -                       '
            print >> f, disp_bins[0] + ' autoMCStats 0 0 1'
            print >> f, disp_bins[1] + ' autoMCStats 0 0 1'
            print >> f, disp_bins[2] + ' autoMCStats 0 0 1'
     
        f.close()


    def make_inputs(self, verbose=False, has_signals=True):
        '''
            ###########################################
            ## reads inputs from the hnl_m_12 stack  ##
            ## and creates the DC + input root files ##
            ###########################################
        '''
        self.vv = False
        self.verbose = verbose

        rates = OrderedDict()

        fout = rt.TFile.Open(self.out_folder + 'hnl_mll_combine_input_%s.root' %self.ch, 'recreate')

        self.disp_bins = []
        in_files = glob(self.in_folder + '*disp*.datacard.root')
        for in_file in in_files:
            disp_bin = sub('.*disp', 'disp', in_file)
            disp_bin = sub('\.datacard\.root','', disp_bin)
            self.disp_bins.append(disp_bin)
            if verbose: print disp_bin
            f_in = rt.TFile(in_file)

            h_dict = OrderedDict()

            h_list = f_in.GetListOfKeys()

            for h in h_list:
                h_name = h.GetName()
                if '_Vp' in h_name: 
                    h_dict[h_name] = f_in.Get(h_name)
                elif 'data' in h_name: 
                    h_dict['data_obs'] = f_in.Get(h_name)
                else: continue

            stack = f_in.Get('hnl_m_12_money_%s_stack' %disp_bin)

            for h in stack.GetHists():
                h_name = h.GetName()
                if 'Conversions_DY' in h_name: h_name = 'conversions' #
                if 'WW'             in h_name: h_name = 'VV'
                if 'nonprompt'      in h_name: h_name = 'non-prompt'
                h.SetName(h_name)
                h.SetLineColor(rt.kBlue+2)
                h.SetMarkerColor(rt.kBlue+2)
                # fail-save for negative integrals of prompt bkg
                #TODO this should be fixed in the plotting tool at some point?
                if h.Integral() < 0: 
                    h.Scale(0.001/h.Integral())
                    print 'WARNING: negative integral for conv in', disp_bin
                
                h_dict[h_name] = h

            if not has_signals: 
                h_dict['dummy_sig'] = h_dict['conversions']
                h_dict['dummy_sig'].SetName('dummy_sig')
                # h_dict['dummy_sig'].Scale(0.01/h_dict['dummy_sig'].Integral())

            # clone prompt bkg to data for blind limits
            if not h_dict.has_key('data_obs'):
                h_dict['data_obs'] = dc(h_dict['conversions'])
                h_dict['data_obs'].SetName('data_obs')

            # make root file with combine-readable structure
            fout.cd()
            rt.gDirectory.mkdir('%s' %disp_bin)
            fout.cd('%s' %disp_bin)

            for h in h_dict.keys():
                if verbose: print 'writing', h
                h_dict[h].Write()

            if has_signals:     signals = [sig for sig in h_dict if 'Vp' in sig]
            if not has_signals:  signals = ['dummy_sig']

            rates[disp_bin] = OrderedDict()

            signals = [h for h in h_dict if 'Vp' in h]

            # set_trace()
            for s in signals:
                if 'maj' in s:      rates[disp_bin][s]  = h_dict[s]
                # elif 'dir_cc' in s: rates[disp_bin][s]  = h_dict[s]
                # rates[disp_bin][s]  = h_dict[s]
            rates[disp_bin]['conv']  = h_dict['conversions']
            rates[disp_bin]['fake']  = h_dict['non-prompt' ]
            # set_trace()
       
        fout.Close()
        signals = [sig for sig in rates[disp_bin] if 'Vp' in sig]
        self.rates = rates
        # set_trace()
        for i, s in enumerate(signals):
            print i, s
            if i == 11: continue
            print rates[self.disp_bins[0]][s].Integral()#, rates[disp_bins[1][s], rates[disp_bins[2][s]
            # self.printDataCards(s)

