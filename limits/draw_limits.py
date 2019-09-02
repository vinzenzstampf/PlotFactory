'''
############################################################################
## IN ORDER TO RUN THIS CODE EXECUTE:                                     ##
source /cvmfs/sft.cern.ch/lcg/views/LCG_94/x86_64-slc6-gcc8-opt/setup.sh  ##
(THE shapely.geometry PACKAGE NEEDS A NEWER python VERSION)               ##
THEN RUN: draw_signals(CH), WHERE CH = 'mmm', eem_OS', ...                ##
############################################################################
'''
#!/usr/bin/env python
from os import environ as env
import ROOT as rt
from collections import OrderedDict
import re
from pdb import set_trace
import numpy as np
from matplotlib import pyplot as plt
import shapely.geometry as sg


def get_signals(verbose=False):
    '''
    ###################################
    ## preparing a signal dictionary ##
    ###################################
    '''
    # read input
    with open('signals.py', 'r') as f_in:
        array = []
        for line in f_in:
            array.append(line)

    # reorganize lines to each signal
    signals = OrderedDict()

    Mass = None; V = None
    for line in array:

        mode = None
        if '_e_' in line:   mode = 'e'  
        if '_mu_' in line:  mode = 'mu' 
        if '_tau_' in line: mode = 'tau'

        line = line.strip()
        line = re.sub('HN3L_', '', line)
        line = re.sub('_massiveAndCKM_LO', '', line)
        line = re.sub('_', '', line)
        line = re.sub(' ', '', line)
        if line == '': continue

        mass = re.sub('V.*', '', line)
        mass = re.sub('M', '', mass)
        v    = re.sub('e.*', '', line)
        v    = re.sub('mu.*', '', v)
        v    = re.sub('tau.*', '', v)
        v    = re.sub('p', '.', v)
        v    = re.sub('M.V', '', v)
        Mass, V = mass, v

        try: signals['M' + Mass + '_V' + V + '_' + mode]['mass'] = float(mass)
        except:
            signals['M' + Mass + '_V' + V + '_' + mode] = OrderedDict()
            signals['M' + Mass + '_V' + V + '_' + mode]['mass'] = float(mass)

        V2 = None
        if 'v2=' in line: 
            V2 = re.sub('.*v2=', '', line) 
            signals['M' + Mass + '_V' + V + '_' + mode]['V2'] = float(V2)
            if verbose: print V2, float(V)**2

        xsec = None
        if 'xs=' in line: 
            xsec = re.sub('.*xs=', '', line) 
            signals['M' + Mass + '_V' + V + '_' + mode]['xsec'] = float(xsec)

        xsec_err = None
        if 'xse=' in line: 
            xsec_err = re.sub('.*xse=', '', line) 
            signals['M' + Mass + '_V' + V + '_' + mode]['xsec_err'] = float(xsec_err)

    if verbose:
        for k in signals.keys(): 
            for kk in signals[k].keys():
                print k, kk, signals[k][kk]
            print '\n'

    return signals

def get_lim_dict(ch='mem', verbose=False):
    '''
    #####################################
    ## preparing the limits dictionary ##
    ## from the output file that       ##
    ## combine is piped to             ## 
    #####################################
    '''
    in_file = '/t3home/vstampf/eos/plots/limits/inputs/data_cards_aug_20/limits_aug_20_%s.txt' %ch
    env['LIM_FILE']   = in_file
    env['OUT_FOLDER'] = '/t3home/vstampf/eos/plots/limits/outputs/'
    with open(in_file, 'r') as f_in:
        array = []
        for line in f_in:
            array.append(line)

    # reorganize lines to each signal
    lim_dict = OrderedDict()

    Mass = None; V = None; mode = None
    for line in array:
        line = line.strip()
        if line == '': continue

        mode = 'mu'
        if '_e_' in line:   mode = 'e'  
        if '_mu_' in line:  mode = 'mu' 
        if '_tau_' in line: mode = 'tau'

        if 'hnl' in line:
            mass = re.sub(r'.*M([0-9])_V.*',r'\1', line)
            v    = re.sub('.*Vp', '', line)
            v    = re.sub('_.*', '', v)
            v    = '0.' + v
            Mass, V = mass, v

        try:
            lim_dict['M' + Mass + '_V' + V + '_' + mode]['mass'] = float(Mass)
            lim_dict['M' + Mass + '_V' + V + '_' + mode]['V']    = float(V)
        except: 
            lim_dict['M' + Mass + '_V' + V + '_' + mode] = OrderedDict()

        ep1s = None; ep2s = None; em1s = None; em2s = None; om1s = None; op1s = None; obs = None; exp = None
        line = re.sub(' ', '', line)

        if 'Observed'     in line: 
            obs  = re.sub('.*r<', '', line) 
            lim_dict['M' + Mass + '_V' + V + '_' + mode]['obs']  = float(obs)

        if 'Expected2.5'  in line:                           
            em2s = re.sub('.*r<', '', line)                  
            lim_dict['M' + Mass + '_V' + V + '_' + mode]['em2s'] = float(em2s)

        if 'Expected16.0' in line:                           
            em1s = re.sub('.*r<', '', line)                  
            lim_dict['M' + Mass + '_V' + V + '_' + mode]['em1s'] = float(em1s)

        if 'Expected50.0' in line:                           
            exp  = re.sub('.*r<', '', line)                  
            lim_dict['M' + Mass + '_V' + V + '_' + mode]['exp']  = float(exp)

        if 'Expected84.0' in line:                           
            ep1s = re.sub('.*r<', '', line)                  
            lim_dict['M' + Mass + '_V' + V + '_' + mode]['ep1s'] = float(ep1s)

        if 'Expected97.5' in line:                           
            ep2s = re.sub('.*r<', '', line)                 
            lim_dict['M' + Mass + '_V' + V + '_' + mode]['ep2s'] = float(ep2s)

    if verbose:
        for k in lim_dict.keys(): 
            for kk in lim_dict[k].keys():
                print k, kk, lim_dict[k][kk]
            print '\n'

    return lim_dict


def draw_limits(ch='mmm', twoD=False, verbose=False): 
    '''
    #############################################################################
    ## producing coupling vs r limits for each signal mass and a given channel ##
    ## also has the option 2D, in order to draw mass vs coupling limits (this  ## 
    ## uses the intersections of the 1D limits and the r=1 line)               ##
    #############################################################################
    '''
    # create signal and limits dictionary
    limits  = get_lim_dict(ch)
    signals = get_signals()

    b     = np.arange(0., 11, 1)
    req1  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    ixs = OrderedDict()
    for m in [2, 5, 8]:
        ixs['M%d' %m] = OrderedDict()

        plt.clf()
        plt.cla()

        y_exp = []; y_ep1s = []; y_ep2s = []; y_em2s = []; y_em1s = [];  
        v2s   = [lim for lim in limits if 'M%d' %m in lim]
        b_V2  = []

        for v2 in v2s:
            if limits[v2].has_key('exp'):  
                y_exp .append(limits [v2]['exp' ]) 
                y_ep1s.append(limits [v2]['ep1s']) 
                y_ep2s.append(limits [v2]['ep2s']) 
                y_em1s.append(limits [v2]['em1s']) 
                y_em2s.append(limits [v2]['em2s']) 
                b_V2  .append(signals[v2]['V2']) 

        x_err = np.zeros(len(b_V2))
        b_V2.sort(reverse=False)

        for i in range(len(y_exp)):
            y_ep1s[i] = abs(y_ep1s[i] - y_exp[i]) 
            y_ep2s[i] = abs(y_ep2s[i] - y_exp[i]) 
            y_em1s[i] = abs(y_em1s[i] - y_exp[i]) 
            y_em2s[i] = abs(y_em2s[i] - y_exp[i]) 
            
        exp = rt.TGraph           (len(b_V2), np.array(b_V2), np.array(y_exp))
        gr1 = rt.TGraphAsymmErrors(len(b_V2), np.array(b_V2), np.array(y_exp), np.array(x_err), np.array(x_err), np.array(y_em1s), np.array(y_ep1s))
        gr2 = rt.TGraphAsymmErrors(len(b_V2), np.array(b_V2), np.array(y_exp), np.array(x_err), np.array(x_err), np.array(y_em2s), np.array(y_ep2s))
            
        
        plt.plot(b_V2, y_exp,  'k--', label = 'exp')
        plt.plot(b_V2, y_ep1s, 'g--', label = 'ep1s')
        plt.plot(b_V2, y_em1s, 'g--', label = 'em1s')
        plt.plot(b_V2, y_ep2s, 'y--', label = 'ep2s')
        plt.plot(b_V2, y_em2s, 'y--', label = 'em2s')

        plt.rc('text', usetex=True)

        rt.gStyle.SetOptStat(0000)
        B_V2 = np.logspace(-6, -3, 10, base=10)
        B_Y  = np.logspace(-2, 4, 10, base=10)
        r1g = rt.TGraph           (len(B_V2), np.array(B_V2), np.ones(len(B_V2)))
        r1g.SetLineColor(rt.kRed+1); r1g.SetLineWidth(1)
        framer = rt.TH2F('framer', 'framer', len(B_V2)-1, B_V2, len(B_Y)-1, B_Y)
        framer.GetYaxis().SetRangeUser(0.01,10000)
        framer.GetXaxis().SetRangeUser(0.000001, 0.001)
        framer.GetXaxis().SetTitleOffset(1.8)

        plt.plot(b,  req1, 'r-')
        if ch == 'mmm': 
            plt.title(r'$M_N = %d \, GeV,\; \mu\mu\mu$' %m)
            framer.SetTitle('m_{N} = %d GeV,  #mu#mu#mu; |V_{#mu N}|^{2}; r' %m)
        if ch == 'eee':
            plt.title(r'$M_N = %d \, GeV,\; eee$' %m)
            framer.SetTitle('m_{N} = %d GeV,  eee; |V_{e N}|^{2}; r' %m)
        if ch == 'mem_OS':
            plt.title(r'$M_N = %d \, GeV,\; \mu\mu e OS$' %m)
            framer.SetTitle('m_{N} = %d GeV,  #mu#mue; |V_{#mu N}|^{2}; r' %m)
        if ch == 'mem_SS':
            plt.title(r'$M_N = %d \, GeV,\; \mu\mu e SS$' %m)
            framer.SetTitle('m_{N} = %d GeV,  #mu#mue; |V_{#mu N}|^{2}; r' %m)
        if ch == 'eem_OS':
            plt.title(r'$M_N = %d \, GeV,\; ee \mu OS$' %m)
            framer.SetTitle('m_{N} = %d GeV,  ee#mu OS; |V_{e N}|^{2}; r' %m)
        if ch == 'eem_SS':
            plt.title(r'$M_N = %d \, GeV,\; ee \mu SS$' %m)
            framer.SetTitle('m_{N} = %d GeV,  ee#mu SS; |V_{e N}|^{2}; r' %m)
        if 'mem' in ch or ch == 'mmm': plt.xlabel(r'${|V_{\mu N}|}^2$')
        if 'eem' in ch or ch == 'eee': plt.xlabel(r'${|V_{e N}|}^2$')
        plt.rc('font', family='serif')
        plt.axis([1e-06, 0.001, 0.1, 50000])
        plt.ylabel('r')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='lower left')

        gr2.SetFillColor(rt.kYellow)
        gr1.SetFillColor(rt.kGreen)

        can = rt.TCanvas('limits', 'limits')
        can.cd(); can.SetLogy(); can.SetLogx()
        can.SetBottomMargin(0.15)
        framer.Draw()
        can.Modified(); can.Update()
        gr2.Draw('3same')
        can.Modified(); can.Update()
        gr1.Draw('3same')
        can.Modified(); can.Update()
        exp.Draw('same')
        can.Modified(); can.Update()
        r1g.Draw('same')
        can.Modified(); can.Update()
        can.SaveAs('/t3home/vstampf/eos/plots/limits/outputs/mmm_M%d_20Aug_%s_root.pdf' %(m, ch))
        can.SaveAs('/t3home/vstampf/eos/plots/limits/outputs/mmm_M%d_20Aug_%s_root.png' %(m, ch))
        can.SaveAs('/t3home/vstampf/eos/plots/limits/outputs/mmm_M%d_20Aug_%s_root.root' %(m, ch))

        r1   = sg.LineString([(min(b_V2), 1), (max(b_V2), 1)])

        exp  = sg.LineString(list(zip(b_V2, y_exp)))
        ep1s = sg.LineString(list(zip(b_V2, y_ep1s)))
        em1s = sg.LineString(list(zip(b_V2, y_em1s)))
        ep2s = sg.LineString(list(zip(b_V2, y_ep2s)))
        em2s = sg.LineString(list(zip(b_V2, y_em2s)))

        int_exp  = np.array(exp .intersection(r1)); int_exp  = int_exp .flatten(); int_exp  = int_exp .tolist()  #ints.append(int_exp)
        int_ep1s = np.array(ep1s.intersection(r1)); int_ep1s = int_ep1s.flatten(); int_ep1s = int_ep1s.tolist() #ints.append(int_ep1s)
        int_em1s = np.array(em1s.intersection(r1)); int_em1s = int_em1s.flatten(); int_em1s = int_em1s.tolist() #ints.append(int_em1s)
        int_ep2s = np.array(ep2s.intersection(r1)); int_ep2s = int_ep2s.flatten(); int_ep2s = int_ep2s.tolist() #ints.append(int_ep2s)
        int_em2s = np.array(em2s.intersection(r1)); int_em2s = int_em2s.flatten(); int_em2s = int_em2s.tolist() #ints.append(int_em2s)

        for lim in ['exp', 'ep1s', 'ep2s', 'em1s', 'em2s']: ixs['M%d' %m][lim] = []
        for i in int_exp:
            if not i == 1.0:
               ixs['M%d' %m]['exp'].append(i)
               break
        for i in int_em1s:
            if not i == 1.0:
               ixs['M%d' %m]['em1s'].append(i)
               break
        for i in int_em2s:
            if not i == 1.0:
               ixs['M%d' %m]['em2s'].append(i)
               break
        for i in int_ep1s:
            if not i == 1.0:
               ixs['M%d' %m]['ep1s'].append(i)
               break
        for i in int_ep2s:
            if not i == 1.0:
               ixs['M%d' %m]['ep2s'].append(i)
               break
         
        if verbose: print ixs['M%d' %m]

        ints = [int_exp, int_ep1s, int_em1s, int_ep2s, int_em2s]
        ints_x = []
        for it in ints:
            for i in it:
                if not i == 1.0:
                   ints_x.append(i)
        if verbose: 
            for it in ints_x:
                print it

        ints_x = np.array(ints_x)
        ones = np.ones(len(ints_x))
        plt.scatter(ints_x, ones, s=10, c='red')

        plt.savefig('/t3home/vstampf/eos/plots/limits/outputs/mmm_M%d_20Aug_%s.pdf' %(m, ch))
 
    if twoD:
        y_exp = []; x_exp = []; x_ep1s = []; y_ep1s = []; x_ep2s = []; y_ep2s = []; x_em1s = []; y_em1s = []; x_em2s = []; y_em2s = [] 
        for m in [2,5,8]:

            for i in ixs['M%d' %m]['exp']:
                x_exp.append(m) 
                y_exp.append(i)

            for i in ixs['M%d' %m]['ep1s']:
                x_ep1s.append(m) 
                y_ep1s.append(i)

            for i in ixs['M%d' %m]['ep2s']:
                x_ep2s.append(m) 
                y_ep2s.append(i)

            for i in ixs['M%d' %m]['em1s']:
                x_em1s.append(m) 
                y_em1s.append(i)

            for i in ixs['M%d' %m]['em2s']:
                x_em2s.append(m) 
                y_em2s.append(i)

        plt.clf() #clearing figure
        plt.rc('text', usetex=True)

        # uncomment for continuous lines:
        # plt.plot(x_exp,  y_exp,  'k--', label = 'exp')
        # plt.plot(x_ep1s, y_ep1s, 'g--', label = 'ep1s')
        # plt.plot(x_em1s, y_em1s, 'g--', label = 'em1s')
        # plt.plot(x_ep2s, y_ep2s, 'y--', label = 'ep2s')
        # plt.plot(x_em2s, y_em2s, 'y--', label = 'em2s')

        # uncomment for separate markers:
        plt.scatter(x_exp,  y_exp,  s=8, c='black',  label='exp')
        plt.scatter(x_ep1s, y_ep1s, s=8, c='green',  label='ep1s')
        plt.scatter(x_em1s, y_em1s, s=8, c='green',  label='em1s')
        plt.scatter(x_ep2s, y_ep2s, s=8, c='yellow', label='ep2s')
        plt.scatter(x_em2s, y_em2s, s=8, c='yellow', label='em2s')

        if ch == 'mmm':    plt.title(r'$\mu\mu\mu$')
        if ch == 'eee':    plt.title(r'$eee$')
        if ch == 'eem_SS': plt.title(r'$ee\mu$ same sign')
        if ch == 'eem_OS': plt.title(r'$ee\mu$ opposite sign')
        if ch == 'mem_SS': plt.title(r'$\mu\mu e$ same sign')
        if ch == 'mem_OS': plt.title(r'$\mu\mu e$ opposite sign')
        plt.rc('font', family='serif')
        plt.axis([1, 10, 1e-06, 0.001])
        plt.xlabel(r'$M_N$ [GeV]')
        if ch in ['mmm', 'mem_SS', 'mem_OS']: plt.ylabel(r'${|V_{\mu N}|}^2$')
        if ch in ['eee', 'eem_SS', 'eem_OS']: plt.ylabel(r'${|V_{e N}|}^2$')
        plt.yscale('log')
        plt.legend(loc='lower left')
        plt.savefig('/t3home/vstampf/eos/plots/limits/outputs/limits_aug_20_%s.pdf' %ch)
