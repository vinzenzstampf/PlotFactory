#!/usr/bin/env python
from math import *
import os, commands
from sys import argv,exit
from optparse import OptionParser
import ROOT
import ROOT as rt
from collections import OrderedDict
import re
from pdb import set_trace
import numpy as np
from matplotlib import pyplot as plt
import shapely.geometry as sg
#from mpl_toolkits.mplot3d import Axes3D

#ROOT.gROOT.SetBatch(True)

def get_signals(verbose=False):

    # read input
    with open('../signals.py', 'r') as f_in:
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
            # print V2, float(V)**2

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

    # with open('combine_output_hnl_22_07_17_blind.txt', 'r') as f_in:
    in_file = '/t3home/vstampf/eos/plots/limits/inputs/data_cards_aug_20/limits_aug_20_%s.txt' %ch
    os.environ['LIM_FILE']   = in_file
    os.environ['OUT_FOLDER'] = '/t3home/vstampf/eos/plots/limits/outputs/'
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

def draw_v2_limits(ch='mmm',root=False): 
    '''here we want to draw limits for 
       one coupling and 3 different masses
    '''

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

        if root:
            for i in range(len(y_exp)):
                y_ep1s[i] = abs(y_ep1s[i] - y_exp[i]) 
                y_ep2s[i] = abs(y_ep2s[i] - y_exp[i]) 
                y_em1s[i] = abs(y_em1s[i] - y_exp[i]) 
                y_em2s[i] = abs(y_em2s[i] - y_exp[i]) 
                
            exp = rt.TGraph           (len(b_V2), np.array(b_V2), np.array(y_exp))
            gr1 = rt.TGraphAsymmErrors(len(b_V2), np.array(b_V2), np.array(y_exp), np.array(x_err), np.array(x_err), np.array(y_em1s), np.array(y_ep1s))
            gr2 = rt.TGraphAsymmErrors(len(b_V2), np.array(b_V2), np.array(y_exp), np.array(x_err), np.array(x_err), np.array(y_em2s), np.array(y_ep2s))

            gr2.SetTitle('#mu#mu#mu limits: m_{N} = %d GeV; |V_{#mu N}|^{2}; r' %m)
            exp.SetTitle('#mu#mu#mu limits: m_{N} = %d GeV; |V_{#mu N}|^{2}; r' %m)

            gr2.SetFillColor(rt.kYellow)
            gr1.SetFillColor(rt.kGreen)

            can = rt.TCanvas('limits', 'limits')
            can.cd(); can.SetLogy(); can.SetLogx()
            gr2.Draw('3')
            can.Modified(); can.Update()
            gr1.Draw('3same')
            can.Modified(); can.Update()
            exp.Draw('same')
            can.Modified(); can.Update()
            set_trace()
            
        
        if not root:
            plt.plot(b_V2, y_exp,  'k--', label = 'exp')
            plt.plot(b_V2, y_ep1s, 'g--', label = 'ep1s')
            plt.plot(b_V2, y_em1s, 'g--', label = 'em1s')
            plt.plot(b_V2, y_ep2s, 'y--', label = 'ep2s')
            plt.plot(b_V2, y_em2s, 'y--', label = 'em2s')

            plt.rc('text', usetex=True)

            plt.plot(b,  req1, 'r-')
            if ch == 'mmm': plt.title(r'$M_N = %d \, GeV,\; \mu\mu\mu$' %m)
            if ch == 'eee': plt.title(r'$M_N = %d \, GeV,\; eee$' %m)
            if ch == 'mem_OS': plt.title(r'$M_N = %d \, GeV,\; \mu\mu e OS$' %m)
            if ch == 'mem_SS': plt.title(r'$M_N = %d \, GeV,\; \mu\mu e SS$' %m)
            if ch == 'eem_OS': plt.title(r'$M_N = %d \, GeV,\; ee \mu OS$' %m)
            if ch == 'eem_SS': plt.title(r'$M_N = %d \, GeV,\; ee \mu SS$' %m)
            if 'mem' in ch or ch == 'mmm': plt.xlabel(r'${|V_{\mu N}|}^2$')
            if 'eem' in ch or ch == 'eee': plt.xlabel(r'${|V_{e N}|}^2$')
            plt.rc('font', family='serif')
            plt.axis([1e-06, 0.001, 0.1, 50000])
            plt.ylabel('r')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend(loc='lower left')

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
             
            print ixs['M%d' %m]

            ints = [int_exp, int_ep1s, int_em1s, int_ep2s, int_em2s]
            ints_x = []
            for it in ints:
                for i in it:
                    if not i == 1.0:
                       ints_x.append(i)
            for it in ints_x: print it

            # fig, ax = plt.subplots()
            # ax.axhline(y=1, color='r', linestyle='-')
            # ax.plot(b_V2, exp, 'b-')
            ints_x = np.array(ints_x)
            ones = np.ones(len(ints_x))
            plt.scatter(ints_x, ones, s=10, c='red')
            # set_trace()
            # plt.show()

            plt.savefig('/t3home/vstampf/eos/plots/limits/outputs/mmm_M%d_20Aug_%s.pdf' %(m, ch))

        print 'done'
 
    if 3 ==4:
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

        # for i in range(len(y_exp)):
            # y_ep1s[i] = abs(y_ep1s[i] - y_exp[i]) 
            # y_ep2s[i] = abs(y_ep2s[i] - y_exp[i]) 
            # y_em1s[i] = abs(y_em1s[i] - y_exp[i]) 
            # y_em2s[i] = abs(y_em2s[i] - y_exp[i]) 
            
        # set_trace()

        plt.rc('text', usetex=True)

        # plt.scatter(x_exp,  y_exp, s=10, c='red')

        plt.plot(x_exp,  y_exp,  'k--', label = 'exp')
        plt.plot(x_ep1s, y_ep1s, 'g--', label = 'ep1s')
        plt.plot(x_em1s, y_em1s, 'g--', label = 'em1s')
        plt.plot(x_ep2s, y_ep2s, 'y--', label = 'ep2s')
        plt.plot(x_em2s, y_em2s, 'y--', label = 'em2s')

        plt.title(r'$\mu\mu\mu$')
        plt.rc('font', family='serif')
        plt.axis([1, 10, 1e-06, 0.001])
        plt.xlabel(r'$M_N$ [GeV]')
        plt.ylabel(r'${|V_{\mu N}|}^2$')
        plt.yscale('log')
        plt.legend(loc='lower left')
        # plt.show()
        plt.savefig('/t3home/vstampf/eos/plots/limits/outputs/limits_aug_20_eee.pdf')
        # plt.scatter(ints_x, ones, s=10, c='red')
        # exp = rt.TGraph           (len(y_exp), np.array(x_exp), np.array(y_exp))
        # gr1 = rt.TGraphAsymmErrors(len(b_V2), np.array(b_V2), np.array(y_exp), np.array(x_err), np.array(x_err), np.array(y_em1s), np.array(y_ep1s))
        # gr2 = rt.TGraphAsymmErrors(len(b_V2), np.array(b_V2), np.array(y_exp), np.array(x_err), np.array(x_err), np.array(y_em2s), np.array(y_ep2s))

        # gr2.SetTitle('#mu#mu#mu limits; m_{N} [GeV]; |V_{#mu N}|^{2}')

        # gr2.SetFillColor(rt.kYellow)
        #'r^', label='exp') gravefig('/t3home/vstampf/eos/plots/limits/outputs/grapasdaasd.pdf').SetFillColor(rt.kGreen)

        # can = rt.TCanvas('limits', 'limits')
        # can.cd(); can.SetLogy(); can.SetLogx()
        # gr2.Draw('3')
        # can.Modified(); can.Update()
        # gr1.Draw('3same')
        # can.Modified(); can.Update()
        # exp.Draw()
        # can.Modified(); can.Update()
        # can.SaveAs('/t3home/vstampf/eos/plots/limits/outputs/graphs.root')

        # os.system('cp $LIM_FILE $OUT_FOLDER')

def draw_mass_limits(): 
    '''here we want to draw limits for 
       one coupling and 3 different masses
    '''

    limits  = get_lim_dict()
    signals = get_signals()

    b    = np.arange(0., 11, 1)
    b_m  = [2, 5, 8]

    req1   = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    y_exp  = [ limits['M2_V0.00244948974278_mu']['exp'],  limits['M5_V0.00244948974278_mu']['exp'],  limits['M8_V0.00244948974278_mu']['exp']  ]
    y_ep1s = [ limits['M2_V0.00244948974278_mu']['ep1s'], limits['M5_V0.00244948974278_mu']['ep1s'], limits['M8_V0.00244948974278_mu']['ep1s'] ]
    y_ep2s = [ limits['M2_V0.00244948974278_mu']['ep2s'], limits['M5_V0.00244948974278_mu']['ep2s'], limits['M8_V0.00244948974278_mu']['ep2s'] ]
    y_em2s = [ limits['M2_V0.00244948974278_mu']['em2s'], limits['M5_V0.00244948974278_mu']['em2s'], limits['M8_V0.00244948974278_mu']['em2s'] ]
    y_em1s = [ limits['M2_V0.00244948974278_mu']['em1s'], limits['M5_V0.00244948974278_mu']['em1s'], limits['M8_V0.00244948974278_mu']['em1s'] ]

    plt.plot(b_m, y_exp,  '^', label = 'exp')
    plt.plot(b_m, y_ep1s, 's', label = 'ep1s')
    plt.plot(b_m, y_em1s, 's', label = 'em1s')
    plt.plot(b_m, y_ep2s, 'o', label = 'ep2s')
    plt.plot(b_m, y_em2s, 'o', label = 'em2s')

    plt.plot(b,  req1, 'r-')
    #plt.axis([0,10,0.1,1000])
    plt.axis([0,10,0.01,10])
    plt.xlabel('neutrino mass [GeV]')
    plt.ylabel('r')
    plt.yscale('log')
    plt.legend(loc='lower left')

    plt.show()

    print 'done'


def draw_2D_limits(verbose=False):

    mpl = False; root = True

    limits  = get_lim_dict()
    signals = get_signals()

    b     = np.arange(0., 11, 1)
    b_m   = [2, 5, 8]
    b_V2  = [6e-06, 8e-06, 1e-05, 2e-05, 3e-05, 5e-05, 7e-05, 0.0001, 0.0002, 0.0003, 0.0005]

    if mpl:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for lim in limits.keys():
            xs = limits[lim]['mass']
            ys = signals[lim]['V2']
            try:    
                zs = limits[lim]['exp'];  m='^'
                p1 = limits[lim]['ep1s']; m='s'
                m1 = limits[lim]['em1s']; m='s'
                p2 = limits[lim]['ep2s']; m='o'
                m2 = limits[lim]['em2s']; m='o'
                ax.scatter(xs, ys, p1, zdir='z', s=20, c='g', marker=m)
                ax.scatter(xs, ys, m1, zdir='z', s=20, c='g', marker=m)
                ax.scatter(xs, ys, p2, zdir='z', s=20, c='y', marker=m)
                ax.scatter(xs, ys, m2, zdir='z', s=20, c='y', marker=m)
                print lim, 'has limits'
                print xs, ys
            except KeyError: 
                zs = 0; c  = 'r'; m='o'
            ax.scatter(xs, ys, zs, zdir='z', s=20, c=c, marker=m)

        ax.set_xlabel('m_N')
        ax.set_ylabel('V^2')
        ax.set_zlabel('r')
    #    ax.set_yscale('log')
    #    ax.set_zscale('log')

        plt.show()

    if root: 

        b     = np.arange(0., 11, 1)
        req1  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        LIM_X = []; LIM_Y = []; LIM_exp = []
        LIM_ep1s = []; LIM_ep2s = []; LIM_em1s = []; LIM_em2s = []

        for k in limits.keys():
            if limits[k].has_key('exp'):  
                LIM_X   .append(limits [k]['mass']) 
                LIM_Y   .append(signals[k]['V2'  ]) 
                LIM_exp .append(limits [k]['exp' ]) 
                LIM_ep1s.append(limits [k]['ep1s']) 
                LIM_em1s.append(limits [k]['em1s']) 
                LIM_ep2s.append(limits [k]['ep2s']) 
                LIM_em2s.append(limits [k]['em2s']) 
        LIM = len(LIM_X)

        LIM_X = np.array(LIM_X); LIM_Y = np.array(LIM_Y); LIM_exp = np.array(LIM_exp)
        LIM_ep1s = np.array(LIM_ep1s); LIM_ep2s = np.array(LIM_ep2s); LIM_em1s = np.array(LIM_em1s); LIM_em2s = np.array(LIM_em2s)

        graph_exp  = rt.TGraph2D('exp' , 'exp' , LIM, LIM_X, LIM_Y, LIM_exp)
        graph_ep1s = rt.TGraph2D('ep1s', 'ep1s', LIM, LIM_X, LIM_Y, LIM_ep1s)
        graph_em1s = rt.TGraph2D('em1s', 'em1s', LIM, LIM_X, LIM_Y, LIM_em1s)
        graph_ep2s = rt.TGraph2D('ep2s', 'ep2s', LIM, LIM_X, LIM_Y, LIM_ep2s)
        graph_em2s = rt.TGraph2D('em2s', 'em2s', LIM, LIM_X, LIM_Y, LIM_em2s)
        graph_r    = rt.TGraph2D('r=1',  'r=1',  LIM, LIM_X, LIM_Y, np.ones(LIM))

        # set_trace()

        cont_exp  = graph_exp .GetContourList(1.0)[0]; cont_exp .SetLineColor(rt.kBlack) ; cont_exp .SetName('cont_exp' ) #  cont_exp .SetFillColor(rt.kBlack) ; cont_exp .SetFillStyle(1001)
        cont_em1s = graph_em1s.GetContourList(1.0)[0]; cont_em1s.SetLineColor(rt.kGreen) ; cont_em1s.SetName('cont_em1s') #  cont_em1s.SetFillColor(rt.kGreen) ; cont_em1s.SetFillStyle(1001) 
        cont_ep1s = graph_ep1s.GetContourList(1.0)[0]; cont_ep1s.SetLineColor(rt.kGreen) ; cont_ep1s.SetName('cont_ep1s') #  cont_ep1s.SetFillColor(rt.kGreen) ; cont_ep1s.SetFillStyle(1001) 
        cont_em2s = graph_em2s.GetContourList(1.0)[0]; cont_em2s.SetLineColor(rt.kYellow); cont_em2s.SetName('cont_em2s') #  cont_em2s.SetFillColor(rt.kYellow); cont_em2s.SetFillStyle(1001)
        cont_ep2s = graph_ep2s.GetContourList(1.0)[0]; cont_ep2s.SetLineColor(rt.kYellow); cont_ep2s.SetName('cont_ep2s') #  cont_ep2s.SetFillColor(rt.kYellow); cont_ep2s.SetFillStyle(1001)


        if verbose:
            for il in range( graph_exp .GetContourList(1).GetSize()): print 'Contour Lists: exp : ', il, graph_exp .GetContourList(1).At(il).GetN()
        # do another test: for i in range(12): print graph_exp.Interpolate( cont_exp.GetX()[i],  cont_exp.GetY()[i] ), cont_exp.GetX()[i],  cont_exp.GetY()[i]
            for il in range( graph_em1s.GetContourList(1).GetSize()): print 'Contour Lists: em1s: ', il, graph_em1s.GetContourList(1).At(il).GetN()
            for il in range( graph_ep1s.GetContourList(1).GetSize()): print 'Contour Lists: ep1s: ', il, graph_ep1s.GetContourList(1).At(il).GetN()
            for il in range( graph_em2s.GetContourList(1).GetSize()): print 'Contour Lists: em2s: ', il, graph_em2s.GetContourList(1).At(il).GetN()
            for il in range( graph_ep2s.GetContourList(1).GetSize()): print 'Contour Lists: ep2s: ', il, graph_ep2s.GetContourList(1).At(il).GetN()

        if verbose:
            cnvs = rt.TCanvas('graphs', 'graphs')
            cnvs.cd()
            # cnvs.Divide(2,3)

            # cnvs.cd(1)
            graph_exp.SetMarkerSize(0.7)
            graph_exp.Draw('surf1')
            graph_exp.Draw('p0same')
            cnvs.SetLogy(); cnvs.SetLogz()
            cnvs.Modified(); cnvs.Update()

            # cnvs.cd(2)
            # graph_exp.Draw('surf1')
            # cnvs.SetLogy(); cnvs.SetLogz()
            # cnvs.Modified(); cnvs.Update()

            # cnvs.cd(3)
            # graph_ep1s.Draw('surf1')
            # cnvs.SetLogy(); cnvs.SetLogz()
            # cnvs.Modified(); cnvs.Update()

            # cnvs.cd(4)
            # graph_em1s.Draw('surf1')
            # cnvs.SetLogy(); cnvs.SetLogz()
            # cnvs.Modified(); cnvs.Update()
            
            # cnvs.cd(5)
            # graph_ep2s.Draw('surf1')
            # cnvs.SetLogy(); cnvs.SetLogz()
            # cnvs.Modified(); cnvs.Update()

            # cnvs.cd(6)
            # graph_em2s.Draw('surf1')
            # cnvs.SetLogy(); cnvs.SetLogz()
            # cnvs.Modified(); cnvs.Update()

            cnvs.SaveAs('/t3home/vstampf/eos/plots/limits/outputs/graphs.pdf')
            cnvs.SaveAs('/t3home/vstampf/eos/plots/limits/outputs/graphs.png')
            cnvs.SaveAs('/t3home/vstampf/eos/plots/limits/outputs/graphs.root')

        if verbose:
            for i in range(len(graph_exp .GetZ())): print 'Entries: exp : ', graph_exp.GetZ()[i],  graph_exp.GetX()[i],  graph_exp.GetY()[i]
            print ''
            for i in range(len(graph_em1s.GetZ())): print 'Entries: em1s: ', graph_em1s.GetZ()[i], graph_em1s.GetX()[i], graph_em1s.GetY()[i]
            print ''
            for i in range(len(graph_ep1s.GetZ())): print 'Entries: ep1s: ', graph_ep1s.GetZ()[i], graph_ep1s.GetX()[i], graph_ep1s.GetY()[i]
            print ''
            for i in range(len(graph_em2s.GetZ())): print 'Entries: em2s: ', graph_em2s.GetZ()[i], graph_em2s.GetX()[i], graph_em2s.GetY()[i]
            print ''
            for i in range(len(graph_ep2s.GetZ())): print 'Entries: ep2s: ', graph_ep2s.GetZ()[i], graph_ep2s.GetX()[i], graph_ep2s.GetY()[i]


        err1_xl = cont_em1s.GetX() 
        err1_yl = cont_em1s.GetY() 

        err1_xh = cont_ep1s.GetX() 
        err1_yh = cont_ep1s.GetY() 

        err2_xl = cont_em2s.GetX() 
        err2_yl = cont_em2s.GetY() 

        err2_xh = cont_ep2s.GetX() 
        err2_yh = cont_ep2s.GetY() 

        print len(cont_exp.GetX()), len(cont_exp.GetY()), len(err1_xl), len(err1_xh), len(err1_yl), len(err1_yh), len(err2_xl), len(err2_xh), len(err2_yl), len(err2_yh)
        MIN = min ( len(cont_exp.GetX()), len(cont_exp.GetY()), len(err1_xl), len(err1_xh), len(err1_yl), len(err1_yh), len(err2_xl), len(err2_xh), len(err2_yl), len(err2_yh) )

        val_x = cont_exp.GetX()
        val_y = cont_exp.GetY()

        for i in range(MIN):
            err1_xh[i] = abs(err1_xh[i] - val_x[i]) 
            err1_xl[i] = abs(err1_xl[i] - val_x[i]) 
            err1_yh[i] = abs(err1_yh[i] - val_y[i]) 
            err1_yl[i] = abs(err1_yl[i] - val_y[i]) 

        for i in range(MIN):
            err2_xh[i] = abs(err2_xh[i] - val_x[i]) 
            err2_xl[i] = abs(err2_xl[i] - val_x[i]) 
            err2_yh[i] = abs(err2_yh[i] - val_y[i]) 
            err2_yl[i] = abs(err2_yl[i] - val_y[i]) 


        # gr1 = rt.TGraphErrors(len(val1_x), np.array(val1_x), np.array(val1_y), np.array(err1_xl), np.array(err1_yl))
        # gr2 = rt.TGraphErrors(len(val2_x), np.array(val2_x), np.array(val2_y), np.array(err2_x), np.array(err2_y))
        gr1 = rt.TGraphAsymmErrors(len(val_x), np.array(val_x), np.array(val_y), np.array(err1_xl), np.array(err1_xh), np.array(err1_yl), np.array(err1_yh))
        gr2 = rt.TGraphAsymmErrors(len(val_x), np.array(val_x), np.array(val_y), np.array(err2_xl), np.array(err2_xh), np.array(err2_yl), np.array(err2_yh))
 
        gr1.SetName('gr1')
        gr2.SetName('gr2')

        gr2.SetTitle('#mu#mu#mu limits; m_{N} [GeV]; |V_{#mu N}|^{2}')

        gr2.SetFillColor(rt.kYellow)
        gr1.SetFillColor(rt.kGreen)

        can = rt.TCanvas('limits', 'limits')
        can.cd()
        gr2.Draw('3')
        can.Modified(); can.Update()
        gr1.Draw('3same')
        can.Modified(); can.Update()
        cont_exp.Draw('same')
        can.Modified(); can.Update()
        cont_ep1s.Draw('same')
        cont_em1s.Draw('same')
        cont_ep2s.Draw('same')
        cont_em2s.Draw('same')

        '''also overlay all error graphs to see it TGraphAsymmError take the right points'''
        set_trace()

        # can.Divide(2,2)

        # can.cd(1)
        # gr2.Draw('a3')
        # # cont_em1s.Draw()
        # cont_ep1s.Draw('same')

        # can.cd(2)
        # gr1.Draw('a3')
        # # cont_em2s.Draw()
        # cont_ep2s.Draw('same')

        # can.cd(3)
        # gr2.Draw('3')
        # gr1.Draw('3same')

        # can.cd(4)
        # gr2.Draw('3')
        # gr1.Draw('3same')
        # cont_exp.Draw('same')

        # can.cd(5)
        # gr2.Draw('a3')
        # gr1.Draw('a3same')

        # can.cd(6)
        # cont_exp.Draw()
        # cont_em1s.Draw('same')
        # cont_ep1s.Draw('same')
        # cont_em2s.Draw('same')
        # cont_ep2s.Draw('same')

        # cont_exp .Draw()
        # cont_em1s.Draw('Fsame')
        # cont_ep1s.Draw('Lsame')
        # cont_em2s.Draw('Fsame')
        # cont_ep2s.Draw('Lsame')

        # can.cd(1)
        # # set_trace()
        # can.GetPad(1).SetLogy()
        # can.GetPad(1).SetLogz()
        # graph_exp.Draw('surf1')


        # can.cd(3)
        # can.GetPad(3).SetLogy()
        # can.GetPad(3).SetLogz()
        # graph_exp.Draw('p0')

        can.SaveAs('/t3home/vstampf/eos/plots/limits/outputs/asdasd.pdf')
        can.SaveAs('/t3home/vstampf/eos/plots/limits/outputs/asdasd.png')
        can.SaveAs('/t3home/vstampf/eos/plots/limits/outputs/asdasd.root')


# def draw_2D_new():


def getLimitYN ( h_lim_mu, r_exluded=1):
    name = h_lim_mu.GetName().replace("mu","yn")
    h_lim_yn = h_lim_mu.Clone(name)
    for ix in range(1,h_lim_yn.GetNbinsX()+1):
        for iy in range(1,h_lim_yn.GetNbinsY()+1):
            r = h_lim_yn.GetBinContent(ix,iy)
            h_lim_yn.SetBinContent(ix, iy, 1e-3 if r>r_exluded else 1 if r>0 else 0)
    return h_lim_yn
    
# def getLimitXS ( h_lim_mu, h_xs):
def getLimitXS (h_lim_mu, verbose=False):
    signals = get_signals()
    name = h_lim_mu.GetName().replace("mu","xs")
    h_lim_xs = h_lim_mu.Clone(name)
    for ix in range(1,h_lim_xs.GetNbinsX()+1):
        m = h_lim_xs.GetXaxis().GetBinCenter(ix)
        for iy in range(1,h_lim_xs.GetNbinsY()+1):
            r = h_lim_xs.GetBinContent(ix,iy)
            V2 = h_lim_xs.GetYaxis().GetBinCenter(iy)
            xs = signals[get_sig_name(V2, m)]['xsec']
            # if r == 0: xs = 0
            if verbose: print r, m, V2, xs
            h_lim_xs.SetBinContent(ix, iy, r*xs)
    return h_lim_xs

def get_sig_name(V2, m):

    signals = get_signals()
    best_guess_v2 = 1000; match = None; v2_match = None

    for sig in [sig for sig in signals.keys() if 'mu' in sig and 'M'+str(int(m)) in sig]:
        guess_v2 = signals[sig]['V2']
        diff      = abs(V2 - guess_v2) 
        best_diff = abs(V2 - best_guess_v2)
        # print diff, best_diff, V2, guess_v2, diff < best_diff
        if diff < best_diff: 
            best_guess_v2 = signals[sig]['V2'] 
            match = sig 
            # print best_guess_v2, sig, '\n'
    # print match, best_guess_v2, V2 
    return match

def prepare_limits(h_lims_mu0=None, h_lims_xs0=None, h_lims_yn0=None):

    signals = get_signals()
    lim_dict  = get_lim_dict(ch)

    rlim = OrderedDict()
    for sig in lim_dict.keys():
        mN = lim_dict[sig]['mass']

        for lim in [lim for lim in lim_dict[sig].keys() if 'e' in lim or 'o' in lim]:
            rlim[lim] = lim_dict[sig][lim]/100. #FIXME divide by 100 to see if code makes sense
 
        # get xs for the given mass
        # print sig, lim
        xs  = signals[sig]['xsec']
        exs = signals[sig]['xsec_err']
        V2  = signals[sig]['V2']

        # unblinded
        # rlim['op1s'] = rlim['obs'] * xs / (xs+exs)
        # rlim['omNs'] = rlim['obs'] * xs / (xs-exs)
    
        #fill the 2d limit histos
        binX=h_lims_mu0[lim].GetXaxis().FindBin(mN)
        binY=h_lims_mu0[lim].GetYaxis().FindBin(V2)
    
        for lim in [lim for lim in lim_dict[sig].keys() if 'e' in lim or 'o' in lim]:
            # limit on signal-strength multiplier
            h_lims_mu0[lim].SetBinContent(binX, binY, rlim[lim])
            # limit on cross-section
            h_lims_xs0[lim].SetBinContent(binX, binY, rlim[lim]*xs)
            # yes/no exclustion plot in terms of signal-strength multiplie #FIXME divide by 100 to see if code makes senser
            h_lims_yn0[lim].SetBinContent(binX, binY, 1 if rlim[lim]<1 else 1e-3) # VS: idk why 1e-3, but does it matter? 1 = excluded, 0 = not-excluded
 

def interpolateDiagonal(hist):
    # interpolate in diagonal direction to fill remaining missing holes
    # start from 15 bins away and finish in the diagonal
    Nx = hist.GetNbinsX() 
    Ny = hist.GetNbinsY()
    for i in range(14,-1,-1): # 14...0
        j=0
        while i+j<Nx and j<Ny:
           j+=1
           val1=hist.GetBinContent(i+j,j)
           if val1==0 or hist.GetBinContent(i+j+1,j+1)!=0:
               continue

           n=2
           while hist.GetBinContent(i+j+n,j+n)==0 and i+j+n<Nx and j+n<Ny:
               n+=1
           val2 = hist.GetBinContent(i+j+n,j+n)
           if val2==0:
               continue
           for nn in range(1,n):                    
               hist.SetBinContent(i+j+nn,j+nn,val1+(val2-val1)*nn/n) 


def extractSmoothedContour(hist, nSmooth=1):
    # if name contains "mu" histogram is signal strenght limit, otherwise it's a Yes/No limit
    isMu = "mu" in hist.GetName()
    #ROOT.gStyle.SetNumberContours(4 if isMu else 2)
    shist = hist.Clone(hist.GetName()+"_smoothed")

    # if smoothing a limit from mu, we need to modify the zeros outside the diagonal, otherwise the smoothing fools us in the diagonal transition
    if isMu:
        for ix in range(1, shist.GetNbinsX()):
            for iy in range(shist.GetNbinsY(),0,-1):
                if shist.GetBinContent(ix,iy)==0:
                    for iyy in range(iy, shist.GetNbinsY()):
                        shist.SetBinContent(ix,iyy, shist.GetBinContent(ix,iy-1))
                else:
                    break
        if model=="T2cc": # for T2cc do it also  bottom-up
            # after smoothing a limit from mu, we need to modify the zeros outside the diagonal, otherwise the contours come wrong for the diagonal 
            interpolateDiagonal(hist)
            for ix in range(1, shist.GetNbinsX()):
                for iy in range(0,shist.GetNbinsY()):
                    if shist.GetBinContent(ix,iy)==0:
                        for iyy in range(iy, 0, -1):
                            shist.SetBinContent(ix,iyy, shist.GetBinContent(ix,iy+1))
                    else:
                        break
            

    for s in range(nSmooth):
        #shist.Smooth() # default smoothing algorithm
        shist.Smooth(1,"k3a")  # k3a smoothing algorithm

    # after smoothing a limit from mu, we need to modify the zeros outside the diagonal, otherwise the contours come wrong for the diagonal
    if isMu:
        for ix in range(1,shist.GetNbinsX()):
            for iy in range(1,shist.GetNbinsY()):
                if hist.GetBinContent(ix,iy)==0:
                    shist.SetBinContent(ix,iy, 1.1)
        
    shist.SetMinimum(0)
    shist.SetMaximum(2 if isMu else 1)
    shist.SetContour(4 if isMu else 2)
    canvas = ROOT.TCanvas()
    shist.Draw("contz list")
    ROOT.gPad.Update()
    obj = ROOT.gROOT.GetListOfSpecials().FindObject("contours")
    list = obj.At(1 if isMu else 0)
    ## take largest graph
    #max_points = -1
    #for l in range(list.GetSize()):
    #    gr = list.At(l).Clone()
    #    n_points = gr.GetN()
    #    if n_points > max_points:
    #        graph = gr
    #        max_points = n_points

    graph = []
    for l in range(list.GetSize()):
        gr = list.At(l).Clone()
        n_points = gr.GetN()
        graph.append((n_points,gr))
    graph.sort(reverse=True)

    #graph = list.First().Clone()
    name = "gr_"
    name += shist.GetName()
    #graph.SetName(name)
    for i,g in enumerate(graph):
        g[1].SetName(name+(str(i)if i>0 else ""))
    #graph.Draw("sameC")
    del canvas
    del shist
    del obj
    del list
    if len(graph)==1: graph.append(graph[0])
    return [graph[0][1], graph[1][1]]  # return list of two largest graphs


def extractSmoothedContourRL(hist, nSmooth=1):
    hR, hL = hist.Clone("muR"), hist.Clone("muL")
    for ix in range(1,hist.GetNbinsX()+1):
        for iy in range(1,hist.GetNbinsY()+1):
            m1, m2 = hist.GetXaxis().GetBinLowEdge(ix), hist.GetYaxis().GetBinLowEdge(iy)
            #if m1-m2>=175:
            #    hL.SetBinContent(ix,iy, 0)
            #if m1-m2<=175:
            #    hR.SetBinContent(ix,iy, 0)
            if m1-m2>=150:
                hL.SetBinContent(ix,iy, 0)
            if m1-m2<=200:
                hR.SetBinContent(ix,iy, 0)

    # check if graph 0 makes sense, otherwise take graph 1
    nL = 0
    g = extractSmoothedContour(hL, nSmooth)
    x,y = ROOT.Double(0), ROOT.Double(0)
    g[0].GetPoint(8,x,y)
    if (x-y)>175: nL=1

    gR,gL = extractSmoothedContour(hR, nSmooth)[0], g[nL]
    gR.SetName("gr_"+hist.GetName()+"_smoothed")
    gL.SetName("gr_"+hist.GetName()+"_smoothed1")
    return [gR,gL]
    

def unexcludeDiagonal(hist, mSplit=175): 
    for ix in range(1,hist.GetNbinsX()+1):
        for iy in range(1,hist.GetNbinsY()+1):
            m1, m2 = hist.GetXaxis().GetBinLowEdge(ix), hist.GetYaxis().GetBinLowEdge(iy)
            val = hist.GetBinContent(ix,iy)
            if m1-m2==mSplit:
                hist.SetBinContent(ix,iy, max(1.5,val))

def mergeGraphs(graphs):
    mg = ROOT.TMultiGraph()
    mg.Add(graphs[0])
    mg.Add(graphs[1])
    g = ROOT.TGraph()
    g.Merge(mg.GetListOfGraphs())
    g.SetName(graphs[0].GetName())
    return g


def main():

    # print "running:", argv

    # if len(argv)<2:
        # print "Usage: "+argv[0]+" fileWithLimits.txt [doOneFold (default=False)]"
        # exit(1)

    # INPUT = argv[1]
    # doOneFold = (argv[2]=='True' or argv[2]=='true') if len(argv)>2 else False

    # get contours separately for the left and right side of the deltaM=Mtop diagonal (T2tt)
    divideTopDiagonal = False

    model   = "hnl"

    limits = ["obs", "exp", "ep1s", "em1s", "ep2s", "em2s", "op1s", "om1s"]
    limits = ["exp", "ep1s", "em1s", "ep2s", "em2s"]
    #limits = ["obs", "exp"]

    # coloum-limit map for txt files (n -> column n+1) 
    fileMap = {"exp":2, "obs":3, "ep1s":4, "em1s":5, "ep2s":6, "em2s":7} #TODO VS: WHAT DOES THIS DO??
    #fileMap = {"exp":2, "ep1s":3, "em1s":4, "ep2s":5, "em2s":6}


    h_lims_mu0 = {} # limits in signal-strength, original binning
    h_lims_yn0 = {} # limits in excluded/non-exluded, original binning
    h_lims_xs0 = {} # limits in cross-section, original binning

    h_lims_mu   = {} # limits in signal-strength, interpolated
    h_lims_yn   = {} # limits in excluded/non-exluded, interpolated
    h_lims_xs   = {} # limits in cross-section, interpolated
    g2_lims_mu  = {} # TGraph2D limits in signal-strength, automatic interpolation

    mN_min, mN_max = 2, 8
    xbinSize = 3

    V2_min, V2_max = 1e-06, 5e-04
    ybinSize = 1e-05
    b_V2_log = np.logspace(-5.09691,-3.30102,11,endpoint=True)

    # log bins central around sample-V2; evenly spaced on log scale
    l_V2_log = [pow(10, 2 * log(6e-06 , 10) - (1/2.) * ( log(8e-06 , 10) + log(6e-06 , 10) ) ), 
                pow(10, (1/2.) * ( log(8e-06 , 10) + log(6e-06 , 10) ) ), 
                pow(10, (1/2.) * ( log(1e-05 , 10) + log(8e-06 , 10) ) ),  
                pow(10, (1/2.) * ( log(2e-05 , 10) + log(01e-05, 10) ) ),  
                pow(10, (1/2.) * ( log(3e-05 , 10) + log(02e-05, 10) ) ),  
                pow(10, (1/2.) * ( log(5e-05 , 10) + log(03e-05, 10) ) ),  
                pow(10, (1/2.) * ( log(7e-05 , 10) + log(05e-05, 10) ) ),  
                pow(10, (1/2.) * ( log(0.0001, 10) + log(07e-05, 10) ) ),  
                pow(10, (1/2.) * ( log(0.0002, 10) + log(0.0001, 10) ) ), 
                pow(10, (1/2.) * ( log(0.0003, 10) + log(0.0002, 10) ) ), 
                pow(10, (1/2.) * ( log(0.0005, 10) + log(0.0003, 10) ) ), 
                pow(10, 2 * log(0.0005, 10) - (1/2.) * ( log(0.0005, 10) + log(0.0003, 10) ) ),] 

    b_V2_log = np.array(l_V2_log)

    mN = "m_{N}" 
    V2 = "|V_{#mu N}|^{2}"

    # create histos
    for lim in limits:
        # uniform 25 GeV binning
        #h_lims_mu0[lim] = ROOT.TH2F(lim+"_mu0", model, int((mN_max-mN_min+binSize)/binSize), mN_min-binSize/2., mN_max+binSize/2., int((V2_max-V2_min+binSize)/binSize), V2_min-binSize/2., V2_max+binSize/2.)
        # h_lims_mu0[lim] = ROOT.TH2F(lim+"_mu0", model, int((mN_max-mN_min+xbinSize)/xbinSize), mN_min-xbinSize/2., mN_max+xbinSize/2., int((V2_max-V2_min+2*ybinSize)/(ybinSize)), V2_min-3*ybinSize/2., V2_max+ybinSize/2.)
        h_lims_mu0[lim] = ROOT.TH2F(lim+"_mu0", model, int((mN_max-mN_min+xbinSize)/xbinSize), mN_min-xbinSize/2., mN_max+xbinSize/2., len(b_V2_log)-1, b_V2_log)
        h_lims_yn0[lim] = ROOT.TH2F(lim+"_yn0", model, int((mN_max-mN_min+xbinSize)/xbinSize), mN_min-xbinSize/2., mN_max+xbinSize/2., len(b_V2_log)-1, b_V2_log)
        h_lims_xs0[lim] = ROOT.TH2F(lim+"_xs0", model, int((mN_max-mN_min+xbinSize)/xbinSize), mN_min-xbinSize/2., mN_max+xbinSize/2., len(b_V2_log)-1, b_V2_log)

        h_lims_mu0[lim].SetXTitle(mN)    
        h_lims_yn0[lim].SetXTitle(mN)    
        h_lims_xs0[lim].SetXTitle(mN)    

        h_lims_mu0[lim].SetYTitle(V2)
        h_lims_yn0[lim].SetYTitle(V2)
        h_lims_xs0[lim].SetYTitle(V2)

    # read txt file with limits (map defined above)
    print "reading file..."
    prepare_limits(h_lims_mu0, h_lims_xs0, h_lims_yn0)

    # output = INPUT.replace(".txt", ("-OneFold" if (model=="T2qq" and doOneFold) else "") + ".root")
    output = 'hnl_test.root'
    fout = ROOT.TFile(output, "RECREATE")
    fout.cd()

    print "interpolating..."
    for lim in limits:
        # set_trace()
        # fillHorizontalBelowZero(h_lims_mu0[lim])
        # interpolation done automatically by TGraph2D using Delaunay method
        g2_lims_mu[lim] = ROOT.TGraph2D(h_lims_mu0[lim])
    '''
        xbinSize_inter = xbinSize/2.
        #xbinSize_inter = xbinSize/2. if model!='T2cc' else ybinSize # bin size of interpolation graph (12.5 GeV as decided in dec7 meeting @ R40) 
        ybinSize_inter = ybinSize/2. if model!='T2cc' else ybinSize # bin size of interpolation graph (12.5 GeV as decided in dec7 meeting @ R40) 
        g2_lims_mu[lim].SetNpx( int((g2_lims_mu[lim].GetXmax()-g2_lims_mu[lim].GetXmin())/xbinSize_inter) )
        g2_lims_mu[lim].SetNpy( int((g2_lims_mu[lim].GetYmax()-g2_lims_mu[lim].GetYmin())/ybinSize_inter) )
        h_lims_mu[lim] = g2_lims_mu[lim].GetHistogram()
        h_lims_mu[lim].SetName( h_lims_mu0[lim].GetName().replace("mu0","mu") )
                 
        #remove negative or nan bins that appear in T2qq for no apparent reason
        for ix in range(1,h_lims_mu[lim].GetNbinsX()+1):
            for iy in range(1,h_lims_mu[lim].GetNbinsY()+1):
                if h_lims_mu[lim].GetBinContent(ix,iy) < 0: #if negative set to zero
                    h_lims_mu[lim].SetBinContent(ix,iy,0)
                if isnan(h_lims_mu[lim].GetBinContent(ix,iy)): #if nan set to neighbour average
                    val = (h_lims_mu[lim].GetBinContent(ix+1,iy) + h_lims_mu[lim].GetBinContent(ix-1,iy) + h_lims_mu[lim].GetBinContent(ix,iy+1) + h_lims_mu[lim].GetBinContent(ix,iy-1) )/4.0
                    h_lims_mu[lim].SetBinContent(ix,iy,val)
    '''


    print "translating to x-sec and yes/no limits and saving 2d histos..."
    for lim in limits:
        #if model=="T2tt":
        #    unexcludeDiagonal( h_lims_mu[lim])
        #if model=="T2bb":  # do this for summary plot as per FKW request
        #    unexcludeDiagonal( h_lims_mu[lim],25 )    
        #    unexcludeDiagonal( h_lims_mu[lim],37.5 )    
        
        # h_lims_yn[lim] = getLimitYN ( h_lims_mu[lim] )
        h_lims_yn[lim] = getLimitYN ( h_lims_mu0[lim] ) #TODO (VS) this should be done with interpolated plots
        # h_lims_xs[lim] = getLimitXS ( h_lims_mu[lim], h_xs )
        h_lims_xs[lim] = getLimitXS ( h_lims_mu0[lim] ) #TODO (VS) this should be done with interpolated plots
        # print lim; set_trace()
        
        h_lims_mu0[lim].Write()
        h_lims_xs0[lim].Write()
        h_lims_yn0[lim].Write()
        # h_lims_mu [lim].Write() #TODO (VS) comes from interpolation
        h_lims_xs [lim].Write()
        h_lims_yn [lim].Write()

    graphs0 = {}
    graphs1 = {}  # smoothed

    print "extracting contours and saving graphs..."
    for lim in limits:
        # get contour. choose the one with maximum number of points
        g_list = g2_lims_mu[lim].GetContourList(1)
        graphs = []
        for il in range(g_list.GetSize()):
            gr = g_list.At(il)
            n_points = gr.GetN()
            graphs.append((n_points,gr))
        graphs.sort(reverse=True)
        graphs0[lim] = graphs[0][1]
        #if model=='T2tt' and (lim=='ep2s' or lim=='ep1s'): # two unconnected contours are obtained for these two guys
        #if model=='T2tt' and (lim=='ep2s' or lim=='ep1s' or lim=='exp'): # two unconnected contours are obtained for these two guys
    #    if model=='T2tt' and (lim=='ep2s' or lim=='ep1s' or lim=='op1s' or lim=='om1s' or lim=='em1s' or lim=='exp'  or lim=='obs'): # two unconnected contours are obtained for these two guys
    #        graphs0[lim]=mergeGraphs([graphs[0][1],graphs[1][1]])
        graphs0[lim].SetName("gr_"+lim)
        graphs0[lim].Write()

    fout.Close()

def new():

    print "smoothing..."
    for lim in limits:
        nSmooth = 1 if model=="T2tt" else 2 if model!="T2qq" else 3
        if model!="T2tt" or not divideTopDiagonal:
            graphs = extractSmoothedContour(h_lims_mu[lim], nSmooth)
            graphs1[lim]=graphs[0]
            #if model=='T2tt' and (lim=='ep2s' or lim=='ep1s'): # two unconnected contours are obtained for these two guys
            #if model=='T2tt' and (lim=='ep2s' or lim=='ep1s' or lim=='exp'): # two unconnected contours are obtained for these two guys
            if model=='T2tt' and (lim=='ep2s' or lim=='om1s' or lim=='exp'  or lim=='ep1s'): # two unconnected contours are obtained for these two guys
                graphs1[lim]=mergeGraphs(graphs)
        else:
            graphs = extractSmoothedContourRL(h_lims_mu[lim], nSmooth)
            graphs1[lim]=mergeGraphs(graphs)
        #if model!="T2tt":
        #    graphs1[lim]=graphs[0]
        #else:
        #    graphs1[lim]=mergeGraphs(graphs)

        graphs1[lim].SetName( graphs1[lim].GetName().replace("_mu","") ) 
        graphs1[lim].Write()


    print "saving x-check plots"
    plotsDir = "xcheckPlots"
    can = ROOT.TCanvas("can","can",600,600)
    if( not os.path.isdir(plotsDir) ):
        os.system("mkdir "+plotsDir)
    for lim in limits:
        ROOT.gStyle.SetNumberContours( 100 )
        xmin = 600 if ("T1qqqq" == model or "T1tttt" == model) else 800 if "T1bbbb" == model else 150 if model=="T2tt" or model=='T2cc' else 300 if model=="T2bb" else 550 if model=="T2qq" else 0
        xmax = 1500 if (model=="T2tt" or model=="T2bb") else 2000 if model=="T2qq" else 800 if model=="T2cc" else 2500 if model=="T1bbbb" else 2400
        ymax = 700 if model=="T2tt" else 1200 if model=="T2bb" or model=="T2cc" else 1600 if model=="T2qq" else 2000 if (model=="T1qqqq" or model=="T1tttt") else 2200 if model=="T1bbbb" else 2000
        print "xmin={} xmax={} ymax={}".format(xmin, xmax, ymax)
        h_lims_yn0[lim].GetXaxis().SetRangeUser(xmin, xmax)
        h_lims_yn0[lim].GetYaxis().SetRangeUser(0   , ymax)
        h_lims_yn0[lim].Draw("colz")
        graphs0[lim].SetLineWidth(2)
        graphs0[lim].Draw("same")
        graphs1[lim].SetLineWidth(2)
        graphs1[lim].SetLineColor(2)
        graphs1[lim].Draw("same")
        can.SaveAs(plotsDir+"/" + model + "_" + lim + ".eps")
        can.SaveAs(plotsDir+"/" + model + "_" + lim + ".png")
        can.SaveAs(plotsDir+"/" + model + "_" + lim + ".pdf")


    print "file "+output+" saved"
    #fout.Close()

# so graph goes below mLSP=0 and there is no cut off above zero
def fillHorizontalBelowZero(hist):
    for ix in range(1,hist.GetNbinsX()+1):
        hist.SetBinContent( ix,1,hist.GetBinContent(ix,2) )
