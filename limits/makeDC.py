import ROOT as rt

##################
# PREPARE INPUTS #
##################

in_folder = '/t3home/vstampf/eos/plots/limits/inputs/'
fin = rt.TFile(in_folder + 'hnl_m_12_low_b3.root')

can = fin.Get('can')
pad = can.GetPrimitive('can_1')

# M2_Vp002   = pad.GetListOfPrimitives()[1]; assert 'M2_V.002'   in M2_Vp002  .GetName(); M2_Vp002  .SetName('M2_Vp002'  )
# M2_Vp022   = pad.GetListOfPrimitives()[2]; assert 'M2_V.022'   in M2_Vp022  .GetName(); M2_Vp022  .SetName('M2_Vp022'  )
# M5_Vp002   = pad.GetListOfPrimitives()[3]; assert 'M5_V.002'   in M5_Vp002  .GetName(); M5_Vp002  .SetName('M5_Vp002'  )
# M5_Vp010   = pad.GetListOfPrimitives()[4]; assert 'M5_V.010'   in M5_Vp010  .GetName(); M5_Vp010  .SetName('M5_Vp010'  )
# M8_Vp002   = pad.GetListOfPrimitives()[5]; assert 'M8_V.002'   in M8_Vp002  .GetName(); M8_Vp002  .SetName('M8_Vp002'  )
# M8_Vp005   = pad.GetListOfPrimitives()[6]; assert 'M8_V.005'   in M8_Vp005  .GetName(); M8_Vp005  .SetName('M8_Vp005'  )


M2_Vp00244948974278_mu  = pad.getlistofprimitives()[1]; assert 'M2_V.0024'   in M2_Vp00244948974278_mu  .GetName(); M2_Vp002  .SetName('M2_Vp00244948974278_mu')
# M2_0.00282842712475_mu  = pad.getlistofprimitives()                            M2_0.00282842712475_mu                                M2_0.00282842712475_mu
# M2_0.00316227766017_mu  = pad.getlistofprimitives()                            M2_0.00316227766017_mu                                M2_0.00316227766017_mu
# M2_0.004472135955_mu    = pad.getlistofprimitives()                            M2_0.004472135955_mu                                  M2_0.004472135955_mu
# M2_0.00547722557505_mu  = pad.getlistofprimitives()                            M2_0.00547722557505_mu                                M2_0.00547722557505_mu
# M2_0.00707106781187_mu  = pad.getlistofprimitives()                            M2_0.00707106781187_mu                                M2_0.00707106781187_mu
# M2_0.00836660026534_mu  = pad.getlistofprimitives()                            M2_0.00836660026534_mu                                M2_0.00836660026534_mu
# M2_0.0141421356237_mu   = pad.getlistofprimitives()                            M2_0.0141421356237_mu                                 M2_0.0141421356237_mu
# M2_0.0173205080757_mu   = pad.getlistofprimitives()                            M2_0.0173205080757_mu                                 M2_0.0173205080757_mu
# M2_0.01_mu              = pad.getlistofprimitives()                            M2_0.01_mu                                            M2_0.01_mu
M2_Vp022360679775_mu    = pad.GetListOfPrimitives()[2]; assert 'M2_V.0223'   in M2_Vp022360679775_mu    .GetName(); M2_Vp022  .SetName('M2_Vp022360679775_mu')
M5_Vp00244948974278_mu  = pad.GetListOfPrimitives()[3]; assert 'M5_V.0024'   in M5_Vp00244948974278_mu  .GetName(); M5_Vp002  .SetName('M5_Vp00244948974278_mu')
# M5_0.00282842712475_mu  = pad.getlistofprimitives()                            M5_0.00282842712475_mu                                M5_0.00282842712475_mu
# M5_0.00316227766017_mu  = pad.getlistofprimitives()                            M5_0.00316227766017_mu                                M5_0.00316227766017_mu
# M5_0.004472135955_mu    = pad.getlistofprimitives()                            M5_0.004472135955_mu                                  M5_0.004472135955_mu
# M5_0.00547722557505_mu  = pad.getlistofprimitives()                            M5_0.00547722557505_mu                                M5_0.00547722557505_mu
# M5_0.00707106781187_mu  = pad.getlistofprimitives()                            M5_0.00707106781187_mu                                M5_0.00707106781187_mu
# M5_0.00836660026534_mu  = pad.getlistofprimitives()                            M5_0.00836660026534_mu                                M5_0.00836660026534_mu
M5_Vp01_mu              = pad.getlistofprimitives()[4]; assert 'M5_V.0100'   in M5_Vp01_mu              .GetName(); M5_Vp010  .SetName('M5_Vp01_mu')
M8_Vp00244948974278_mu  = pad.getlistofprimitives()[5]; assert 'M8_V.0024'   in M8_Vp002                .GetName(); M8_Vp002  .SetName('M8_Vp00244948974278_mu')
# M8_0.00282842712475_mu  = pad.getlistofprimitives()                            M8_0.00282842712475_mu                                M8_0.00282842712475_mu
# M8_0.00316227766017_mu  = pad.GetListOfPrimitives()                            M8_0.00316227766017_mu                                M8_0.00316227766017_mu
# M8_0.004472135955_mu    = pad.GetListOfPrimitives()                            M8_0.004472135955_mu                                  M8_0.004472135955_mu
M8_Vp00547722557505_mu  = pad.GetListOfPrimitives()[6]; assert 'M8_V.0054'   in M8_Vp00547722557505_mu  .GetName(); M8_Vp005  .SetName('M8_Vp00547722557505_mu')

data_2017B = pad.GetListOfPrimitives()[7]; assert 'data_2017B' in data_2017B.GetName(); data_2017B.SetName('data_obs')

stack = pad.GetPrimitive('hnl_m_12_low_b3_stack')

VV = stack.GetHists()[0]; assert 'WW' in VV.GetName(); VV.SetName('VV')
DY = stack.GetHists()[1]; assert 'DY' in DY.GetName(); DY.SetName('DY')


###################
# PREPARE OUTPUTS #
###################

fout = rt.TFile.Open('hnl_mll_b3.input.root', 'recreate')
fout.cd()
rt.gDirectory.mkdir('mll')
fout.cd('mll')

M2_Vp002.Write() 
M2_Vp022.Write()   
M5_Vp002.Write()   
M5_Vp010.Write()   
M8_Vp002.Write()   
M8_Vp005.Write()   

data_2017B.Write() 

VV.Write() 
DY.Write() 

fout.Close()

'''
M2_0.00244948974278_mu
M2_0.00282842712475_mu
M2_0.00316227766017_mu
M2_0.004472135955_mu
M2_0.00547722557505_mu
M2_0.00707106781187_mu
M2_0.00836660026534_mu
M2_0.0141421356237_mu
M2_0.0173205080757_mu
M2_0.01_mu
M2_0.022360679775_mu
M5_0.00244948974278_mu
M5_0.00282842712475_mu
M5_0.00316227766017_mu
M5_0.004472135955_mu
M5_0.00547722557505_mu
M5_0.00707106781187_mu
M5_0.00836660026534_mu
M5_0.01_mu
M8_0.00244948974278_mu
M8_0.00282842712475_mu
M8_0.00316227766017_mu
M8_0.004472135955_mu
M8_0.00547722557505_mu

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
