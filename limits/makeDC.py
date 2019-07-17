import ROOT as rt

##################
# PREPARE INPUTS #
##################

fin = rt.TFile('hnl_m_12_low_b3.root')

can = fin.Get('can')
pad = can.GetPrimitive('can_1')

M2_Vp002   = pad.GetListOfPrimitives()[1]; assert 'M2_V.002'   in M2_Vp002  .GetName(); M2_Vp002  .SetName('M2_Vp002'  )
M2_Vp022   = pad.GetListOfPrimitives()[2]; assert 'M2_V.022'   in M2_Vp022  .GetName(); M2_Vp022  .SetName('M2_Vp022'  )
M5_Vp002   = pad.GetListOfPrimitives()[3]; assert 'M5_V.002'   in M5_Vp002  .GetName(); M5_Vp002  .SetName('M5_Vp002'  )
M5_Vp010   = pad.GetListOfPrimitives()[4]; assert 'M5_V.010'   in M5_Vp010  .GetName(); M5_Vp010  .SetName('M5_Vp010'  )
M8_Vp002   = pad.GetListOfPrimitives()[5]; assert 'M8_V.002'   in M8_Vp002  .GetName(); M8_Vp002  .SetName('M8_Vp002'  )
M8_Vp005   = pad.GetListOfPrimitives()[6]; assert 'M8_V.005'   in M8_Vp005  .GetName(); M8_Vp005  .SetName('M8_Vp005'  )
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


