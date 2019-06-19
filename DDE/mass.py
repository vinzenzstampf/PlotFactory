import ROOT as rt

def mass():#l1_px, l1_py, l1_pz, l1_e, l2_px, l2_py, l2_pz, l2_e):

#      v1 = rt.TLorentzVector();
#      v2 = rt.TLorentzVector();
#      v1.SetPxPyPzE('l1_px','l1_py','l1_pz','l1_e');
#      v2.SetPxPyPzE('l2_px','l2_py','l2_pz','l2_e');
#      v3 = rt.TLorentzVector();
#      v3=v1+v2;
#      return  v3.M();
    return '(TLorentzVector(l1_px, l1_py, l1_pz, l1_e) + TLorentzVector(l2_px, l2_py, l2_pz, l2_e)).M()'
#    return 'l1_px + l2_px'
    

