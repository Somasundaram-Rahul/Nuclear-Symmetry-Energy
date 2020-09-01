import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import gvar as gv
import lsqfit
from sklearn.metrics import r2_score

spe = np.zeros([100,21,6,2])
spe_2 = np.zeros([100,21,6,2])
k = np.zeros([100,21,6,2])

########### Read data for SM ###############

h=1

for i in range(21):
    if (i<9) :
        f  = np.loadtxt("data/Effective mass/Symmetric matter/h1/SPE_n_0.0"+str(i+1)+"00000_x_0.5_N3LO_EM500_lam_1.80_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.26400_cE_-0.12000_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt" )
    else :
        f  = np.loadtxt("data/Effective mass/Symmetric matter/h1/SPE_n_0."+str(i+1)+"00000_x_0.5_N3LO_EM500_lam_1.80_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.26400_cE_-0.12000_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt" )
    k[:,i,h-1,0] = f[:,0]
    spe[:,i,h-1,0] = f[:,4]
    spe_2[:,i,h-1,0] = f[:,1] + f[:,2]    
    

h=2

for i in range(21):
    if (i<9) :
        f  = np.loadtxt("data/Effective mass/Symmetric matter/h2/SPE_n_0.0"+str(i+1)+"00000_x_0.5_N3LO_EM500_lam_2.00_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.27100_cE_-0.13100_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt"  )
    else :
        f  = np.loadtxt("data/Effective mass/Symmetric matter/h2/SPE_n_0."+str(i+1)+"00000_x_0.5_N3LO_EM500_lam_2.00_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.27100_cE_-0.13100_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt"  )
    k[:,i,h-1,0] = f[:,0]
    spe[:,i,h-1,0] = f[:,4]
    spe_2[:,i,h-1,0] = f[:,1] + f[:,2]    

h=3

for i in range(21):
    if (i<9) :
        f  = np.loadtxt("data/Effective mass/Symmetric matter/h3/SPE_n_0.0"+str(i+1)+"00000_x_0.5_N3LO_EM500_lam_2.00_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_-0.29200_cE_-0.59200_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.50000_L2_2.53387_nexp_4_nexpNN_4.txt"   )
    else :
        f  = np.loadtxt("data/Effective mass/Symmetric matter/h3/SPE_n_0."+str(i+1)+"00000_x_0.5_N3LO_EM500_lam_2.00_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_-0.29200_cE_-0.59200_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.50000_L2_2.53387_nexp_4_nexpNN_4.txt"   )
    k[:,i,h-1,0] = f[:,0]
    spe[:,i,h-1,0] = f[:,4]
    spe_2[:,i,h-1,0] = f[:,1] + f[:,2]    
    

h=4

for i in range(21):
    if (i<9) :
        f  = np.loadtxt("data/Effective mass/Symmetric matter/h4/SPE_n_0.0"+str(i+1)+"00000_x_0.5_N3LO_EM500_lam_2.20_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.21400_cE_-0.13700_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt"  )
    else :
        f  = np.loadtxt("data/Effective mass/Symmetric matter/h4/SPE_n_0."+str(i+1)+"00000_x_0.5_N3LO_EM500_lam_2.20_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.21400_cE_-0.13700_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt"  )
    k[:,i,h-1,0] = f[:,0]
    spe[:,i,h-1,0] = f[:,4]
    spe_2[:,i,h-1,0] = f[:,1] + f[:,2]    
    
h=5

for i in range(21):
    if (i<9) :
        f  = np.loadtxt("data/Effective mass/Symmetric matter/h5/SPE_n_0.0"+str(i+1)+"00000_x_0.5_N3LO_EM500_lam_2.80_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.27800_cE_-0.07800_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt"  )
    else :
        f  = np.loadtxt("data/Effective mass/Symmetric matter/h5/SPE_n_0."+str(i+1)+"00000_x_0.5_N3LO_EM500_lam_2.80_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.27800_cE_-0.07800_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt"  )
    k[:,i,h-1,0] = f[:,0]
    spe[:,i,h-1,0] = f[:,4]
    spe_2[:,i,h-1,0] = f[:,1] + f[:,2]    
    
h=6

for i in range(21):
    if (i<9) :
        f  = np.loadtxt("data/Effective mass/Symmetric matter/h6/SPE_n_0.0"+str(i+1)+"00000_x_0.5_N3LO_EM500_lam_2.00_Np_100_limit_10.0_c1_-0.14997_c3_-0.94322_c4_0.78141_cD_-3.00700_cE_-0.68600_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt"  )
    else :
        f  = np.loadtxt("data/Effective mass/Symmetric matter/h6/SPE_n_0."+str(i+1)+"00000_x_0.5_N3LO_EM500_lam_2.00_Np_100_limit_10.0_c1_-0.14997_c3_-0.94322_c4_0.78141_cD_-3.00700_cE_-0.68600_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt"   )
    k[:,i,h-1,0] = f[:,0]
    spe[:,i,h-1,0] = f[:,4]
    spe_2[:,i,h-1,0] = f[:,1] + f[:,2]    

########### Read data for NM ###############

h=1

for i in range(1,21):
    if (i<9) :
        f  = np.loadtxt("data/Effective mass/Neutron matter/h1/SPE_n_0.0"+str(i+1)+"00000_x_0.0_N3LO_EM500_lam_1.80_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.26400_cE_-0.12000_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt" )
    else :
        f  = np.loadtxt("data/Effective mass/Neutron matter/h1/SPE_n_0."+str(i+1)+"00000_x_0.0_N3LO_EM500_lam_1.80_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.26400_cE_-0.12000_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt" )
    k[:,i,h-1,1] = f[:,0]
    spe[:,i,h-1,1] = f[:,4]
    spe_2[:,i,h-1,1] = f[:,1] + f[:,2]      

h=2

for i in range(21):
    if (i<9) :
        f  = np.loadtxt("data/Effective mass/Neutron matter/h2/SPE_n_0.0"+str(i+1)+"00000_x_0.0_N3LO_EM500_lam_2.00_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.27100_cE_-0.13100_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt"  )
    else :
        f  = np.loadtxt("data/Effective mass/Neutron matter/h2/SPE_n_0."+str(i+1)+"00000_x_0.0_N3LO_EM500_lam_2.00_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.27100_cE_-0.13100_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt"  )
    k[:,i,h-1,1] = f[:,0]
    spe[:,i,h-1,1] = f[:,4]
    spe_2[:,i,h-1,1] = f[:,1] + f[:,2]    
    
            
### Rough fix for corrupt file for spe,,spe_2,k[:,0,0,1]
k[:,0,0,1] = k[:,0,1,1]
spe[:,0,0,1] = spe[:,0,1,1] 
spe_2[:,0,0,1] = spe_2[:,0,1,1] 

h=3

for i in range(21):
    if (i<9) :
        f  = np.loadtxt("data/Effective mass/Neutron matter/h3/SPE_n_0.0"+str(i+1)+"00000_x_0.0_N3LO_EM500_lam_2.00_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_-0.29200_cE_-0.59200_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.50000_L2_2.53387_nexp_4_nexpNN_4.txt"   )
    else :
        f  = np.loadtxt("data/Effective mass/Neutron matter/h3/SPE_n_0."+str(i+1)+"00000_x_0.0_N3LO_EM500_lam_2.00_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_-0.29200_cE_-0.59200_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.50000_L2_2.53387_nexp_4_nexpNN_4.txt"   )
    k[:,i,h-1,1] = f[:,0]
    spe[:,i,h-1,1] = f[:,4]
    spe_2[:,i,h-1,1] = f[:,1] + f[:,2]    
    

h=4

for i in range(21):
    if (i<9) :
        f  = np.loadtxt("data/Effective mass/Neutron matter/h4/SPE_n_0.0"+str(i+1)+"00000_x_0.0_N3LO_EM500_lam_2.20_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.21400_cE_-0.13700_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt"  )
    else :
        f  = np.loadtxt("data/Effective mass/Neutron matter/h4/SPE_n_0."+str(i+1)+"00000_x_0.0_N3LO_EM500_lam_2.20_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.21400_cE_-0.13700_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt"  )
    k[:,i,h-1,1] = f[:,0]
    spe[:,i,h-1,1] = f[:,4]
    spe_2[:,i,h-1,1] = f[:,1] + f[:,2]    
    
h=5

for i in range(21):
    if (i<9) :
        f  = np.loadtxt("data/Effective mass/Neutron matter/h5/SPE_n_0.0"+str(i+1)+"00000_x_0.0_N3LO_EM500_lam_2.80_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.27800_cE_-0.07800_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt"  )
    else :
        f  = np.loadtxt("data/Effective mass/Neutron matter/h5/SPE_n_0."+str(i+1)+"00000_x_0.0_N3LO_EM500_lam_2.80_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.27800_cE_-0.07800_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt"  )
    k[:,i,h-1,1] = f[:,0]
    spe[:,i,h-1,1] = f[:,4]
    spe_2[:,i,h-1,1] =f[:,1] + f[:,2]    
    
h=6

for i in range(21):
    if (i<9) :
        f  = np.loadtxt("data/Effective mass/Neutron matter/h6/SPE_n_0.0"+str(i+1)+"00000_x_0.0_N3LO_EM500_lam_2.00_Np_100_limit_10.0_c1_-0.14997_c3_-0.94322_c4_0.78141_cD_-3.00700_cE_-0.68600_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt"  )
    else :
        f  = np.loadtxt("data/Effective mass/Neutron matter/h6/SPE_n_0."+str(i+1)+"00000_x_0.0_N3LO_EM500_lam_2.00_Np_100_limit_10.0_c1_-0.14997_c3_-0.94322_c4_0.78141_cD_-3.00700_cE_-0.68600_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt"   )
    k[:,i,h-1,1] = f[:,0]
    spe[:,i,h-1,1] = f[:,4]
    spe_2[:,i,h-1,1] = f[:,1] + f[:,2]    
    
################################## Plot spe #############################################################

kf_SM = (1.5*np.pi*np.pi*0.16)**0.333  #at saturation density
kf_NM = (3*np.pi*np.pi*0.16)**0.333    #at saturation density


# fig = plt.figure(figsize=(7,5))
# for h in range(6):
#     if h==5:
#         plt.plot (k[:,15,h,0],spe[:,15,h,0]*197.3269631,label='H'+str(h+2)+'',color='C'+str(h)+'')
#         plt.plot (k[:,15,h,1],spe[:,15,h,1]*197.3269631,color='C'+str(h)+'')
#         break
#     plt.plot (k[:,15,h,0],spe[:,15,h,0]*197.3269631,label='H'+str(h+1)+'',color='C'+str(h)+'')
#     plt.plot (k[:,15,h,1],spe[:,15,h,1]*197.3269631,color='C'+str(h)+'')

# spe_max = np.zeros([100,2])
# spe_min = np.zeros([100,2])

# for i in range(100):
#     for j in range(2):
#         spe_max[i,j] = np.amax(spe[i,15,:,j])
#         spe_min[i,j] = np.amin(spe[i,15,:,j])

# plt.fill_between (k[:,15,0,0] , spe_max[:,0]*197.3269631, spe_min[:,0]*197.3269631, color='grey',alpha=0.5)
# plt.fill_between (k[:,15,0,1] , spe_max[:,1]*197.3269631, spe_min[:,1]*197.3269631, color='grey',alpha=0.5)

# plt.axvline(x=kf_SM,linestyle=':')
# plt.axvline(x=kf_NM,linestyle='-.')

# plt.legend()

# plt.xlim(right=2.0)
# plt.xlim(left=0)
# plt.ylim(top=50)
# plt.ylim(bottom=-110)

# plt.text(0.75,-68,'SNM',fontsize='15')
# plt.text(0.75,-9,'PNM',fontsize='15')

# plt.text(kf_SM+0.02,-105,'$k_{F,\mathrm{SNM}}$',fontsize='15')
# plt.text(kf_NM+0.02,-105,'$k_{F,\mathrm{PNM}}$',fontsize='15')
# plt.text(0.5,17,'NN+3N \n \n $n=n^{\mathrm{emp}}_{\mathrm{sat}}$',fontsize='15')

# plt.xlabel ("$k$ (fm$^{-1}$)",fontsize='15')
# plt.ylabel ("$\epsilon_n$ (MeV)",fontsize='15')
# plt.legend(fontsize='14')
# plt.xticks(fontsize='14' )
# plt.yticks(fontsize='14' )
# plt.tick_params(right=True)
# plt.tick_params(top=True)
# plt.tick_params(direction='in')
# plt.tight_layout()
# plt.show()  
# 
# 
# #### Plot spe_2  
# 
kf_SM = (1.5*np.pi*np.pi*0.16)**0.333  #at saturation density
kf_NM = (3*np.pi*np.pi*0.16)**0.333    #at saturation density


# fig = plt.figure(figsize=(7,5))
# for h in range(6):
#     if h==5:
#         plt.plot (k[:,15,h,0],spe_2[:,15,h,0]*197.3269631,label='H'+str(h+2)+'',color='C'+str(h)+'')
#         plt.plot (k[:,15,h,1],spe_2[:,15,h,1]*197.3269631,color='C'+str(h)+'')
#         break 
#     plt.plot (k[:,15,h,0],spe_2[:,15,h,0]*197.3269631,label='H'+str(h+1)+'',color='C'+str(h)+'')
#     plt.plot (k[:,15,h,1],spe_2[:,15,h,1]*197.3269631,color='C'+str(h)+'')

# spe2_max = np.zeros([100,2])
# spe2_min = np.zeros([100,2])

# for i in range(100):
#     for j in range(2):
#         spe2_max[i,j] = np.amax(spe_2[i,15,:,j])
#         spe2_min[i,j] = np.amin(spe_2[i,15,:,j])

# plt.fill_between (k[:,15,0,0] , spe2_max[:,0]*197.3269631, spe2_min[:,0]*197.3269631, color='grey',alpha=0.5)
# plt.fill_between (k[:,15,0,1] , spe2_max[:,1]*197.3269631, spe2_min[:,1]*197.3269631, color='grey',alpha=0.5)


# plt.axvline(x=kf_SM,linestyle=':')
# plt.axvline(x=kf_NM,linestyle='-.')

# plt.xlim(right=2.0)
# plt.xlim(left=0)
# plt.ylim(top=50)
# plt.ylim(bottom=-110)

# plt.text(0.45,-102.5,'SNM',fontsize='15')
# plt.text(0.3,-46,'PNM',fontsize='15')
# plt.text(0.25,17,'NN-only \n \n $n=n^{\mathrm{emp}}_{\mathrm{sat}}$',fontsize='15')
# plt.text(kf_SM+0.02,-105,'$k_{F,\mathrm{SNM}}$',fontsize='15')
# plt.text(kf_NM+0.02,-105,'$k_{F,\mathrm{PNM}}$',fontsize='15')

# plt.xlabel ("$k$ (fm$^{-1}$)",fontsize='15')
# plt.ylabel ("$\epsilon_n$ (MeV)",fontsize='15')
# plt.xticks(fontsize='14' )
# plt.yticks(fontsize='14' )
# plt.tick_params(right=True)
# plt.tick_params(top=True)
# plt.tick_params(direction='in')
# plt.tight_layout()
# plt.show()  
    
############################### Effective mass as function of k ###########################################

m_e  = np.zeros([100,21,6,2])
de_dk = np.zeros([100,21,6,2])

m_e_2  = np.zeros([100,21,6,2])
de_2_dk = np.zeros([100,21,6,2])

m = 4.758187377097

for delta in range(2):
    for h in range(6):
        for density in range(21):
            tck = interpolate.splrep(k[:,density,h,delta], spe[:,density,h,delta])
            de_dk[:,density,h,delta] = interpolate.splev(k[:,density,h,delta], tck, der=1)
            
            tck = interpolate.splrep(k[:,density,h,delta], spe_2[:,density,h,delta])
            de_2_dk[:,density,h,delta] = interpolate.splev(k[:,density,h,delta], tck, der=1)

m_e = de_dk**(-1) * k/m

m_e_2 = de_2_dk**(-1) * k/m


kf_SM = (1.5*np.pi*np.pi*0.16)**0.333  #at saturation density
kf_NM = (3*np.pi*np.pi*0.16)**0.333    #at saturation density


# fig = plt.figure(figsize=(7,5))
# for h in range(6):
#     if h==5:
#         plt.plot (k[:,15,h,0],m_e[:,15,h,0],label='H'+str(h+2)+'',color='C'+str(h)+'')
#         plt.plot (k[:,15,h,1],m_e[:,15,h,1],color='C'+str(h)+'')
#         break
#     plt.plot (k[:,15,h,0],m_e[:,15,h,0],label='H'+str(h+1)+'',color='C'+str(h)+'')
#     plt.plot (k[:,15,h,1],m_e[:,15,h,1],color='C'+str(h)+'')

# m_e_max = np.zeros([100,2])
# m_e_min = np.zeros([100,2])

# for i in range(100):
#     for j in range(2):
#         m_e_max[i,j] = np.amax(m_e[i,15,:,j])
#         m_e_min[i,j] = np.amin(m_e[i,15,:,j])

# plt.fill_between (k[:,15,0,0] , m_e_max[:,0], m_e_min[:,0], color='grey',alpha=0.4)
# plt.fill_between (k[:,15,0,1] , m_e_max[:,1], m_e_min[:,1], color='grey',alpha=0.4)

# plt.axvline(x=kf_SM,linestyle=':')
# plt.axvline(x=kf_NM,linestyle='-.')

# plt.xlim(left=0.01)
# plt.xlim(right=2)
# plt.ylim(bottom=0.5)
# plt.ylim(top=1)

# plt.text(0.75,0.63,'SNM',fontsize='15')
# plt.text(0.75,0.79,'PNM',fontsize='15')
# plt.text(kf_SM+0.02,0.52,'$k_{F,\mathrm{SNM}}$',fontsize='15')
# plt.text(kf_NM+0.02,0.52,'$k_{F,\mathrm{PNM}}$',fontsize='15')
# plt.text(0.45,0.88,'NN+3N \n \n $n=n^{\mathrm{emp}}_{\mathrm{sat}}$',fontsize='15')

# plt.xlabel ("$k$ (fm$^{-1}$)",fontsize='15')
# plt.ylabel ("$m_n^{*}/m$",fontsize='15')
# plt.legend(fontsize='14')
# plt.xticks(fontsize='14' )
# plt.yticks(fontsize='14' )
# plt.tick_params(right=True)
# plt.tick_params(top=True)
# plt.tick_params(direction='in')
# plt.tight_layout()
# plt.show()

# 
# ######
# 
# fig = plt.figure(figsize=(7,5))
# for h in range(6):
#     if h==5:
#         plt.plot (k[:,15,h,0],m_e_2[:,15,h,0],label='H'+str(h+2)+'',color='C'+str(h)+'')
#         plt.plot (k[:,15,h,1],m_e_2[:,15,h,1],color='C'+str(h)+'')
#         break
#     plt.plot (k[:,15,h,0],m_e_2[:,15,h,0],label='H'+str(h+1)+'',color='C'+str(h)+'')
#     plt.plot (k[:,15,h,1],m_e_2[:,15,h,1],color='C'+str(h)+'')

# m_e2_max = np.zeros([100,2])
# m_e2_min = np.zeros([100,2])

# for i in range(100):
#     for j in range(2):
#         m_e2_max[i,j] = np.amax(m_e_2[i,15,:,j])
#         m_e2_min[i,j] = np.amin(m_e_2[i,15,:,j])

# plt.fill_between (k[:,15,0,0] , m_e2_max[:,0], m_e2_min[:,0], color='grey',alpha=0.4)
# plt.fill_between (k[:,15,0,1] , m_e2_max[:,1], m_e2_min[:,1], color='grey',alpha=0.4)

# plt.axvline(x=kf_SM,linestyle=':')
# plt.axvline(x=kf_NM,linestyle='-.')

# plt.xlim(left=0.01)
# plt.xlim(right=2)
# plt.ylim(bottom=0.5)
# plt.ylim(top=1)

# plt.text(0.45,0.54,'SNM',fontsize='15')
# plt.text(0.3,0.8,'PNM',fontsize='15')
# plt.text(kf_SM+0.02,0.52,'$k_{F,\mathrm{SNM}}$',fontsize='15')
# plt.text(kf_NM+0.02,0.52,'$k_{F,\mathrm{PNM}}$',fontsize='15')
# plt.text(0.4,0.88,'NN-only \n \n $n=n^{\mathrm{emp}}_{\mathrm{sat}}$',fontsize='15')

# plt.xlabel ("$k$ (fm$^{-1}$)",fontsize='15')
# plt.ylabel ("$m_n^{*}/m$",fontsize='15')
# #plt.legend(fontsize='14')
# plt.xticks(fontsize='14' )
# plt.yticks(fontsize='14' )
# plt.tick_params(right=True)
# plt.tick_params(top=True)
# plt.tick_params(direction='in')
# plt.tight_layout()
# plt.show()

############################### Effective mass at k_F ####################################################

m_e_kf  = np.zeros([21,6,2])
de_dk_kf = np.zeros([21,6,2])

m_e_2_kf  = np.zeros([21,6,2])
de_2_dk_kf = np.zeros([21,6,2])

m = 4.758187377097

for delta in range(0,2):
    for h in range(0,6):
        for density in range(0,21):
            tck = interpolate.splrep(k[:,density,h,delta], spe[:,density,h,delta])
            tck2 = interpolate.splrep(k[:,density,h,delta], spe_2[:,density,h,delta])
            
            if delta == 0:
                kf = (1.5*np.pi**2 * (density+1)*0.01)**(1/3)
                
                de_dk_kf[density,h,delta] = interpolate.splev(kf, tck, der=1)
                m_e_kf[density,h,delta] = de_dk_kf[density,h,delta]**(-1) * kf/m
                
                de_2_dk_kf[density,h,delta] = interpolate.splev(kf, tck2, der=1)
                m_e_2_kf[density,h,delta] = de_2_dk_kf[density,h,delta]**(-1) * kf/m
                        
            if delta == 1:
                kf = (3*np.pi**2 * (density+1)*0.01)**(1/3)
                de_dk_kf[density,h,delta] = interpolate.splev(kf, tck, der=1)
                m_e_kf[density,h,delta] = de_dk_kf[density,h,delta]**(-1) * kf/m 
                    
                de_2_dk_kf[density,h,delta] = interpolate.splev(kf, tck2, der=1)
                m_e_2_kf[density,h,delta] = de_2_dk_kf[density,h,delta]**(-1) * kf/m      

density = np.arange(0.01,0.22,0.01)

fig = plt.figure(figsize=(7,5))
for h in range(6):
    if h==5:
        plt.plot (density, m_e_kf[:,h,0],label='H'+str(h+2)+'',color='C'+str(h)+'')
        plt.plot (density, m_e_kf[:,h,1],color='C'+str(h)+'')
        break
    plt.plot (density, m_e_kf[:,h,0],label='H'+str(h+1)+'',color='C'+str(h)+'')
    plt.plot (density, m_e_kf[:,h,1],color='C'+str(h)+'')

m_e_kf_max = np.zeros([np.size(density),2])
m_e_kf_min = np.zeros([np.size(density),2])

m_e_2_kf_max = np.zeros([np.size(density),2])
m_e_2_kf_min = np.zeros([np.size(density),2])

for i in range(np.size(density)):
    for j in range(2):
        m_e_kf_max[i,j] = np.amax(m_e_kf[i,:,j])
        m_e_kf_min[i,j] = np.amin(m_e_kf[i,:,j])
        
        m_e_2_kf_max[i,j] = np.amax(m_e_2_kf[i,:,j])
        m_e_2_kf_min[i,j] = np.amin(m_e_2_kf[i,:,j])

plt.fill_between (density , m_e_kf_max[:,0], m_e_kf_min[:,0], color='grey',alpha=0.4)
plt.fill_between (density , m_e_kf_max[:,1], m_e_kf_min[:,1], color='grey',alpha=0.4)

plt.plot (density , m_e_2_kf_max[:,0], 'k--')
plt.plot (density , m_e_2_kf_min[:,0], 'k--')
plt.plot (density , m_e_2_kf_max[:,1], 'k--')
plt.plot (density , m_e_2_kf_min[:,1], 'k--')

plt.text(0.05,0.69,'SNM',fontsize='15')
plt.text(0.05,0.91,'PNM',fontsize='15')
plt.text(0.025,0.97,'$k=k_F$',fontsize='15')

plt.arrow(0.153,0.75,0.16-0.153,0.795-0.75,width=0.002)
plt.text(0.14,0.73,'NN-only',fontsize='15')

plt.arrow(0.171,0.988,0.185-0.171,0.97-0.988,width=0.002)
plt.text(0.136,0.99,'NN+3N',fontsize='15')

plt.xlabel ("$n$ (fm$^{-3}$)",fontsize='15')
plt.ylabel ("$m_n^{*}/m$",fontsize='15')
plt.xticks(np.arange(0, 0.2+0.01, 0.05),fontsize='14' )
plt.yticks(fontsize='14' )
plt.tick_params(right=True)
plt.tick_params(top=True)
plt.tick_params(direction='in')
plt.tight_layout()
plt.show()
# 
# 
# # plot m/m^*
# 
fig = plt.figure(figsize=(7,5))
for h in range(6):
    if h==5:
        plt.plot (density, 1/m_e_kf[:,h,0],label='H'+str(h+2)+'',color='C'+str(h)+'')
        plt.plot (density, 1/m_e_kf[:,h,1],color='C'+str(h)+'')
        break
    plt.plot (density, 1/m_e_kf[:,h,0],label='H'+str(h+1)+'',color='C'+str(h)+'')
    plt.plot (density, 1/m_e_kf[:,h,1],color='C'+str(h)+'')
    

plt.fill_between (density , 1/m_e_kf_max[:,0], 1/m_e_kf_min[:,0], color='grey',alpha=0.4)
plt.fill_between (density , 1/m_e_kf_max[:,1], 1/m_e_kf_min[:,1], color='grey',alpha=0.4)

plt.plot (density , 1/m_e_2_kf_max[:,0], 'k--')      
plt.plot (density , 1/m_e_2_kf_min[:,0], 'k--')      
plt.plot (density , 1/m_e_2_kf_max[:,1], 'k--')      
plt.plot (density , 1/m_e_2_kf_min[:,1], 'k--')      
      
plt.text(0.05,1.06,'PNM',fontsize='15')
plt.text(0.05,1.4,'SNM',fontsize='15')
plt.text(0.05,1.66,'$k=k_F$',fontsize='15')

plt.arrow(0.173,1.34,0.18-0.173,1.27-1.34,width=0.002)
plt.text(0.14,1.35,'NN-only',fontsize='15')

plt.arrow(0.1539,1.02,0.161-0.1539,1.075-1.02,width=0.002)
plt.text(0.136,0.99,'NN+3N',fontsize='15')

plt.xlabel ("$n$ (fm$^{-3}$)",fontsize='15')
plt.ylabel ("$m/m_n^{*}$",fontsize='15')
plt.xticks(np.arange(0, 0.2+0.01, 0.05),fontsize='14' )
plt.yticks(fontsize='14' )
plt.tick_params(right=True)
plt.tick_params(top=True)
plt.tick_params(direction='in')
plt.legend(fontsize='14')  
plt.tight_layout()
plt.show()

################################# Obtain kappa_sat and kappa_NM ##########################################

min_den = 14
max_den = 17

m_e_inv_kf_red = np.zeros([max_den-min_den,6,2])

for i in range(min_den,max_den):
        m_e_inv_kf_red[i-min_den,:,:] = 1/m_e_kf[i,:,:]

y_SM_1 = []
for h in range(6):
          y_SM_1.append (m_e_inv_kf_red[:,h,0])


s = gv.dataset.svd_diagnosis(y_SM_1)
y_SM_1 = gv.dataset.avg_data(y_SM_1,spread=True)

e,ev = np.linalg.eig (gv.evalcorr (y_SM_1) )
d2 = np.std(  np.absolute(ev[0]) ) **2 
print ("l_corr (linear,SM) = ", 1 - (max_den-min_den)*d2  )

def f_SM_1 (x,p):
    ans =  1 + x*p['k1'] 
    return ans
    
prior_m_e_inv_SM_1 = {}
prior_m_e_inv_SM_1['k1'] = gv.gvar(0,100)


x = np.arange(min_den+1 , max_den+1, 1)
x = x*0.01

fit = lsqfit.nonlinear_fit(data=(x, y_SM_1), prior=prior_m_e_inv_SM_1, fcn=f_SM_1, debug=True
                              ,svdcut=s.svdcut,add_svdnoise=False)
#print (fit)

par_SM_1 = fit.p

### Quadratic Fit

min_den = 6     
max_den = 20

m_e_inv_kf_red = np.zeros([max_den-min_den,6,2])

for i in range(min_den,max_den):
        m_e_inv_kf_red[i-min_den,:,:] = 1/m_e_kf[i,:,:]

y_SM_2 = []
for h in range(6):
          y_SM_2.append (m_e_inv_kf_red[:,h,0])


s = gv.dataset.svd_diagnosis(y_SM_2)
y_SM_2 = gv.dataset.avg_data(y_SM_2,spread=True)

e,ev = np.linalg.eig (gv.evalcorr (y_SM_2) )
d2 = np.std(  np.absolute(ev[0]) ) **2 
print ("l_corr (quadratic,SM) = ", 1 - (max_den-min_den)*d2  )

def f_SM_2 (x,p):
    ans =  1 + x*p['k1']  + x**2 * p['k2']
    return ans
    
prior_m_e_inv_SM_2 = {}
prior_m_e_inv_SM_2['k1'] = gv.gvar(0,100)
prior_m_e_inv_SM_2['k2'] = gv.gvar(0,100)

x = np.arange(min_den+1 , max_den+1, 1)
x = x*0.01

fit = lsqfit.nonlinear_fit(data=(x, y_SM_2), prior=prior_m_e_inv_SM_2, fcn=f_SM_2, debug=True
                              ,svdcut=s.svdcut,add_svdnoise=False)

par_SM_2 = fit.p

##### QQ plot 
# 
# residuals = fit.residuals
# residuals = np.sort(residuals)
# np.random.seed(73568478)
# quantiles = np.random.normal(0, 1, np.size(residuals))
# quantiles = np.sort(quantiles)
# 
# r2 = r2_score(residuals, quantiles)
# r2 = np.around(r2,2)
# 
# z = np.polyfit(quantiles, residuals ,deg = 1 )
# p = np.poly1d(z)
# x = np.arange(np.min(quantiles),np.max(quantiles),0.001)
# 
# fig,ax = plt.subplots(1)
# plt.plot (quantiles, residuals, 'ob')
# plt.plot (x,p(x), color='blue')
# plt.plot (x,x,'r--')
# ax.text(0.1, 0.9, 'R ='+str(r2)+'' , transform = ax.transAxes,fontsize='13')
# plt.xlabel ('Theoretical quantiles',fontsize='15')
# plt.ylabel ('Ordered fit residuals',fontsize='15')
# ax.tick_params(labelsize='14')
# plt.show()

########

x_plot = np.arange(1,22)
x_plot = x_plot*0.01

y_plot = []
for h in range(6):
          y_plot.append (m_e_kf[:,h,0])

y_plot = gv.dataset.avg_data(y_plot,spread=True)

# fig = plt.figure()

# plt.errorbar(x_plot,gv.mean (y_plot) ,gv.sdev(y_plot), fmt='ok',label='data (68% CL)')

# plt.fill_between (x_plot, gv.mean(1/f_SM_2(x_plot,par_SM_2))+ gv.sdev(1/f_SM_2(x_plot,par_SM_2))
#                   ,gv.mean(1/f_SM_2(x_plot,par_SM_2))-gv.sdev(1/f_SM_2(x_plot,par_SM_2)),label='quadratic fit (68% CL)'
#                   ,color ='red', alpha=0.6)
                
# plt.fill_between (x_plot, gv.mean(1/f_SM_2(x_plot,par_SM_2))+ 2*gv.sdev(1/f_SM_2(x_plot,par_SM_2))
#                   ,gv.mean(1/f_SM_2(x_plot,par_SM_2))-2*gv.sdev(1/f_SM_2(x_plot,par_SM_2)),label='quadratic fit (95% CL)'
#                     ,color ='red', alpha=0.4)      

# plt.fill_between (x_plot, gv.mean(1/f_SM_1(x_plot,par_SM_1))+ gv.sdev(1/f_SM_1(x_plot,par_SM_1))
#                   ,gv.mean(1/f_SM_1(x_plot,par_SM_1))-gv.sdev(1/f_SM_1(x_plot,par_SM_1)),label='linear fit (68% CL)'
#                   ,color ='blue', alpha=0.6)
                
# plt.fill_between (x_plot, gv.mean(1/f_SM_1(x_plot,par_SM_1))+ 2*gv.sdev(1/f_SM_1(x_plot,par_SM_1))
#                   ,gv.mean(1/f_SM_1(x_plot,par_SM_1))-2*gv.sdev(1/f_SM_1(x_plot,par_SM_1)),label='linear fit (95% CL)'
#                     ,color ='blue', alpha=0.4)  

# for h in range(6):
#     plt.plot (x_plot, m_e_kf[:,h,0],color='k', alpha=0.6)     
  
# plt.text(0.212,1/m_e_inv_kf_red[max_den-min_den-1,0,0]-0.01,'H'+str(1)+'')
# plt.text(0.212,1/m_e_inv_kf_red[max_den-min_den-1,1,0]-0.008,'H'+str(2)+'')
# plt.text(0.212,1/m_e_inv_kf_red[max_den-min_den-1,2,0]-0.015,'H'+str(3)+'')
# plt.text(0.212,1/m_e_inv_kf_red[max_den-min_den-1,3,0]+0.008,'H'+str(4)+'')
# plt.text(0.212,1/m_e_inv_kf_red[max_den-min_den-1,4,0]+0.005,'H'+str(5)+'')
# plt.text(0.212,1/m_e_inv_kf_red[max_den-min_den-1,5,0]-0.003,'H'+str(7)+'')

# plt.xlabel ("$n$ (fm$^{-3}$)",fontsize='15')
# plt.ylabel ("$m_n^{*}/m$",fontsize='15')
# plt.text (0.02,0.95,'Symmetric matter',fontsize='15')
# plt.xticks(np.arange(0, 0.2+0.01, 0.05),fontsize='14' )
# plt.yticks(fontsize='14' )
# plt.tick_params(right=True)
# plt.tick_params(top=True)
# plt.tick_params(direction='in')

# plt.legend(loc='upper right',fontsize='14')  
# plt.show()



######## Neutron Matter ###############

min_den = 14
max_den = 17

m_e_inv_kf_red = np.zeros([max_den-min_den,6,2])

for i in range(min_den,max_den):
        m_e_inv_kf_red[i-min_den,:,:] = 1/m_e_kf[i,:,:]

y_NM_1 = []
for h in range(6):
          y_NM_1.append (m_e_inv_kf_red[:,h,1])


s = gv.dataset.svd_diagnosis(y_NM_1)
y_NM_1 = gv.dataset.avg_data(y_NM_1,spread=True)

e,ev = np.linalg.eig (gv.evalcorr (y_NM_1) )
d2 = np.std(  np.absolute(ev[0]) ) **2 
print ("l_corr (linear,NM) = ", 1 - (max_den-min_den)*d2  )

def f_NM_1 (x,p):
    ans =  1 + x*p['k1'] 
    return ans
    
prior_m_e_inv_NM_1 = {}
prior_m_e_inv_NM_1['k1'] = gv.gvar(0,100)


x = np.arange(min_den+1 , max_den+1, 1)
x = x*0.01

fit = lsqfit.nonlinear_fit(data=(x, y_NM_1), prior=prior_m_e_inv_NM_1, fcn=f_NM_1, debug=True
                              ,svdcut=0.25,add_svdnoise=False)
#print (fit)

par_NM_1 = fit.p

## Quadratic fit

min_den = 6     
max_den = 20

m_e_inv_kf_red = np.zeros([max_den-min_den,6,2])

for i in range(min_den,max_den):
        m_e_inv_kf_red[i-min_den,:,:] = 1/m_e_kf[i,:,:]

y_NM_2 = []
for h in range(6):
          y_NM_2.append (m_e_inv_kf_red[:,h,1])


s = gv.dataset.svd_diagnosis(y_NM_2)
y_NM_2 = gv.dataset.avg_data(y_NM_2,spread=True)

e,ev = np.linalg.eig (gv.evalcorr (y_NM_2) )
d2 = np.std(  np.absolute(ev[0]) ) **2 
print ("l_corr (quadratic,NM) = ", 1 - (max_den-min_den)*d2  )

def f_NM_2 (x,p):
    ans =  1 + x*p['k1']  + x**2 * p['k2']
    return ans
    
prior_m_e_inv_NM_2 = {}
prior_m_e_inv_NM_2['k1'] = gv.gvar(0,100)
prior_m_e_inv_NM_2['k2'] = gv.gvar(0,100)

x = np.arange(min_den+1 , max_den+1, 1)
x = x*0.01

fit = lsqfit.nonlinear_fit(data=(x, y_NM_2), prior=prior_m_e_inv_NM_2, fcn=f_NM_2, debug=True
                              ,svdcut=s.svdcut,add_svdnoise=False)
#print (fit)

par_NM_2 = fit.p

x_plot = np.arange(1,22)
x_plot = x_plot*0.01

y_plot = []
for h in range(6):
          y_plot.append (m_e_kf[:,h,1])

y_plot = gv.dataset.avg_data(y_plot,spread=True)

# fig = plt.figure()

# plt.errorbar(x_plot,gv.mean (y_plot) ,gv.sdev(y_plot), fmt='ok',label='data (68% CL)')

# plt.fill_between (x_plot, gv.mean(1/f_NM_2(x_plot,par_NM_2))+ gv.sdev(1/f_NM_2(x_plot,par_NM_2))
#                 ,gv.mean(1/f_NM_2(x_plot,par_NM_2))-gv.sdev(1/f_NM_2(x_plot,par_NM_2)),label='quadratic fit (68% CL)'
#                 ,color ='red', alpha=0.6)
                
# plt.fill_between (x_plot, gv.mean(1/f_NM_2(x_plot,par_NM_2))+ 2*gv.sdev(1/f_NM_2(x_plot,par_NM_2))
#                 ,gv.mean(1/f_NM_2(x_plot,par_NM_2))-2*gv.sdev(1/f_NM_2(x_plot,par_NM_2)),label='quadratic fit (95% CL)'
#                 ,color ='red', alpha=0.4)    

# plt.fill_between (x_plot, gv.mean(1/f_NM_1(x_plot,par_NM_1))+ gv.sdev(1/f_NM_1(x_plot,par_NM_1))
#                 ,gv.mean(1/f_NM_1(x_plot,par_NM_1))-gv.sdev(1/f_NM_1(x_plot,par_NM_1)),label='linear fit (68% CL)'
#                 ,color ='blue', alpha=0.6)
                
# plt.fill_between (x_plot, gv.mean(1/f_NM_1(x_plot,par_NM_1))+ 2*gv.sdev(1/f_NM_1(x_plot,par_NM_1))
#                 ,gv.mean(1/f_NM_1(x_plot,par_NM_1))-2*gv.sdev(1/f_NM_1(x_plot,par_NM_1)),label='linear fit (95% CL)'
#                 ,color ='blue', alpha=0.4)     
                              
# for h in range(6):
#     plt.plot (x_plot, m_e_kf[:,h,1],color='k', alpha=0.6)    

# plt.text(0.212,m_e_kf[20,0,1],'H'+str(1)+'')
# plt.text(0.212,m_e_kf[20,1,1]-0.005,'H'+str(2)+'')
# plt.text(0.212,m_e_kf[20,2,1],'H'+str(3)+'')
# plt.text(0.212,m_e_kf[20,3,1]-0.006,'H'+str(4)+'')
# plt.text(0.212,m_e_kf[20,4,1],'H'+str(5)+'')
# plt.text(0.212,m_e_kf[20,5,1],'H'+str(7)+'')

# #legend = plt.legend (loc='lower left')
# plt.xlabel ("$n$ (fm$^{-3}$)",fontsize='15')
# plt.ylabel ("$m_n^{*}/m$",fontsize='15')
# plt.text (0.06,1.03,'Neutron matter',fontsize='15')
# plt.xticks(np.arange(0, 0.2+0.01, 0.05),fontsize='14' )
# plt.yticks(fontsize='14' )
# plt.tick_params(right=True)
# plt.tick_params(top=True)
# plt.tick_params(direction='in')
# plt.show()

######################## Print Fit Parameter values #########################################
print ('\n -------------Fit parameters-------------')
print ('SNM')
print (par_SM_1)
print (par_SM_2)
print ('PNM')
print (par_NM_1)
print (par_NM_2)


