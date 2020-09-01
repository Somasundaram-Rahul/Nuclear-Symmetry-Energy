import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import gvar as gv
import lsqfit
from scipy.optimize import curve_fit

d=np.zeros([35,11])
e=np.zeros([35,11,6])


################ Read data ##############################

for i in range(0,10):
    f = np.loadtxt("data/Jerome_EOS_Drischler/EOS_spec_4_beta_0."+str(i)+".txt")
    d[:,i] = f[:,0]
    for k in range(0,6):
        if k == 5:
            e[:,i,k] = f[:,k+2]
        else:
            e[:,i,k] = f[:,k+1]


f = np.loadtxt("data/Jerome_EOS_Drischler/EOS_spec_4_beta_1.0.txt")     
d[:,10] = f[:,0]
for k in range(0,6):
    if k == 5:
        e[:,10,k] = f[:,k+2]
    else:
        e[:,10,k] = f[:,k+1]



######## Numerical calculation of e_sat... e_symnq (INTERPOLATION) ############

td = np.arange(0.02, 0.2, 0.01)
te = np.zeros([td.size,11,6])

for i in range(0,11):
    for k in range(0,6):
        tck = interpolate.splrep(d[:,i], e[:,i,k])
        te[:,i,k] = interpolate.splev(td, tck, der=0)

e_sat = te[:,0,:] 
e_NM = te[:,10,:]
e_sym = te[:,10,:] - te[:,0,:] 
#e_sym2 = (te[:,1,:] - te[:,0,:])/ (0.01) 

def fit(delta,e0,e2,e4):
    return e0 + e2*delta**2 + e4*delta**4

delta_max = 6
indices = np.arange(0,delta_max)
delta = indices*0.1

e_sym2 = np.zeros([td.size,6])

for i in range(td.size):
    for h in range(6):
        popt, pcov  = curve_fit(fit, delta,  np.take(te[i,:,h],indices) )
        e_sym2[i,h] = popt[1] 
        
e_symnq = e_sym-e_sym2

#### Kinetic energy function ##############

def T(x,y):
    m = 938.919
    hbar = 197.3
    e_F = (  hbar**2 / (2*m) )  *  (1.5* np.pi**2 * x )**(2./3.) 
    f = (1+y)**(5/3) + (1-y)**(5/3)
    ans = 3/5 * e_F * f/2
    return ans


#Linear fit !!
def m_e_inv_SM(x):
     k1 =  gv.gvar ('3.33 (18)')
     return 1+ x*k1   

#Linear fit !!    
def m_e_inv_NM(x):
    k1 =  gv.gvar ('0.89 (19)')
    return 1+ x*k1  


def T_SM (x):
    m = 938.919
    hbar = 197.3
    ans  = 3*hbar**2/(10*m) * (3*np.pi*np.pi*x/2)**(2/3)
    return ans
    
def T_NM (x):
    m = 938.919
    hbar = 197.3
    ans  = 3*hbar**2/(10*m) * (3*np.pi*np.pi*x)**(2/3)
    return ans
    
def T_SM_eff (x):
    return T_SM(x)*m_e_inv_SM(x)

def T_NM_eff (x):
    return T_NM(x)*m_e_inv_NM(x)

def T_2_eff(x):
    return T_SM(x)* 5/9 * ( m_e_inv_SM(x) + 3*( m_e_inv_NM(x) - m_e_inv_SM (x) )     ) 

####### Expansion in eta (calculation via fits and not finite diffference) #############

eta= np.arange(0,4,1)
eta = 0.1 *eta

te_eta = np.zeros([td.size,eta.size,6])   #second index is for eta

c1 = np.zeros([td.size,6])   #stores answer from polynomial fit in eta
c2 = np.zeros([td.size,6])   #stores answer from polynomial fit in eta

for i in range(td.size):
    for k in range(6):
        for j in range(eta.size):
            te_eta[i,j,k] = te[i,10-j,k] 

order = 2
for i in range(td.size):
    for k in range(6):
        p = np.polyfit(eta,te_eta[i,:,k],order)
        c1[i,k] = p[order-1] 
        c2[i,k] = p[order-2] 
        
esym2_eta = np.zeros([td.size,6])
esymnq_eta = np.zeros([td.size,6])

for i in range(td.size):
    for h in range(6):
#        esym2_eta[i,h] = 2*e_sym[i,h]   + c1[i,h]/2
        esym2_eta[i,h] = -(3*c1[i,h] + 2*c2[i,h] )/4
        esymnq_eta[i,h] = e_sym[i,h] - esym2_eta[i,h]
        
###################### Fits for e_sym2 (delta)  #############################


e_sym2_av =[]


for h in range(6):
    e_sym2_av.append ( e_sym2[:,h]  )
    
##### Data for plottting purposes


e_sym2_pot_av = gv.dataset.avg_data(e_sym2_av,spread=True)- 5/9*T_SM(td)

e_sym2_pot_eff_av = gv.dataset.avg_data(e_sym2_av,spread=True) - T_2_eff(td)



##### Data for Fitting purposes

s = gv.dataset.svd_diagnosis(e_sym2_av)
e_sym2_av = gv.dataset.avg_data(e_sym2_av,spread=True)

e,ev = np.linalg.eig (gv.evalcorr (e_sym2_av) )
d2 = np.std(  np.absolute(ev[0]) ) **2 
print ("N (delta) = ",np.size(td))
print ("l_corr (delta) = ", 1 - np.size(td)*d2 )

def u(alpha,x):
    N=4
    b_sat = 17
    return 1 - (-3*x)**(N+1-alpha) * np.exp(-b_sat*(1+3*x))


def V2(den,p):
    b_sym = 42 - 17
    x = (den-p['n_sat'])/(3*p['n_sat']) 
    
    v0_sat = p['E_sat'] - T_SM(p['n_sat'])
    v1_sat = -2*T_SM(p['n_sat'])
    v2_sat = p['K_sat'] + 2*T_SM(p['n_sat'])
    v3_sat = p['Q_sat'] -8*T_SM(p['n_sat'])
    v4_sat = p['Z_sat'] + 56*T_SM(p['n_sat'])
    
    v0_sym2 = p['E_sym2'] - 5/9 * T_SM(p['n_sat'])
    v1_sym2 = p['L_sym2'] - 10/9  *T_SM(p['n_sat'])
    v2_sym2 = p['K_sym2'] + 10/9  *T_SM(p['n_sat'])
    v3_sym2 = p['Q_sym2'] - 40/9  *T_SM(p['n_sat'])
    v4_sym2 = p['Z_sym2'] + 280/9  *T_SM(p['n_sat'])
    
    ans = (  v0_sym2*u(0,x) - v0_sat*(u(0,x)-1)*b_sym*(1+3*x) )* x**(0)/np.math.factorial(0)
    ans = ans + (  v1_sym2*u(1,x) - v1_sat*(u(1,x)-1)*b_sym*(1+3*x) )* x**(1)/np.math.factorial(1)
    ans = ans + (  v2_sym2*u(2,x) - v2_sat*(u(2,x)-1)*b_sym*(1+3*x) )* x**(2)/np.math.factorial(2)
    ans = ans + (  v3_sym2*u(3,x) - v3_sat*(u(3,x)-1)*b_sym*(1+3*x) )* x**(3)/np.math.factorial(3)
    ans = ans + (  v4_sym2*u(4,x) - v4_sat*(u(4,x)-1)*b_sym*(1+3*x) )* x**(4)/np.math.factorial(4)
    
    return ans
    
def f_esym2_c(den,p):
    return 5/9*T_SM(den) + V2(den,p)
    

prior_esym2 = {}
prior_esym2['n_sat'] =  gv.gvar ('0.1606(74)')    #Sat parameters taken from posterior of SM_3* (with fixed b) 
prior_esym2['E_sat'] = gv.gvar ('-15.17(57)')
prior_esym2['K_sat'] = gv.gvar ('226(18)')
prior_esym2['Q_sat'] = gv.gvar ('-306(186)')
prior_esym2['Z_sat'] = gv.gvar (' 1324(497)')

prior_esym2['E_sym2'] = gv.gvar ('31.5(3.5)')
prior_esym2['L_sym2'] = gv.gvar ('50(10)')
prior_esym2['K_sym2'] = gv.gvar ('-130 (110)')
prior_esym2['Q_sym2'] = gv.gvar ('-300(600)')
prior_esym2['Z_sym2'] = gv.gvar ('-1800(800)')


x = td
y = e_sym2_av
fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior_esym2, fcn=f_esym2_c, debug=True,svdcut=s.svdcut)
e_sym2_par = fit.p

fig, axes =  plt.subplots(3, 3  , sharex='col', sharey='row',figsize=(15,8))

#Delta e_{sym,2}

axes[0,0].errorbar( td, gv.mean(e_sym2_av), gv.sdev(e_sym2_av),fmt='ob', label='data (68% CL)'  )
axes[0,0].fill_between (td,gv.mean(f_esym2_c(td,e_sym2_par))+gv.sdev(f_esym2_c(td,e_sym2_par)),gv.mean(f_esym2_c(td,e_sym2_par))-gv.sdev(f_esym2_c(td,e_sym2_par)),label='fit (68% CL)',color='red',alpha=0.8)
axes[0,0].fill_between (td,gv.mean(f_esym2_c(td,e_sym2_par))+2*gv.sdev(f_esym2_c(td,e_sym2_par)),gv.mean(f_esym2_c(td,e_sym2_par))-2*gv.sdev(f_esym2_c(td,e_sym2_par)),label='fit (95% CL)',color='red',alpha=0.2)

axes[0,0].set_ylabel('$e_{\mathrm{sym,2}}$ (MeV)',fontsize='15')
axes[0,0].text(0.1, 0.9, 'Delta expansion',fontsize='14' , transform = axes[0,0].transAxes)
axes[0,0].tick_params(labelsize='14')
axes[0,0].tick_params(right=True)
axes[0,0].tick_params(top=True)
axes[0,0].tick_params(direction='in')

#Delta e_{sym,2}^{pot}

axes[0,1].errorbar( td, gv.mean(e_sym2_pot_av), gv.sdev(e_sym2_pot_av),fmt='ob', label='data (68% CL)'  )
axes[0,1].fill_between (td,gv.mean(f_esym2_c(td,e_sym2_par)- 5/9*T_SM(td))+gv.sdev(f_esym2_c(td,e_sym2_par)- 5/9*T_SM(td)),gv.mean(f_esym2_c(td,e_sym2_par)- 5/9*T_SM(td))-gv.sdev(f_esym2_c(td,e_sym2_par)- 5/9*T_SM(td)),label='fit (68% CL)',color='red',alpha=0.8)
axes[0,1].fill_between (td,gv.mean(f_esym2_c(td,e_sym2_par)- 5/9*T_SM(td))+2*gv.sdev(f_esym2_c(td,e_sym2_par)- 5/9*T_SM(td)),gv.mean(f_esym2_c(td,e_sym2_par)- 5/9*T_SM(td))-2*gv.sdev(f_esym2_c(td,e_sym2_par)- 5/9*T_SM(td)),label='fit (95% CL)',color='red',alpha=0.2)

axes[0,1].set_ylabel('$e_{\mathrm{sym,2}}^{\mathrm{pot}}$ (MeV)',fontsize='15')
axes[0,1].text(0.1, 0.9, 'Delta expansion',fontsize='14' , transform = axes[0,1].transAxes)
axes[0,1].tick_params(right=True)
axes[0,1].tick_params(top=True)
axes[0,1].tick_params(direction='in')

#Delta e_{sym,2}^{pot*}

axes[0,2].errorbar( td, gv.mean(e_sym2_pot_eff_av), gv.sdev(e_sym2_pot_eff_av),fmt='ob', label='data (68% CL)'  )
axes[0,2].fill_between (td,gv.mean(f_esym2_c(td,e_sym2_par)- T_2_eff(td))+gv.sdev(f_esym2_c(td,e_sym2_par)- T_2_eff(td)),gv.mean(f_esym2_c(td,e_sym2_par)- T_2_eff(td))-gv.sdev(f_esym2_c(td,e_sym2_par)- T_2_eff(td)),label='fit (68% CL)',color='red',alpha=0.8)
axes[0,2].fill_between (td,gv.mean(f_esym2_c(td,e_sym2_par)- T_2_eff(td))+2*gv.sdev(f_esym2_c(td,e_sym2_par)- T_2_eff(td)),gv.mean(f_esym2_c(td,e_sym2_par)- T_2_eff(td))-2*gv.sdev(f_esym2_c(td,e_sym2_par)- T_2_eff(td)),label='fit (95% CL)',color='red',alpha=0.2)

axes[0,2].set_ylabel('$e_{\mathrm{sym,2}}^{\mathrm{pot*}}$ (MeV)',fontsize='15')
axes[0,2].text(0.1, 0.9, 'Delta expansion',fontsize='14' , transform = axes[0,2].transAxes)
axes[0,2].legend(loc='lower right',fontsize='12')
axes[0,2].tick_params(right=True)
axes[0,2].tick_params(top=True)
axes[0,2].tick_params(direction='in')


###################### Fits for e_sym2 (eta)  #############################

e_sym2_eta_av =[]


for h in range(6):
    e_sym2_eta_av.append ( esym2_eta[:,h]  )
    
##### Data for plottting purposes


e_sym2_eta_pot_av = gv.dataset.avg_data(e_sym2_eta_av,spread=True)- 5/9*T_SM(td)

e_sym2_eta_pot_eff_av = gv.dataset.avg_data(e_sym2_eta_av,spread=True) -T_2_eff(td)



##### Data for Fitting purposes

s_eta = gv.dataset.svd_diagnosis(e_sym2_eta_av)
e_sym2_eta_av = gv.dataset.avg_data(e_sym2_eta_av,spread=True)

e,ev = np.linalg.eig (gv.evalcorr (e_sym2_eta_av) )
d2 = np.std(  np.absolute(ev[0]) ) **2 
print ("N (eta) = ",np.size(td))
print ("l_corr (eta) = ", 1 - np.size(td)*d2 )

x = td
y = e_sym2_eta_av
fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior_esym2, fcn=f_esym2_c, debug=True,svdcut=s_eta.svdcut)
e_sym2_eta_par = fit.p


# Eta e_{sym,2}

axes[1,0].errorbar( td, gv.mean(e_sym2_eta_av), gv.sdev(e_sym2_eta_av),fmt='ob', label='data (68% CL)'  )
axes[1,0].fill_between (td,gv.mean(f_esym2_c(td,e_sym2_eta_par))+gv.sdev(f_esym2_c(td,e_sym2_eta_par)),gv.mean(f_esym2_c(td,e_sym2_eta_par))-gv.sdev(f_esym2_c(td,e_sym2_eta_par)),label='fit (68% CL)',color='red',alpha=0.8)
axes[1,0].fill_between (td,gv.mean(f_esym2_c(td,e_sym2_eta_par))+2*gv.sdev(f_esym2_c(td,e_sym2_eta_par)),gv.mean(f_esym2_c(td,e_sym2_eta_par))-2*gv.sdev(f_esym2_c(td,e_sym2_eta_par)),label='fit (95% CL)',color='red',alpha=0.2)

axes[1,0].set_ylabel('$e_{\mathrm{sym,2}}^{\mathrm{PNM}}$ (MeV)',fontsize='15')
axes[1,0].text(0.1, 0.9, 'Eta expansion' ,fontsize='14', transform = axes[1,0].transAxes)
axes[1,0].tick_params(labelsize='14')
axes[1,0].tick_params(right=True)
axes[1,0].tick_params(top=True)
axes[1,0].tick_params(direction='in')

# Eta e_{sym,2}^{pot}

axes[1,1].errorbar( td, gv.mean(e_sym2_eta_pot_av), gv.sdev(e_sym2_eta_pot_av),fmt='ob', label='data (68% CL)'  )
axes[1,1].fill_between (td,gv.mean(f_esym2_c(td,e_sym2_eta_par)- 5/9*T_SM(td))+gv.sdev(f_esym2_c(td,e_sym2_eta_par)- 5/9*T_SM(td)),gv.mean(f_esym2_c(td,e_sym2_eta_par)- 5/9*T_SM(td))-gv.sdev(f_esym2_c(td,e_sym2_eta_par)- 5/9*T_SM(td)),label='fit (68% CL)',color='red',alpha=0.8)
axes[1,1].fill_between (td,gv.mean(f_esym2_c(td,e_sym2_eta_par)- 5/9*T_SM(td))+2*gv.sdev(f_esym2_c(td,e_sym2_eta_par)- 5/9*T_SM(td)),gv.mean(f_esym2_c(td,e_sym2_eta_par)- 5/9*T_SM(td))-2*gv.sdev(f_esym2_c(td,e_sym2_eta_par)- 5/9*T_SM(td)),label='fit (95% CL)',color='red',alpha=0.2)

axes[1,1].set_ylabel('$e_{\mathrm{sym,2}}^{\mathrm{pot,PNM}}$ (MeV)',fontsize='15')
axes[1,1].text(0.1, 0.9, 'Eta expansion',fontsize='14' , transform = axes[1,1].transAxes)
axes[1,1].tick_params(right=True)
axes[1,1].tick_params(top=True)
axes[1,1].tick_params(direction='in')

# Eta e_{sym,2}^{pot*}

axes[1,2].errorbar( td, gv.mean(e_sym2_eta_pot_eff_av), gv.sdev(e_sym2_eta_pot_eff_av),fmt='ob', label='data (68% CL)'  )
axes[1,2].fill_between (td,gv.mean(f_esym2_c(td,e_sym2_eta_par)- T_2_eff(td))+gv.sdev(f_esym2_c(td,e_sym2_eta_par)- T_2_eff(td)),gv.mean(f_esym2_c(td,e_sym2_eta_par)-T_2_eff(td))-gv.sdev(f_esym2_c(td,e_sym2_eta_par)- T_2_eff(td)),label='fit (68% CL)',color='red',alpha=0.8)
axes[1,2].fill_between (td,gv.mean(f_esym2_c(td,e_sym2_eta_par)- T_2_eff(td))+2*gv.sdev(f_esym2_c(td,e_sym2_eta_par)- T_2_eff(td)),gv.mean(f_esym2_c(td,e_sym2_eta_par)- T_2_eff(td))-2*gv.sdev(f_esym2_c(td,e_sym2_eta_par)- T_2_eff(td)),label='fit (95% CL)',color='red',alpha=0.2)

axes[1,2].set_ylabel('$e_{\mathrm{sym,2}}^{\mathrm{pot*,PNM}}$ (MeV)',fontsize='15')
axes[1,2].text(0.1, 0.9, 'Eta expansion',fontsize='14' , transform = axes[1,2].transAxes)
axes[1,2].tick_params(right=True)
axes[1,2].tick_params(top=True)
axes[1,2].tick_params(direction='in')


### Difference

diff_data = e_sym2_av - e_sym2_eta_av
diff_fit = f_esym2_c(td,e_sym2_par) - f_esym2_c(td,e_sym2_eta_par)

axes[2,0].errorbar( td, gv.mean(diff_data), gv.sdev(diff_data),fmt='ob', label='data (68% CL)'  )
axes[2,0].fill_between (td,gv.mean(diff_fit)+gv.sdev(diff_fit),gv.mean(diff_fit)-gv.sdev(diff_fit),label='fit (68% CL)',color='red',alpha=0.8)
axes[2,0].fill_between (td,gv.mean(diff_fit)+2*gv.sdev(diff_fit),gv.mean(diff_fit)-2*gv.sdev(diff_fit),label='fit (95% CL)',color='red',alpha=0.2)
axes[2,0].axhline(color='black')
axes[2,0].set_xlabel('$n$ (fm$^{-3}$)',fontsize='14')
axes[2,0].set_ylabel('Difference (MeV)',fontsize='14')
axes[2,0].tick_params(labelsize='14')
axes[2,0].set_xticks(np.arange(0, 0.2+0.01, 0.05))
axes[2,0].tick_params(right=True)
axes[2,0].tick_params(top=True)
axes[2,0].tick_params(direction='in')

### Difference_pot

diff_data = e_sym2_pot_av - e_sym2_eta_pot_av
diff_fit = f_esym2_c(td,e_sym2_par)- 5/9*T_SM(td) - f_esym2_c(td,e_sym2_eta_par)+ 5/9*T_SM(td)

axes[2,1].errorbar( td, gv.mean(diff_data), gv.sdev(diff_data),fmt='ob', label='data (68% CL)'  )
axes[2,1].fill_between (td,gv.mean(diff_fit)+gv.sdev(diff_fit),gv.mean(diff_fit)-gv.sdev(diff_fit),label='fit (68% CL)',color='red',alpha=0.8)
axes[2,1].fill_between (td,gv.mean(diff_fit)+2*gv.sdev(diff_fit),gv.mean(diff_fit)-2*gv.sdev(diff_fit),label='fit (95% CL)',color='red',alpha=0.2)

axes[2,1].axhline(color='black')
axes[2,1].set_xlabel('$n$ (fm$^{-3}$)',fontsize='14')
axes[2,1].set_ylabel('Difference (MeV)',fontsize='14')
axes[2,1].tick_params(labelsize='14')
axes[2,1].set_xticks(np.arange(0, 0.2+0.01, 0.05))
axes[2,1].tick_params(right=True)
axes[2,1].tick_params(top=True)
axes[2,1].tick_params(direction='in')

### Difference_pot_eff

diff_data = e_sym2_pot_eff_av - e_sym2_eta_pot_eff_av
diff_fit = f_esym2_c(td,e_sym2_par)- T_2_eff(td) - f_esym2_c(td,e_sym2_eta_par)+ T_2_eff(td)

axes[2,2].errorbar( td, gv.mean(diff_data), gv.sdev(diff_data),fmt='ob', label='data (68% CL)'  )
axes[2,2].fill_between (td,gv.mean(diff_fit)+gv.sdev(diff_fit),gv.mean(diff_fit)-gv.sdev(diff_fit),label='fit (68% CL)',color='red',alpha=0.8)
axes[2,2].fill_between (td,gv.mean(diff_fit)+2*gv.sdev(diff_fit),gv.mean(diff_fit)-2*gv.sdev(diff_fit),label='fit (95% CL)',color='red',alpha=0.2)

axes[2,2].axhline(color='black')
axes[2,2].set_ylim(top=+3)
axes[2,2].set_ylim(bottom=-3)

axes[2,2].set_xlabel('$n$ (fm$^{-3}$)',fontsize='14')
axes[2,2].set_ylabel('Difference (MeV)',fontsize='14')
axes[2,2].tick_params(labelsize='14')
axes[2,2].set_xticks(np.arange(0, 0.2+0.01, 0.05))
axes[2,2].tick_params(right=True)
axes[2,2].tick_params(top=True)
axes[2,2].tick_params(direction='in')

plt.tight_layout()
fig.show()


print ('\n','-----Delta----')
print (e_sym2_par)
print ('\n','-----Eta-----')
print (e_sym2_eta_par)