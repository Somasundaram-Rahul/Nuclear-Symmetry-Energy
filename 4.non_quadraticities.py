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


#Linear fit !!
def m_e_inv_SM(x):
     k1 = gv.gvar ('3.33 (18)')
     return 1+ x*k1   

#Linear fit !!    
def m_e_inv_NM(x):
    k1 = gv.gvar ('0.89 (19)')
    return 1+ x*k1  
    
def T(x,y):
    m = 938.919
    hbar = 197.3
    e_F = (  hbar**2 / (2*m) )  *  (1.5* np.pi**2 * x )**(2./3.) 
    f = (1+y)**(5/3) + (1-y)**(5/3)
    ans = 3/5 * e_F * f/2
    return ans
    
def T_eff(x,y):
    m = 938.919
    hbar = 197.3
    e_F = (  hbar**2 / (2*m) )  *  (1.5* np.pi**2 * x )**(2./3.) 
    f = (m_e_inv_SM(x) + y* (m_e_inv_NM(x) - m_e_inv_SM(x)) )*(1+y)**(5/3) + (m_e_inv_SM(x) - y* (m_e_inv_NM(x) - m_e_inv_SM(x)) )*(1-y)**(5/3)
    ans = 3/5 * e_F * f/2
    return ans    

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

###################### interpolation####################################

td = np.arange(2, 21,)
td = td*0.01
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

te_SM_av=[]
te_NM_av=[]

for h in range(6):
    te_SM_av.append ( e_sat[:,h]   )
    te_NM_av.append ( e_NM[:,h]    )
    
ts_SM = gv.dataset.svd_diagnosis(te_SM_av)
te_SM_av = gv.dataset.avg_data(te_SM_av,spread=True)


ts_NM = gv.dataset.svd_diagnosis(te_NM_av)
te_NM_av = gv.dataset.avg_data(te_NM_av,spread=True)

############################ Fit for SM ##################################

def f_pot_SM(x,p):
    xt = (x-p['n_sat'])/(3*p['n_sat']) 
    
    v0 = p['E_sat'] - T_SM(p['n_sat'])
    v1 = -2*T_SM(p['n_sat'])
    v2 = p['K_sat'] + 2*T_SM(p['n_sat'])
    v3 = p['Q_sat'] -8*T_SM(p['n_sat'])
    v4 = p['Z_sat'] + 56*T_SM(p['n_sat'])
    
    ans = v0 + v1*xt + (v2/2.)*xt**2 + (v3/6.)*xt**3 + (v4/24.)*(xt)**4 
    return ans
    
def f_pot_SM_c(x,p):
    xt = (x-p['n_sat'])/(3*p['n_sat']) 
    lam = f_pot_SM(0,p) * 3.**5 
    beta=3
    return f_pot_SM(x,p)  + lam * xt**5  * np.exp( -17* (x/0.16)**(beta/3) )
    
def f_SM(x,p):
    return T_SM(x) + f_pot_SM_c(x,p)
    

# prior_e_SM = {}            # Drischler priors
# prior_e_SM['n_sat'] = gv.gvar(0.171, 0.016)
# prior_e_SM['E_sat'] = gv.gvar(-15.16, 1.24)
# prior_e_SM['K_sat'] = gv.gvar(214, 22)
# prior_e_SM['Q_sat'] = gv.gvar(-139, 104)
# prior_e_SM['Z_sat'] = gv.gvar(1306, 214)
# prior_e_SM['b'] = gv.gvar(0,50)

prior_e_SM = {}               # Jerome priors
prior_e_SM['n_sat'] = gv.gvar(0.16, 0.01)
prior_e_SM['E_sat'] = gv.gvar(-15.5,1.0)
prior_e_SM['K_sat'] = gv.gvar(230, 20)
prior_e_SM['Q_sat'] = gv.gvar(-300, 400)
prior_e_SM['Z_sat'] = gv.gvar(1300, 500)

x = td
y = te_SM_av

fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior_e_SM, fcn=f_SM, debug=True,svdcut=ts_SM.svdcut)
SM3_par = fit.p




############################ Fit for NM ##################################

def f_pot_NM(x,p):
    xt = (x-p['n_sat'])/(3*p['n_sat']) 
    
    v0 = p['E_sat+E_sym'] - T_SM(p['n_sat']) - T_SM(p['n_sat']) *(2**(2/3)-1)
    v1 = p['L_sym'] -2*T_SM(p['n_sat']) -2*T_SM(p['n_sat'])*(2**(2/3)-1)
    v2 = p['K_sat+K_sym'] + 2*T_SM(p['n_sat'])-2*T_SM(p['n_sat'])*(-1*2**(2/3)+1)
    v3 = p['Q_sat+Q_sym'] -8*T_SM(p['n_sat'])-2*T_SM(p['n_sat'])*(4*2**(2/3)-4)
    v4 = p['Z_sat+Z_sym'] + 56*T_SM(p['n_sat']) -8*T_SM(p['n_sat'])*(-7*2**(2/3)+7)
    
    ans = v0 + v1*xt + (v2/2.)*xt**2 + (v3/6.)*xt**3 + (v4/24.)*(xt)**4 
    return ans
    
def f_pot_NM_c(x,p):
    xt = (x-p['n_sat'])/(3*p['n_sat']) 
    lam = f_pot_NM(0,p) * 3.**5 
    beta=3
    return f_pot_NM(x,p)  + lam * xt**5  * np.exp( -42* (x/0.16)**(beta/3) )
    
def f_NM(x,p):
    return T_NM(x) + f_pot_NM_c(x,p)
    

    
# prior_eNM = {}     # Drischler priors
# prior_eNM['n_sat'] = SM1_par['n_sat']
# prior_eNM['E_sat+E_sym'] = gv.gvar(16.85, 3.33)
# prior_eNM['L_sym'] = gv.gvar(48.1, 3.6)
# prior_eNM['K_sat+K_sym'] = gv.gvar(42,62)
# prior_eNM['Q_sat+Q_sym'] = gv.gvar(-303, 338)
# prior_eNM['Z_sat+Z_sym'] = gv.gvar(-1011, 593)
# prior_eNM['b'] = gv.gvar(0,50)

prior_eNM = {}          # Jerome priors
prior_eNM['n_sat'] = SM3_par['n_sat']
prior_eNM['E_sat+E_sym'] = gv.gvar(16.0, 3.0)
prior_eNM['L_sym'] = gv.gvar(50, 10)
prior_eNM['K_sat+K_sym'] = gv.gvar(100,100)
prior_eNM['Q_sat+Q_sym'] = gv.gvar(0, 400)
prior_eNM['Z_sat+Z_sym'] = gv.gvar(-500, 500)

x = td
y = te_NM_av

fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior_eNM, fcn=f_NM, debug=True,svdcut=ts_NM.svdcut)
NM3_par = fit.p

############ E_sym #####################


e_sym_res = f_NM(td,NM3_par) - f_SM(td,SM3_par)
e_sym_pot_res = f_NM(td,NM3_par) - f_SM(td,SM3_par) - T_NM(td) + T_SM(td)
e_sym_pot_eff_res = f_NM(td,NM3_par) - f_SM(td,SM3_par) - T_NM_eff(td) + T_SM_eff(td)


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
esym4_eta = np.zeros([td.size,6])

for i in range(td.size):
    for h in range(6):
#        esym2_eta[i,h] = 2*e_sym[i,h]   + c1[i,h]/2
        esym2_eta[i,h] = -(3*c1[i,h] + 2*c2[i,h] )/4
        esym4_eta[i,h] = (c1[i,h] + 2*c2[i,h] )/8

###################### Fits for e_sym2 (delta)  #############################

e_sym2_av =[]


for h in range(6):
    e_sym2_av.append ( e_sym2[:,h]  )
    
##### Data for Fitting purposes

s = gv.dataset.svd_diagnosis(e_sym2_av)
e_sym2_av = gv.dataset.avg_data(e_sym2_av,spread=True)


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

###################### Fits for e_sym2 (eta)  #############################

e_sym2_eta_av =[]


for h in range(6):
    e_sym2_eta_av.append ( esym2_eta[:,h]  )
    

s_eta = gv.dataset.svd_diagnosis(e_sym2_eta_av)
e_sym2_eta_av = gv.dataset.avg_data(e_sym2_eta_av,spread=True)

x = td
y = e_sym2_eta_av
fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior_esym2, fcn=f_esym2_c, debug=True,svdcut=s_eta.svdcut)
e_sym2_eta_par = fit.p


############### Non-Quadraticities ##########################################

### e_symnq

e_symnq_res = e_sym_res - f_esym2_c(td,e_sym2_par)
e_symnq_data = te_NM_av - te_SM_av - e_sym2_av

e_symnq_eta_res = e_sym_res - f_esym2_c(td,e_sym2_eta_par)
e_symnq_eta_data = te_NM_av - te_SM_av - e_sym2_eta_av


fig, axes =  plt.subplots(1,3,figsize=(11,4), sharey='row')

for h in range(6):
    if h==5:
        axes[0].plot(td,e_NM[:,h] - e_sat[:,h] -e_sym2[:,h],color='C'+str(h) ,label='H'+str(h+2))
    else:
        axes[0].plot(td,e_NM[:,h] - e_sat[:,h] -e_sym2[:,h],color='C'+str(h) ,label='H'+str(h+1))

# axes[0].plot (td , gv.mean(e_symnq_data) , 'bs',label='Data (delta)')       # without error-bars
# axes[0].plot (td , gv.mean(e_symnq_eta_data) , 'rs',label='Data (eta)')
axes[0].errorbar (td , gv.mean(e_symnq_data),gv.sdev(e_symnq_data)  ,fmt='ob')
axes[0].errorbar (td , gv.mean(e_symnq_eta_data),gv.sdev(e_symnq_eta_data),fmt='or')
axes[0].fill_between (td,gv.mean(e_symnq_res)+gv.sdev(e_symnq_res),gv.mean(e_symnq_res)-gv.sdev(e_symnq_res),color='blue',alpha=0.1)
axes[0].fill_between (td,gv.mean(e_symnq_eta_res)+gv.sdev(e_symnq_eta_res),gv.mean(e_symnq_eta_res)-gv.sdev(e_symnq_eta_res),color='red',alpha=0.2)
axes[0].axhline(color='black')

axes[0].set_xlabel('$n$ (fm$^{-3}$)',fontsize='13')
axes[0].set_ylabel('$e_{\mathrm{sym,nq}}$ (MeV)',fontsize='13')
axes[0].tick_params(labelsize='13')
axes[0].tick_params(right=True)
axes[0].tick_params(top=True)
axes[0].tick_params(direction='in')
axes[0].legend(loc='upper left')

## e_symnq_pot

for h in range(6):
    axes[1].plot(td,e_NM[:,h] - T_NM(td) - e_sat[:,h] + T_SM(td) -e_sym2[:,h]+ 5/9*T_SM(td),color='C'+str(h) )

e_symnq_pot_res = e_sym_pot_res - f_esym2_c(td,e_sym2_par) + 5/9*T_SM(td)

e_symnq_pot_data = te_NM_av- T_NM(td) - te_SM_av+ T_SM(td) - e_sym2_av+ 5/9*T_SM(td)

e_symnq_pot_eta_res = e_sym_pot_res - f_esym2_c(td,e_sym2_eta_par)+ 5/9*T_SM(td)

e_symnq_pot_eta_data = te_NM_av- T_NM(td) - te_SM_av+ T_SM(td)  - e_sym2_eta_av+ 5/9*T_SM(td)


# axes[1].plot (td , gv.mean(e_symnq_pot_data) , 'bs',label='Data (delta)')   
# axes[1].plot (td , gv.mean(e_symnq_pot_eta_data) , 'rs',label='Data (eta)')
axes[1].errorbar (td , gv.mean(e_symnq_pot_data),gv.sdev(e_symnq_pot_data)  ,fmt='ob',label='Data (delta) (68% CL)')
axes[1].errorbar (td , gv.mean(e_symnq_pot_eta_data),gv.sdev(e_symnq_pot_eta_data),fmt='or',label='Data (eta) (68% CL)')
axes[1].fill_between (td,gv.mean(e_symnq_pot_res)+gv.sdev(e_symnq_pot_res),gv.mean(e_symnq_pot_res)-gv.sdev(e_symnq_pot_res),color='blue',alpha=0.1)
axes[1].fill_between (td,gv.mean(e_symnq_pot_eta_res)+gv.sdev(e_symnq_pot_eta_res),gv.mean(e_symnq_pot_eta_res)-gv.sdev(e_symnq_pot_eta_res),color='red',alpha=0.2)
axes[1].axhline(color='black')

axes[1].set_xlabel('$n$ (fm$^{-3}$)',fontsize='13')
axes[1].set_ylabel('$e_{\mathrm{sym,nq}}^{\mathrm{pot}}$ (MeV)',fontsize='13')
axes[1].tick_params(labelsize='13')
axes[1].tick_params(right=True)
axes[1].tick_params(top=True)
axes[1].tick_params(direction='in')
axes[1].legend(loc='upper left')

### e_symnq_pot_eff

for h in range(6):
    axes[2].plot(td,e_NM[:,h] - gv.mean(T_NM_eff(td)) - e_sat[:,h] + gv.mean(T_SM_eff(td)) -e_sym2[:,h]+ gv.mean(T_2_eff(td)),color='C'+str(h) )


e_symnq_pot_eff_res = e_sym_pot_eff_res - f_esym2_c(td,e_sym2_par) + T_2_eff(td)

e_symnq_pot_eff_data = te_NM_av- T_NM_eff(td) - te_SM_av+ T_SM_eff(td) - e_sym2_av+ T_2_eff(td)

e_symnq_pot_eff_eta_res = e_sym_pot_eff_res - f_esym2_c(td,e_sym2_eta_par)+ T_2_eff(td)

e_symnq_pot_eff_eta_data = te_NM_av- T_NM_eff(td) - te_SM_av+ T_SM_eff(td)  - e_sym2_eta_av + T_2_eff(td)

# axes[2].plot (td , gv.mean(e_symnq_pot_eff_data) , 'bs',label='Data (delta)')  
# axes[2].plot (td , gv.mean(e_symnq_pot_eff_eta_data) , 'rs',label='Data (eta)')
axes[2].errorbar (td , gv.mean(e_symnq_pot_eff_data),gv.sdev(e_symnq_pot_eff_data)  ,fmt='ob')
axes[2].errorbar (td , gv.mean(e_symnq_pot_eff_eta_data),gv.sdev(e_symnq_pot_eff_eta_data),fmt='or')
axes[2].fill_between (td,gv.mean(e_symnq_pot_eff_res)+gv.sdev(e_symnq_pot_eff_res),gv.mean(e_symnq_pot_eff_res)-gv.sdev(e_symnq_pot_eff_res),label='Fit (delta) (68% CL)',color='blue',alpha=0.1)
axes[2].fill_between (td,gv.mean(e_symnq_pot_eff_eta_res)+gv.sdev(e_symnq_pot_eff_eta_res),gv.mean(e_symnq_pot_eff_eta_res)-gv.sdev(e_symnq_pot_eff_eta_res),label='Fit (eta) (68% CL)',color='red',alpha=0.2)
axes[2].axhline(color='black')

axes[2].set_ylim(bottom=-1.5)

axes[2].set_xlabel('$n$ (fm$^{-3}$)',fontsize='13')
axes[2].set_ylabel('$e_{\mathrm{sym,nq}}^{\mathrm{pot*}}$ (MeV)',fontsize='13')
axes[2].tick_params(labelsize='13')
axes[2].tick_params(right=True)
axes[2].tick_params(top=True)
axes[2].tick_params(direction='in')
axes[2].legend(loc='upper left')

plt.tight_layout()
fig.show()


##################################### Residues ################################################# 

delta = np.arange(0,11,1)
delta = 0.1 * delta

def fit(den,delta):
    ans = f_SM(den,SM3_par)  + f_esym2_c(den,e_sym2_par)* delta**2  
    ans = ans + ( f_NM(den,NM3_par) - f_SM(den,SM3_par) - f_esym2_c(den,e_sym2_par) )* delta**4
    return ans
    
def fit_eta(den,delta):
    ans = f_SM(den,SM3_par)  + f_esym2_c(den,e_sym2_eta_par)* delta**2  
    ans = ans + ( f_NM(den,NM3_par) - f_SM(den,SM3_par) - f_esym2_c(den,e_sym2_eta_par) )* delta**4
    return ans
    
def fit_pot(den,delta):
    ans = f_SM(den,SM3_par) - T_SM(den)  + (f_esym2_c(den,e_sym2_par) - 5/9*T_SM(den))* delta**2  
    ans = ans + ( f_NM(den,NM3_par) - T_NM(den) - f_SM(den,SM3_par) + T_SM(den) - f_esym2_c(den,e_sym2_par) + 5/9*T_SM(den) )* delta**4
    return ans
    
def fit_eta_pot(den,delta):
    ans = f_SM(den,SM3_par) - T_SM(den) + ( f_esym2_c(den,e_sym2_eta_par)  - 5/9*T_SM(den))* delta**2  
    ans = ans + ( f_NM(den,NM3_par) - T_NM(den)- f_SM(den,SM3_par) + T_SM(den)- f_esym2_c(den,e_sym2_eta_par)+ 5/9*T_SM(den) )* delta**4
    return ans
    
def fit_pot_eff(den,delta):
    ans = f_SM(den,SM3_par) - T_SM_eff(den)  + (f_esym2_c(den,e_sym2_par) - T_2_eff(den))* delta**2  
    ans = ans + ( f_NM(den,NM3_par) - T_NM_eff(den) - f_SM(den,SM3_par) + T_SM_eff(den) - f_esym2_c(den,e_sym2_par) + T_2_eff(den) )* delta**4
    return ans

def fit_pot_eff_meanmass(den,delta):
    ans = f_SM(den,SM3_par) - gv.mean(T_SM_eff(den))  + (f_esym2_c(den,e_sym2_par) - gv.mean(T_2_eff(den)))* delta**2  
    ans = ans + ( f_NM(den,NM3_par) - gv.mean(T_NM_eff(den)) - f_SM(den,SM3_par) + gv.mean(T_SM_eff(den)) - f_esym2_c(den,e_sym2_par) + gv.mean(T_2_eff(den)) )* delta**4
    return ans
    
                    
def fit_eta_pot_eff(den,delta):
    ans = f_SM(den,SM3_par) - T_SM_eff(den) + ( f_esym2_c(den,e_sym2_eta_par)  - T_2_eff(den))* delta**2  
    ans = ans + ( f_NM(den,NM3_par) - T_NM_eff(den)- f_SM(den,SM3_par) + T_SM_eff(den)- f_esym2_c(den,e_sym2_eta_par)+ T_2_eff(den) )* delta**4
    return ans



fig, axes =  plt.subplots(3, 3, sharex='col', sharey = 'row',figsize=(15,8))

## First row (den=0.06)
data = []
data_pot=[]
data_pot_eff = []
data_pot_eff_meanmass = []

for h in range(6):
    data.append (te[4,:,h])
    data_pot.append ( te[4,:,h] )
    data_pot_eff.append ( te[4,:,h] )
    data_pot_eff_meanmass.append ( te[4,:,h] )

data = gv.dataset.avg_data(data,spread=True)
data_pot = gv.dataset.avg_data(data_pot,spread=True) - T(0.06,delta)
data_pot_eff = gv.dataset.avg_data(data_pot_eff,spread=True)- T_eff(0.06,delta)
data_pot_eff_meanmass = gv.dataset.avg_data(data_pot_eff_meanmass,spread=True)- gv.mean(T_eff(0.06,delta))

res = fit(0.06,delta) - data
res_eta = fit_eta(0.06,delta) - data

res_pot = fit_pot(0.06,delta) - data_pot
res_eta_pot = fit_eta_pot(0.06,delta) - data_pot

res_pot_eff = fit_pot_eff(0.06,delta) - data_pot_eff
res_eta_pot_eff = fit_eta_pot_eff(0.06,delta) - data_pot_eff

res_pot_eff_meanmass = fit_pot_eff_meanmass(0.06,delta) - data_pot_eff_meanmass

for h in range(6):
    axes[0,0].plot (delta, gv.mean ( fit(0.06,delta) - te[4,:,h] ) ,color='C'+str(h)+'')
    axes[0,1].plot (delta, gv.mean ( fit_pot(0.06,delta) - te[4,:,h] + T(0.06,delta) ) ,color='C'+str(h)+'' )
    axes[0,2].plot (delta, gv.mean ( fit_pot_eff(0.06,delta) - te[4,:,h] + T_eff(0.06,delta) ) ,color='C'+str(h)+'' )

axes[0,0].plot (delta, gv.mean(res) ,'bs' ,label='Mean R (Delta)')
axes[0,0].plot (delta, gv.mean(res_eta) ,'rs',label='Mean R (Eta)')
axes[0,0].fill_between (delta,gv.mean(res)+gv.sdev(res),gv.mean(res)-gv.sdev(res),color='blue',alpha=0.1,label='$\pm \sigma$(R) (Delta)')
axes[0,0].fill_between (delta,gv.mean(res_eta)+gv.sdev(res_eta),gv.mean(res_eta)-gv.sdev(res_eta),color='red',alpha=0.2,label='$\pm \sigma$(R) (Eta)')

axes[0,0].set_ylabel('R = Fit - Data (MeV)',fontsize='14')
axes[0,0].text(0.3, 0.05, '$n = 0.06$ fm$^{-3}$ ' ,fontsize='14' , transform = axes[0,0].transAxes)
axes[0,0].text(0.5, 0.5, '$y = e$',fontsize='14'  )
axes[0,0].tick_params(right=True)
axes[0,0].tick_params(top=True)
axes[0,0].tick_params(direction='in')
axes[0,0].axhline(color='black',alpha=1)
axes[0,0].tick_params(labelsize='14')

axes[0,1].plot (delta, gv.mean(res_pot) ,'bs' ,label='Mean R (Delta)')
axes[0,1].plot (delta, gv.mean(res_eta_pot) ,'rs',label='Mean R (Eta)')
axes[0,1].fill_between (delta,gv.mean(res_pot)+gv.sdev(res_pot),gv.mean(res_pot)-gv.sdev(res_pot),color='blue',alpha=0.1,label='$\pm \sigma$(R) (Delta)')
axes[0,1].fill_between (delta,gv.mean(res_eta_pot)+gv.sdev(res_eta_pot),gv.mean(res_eta_pot)-gv.sdev(res_eta_pot),color='red',alpha=0.2,label='$\pm \sigma$(R) (Eta)')

axes[0,1].text(0.3, 0.05, '$n = 0.06$ fm$^{-3}$ ',fontsize='14'  , transform = axes[0,1].transAxes)
axes[0,1].axhline(color='black',alpha=1)
axes[0,1].tick_params(right=True)
axes[0,1].tick_params(top=True)
axes[0,1].tick_params(direction='in')
axes[0,1].text(0.5, 0.5, '$y = e^{\mathrm{pot}}$',fontsize='14'  )


axes[0,2].plot (delta, gv.mean(res_pot_eff) ,'bs' ,label='Mean R (Delta)')
axes[0,2].plot (delta, gv.mean(res_eta_pot_eff) ,'rs',label='Mean R (Eta)')
axes[0,2].fill_between (delta,gv.mean(res_pot_eff)+gv.sdev(res_pot_eff),gv.mean(res_pot_eff)-gv.sdev(res_pot_eff),color='blue',alpha=0.1,label='$\pm \sigma$(R) (Delta)')
axes[0,2].fill_between (delta,gv.mean(res_eta_pot_eff)+gv.sdev(res_eta_pot_eff),gv.mean(res_eta_pot_eff)-gv.sdev(res_eta_pot_eff),color='red',alpha=0.2,label='$\pm \sigma$(R) (Eta)')
axes[0,2].plot (delta,gv.mean(res_pot_eff_meanmass)+gv.sdev(res_pot_eff_meanmass),'k--')
axes[0,2].plot (delta,gv.mean(res_pot_eff_meanmass)-gv.sdev(res_pot_eff_meanmass),'k--')

axes[0,2].text(0.3, 0.05, '$n = 0.06$ fm$^{-3}$ ' ,fontsize='14' , transform = axes[0,2].transAxes)
axes[0,2].axhline(color='black',alpha=1)
axes[0,2].tick_params(right=True)
axes[0,2].tick_params(top=True)
axes[0,2].tick_params(direction='in')
axes[0,2].text(0.5, 0.5, '$y = e^{\mathrm{pot*}}$' ,fontsize='14' )

## Second row (den=0.12)
data = []
data_pot=[]
data_pot_eff = []
data_pot_eff_meanmass = []

for h in range(6):
    data.append (te[10,:,h])
    data_pot.append ( te[10,:,h] )
    data_pot_eff.append ( te[10,:,h] )
    data_pot_eff_meanmass.append ( te[10,:,h] )

data = gv.dataset.avg_data(data,spread=True)
data_pot = gv.dataset.avg_data(data_pot,spread=True) - T(0.12,delta)
data_pot_eff = gv.dataset.avg_data(data_pot_eff,spread=True)- T_eff(0.12,delta)
data_pot_eff_meanmass = gv.dataset.avg_data(data_pot_eff_meanmass,spread=True)- gv.mean(T_eff(0.12,delta))

res = fit(0.12,delta) - data
res_eta = fit_eta(0.12,delta) - data

res_pot = fit_pot(0.12,delta) - data_pot
res_eta_pot = fit_eta_pot(0.12,delta) - data_pot

res_pot_eff = fit_pot_eff(0.12,delta) - data_pot_eff
res_eta_pot_eff = fit_eta_pot_eff(0.12,delta) - data_pot_eff

res_pot_eff_meanmass = fit_pot_eff_meanmass(0.12,delta) - data_pot_eff_meanmass

for h in range(6):
    axes[1,0].plot (delta, gv.mean ( fit(0.12,delta) - te[10,:,h] ) ,color='C'+str(h)+'' )
    axes[1,1].plot (delta, gv.mean ( fit_pot(0.12,delta) - te[10,:,h] + T(0.12,delta) ) ,color='C'+str(h)+'' )
    axes[1,2].plot (delta, gv.mean ( fit_pot_eff(0.12,delta) - te[10,:,h] + T_eff(0.12,delta) ) ,color='C'+str(h)+'' )
    
axes[1,0].plot (delta, gv.mean(res) ,'bs' ,label='Mean R (Delta)')
axes[1,0].plot (delta, gv.mean(res_eta) ,'rs',label='Mean R (Eta)')
axes[1,0].fill_between (delta,gv.mean(res)+gv.sdev(res),gv.mean(res)-gv.sdev(res),color='blue',alpha=0.1,label='$\pm \sigma$(R) (Delta)')
axes[1,0].fill_between (delta,gv.mean(res_eta)+gv.sdev(res_eta),gv.mean(res_eta)-gv.sdev(res_eta),color='red',alpha=0.2,label='$\pm \sigma$(R) (Eta)')

axes[1,0].set_ylabel('R = Fit - Data (MeV)',fontsize='14')
axes[1,0].text(0.3, 0.05, '$n = 0.12$ fm$^{-3}$ ',fontsize='14'  , transform = axes[1,0].transAxes)
axes[1,0].axhline(color='black',alpha=1)
axes[1,0].tick_params(right=True)
axes[1,0].tick_params(top=True)
axes[1,0].tick_params(direction='in')
axes[1,0].text(0.5, 1.75, '$y = e$' ,fontsize='14' )
axes[1,0].tick_params(labelsize='14')

axes[1,1].plot (delta, gv.mean(res_pot) ,'bs' ,label='Mean R (Delta)')
axes[1,1].plot (delta, gv.mean(res_eta_pot) ,'rs',label='Mean R (Eta)')
axes[1,1].fill_between (delta,gv.mean(res_pot)+gv.sdev(res_pot),gv.mean(res_pot)-gv.sdev(res_pot),color='blue',alpha=0.1)
axes[1,1].fill_between (delta,gv.mean(res_eta_pot)+gv.sdev(res_eta_pot),gv.mean(res_eta_pot)-gv.sdev(res_eta_pot),color='red',alpha=0.2)

axes[1,1].legend(loc='upper center',fontsize='12' )
axes[1,1].text(0.3, 0.05, '$n = 0.12$ fm$^{-3}$ ' ,fontsize='14' , transform = axes[1,1].transAxes)
axes[1,1].axhline(color='black',alpha=1)
axes[1,1].tick_params(right=True)
axes[1,1].tick_params(top=True)
axes[1,1].tick_params(direction='in')
axes[1,1].text(0.85, 1.75, '$y = e^{\mathrm{pot}}$',fontsize='14'  )


axes[1,2].plot (delta, gv.mean(res_pot_eff) ,'bs' )
axes[1,2].plot (delta, gv.mean(res_eta_pot_eff) ,'rs')
axes[1,2].fill_between (delta,gv.mean(res_pot_eff)+gv.sdev(res_pot_eff),gv.mean(res_pot_eff)-gv.sdev(res_pot_eff),color='blue',alpha=0.1,label='$\pm \sigma$(R) (Delta)')
axes[1,2].fill_between (delta,gv.mean(res_eta_pot_eff)+gv.sdev(res_eta_pot_eff),gv.mean(res_eta_pot_eff)-gv.sdev(res_eta_pot_eff),color='red',alpha=0.2,label='$\pm \sigma$(R) (Eta)')
axes[1,2].plot (delta,gv.mean(res_pot_eff_meanmass)+gv.sdev(res_pot_eff_meanmass),'k--')
axes[1,2].plot (delta,gv.mean(res_pot_eff_meanmass)-gv.sdev(res_pot_eff_meanmass),'k--')

axes[1,2].text(0.3, 0.05, '$n = 0.12$ fm$^{-3}$ ',fontsize='14' , transform = axes[1,2].transAxes)
axes[1,2].axhline(color='black',alpha=1)
axes[1,2].tick_params(right=True)
axes[1,2].tick_params(top=True)
axes[1,2].tick_params(direction='in')
axes[1,2].text(0.75, 1.75, '$y = e^{\mathrm{pot*}}$' ,fontsize='14' )
axes[1,2].legend(loc='upper center',fontsize='12' )

## Third row (den=0.16)
data = []
data_pot=[]
data_pot_eff = []
data_pot_eff_meanmass = []

for h in range(6):
    data.append (te[14,:,h])
    data_pot.append ( te[14,:,h] )
    data_pot_eff.append ( te[14,:,h] )
    data_pot_eff_meanmass.append ( te[14,:,h] )

data = gv.dataset.avg_data(data,spread=True)
data_pot = gv.dataset.avg_data(data_pot,spread=True) - T(0.16,delta)
data_pot_eff = gv.dataset.avg_data(data_pot_eff,spread=True)- T_eff(0.16,delta)
data_pot_eff_meanmass = gv.dataset.avg_data(data_pot_eff_meanmass,spread=True)- gv.mean( T_eff(0.16,delta))

res = fit(0.16,delta) - data
res_eta = fit_eta(0.16,delta) - data

res_pot = fit_pot(0.16,delta) - data_pot
res_eta_pot = fit_eta_pot(0.16,delta) - data_pot

res_pot_eff = fit_pot_eff(0.16,delta) - data_pot_eff
res_eta_pot_eff = fit_eta_pot_eff(0.16,delta) - data_pot_eff

res_pot_eff_meanmass = fit_pot_eff_meanmass(0.16,delta) - data_pot_eff_meanmass


for h in range(6):
    if h<=2:
        axes[2,0].plot (delta, gv.mean ( fit(0.16,delta) - te[14,:,h] ) ,color='C'+str(h)+'' ,label='H'+str(h+1)+'')
    else:
        axes[2,0].plot (delta, gv.mean ( fit(0.16,delta) - te[14,:,h] ) ,color='C'+str(h)+'' )
    if h >2:
        if h==5:
            axes[2,1].plot (delta, gv.mean ( fit_pot(0.16,delta) - te[14,:,h] + T(0.16,delta) ) ,color='C'+str(h)+''  ,label='H'+str(h+2)+'')
        else:
            axes[2,1].plot (delta, gv.mean ( fit_pot(0.16,delta) - te[14,:,h] + T(0.16,delta) ) ,color='C'+str(h)+''  ,label='H'+str(h+1)+'')
    else:
        axes[2,1].plot (delta, gv.mean ( fit_pot(0.16,delta) - te[14,:,h] + T(0.16,delta) ) ,color='C'+str(h)+'' )

    axes[2,2].plot (delta, gv.mean ( fit_pot_eff(0.16,delta) - te[14,:,h] + T_eff(0.16,delta) ) ,color='C'+str(h)+'' )
    

axes[2,0].plot (delta, gv.mean(res) ,'bs' )
axes[2,0].plot (delta, gv.mean(res_eta) ,'rs')
axes[2,0].fill_between (delta,gv.mean(res)+gv.sdev(res),gv.mean(res)-gv.sdev(res),color='blue',alpha=0.1)
axes[2,0].fill_between (delta,gv.mean(res_eta)+gv.sdev(res_eta),gv.mean(res_eta)-gv.sdev(res_eta),color='red',alpha=0.2)

axes[2,0].set_ylabel('R = Fit - Data (MeV)',fontsize='14')
axes[2,0].set_xlabel('$\delta$',fontsize='14')
axes[2,0].text(0.3, 0.05, '$n = 0.16$ fm$^{-3}$ ',fontsize='14' , transform = axes[2,0].transAxes)
axes[2,0].axhline(color='black',alpha=1)
axes[2,0].text(0.5, 3, '$y = e$',fontsize='14' )
axes[2,0].tick_params(right=True)
axes[2,0].tick_params(top=True)
axes[2,0].tick_params(direction='in')
axes[2,0].tick_params(labelsize='14')
axes[2,0].legend()

axes[2,1].plot (delta, gv.mean(res_pot) ,'bs' )
axes[2,1].plot (delta, gv.mean(res_eta_pot) ,'rs')
axes[2,1].fill_between (delta,gv.mean(res_pot)+gv.sdev(res_pot),gv.mean(res_pot)-gv.sdev(res_pot),color='blue',alpha=0.1)
axes[2,1].fill_between (delta,gv.mean(res_eta_pot)+gv.sdev(res_eta_pot),gv.mean(res_eta_pot)-gv.sdev(res_eta_pot),color='red',alpha=0.2)

axes[2,1].set_xlabel('$\delta$',fontsize='14')
axes[2,1].text(0.3, 0.05, '$n = 0.16$ fm$^{-3}$ ',fontsize='14' , transform = axes[2,1].transAxes)
axes[2,1].axhline(color='black',alpha=1)
axes[2,1].text(0.5, 3, '$y = e^{\mathrm{pot}}$',fontsize='14' )
axes[2,1].tick_params(right=True)
axes[2,1].tick_params(top=True)
axes[2,1].tick_params(direction='in')
axes[2,1].tick_params(labelsize='14')
axes[2,1].legend()

axes[2,2].plot (delta, gv.mean(res_pot_eff) ,'bs' )
axes[2,2].plot (delta, gv.mean(res_eta_pot_eff) ,'rs')
axes[2,2].fill_between (delta,gv.mean(res_pot_eff)+gv.sdev(res_pot_eff),gv.mean(res_pot_eff)-gv.sdev(res_pot_eff),color='blue',alpha=0.1)
axes[2,2].fill_between (delta,gv.mean(res_eta_pot_eff)+gv.sdev(res_eta_pot_eff),gv.mean(res_eta_pot_eff)-gv.sdev(res_eta_pot_eff),color='red',alpha=0.2)
axes[2,2].plot (delta,gv.mean(res_pot_eff_meanmass)+gv.sdev(res_pot_eff_meanmass),'k--')
axes[2,2].plot (delta,gv.mean(res_pot_eff_meanmass)-gv.sdev(res_pot_eff_meanmass),'k--')

axes[2,2].set_xlabel('$\delta$',fontsize='14')
axes[2,2].text(0.3, 0.05, '$n = 0.16$ fm$^{-3}$ ' ,fontsize='14', transform = axes[2,2].transAxes)
axes[2,2].axhline(color='black',alpha=1)
axes[2,2].tick_params(right=True)
axes[2,2].tick_params(top=True)
axes[2,2].tick_params(direction='in')
axes[2,2].text(0.5, 3, '$y = e^{\mathrm{pot*}}$' ,fontsize='14')
axes[2,2].tick_params(labelsize='14')

plt.tight_layout()
fig.show()

######################## Posterior values #############################################

print ('----------Delta----------')
print ("E_sym,nq = ", NM3_par['E_sat+E_sym'] - SM3_par['E_sat'] - e_sym2_par['E_sym2'] )
print ("L_sym,nq = ", NM3_par['L_sym']  - e_sym2_par['L_sym2'] )
print ("K_sym,nq = ", NM3_par['K_sat+K_sym'] - SM3_par['K_sat'] - e_sym2_par['K_sym2'] )
print ("Q_sym,nq = ", NM3_par['Q_sat+Q_sym'] - SM3_par['Q_sat'] - e_sym2_par['Q_sym2'] )
print ("Z_sym,nq = ", NM3_par['Z_sat+Z_sym'] - SM3_par['Z_sat'] - e_sym2_par['Z_sym2'] )

print ('----------Eta----------')
print ("E_sym,nq = ", NM3_par['E_sat+E_sym'] - SM3_par['E_sat'] - e_sym2_eta_par['E_sym2'] )
print ("L_sym,nq = ", NM3_par['L_sym']  - e_sym2_eta_par['L_sym2'] )
print ("K_sym,nq = ", NM3_par['K_sat+K_sym'] - SM3_par['K_sat'] - e_sym2_eta_par['K_sym2'] )
print ("Q_sym,nq = ", NM3_par['Q_sat+Q_sym'] - SM3_par['Q_sat'] - e_sym2_eta_par['Q_sym2'] )
print ("Z_sym,nq = ", NM3_par['Z_sat+Z_sym'] - SM3_par['Z_sat'] - e_sym2_eta_par['Z_sym2'] )


################# e_sym_4 VS e_sym,nq  (Eta calculations)   #####################################

e_sym4_eta_av =[]


for h in range(6):
    e_sym4_eta_av.append ( esym4_eta[:,h]  )
    

s4_eta = gv.dataset.svd_diagnosis(e_sym4_eta_av)
e_sym4_eta_av = gv.dataset.avg_data(e_sym4_eta_av,spread=True)

prior_esym4 = {}     # comes from posterior of (e_sym - e_sym2) fit subtraction
prior_esym4['n_sat']  = gv.gvar ('0.1606(74)') 
prior_esym4['E_sym4'] = gv.gvar ('1.3(1.5)')
prior_esym4['L_sym4'] = gv.gvar ('0.7(5.7)')
prior_esym4['K_sym4'] = gv.gvar ('-20(57)')
prior_esym4['Q_sym4'] = gv.gvar ('107(432)')
prior_esym4['Z_sym4'] = gv.gvar ('101(1058)')

def f_esym4(x,p):    
    xt = (x-p['n_sat'])/(3*p['n_sat'])
    
    ans = p['E_sym4'] + (p['K_sym4']/2)*xt**2 \
        + (p['Q_sym4']/6)*xt**3 + (p['Z_sym4']/24)*(xt)**4 \
        + p['L_sym4']*xt
    return ans
    
# def f_esym4_c(x,p):
#     xt = (x-p['n_sat'])/(3*p['n_sat']) 
#     b = 6.93
#     lam = f_esym4(0,p) * 3.**5 
#     return f_esym4(x,p)  + lam * xt**5  * np.exp(-b*x/0.16)


x = td
y = e_sym4_eta_av
fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior_esym4, fcn=f_esym4, debug=True,svdcut=0.1)

print ('------------e_sym4_eta---------------')
print (fit.p)


prior_esymnq = {}      # comes from posterior of (e_sym - e_sym2) fit subtraction
prior_esymnq['n_sat']  = gv.gvar ('0.1606(74)') 
prior_esymnq['E_symnq'] = gv.gvar ('1.3(1.5)')
prior_esymnq['L_symnq'] = gv.gvar ('0.7(5.7)')
prior_esymnq['K_symnq'] = gv.gvar ('-20(57)')
prior_esymnq['Q_symnq'] = gv.gvar ('107(432)')
prior_esymnq['Z_symnq'] = gv.gvar ('101(1058)')

def f_esymnq(x,p):    
    xt = (x-p['n_sat'])/(3*p['n_sat'])
    
    ans = p['E_symnq'] + (p['K_symnq']/2)*xt**2 \
        + (p['Q_symnq']/6)*xt**3 + (p['Z_symnq']/24)*(xt)**4 \
        + p['L_symnq']*xt
    return ans
    
prior_esymnq_red1 = {}     
prior_esymnq_red1['n_sat']  = gv.gvar ('0.1606(74)') 
prior_esymnq_red1['E_symnq'] = gv.gvar ('1.3(1.5)')
prior_esymnq_red1['L_symnq'] = gv.gvar ('0.7(5.7)')


def f_esymnq_red1(x,p):    
    xt = (x-p['n_sat'])/(3*p['n_sat'])
    
    ans = p['E_symnq']  + p['L_symnq']*xt
    return ans


prior_esymnq_red2 = {}      
prior_esymnq_red2['n_sat']  = gv.gvar ('0.1606(74)') 
prior_esymnq_red2['E_symnq'] = gv.gvar ('1.3(1.5)')
prior_esymnq_red2['K_symnq'] = gv.gvar ('-20(57)')

def f_esymnq_red2(x,p):    
    xt = (x-p['n_sat'])/(3*p['n_sat'])
    
    ans = p['E_symnq'] + (p['K_symnq']/2)*xt**2 
    return ans
    
e_symnq_eta_av = []   
for h in range(6):
    e_symnq_eta_av.append ( e_NM[:,h] -  e_sat[:,h]  -  esym2_eta[:,h]  )
 
snq_eta = gv.dataset.svd_diagnosis(e_symnq_eta_av)
e_symnq_eta_av = gv.dataset.avg_data(e_symnq_eta_av,spread=True)

x = td
y = e_symnq_eta_av
fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior_esymnq, fcn=f_esymnq, debug=True,svdcut=0.2)
full = fit.p

fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior_esymnq_red1, fcn=f_esymnq_red1, debug=True,svdcut=0.2)
red1 = fit.p

fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior_esymnq_red2, fcn=f_esymnq_red2, debug=True,svdcut=0.2)
red2 = fit.p

print ('------------e_sym,nq_eta   (New fit and not fit subtraction) ---------------')
print (full)
print (red1)
print (red2)



# fig = plt.figure()
# plt.plot (td, gv.mean (f_esymnq(td,full)), label = 'Full fit')
# plt.plot (td, gv.mean (f_esymnq_red1(td,red1)), label = 'only $E_{sym,nq}$ and $L_{sym,nq}$')
# plt.plot (td, gv.mean (f_esymnq_red2(td,red2)), label = 'only $E_{sym,nq}$ and $K_{sym,nq}$')
# plt.plot(td,gv.mean (y) ,'ob',label='data')
# plt.ylabel ('$e_{sym,nq}$')
# plt.xlabel ('n $(fm^{-3})$')
# plt.legend()
# plt.show()