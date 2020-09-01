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

########################## Crust-Core transition #######################################

NM3_par.pop('n_sat')
e_sym2_par.pop('Q_sat')
e_sym2_par.pop('n_sat')
e_sym2_par.pop('E_sat')
e_sym2_par.pop('K_sat')
e_sym2_par.pop('Z_sat')

model_par_1={}              # Meta-model parameters with no quartic term 
model_par_1['n_sat'] = SM3_par['n_sat']

model_par_1['v0_SM'] = SM3_par['E_sat'] - T_SM(SM3_par['n_sat'])
model_par_1['v1_SM'] = -2*T_SM(SM3_par['n_sat'])
model_par_1['v2_SM'] = SM3_par['K_sat'] + 2*T_SM(SM3_par['n_sat'])
model_par_1['v3_SM'] = SM3_par['Q_sat'] -8*T_SM(SM3_par['n_sat'])
model_par_1['v4_SM'] = SM3_par['Z_sat'] + 56*T_SM(SM3_par['n_sat'])

model_par_1['v0_sym2'] = e_sym2_par['E_sym2'] - 5/9*T_SM(SM3_par['n_sat'])
model_par_1['v1_sym2'] = e_sym2_par['L_sym2'] - 10/9*T_SM(SM3_par['n_sat'])
model_par_1['v2_sym2'] = e_sym2_par['K_sym2'] + 10/9*T_SM(SM3_par['n_sat'])
model_par_1['v3_sym2'] = e_sym2_par['Q_sym2'] - 40/9*T_SM(SM3_par['n_sat'])
model_par_1['v4_sym2'] = e_sym2_par['Z_sym2'] + 280/9*T_SM(SM3_par['n_sat'])

model_par_1['v0_sym4'] = 0
model_par_1['v1_sym4'] = 0
model_par_1['v2_sym4'] = 0
model_par_1['v3_sym4'] = 0
model_par_1['v4_sym4'] = 0



model_par_2={}           # Meta-model parameters with quartic term 
model_par_2['n_sat'] = SM3_par['n_sat']

model_par_2['v0_SM'] = SM3_par['E_sat'] - T_SM(SM3_par['n_sat'])
model_par_2['v1_SM'] = -2*T_SM(SM3_par['n_sat'])
model_par_2['v2_SM'] = SM3_par['K_sat'] + 2*T_SM(SM3_par['n_sat'])
model_par_2['v3_SM'] = SM3_par['Q_sat'] -8*T_SM(SM3_par['n_sat'])
model_par_2['v4_SM'] = SM3_par['Z_sat'] + 56*T_SM(SM3_par['n_sat'])

model_par_2['v0_sym2'] = e_sym2_par['E_sym2'] - 5/9*T_SM(SM3_par['n_sat'])
model_par_2['v1_sym2'] = e_sym2_par['L_sym2'] - 10/9*T_SM(SM3_par['n_sat'])
model_par_2['v2_sym2'] = e_sym2_par['K_sym2'] + 10/9*T_SM(SM3_par['n_sat'])
model_par_2['v3_sym2'] = e_sym2_par['Q_sym2'] - 40/9*T_SM(SM3_par['n_sat'])
model_par_2['v4_sym2'] = e_sym2_par['Z_sym2'] + 280/9*T_SM(SM3_par['n_sat'])

model_par_2['v0_sym4'] = NM3_par['E_sat+E_sym']  - T_SM(SM3_par['n_sat']) *(2**(2/3)-1) - SM3_par['E_sat'] - model_par_2['v0_sym2']
model_par_2['v1_sym4'] = NM3_par['L_sym']  -2*T_SM(SM3_par['n_sat'])*(2**(2/3)-1) - model_par_2['v1_sym2']
model_par_2['v2_sym4'] = NM3_par['K_sat+K_sym'] -2*T_SM(SM3_par['n_sat'])*(-1*2**(2/3)+1)-SM3_par['K_sat'] - model_par_2['v2_sym2']
model_par_2['v3_sym4'] = NM3_par['Q_sat+Q_sym'] -2*T_SM(SM3_par['n_sat'])*(4*2**(2/3)-4)- SM3_par['Q_sat'] - model_par_2['v3_sym2']
model_par_2['v4_sym4'] = NM3_par['Z_sat+Z_sym'] -8*T_SM(SM3_par['n_sat'])*(-7*2**(2/3)+7)- SM3_par['Z_sat'] - model_par_2['v4_sym2']



def low(alpha,x,delta):         # low density correction term (u in the notes)
    N=4
    b = 17 + 22*delta**2 
    return 1 - (-3*x)**(N+1-alpha) * np.exp(-b*(1+3*x)) 


def e_delta(den,delta,p):       # first derivative of e wrt delta
    f = (1-delta)**(2/3) - (1+delta)**(2/3)
    T_delta = -5/6 * T_SM(den)*f

    x = (den-p['n_sat'])/(3*p['n_sat']) 
    
    v = [ p['v0_SM'] + p ['v0_sym2']*delta**2 + p ['v0_sym4']*delta**4,
          p['v1_SM'] + p ['v1_sym2']*delta**2 + p ['v1_sym4']*delta**4, 
          p['v2_SM'] + p ['v2_sym2']*delta**2 + p ['v2_sym4']*delta**4, 
          p['v3_SM'] + p ['v3_sym2']*delta**2 + p ['v3_sym4']*delta**4, 
          p['v4_SM'] + p ['v4_sym2']*delta**2 + p ['v4_sym4']*delta**4]    
    
    vp = [ 2*p['v0_sym2']*delta + 4*p['v0_sym4']*delta**3,
           2*p['v1_sym2']*delta + 4*p['v1_sym4']*delta**3, 
           2*p['v2_sym2']*delta + 4*p['v2_sym4']*delta**3, 
           2*p['v3_sym2']*delta + 4*p['v3_sym4']*delta**3, 
           2*p['v4_sym2']*delta + 4*p['v4_sym4']*delta**3]   
           
    ans = 0
    b_sym = 22
    for alpha in range(0,5):
        ans = ans +  (  vp[alpha]*low(alpha,x,delta) + v[alpha]* (1 - low(alpha,x,delta) ) *2*b_sym*delta* (1+3*x) )* x**(alpha)/np.math.factorial(alpha)
    
    v_delta = ans
    return T_delta + v_delta



def e_den(den,delta,p):    #First derivative of e wrt density 
    m = 938.919
    hbar = 197.3
    f = (1+delta)**(5/3) + (1-delta)**(5/3) 
    T_den =  3/5 * hbar**2/(2*m) * (1.5 * np.pi**2)**(2/3) *2/3* den**(-1/3) * f/2
    
    x = (den-p['n_sat'])/(3*p['n_sat']) 
    v = [ p['v0_SM'] + p ['v0_sym2']*delta**2 + p ['v0_sym4']*delta**4,
          p['v1_SM'] + p ['v1_sym2']*delta**2 + p ['v1_sym4']*delta**4, 
          p['v2_SM'] + p ['v2_sym2']*delta**2 + p ['v2_sym4']*delta**4, 
          p['v3_SM'] + p ['v3_sym2']*delta**2 + p ['v3_sym4']*delta**4, 
          p['v4_SM'] + p ['v4_sym2']*delta**2 + p ['v4_sym4']*delta**4]
          
    sumation = 0
    N=4
    b = 17 + 22*delta**2 
    for alpha in range(1,5):
        fac1 = alpha * low(alpha,x,delta)
        fac2 = (low(alpha,x,delta) - 1)*(N+1-alpha-3*b*x)
        sumation = sumation + v[alpha] * x**(alpha-1)/np.math.factorial(alpha) * ( fac1 + fac2 )
    
    v_den = 1/(3*p['n_sat']) * sumation
    
    return T_den + v_den
    

def e_den2(den,delta,p):    #Second derivative of e wrt density 
    m = 938.919
    hbar = 197.3
    f = (1+delta)**(5/3) + (1-delta)**(5/3) 
    T_den2 =  -6/45 * hbar**2/(2*m) * f/2 * (1.5 * np.pi**2)**(2/3) * den**(-4/3)
    
    x = (den-p['n_sat'])/(3*p['n_sat']) 
    v = [ p['v0_SM'] + p ['v0_sym2']*delta**2 + p ['v0_sym4']*delta**4,
          p['v1_SM'] + p ['v1_sym2']*delta**2 + p ['v1_sym4']*delta**4, 
          p['v2_SM'] + p ['v2_sym2']*delta**2 + p ['v2_sym4']*delta**4, 
          p['v3_SM'] + p ['v3_sym2']*delta**2 + p ['v3_sym4']*delta**4, 
          p['v4_SM'] + p ['v4_sym2']*delta**2 + p ['v4_sym4']*delta**4]
          
    sumation = 0
    N=4
    b = 17 + 22*delta**2 
    for alpha in range(2,5):
        fac1 = (alpha-1) * ( alpha* low(alpha,x,delta) + (low(alpha,x,delta)-1)*(N+1-alpha-3*b*x) )
        fac2 = alpha * (low(alpha,x,delta)-1)*(N+1-alpha-3*b*x)
        fac3 = (low(alpha,x,delta)-1)*(N+1-alpha-3*b*x)
        fac4 = (low(alpha,x,delta)-1) * ( (N+1-alpha-3*b*x)*(N-alpha-3*b*x) -3*b*x)
        sumation = sumation + v[alpha] * x**(alpha-2)/np.math.factorial(alpha) * ( fac1 + fac2 + fac3 + fac4)
    
    v_den2 = 1/((3*p['n_sat'])**2) * sumation
    
    return T_den2 + v_den2
     
           
def e_delta2(den,delta,p):           #Second derivative of e wrt delta
    fac = (1-delta)**(-1/3) + (1+delta)**(-1/3)
    T_delta2 =  10/18 * T_SM(den)*fac
    
    x = (den-p['n_sat'])/(3*p['n_sat']) 
    
    v = [ p['v0_SM'] + p ['v0_sym2']*delta**2 + p ['v0_sym4']*delta**4,
          p['v1_SM'] + p ['v1_sym2']*delta**2 + p ['v1_sym4']*delta**4, 
          p['v2_SM'] + p ['v2_sym2']*delta**2 + p ['v2_sym4']*delta**4, 
          p['v3_SM'] + p ['v3_sym2']*delta**2 + p ['v3_sym4']*delta**4, 
          p['v4_SM'] + p ['v4_sym2']*delta**2 + p ['v4_sym4']*delta**4]

    v_delta = [ 2*p['v0_sym2']*delta + 4*p['v0_sym4']*delta**3,
           2*p['v1_sym2']*delta + 4*p['v1_sym4']*delta**3, 
           2*p['v2_sym2']*delta + 4*p['v2_sym4']*delta**3, 
           2*p['v3_sym2']*delta + 4*p['v3_sym4']*delta**3, 
           2*p['v4_sym2']*delta + 4*p['v4_sym4']*delta**3]  
 
    v_delta2 = [ 2*p['v0_sym2'] + 4*3*p['v0_sym4']*delta**2,
           2*p['v1_sym2'] + 4*3*p['v1_sym4']*delta**2, 
           2*p['v2_sym2'] + 4*3*p['v2_sym4']*delta**2, 
           2*p['v3_sym2'] + 4*3*p['v3_sym4']*delta**2, 
           2*p['v4_sym2'] + 4*3*p['v4_sym4']*delta**2]   
    
    sumation = 0
    N=4
    b = 17 + 22*delta**2 
    b_sym =22
    for alpha in range(0,5):
        fac1 = v_delta2[alpha ]* low(alpha,x,delta)
        fac2 = 2* v_delta[alpha] * (1 + 3*x) * ( 1 - low(alpha,x,delta) )*2*b_sym*delta
        fac3 = v[alpha] * (1+3*x)* 2*b_sym*(1 - low(alpha,x,delta) ) * (1 - (1+3*x)*2*b_sym*delta**2 )
        sumation = sumation + x**(alpha)/np.math.factorial(alpha) * ( fac1 + fac2 + fac3 )  
        
    return T_delta2 + sumation
    
def e_delta_den(den,delta,p):        # Mixed derivative of e wrt to density and delta
    m = 938.919
    hbar = 197.3
    f = (1+delta)**(2/3) - (1-delta)**(2/3) 
    T_delta_den =  30/45 * hbar**2/(2*m) * (1.5 * np.pi**2)**(2/3) * den**(-1/3) * f/2    

    x = (den-p['n_sat'])/(3*p['n_sat']) 
    
    v = [ p['v0_SM'] + p ['v0_sym2']*delta**2 + p ['v0_sym4']*delta**4,
          p['v1_SM'] + p ['v1_sym2']*delta**2 + p ['v1_sym4']*delta**4, 
          p['v2_SM'] + p ['v2_sym2']*delta**2 + p ['v2_sym4']*delta**4, 
          p['v3_SM'] + p ['v3_sym2']*delta**2 + p ['v3_sym4']*delta**4, 
          p['v4_SM'] + p ['v4_sym2']*delta**2 + p ['v4_sym4']*delta**4]

    v_delta = [ 2*p['v0_sym2']*delta + 4*p['v0_sym4']*delta**3,
           2*p['v1_sym2']*delta + 4*p['v1_sym4']*delta**3, 
           2*p['v2_sym2']*delta + 4*p['v2_sym4']*delta**3, 
           2*p['v3_sym2']*delta + 4*p['v3_sym4']*delta**3, 
           2*p['v4_sym2']*delta + 4*p['v4_sym4']*delta**3]  
 
    sumation = 0
    N=4
    b = 17 + 22*delta**2 
    b_sym =22
    for alpha in range(1,5):
        fac1 = v_delta[alpha ]* low(alpha,x,delta)
        fac2 = v[alpha] * (1 - low(alpha,x,delta) ) * (1+3*x) * 2*b_sym*delta
        fac3 = v_delta[alpha ] * (N+1-alpha-3*b*x) * (low(alpha,x,delta) - 1)
        fac4 = v[alpha] * 2*b_sym*delta*(1 - low(alpha,x,delta) ) * (3*x + (1+3*x)*(N+1-alpha-3*b*x) )
        sumation = sumation + x**(alpha-1)/np.math.factorial(alpha) * ( alpha*(fac1+fac2) +fac3+fac4 )
    
    v_delta_den = sumation/(3*p['n_sat'])
    
    return T_delta_den + v_delta_den



def beta(den,delta,p):     # Beta equilibrium implies this function is zero
    m_e = 0.511
    hbar = 197.3
    
    fac1 = 2 * ( e_delta(den,delta,p) )
    fac2 = np.sqrt ( m_e**2 + hbar**2 * ( 3*np.pi**2*(1-delta)*den/2 )**(2/3) )
    return fac1-fac2
 
def delta_beta(den,p):      # Gives delta determined by beta equilibrium as function of density 
    def beta_f(delta):
        return beta(den,delta,p)
    interval = gv.root.search(beta_f, 0.1)
    root = gv.root.refine (beta_f, interval)
    return root
    
    
def det_C(den,delta,p):      #Determinant of curvature matrix
    term_11 = den*e_den2(den,delta,p) + 2*(1-delta)*e_delta_den(den,delta,p) \
             + (1-delta)**2/den * e_delta2(den,delta,p) + 2*e_den(den,delta,p)
               
    term_22 = den*e_den2(den,delta,p) - 2*(1+delta)*e_delta_den(den,delta,p) \
             + (1+delta)**2/den * e_delta2(den,delta,p) + 2*e_den(den,delta,p)
               
    term_12 =  den*e_den2(den,delta,p) - 2*delta*e_delta_den(den,delta,p) \
             - (1-delta**2)/den * e_delta2(den,delta,p) + 2*e_den(den,delta,p)
               
    return term_11*term_22 - term_12**2

    

def transition(delta,p):    #Spinodal density as function of delta 
    def det_C_f(den):  
        return det_C(den,delta,p)
    interval = gv.root.search(det_C_f, 0.03)
    root = gv.root.refine (det_C_f, interval)
    return root 

def transition_density(p):     
    def det_C_f(den):
        delta = delta_beta(den,p)
        return det_C(den,delta,p)
    interval = gv.root.search(det_C_f, 0.03)
    root = gv.root.refine (det_C_f, interval)
    return root


print ('-----Transition Point (Quadratic symmetry energy)-----')
print ('density = ',transition_density(model_par_1) )
print ('delta = ', delta_beta ( transition_density(model_par_1) ,model_par_1  ) )

print ('-----Transition Point (Global symmetry energy)--------')
print ('density = ',transition_density(model_par_2) )
print ('delta = ', delta_beta ( transition_density(model_par_2), model_par_2 ) )


density = np.arange(2,16)
density = density * 0.01
d1 = []
d2 = []

for i in density:
    d1.append(delta_beta(i,model_par_1))
    d2.append(delta_beta(i,model_par_2))
   
   
t1 = []
t2 = []
delta = np.arange(0,99)
delta = delta*0.01

for i in delta:
    t1.append (transition( i ,model_par_1) )   
    t2.append (transition( i ,model_par_2) )
    

fig, ax = plt.subplots(figsize=(7,5))

ax.fill_between ( density ,gv.mean(d1) +gv.sdev(d1),gv.mean(d1) -gv.sdev(d1) ,label='$\delta^2$ only',color='red',alpha=0.4)
ax.fill_between ( density ,gv.mean(d2) +gv.sdev(d2),gv.mean(d2) -gv.sdev(d2) ,label='$\delta^2 + \delta^4$',color='blue',alpha=0.4)

ax.fill_betweenx ( delta ,gv.mean(t1) +gv.sdev(t1),gv.mean(t1) -gv.sdev(t1) ,color='red',alpha=0.4)
ax.fill_betweenx ( delta ,gv.mean(t2) +gv.sdev(t2),gv.mean(t2) -gv.sdev(t2) ,color='blue',alpha=0.4)

plt.legend()
plt.xlabel('$n$ (fm$^{-3}$)',fontsize='15')
plt.ylabel (r'$ \delta$',fontsize='15')
plt.text(0.12, 0.85, r'$\beta$-equilibrium ',fontsize='14')
plt.text(0.109, 0.314, r'spinodal',fontsize='14')
ax.tick_params(labelsize='14')
ax.tick_params(right=True)
ax.tick_params(top=True)
ax.tick_params(direction='in')
ax.legend(loc = 'lower left',fontsize='13.0')

## Inset

density_inset = np.arange(5,13)
density_inset = density_inset * 0.01
d1_inset = []
d2_inset = []

for i in density_inset:
    d1_inset.append(delta_beta(i,model_par_1))
    d2_inset.append(delta_beta(i,model_par_2))
   
   
t1_inset = []
t2_inset = []
delta_inset = np.arange(90,99)
delta_inset = delta_inset*0.01

for i in delta_inset:
    t1_inset.append (transition( i ,model_par_1) )   
    t2_inset.append (transition( i ,model_par_2) )
    
    
left, bottom, width, height = [0.2, 0.45, 0.33, 0.41]

ax2 = fig.add_axes([left, bottom, width, height])

ax2.plot ( gv.mean(transition_density(model_par_1)) ,gv.mean(delta_beta ( transition_density(model_par_1) ,model_par_1  )) , 'rs' )
ax2.plot ( gv.mean(transition_density(model_par_2)) ,gv.mean(delta_beta ( transition_density(model_par_2) ,model_par_2  )) , 'bs' )


ax2.fill_between ( density_inset ,gv.mean(d1_inset) +gv.sdev(d1_inset),gv.mean(d1_inset) -gv.sdev(d1_inset) ,color='red',alpha=0.2)
ax2.fill_between ( density_inset ,gv.mean(d2_inset) +gv.sdev(d2_inset),gv.mean(d2_inset) -gv.sdev(d2_inset) ,color='blue',alpha=0.2)

ax2.fill_betweenx ( delta_inset ,gv.mean(t1_inset) +gv.sdev(t1_inset),gv.mean(t1_inset) -gv.sdev(t1_inset) ,color='red',alpha=0.4)
ax2.fill_betweenx ( delta_inset ,gv.mean(t2_inset) +gv.sdev(t2_inset),gv.mean(t2_inset) -gv.sdev(t2_inset) ,color='blue',alpha=0.4)

ax2.set_xlim(left=0.06)
ax2.set_xlim(right=0.12)
ax2.set_ylim(bottom=0.9)
ax2.set_ylim(top=0.98)

ax2.tick_params(labelsize='11')
ax2.tick_params(right=True)
ax2.tick_params(top=True)
ax2.tick_params(direction='in')

plt.tight_layout()
fig.show()

    