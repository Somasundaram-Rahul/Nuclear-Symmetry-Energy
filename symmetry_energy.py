import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import gvar as gv
import lsqfit

import EM
import SM_NM


def main():
    
    # Import results from Effective Mass module 
    # EM = Effective Mass; par = Best fit parameter values
    # SM = Symmetric matter; NM = Neutron matter
    # 1 = liner fit; 2 = quadratic fit
    global EM_par_SM_1, EM_par_SM_2 , EM_par_NM_1 , EM_par_NM_2
    EM_par_SM_1, EM_par_SM_2 , EM_par_NM_1 , EM_par_NM_2 = EM.EM_results()
    
    
    # Import results from SM_NM module
    # e_SM and e_NM are energy/particle in symmetric and neutron matter
    # d_SM and d_NM are the correspondind densities
    # td = target density with unifrom grid
    # te_SM, te_NM = target energies corresponding to td.   
    global e_SM,e_NM, d_SM,d_NM,te_SM, te_NM,td
    e_SM,e_NM, d_SM,d_NM,te_SM, te_NM,td = SM_NM.SM_NM_results()



    # Prepare data for fitting and plotting
    # av refers to an average over the 6 Hamiltonians
    # s refers to svd cut imposed to regulate 0 eigenvalues of correlation matrix obtained during the averaging
    global te_SM_av,te_NM_av,ts_SM,ts_NM,te_SM_pot_av
    global te_NM_pot_av,te_SM_pot_eff_av,te_NM_pot_eff_av
    global te_SM_pot_eff_1_av,te_NM_pot_eff_1_av
    te_SM_av,te_NM_av,ts_SM,ts_NM,te_SM_pot_av,\
    te_NM_pot_av,te_SM_pot_eff_av,te_NM_pot_eff_av,\
    te_SM_pot_eff_1_av,te_NM_pot_eff_1_av = data_preparation()
    
    
    # Fit to energy per particle in SM. 
    # This performs Scaling 3*
    global f_SM,SM3_par
    f_SM,SM3_par = Analyse_SM()
    
    # Fit to energy per particle in NM. 
    # This performs Scaling 3*
    global f_NM,NM3_par
    f_NM,NM3_par = Analyse_NM()
    
    
    # Calculate and plot e_sym
    calculate_and_plot_esym()


####################### Modules called in the Main ##############################

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
    k = EM_par_SM_2
    m_e_inv  =  1+ x*k['k1']  + x**2 *k['k2']
    return T_SM(x)*m_e_inv


def T_NM_eff (x):
    k = EM_par_NM_2
    m_e_inv  =  1+ x*k['k1']  + x**2 *k['k2']
    return T_NM(x)*m_e_inv


def T_SM_eff_1 (x):
    k = EM_par_SM_1
    m_e_inv  =  1+ x*k['k1'] 
    return T_SM(x)*m_e_inv


def T_NM_eff_1 (x):
    k = EM_par_NM_1
    m_e_inv  =  1+ x*k['k1'] 
    return T_NM(x)*m_e_inv



def data_preparation():
    te_SM_av=[]
    te_NM_av=[]
    
    for h in range(6):
        te_SM_av.append ( te_SM[:,h]    )
        te_NM_av.append ( te_NM[:,h]    )
        
    
    ##### Data for plottting purposes
    
    
    te_SM_pot_av = gv.dataset.avg_data(te_SM_av,spread=True)- T_SM(td)
    
    te_NM_pot_av = gv.dataset.avg_data(te_NM_av,spread=True)- T_NM(td)
    
    
    te_SM_pot_eff_av = gv.dataset.avg_data(te_SM_av,spread=True) - T_SM_eff (td)
    
    te_NM_pot_eff_av = gv.dataset.avg_data(te_NM_av,spread=True) - T_NM_eff (td)
    
    te_SM_pot_eff_1_av = gv.dataset.avg_data(te_SM_av,spread=True) - T_SM_eff_1 (td)
    
    te_NM_pot_eff_1_av = gv.dataset.avg_data(te_NM_av,spread=True) - T_NM_eff_1 (td)
    
    ##### Data for Fitting purposes
    ts_SM = gv.dataset.svd_diagnosis(te_SM_av)
    te_SM_av = gv.dataset.avg_data(te_SM_av,spread=True)
    
    
    ts_NM = gv.dataset.svd_diagnosis(te_NM_av)
    te_NM_av = gv.dataset.avg_data(te_NM_av,spread=True)
    
    return te_SM_av,te_NM_av,ts_SM,ts_NM,te_SM_pot_av\
        ,te_NM_pot_av,te_SM_pot_eff_av,te_NM_pot_eff_av\
            ,te_SM_pot_eff_1_av,te_NM_pot_eff_1_av
            





def Analyse_SM():

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
        
    
    # prior_e_SM = {}          # Drischler prior
    # prior_e_SM['n_sat'] = gv.gvar(0.171, 0.016)
    # prior_e_SM['E_sat'] = gv.gvar(-15.16, 1.24)
    # prior_e_SM['K_sat'] = gv.gvar(214, 22)
    # prior_e_SM['Q_sat'] = gv.gvar(-139, 104)
    # prior_e_SM['Z_sat'] = gv.gvar(1306, 214)
    
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
    
    return f_SM,SM3_par






def Analyse_NM():

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
        
    
        
    # prior_eNM = {}    # Drischler prior
    # prior_eNM['n_sat'] = SM3_par['n_sat']
    # prior_eNM['E_sat+E_sym'] = gv.gvar(16.85, 3.33)
    # prior_eNM['L_sym'] = gv.gvar(48.1, 3.6)
    # prior_eNM['K_sat+K_sym'] = gv.gvar(42,62)
    # prior_eNM['Q_sat+Q_sym'] = gv.gvar(-303, 338)
    # prior_eNM['Z_sat+Z_sym'] = gv.gvar(-1011, 593)
    
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
    
    return f_NM,NM3_par




def calculate_and_plot_esym():

    den = np.arange(0.001, 0.2, 0.01)
    
    
    e_sym_s3 = f_NM(den,NM3_par) - f_SM(den,SM3_par)
    data_esym = te_NM_av - te_SM_av
    
    
    fig, axes =  plt.subplots(1,3,figsize=(11,4), sharey='row')
     
    axes[0].errorbar( td, gv.mean(data_esym), gv.sdev(data_esym),fmt='ob', label='data (68% CL)'  )
    axes[0].fill_between (den,gv.mean(e_sym_s3)+gv.sdev(e_sym_s3),gv.mean(e_sym_s3)-gv.sdev(e_sym_s3),label='fit (68% CL)',color='red',alpha=0.8)
    axes[0].fill_between (den,gv.mean(e_sym_s3)+2*gv.sdev(e_sym_s3),gv.mean(e_sym_s3)-2*gv.sdev(e_sym_s3),label='fit (95% CL)',color='red',alpha=0.2)
    
    axes[0].set_xlabel('$n$ (fm$^{-3}$)',fontsize='13')
    axes[0].set_ylabel('$e_{\mathrm{sym}}$ (MeV)',fontsize='13')
    axes[0].tick_params(right=True)
    axes[0].tick_params(labelsize='13')
    axes[0].set_xticks(np.arange(0, 0.2+0.01, 0.05))
    axes[0].tick_params(right=True)
    axes[0].tick_params(top=True)
    axes[0].tick_params(direction='in')
    
    ############ E_sym^{pot} #####################
    
    den = np.arange(0.001, 0.2, 0.01)
    
    
    e_sym_pot_s3 = f_NM(den,NM3_par) - f_SM(den,SM3_par) - T_NM(den) + T_SM(den)
    data_esym_pot = te_NM_pot_av - te_SM_pot_av
    
    
    axes[1].errorbar( td, gv.mean(data_esym_pot), gv.sdev(data_esym_pot),fmt='ob', label='data (68% CL)'  )
    axes[1].fill_between (den,gv.mean(e_sym_pot_s3)+gv.sdev(e_sym_pot_s3),gv.mean(e_sym_pot_s3)-gv.sdev(e_sym_pot_s3),label='fit (68% CL)',color='red',alpha=0.8)
    axes[1].fill_between (den,gv.mean(e_sym_pot_s3)+2*gv.sdev(e_sym_pot_s3),gv.mean(e_sym_pot_s3)-2*gv.sdev(e_sym_pot_s3),label='fit (95% CL)',color='red',alpha=0.2)
    
    axes[1].set_xlabel('$n$ (fm$^{-3}$)',fontsize='13')
    axes[1].set_ylabel('$e_{\mathrm{sym}}^{\mathrm{pot}}$ (MeV)',fontsize='13')
    axes[1].tick_params(right=True)
    axes[1].tick_params(labelsize='13')
    axes[1].set_xticks(np.arange(0, 0.2+0.01, 0.05))
    axes[1].legend(loc='upper left',fontsize='13')
    axes[1].tick_params(right=True)
    axes[1].tick_params(top=True)
    axes[1].tick_params(direction='in')
    
    ############ E_sym^{pot*} #####################
    
    
    den = np.arange(0.001, 0.2, 0.01)
    
    e_sym_pot_eff_s3 = f_NM(den,NM3_par) - f_SM(den,SM3_par) - T_NM_eff(den) + T_SM_eff(den)
    data_esym_pot_eff = te_NM_pot_eff_av - te_SM_pot_eff_av
    
    e_sym_pot_eff_1_s3 = f_NM(den,NM3_par) - f_SM(den,SM3_par) - T_NM_eff_1(den) + T_SM_eff_1(den)
    data_esym_pot_eff_1 = te_NM_pot_eff_1_av - te_SM_pot_eff_1_av
    
    
    
    axes[2].errorbar( td, gv.mean(data_esym_pot_eff_1), gv.sdev(data_esym_pot_eff_1),fmt='ob',alpha=0.7)
    axes[2].fill_between (den,gv.mean(e_sym_pot_eff_1_s3)+gv.sdev(e_sym_pot_eff_1_s3),gv.mean(e_sym_pot_eff_1_s3)-gv.sdev(e_sym_pot_eff_1_s3),color='red',alpha=0.8)
    axes[2].fill_between (den,gv.mean(e_sym_pot_eff_1_s3)+2*gv.sdev(e_sym_pot_eff_1_s3),gv.mean(e_sym_pot_eff_1_s3)-2*gv.sdev(e_sym_pot_eff_1_s3),color='red',alpha=0.2)
    
    axes[2].plot( td, gv.mean(data_esym_pot_eff),'xk',markersize=8, label='data (Quad. fit)' )
    axes[2].plot (den,gv.mean(e_sym_pot_eff_s3)+gv.sdev(e_sym_pot_eff_s3),'--k',label='Quadratic fit \n(68% CL)')
    axes[2].plot (den,gv.mean(e_sym_pot_eff_s3)-gv.sdev(e_sym_pot_eff_s3),'--k')
    # axes[2].fill_between (den,gv.mean(e_sym_pot_eff_s3)+gv.sdev(e_sym_pot_eff_s3),gv.mean(e_sym_pot_eff_s3)-gv.sdev(e_sym_pot_eff_s3),label='fit (68% CL)',color='red',alpha=0.8)
    # axes[2].fill_between (den,gv.mean(e_sym_pot_eff_s3)+2*gv.sdev(e_sym_pot_eff_s3),gv.mean(e_sym_pot_eff_s3)-2*gv.sdev(e_sym_pot_eff_s3),label='fit (95% CL)',color='red',alpha=0.2)
    
    
    axes[2].set_xlabel('$n$ (fm$^{-3}$)',fontsize='13')
    axes[2].set_ylabel('$e_{\mathrm{sym}}^{\mathrm{pot*}}$ (MeV)',fontsize='13')
    axes[2].tick_params(right=True)
    axes[2].legend(loc='upper left',fontsize='11')
    axes[2].tick_params(labelsize='13')
    axes[2].set_xticks(np.arange(0, 0.2+0.01, 0.05))
    axes[2].tick_params(right=True)
    axes[2].tick_params(top=True)
    axes[2].tick_params(direction='in')
    
    plt.tight_layout()
    plt.show()


################# Launch main program #####################################

if __name__ == '__main__':
    main()


###### Transfer results to subsequent programs ###############

def e_sym_results():
    global EM_par_SM_1, EM_par_SM_2 , EM_par_NM_1 , EM_par_NM_2
    EM_par_SM_1, EM_par_SM_2 , EM_par_NM_1 , EM_par_NM_2 = EM.EM_results()
    
    global e_SM,e_NM, d_SM,d_NM,te_SM, te_NM,td
    e_SM,e_NM, d_SM,d_NM,te_SM, te_NM,td = SM_NM.SM_NM_results()

    global te_SM_av,te_NM_av,ts_SM,ts_NM,te_SM_pot_av
    global te_NM_pot_av,te_SM_pot_eff_av,te_NM_pot_eff_av
    global te_SM_pot_eff_1_av,te_NM_pot_eff_1_av
    te_SM_av,te_NM_av,ts_SM,ts_NM,te_SM_pot_av,\
    te_NM_pot_av,te_SM_pot_eff_av,te_NM_pot_eff_av,\
    te_SM_pot_eff_1_av,te_NM_pot_eff_1_av = data_preparation()
        
    global f_SM,SM3_par
    f_SM,SM3_par = Analyse_SM()
    
    global f_NM,NM3_par
    f_NM,NM3_par = Analyse_NM()
    
    return te_SM_av,te_NM_av,f_SM,SM3_par,f_NM,NM3_par
    
