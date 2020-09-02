import matplotlib.pyplot as plt
import numpy as np
import gvar as gv
import lsqfit

import EM, symmetry_energy, quadratic_symmetry_energy


def main():
    
    # Import results from Effective Mass module 
    # EM = Effective Mass; par = Best fit parameter values
    # SM = Symmetric matter; NM = Neutron matter
    global EM_par_SM_1 , EM_par_NM_1 
    EM_par_SM_1, _ , EM_par_NM_1 , _ = EM.EM_results()
    
    
    # Import results from symmetry energy module
    # Refer to this module for explanation of variables
    global te_SM_av,te_NM_av,f_SM,SM3_par,f_NM,NM3_par
    te_SM_av,te_NM_av,f_SM,SM3_par,f_NM,NM3_par = symmetry_energy.e_sym_results()

    # Import results from quadratic_symmetry energy module
    # Refer to this module for explanation of variables
    global d,e,td,te,e_sym2,esym2_eta,esym4_eta,e_sym2_av,e_sym2_eta_av,f_esym2_c,e_sym2_par,e_sym2_eta_par
    d,e,td,te,e_sym2,esym2_eta,esym4_eta,e_sym2_av,e_sym2_eta_av,f_esym2_c,e_sym2_par,e_sym2_eta_par = quadratic_symmetry_energy.quadratic_results()
    
    
    # Calculate and plot non-quadratic symmetry energies
    plot_e_symnq()
    
    # Calculate and plot Final residuals of the fit wrt the data
    plot_residues()
    
    
    
    # Print best fit value of parameters
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

    print ('----------Fit to E_sym4 without meta-model----------')
    e_sym4_eta_par = Fit_e_sym4_eta()
    print (e_sym4_eta_par)
    print ('----------Fit to E_symnq without meta-model----------')
    e_symnq_eta_par = Fit_e_symnq_eta()
    print (e_symnq_eta_par)


############################## Sub-modules #################################

#Linear fit !!
def m_e_inv_SM(x):
    k1 = gv.gvar ('3.33(18)')
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



# calculates and plots e_sym,nq 
def plot_e_symnq():

    #### E_sym 
    
    e_sym_res = f_NM(td,NM3_par) - f_SM(td,SM3_par)
    e_sym_pot_res = f_NM(td,NM3_par) - f_SM(td,SM3_par) - T_NM(td) + T_SM(td)
    e_sym_pot_eff_res = f_NM(td,NM3_par) - f_SM(td,SM3_par) - T_NM_eff(td) + T_SM_eff(td)
    
    
    
    ######################## Non-Quadraticities ############################
    
    ### e_symnq
    
    e_symnq_res = e_sym_res - f_esym2_c(td,e_sym2_par)
    e_symnq_data = te_NM_av - te_SM_av - e_sym2_av
    
    e_symnq_eta_res = e_sym_res - f_esym2_c(td,e_sym2_eta_par)
    e_symnq_eta_data = te_NM_av - te_SM_av - e_sym2_eta_av
    
    
    fig, axes =  plt.subplots(1,3,figsize=(11,4), sharey='row')
    
    for h in range(6):
        if h==5:
            axes[0].plot(td,te[:,10,h] - te[:,0,h] -e_sym2[:,h],color='C'+str(h) ,label='H'+str(h+2))
        else:
            axes[0].plot(td,te[:,10,h] - te[:,0,h] -e_sym2[:,h],color='C'+str(h) ,label='H'+str(h+1))
    
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
        axes[1].plot(td,te[:,10,h] - T_NM(td) - te[:,0,h] + T_SM(td) -e_sym2[:,h]+ 5/9*T_SM(td),color='C'+str(h) )
    
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
        axes[2].plot(td,te[:,10,h] - gv.mean(T_NM_eff(td)) - te[:,0,h] + gv.mean(T_SM_eff(td)) -e_sym2[:,h]+ gv.mean(T_2_eff(td)),color='C'+str(h) )
    
    
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






# calculate and plot the residuals
def plot_residues():

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





def Fit_e_sym4_eta():

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
    
    e_sym4_eta_par = fit.p 
    
    return e_sym4_eta_par



def Fit_e_symnq_eta():

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
        
    
        
    e_symnq_eta_av = []   
    for h in range(6):
        e_symnq_eta_av.append ( te[:,10,h] -  te[:,0,h]  -  esym2_eta[:,h]  )
     
    #snq_eta = gv.dataset.svd_diagnosis(e_symnq_eta_av)
    e_symnq_eta_av = gv.dataset.avg_data(e_symnq_eta_av,spread=True)
    
    x = td
    y = e_symnq_eta_av
    fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior_esymnq, fcn=f_esymnq, debug=True,svdcut=0.2)
    e_symnq_eta_par = fit.p
    
    return e_symnq_eta_par




################# Launch main program #####################################

if __name__ == '__main__':
    main()



    
