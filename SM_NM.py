import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import gvar as gv
import lsqfit
from sklearn.metrics import r2_score



def main():
    # Read input data of energy per particle
    # e_SM and e_NM are energy/particle in symmetric and neutron matter
    # d_SM and d_NM are the correspondind densities
    global e_SM,e_NM,d_SM,d_NM
    e_SM,e_NM,d_SM,d_NM = read_data()
    



    # Perform interpolation to go to uniform grid in density
    # td = target density with unifrom grid
    # te_SM, te_NM = target energies corresponding to td.
    global te_SM, te_NM,td
    te_SM, te_NM,td = interpolation()

    
    
    # Prepare data for fitting
    # mod refers to division of total energy by kinetic energy
    # av refers to an average over the 6 Hamiltonians
    # s refers to svd cut imposed to regulate 0 eigenvalues of correlation matrix obtained during the averaging
    global e_SM_mod_av,te_SM_av,te_SM_mod_av
    global e_NM_mod_av,te_NM_av,te_NM_mod_av
    global s_SM_mod,ts_SM,ts_SM_mod
    global s_NM_mod,ts_NM,ts_NM_mod
    e_SM_mod_av,te_SM_av,te_SM_mod_av,e_NM_mod_av,te_NM_av,te_NM_mod_av,\
    s_SM_mod,ts_SM,ts_SM_mod,s_NM_mod,ts_NM,ts_NM_mod  = data_preparation()
        
    
    
    # Perform the fit in Syummetric matter.
    # f_SM and f_SM_mod are the used fit functions for the energy and energy/kinetic energy
    # The other variables are the best fit parameters for the three scalings
    global f_SM,f_SM_mod,SM1_par,SM2_par,SM3_par
    f_SM,f_SM_mod,SM1_par,SM2_par,SM3_par = Analyse_SM()

    
    
    # Same as above for Neutron matter
    global f_NM,f_NM_mod ,NM1_par,NM2_par,NM3_par
    f_NM,f_NM_mod ,NM1_par,NM2_par,NM3_par = Analyse_NM()


    # Generate the various plots (uncomment as required):

    # plot_SNM_PNM()
    # plot_additional_data_scaling3()
    # plot_additional_data_scaling1()


    # Print fit results
    print ('---------Symmetric matter--------------')
    print ('Scaling 1:')
    print (SM1_par,'\n')
    print ('Scaling 2:')
    print (SM2_par,'\n')
    print ('Scaling 3:')
    print (SM3_par,'\n')
    print ('---------Neutron matter--------------')
    print ('Scaling 1:')
    print (NM1_par,'\n')
    print ('Scaling 2:')
    print (NM2_par,'\n')
    print ('Scaling 3:')
    print (NM3_par)
    

    
####################### Modules called in the Main ##############################


def read_data():


    d=np.zeros([35,11])
    e=np.zeros([35,11,6])
    
    
    ################ Read data ##############################
    
    for i in range(0,10):
        f = np.loadtxt("data/EOS_Drischler/EOS_spec_4_beta_0."+str(i)+".txt")
        d[:,i] = f[:,0]
        for k in range(0,6):
            if k == 5:
                e[:,i,k] = f[:,k+2]
            else:
                e[:,i,k] = f[:,k+1]
    
    
    f = np.loadtxt("data/EOS_Drischler/EOS_spec_4_beta_1.0.txt")     
    d[:,10] = f[:,0]
    for k in range(0,6):
        if k == 5:
            e[:,10,k] = f[:,k+2]
        else:
            e[:,10,k] = f[:,k+1]
            
    e_SM = e[:,0,:]
    e_NM = e[:,10,:]

    d_SM = d[:,0]
    d_NM = d[:,10]
            
    return e_SM,e_NM,d_SM,d_NM



def interpolation():
    td = np.arange(0.02, 0.2, 0.01)

    te_SM = np.zeros([td.size,6])
    te_NM = np.zeros([td.size,6])
    
    
    for k in range(0,6):
        tck = interpolate.splrep(d_SM, e_SM[:,k])
        te_SM[:,k] = interpolate.splev(td, tck, der=0)
    
    for k in range(0,6):
        tck = interpolate.splrep(d_NM, e_NM[:,k])
        te_NM[:,k] = interpolate.splev(td, tck, der=0)
        
    return te_SM, te_NM,td
        


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



def data_preparation():
    e_SM_mod = np.zeros([35,6])
    e_NM_mod = np.zeros([35,6])
    
    for k in range(0,6):
        e_SM_mod[:,k] = e_SM[:,k]/T_SM(d_SM)
        e_NM_mod[:,k] = e_NM[:,k]/T_NM(d_NM)
        
    
    e_SM_mod_av=[]
    e_NM_mod_av=[]
    
    
    for h in range(6):
        e_SM_mod_av.append (e_SM_mod[:,h] )
        e_NM_mod_av.append (e_NM_mod[:,h] )
    
    
    s_SM_mod = gv.dataset.svd_diagnosis(e_SM_mod_av)
    e_SM_mod_av = gv.dataset.avg_data(e_SM_mod_av,spread=True)
    
    
    s_NM_mod = gv.dataset.svd_diagnosis(e_NM_mod_av)
    e_NM_mod_av = gv.dataset.avg_data(e_NM_mod_av,spread=True)

    
    
    te_SM_mod = np.zeros([td.size,6])
    te_NM_mod = np.zeros([td.size,6])
    
    for k in range(0,6):
        te_SM_mod[:,k] = te_SM[:,k]/T_SM(td)
        te_NM_mod[:,k] = te_NM[:,k]/T_NM(td)
        
    
    te_SM_av=[]
    te_NM_av=[]
    
    te_SM_mod_av=[]
    te_NM_mod_av=[]
    
    for h in range(6):
        te_SM_av.append ( te_SM[:,h]    )
        te_NM_av.append ( te_NM[:,h]    )
        
        te_SM_mod_av.append ( te_SM_mod[:,h]    )
        te_NM_mod_av.append ( te_NM_mod[:,h]    )
    
    
    ts_SM = gv.dataset.svd_diagnosis(te_SM_av)
    te_SM_av = gv.dataset.avg_data(te_SM_av,spread=True)
    
    
    ts_NM = gv.dataset.svd_diagnosis(te_NM_av)
    te_NM_av = gv.dataset.avg_data(te_NM_av,spread=True)
    
    
    ts_SM_mod = gv.dataset.svd_diagnosis(te_SM_mod_av)
    te_SM_mod_av = gv.dataset.avg_data(te_SM_mod_av,spread=True)
    
    
    ts_NM_mod = gv.dataset.svd_diagnosis(te_NM_mod_av)
    te_NM_mod_av = gv.dataset.avg_data(te_NM_mod_av,spread=True)
    
    
    
    
    return e_SM_mod_av,te_SM_av,te_SM_mod_av,e_NM_mod_av,te_NM_av,te_NM_mod_av,\
           s_SM_mod,ts_SM,ts_SM_mod,s_NM_mod,ts_NM,ts_NM_mod


        
        
        
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
        return f_pot_SM(x,p)  + lam * xt**5  * np.exp( -p['b']* (x/0.16)**(beta/3) )
        
    def f_SM(x,p):
        return T_SM(x) + f_pot_SM_c(x,p)
        
    def f_SM_mod(x,p):
        return f_SM(x,p)/T_SM(x)
        
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
    prior_e_SM['b'] = gv.gvar(0,50)
    
    x = d_SM
    y = e_SM_mod_av
    
    fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior_e_SM, fcn=f_SM_mod, debug=True ,svdcut=s_SM_mod.svdcut)
    SM1_par = fit.p
    
    x = td
    y = te_SM_mod_av
    
    fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior_e_SM, fcn=f_SM_mod, debug=True,svdcut=ts_SM_mod.svdcut)
    SM2_par = fit.p
    
    x = td
    y = te_SM_av
    
    fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior_e_SM, fcn=f_SM, debug=True,svdcut=ts_SM.svdcut)
    SM3_par = fit.p
    
    
    # e,ev = np.linalg.eig (gv.evalcorr (e_SM_mod_av) )
    # d2 = np.std(  np.absolute(ev[0]) ) **2 
    # print ("N (Scale 1,SM) = ",np.size(d_SM))
    # print ("l_corr (Scale 1,SM) = ",  1 - np.size(d_SM)*d2  )
    
    # print ('\n')
    
    # e,ev = np.linalg.eig (gv.evalcorr (te_SM_mod_av) )
    # d2 = np.std(  np.absolute(ev[0]) ) **2 
    # print ("N (Scale 2,SM) = ",np.size(td))
    # print ("l_corr (Scale 2,SM) = ",  1 - np.size(td)*d2  )
    
    # print ('\n')
    
    # e,ev = np.linalg.eig (gv.evalcorr (te_SM_av) )
    # d2 = np.std(  np.absolute(ev[0]) ) **2 
    # print ("N (Scale 3,SM) = ",np.size(td))
    # print ("l_corr (Scale 3,SM) = ",  1 - np.size(td)*d2  )
    
    
    return f_SM,f_SM_mod ,SM1_par,SM2_par,SM3_par




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
        return f_pot_NM(x,p)  + lam * xt**5  * np.exp( -p['b']* (x/0.16)**(beta/3) )
        
    def f_NM(x,p):
        return T_NM(x) + f_pot_NM_c(x,p)
        
    def f_NM_mod(x,p):
        return f_NM(x,p)/T_NM(x)
        
    # prior_eNM = {}     # Drischler priors
    # prior_eNM['n_sat'] = SM1_par['n_sat']
    # prior_eNM['E_sat+E_sym'] = gv.gvar(16.85, 3.33)
    # prior_eNM['L_sym'] = gv.gvar(48.1, 3.6)
    # prior_eNM['K_sat+K_sym'] = gv.gvar(42,62)
    # prior_eNM['Q_sat+Q_sym'] = gv.gvar(-303, 338)
    # prior_eNM['Z_sat+Z_sym'] = gv.gvar(-1011, 593)
    # prior_eNM['b'] = gv.gvar(0,50)
    
    prior_eNM = {}          # Jerome priors
    prior_eNM['n_sat'] = SM1_par['n_sat']
    prior_eNM['E_sat+E_sym'] = gv.gvar(16.0, 3.0)
    prior_eNM['L_sym'] = gv.gvar(50, 10)
    prior_eNM['K_sat+K_sym'] = gv.gvar(100,100)
    prior_eNM['Q_sat+Q_sym'] = gv.gvar(0, 400)
    prior_eNM['Z_sat+Z_sym'] = gv.gvar(-500, 500)
    prior_eNM['b'] = gv.gvar(0,50)
    
    x = d_NM
    y = e_NM_mod_av
    
    fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior_eNM, fcn=f_NM_mod, debug=True,svdcut=s_NM_mod.svdcut)
    NM1_par = fit.p
    
    prior_eNM['n_sat'] =  SM2_par['n_sat']
    x = td
    y = te_NM_mod_av
    
    fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior_eNM, fcn=f_NM_mod, debug=True,svdcut=ts_NM_mod.svdcut)
    NM2_par = fit.p
    
    
    prior_eNM['n_sat'] =  SM3_par['n_sat']
    x = td
    y = te_NM_av
    
    fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior_eNM, fcn=f_NM, debug=True,svdcut=ts_NM.svdcut,add_svdnoise=False)
    NM3_par = fit.p
    
    ######### QQ plot 
    
    # residuals = fit.residuals
    # residuals = np.sort(residuals)
    # np.random.seed(36725786)
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
    
    ############
    

    # e,ev = np.linalg.eig (gv.evalcorr (e_NM_mod_av) )
    # d2 = np.std(  np.absolute(ev[0]) ) **2 
    # print ("N (Scale 1,NM) = ",np.size(d_NM))
    # print ("l_corr (Scale 1,NM) = ",1 - np.size(d_NM)*d2  )
    
    # print ('\n')
    
    # e,ev = np.linalg.eig (gv.evalcorr (te_NM_mod_av) )
    # d2 = np.std(  np.absolute(ev[0]) ) **2 
    # print ("N (Scale 2,NM) = ",np.size(td))
    # print ("l_corr (Scale 2,NM) = ",1 - np.size(td)*d2  )
    
    # print ('\n')
    
    # e,ev = np.linalg.eig (gv.evalcorr (te_NM_av) )
    # d2 = np.std(  np.absolute(ev[0]) ) **2 
    # print ("N (Scale 3,NM) = ",np.size(td))
    # print ("l_corr (Scale 3,NM) = ",1 - np.size(td)*d2  )
    
    
    return f_NM,f_NM_mod ,NM1_par,NM2_par,NM3_par




def plot_SNM_PNM():
    fig, axes =  plt.subplots(2, 3, sharex='col',figsize=(15,8))
    
    #S1,SM
    
    axes[0,0].errorbar (  (3*np.pi*np.pi*d_SM/2)**(1/3) ,gv.mean(e_SM_mod_av),gv.sdev(e_SM_mod_av) ,fmt='ob',label='data (68% CL)')
    plot_Kf = np.arange(0.001, 2, 0.01)
    plot_x = plot_Kf**3 * (2/3) * np.pi**(-2)
    axes[0,0].fill_between (  plot_Kf ,gv.mean(f_SM_mod(plot_x,SM1_par)) +gv.sdev(f_SM_mod(plot_x,SM1_par)),gv.mean(f_SM_mod(plot_x,SM1_par)) -gv.sdev(f_SM_mod(plot_x,SM1_par)) ,label='fit (68% CL)',color='red',alpha=0.8)
    axes[0,0].fill_between (  plot_Kf ,gv.mean(f_SM_mod(plot_x,SM1_par)) +2*gv.sdev(f_SM_mod(plot_x,SM1_par)),gv.mean(f_SM_mod(plot_x,SM1_par)) -2*gv.sdev(f_SM_mod(plot_x,SM1_par)) ,label='fit (95% CL)',color='red',alpha=0.2)
    
    axes[0,0].set_ylabel('$e_{\mathrm{SNM}}/e^{\mathrm{FFG}}_{\mathrm{SNM}}$',fontsize='15')
    axes[0,0].tick_params(right=True)
    axes[0,0].text(0.1, 0.9, ' Scaling 1, SNM',fontsize='15' , transform = axes[0,0].transAxes)
    axes[0,0].tick_params(labelsize=13)
    axes[0,0].tick_params(right=True)
    axes[0,0].tick_params(top=True)
    axes[0,0].tick_params(direction='in')
    
    #S2,SM
    
    axes[0,1].errorbar (  td ,gv.mean(te_SM_mod_av),gv.sdev(te_SM_mod_av) ,fmt='ob',label='data (68% CL)')
    axes[0,1].fill_between (  td ,gv.mean(f_SM_mod(td,SM2_par)) +gv.sdev(f_SM_mod(td,SM2_par)),gv.mean(f_SM_mod(td,SM2_par)) -gv.sdev(f_SM_mod(td,SM2_par)) ,label='fit (68% CL)',color='red',alpha=0.8)
    axes[0,1].fill_between (  td ,gv.mean(f_SM_mod(td,SM2_par)) +2*gv.sdev(f_SM_mod(td,SM2_par)),gv.mean(f_SM_mod(td,SM2_par)) -2*gv.sdev(f_SM_mod(td,SM2_par)) ,label='fit (95% CL)',color='red',alpha=0.2)
    
    axes[0,1].set_ylabel('$e_{\mathrm{SNM}}/e^{\mathrm{FFG}}_{\mathrm{SNM}}$',fontsize='15')
    axes[0,1].tick_params(right=True)
    axes[0,1].text(0.1, 0.9, ' Scaling 2, SNM',fontsize='15' , transform = axes[0,1].transAxes)
    axes[0,1].tick_params(labelsize=13)
    axes[0,1].tick_params(right=True)
    axes[0,1].tick_params(top=True)
    axes[0,1].tick_params(direction='in')
    
    #S3,SM
    
    axes[0,2].errorbar (  td ,gv.mean(te_SM_av),gv.sdev(te_SM_av) ,fmt='ob',label='data (68% CL)')
    axes[0,2].fill_between (  td ,gv.mean(f_SM(td,SM3_par)) +gv.sdev(f_SM(td,SM3_par)),gv.mean(f_SM(td,SM3_par)) -gv.sdev(f_SM(td,SM3_par)) ,label='fit (68% CL)',color='red',alpha=0.8)
    axes[0,2].fill_between (  td ,gv.mean(f_SM(td,SM3_par)) +2*gv.sdev(f_SM(td,SM3_par)),gv.mean(f_SM(td,SM3_par)) -2*gv.sdev(f_SM(td,SM3_par)) ,label='fit (95% CL)',color='red',alpha=0.2)
    
    axes[0,2].set_ylabel('$e_{\mathrm{SNM}}$ (MeV)',fontsize='15')
    axes[0,2].tick_params(right=True)
    axes[0,2].text(0.1, 0.9, ' Scaling 3, SNM' ,fontsize='15', transform = axes[0,2].transAxes)
    axes[0,2].tick_params(labelsize=13)
    axes[0,2].tick_params(right=True)
    axes[0,2].tick_params(top=True)
    axes[0,2].tick_params(direction='in')
    
    #S1, NM
    
    axes[1,0].errorbar (  (3*np.pi*np.pi*d_NM)**(1/3) ,gv.mean(e_NM_mod_av),gv.sdev(e_NM_mod_av) ,fmt='ob',label='data (68% CL)')
    plot_Kf = np.arange(0.001, 2, 0.01)
    plot_x = plot_Kf**3 * (1/3) * np.pi**(-2)
    axes[1,0].fill_between (  plot_Kf ,gv.mean(f_NM_mod(plot_x,NM1_par)) +gv.sdev(f_NM_mod(plot_x,NM1_par)),gv.mean(f_NM_mod(plot_x,NM1_par)) -gv.sdev(f_NM_mod(plot_x,NM1_par)) ,label='fit (68% CL)',color='red',alpha=0.8)
    axes[1,0].fill_between (  plot_Kf ,gv.mean(f_NM_mod(plot_x,NM1_par)) +2*gv.sdev(f_NM_mod(plot_x,NM1_par)),gv.mean(f_NM_mod(plot_x,NM1_par)) -2*gv.sdev(f_NM_mod(plot_x,NM1_par)) ,label='fit (95% CL)',color='red',alpha=0.2)
    
    axes[1,0].set_xlabel('$k_F$ (fm$^{-1}$)',fontsize='15')
    axes[1,0].set_ylabel('$e_{\mathrm{PNM}}/e^{\mathrm{FFG}}_{\mathrm{PNM}}$',fontsize='15')
    axes[1,0].tick_params(right=True)
    axes[1,0].text(0.1, 0.9, ' Scaling 1, PNM',fontsize='15' , transform = axes[1,0].transAxes)
    #axes[1,0].legend(loc='center right',fontsize='15')
    axes[1,0].legend(loc='center right', bbox_to_anchor=(1,0.6),fontsize='15')
    axes[1,0].tick_params(labelsize=13)
    axes[1,0].tick_params(right=True)
    axes[1,0].tick_params(top=True)
    axes[1,0].tick_params(direction='in')
    
    #S2, NM
    
    axes[1,1].errorbar (  td ,gv.mean(te_NM_mod_av),gv.sdev(te_NM_mod_av) ,fmt='ob',label='data (68% CL)')
    axes[1,1].fill_between (  td ,gv.mean(f_NM_mod(td,NM2_par)) +gv.sdev(f_NM_mod(td,NM2_par)),gv.mean(f_NM_mod(td,NM2_par)) -gv.sdev(f_NM_mod(td,NM2_par)) ,label='fit (68% CL)',color='red',alpha=0.8)
    axes[1,1].fill_between (  td ,gv.mean(f_NM_mod(td,NM2_par)) +2*gv.sdev(f_NM_mod(td,NM2_par)),gv.mean(f_NM_mod(td,NM2_par)) -2*gv.sdev(f_NM_mod(td,NM2_par)) ,label='fit (95% CL)',color='red',alpha=0.2)
    
    axes[1,1].set_xlabel('$n$ (fm$^{-3}$)',fontsize='15')
    axes[1,1].set_ylabel('$e_{\mathrm{PNM}}/e^{\mathrm{FFG}}_{\mathrm{PNM}}$',fontsize='15')
    axes[1,1].tick_params(right=True)
    axes[1,1].text(0.1, 0.9, ' Scaling 2, PNM' ,fontsize='15', transform = axes[1,1].transAxes)
    axes[1,1].tick_params(labelsize=13)
    axes[1,1].tick_params(right=True)
    axes[1,1].tick_params(top=True)
    axes[1,1].tick_params(direction='in')
    
    
    #S3, NM
    
    axes[1,2].errorbar (  td ,gv.mean(te_NM_av),gv.sdev(te_NM_av) ,fmt='ob',label='data (68% CL)')
    axes[1,2].fill_between (  td ,gv.mean(f_NM(td,NM3_par)) +gv.sdev(f_NM(td,NM3_par)),gv.mean(f_NM(td,NM3_par)) -gv.sdev(f_NM(td,NM3_par)) ,label='fit (68% CL)',color='red',alpha=0.8)
    axes[1,2].fill_between (  td ,gv.mean(f_NM(td,NM3_par)) +2*gv.sdev(f_NM(td,NM3_par)),gv.mean(f_NM(td,NM3_par)) -2*gv.sdev(f_NM(td,NM3_par)) ,label='fit (95% CL)',color='red',alpha=0.2)
    
    axes[1,2].set_xlabel('$n$ (fm$^{-3}$)',fontsize='15')
    axes[1,2].set_ylabel('$e_{\mathrm{PNM}}$ (MeV)',fontsize='15')
    axes[1,2].tick_params(right=True)
    axes[1,2].text(0.1, 0.9, ' Scaling 3, PNM' ,fontsize='15', transform = axes[1,2].transAxes)
    axes[1,2].tick_params(labelsize=13)
    axes[1,2].tick_params(right=True)
    axes[1,2].tick_params(top=True)
    axes[1,2].tick_params(direction='in')
    
    
    plt.tight_layout()
    fig.show()





def plot_additional_data_scaling3():
    f = np.loadtxt("data/EOS_Drischler/Bulgac-QMC.dat")
    A_den = f[:,0]
    A_E = f[:,2]
    
    f = np.loadtxt("data/EOS_Drischler/Tews-2016-QMC-band.dat")
    I_den = f[:,0]
    I_E = f[:,1]
    I_E_err = f[:,2]
    
    f = np.loadtxt("data/EOS_Drischler/APR.dat")
    apr_den = f[:,0]
    apr_E = f[:,2]
    
    
    fig,ax = plt.subplots(1,figsize=(7,5))
    
    ax.errorbar (  I_den ,I_E,I_E_err ,fmt='.k',alpha=0.7,label='Tews $et$ $al.$ (2016)')
    ax.plot(A_den, A_E,"vc",label='Wlazlowski $et$ $al.$ (2014)')
    ax.plot(apr_den, apr_E,"sg",label='APR (1998)')
    ax.errorbar (  td ,gv.mean(te_NM_av),gv.sdev(te_NM_av) ,fmt='ob',label='Drischler $et$ $al.$ (2016)')
    
    ax.fill_between (  td ,gv.mean(f_NM(td,NM3_par)) +gv.sdev(f_NM(td,NM3_par)),gv.mean(f_NM(td,NM3_par)) -gv.sdev(f_NM(td,NM3_par)) ,label='fit (68% CL)',color='red',alpha=0.8)
    ax.fill_between (  td ,gv.mean(f_NM(td,NM3_par)) +2*gv.sdev(f_NM(td,NM3_par)),gv.mean(f_NM(td,NM3_par)) -2*gv.sdev(f_NM(td,NM3_par)) ,label='fit (95% CL)',color='red',alpha=0.2)
    
    ax.set_xlabel('$n$ (fm$^{-3}$)',fontsize='15')
    ax.set_ylabel('$e_{\mathrm{PNM}}$ (MeV)',fontsize='15')
    plt.tick_params(right=True)
    plt.tick_params(top=True)
    plt.tick_params(direction='in')
    ax.set_xlim(right=0.21)
    ax.set_xlim(left=0.001)
    ax.set_ylim(top=21)
    ax.set_ylim(bottom=0)
    
    ax.tick_params(labelsize='14')
    
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[1], handles[0], handles[4], handles[5], handles[2], handles[3]]
    labels = [labels[1], labels[0], labels[4], labels[5], labels[2], labels[3]]
    ax.legend(handles,labels,loc='lower right',fontsize='13.5')
    
    # Inset
    
    left, bottom, width, height = [0.2, 0.63, 0.26, 0.32]
    ax2 = fig.add_axes([left, bottom, width, height])
    
    ax2.errorbar (  I_den ,I_E,I_E_err ,fmt='.k',alpha=0.7,label='Tews 2016(68% CL)')
    ax2.plot(A_den, A_E,"vc",label='Wlazlowski 2014')
    ax2.plot(apr_den, apr_E,"sg",label='APR')
    ax2.errorbar (  td ,gv.mean(te_NM_av),gv.sdev(te_NM_av) ,fmt='ob',label='Drischler 2016 (68% CL)')
    
    ax2.fill_between (  td ,gv.mean(f_NM(td,NM3_par)) +gv.sdev(f_NM(td,NM3_par)),gv.mean(f_NM(td,NM3_par)) -gv.sdev(f_NM(td,NM3_par)) ,label='fit (68% CL)',color='red',alpha=0.8)
    ax2.fill_between (  td ,gv.mean(f_NM(td,NM3_par)) +2*gv.sdev(f_NM(td,NM3_par)),gv.mean(f_NM(td,NM3_par)) -2*gv.sdev(f_NM(td,NM3_par)) ,label='fit (95% CL)',color='red',alpha=0.2)
    
    ax2.set_xlim(right=0.125)
    ax2.set_xlim(left=0.075)
    ax2.set_ylim(top=14)
    ax2.set_ylim(bottom=8)
    
    plt.tick_params(right=True)
    plt.tick_params(top=True)
    plt.tick_params(direction='in')
    ax2.tick_params(labelsize='12')
    
    plt.tight_layout()
    plt.show()




def plot_additional_data_scaling1():
    f = np.loadtxt("data/EOS_Drischler/Bulgac-QMC.dat")
    A_den = f[:,0]
    A_E = f[:,2]
    
    f = np.loadtxt("data/EOS_Drischler/Tews-2016-QMC-band.dat")
    I_den = f[:,0]
    I_E = f[:,1]
    I_E_err = f[:,2]
    
    f = np.loadtxt("data/EOS_Drischler/APR.dat")
    apr_den = f[:,0]
    apr_E = f[:,2]
    
    
    fig,ax = plt.subplots(1,figsize=(7,5))
    
    ax.errorbar (  (3*np.pi*np.pi*I_den)**(1/3) ,I_E/T_NM (I_den),I_E_err/T_NM (I_den) ,fmt='.k',alpha=0.7,label='Tews 2016(68% CL)')
    ax.plot((3*np.pi*np.pi*A_den)**(1/3), A_E/T_NM (A_den),"vc",label='Wlazlowski 2014')
    ax.plot((3*np.pi*np.pi*apr_den)**(1/3), apr_E/T_NM (apr_den),"sg",label='APR')
    ax.errorbar (  (3*np.pi*np.pi*d_NM)**(1/3) ,gv.mean(e_NM_mod_av),gv.sdev(e_NM_mod_av) ,fmt='ob',label='data (68% CL)')
    
    plot_Kf = np.arange(0.001, 2, 0.01)
    plot_x = plot_Kf**3 * (1/3) * np.pi**(-2)
    ax.fill_between (  plot_Kf ,gv.mean(f_NM_mod(plot_x,NM1_par)) +gv.sdev(f_NM_mod(plot_x,NM1_par)),gv.mean(f_NM_mod(plot_x,NM1_par)) -gv.sdev(f_NM_mod(plot_x,NM1_par)) ,label='fit (68% CL)',color='red',alpha=0.8)
    ax.fill_between (  plot_Kf ,gv.mean(f_NM_mod(plot_x,NM1_par)) +2*gv.sdev(f_NM_mod(plot_x,NM1_par)),gv.mean(f_NM_mod(plot_x,NM1_par)) -2*gv.sdev(f_NM_mod(plot_x,NM1_par)) ,label='fit (95% CL)',color='red',alpha=0.2)
    
    ax.set_xlabel('$k_F$ (fm$^{-1}$)',fontsize='15')
    ax.set_ylabel('$e_{\mathrm{PNM}}/e^{\mathrm{FFG}}_{\mathrm{PNM}}$',fontsize='15')
    plt.tick_params(right=True)
    plt.tick_params(top=True)
    plt.tick_params(direction='in')
    ax.set_xlim(right=2.1)
    ax.set_xlim(left=0.)
    ax.set_ylim(top=1)
    ax.set_ylim(bottom=0.35)
    
    ax.tick_params(labelsize='14')
    
    # Inset
    
    left, bottom, width, height = [0.4, 0.5, 0.4, 0.43]
    ax2 = fig.add_axes([left, bottom, width, height])
    
    ax2.errorbar (  (3*np.pi*np.pi*I_den)**(1/3) ,I_E/T_NM (I_den),I_E_err/T_NM (I_den) ,fmt='.k',alpha=0.7,label='Tews 2016(68% CL)')
    ax2.plot((3*np.pi*np.pi*A_den)**(1/3), A_E/T_NM (A_den),"vc",label='Wlazlowski 2014')
    ax2.plot((3*np.pi*np.pi*apr_den)**(1/3), apr_E/T_NM (apr_den),"sg",label='APR 1998')
    ax2.errorbar (  (3*np.pi*np.pi*d_NM)**(1/3) ,gv.mean(e_NM_mod_av),gv.sdev(e_NM_mod_av) ,fmt='ob',label='data (68% CL)')
    
    plot_Kf = np.arange(0.001, 2, 0.01)
    plot_x = plot_Kf**3 * (1/3) * np.pi**(-2)
    ax2.fill_between (  plot_Kf ,gv.mean(f_NM_mod(plot_x,NM1_par)) +gv.sdev(f_NM_mod(plot_x,NM1_par)),gv.mean(f_NM_mod(plot_x,NM1_par)) -gv.sdev(f_NM_mod(plot_x,NM1_par)) ,label='fit (68% CL)',color='red',alpha=0.8)
    ax2.fill_between (  plot_Kf ,gv.mean(f_NM_mod(plot_x,NM1_par)) +2*gv.sdev(f_NM_mod(plot_x,NM1_par)),gv.mean(f_NM_mod(plot_x,NM1_par)) -2*gv.sdev(f_NM_mod(plot_x,NM1_par)) ,label='fit (95% CL)',color='red',alpha=0.2)
    
    ax2.set_xlim(right=1.75)
    ax2.set_xlim(left=1.)
    ax2.set_ylim(top=0.6)
    ax2.set_ylim(bottom=0.35)
    
    plt.tick_params(right=True)
    plt.tick_params(top=True)
    plt.tick_params(direction='in')
    ax2.tick_params(labelsize='12')
    
    plt.tight_layout()
    plt.show()


################# Launch main program #####################################

if __name__ == '__main__':
    main()


###### Transfer results to subsequent programs ##############

def SM_NM_results():
    global e_SM,e_NM,d_SM,d_NM
    e_SM,e_NM,d_SM,d_NM = read_data()
    
    global te_SM, te_NM,td
    te_SM, te_NM,td = interpolation()
    
    return e_SM,e_NM, d_SM,d_NM,te_SM, te_NM,td
    
