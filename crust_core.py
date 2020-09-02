import matplotlib.pyplot as plt
import numpy as np
import gvar as gv

import symmetry_energy, quadratic_symmetry_energy


def main():
    
    # Import results from symmetry energy module
    # Refer to this module for explanation of variables
    global NM3_par, SM3_par
    te_SM_av,te_NM_av,f_SM,SM3_par,f_NM,NM3_par = symmetry_energy.e_sym_results()
    
    
    # Import results from quadratic_symmetry energy module
    # Refer to this module for explanation of variables
    global e_sym2_par
    d,e,td,te,e_sym2,esym2_eta,esym4_eta,e_sym2_av,e_sym2_eta_av,f_esym2_c,e_sym2_par,e_sym2_eta_par  = quadratic_symmetry_energy.quadratic_results()
    
    # Creates dictionary of NEPs for quadratic symmetry energy model
    global model_par_1
    model_par_1 = make_model_par_1()

    # Creates dictionary of NEPs for quadratic+quartic symmetry energy model
    global model_par_2
    model_par_2 = make_model_par_2()
    
    # Print the transition point for the two models 
    print ('-----Transition Point (Quadratic symmetry energy)-----')
    print ('density = ',transition_density(model_par_1) )
    print ('delta = ', delta_beta ( transition_density(model_par_1) ,model_par_1  ) )
    
    print ('-----Transition Point (Quadratic+Quartic symmetry energy)--------')
    print ('density = ',transition_density(model_par_2) )
    print ('delta = ', delta_beta ( transition_density(model_par_2), model_par_2 ) )
    
        
    # plots the spinodal and beta-equilibrium path and their intersection
    plot_crust_core_transition_plot()
    
    
############################## Sub-modules #################################
 
    
def T_SM (x):
    m = 938.919
    hbar = 197.3
    ans  = 3*hbar**2/(10*m) * (3*np.pi*np.pi*x/2)**(2/3)
    return ans
    

def make_model_par_1():   # Meta-model parameters with no quartic term 
    
    NM3_par.pop('n_sat')
    e_sym2_par.pop('Q_sat')
    e_sym2_par.pop('n_sat')
    e_sym2_par.pop('E_sat')
    e_sym2_par.pop('K_sat')
    e_sym2_par.pop('Z_sat')

    model_par_1={}              
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
    
    return model_par_1



def make_model_par_2():  # Meta-model parameters with quartic term 
    

    model_par_2={}           
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

    return model_par_2



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



def plot_crust_core_transition_plot():

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

################# Launch main program #####################################

if __name__ == '__main__':
    main()



