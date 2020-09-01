#
# Gnuplot Fit
#
set fit logfile "fit.log"
set fit errorvariables
#
# Fit the single particle energy for hamiltonian h=1,6
#

#############################################################

h=1

    set print "Effective_mass(SM)_".h.".dat" 
    print "#density    m*/m"

    do for [i=1:21:1] {
       den=i*0.01
       k_F=(1.5*3.14159*3.14159*den)**0.3333

       a=1
       b=1
       c=1
       d=1
       e=1
       esp(x) = a + b*x + c*x**2 + d*x**3 + e*x**4

       if (i < 10) {
       fit [x=0.0:k_F*0.25] esp(x) "h1/SPE_n_0.0".i."00000_x_0.5_N3LO_EM500_lam_1.80_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.26400_cE_-0.12000_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt" using 1:5 via a,b,c,d,e }


       else {
       fit [x=0.0:k_F*0.25] esp(x) "h1/SPE_n_0.".i."00000_x_0.5_N3LO_EM500_lam_1.80_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.26400_cE_-0.12000_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt" using 1:5 via a,b,c,d,e }

    m=1/(2*c)
     
    print den,"  ",m/4.758187377097

    }

    unset print

###############################################################

h=2

    set print "Effective_mass(SM)_".h.".dat" 
    print "#density    m*/m"

    do for [i=1:21:1] {
       den=i*0.01
       k_F=(1.5*3.14159*3.14159*den)**0.3333

       a=1
       b=1
       c=1
       d=1
       e=1
       esp(x) = a + b*x + c*x**2 + d*x**3 + e*x**4

       if (i < 10) {
       fit [x=0.0:k_F*0.25] esp(x) "h2/SPE_n_0.0".i."00000_x_0.5_N3LO_EM500_lam_2.00_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.27100_cE_-0.13100_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt" using 1:5 via a,b,c,d,e }


       else {
       fit [x=0.0:k_F*0.25] esp(x) "h2/SPE_n_0.".i."00000_x_0.5_N3LO_EM500_lam_2.00_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.27100_cE_-0.13100_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt" using 1:5 via a,b,c,d,e }

    m=1/(2*c)
     
    print den,"  ",m/4.758187377097

    }

    unset print

#################################################################

h=3

    set print "Effective_mass(SM)_".h.".dat" 
    print "#density    m*/m"

    do for [i=1:21:1] {
       den=i*0.01
       k_F=(1.5*3.14159*3.14159*den)**0.3333

       a=1
       b=1
       c=1
       d=1
       e=1
       esp(x) = a + b*x + c*x**2 + d*x**3 + e*x**4

       if (i < 10) {
       fit [x=0.0:k_F*0.25] esp(x) "h3/SPE_n_0.0".i."00000_x_0.5_N3LO_EM500_lam_2.00_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_-0.29200_cE_-0.59200_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.50000_L2_2.53387_nexp_4_nexpNN_4.txt" using 1:5 via a,b,c,d,e }


       else {
       fit [x=0.0:k_F*0.25] esp(x) "h3/SPE_n_0.".i."00000_x_0.5_N3LO_EM500_lam_2.00_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_-0.29200_cE_-0.59200_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.50000_L2_2.53387_nexp_4_nexpNN_4.txt" using 1:5 via a,b,c,d,e }

    m=1/(2*c)
     
    print den,"  ",m/4.758187377097

    }

    unset print

#################################################################


h=4

    set print "Effective_mass(SM)_".h.".dat" 
    print "#density    m*/m"

    do for [i=1:21:1] {
       den=i*0.01
       k_F=(1.5*3.14159*3.14159*den)**0.3333

       a=1
       b=1
       c=1
       d=1
       e=1
       esp(x) = a + b*x + c*x**2 + d*x**3 + e*x**4

       if (i < 10) {
       fit [x=0.0:k_F*0.25] esp(x) "h4/SPE_n_0.0".i."00000_x_0.5_N3LO_EM500_lam_2.20_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.21400_cE_-0.13700_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt" using 1:5 via a,b,c,d,e }


       else {
       fit [x=0.0:k_F*0.25] esp(x) "h4/SPE_n_0.".i."00000_x_0.5_N3LO_EM500_lam_2.20_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.21400_cE_-0.13700_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt" using 1:5 via a,b,c,d,e }

    m=1/(2*c)
     
    print den,"  ",m/4.758187377097

    }

    unset print

#################################################################

h=5

    set print "Effective_mass(SM)_".h.".dat" 
    print "#density    m*/m"

    do for [i=1:21:1] {
       den=i*0.01
       k_F=(1.5*3.14159*3.14159*den)**0.3333

       a=1
       b=1
       c=1
       d=1
       e=1
       esp(x) = a + b*x + c*x**2 + d*x**3 + e*x**4

       if (i < 10) {
       fit [x=0.0:k_F*0.25] esp(x) "h5/SPE_n_0.0".i."00000_x_0.5_N3LO_EM500_lam_2.80_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.27800_cE_-0.07800_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt" using 1:5 via a,b,c,d,e }


       else {
       fit [x=0.0:k_F*0.25] esp(x) "h5/SPE_n_0.".i."00000_x_0.5_N3LO_EM500_lam_2.80_Np_100_limit_10.0_c1_-0.15983_c3_-0.63145_c4_1.06557_cD_1.27800_cE_-0.07800_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt" using 1:5 via a,b,c,d,e }

    m=1/(2*c)
     
    print den,"  ",m/4.758187377097

    }

    unset print

#################################################################

h=6

    set print "Effective_mass(SM)_".h.".dat" 
    print "#density    m*/m"

    do for [i=1:21:1] {
       den=i*0.01
       k_F=(1.5*3.14159*3.14159*den)**0.3333

       a=1
       b=1
       c=1
       d=1
       e=1
       esp(x) = a + b*x + c*x**2 + d*x**3 + e*x**4

       if (i < 10) {
       fit [x=0.0:k_F*0.25] esp(x) "h6/SPE_n_0.0".i."00000_x_0.5_N3LO_EM500_lam_2.00_Np_100_limit_10.0_c1_-0.14997_c3_-0.94322_c4_0.78141_cD_-3.00700_cE_-0.68600_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt" using 1:5 via a,b,c,d,e }


       else {
       fit [x=0.0:k_F*0.25] esp(x) "h6/SPE_n_0.".i."00000_x_0.5_N3LO_EM500_lam_2.00_Np_100_limit_10.0_c1_-0.14997_c3_-0.94322_c4_0.78141_cD_-3.00700_cE_-0.68600_c2pi1pi_0_c2piCont_0_c2pi_0_crelCS_0_crelCT_0_crel2pi_0_crings_0_L3_2.00000_L2_2.53387_nexp_4_nexpNN_4.txt" using 1:5 via a,b,c,d,e }

    m=1/(2*c)
     
    print den,"  ",m/4.758187377097

    }

    unset print

#################################################################

exit
