#Import packages and necessary functions used in these functions
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize

#Define measured values and number of colors 
mσ=599.99
mπ=140
mq=300
fπ=93
Nc = 3

def step(x):
    if x >= 0:
        return 1
    else:
        return 0

#General spectra needed
def E(p, m):
    return np.sqrt(p**2 + m**2)
    
def EΔ(p, m, Δ, μ_bar):  
    return np.sqrt((E(p, m) + μ_bar)**2 + Δ**2)

def C(p):
    return np.real(2-2*np.emath.sqrt(4*mq**2/(p**2)-1)*np.arctan(1/(np.emath.sqrt(4*mq**2/(p**2)-1))))
    
def dC(p):
    return np.real(4*mq**2/(p**4*np.emath.sqrt(4*mq**2/(p**2)-1))*np.arctan(1/(np.emath.sqrt(4*mq**2/(p**2)-1)))-1/(p**2))

Cσ = C(mσ)
Cπ = C(mπ)
dCπ = dC(mπ)

#m02 means m0^2
m02 = (mσ**2-3*mπ**2)/2 + 2*Nc*mq**2/((4*np.pi)**2*fπ**2)*(4*mq**2 + (mσ**2-4*mq**2)*Cσ - mσ**2*Cπ - (mσ**2-3*mπ**2)*mπ**2*dCπ)

λ0 = 3*(mσ**2-mπ**2)/fπ**2 + 12*Nc*mq**2/((4*np.pi)**2*fπ**4)*((mσ**2-4*mq**2)*(Cσ - Cπ - mπ**2*dCπ) + mπ**4*dCπ)
    
h0 = -mπ**2*fπ**2*(1-4*Nc*mq**2/((4*np.pi)**2*fπ**2)*mπ**2*dCπ)/fπ
    
g0 = mq/fπ

#Grand potential for NQM only need 
def grand_potential_NQM(ϕ, μ, μe):
    
    def Ωe(μe):
        return -1/(12*np.pi**2)*μe**4
    
    def Ωϕ(ϕ):
        if ϕ == 0:
            return 0 
        else: 
            return  h0*fπ*ϕ/mq - 1/2*m02*fπ**2*ϕ**2/(mq**2) + 1/24*λ0*fπ**4*ϕ**4/(mq**4) + 2*mq**4/((4*np.pi)**2)*(np.log(mq**2/(ϕ**2)) -(Cπ+mπ**2*dCπ))*ϕ**4/(mq**4) + 2*Nc*mq**4/((4*np.pi)**2)*(3/2)*ϕ**4/(mq**4) +  4*mq**4/((4*np.pi)**2)*(np.log((mq**2)/(ϕ**2)) -(Cπ+mπ**2*dCπ))*ϕ**4/(mq**4)

    def Ωμ(ϕ, μ, μe):
    
        μ_u = μ - 2*μe/3
        μ_d = μ + μe/3
        
        def integrand_u(p):
            if μ_u < 0:
                return p**2*(-μ_u - E(p, ϕ))*step(-μ_u - E(p, ϕ))
            else:
                return p**2*(μ_u - E(p, ϕ))*step(μ_u - E(p, ϕ))
    
        def integrand_d(p):
            if μ_d < 0:
                return p**2*(-μ_d - E(p, ϕ))*step(-μ_d - E(p, ϕ))
            else:
                return p**2*(μ_d - E(p, ϕ))*step(μ_d - E(p, ϕ))
    
        integral_u, err_u = integrate.quad(integrand_u, 0, np.real(np.emath.sqrt(μ_u**2 - ϕ**2)))
        integral_d, err_d = integrate.quad(integrand_d, 0, np.real(np.emath.sqrt(μ_d**2 - ϕ**2)))
    
        return -2*Nc*4*np.pi*(integral_u + integral_d)/(2*np.pi)**3
    #We define one potential without the original temperature part Ω1 and the total as Ω_tot.
    def Ω1(ϕ, Δ, μ, μe, μ8):
        return (Ωϕ(ϕ) + Ωe(μe))/fπ**4
    
    def Ω_tot(ϕμe, μ):
        ϕ, μe = ϕμe
        return (Ωϕ(ϕ) + Ωμ(ϕ, μ, μe) + Ωe(μe))/fπ**4

    return Ω_tot([ϕ, μe], μ)

def grand_potential_2SC(ϕ, Δ, μ, μe, μ8, gΔ, mΔ, λ3, λΔ, Λ_cutoff):
    
    def Ωe(μe):
        return -1/(12*np.pi**2)*μe**4
    
    def Ωϕ(ϕ):
        if ϕ == 0:
            return 0 
        else: 
            return  h0*fπ*ϕ/mq - 1/2*m02*fπ**2*ϕ**2/(mq**2) + 1/24*λ0*fπ**4*ϕ**4/(mq**4) + 2*mq**4/((4*np.pi)**2)*(np.log(mq**2/(ϕ**2)) - (Cπ+mπ**2*dCπ))*ϕ**4/(mq**4) + 2*Nc*mq**4/((4*np.pi)**2)*(3/2)*ϕ**4/(mq**4)

    def ΩϕΔ(ϕ, Δ, μ, μe, μ8):
        μ_bar = μ - μe/6 + μ8/3 
        if ϕ == 0 and Δ == 0:
            return 0
        if ϕ == 0:
            return  (mΔ**2 - 4*μ_bar**2)/(gΔ**2)*Δ**2 - 16/((4*np.pi)**2)*(np.log(mq**2/(Δ**2))-(Cπ+mπ**2*dCπ))*μ_bar**2*Δ**2 + λΔ/(6*gΔ**4)*Δ**4 + 6/((4*np.pi)**2)*Δ**4 + 4/((4*np.pi)**2)*(np.log(mq**2/(Δ**2))-(Cπ + mπ**2*dCπ))*(Δ**4)
        if Δ ==0:
            return   4*mq**4/((4*np.pi)**2)*(np.log((mq**2)/(ϕ**2)) -(Cπ+mπ**2*dCπ))*ϕ**4/(mq**4)
        else:
            return 1/12*λ3*fπ**2/(gΔ**2*mq**2)*Δ**2*ϕ**2 + 12/((4*np.pi)**2)*ϕ**2*Δ**2 + (mΔ**2-4*μ_bar**2)/(gΔ**2)*Δ**2 - 16/((4*np.pi)**2)*(np.log(mq**2/(Δ**2+ϕ**2))-(Cπ+mπ**2*dCπ))*μ_bar**2*Δ**2 + λΔ/(6*gΔ**4)*Δ**4 + 6/((4*np.pi)**2)*Δ**4 + 4/((4*np.pi)**2)*(np.log(mq**2/(Δ**2+ϕ**2))-(Cπ + mπ**2*dCπ))*(Δ**4) +  8/((4*np.pi)**2)*(np.log(mq**2/(Δ**2+ϕ**2))-(Cπ + mπ**2*dCπ))*Δ**2*ϕ**2 + 4*mq**4/((4*np.pi)**2)*(np.log((mq**2)/(ϕ**2+Δ**2)) -(Cπ+mπ**2*dCπ))*ϕ**4/(mq**4)
 
        
    def Ω_num(ϕ, Δ, μ, μe, μ8):
        μ_bar = μ - μe/6 + μ8/3
        def integrand(p):
            if Δ == 0:
                return 0
            else:
                return p**2*(EΔ(p, ϕ, Δ, μ_bar) + EΔ(p, ϕ, Δ, -μ_bar) - 2*np.sqrt(p**2+ϕ**2+Δ**2) - Δ**2*μ_bar**2/((p**2+ϕ**2+Δ**2)**(3/2))
                            )
    
        integral, err = integrate.quad(integrand, 0, Λ_cutoff, limit = 50)
    
        return 2*integral/(np.pi**2)

    
        
    def Ωμ(ϕ, Δ, μ, μe, μ8):

        μ_ub = μ - 2*μe/3 - 2*μ8/3 
        μ_db = μ + μe/3 - 2*μ8/3
        μ_ur = μ - 2*μe/3 + 1/3*μ8
        μ_dr = μ + 1/3*μe + 1/3*μ8
        μ_ug = μ_ur
        μ_dg = μ_dr
    
        μ_bar = μ - μe/6 + μ8/3
        δμ = μe/2
    
        if Δ != 0:
    
            def integrand_ub(p):
                if μ_ub < 0:
                    return p**2*(-μ_ub - E(p, ϕ))*step(-μ_ub - E(p, ϕ))
                else:
                    return p**2*(μ_ub - E(p, ϕ))*step(μ_ub - E(p, ϕ))
    
            def integrand_db(p):
                if μ_db <0:
                    return p**2*(-μ_db - E(p, ϕ))*step(-μ_db - E(p, ϕ))
                else:
                    return p**2*(μ_db - E(p, ϕ))*step(μ_db - E(p, ϕ))
            def integral_limit_eqn_plus(p):
                return δμ - EΔ(p, ϕ, Δ, μ_bar)
            
            if np.sign(integral_limit_eqn_plus(0)) == np.sign(integral_limit_eqn_plus(Λ_cutoff)):
                integral_limit_plus = 0
            else:
                integral_limit_plus = optimize.brentq(integral_limit_eqn_plus, 0, Λ_cutoff)
        
            def integral_limit_eqn_min(p):
                return δμ - EΔ(p, ϕ, Δ, -μ_bar)
    
            if np.sign(integral_limit_eqn_min(0)) == np.sign(integral_limit_eqn_min(Λ_cutoff)):
                integral_limit_min = 0
            else:
                integral_limit_min = optimize.brentq(integral_limit_eqn_min, 0, Λ_cutoff)
            
            def integrand_bar_p(p):
                return p**2*(δμ - EΔ(p, ϕ, Δ, μ_bar))*step(δμ - EΔ(p, ϕ, Δ, μ_bar))
    
            def integrand_bar_m(p):
                return p**2*(δμ - EΔ(p, ϕ, Δ, -μ_bar))*step(δμ - EΔ(p, ϕ, Δ, -μ_bar))
    
            integral_ub, err_ub = integrate.quad(integrand_ub, 0, np.real(np.emath.sqrt(μ_ub**2 - ϕ**2)))
            integral_db, err_db = integrate.quad(integrand_db, 0, np.real(np.emath.sqrt(μ_db**2 - ϕ**2)))
            integral_bar_p, err_bar_p = integrate.quad(integrand_bar_p, 0, integral_limit_plus)
            integral_bar_m, err_bar_m = integrate.quad(integrand_bar_m, 0, integral_limit_min)
            #return -integral_bar_m
            return -2*4*np.pi*(integral_ub + integral_db + 2*integral_bar_p + 2*integral_bar_m)/(2*np.pi)**3

        else:
            #Blue quarks
            def integrand_ub(p):
                if μ_ub < 0:
                    return p**2*(-μ_ub - E(p, ϕ))*step(-μ_ub - E(p, ϕ))
                else:
                    return p**2*(μ_ub - E(p, ϕ))*step(μ_ub - E(p, ϕ))
    
            def integrand_db(p):
                if μ_db < 0:
                    return p**2*(-μ_db - E(p, ϕ))*step(-μ_db - E(p, ϕ))
                else:
                    return p**2*(μ_db - E(p, ϕ))*step(μ_db - E(p, ϕ))
    
            #Red quarks
            def integrand_ur(p):
                if μ_ur <0:
                    return p**2*(-μ_ur - E(p, ϕ))*step(-μ_ur - E(p, ϕ))
                else:
                    return p**2*(μ_ur - E(p, ϕ))*step(μ_ur - E(p, ϕ))
    
            def integrand_dr(p):
                if μ_dr <0:
                    return p**2*(-μ_dr - E(p, ϕ))*step(-μ_dr - E(p, ϕ))
                else:
                    return p**2*(μ_dr - E(p, ϕ))*step(μ_dr - E(p, ϕ))
    
            #Green Quarks
            def integrand_ug(p):
                if μ_ug <0:
                    return p**2*(-μ_ug - E(p, ϕ))*step(-μ_ug - E(p, ϕ))
                else:
                    return p**2*(μ_ug - E(p, ϕ))*step(μ_ug - E(p, ϕ))
    
            def integrand_dg(p):
                if μ_dg < 0:
                    return p**2*(-μ_dg - E(p, ϕ))*step(-μ_dg - E(p, ϕ))
                else:
                    return p**2*(μ_dg - E(p, ϕ))*step(μ_dg - E(p, ϕ))
    
            integral_ub, err_ub = integrate.quad(integrand_ub, 0, np.real(np.emath.sqrt(μ_ub**2 - ϕ**2)))
            integral_db, err_db = integrate.quad(integrand_db, 0, np.real(np.emath.sqrt(μ_db**2 - ϕ**2)))
    
            integral_ur, err_ur = integrate.quad(integrand_ur, 0, np.real(np.emath.sqrt(μ_ur**2 - ϕ**2)))
            integral_dr, err_dr = integrate.quad(integrand_dr, 0, np.real(np.emath.sqrt(μ_dr**2 - ϕ**2)))
    
            integral_ug, err_ug = integrate.quad(integrand_ug, 0, np.real(np.emath.sqrt(μ_ug**2 - ϕ**2)))
            integral_dg, err_dg = integrate.quad(integrand_dg, 0, np.real(np.emath.sqrt(μ_dg**2 - ϕ**2)))
    
            return -2*4*np.pi*(integral_ub + integral_db + integral_ur + integral_dr + integral_ug + integral_dg)/(2*np.pi)**3
    #We define one potential without the original temperature part Ω1 and the total as Ω_tot.
    def Ω1(ϕ, Δ, μ, μe, μ8):
        return (Ωϕ(ϕ) + ΩϕΔ(ϕ, Δ, μ, μe, μ8)  - Ω_num(ϕ, Δ, μ, μe, μ8) + Ωe(μe))/fπ**4
    
    def Ω_tot(ϕΔμeμ8, μ):
        ϕ, Δ, μe, μ8 = ϕΔμeμ8
        return (Ωϕ(ϕ) + ΩϕΔ(ϕ, Δ, μ, μe, μ8)  - Ω_num(ϕ, Δ, μ, μe, μ8) + Ωμ(ϕ, Δ, μ, μe, μ8) + Ωe(μe))/fπ**4

    return Ω_tot([ϕ, Δ, μe, μ8], μ)

#Functions needed to calculate ne, n8 and nq
def dΩdμ_ub(ϕ, μ, μe, μ8, Λ_cutoff):
    μ_ub = μ - 2/3*μe - 2/3*μ8
    return -16/(3*(4*np.pi)**2)*np.real(np.emath.sqrt(μ_ub**2 - ϕ**2)**3)
    #return -np.sign(μ_ub)*16/(3*(4*np.pi)**2)*np.real(np.emath.sqrt(μ_ub**2 - ϕ**2)**3)

def dΩdμ_db(ϕ, μ, μe, μ8, Λ_cutoff):
    μ_db = μ + 1/3*μe - 2/3*μ8
    return -16/(3*(4*np.pi)**2)*np.real(np.emath.sqrt(μ_db**2 - ϕ**2)**3)
    #return -np.sign(μ_db)*16/(3*(4*np.pi)**2)*np.real(np.emath.sqrt(μ_db**2 - ϕ**2)**3)

def dΩdμ_bar(ϕ, Δ, μ, μe, μ8, gΔ, Λ_cutoff):
    μ_bar = μ - 1/6*μe + 1/3*μ8
    δμ = μe/2
    def integral_limit_eqn_plus(p):
        return δμ - EΔ(p, ϕ, Δ, μ_bar)
    
    if np.sign(integral_limit_eqn_plus(0)) == np.sign(integral_limit_eqn_plus(Λ_cutoff)):
        integral_limit_plus = 0
    else:
        integral_limit_plus = optimize.brentq(integral_limit_eqn_plus, 0, Λ_cutoff)

    def integral_limit_eqn_min(p):
        return δμ - EΔ(p, ϕ, Δ, -μ_bar)
    
    if np.sign(integral_limit_eqn_min(0)) == np.sign(integral_limit_eqn_min(Λ_cutoff)):
        integral_limit_min = 0
    else:
        integral_limit_min = optimize.brentq(integral_limit_eqn_min, 0, Λ_cutoff)

    def integrand_finite(p):
        return p**2*((E(p, ϕ) + μ_bar)/EΔ(p, ϕ, Δ, μ_bar) - (E(p, ϕ) - μ_bar)/EΔ(p, ϕ, Δ, -μ_bar) - 2*Δ**2*μ_bar/((p**2 + ϕ**2 + Δ**2)**(3/2))
                    )
    
    def integrand_1(p):
        return p**2*(E(p, ϕ) + μ_bar)/EΔ(p, ϕ, Δ, μ_bar)
        
    def integrand_2(p):
        return p**2*(E(p, ϕ) - μ_bar)/EΔ(p, ϕ, Δ, -μ_bar)
        
    integral_finite, err_finite = integrate.quad(integrand_finite, 0, Λ_cutoff)
    integral_1, err_1 = integrate.quad(integrand_1, 0, integral_limit_plus)
    integral_2, err_2 = integrate.quad(integrand_2, 0, integral_limit_min)
        
    term1 = - 8*μ_bar/(gΔ**2)*(1 + 4*gΔ**2/((4*np.pi)**2)*(np.log(mq**2/(ϕ**2 + Δ**2)) - Cπ - mπ**2*dCπ))*Δ**2
    term2 = - 4*integral_finite*(4*np.pi)/((2*np.pi)**3)
    term3 = + 4*(integral_1 - integral_2)*(4*np.pi)/((2*np.pi)**3)
    return term1 + term2 + term3

def dΩdδμ(ϕ, Δ, μ, μe, μ8, Λ_cutoff):
    μ_bar = μ - 1/6*μe + 1/3*μ8
    δμ = μe/2
    def integral_limit_eqn_plus(p):
        return δμ - EΔ(p, ϕ, Δ, μ_bar)
    
    if np.sign(integral_limit_eqn_plus(0)) == np.sign(integral_limit_eqn_plus(Λ_cutoff)):
        integral_limit_plus = 0
    else:
        integral_limit_plus = optimize.brentq(integral_limit_eqn_plus, 0, Λ_cutoff)

    def integral_limit_eqn_min(p):
        return δμ - EΔ(p, ϕ, Δ, -μ_bar)
    
    if np.sign(integral_limit_eqn_min(0)) == np.sign(integral_limit_eqn_min(Λ_cutoff)):
        integral_limit_min = 0
    else:
        integral_limit_min = optimize.brentq(integral_limit_eqn_min, 0, Λ_cutoff)
    
    #def integrand_1(p):
    #    return p**2*(E(p, ϕ) + μ_bar)/EΔ(p, ϕ, Δ, μ_bar)
        
    #def integrand_2(p):
    #    return p**2*(E(p, ϕ) - μ_bar)/EΔ(p, ϕ, Δ, -μ_bar)

    def integrand_1(p):
        return p**2#*(E(p, ϕ) + μ_bar)/EΔ(p, ϕ, Δ, μ_bar)
        
    def integrand_2(p):
        return p**2#*(E(p, ϕ) - μ_bar)/EΔ(p, ϕ, Δ, -μ_bar)
    
    integral_1, err_1 = integrate.quad(integrand_1, 0, integral_limit_plus)
    integral_2, err_2 = integrate.quad(integrand_2, 0, integral_limit_min)
    
    term1 = -4*(integral_1 - integral_2)*(4*np.pi)/((2*np.pi)**3)

    return term1

#Number density ne, n8 and nq
def num_density_e_NQM(ϕ, μ, μe):
    
    def dΩ_NQMdμu(ϕ, μ, μe):
        μ_u = μ - 2/3*μe
        return -16/(3*(4*np.pi)**2)*np.real(np.emath.sqrt(μ_u**2 - ϕ**2)**3)
    
    def dΩ_NQMdμd(ϕ, μ, μe):
        μ_d = μ + 1/3*μe
        return -16/(3*(4*np.pi)**2)*np.real(np.emath.sqrt(μ_d**2 - ϕ**2)**3)
        
    return 2/3*dΩ_NQMdμu(ϕ, μ, μe) - 1/3*dΩ_NQMdμd(ϕ, μ, μe) + 16*μe**3/(3*(4*np.pi)**2)

def num_density_e_2SC(ϕ, Δ, μ, μe, μ8, gΔ, Λ_cutoff):
    return 2/3*dΩdμ_ub(ϕ, μ, μe, μ8, Λ_cutoff) - 1/3*dΩdμ_db(ϕ, μ, μe, μ8, Λ_cutoff) + 1/6*dΩdμ_bar(ϕ, Δ, μ, μe, μ8, gΔ, Λ_cutoff) - 1/2*dΩdδμ(ϕ, Δ, μ, μe, μ8, Λ_cutoff) + 16/(3*(4*np.pi)**2)*μe**3

def num_density_8_2SC(ϕ, Δ, μ, μe, μ8, gΔ, Λ_cutoff):
    return 2/3*dΩdμ_ub(ϕ, μ, μe, μ8, Λ_cutoff) + 2/3*dΩdμ_db(ϕ, μ, μe, μ8, Λ_cutoff) - 1/3*dΩdμ_bar(ϕ, Δ, μ, μe, μ8, gΔ, Λ_cutoff)

def num_density_q_2SC(ϕ, Δ, μ, μe, μ8, gΔ, Λ_cutoff):
    return -dΩdμ_ub(ϕ, μ, μe, μ8, Λ_cutoff) - dΩdμ_db(ϕ, μ, μe, μ8, Λ_cutoff) - dΩdμ_bar(ϕ, Δ, μ, μe, μ8, gΔ, Λ_cutoff)

#Equation of state - returns pressure and energy density
Bag_const = grand_potential_NQM(300, 0, 0)
def Equation_of_state_2SC(ϕ, Δ, μ, μe, μ8, gΔ, mΔ, λ3, λΔ, Λ_cutoff):
    
    def Ωe(μe):
        return -1/(12*np.pi**2)*μe**4
    
    def Ωϕ(ϕ):
        if ϕ == 0:
            return 0 
        else: 
            return  h0*fπ*ϕ/mq - 1/2*m02*fπ**2*ϕ**2/(mq**2) + 1/24*λ0*fπ**4*ϕ**4/(mq**4) + 2*mq**4/((4*np.pi)**2)*(np.log(mq**2/(ϕ**2)) -(Cπ + mπ**2*dCπ))*ϕ**4/(mq**4) + 2*Nc*mq**4/((4*np.pi)**2)*(3/2)*ϕ**4/(mq**4)

    def ΩϕΔ(ϕ, Δ, μ, μe, μ8):
        μ_bar = μ - μe/6 + μ8/3 
        if ϕ == 0 and Δ == 0:
            return 0
        if ϕ == 0:
            return  (mΔ**2 - 4*μ_bar**2)/(gΔ**2)*Δ**2 - 16/((4*np.pi)**2)*(np.log(mq**2/(Δ**2))-(Cπ+mπ**2*dCπ))*μ_bar**2*Δ**2 + λΔ/(6*gΔ**4)*Δ**4 + 6/((4*np.pi)**2)*Δ**4 + 4/((4*np.pi)**2)*(np.log(mq**2/(Δ**2))-(Cπ + mπ**2*dCπ))*(Δ**4)
        if Δ == 0:
            return   4*mq**4/((4*np.pi)**2)*(np.log((mq**2)/(ϕ**2)) -(Cπ+mπ**2*dCπ))*ϕ**4/(mq**4)
        else:
            return 1/12*λ3*fπ**2/(gΔ**2*mq**2)*Δ**2*ϕ**2 + 12/((4*np.pi)**2)*ϕ**2*Δ**2 + (mΔ**2 - 4*μ_bar**2)/(gΔ**2)*Δ**2 - 16/((4*np.pi)**2)*(np.log(mq**2/(Δ**2+ϕ**2))-(Cπ+mπ**2*dCπ))*μ_bar**2*Δ**2 + λΔ/(6*gΔ**4)*Δ**4 + 6/((4*np.pi)**2)*Δ**4 + 4/((4*np.pi)**2)*(np.log(mq**2/(Δ**2+ϕ**2))-(Cπ + mπ**2*dCπ))*(Δ**4) +  8/((4*np.pi)**2)*(np.log(mq**2/(Δ**2+ϕ**2))-(Cπ + mπ**2*dCπ))*Δ**2*ϕ**2 + 4*mq**4/((4*np.pi)**2)*(np.log((mq**2)/(ϕ**2+Δ**2)) -(Cπ+mπ**2*dCπ))*ϕ**4/(mq**4)
        
    def Ω_num(ϕ, Δ, μ, μe, μ8):
        if Δ ==0:
            return 0
        else:
            μ_bar = μ - μe/6 + μ8/3
            def integrand(p):
                return p**2*(EΔ(p, ϕ, Δ, μ_bar) + EΔ(p, ϕ, Δ, -μ_bar) - 2*np.sqrt(p**2+ϕ**2+Δ**2) - Δ**2*μ_bar**2/((p**2+ϕ**2+Δ**2)**(3/2))
                                )
            integral, err = integrate.quad(integrand, 0, Λ_cutoff)
        
            return 2*integral/(np.pi**2)

    #We define one potential without the original temperature part Ω1 and the total as Ω_tot.
    def Ω1(ϕ, Δ, μ, μe, μ8):
        return (Ωϕ(ϕ) + ΩϕΔ(ϕ, Δ, μ, μe, μ8)  - Ω_num(ϕ, Δ, μ, μe, μ8) + Ωe(μe))/fπ**4
    
    def Pressure(ϕ, Δ, μ, μe, μ8):
        ϕΔμeμ8 = [ϕ, Δ, μe, μ8]
        return -grand_potential_2SC(ϕ, Δ, μ, μe, μ8, gΔ, mΔ, λ3, λΔ, Λ_cutoff) - Bag_const
    
    def Energy_Density(ϕ, Δ, μ, μe, μ8):
        
        μ_ub = μ - 2*μe/3 - 2*μ8/3 
        μ_db = μ + μe/3 - 2*μ8/3
        
        μ_bar = μ - μe/6 + μ8/3
        δμ = μe/2

        def integral_limit_eqn_plus(p):
            return δμ - EΔ(p, ϕ, Δ, μ_bar)
    
        if np.sign(integral_limit_eqn_plus(0)) == np.sign(integral_limit_eqn_plus(Λ_cutoff)):
            integral_limit_plus = 0
        else:
            integral_limit_plus = optimize.brentq(integral_limit_eqn_plus, 0, Λ_cutoff)
    
        def integral_limit_eqn_min(p):
            return δμ - EΔ(p, ϕ, Δ, -μ_bar)
        
        if np.sign(integral_limit_eqn_min(0)) == np.sign(integral_limit_eqn_min(Λ_cutoff)):
            integral_limit_min = 0
        else:
            integral_limit_min = optimize.brentq(integral_limit_eqn_min, 0, Λ_cutoff)
        
        def integrand1(p):
            return p**2*(μ_ub - E(p, ϕ))*step(μ_ub - E(p, ϕ))
        
        def integrand2(p):
            return p**2*(μ_db - E(p, ϕ))*step(μ_db - E(p, ϕ))
        
        def integrand3(p):
            return p**2*(δμ - EΔ(p, ϕ, Δ, μ_bar))*step(δμ - EΔ(p, ϕ, Δ, μ_bar))
        
        def integrand4(p):
            return p**2*(δμ - EΔ(p, ϕ, Δ, -μ_bar))*step(δμ - EΔ(p, ϕ, Δ, -μ_bar))
        
        integral1, err1 = integrate.quad(integrand1, 0, np.real(np.emath.sqrt(μ_ub**2 - ϕ**2)))
        integral2, err2 = integrate.quad(integrand2, 0, np.real(np.emath.sqrt(μ_db**2 - ϕ**2)))
        integral3, err3 = integrate.quad(integrand3, 0, integral_limit_plus)
        integral4, err4 = integrate.quad(integrand4, 0, integral_limit_min)
        
        return (μ*num_density_q_2SC(ϕ, Δ, μ, μe, μ8, gΔ, Λ_cutoff) + μe*num_density_e_2SC(ϕ, Δ, μ, μe, μ8, gΔ, Λ_cutoff) + μ8*num_density_8_2SC(ϕ, Δ, μ, μe, μ8, gΔ, Λ_cutoff) + Ω1(ϕ, Δ, μ, μe, μ8)*fπ**4 - 2*(integral1 + integral2)*4*np.pi/(2*np.pi)**3 - 4*(integral3 + integral4)*4*np.pi/(2*np.pi)**3)/fπ**4 + Bag_const
    return Pressure(ϕ, Δ, μ, μe, μ8), Energy_Density(ϕ, Δ, μ, μe, μ8)

#Functions needed to solve the gap equations
def dΩϕdϕ(ϕ):
    return h0/g0 - m02*ϕ/(g0**2) + 1/6*λ0*ϕ**3/(g0**4) + 8/((4*np.pi)**2)*(np.log(mq**2/(ϕ**2)) - Cπ - mπ**2*dCπ)*ϕ**3 + 12*Nc/((4*np.pi)**2)*ϕ**3 - 4/((4*np.pi)**2)*ϕ**3

def dlogdϕ(ϕ, Δ):
    return 2*ϕ/(ϕ**2 + Δ**2)

def dlogdΔ(ϕ, Δ):
    return 2*Δ/(ϕ**2 + Δ**2)

def dΩvacdϕ(ϕ, Δ, μ, μe, μ8, gΔ, λ3):
    μ_bar = μ - μe/6 + μ8/3
    line1 = h0/g0 - m02*ϕ/(g0**2) + 1/6*λ0*ϕ**3/(g0**4) + λ3/6*ϕ*Δ**2/(g0**2*gΔ**2) + 24/((4*np.pi)**2)*ϕ*Δ**2 + 12*Nc/((4*np.pi)**2)*ϕ**3 
    line2 = 8/((4*np.pi)**2)*(np.log(mq**2/(ϕ**2)) - Cπ - mπ**2*dCπ)*ϕ**3 - 4/((4*np.pi)**2)*ϕ**3
    line3 = 16/((4*np.pi)**2)*(np.log(mq**2/(ϕ**2 + Δ**2)) - Cπ - mπ**2*dCπ)*ϕ*(ϕ**2 + Δ**2)
    line4 = -8/((4*np.pi)**2)*ϕ/(ϕ**2 + Δ**2)*((ϕ**2 + Δ**2)**2 - 4*μ_bar**2*Δ**2)
    return line1 + line2 + line3 + line4

def dΩvacdΔ(ϕ, Δ, μ, μe, μ8, gΔ, mΔ, λ3, λΔ):
    μ_bar = μ - μe/6 + μ8/3
    line1 = 1/6*λ3/(gΔ**2*g0**2)*ϕ**2*Δ + 2*(mΔ**2 - 4*μ_bar**2)/(gΔ**2)*Δ + 2/3*λΔ/(gΔ**4)*Δ**3
    line2 = 24/((4*np.pi)**2)*ϕ**2*Δ + 24/((4*np.pi)**2)*Δ**3 - 8/((4*np.pi)**2)*Δ/(ϕ**2 + Δ**2)*((ϕ**2 + Δ**2)**2 - 4*μ_bar**2*Δ**2)
    line3 = 16/((4*np.pi)**2)*(np.log(mq**2/(ϕ**2 + Δ**2)) - Cπ - mπ**2*dCπ)*(Δ*(ϕ**2 + Δ**2) - 2*μ_bar**2*Δ)
    return line1 + line2 + line3

def dΩ1findϕ(ϕ, Δ, μ_bar, Λ_cutoff):
    def integrand(p):
        return p**2*((E(p, ϕ) + μ_bar)/(E(p, ϕ)*EΔ(p, ϕ, Δ, μ_bar))
                     + (E(p, ϕ) - μ_bar)/(E(p, ϕ)*EΔ(p, ϕ, Δ, -μ_bar))
                    -2/np.sqrt(p**2 + ϕ**2 + Δ**2)
                    +3*Δ**2*μ_bar**2/((p**2+ϕ**2+Δ**2)**(5/2))
                    )
    integral, err = integrate.quad(integrand, 0, Λ_cutoff)
    return -4*ϕ*integral*(4*np.pi)/((2*np.pi)**3)

def dΩ1findΔ(ϕ, Δ, μ_bar, Λ_cutoff):
    def integrand(p):
        return p**2*(1/EΔ(p, ϕ, Δ, μ_bar) 
                     + 1/EΔ(p, ϕ, Δ, -μ_bar) 
                     - 2/np.sqrt(p**2 + ϕ**2 + Δ**2) 
                     - 2*μ_bar**2/((p**2 + ϕ**2 + Δ**2)**(3/2)) 
                     + 3*Δ**2*μ_bar**2/((p**2 + ϕ**2 + Δ**2)**(5/2))
                    )
    integral, err = integrate.quad(integrand, 0, Λ_cutoff)
    return -4*Δ*integral*(4*np.pi)/((2*np.pi)**3)

def dΩμdϕ(ϕ, Δ, μ, μe, μ8, Λ_cutoff):
    μ_bar = μ - μe/6 + μ8/3
    δμ = μe/2
    μ_ub = μ -2/3*μe - 2/3*μ8
    μ_db = μ + 1/3*μe - 2/3*μ8
    
    def integrand_ub(p):
        return p**2/(E(p, ϕ))
    
    def integrand_db(p):
        return p**2/(E(p, ϕ))
    
    p_max_ub = np.real(np.emath.sqrt(μ_ub**2 - ϕ**2))
    p_max_db = np.real(np.emath.sqrt(μ_db**2 - ϕ**2))
    
    integral_ub, err_ub = integrate.quad(integrand_ub, 0, p_max_ub)
    integral_db, err_db = integrate.quad(integrand_db, 0, p_max_db)
    
    def integral_limit_eqn_plus(p):
        return δμ - EΔ(p, ϕ, Δ, μ_bar)
    
    if np.sign(integral_limit_eqn_plus(0)) == np.sign(integral_limit_eqn_plus(Λ_cutoff)):
        p_max_δp = 0
    else:
        p_max_δp = optimize.brentq(integral_limit_eqn_plus, 0, Λ_cutoff)

    def integral_limit_eqn_min(p):
        return δμ - EΔ(p, ϕ, Δ, -μ_bar)
    
    if np.sign(integral_limit_eqn_min(0)) == np.sign(integral_limit_eqn_min(Λ_cutoff)):
        p_max_δm = 0
    else:
        p_max_δm = optimize.brentq(integral_limit_eqn_min, 0, Λ_cutoff)
    
    def integrand_δp(p):
        return 2*p**2*(E(p, ϕ) + μ_bar)/(E(p, ϕ)*EΔ(p, ϕ, Δ, μ_bar))
    
    def integrand_δm(p):
        return 2*p**2*(E(p, ϕ) - μ_bar)/(E(p, ϕ)*EΔ(p, ϕ, Δ, -μ_bar))

    integral_δp, err_δp = integrate.quad(integrand_δp, 0, p_max_δp)
    integral_δm, err_δm = integrate.quad(integrand_δm, 0, p_max_δm)
    return 2*ϕ*(integral_ub + integral_db + integral_δp + integral_δm)*(4*np.pi)/((2*np.pi)**3)

def dΩμdΔ(ϕ, Δ, μ_bar, δμ, Λ_cutoff):
    def integral_limit_eqn_plus(p):
        return δμ - EΔ(p, ϕ, Δ, μ_bar)
    
    if np.sign(integral_limit_eqn_plus(0)) == np.sign(integral_limit_eqn_plus(Λ_cutoff)):
        p_max_δp = 0
    else:
        p_max_δp = optimize.brentq(integral_limit_eqn_plus, 0, Λ_cutoff)

    def integral_limit_eqn_min(p):
        return δμ - EΔ(p, ϕ, Δ, -μ_bar)
    
    if np.sign(integral_limit_eqn_min(0)) == np.sign(integral_limit_eqn_min(Λ_cutoff)):
        p_max_δm = 0
    else:
        p_max_δm = optimize.brentq(integral_limit_eqn_min, 0, Λ_cutoff)
    
    def integrand_δp(p):
        return p**2/EΔ(p, ϕ, Δ, μ_bar)
    
    def integrand_δm(p):
        return p**2/EΔ(p, ϕ, Δ, -μ_bar)

    integral_δp, err_δp = integrate.quad(integrand_δp, 0, p_max_δp)
    integral_δm, err_δm = integrate.quad(integrand_δm, 0, p_max_δm)
    return 4*(integral_δp + integral_δm)*Δ*(4*np.pi)/((2*np.pi)**3)

##Gap equations
#Gap equation in NQM
def dΩ_NQMdϕ(ϕ, μ, μe):
    
    μ_u = μ - 2/3*μe
    μ_d = μ + 1/3*μe
    
    def integrand(p):
        return p**2/E(p, ϕ)
    
    integral_u, err_u = integrate.quad(integrand, 0, np.real(np.emath.sqrt(μ_u**2 - ϕ**2)))
    integral_d, err_d = integrate.quad(integrand, 0, np.real(np.emath.sqrt(μ_d**2 - ϕ**2)))
    
    return h0/g0 - m02*ϕ/(g0**2) + 1/6*λ0*ϕ**3/(g0**4) + 8*Nc/((4*np.pi)**2)*ϕ**3*(np.log(mq**2/(ϕ**2)) - Cπ - mπ**2*dCπ) - 4*Nc/((4*np.pi)**2)*ϕ**3  + 12*Nc/((4*np.pi)**2)*ϕ**3 + 2*Nc*ϕ*(integral_u + integral_d)*(4*np.pi)/((2*np.pi)**3)

#Gap equations for 2SC
#The gap equation for ϕ
def dΩ_2SCdϕ(ϕ, Δ, μ, μe, μ8, gΔ, λ3, Λ_cutoff):
    μ_bar = μ - μe/6 + μ8/3
    δμ = μe/2
    return dΩvacdϕ(ϕ, Δ, μ, μe, μ8, gΔ, λ3) + dΩ1findϕ(ϕ, Δ, μ_bar, Λ_cutoff) + dΩμdϕ(ϕ, Δ, μ, μe, μ8, Λ_cutoff)

#The gap equation for Δ
def dΩ_2SCdΔ(ϕ, Δ, μ, μe, μ8, gΔ, mΔ, λ3, λΔ, Λ_cutoff):
    μ_bar = μ - μe/6 + μ8/3
    δμ = μe/2
    return dΩvacdΔ(ϕ, Δ, μ, μe, μ8, gΔ, mΔ, λ3, λΔ) + dΩ1findΔ(ϕ, Δ, μ_bar, Λ_cutoff) + dΩμdΔ(ϕ, Δ, μ_bar, δμ, Λ_cutoff)
