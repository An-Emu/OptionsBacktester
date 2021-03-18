import datetime
import math
import scipy.stats as stats
import numpy as np

class Contract:
    def __init__(self, inst, K, exp, ctype, quantity):
        self.instrument = inst
        self.strike = K
        self.expiration = exp
        self.opt_type = ctype
        self.qty = quantity

class Position:
    def __init__(self, opndate, contracts):
        self.open_date = opndate
        self.is_open = 1
        self.contracts = contracts
        self.value_history = []
        self.xt_history = [] #total value of position (including BPR)
        self.dyt_history = [] #history of ln(x_{t+dt}/x_t)
        
    def close_pos(self, clsdate):
        self.is_open = 0        
        self.close_date = clsdate
        self.close_price = self.value
        if self.close_price >= self.init_value:
            self.win=1
        else:
            self.win=0
        
    def update_value(self,date):
        self.value = 0
        self.delta = 0
        self.theta = 0
        self.rho = 0
        self.vega = 0
        self.gamma = 0
        for c in self.contracts:
            self.value = self.value + c.price
            self.delta = self.delta + c.delta
            self.theta = self.theta + c.theta
            self.rho = self.rho + c.rho
            self.vega = self.vega + c.vega
            self.gamma = self.gamma + c.gamma
        
        self.days_held = (date - self.open_date).days
        self.value_history.append([date,self.value])
        
        if date == self.open_date:
            self.init_value = self.value
            bpe = 0
            for c in self.contracts:
                bpe = bpe + c.strike*c.qty*100                    
            if bpe>0: bpe=0
            self.bpr = -(bpe - self.init_value*100) #initial investment or margin requirement
        
        self.pl = (self.value - self.init_value)*100
        self.pl_pct = self.pl/self.bpr #percent gain or loss of initial investment
        self.xt_history.append(self.pl+self.bpr) #total position value at time t
        self.rt = self.xt_history[-1]/self.bpr #position return (multiple on investment)

        
        if date > self.open_date:
            #history of daily log returns
            self.dyt_history.append(math.log(max(self.xt_history[-1]/self.xt_history[-2], 1e-8)))
            self.dgr = self.rt**(1/self.days_held) #daily growth rate
            self.cagr = self.dgr**365 #annual growth multiple
            self.cagr_pct = (self.cagr-1) #annual growth rate (%)
            self.ccgr = math.log(max(self.cagr,1e-8)) #continuously compounded growth rate per year
            self.volatility_annual = np.std(self.dyt_history)*math.sqrt(365)
            self.gsd = math.exp(self.volatility_annual) #geometric standard dev
            self.gsd_pct = (self.gsd-1)
            
        #phistory=[row[1] for row in self.value_history]
        
def est_atm_vol(VIX, VXST, t):
    """
    Estimates at-the-money volatility for SPX or SPY options with time to 
    expiration, t, based on correlations from a small set of historical data.
    
    INPUTS
    VIX = Price of the VIX on date of interest (i.e. 15)
    VXST = Price of VXST (9-day volatility index) on date of interest (i.e. 15)
    t = time to expiration, in years
    
    CONSTANTS
    c1 = fit coefficient for equation sigma_BS(0,30/365) ~= c1*VIX+c0, based on
    historical data
    c0 = fit coefficient for equation sigma_BS(0,30/365) ~= c1*VIX+c0, based on
    historical data
    
    OUTPUTS
    sigma_BS(0,t) = at-the-money volatility for options expiring in t years, on
    the date of interest.
    """

    c1 = 0.899
    c0 = -0.843/100
    sigmaVIX = VIX/100
    sigmaVXST = VXST/100

    if (t <= 9/365):
        iv = c1*sigmaVXST + c0 # If less than 9 days to expiration, use VXST 
        
    elif (t<=30/365):
        # If 9 < days to expiration <= 30, use linear interpolation between VXST and VIX
        iv = 365/21*c1*(sigmaVIX-sigmaVXST)*(t-9/365) + c1*sigmaVXST+c0 
    else:
        iv = c1*sigmaVIX + c0 # If greater than 30 days to expiration, use VIX
        
    return iv

def ssvi(beta1, beta2, gamma1, gamma2, eta, rho, k, t, theta):
    """
    Calculate implied volatility based on the SSVI model: 
    https://mfe.baruch.cuny.edu/wp-content/uploads/2013/04/BloombergSVI2013.pdf
    
    INPUTS
    beta1,beta2,gamma1,gamma2 = constants that do not vary with time. 
                                specific to each underlying. For SPX 
                                (fit from 2001-2011): beta1= 177.682811, 
                                beta2 = 0.04978706837, gamma1 = 0.238, 
                                gamma2 = 0.253
    eta, rho = time dependent constants. Avg for SPX from 2001-2011:
               eta ~= 3.2441 ± 0.6, rho ~= -.6814 ± 0.11
    k = log strike. k=ln(K/F), F=e^(r*t)*S, where r is the risk free interest
        rate, t is time to expiration in years, and S is the price of the 
        underlying.
    t = time to expiration, in years
    theta = (ATM volatility)^2*t. for first cut, can assume VIX for ATM
            volatility on SPX. For better model, would need true ATM volatility
            for the expiration cycle of interest.
    
    OUTPUTS
    ImpliedVolatility in black-scholes form.
    """
    #catch for edge cases. needs review
    if t < 5.708e-5:
        return .15
    
    phi=eta/(theta**gamma1*(1+beta1*theta)**gamma2*(1+beta2*theta)**(1-gamma1-gamma2))
    w = theta/2*(1+rho*phi*k+((phi*k+rho)**2+(1-rho**2))**0.5)

    # skew = (phi*rho*theta**0.5)/(2*t**0.5)
    # Return Implied Volatility
    return (w/t)**0.5

def bs_model(contract, stock, rate, div, sigma, time):
    """
    Calculate prices of calls and puts based on the Black-Scholes model
    https://quantpie.co.uk/bsm_formula/bs_summary.php
    
    Checked outputs against https://www.math.drexel.edu/~pg/fin/VanillaCalculator.html
    
    INPUTS
    stock = Price of the underlying
    strike = strike price of the option
    rate = risk free interest rate (annualized)
    div = dividend yield (annualized)
    sigma = Volatility
    time = time to expiration [years]
    opt_type = Specifies whether option is call or put. To specify call, use "Call","call","C", or "c". Otherwise opt_type is assumed to be Put.
    
    OUTPUTS
    Outputs an array, [Price,delta,theta_daily,vega,gamma,rho], where Price is the price of the option, and the option greeks are given in elements 2-6.
    
    To return a specific element of the array, use the index function. For example,
    index(BSOption(S,K,r,d,sigma,t,"c"),2)
    would return the delta of the option price.
    
    To calculate price, greeks, of a multi-leg position can use ArrayFormula. For example,
    ArrayFormula(-5*BSOption(S, Kshort, r, d, sigmaShort, t, "p")+5*BSOption(S, Klong, r, d, sigmaLong, t, "p"))
    would return price & net-greeks for QTY 5 vertical put spreads with strikes Kshort and Klong, and time to expiration t.
    """
    
    strike = contract.strike
    opt_type = contract.opt_type
    
    if (time <= 5.708e-5) :
        time = 5.708e-5 #1hr left
#        d1 = (math.log(stock/strike))
#    else:
#        d1 = (math.log(stock/strike) + (rate - div + sigma*sigma/2.0)*time)/(sigma*math.sqrt(time))    
    d1 = (math.log(stock/strike) + (rate - div + sigma*sigma/2.0)*time)/(sigma*math.sqrt(time))     
    d2 = d1 - sigma*math.sqrt(time)
    
    # useful calculations
    nd1 = (2*math.pi)**-0.5*math.exp(-0.5*d1**2); # dN(d1)/d(d1) = N'(d1)
    emdivt = math.exp(-div*time)
    emrt = math.exp(-rate * time)
    Nd1 = stats.norm.cdf(d1)
    Nd2 = stats.norm.cdf(d2)
    Nmd1 = stats.norm.cdf(-d1)
    Nmd2 = stats.norm.cdf(-d2)
        
    # non-type dependent partial derivatives (gamma and vega)
    contract.gamma = emdivt*nd1/(stock*sigma*time**0.5)
    contract.vega = stock*emdivt*(time**0.5)*nd1
    
    # Calculate option prices and type-specific partial derivatives (delta,rho,theta)
    if (opt_type == "c" or opt_type == "Call" or opt_type == "call" or opt_type == "C"):
        # call
        contract.price = contract.qty*max(0,(emdivt*stock*Nd1 - strike*emrt*Nd2))
        contract.delta = contract.qty*(emdivt*Nd1)
        contract.rho = contract.qty*(strike*emrt*time*Nd2)
        theta_yr = contract.qty*(stock*emdivt*div*Nd1 - strike*emrt*rate*Nd2 - stock*emdivt*0.5*sigma*nd1*time**-0.5)
        contract.theta = theta_yr/365
    else:
        # put 
        contract.price = contract.qty*max(0,(strike*emrt + (emdivt*stock*Nd1 - strike*emrt*Nd2) - emdivt*stock))
        contract.delta = contract.qty*(-emdivt*Nmd1)
        contract.rho = contract.qty*(-strike*emrt*time*Nmd2)
        theta_yr = contract.qty*(-stock*emdivt*div*Nmd1 + strike*emrt*rate*Nmd2 - stock*emdivt*(0.5*sigma)*nd1*(time**-0.5))
        contract.theta = theta_yr/365

    #return [Price,delta,theta_daily,vega,gamma,rho]

def strike_from_delta(stock, delta, rate, div, sigma, time, opt_type):
    """
    Given the delta of an option, solve for the option's strike price per the Black-Scholes model.

    INPUTS
    stock = Price of the underlying
    delta = delta (d/dS) of the option
    rate = risk free interest rate (annualized)
    div = dividend yield (annualized)
    sigma = Volatility
    time = time to expiration [years]
    opt_type = Specifies whether option is call or put. To specify call, use "Call","call","C", or "c". Otherwise opt_type is assumed to be Put.
    
    OUTPUTS
    Returns the strike of the specified option
    """
    if time<5.708e-5: 
        time=5.708e-5
    
    if (opt_type == "c" or opt_type == "Call" or opt_type == "call" or opt_type == "c"):
        d1 = stats.norm.ppf(delta*math.exp(div*time))
    else:
        d1 = stats.norm.ppf((delta+math.exp(-div*time))*math.exp(div*time))

    # Return strike
    return stock*math.exp(-(d1*sigma*math.sqrt(time)-(rate-div+sigma*sigma/2)*time))

def strike_from_delta2(stock, delta, rate, div, atm_vol, time, opt_type, beta1, beta2, gamma1, gamma2, eta, rho, tol):
    
    #init
    if time<5.708e-5: 
        time=5.708e-5
    vol = atm_vol
    dK = tol*2
    K0 = 0
    #loop to converge on strike
    while dK > tol:
        K = strike_from_delta(stock, delta, rate, 0, vol, time, opt_type)
        k = math.log(K/(stock*math.exp(rate*time)))
        theta = atm_vol**2*time
        vol = ssvi(beta1, beta2, gamma1, gamma2, eta, rho, k, time, theta)
        dK = abs(K-K0)
        K0 = K
    #round strike to nearest dollar (update to enable .50?)
    K = round(float(K), 0)
    return K
        

def get_next_high_val(val, valuelist):
    valuelist = (int(t) for t in valuelist if t != '')
    valuelist = [t for t in valuelist if t <= int(val)]
    if valuelist: return max(valuelist)
    else: return None                   # or raise an error
    
def get_third_friday(d, lbnd, ubnd):
    #search between lbnd and ubnd days out from d
    for i in range(lbnd, ubnd):
        dexp = d + datetime.timedelta(days=i)
        if (dexp.weekday() == 4 and 15 <= dexp.day <= 21):
            return dexp
    
    raise RuntimeError('Could not find valid 3rd friday within specified range')

def xirr(capital_array,time_held_array,final_acct_value,r=0.1,tol=0.1):
    """ INPUTS 
    capital_array = array of numbers indicating how much was invested in each
        individual contribution.
    time_held_array = array indicating time (in years) for which each 
        corresponding investment in capital_array was held.
    
                
    
    """
    #input checking
    c = np.array(capital_array)
    t = np.array(time_held_array)
    l = len(np.atleast_1d(c))
    F = float(final_acct_value)
    if len(np.atleast_1d(t)) != l:
        raise RuntimeError('Capital history and time history must be equal')
    elif r<-1:
        raise Warning('r must be >=-1')
        r=-1
    elif final_acct_value < 0:
        'Invalid final acct value'
    
    #init    
    computed_final_val = F - tol*10
    while (abs(computed_final_val - F) > tol):
        computed_final_val = np.sum(c*(1+r)**t)
                
        #calculate derivative and update r
        dFdr = np.sum(t*c*(1+r)**(t-1))
        r = (F - computed_final_val)/dFdr + r
        if r<-1:
            r=-1
    
    return r #return rate of return
                    