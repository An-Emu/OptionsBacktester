import datetime
import math
import scipy.stats as stats

class Contract:
    def __init__(self, inst, K, exp, ctype, quantity):
        self.instrument = inst
        self.strike = K
        self.expiration = exp
        self.optType = ctype
        self.qty = quantity
        #need to track price history for each contract

class Position:
    def __init__(self, opndate, contractList):
        self.openDate = opndate
        self.isOpen = 1
        self.contractList = contractList
        #need to track price history for each contract
        #should assign open price
        
    def closePos(self, clsdate):
        self.isOpen = 0
        self.closeDate = clsdate
        #should assign close price, close date, cacl P/L
        
    def updateValue(self):
        self.value = 0
        self.delta = 0
        self.theta = 0
        self.rho = 0
        self.vega = 0
        self.gamma = 0
        for c in self.contractList:
            self.value = self.value + c.price
            self.delta = self.delta + c.delta
            self.theta = self.theta + c.theta
            self.rho = self.rho + c.rho
            self.vega = self.vega + c.vega
            self.gamma = self.gamma + c.gamma

#need to track price history for account
        
def estATMVol(VIX, VXST, t):
    """
    Estimates at-the-money volatility for SPX or SPY options with time to 
    expiration, t, based on correlations from a small set of historical data.
    
    INPUTS
    VIX = Price of the VIX on date of interest (i.e. 15)
    VXST = Price of VXST (9-day volatility index) on date of interest (i.e. 15)
    t = time to expiration, in years
    
    CONSTANTS
    c1 = fit coefficient for equation Sigma_BS(0,30/365) ~= c1*VIX+c0, based on
    historical data
    c0 = fit coefficient for equation Sigma_BS(0,30/365) ~= c1*VIX+c0, based on
    historical data
    
    OUTPUTS
    Sigma_BS(0,t) = at-the-money volatility for options expiring in t years, on
    the date of interest.
    """

    c1 = 0.899
    c0 = -0.843/100
    SigmaVIX = VIX/100
    SigmaVXST = VXST/100

    if (t <= 9/365):
        IV = c1*SigmaVXST + c0 # If less than 9 days to expiration, use VXST 
        
    elif (t<=30/365):
        # If 9 < days to expiration <= 30, use linear interpolation between VXST and VIX
        IV = 365/21*c1*(SigmaVIX-SigmaVXST)*(t-9/365) + c1*SigmaVXST+c0 
    else:
        IV = c1*SigmaVIX + c0 # If greater than 30 days to expiration, use VIX
        
    return IV

def SSVI(beta1, beta2, gamma1, gamma2, eta, rho, k, t, theta):
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

def BSmodel(contract, Stock, Rate, Div, Sigma, Time):
    """
    Calculate prices of calls and puts based on the Black-Scholes model
    https://quantpie.co.uk/bsm_formula/bs_summary.php
    
    Checked outputs against https://www.math.drexel.edu/~pg/fin/VanillaCalculator.html
    
    INPUTS
    Stock = Price of the underlying
    Strike = Strike price of the option
    Rate = risk free interest rate (annualized)
    Div = dividend yield (annualized)
    Sigma = Volatility
    Time = time to expiration [years]
    OptType = Specifies whether option is call or put. To specify call, use "Call","call","C", or "c". Otherwise OptType is assumed to be Put.
    
    OUTPUTS
    Outputs an array, [Price,delta,theta_daily,vega,gamma,rho], where Price is the price of the option, and the option greeks are given in elements 2-6.
    
    To return a specific element of the array, use the index function. For example,
    index(BSOption(S,K,r,d,sigma,t,"c"),2)
    would return the delta of the option price.
    
    To calculate price, greeks, of a multi-leg position can use ArrayFormula. For example,
    ArrayFormula(-5*BSOption(S, Kshort, r, d, sigmaShort, t, "p")+5*BSOption(S, Klong, r, d, sigmaLong, t, "p"))
    would return price & net-greeks for QTY 5 vertical put spreads with strikes Kshort and Klong, and time to expiration t.
    """
    
    Strike = contract.strike
    OptType = contract.optType
    
    if (Time <= 5.708e-5) :
        Time = 5.708e-5 #1hr left
        d1 = (math.log(Stock/Strike))
    else:
        d1 = (math.log(Stock/Strike) + (Rate - Div + Sigma*Sigma/2.0)*Time)/(Sigma*math.sqrt(Time))    
        
    d2 = d1 - Sigma*math.sqrt(Time)
    
    # useful calculations
    nd1 = (2*math.pi)**-0.5*math.exp(-0.5*d1**2); # dN(d1)/d(d1) = N'(d1)
    emdivt = math.exp(-Div*Time)
    emrt = math.exp(-Rate * Time)
    Nd1 = stats.norm.cdf(d1)
    Nd2 = stats.norm.cdf(d2)
    Nmd1 = stats.norm.cdf(-d1)
    Nmd2 = stats.norm.cdf(-d2)
        
    # non-type dependent partial derivatives (gamma and vega)
    contract.gamma = emdivt*nd1/(Stock*Sigma*Time**0.5)
    contract.vega = Stock*emdivt*(Time**0.5)*nd1
    
    # Calculate option prices and type-specific partial derivatives (delta,rho,theta)
    if (OptType == "c" or OptType == "Call" or OptType == "call" or OptType == "C"):
        # call
        contract.price = contract.qty*max(0,(emdivt*Stock*Nd1 - Strike*emrt*Nd2))
        contract.delta = contract.qty*(emdivt*Nd1)
        contract.rho = contract.qty*(Strike*emrt*Time*Nd2)
        theta_yr = contract.qty*(Stock*emdivt*Div*Nd1 - Strike*emrt*Rate*Nd2 - Stock*emdivt*0.5*Sigma*nd1*Time**-0.5)
        contract.theta = theta_yr/365
    else:
        # put 
        contract.price = contract.qty*max(0,(Strike*emrt + (emdivt*Stock*Nd1 - Strike*emrt*Nd2) - emdivt*Stock))
        contract.delta = contract.qty*(-emdivt*Nmd1)
        contract.rho = contract.qty*(-Strike*emrt*Time*Nmd2)
        theta_yr = contract.qty*(-Stock*emdivt*Div*Nmd1 + Strike*emrt*Rate*Nmd2 - Stock*emdivt*(0.5*Sigma)*nd1*(Time**-0.5))
        contract.theta = theta_yr/365

    #return [Price,delta,theta_daily,vega,gamma,rho]

def strikeFromDelta(Stock, Delta, Rate, Div, Sigma, Time, OptType):
    """
    Given the delta of an option, solve for the option's strike price per the Black-Scholes model.

    INPUTS
    Stock = Price of the underlying
    Delta = delta (d/dS) of the option
    Rate = risk free interest rate (annualized)
    Div = dividend yield (annualized)
    Sigma = Volatility
    Time = time to expiration [years]
    OptType = Specifies whether option is call or put. To specify call, use "Call","call","C", or "c". Otherwise OptType is assumed to be Put.
    
    OUTPUTS
    Returns the Strike of the specified option
    """
    if Time<5.708e-5: 
        Time=5.708e-5
    
    if (OptType == "c" or OptType == "Call" or OptType == "call" or OptType == "c"):
        d1 = stats.norm.ppf(Delta*math.exp(Div*Time))
    else:
        d1 = stats.norm.ppf((Delta+math.exp(-Div*Time))*math.exp(Div*Time))

    # Return Strike
    return Stock*math.exp(-(d1*Sigma*math.sqrt(Time)-(Rate-Div+Sigma*Sigma/2)*Time))

def strikeFromDelta2(Stock, Delta, Rate, Div, ATMVol, Time, OptType, beta1, beta2, gamma1, gamma2, eta, rho, tol):
    
    #init
    vol = ATMVol
    dK = tol*2
    K0 =0
    #loop to converge on strike
    while dK > tol:
        K=strikeFromDelta(Stock, Delta, Rate, 0, vol, Time, OptType)
        k = math.log(K/(Stock*math.exp(Rate*Time)))
        theta = ATMVol**2*Time
        vol = SSVI(beta1,beta2,gamma1,gamma2,eta,rho,k,Time,theta)
        dK = abs(K-K0)
        K0 = K
    #round strike to nearest dollar (update to enable .50?)
    K=round(float(K),0)
    return K
        

def GetNextHighVal(val, valuelist):
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