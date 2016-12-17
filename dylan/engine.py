import abc
import numpy as np
from scipy.stats import binom


class PricingEngine(object, metaclass=abc.ABCMeta):
    """
    An option pricing engine interface.

    """

    @abc.abstractmethod
    def calculate(self):
        """
        A method to implement an option pricing model.

        The pricing method may be either an analytic model (i.e. Black-Scholes or Heston) or
        a numerical method such as lattice methods or Monte Carlo simulation methods.

        """

        pass


class BinomialPricingEngine(PricingEngine):
    """
    A concrete PricingEngine class that implements the Binomial model.

    Args:
        

    Attributes:


    """

    def __init__(self, steps, pricer):
        self.__steps = steps
        self.__pricer = pricer

    @property
    def steps(self):
        return self.__steps

    @steps.setter
    def steps(self, new_steps):
        self.__steps = new_steps

    def calculate(self, option, data):
        return self.__pricer(self, option, data)


def EuropeanBinomialPricer(pricing_engine, option, data):
    """
    The binomial option pricing model for a plain vanilla European option.

    Args:
        pricing_engine (PricingEngine): a pricing method via the PricingEngine interface
        option (Payoff):                an option payoff via the Payoff interface
        data (MarketData):              a market data variable via the MarketData interface

    """

    expiry = option.expiry
    strike = option.strike
    (spot, rate, volatility, dividend) = data.get_data()
    steps = pricing_engine.steps
    nodes = steps + 1
    dt = expiry / steps 
    u = np.exp((rate * dt) + volatility * np.sqrt(dt)) 
    d = np.exp((rate * dt) - volatility * np.sqrt(dt))
    pu = (np.exp(rate * dt) - d) / (u - d)
    pd = 1 - pu
    disc = np.exp(-rate * expiry)
    spotT = 0.0
    payoffT = 0.0
    
    for i in range(nodes):
        spotT = spot * (u ** (steps - i)) * (d ** (i))
        payoffT += option.payoff(spotT)  * binom.pmf(steps - i, steps, pu)  
    price = disc * payoffT 
     
    return price 

def AmericanBinomialPricer(pricing_engine, option, data):
    """
    The binomial option pricing model for a plain vanilla American option.

    Args:
        pricing_engine (PricingEngine): a pricing method via the PricingEngine interface
        option (Payoff):                an option payoff via the Payoff interface
        data (MarketData):              a market data variable via the MarketData interface

    """

    expiry = option.expiry
    strike = option.strike
    (spot, rate, volatility, dividend) = data.get_data()
    steps = pricing_engine.steps
    nodes = steps + 1
    dt = expiry / steps 
    u = np.exp((rate * dt) + volatility * np.sqrt(dt)) 
    d = np.exp((rate * dt) - volatility * np.sqrt(dt))
    pu = (np.exp(rate * dt) - d) / (u - d)
    pd = 1 - pu
    disc = np.exp(-rate * expiry)
    dpu = disc * pu
    dpd = disc * pd
    spotT = np.zeros(nodes)
    payoffT = np.zeros(nodes)
    
    for i in range(nodes):
        spotT[i] = spot * (u ** (steps - i)) * (d ** (i))
        payoffT[i] = option.payoff(spotT[i])  
        
    for i in range(steps- 1, -1, -1):
        for j in range(i + 1):
            payoffT[j] = (dpu * payoffT[j]) + (dpd * payoffT[j+1])
            spotT[j] = spotT[j] / u
            payoffT[j] = np.maximum(payoffT[j], option.payoff(spotT[j]))
            
    return payoffT[0]



class MonteCarloPricingEngine(PricingEngine):
    """
    Doc string
    """

    def __init__(self, reps, steps, pricer):
        self.__reps = reps
        self.__steps = steps
        self.__pricer = pricer

    @property
    def reps(self):
        return self.__reps

    @reps.setter
    def reps(self, new_reps):
        self.__reps = new_reps

    @property
    def steps(self):
        return self.__steps

    @steps.setter
    def steps(self, new_steps):
        self.__steps = new_steps

    def calculate(self, option, data):
        return self.__pricer(self, option, data)


def NaiveMonteCarloPricer(pricing_engine, option, data):
    """
    Doc string
    """

    expiry = option.expiry
    strike = option.strike
    (spot, rate, volatility, dividend) = data.get_data()
    reps = pricing_engine.reps
    steps = pricing_engine.steps
    disc = np.exp(-rate * expiry)
    dt = expiry / steps

    nudt = (rate - dividend - 0.5 * volatility * volatility) * dt
    sigsdt = volatility * np.sqrt(dt)
    z = np.random.normal(size=reps)

    spotT = spot * np.exp(nudt + sigsdt * z)
    callT = option.payoff(spotT)

    return callT.mean() * disc

def GeometricAsianCall(spot, strike, rate, volatility, dividend, expiry, steps):
    dt = expiry/steps
    nu = rate - dividend - 0.5 * volatility * volatility
    a = steps * (steps - 1) + (2.0 * steps + 1.0) / 6.0
    V = np.exp(-rate * expiry) * spot * np.exp(((steps + 1) * nu / 2.0 + volatility * volatility * a / (2.0 * steps * steps)) * dt)
    vang = volatility * np.sqrt(a) / (pow(steps, 1.5))
    
    price = BlackScholes(V, strike, rate, vang, 0, expiry)

    return price

# For Call option Only.

def BlackScholes(spot, strike, rate, volatility, dividend, expiry):
    N = norm.cdf
    d1 = (np.log(spot/strike) + (rate - dividend - 0.5 * volatility * volatility) * expiry) / (volatility * np.sqrt(expiry))
    d2 = d1 - volatility * np.sqrt(expiry)
    price = spot * np.exp(-dividend * expiry) * N(d1) -  strike * np.exp(-rate * expiry) * N(d2)
    
    return price

def StratifiedMonteCarloPricer(pricing_engine, option, data):
    """
    Doc string
    """

    expiry = option.expiry
    strike = option.strike
    (spot, rate, volatility, dividend) = data.get_data()
    reps = pricing_engine.reps
    steps = pricing_engine.steps
    disc = np.exp(-rate * expiry)
    dt = expiry / steps
    b = -1.0
    
    z = np.random.normal(size=(reps, steps))
    drift = (rate - dividend - 0.5 * volatility * volatility) * dt
    diffusion = volatility * np.sqrt(dt) 
    disc = np.exp(-rate * expiry)

    Gstar = GeometricAsianCall(spot, strike, rate, volatility, dividend, expiry, steps)
    spotT = np.zeros((reps, steps))
    spotT[:,0] = spot
    A = np.zeros(reps)
    G = np.zeros(reps)
    W = np.zeros(reps)

    for i in range(reps):
        for j in range(1, steps):
            spotT[i,j] = spotT[i,j-1] * np.exp(drift + diffusion * z[i,j])
        
        A_mean = spotT[i].mean()
        G_mean = pow(spotT[i].prod(), 1 / steps)
        A[i] = option.payoff(A_mean)
        G[i] = option.payoff(G_mean)
        W[i] = A[i] + (b * (Gstar - G[i]))
    
    
    W = W.std()
    W = W/np.sqrt(reps)
    price = disc * A.mean() + b * (Gstar - G.mean())
    
    print("Standard Error: {0:0.3f}".format(W))
    return price
