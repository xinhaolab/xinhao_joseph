import pandas as pd

def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns
        returns a dataframe with columns for 
        the wealth index,
        the previous peaks, and 
        the percentage drawdown
    """
    wealth_index= 1000*(1+return_series).cumprod()
    previous_peaks=wealth_index.cummax()
    drawdowns=(wealth_index - previous_peaks)/ previous_peaks
    return pd.DataFrame({"Wealth": wealth_index,
                        "Previous Peak": previous_peaks,
                        "Drawdown": drawdowns})

def get_ffme_returns():
    """
    load the fama_french dataset for the returns of the top and bottom deciles by marketcap
    """
    me_m=pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                    header=0, index_col=0, na_values=-99.99)
    rets=me_m[['Lo 10','Hi 10']]
    rets.columns=['SmallCap', 'LargeCap']
    rets=rets/100
    rets.index=pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets


def get_hfi_returns():
    hfi=pd.read_csv("data/edhec-hedgefundindices.csv",
                   header=0, index_col=0, parse_dates=True)
    hfi=hfi/100
    hfi.index=hfi.index.to_period('M')
    return hfi

def get_ind_returns():
    ind= pd.read_csv("data/ind30_m_vw_rets.csv",header=0, index_col=0, parse_dates=True)/100
    ind.index=pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns= ind.columns.str.strip()
    return ind


def get_ind_size():
    ind= pd.read_csv("data/ind30_m_size.csv",header=0, index_col=0)
    ind.index=pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns= ind.columns.str.strip()
    return ind


def get_ind_nfirms():
    ind= pd.read_csv("data/ind30_m_nfirms.csv",header=0, index_col=0)
    ind.index=pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns= ind.columns.str.strip()
    return ind


def get_total_market_index_returns():
    ind_mktcap=get_ind_nfirms()* get_ind_size()
    total_mktcap=ind_mktcap.sum(axis="columns")
    ind_capweight= ind_mktcap.divide(total_mktcap, axis="rows")
    total_market_return = (ind_capweight * get_ind_returns()).sum(axis="columns")
    return total_market_return
    
    
    


def skewness(r):
    """
    alternative to scipy.stats.skew()
    computes the skewness of the supplied series or dataframe
    returns a float or a series
    """
    demeaned_r=r-r.mean()
    #use the population stdev, so set dof=0
    sigma_r= r.std(ddof=0)
    exp=(demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    alternative to scipy.stats.kurtosis()
    computes the kurtosis of the supplied series or dataframe
    returns a float or a series
    """
    demeaned_r=r-r.mean()
    #use the population stdev, so set dof=0
    sigma_r= r.std(ddof=0)
    exp=(demeaned_r**4).mean()
    return exp/sigma_r**4

def semideviation(r):
    """r must be a series or dataframe
    """
    is_negative= r<0 # create a boolean mask first
    return r[is_negative].std(ddof=0)


import scipy.stats

def is_normal(r, level=0.01):
    """
     applies the jarque_bera test to determine if a series is normal or not, test is applied at 1% level by default. returns true if the hypothesis of normality is accepted, false otherwise.
    """
    statist, p_value= scipy.stats.jarque_bera(r)
    return p_value > level

import numpy as np

def var_historic(r, level=5):
    """VaR Historic
    """
    if isinstance(r, pd.DataFrame):
        # if r is an instance of dataframe, then....
        return r.aggregate(var_historic, level=level)
        # call var_historic function on every col of r
    elif isinstance(r,pd.Series):
        return -np.percentile(r,level)
    else: 
        raise TypeError("Expected r to be Series or DataFrame")
        
from scipy.stats import norm


def var_gaussian(r, level=5, modified=False):
    """returns the parametric gaussian VaR of a series or a dataframe"""
    # compute the Z score assuming it was Gaussian
    z= norm.ppf(level/100)
    if modified: # modify the z score based on observed skewness and kurtosis
        s= skewness(r)
        k=kurtosis(r)
        z=(z+
              (z**2 -1)*s/6 +
              (z**3 -3*z)*(k-3)/24 -
              (2*z**3 - 5*z)*(s**2)/36
          )
    return -(r.mean() + z* r.std(ddof=0))


def cvar_historic(r, level=5):
    """computes the conditional VaR of series or dataframe
    """
    if isinstance(r,pd.Series):
        is_beyond= r<= -var_historic(r,level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a series or dataframe")

def annualize_rets(r, periods_per_year):
    compounded_growth= (1+r).prod()
    n_periods=r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1
        
def annualize_vol(r, periods_per_year):
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate,periods_per_year):
    # convert the annual risk free rate to per period
    rf_per_period= (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret=r-rf_per_period
    ann_ex_ret= annualize_rets(excess_ret, periods_per_year)
    ann_vol= annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def portfolio_return(weights, returns):
    """
    weigths > returns
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    weights > vol
    """
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(n_points, er, cov,style=".-"):
    """
    plot the 2 asset efficient frontier
    """
    if er.shape[0]!= 2 or cov.shape[0] != 2:
        raise ValueError("Plot_ef2 can only plot 2-asset frontier")
    weights=[np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    rets=[portfolio_return(w,er) for w in weights]
    vols=[portfolio_vol(w,cov) for w in weights]
    ef=pd.DataFrame({
            "R":rets,"Vol":vols
        })
    return ef.plot.line(x="Vol",y="R",style=style)

       
from scipy.optimize import minimize

def minimize_vol(target_return, er, cov):
    """
    target_return >> weight vector
    """
    n =er.shape[0]
    init_guess= np.repeat(1/n,n)
    bounds= ((0.0, 1.0),)*n
    return_is_target= {
        'type':'eq', 
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights,er) 
    }
    weights_sum_to_1={
        'type':'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    results=minimize(portfolio_vol, init_guess,
                    args=(cov,), method="SLSQP", 
                    options={'disp':False},
                    constraints=(return_is_target, weights_sum_to_1),
                    bounds=bounds
                    )
    return results.x

def optimal_weights(n_points, er, cov):
    """
    > list of weights to run the optimizer on to minimize the vol
    """
    target_rs= np.linspace(er.min(), er.max(), n_points)
    weights= [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights



def msr(riskfree_rate, er, cov):
    """
    riskfree_Rate +ER + COV -> W
    """
    n =er.shape[0]
    init_guess= np.repeat(1/n,n)
    bounds= ((0.0, 1.0),)*n
    weights_sum_to_1={
        'type':'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        returns the negative of the sharpe ratio, given weights
        """
        r = portfolio_return(weights, er)
        vol=portfolio_vol(weights,cov)
        return -(r-riskfree_rate)/vol
    
    results=minimize(neg_sharpe_ratio, init_guess,
                    args=(riskfree_rate,er,cov), method="SLSQP", 
                    options={'disp':False},
                    constraints=(weights_sum_to_1),
                    bounds=bounds
                    )
    return results.x

def gmv(cov):
    """return the weights of the global minimum variance portfolio given the cov matrix
    """
    n= cov.shape[0]
    return msr(0,np.repeat(1,n), cov)
    
    
def plot_ef(n_points, er, cov,style=".-", show_cml=False,riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    plot the N asset efficient frontier
    """
    weights=optimal_weights(n_points, er,cov)
    rets=[portfolio_return(w,er) for w in weights]
    vols=[portfolio_vol(w,cov) for w in weights]
    ef=pd.DataFrame({
            "R":rets,"Vol":vols
        })
    ax= ef.plot.line(x="Vol",y="R",style=style)
    if show_ew:
        n=er.shape[0]
        w_ew=np.repeat(1/n,n)
        r_ew=portfolio_return(w_ew, er)
        vol_ew=portfolio_vol(w_ew,cov)
        # display Equally weighted portfolio
        ax.plot([vol_ew],[r_ew],color="red",marker="o",markersize=10)
 
    if show_gmv:
        w_gmv=gmv(cov)
        r_gmv=portfolio_return(w_gmv, er)
        vol_gmv=portfolio_vol(w_gmv,cov)
        # display GMV
        ax.plot([vol_gmv],[r_gmv],color="midnightblue",marker="o",markersize=10)
    
    if show_cml:
        ax.set_xlim(left=0)
        w_msr=msr(riskfree_rate,er,cov)
        r_msr=portfolio_return(w_msr,er)
        vol_msr=portfolio_vol(w_msr,cov)
        # add CML
        cml_x=[0, vol_msr]
        cml_y=[riskfree_rate, r_msr]
        ax.plot(cml_x,cml_y,color='gold',marker='o',linestyle='dashed', markersize=12, linewidth=2)
    return ax


def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    run a backtest of the CPPI strategy, given a set of returns for the risky asset 
    returns a dictionary containing: asset value history, risk budget history, risky weight history
    """
    #set up the CPPI parameters.
    dates = risky_r.index
    n_steps = len(dates)
    account_value =start
    floor_value = floor * start
    peak= start
    
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:]= riskfree_rate/12
    account_history= pd.DataFrame().reindex_like(risky_r)
    cushion_history= pd.DataFrame().reindex_like(risky_r)
    risky_w_history= pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_steps):
        if drawdown is not None:
            peak=np.maximum(peak,account_value)
            floor_value= peak*(1-drawdown)
        cushion = (account_value - floor_value) / account_value
        risky_w = m * cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w= 1 - risky_w
        risky_alloc= account_value * risky_w
        safe_alloc= account_value * safe_w
        ## update the account value for this time step.
        account_value= risky_alloc*(1+risky_r.iloc[step])+ safe_alloc*(1+safe_r.iloc[step])
        ## save the values so i can look at the history and plot it
        cushion_history.iloc[step]=cushion
        risky_w_history.iloc[step]=risky_w
        account_history.iloc[step]=account_value
        
        
    risky_wealth = start*(1+risky_r).cumprod()
    #pack everything into a nice dictionary
    backtest_result={
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m":m,
        "start":start,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r":safe_r }
    return backtest_result


def summary_stats(r, riskfree_rate=0.03):
    """
    return a dataframe that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r=r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol=r.aggregate(annualize_vol,periods_per_year=12)
    ann_sr= r.aggregate(sharpe_ratio,riskfree_rate=riskfree_rate,periods_per_year=12)
    dd=r.aggregate(lambda r:drawdown(r).Drawdown.min())
    skew=r.aggregate(skewness)
    kurt=r.aggregate(kurtosis)
    cf_var5= r.aggregate(var_gaussian, modified=True)
    hist_cvar5= r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Returns":ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Corner_Fisher Var (5%)": cf_var5,
        "Historic CVaR (5%)" : hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
        })

def discount(t,r):
    """
    compute the price of a pure discount bond that pays a dollar at time period t and r is the per_period interest rate
    returns a |t| * |r| series or dataframe 
    r can be a float, series or dataframe
    returns a dataframe indexed by t
    """
    discounts= pd.DataFrame([(r+1)**-i for i in t])
    discounts.index=t
    return discounts

def pv(flows,r):
    """compute the pv of a sequence of liabilities
    l is indexed by the time, and the values are the amounts of each liabilities
    it returns the pv """
    dates = flows.index
    discounts= discount(dates, r)
    return discounts.multiply(flows, axis='rows').sum()

def funding_ratio(assets, liabilities, r):
    return pv(assets,r)/pv(liabilities, r)
    
    
def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    returns a series of cash flows generated by a bond,
    indexed by a coupon number. maturity is in years
    """
    n_coupons =round(maturity* coupons_per_year)
    coupon_amt= principal*coupon_rate/coupons_per_year
    coupon_times= np.arange(1,n_coupons+1)
    cash_flows= pd.Series(data=coupon_amt, index= coupon_times)
    cash_flows.iloc[-1]+= principal
    return cash_flows

def bond_cash_flows_online(principal=100, maturity=10, coupon_rate=0.03, coupons_per_year=12):
    '''
    Generates a pd.Series of cash flows of a regular bond. Note that:
    '''
    # total number of coupons 
    n_coupons = round(maturity * coupons_per_year)
    
    # coupon amount 
    coupon_amount = (coupon_rate / coupons_per_year) * principal 
    
    # Cash flows
    cash_flows = pd.DataFrame(coupon_amount, index = np.arange(1,n_coupons+1), columns=[0])
    cash_flows.iloc[-1] = cash_flows.iloc[-1] + principal 
        
    return cash_flows



def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, 
                discount_rate=0.03):
    """
    compute the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    this is not designed to be efficient, rather.
    it is to illustrate the underlying principal behind bond pricing!
    if discount_rate is a dataframe, then this is asssumed to be the rate on each coupon date and the bond value is computed over time
    i.e. the index of the discount_rate dataframe is assumed to be the coupon number 
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates= discount_rate.index
        prices=pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t]=bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year, discount_rate.loc[t])
        return prices
    else: #base case... single time period
        if maturity<=0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows= bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)

def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    computes the total return of a bond based on monthly bond prices and coupon payments
    assume that dividends (coupons) are paid out at the end of the period(e.g. end of 3 months for quarterly div)
    and that dividends are reinvested in the bond
    """
    coupons = pd.DataFrame(data=0, index= monthly_prices.index, columns= monthly_prices.columns)
    t_max= monthly_prices.index.max()
    pay_date= np.linspace(12/coupons_per_year, t_max, int(coupons_per_year* t_max/12), dtype=int)
    coupons.iloc[pay_date]=principal* coupon_rate/coupons_per_year
    total_returns= (monthly_prices+ coupons)/monthly_prices.shift()-1
    return total_returns.dropna()
    

def macaulay_duration(cash_flows, discount_rate):
    '''
    Computed the Macaulay duration of an asset involving regular cash flows a given discount rate
    Note that if the cash_flows dates are normalized, then the discount_rate is simply the YTM. 
    Otherwise, it has to be the YTM divided by the coupons per years.
    '''
    if not isinstance(cash_flows,pd.DataFrame):
        raise ValueError("Expected a pd.DataFrame of cash_flows")

    dates = cash_flows.index

    # present value of single cash flows (discounted cash flows)
    discount_cf = discount( dates, discount_rate ) * cash_flows
    
    # weights: the present value of the entire payment, i.e., discount_cf.sum() is equal to the principal 
    weights = discount_cf / discount_cf.sum()
    
    # sum of weights * dates
    return ( weights * pd.DataFrame(dates,index=weights.index) ).sum()[0]





def match_duration(cf_target, cf_shortbond, cf_longbond, 
                   discount_rate):
    """
    returns the weight W in cf_shortbond that, along with 1-W in cf_1 will 
    have a effective duration that matches cf_target
    """
    d_t=macaulay_duration(cf_target,discount_rate)
    d_s=macaulay_duration(cf_shortbond, discount_rate)
    d_l=macaulay_duration(cf_longbond, discount_rate)
    return (d_l - d_t)/(d_l - d_s)


import math
def cir(n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    generate random interest rate evolution over time using the CIR model 
    b and r_0 are assumed to be the annualized rates, not the short rate 
    and the returned values are the annualized rates as well.
    """
    if r_0 is None: r_0 = b
    r_0=ann_to_inst(r_0)
    dt= 1/steps_per_year
    
    num_steps=int(n_years*steps_per_year)+1 # '+1' is b/c row 0 is initial rates.
    shock= np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates=np.empty_like(shock)
    rates[0]=r_0
    
    ## for price generation
    h=math.sqrt(a**2 + 2*sigma**2)
    prices= np.empty_like(shock)
    ###
    
    def price(ttm,r):
        _A= ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1))) ** (2*a*b/sigma**2)
        _B= (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P= _A * np.exp(-_B*r)
        return _P
    prices[0]= price(n_years, r_0)
    ####
    
    for step in range(1,num_steps):
        r_t =rates[step-1]
        d_r_t=a*(b-r_t)*dt+sigma* np.sqrt(r_t) * shock[step]
        rates[step]= abs(r_t + d_r_t)
        #generate prices at time t as well
        prices[step]=price(n_years-step*dt, rates[step])
    
    rates= pd.DataFrame(data=inst_to_ann(rates),index=range(num_steps))
    ### for prices
    prices= pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates,prices



def inst_to_ann(r):
    """
    converts short rate to annualized rate
    """
    return np.exp(r)-1

def ann_to_inst(r):
    """
    convert annualized to short rate
    """
    return np.log(1+r)

def show_cir_prices(r_0=0.03, a=0.5, b=0.03, sigma=0.05,n_scenarios=5):
    cir(r_0=r_0, a=a,b=b, sigma=sigma, n_scenarios=n_scenarios)[1].plot(legend=False, figsize=(12,5))
    

def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0):
    """
    evolution of a stock price using a geometric brownian motion model
    """
    dt=1/steps_per_year
    n_steps=int(n_years*steps_per_year)
    xi=np.random.normal( size=(n_steps+1, n_scenarios))
    xi[0]=0
    rets= mu*dt+ sigma*np.sqrt(dt)*xi
    rets=pd.DataFrame(rets)
    ## to prices
    prices= s_0 * (1+rets).cumprod()
    return prices


def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    produces a time series over T steps of allocations b/w PSP and GHP across N scenarios
    PSP and GHP are T x N dataframes that represents the returns of PSP and GHP such that:
    each column is a scenario 
    each row is the price for a timestep
    returns a Tx N dataframe of PSP weights.
    """
    return pd.DataFrame(data=w1, index=r1.index, columns=r1.columns)


def bt_mix(r1, r2, allocator, **kwargs):
    """
    run a back test (simulation) of allocating b/w a two set of returns 
    r1 and r2 are TxN dataframes or returns where T is the time step index and N is the number of scenarios.
    allocator is a function that takes two sets of retursn and allocator specific parameters, and produces 
    an allocation to the first portfolio (the rest of the money in the LHP/GHP) as a Tx1 dataframe, 
    returns a TxN dataframe of the resulting N portfolio scenarios.
    """
    if not r1.shape== r2.shape:
        raise ValueError("r1 and r2 need to be the same shape")
    weights = allocator(r1,r2,**kwargs)
    if not weights.shape==r1.shape:
        raise ValueError("allocator returned weights that don't match r1")
    r_mix=weights*r1+(1-weights)*r2
    return r_mix


def terminal_values(rets):
    """
    returns the final values at the end of the return period for each scenario
    """
    return (rets+1).prod()


def terminal_stats(rets, floor=0.8, cap=np.inf, name="Stats"):
    """
    produce a summary statistics on the terminal values per invested dollar across a range of N scenarios
    rets is T by N dataframe of returns
    returns a 1 column dataframe of summary stats indexed by the stat name
    """
    terminal_wealth=(rets+1).prod()
    breach=terminal_wealth< floor
    reach= terminal_wealth >= cap
    p_breach= breach.mean() if breach.sum()>0 else np.nan
    p_reach= reach.mean() if reach.sum()>0 else np.nan
    e_short= (floor- terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus=(cap- terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats= pd.DataFrame.from_dict({
        "mean":terminal_wealth.mean(),
        "std" :terminal_wealth.std(),
        "p-breach": p_breach,
        "e-shortfall": e_short,
        "p-reach": p_reach,
        "e-surplus":e_surplus
    }, orient="index", columns=[name])
    return sum_stats

def glidepath_allocator(r1,r2, start_glide=1, end_glide=0):
    """
    simulate a target_date_fund style gradual move from r1 to r2
    """
    n_points=r1.shape[0]
    n_col=r1.shape[1]
    path= pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    paths = pd.concat([path]*n_col,axis=1)
    paths.index=r1.index
    paths.columns=r1.columns
    return paths

def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    allocate b/w PSP and CHP with the goal to provide exposure to the upside 
    of the PSP without going violating the floor
    Uses a CPPI_style dynamic risk budgeting algorithm by investing a multiple 
    of the cushion in the PSP
    Returns a DataFRAME WITH THE same shape as the PSP/GHP representing the weights in the PSP
    """
    if zc_prices.shape!= psp_r.shape:
        raise ValueError("PSP and ZC prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value= np.repeat(1,n_scenarios)
    floor_value= np.repeat(1, n_scenarios)
    w_history= pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value= floor*zc_prices.iloc[step] ## PV of floor assuming today's rates and flat YC
        cushion= (account_value- floor_value)/account_value
        psp_w= (m*cushion).clip(0,1) #same as applying min and max
        ghp_w= 1-psp_w
        psp_alloc= account_value*psp_w
        ghp_alloc= account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value= psp_alloc*(1+psp_r.iloc[step])+ ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step]=psp_w
    return w_history


def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    allocate b/w PSP and CHP with the goal to provide exposure to the upside 
    of the PSP without going violating the floor
    Uses a CPPI_style dynamic risk budgeting algorithm by investing a multiple 
    of the cushion in the PSP
    Returns a DataFRAME WITH THE same shape as the PSP/GHP representing the weights in the PSP
    """
    n_steps, n_scenarios = psp_r.shape
    account_value= np.repeat(1,n_scenarios)
    floor_value= np.repeat(1, n_scenarios)
    peak_value=np.repeat(1, n_scenarios)
    w_history= pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value= (1-maxdd)*peak_value  ## Floor is based on Peak Value
        cushion= (account_value- floor_value)/account_value
        psp_w= (m*cushion).clip(0,1) #same as applying min and max
        ghp_w= 1-psp_w
        psp_alloc= account_value*psp_w
        ghp_alloc= account_value*ghp_w
        # recompute the new account value and prev peak at the end of this step
        account_value= psp_alloc*(1+psp_r.iloc[step])+ ghp_alloc*(1+ghp_r.iloc[step])
        peak_value=np.maximum(peak_value, account_value)
        w_history.iloc[step]=psp_w
    return w_history