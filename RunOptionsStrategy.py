# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 23:45:23 2021

"""
import QuantPy as qp
import math
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly
#import scipy.stats as stats

#have data from 5/19/06 to 3/4/21
start_date = datetime.date(2006,8, 3)
days_to_run = 5300
cashbal = 0
#start_date = datetime.date(2015,7,23)
#days_to_run = 100

#Conditions for new position
#everyday?
open_ruleset = "current_date.weekday() < 5"
#open_ruleset = "x == 0"
min_dte = 30
max_dte = 65

#Position details
short_delta = -0.2;
long_delta = -0.1;

#When to Close?
stop_loss = -1
take_profit = 0.5
close_by_dte = 7

""" Model """
#Fit coeffs for ssvi model
beta1 = 177.6828
beta2 = .049787
gamma1 = .238
gamma2 = .253

##Import stock market data
dataFile = open("AllData.csv")
data_start_date = datetime.date(2006, 5, 19)
rawdata = np.loadtxt(dataFile, delimiter=",", skiprows=1)
deldaysarr = rawdata[:, 0]

# For each day in study
DateHistory = pd.date_range(start_date,start_date + datetime.timedelta(days=days_to_run-1))
acct_history = np.zeros((days_to_run))
cash_history = np.zeros((days_to_run))
inst_history = np.zeros((days_to_run))
positions = []
for x in range(days_to_run):
    #Get current date
    current_date = datetime.timedelta(days=x) + start_date
    datedelta = current_date - data_start_date
    current_row = np.where(deldaysarr == qp.get_next_high_val(datedelta.days, deldaysarr))
    
    # Pull data from current Date
    stock = rawdata[current_row, 1]
    Sadj = rawdata[current_row, 2]
    VIX = rawdata[current_row, 3]
    rate = rawdata[current_row, 4]
    eta = rawdata[current_row, 5]
    rho = rawdata[current_row, 6]
    VXST = rawdata[current_row, 7]    
    #print("\n",current_date)
    #print("SPY =%7.2f" % (stock))
    
    #Figure out if we should open a new position today
    if (eval(open_ruleset)):
#    if (VIX < 25):
#    if (current_date.weekday() == 3 and VXST > 25):
#    if (x == 0):
        #If opening new position,
        #Find expiration date within target range
        expiration = qp.get_third_friday(current_date, min_dte, max_dte)
        
        dte = expiration-current_date
        t = dte.days/365 #time to expiration (years)   

        #find strike at short and long deltas
        atm_vol = qp.est_atm_vol(VIX, VXST, t)
        Kshort = qp.strike_from_delta2(stock, short_delta, rate, 0, atm_vol, t, "Put", beta1,
                                     beta2, gamma1, gamma2, eta, rho, 0.2)
        Klong = qp.strike_from_delta2(stock, long_delta, rate, 0, atm_vol, t, "Put", beta1,
                                     beta2, gamma1, gamma2, eta, rho, 0.2)
        if Klong == Kshort:
            Klong = Kshort - 1 #Catch identical deltas
            
        #Add contracts to position
        contracts=[]
        contracts.append(qp.Contract("SPY", Kshort, expiration, "Put", -1))
        contracts.append(qp.Contract("SPY", Klong, expiration, "Put", 1))
        positions.append(qp.Position(current_date, contracts))
        
    #Every day, update prices for each options contract
    pval=0
    #For each position
    for position in positions:
        #If the position is still open
        closeflag=0
        if position.is_open:
            #Calculate value of each contract in the position
            for c in position.contracts:
                c.dte = c.expiration - current_date
                t = max(c.dte.days/365, 5.708e-5) #time to expiration (years)   
        
                k = math.log(c.strike/(stock*math.exp(rate*t)))
                theta = qp.est_atm_vol(VIX, VXST, t)**2*t
                c.IV = qp.ssvi(beta1, beta2, gamma1, gamma2, eta, rho, k, t, theta) #Implied Volatility
        
                qp.bs_model(c, stock, rate, 0, c.IV, t)
                
                #kinda messed up cause closes position if 1 leg expires
                if ((t <= max(5.708e-5, close_by_dte/365)) and (current_date.weekday() < 5)): #if time limit reached
                    closeflag = 1
                    #print("time")
                #print(c.expiration)
                #print("%3.1f %s price=%7.3f" % (c.strike, c.optType, c.price))
            position.update_value(current_date)
            #print(position.value)
            
            #Update Account values
            if (current_date == position.open_date):
#                position.init_value = position.value
#                bpe = 0
#                for c in position.contracts:
#                    bpe = bpe + c.strike*c.qty*100                    
#                if bpe>0: bpe=0
#                position.bpr = -(bpe - position.init_value*100)
                cashbal = cashbal - position.value*100 #TODO: adjust cash by BPR

                
                #print("initial acct value=",pval)            
            
            #add position value to value of all other positions
            pval = pval + position.value*100
            
            #if stop loss or takeprofit is triggered
            if ((position.init_value*(1-stop_loss) >= position.value) and
                (current_date.weekday() < 5)):
                closeflag = 1
                #print("l")
            elif ((position.value >= (1-take_profit)*position.init_value) and 
                (current_date.weekday() < 5)):
                closeflag = 1
                #print("w")
        
        #if this position has been marked for closure,
        if closeflag:
            cashbal = cashbal + position.value*100 #reverse the position and adjust cash balance
            pval = pval - position.value*100
            position.close_pos(current_date)
            
    #store history
    inst_history[x] = stock
    cash_history[x] = cashbal
    acct_history[x] = cashbal + pval #acct value is cash balance + value of all positions
    #print("Position Value =%3.3f delta=%6.3f theta=%6.3f vega=%6.2f" % (positions[0].value, positions[0].delta, positions[0].theta , positions[0].vega))


"""After Simulation has run, collect statistics"""
#initialize    
closed_positions = [position for position in positions if position.is_open==0]
total_days_held = np.sum([position.days_held for position in closed_positions])
n_closed_positions = len(closed_positions)
#win_pct = np.sum([position.win for position in closed_positions])/len(closed_positions)

total_pl = 0
pl_array = np.zeros(n_closed_positions)
roi_array = np.zeros(n_closed_positions)
days_held_array = np.zeros(n_closed_positions)
win_array = np.zeros(n_closed_positions)
cagr_array = np.zeros(n_closed_positions)
ccgr_array = np.zeros(n_closed_positions)
gsd_array = np.zeros(n_closed_positions)
bpr_array = np.zeros(n_closed_positions)
dyt_array = np.zeros(total_days_held)
#For each closed position 
i=0
for position in closed_positions: #if position is closed
    #collect arrays of statistics
    total_pl = total_pl + position.pl #Count total P/L
    pl_array[i] = position.pl
    dyt_array[int(np.sum(days_held_array)):
        int(np.sum(days_held_array))+position.days_held] = position.dyt_history
    days_held_array[i] = position.days_held
    win_array[i] = position.win
    roi_array[i] = position.pl_pct
    #cagr_array[i] = position.cagr_pct
    #ccgr_array[i] = position.ccgr
    #gsd_array[i] = position.gsd_pct
    bpr_array[i] = position.bpr
    
    #increase index
    i=i+1

# Compute stats from collected arrays        
win_pct = np.sum(win_array)/n_closed_positions
avg_pl = np.mean(pl_array)
stdPL = np.std(pl_array)
avg_roi = np.mean(roi_array)
std_roi = np.std(roi_array)

avg_days_held = np.mean(days_held_array)
std_days_held = np.std(days_held_array)
annualized_pl = 365*avg_pl/avg_days_held
dAnnualizedPL = annualized_pl*math.sqrt((stdPL/avg_pl)**2 + (std_days_held/avg_days_held)**2)
annualized_roi = 365*avg_roi/avg_days_held
avgWin = np.mean(pl_array[pl_array>=0])
std_win = np.std(pl_array[pl_array>=0])
avgLoss = np.mean(pl_array[pl_array<0])
std_loss = np.std(pl_array[pl_array<0])
biggestW = max(pl_array)
biggestL = min(pl_array)
dAnnualizedROI = annualized_roi*math.sqrt((std_roi/avg_roi)**2 + (std_days_held/avg_days_held)**2)
#cagr = math.sqrt(max(0,(1+annualized_roi)**2-dAnnualizedROI**2))-1

overall_roi = total_pl/np.sum(bpr_array)
overall_dgr = (1+overall_roi)**(1/avg_days_held)
overall_cagr = overall_dgr**365-1

cagr = qp.xirr(bpr_array,days_held_array/365,total_pl+np.sum(bpr_array))
r_estimate = math.log(max(1+cagr, 1e-8))
nu = total_days_held-1 # degrees of freedom
vol = np.std(dyt_array)*(365**0.5)
max_ccgr = r_estimate + 1.96*vol/(nu**0.5) #95% confidence interval on cagr
max_cagr = math.exp(max_ccgr)-1
min_ccgr = r_estimate - 1.96*vol/(nu**0.5)
min_cagr = math.exp(min_ccgr)-1

timeslice_start = (closed_positions[0].open_date - start_date).days
timeslice_delta = (closed_positions[-1].close_date - closed_positions[0].open_date).days
spy_init = inst_history[timeslice_start]
spy_final = inst_history[timeslice_start+timeslice_delta]
spy_ror = inst_history[timeslice_start+timeslice_delta]/inst_history[timeslice_start]
spy_dgr = (spy_ror)**(1/timeslice_delta)
spy_cagr = spy_dgr**365-1
sharpe = (cagr-spy_cagr)/vol #uses spy for risk free interest rate

"""------Plotting------"""
fig = plotly.subplots.make_subplots(rows=2,cols=1,vertical_spacing=0.08,
                                    specs=[[{"secondary_y": True}],
                                           [{"type": "table"}]],)

trace1=go.Scatter(x=DateHistory,y=acct_history,name="Account Value",yaxis='y1')
trace2=go.Scatter(x=DateHistory,y=cash_history,name="Cash available",yaxis='y1')
trace3=go.Scatter(x=DateHistory,y=inst_history,name="SPY",yaxis='y2')
table4=go.Table(header=dict(values=['Parameter', 'Value','Uncertainty'],
                            font=dict(color='black', size=12),
                            line_color='darkslategray',
                            ),
    cells=dict(values=[['Win Percentage', 'Avg P/L', 'Avg ROI', 
                        'Avg Days Held', 'Annualized P/L', 'Annualized ROI',
                        'Avg Win', 'Avg Loss', 'Biggest Winner', 
                        'Biggest Loser', 'CAGR', 'CAGR (method 2)',
                        'Volatility of Returns', 'SPY CAGR', 
                        'Sharpe Ratio (to SPY)'],
    [win_pct, avg_pl, avg_roi, avg_days_held, annualized_pl, annualized_roi,
     avgWin, avgLoss, biggestW, biggestL, cagr, overall_cagr, vol, spy_cagr, 
     sharpe],
     ['', '±${:.0f}'.format(stdPL), '±{:.0%}'.format(std_roi),
      '±{:.0f}'.format(std_days_held), '±${:.0f}'.format(dAnnualizedPL),
      '±{:.0%}'.format(dAnnualizedROI), '±${:.0f}'.format(std_win),
      '±${:.0f}'.format(std_loss), '', '', 
      '{:.0%} - {:.0%}'.format(min_cagr,max_cagr),'','','']],
     fill=dict(color=['#C8D4E3','white','white']),
                      font=dict(color='black', size=12),
                      line_color='darkslategray',
                      format=[[],
                              ['.1%', '$.2f', '.1%', '.1f', '$.0f',
                               '.1%', '$.0f', '$.0f', '$.0f', '$.0f', '.1%',
                               '.1%', '.0%', '.1%', '.2f'],
                               []]
                      ),
     columnwidth = [50,50]
 )

data = [trace3, trace1, trace2,table4]
layout = dict(showlegend=True,
        legend=dict(x=1.1),
        height=1000,
        width=800,
        title="Strategy Performance",
        hovermode="x unified",
        plot_bgcolor='rgba(1,1,1,0)',
        yaxis=dict(zeroline = True, 
           hoverformat = '$.2f', 
           title = 'Balance',
           tickformat = "$",
           gridcolor='LightGray',
           zerolinecolor = 'Black',
           ),
        yaxis2=dict(hoverformat = '.2f', 
           title = 'SPY',
           tickformat = "",
           overlaying="y",
           side = "right",
           showgrid = False,
           ),
        xaxis=dict(gridcolor='LightGray'),
        )
        
fig.add_trace(trace1,row=1,col=1)
fig.add_trace(trace2,row=1,col=1)
fig.add_trace(trace3,row=1,col=1,secondary_y=True)
fig.add_trace(table4,row=2,col=1)
fig.update_layout(layout)
#fig = dict(data = data, layout = layout)

plotly.offline.plot(fig)

