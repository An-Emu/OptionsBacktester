# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 23:45:23 2021

"""
import QuantPy as qp
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

#have data from 5/19/06 to 3/4/21
StartDate = datetime.date(2007, 1, 14)
cashBal = 0
daystoRun = 5000

#Conditions for new position
#everyday?
minDTE = 30
maxDTE = 65

#Position details
shortDelta = -0.3;
longDelta = -0.15;

#When to Close?
stopLoss = -2
takeProfit = 0.75
closeByDTE = 7

#Fit coeffs for SSVI model
beta1 = 177.6828
beta2 = .049787
gamma1 = .238
gamma2 = .253

##Import stock market data
dataFile = open("AllData.csv")
dataStartDate = datetime.date(2006, 5, 19)
rawdata = np.loadtxt(dataFile, delimiter=",", skiprows=1)
deldaysarr = rawdata[:, 0]

# For each day in study
DateHistory = pd.date_range(StartDate,StartDate + datetime.timedelta(days=daystoRun-1))
AcctHistory = np.zeros((daystoRun))
pList = []
for x in range(daystoRun):
    #Get current date
    currDate = datetime.timedelta(days=x) + StartDate
    datedelta = currDate - dataStartDate
    currRow = np.where(deldaysarr == qp.GetNextHighVal(datedelta.days, deldaysarr))
    
    # Pull data from current Date
    S = rawdata[currRow, 1]
    Sadj = rawdata[currRow, 2]
    VIX = rawdata[currRow, 3]
    rate = rawdata[currRow, 4]
    eta = rawdata[currRow, 5]
    rho = rawdata[currRow, 6]
    VXST = rawdata[currRow, 7]    
    #print("\n",currDate)
    #print("SPY =%7.2f" % (S))
    
    #Figure out if we should open a new position today
    if (currDate.weekday() == 3):
        #If opening new position,
        #Find expiration date within target range
        expiration = qp.get_third_friday(currDate, minDTE, maxDTE)
        dte = expiration-currDate
        t = dte.days/365 #time to expiration (years)   

        #find strike at short and long deltas
        ATMVol = qp.estATMVol(VIX, VXST, t)
        Kshort = qp.strikeFromDelta2(S, shortDelta, rate, 0, ATMVol, t, "Put", beta1,
                                     beta2, gamma1, gamma2, eta, rho, 0.2)
        Klong = qp.strikeFromDelta2(S, longDelta, rate, 0, ATMVol, t, "Put", beta1,
                                     beta2, gamma1, gamma2, eta, rho, 0.2)
        #Add contracts to position
        cList=[]
        cList.append(qp.Contract("SPY", Kshort, expiration, "Put", -1))
        cList.append(qp.Contract("SPY", Klong, expiration, "Put", 1))
        pList.append(qp.Position(currDate, cList))
        
    #Every day, update prices for each options contract
    pval=0
    #For each position
    for pos in pList:
        #If the position is still open
        closeflag=0
        if pos.isOpen:
            #Calculate value of each contract in the position
            for c in pos.contractList:
                c.dte = c.expiration - currDate
                t = max(c.dte.days/365, 5.708e-5) #time to expiration (years)   
        
                k = math.log(c.strike/(S*math.exp(rate*t)))
                theta = qp.estATMVol(VIX, VXST, t)**2*t
                c.IV = qp.SSVI(beta1, beta2, gamma1, gamma2, eta, rho, k, t, theta) #Implied Volatility
        
                qp.BSmodel(c, S, rate, 0, c.IV, t)
                
                #kinda messed up cause closes position if 1 leg expires
                if t <= closeByDTE/365: #if time limit reached
                    closeflag = 1
                    #print("time")
                
                #print("%3.1f %s price=%7.3f" % (c.strike, c.optType, c.price))
            pos.updateValue()
            
            #Update Account values
            if (currDate == pos.openDate):
                pos.initValue = pos.value
                cashBal = cashBal - pos.value*100 # open position and adjust cash balance
                #TODO: update buying power
                #print("initial acct value=",pval)            
            
            #add position value to value of all other positions
            pval = pval + pos.value*100
            
            #if stop loss or takeprofit is triggered
            if (pos.initValue*(1-stopLoss) >= pos.value):
                closeflag = 1
                #print("l")
            elif (pos.value >= (1-takeProfit)*pos.initValue):
                closeflag = 1
                #print("w")
        
        #if this position has been marked for closure,
        if closeflag:
            cashBal = cashBal + pos.value*100 #reverse the position and adjust cash balance
            pos.closePos(currDate)
            
    #store history
    AcctHistory[x] = cashBal + pval #acct value is cash balance + value of all positions
    #print("Position Value =%3.3f delta=%6.3f theta=%6.3f vega=%6.2f" % (pList[0].value, pList[0].delta, pList[0].theta , pList[0].vega))

#Plotting    
plt.clf()
plt.cla()
plt.close()    
plt.plot(DateHistory, AcctHistory)
plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 90-degrees
plt.show
