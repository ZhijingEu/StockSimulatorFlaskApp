import pandas as pd
from pandas_datareader import data

import numpy as np, numpy.random
from numpy import mean

import random

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from datetime import datetime, timedelta

from scipy.stats import norm 
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import stats

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

import pmdarima
import arch

import time
from scipy.stats import zscore
import os

#Based loosely on https://towardsdatascience.com/how-to-simulate-financial-portfolios-with-python-d0dc4b52a278

def extract_prices(start_date,end_date,symbols,portfolioWeights,portfolioValue,backtestduration=0):
    dim=len(symbols)
    for symbol in symbols:
        dfprices = data.DataReader(symbols, start=start_date, end=end_date, data_source='yahoo')
        dfprices = dfprices[['Adj Close']]
    dfprices.columns=[' '.join(col).strip() for col in dfprices.columns.values]

    priceAtEndDate=[]
    for symbol in symbols:
        priceAtEndDate.append(dfprices[[f'Adj Close {symbol}']][-(backtestduration+1):].values[0][0])
        
    noOfShares=[]
    portfolioValPerSymbol=[x * portfolioValue for x in portfolioWeights]
    for i in range(0,len(symbols)):
        noOfShares.append(portfolioValPerSymbol[i]/priceAtEndDate[i])
    noOfShares=[round(element, 5) for element in noOfShares]
    listOfColumns=dfprices.columns.tolist()   
    dfprices["Adj Close Portfolio"]=dfprices[listOfColumns].mul(noOfShares).sum(1)
    
    share_split_table=dfprices.tail(1).T
    share_split_table=share_split_table.iloc[:-1]
    share_split_table["Share"]=symbols
    share_split_table["No Of Shares"]=noOfShares
    share_split_table.columns=["Price At "+end_date,"Share Name","No Of Shares"]
    share_split_table["Value At "+end_date]=share_split_table["No Of Shares"]*share_split_table["Price At "+end_date]
    share_split_table.index=share_split_table["Share Name"]
    share_split_table=share_split_table[["Share Name","Price At "+end_date,"No Of Shares","Value At "+end_date]]
    share_split_table=share_split_table.round(3)
    share_split_table=share_split_table.append(share_split_table.sum(numeric_only=True), ignore_index=True)
    share_split_table.at[len(symbols),'No Of Shares']=np.nan
    share_split_table.at[len(symbols),'Price At '+end_date]=np.nan
    share_split_table.at[len(symbols),'Share Name']="Portfolio"
    share_split_table["Weights"]=portfolioWeights+["1"]
    share_split_table = share_split_table[['Share Name', 'Weights', 'Price At '+end_date, 'No Of Shares', "Value At "+end_date]] 
    
    print(f"Extracted {len(dfprices)} days worth of data for {len(symbols)} counters with {dfprices.isnull().sum().sum()} missing data")
    
    return dfprices, noOfShares, share_split_table

def plotprices(dfprices,symbols,imagecounter,targetfolder):
    dfprices.plot(subplots=True, figsize=(15,7.5*len(symbols)))
    plt.savefig(f'static/{targetfolder}/{imagecounter}_02adjclosingprices.png')
    
def plotpiechart(symbols,portfolioWeights,imagecounter,targetfolder):    
    labels = symbols
    sizes = portfolioWeights
    fig1, ax1 = plt.subplots()
    ax1.pie(portfolioWeights, labels=symbols, autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Portfolio Weights")
    plt.savefig(f'static/{targetfolder}/{imagecounter}_01portfolioweights.png')
    
#Modified from https://medium.com/swlh/generating-candlestick-charts-from-scratch-ef6e1d3cf0e9

# Function to draw candlestick
def draw_candlestick(axis, data, color_up, color_down):
    
    # Check if stock closed higher or not
    if data['Close'] > data['Open']:
        color = color_up
    else:
        color = color_down

    # Plot the candle wick
    axis.plot([data['day_num'], data['day_num']], [data['Low'], data['High']], linewidth=2, color='black', solid_capstyle='round', zorder=2)
    
    # Draw the candle body
    rect = mpl.patches.Rectangle((data['day_num'] - 0.5, data['Open']), 1.0, (data['Close'] - data['Open']), facecolor=color, edgecolor='black', linewidth=1, zorder=3)

    # Add candle body to the axis
    axis.add_patch(rect)
    
    # Return modified axis
    return axis

# Function to draw all candlesticks
def draw_all_candlesticks(axis, data, color_up='white', color_down='black'):
    for day in range(data.shape[0]):
        axis = draw_candlestick(axis, data.iloc[day], color_up, color_down)
    return axis

def plot_candlesticks(symbols,start_date,end_date,imagecounter,targetfolder):
    for i in range(0,len(symbols)):
        tkr_str = str(symbols[i])
        tkr_history = data.DataReader(tkr_str, start=start_date, end=end_date, data_source='yahoo')
        tkr_history['Date']=tkr_history.index
        base_date = tkr_history['Date'][0]
        tkr_history['day_num'] = tkr_history['Date'].map(lambda date:(date - base_date).days)

        # Create figure and axes
        fig = plt.figure(figsize=(20, 10), facecolor='white')
        ax = fig.add_subplot(111)

        # Colors for candlesticks
        colors = ['#00FF00', '#FF0000']

        # Grid lines
        ax.grid(linestyle='-', linewidth=4, color='white', zorder=1)

        # Draw candlesticks
        ax = draw_all_candlesticks(ax, tkr_history, colors[0], colors[1])

        # Set ticks to every 5th day
        ax.set_xticks(list(tkr_history['day_num'])[::15])
        ax.set_xticklabels(list(tkr_history['Date'].dt.strftime('%Y-%m-%d'))[::15])
        ax.tick_params(labelsize=14)
        plt.xticks(rotation=50)

#         # Add dollar signs
#         formatter = mpl.ticker.FormatStrFormatter('$%.2f')
#         ax.yaxis.set_major_formatter(formatter)

        # Append ticker symbol
        ax.text(0, 1.05, tkr_str, va='baseline', ha='left', size=20, transform=ax.transAxes)

        # Set axis limits
        ax.set_xlim(-1, tkr_history['day_num'].iloc[-1] + 1)

        # Show plot
        plt.savefig(f'static/{targetfolder}/{imagecounter}_candlestick{i}.png')
        #plt.show()
    
 #Modified from https://tcoil.info/compute-bollinger-bands-for-stocks-with-python-and-pandas/

    # n = smoothing length eg 20
    # m = number of standard deviations away from MA eg 2
    
def bollinger_bands(start_date,end_date,symbol, n, m,i):

    df=data.DataReader(symbol, start=start_date, end=end_date, data_source='yahoo')
    
    #typical price
    TP = (df['High'] + df['Low'] + df['Close']) / 3
    # but we will use Adj close instead for now, depends
    
    datax = TP
    #data = df['Adj Close']
    
    # takes one column from dataframe
    B_MA = pd.Series((datax.rolling(n, min_periods=n).mean()), name='B_MA')
    sigma = datax.rolling(n, min_periods=n).std() 
    
    BU = pd.Series((B_MA + m * sigma), name='BU')
    BL = pd.Series((B_MA - m * sigma), name='BL')
    
    df = df.join(B_MA)
    df = df.join(BU)
    df = df.join(BL)  
    
    return df

def plot_bollingerbands(symbols,start_date,end_date,n,m,imagecounter,targetfolder):
    for i in range(0,len(symbols)):
        df=bollinger_bands(start_date,end_date,str(symbols[i]), n, m,i)
        # plot correspondingRSI values and significant levels
        plt.figure(figsize=(15,5))
        plt.title(symbols[i]+': Bollinger Bands For Smoothing Length Of '+str(n)+' Days & '+str(m)+' Std Devs From MA')
        plt.plot(df.index, df['Adj Close'])
        plt.plot(df.index, df['BU'], alpha=0.3)
        plt.plot(df.index, df['BL'], alpha=0.3)
        plt.plot(df.index, df['B_MA'], alpha=0.3)
        plt.fill_between(df.index, df['BU'], df['BL'], color='grey', alpha=0.1)
        plt.savefig(f'static/{targetfolder}/{imagecounter}_bollingerband{i}.png')
        #plt.show()   
    
#Modified from https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas

def calcRSI(start_date,end_date,symbol,time_window,RSItype="EWMA"):

    df=data.DataReader(symbol, start=start_date, end=end_date, data_source='yahoo')
    
    diff = df["Adj Close"].diff(1).dropna()        # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    if RSItype=="EWMA":
        up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window, adjust=False).mean()
        down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window, adjust=False).mean()
    
    elif RSItype=="SMA":
    # Calculate the SMA
        up_chg_avg = up_chg.rolling(time_window).mean()
        down_chg_avg = down_chg.abs().rolling(time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    df["RSI"]=rsi
    return df    
    
def plot_RSI(symbols,start_date,end_date,time_window,RSItype,imagecounter,targetfolder):
    for i in range(0,len(symbols)):
        dfrsi=calcRSI(start_date,end_date,str(symbols[i]),time_window,RSItype)
        plt.figure(figsize=(15,5))
        dfrsi['Adj Close'][time_window::].plot()
        plt.title(str(symbols[i])+": Adj Close Price")
        plt.savefig(f'static/{targetfolder}/{imagecounter}_{i}adjclosingprice.png')
        #plt.show
        
        plt.figure(figsize=(15,5))
        dfrsi["RSI"].plot()
        plt.axhline(0, linestyle='--', alpha=0.1)
        plt.axhline(20, linestyle='--', alpha=0.5)
        plt.axhline(30, linestyle='--')
        plt.axhline(70, linestyle='--')
        plt.axhline(80, linestyle='--', alpha=0.5)
        plt.axhline(100, linestyle='--', alpha=0.1)
        plt.title(str(symbols[i])+": RSI Plot: "+RSItype+" With Period Of "+str(time_window)+" Days")
        plt.savefig(f'static/{targetfolder}/{imagecounter}_{i}relativestrengthindex.png')
        #plt.show()

def calcMACD(start_date,end_date,symbol,timeperiod1,timeperiod2,timeperiod3,imagecounter,targetfolder):
    df=data.DataReader(symbol, start=start_date, end=end_date, data_source='yahoo')
    
    df["EWMA "+str(timeperiod1)+" Days"]=df["Adj Close"].ewm(span=timeperiod1,min_periods=timeperiod1,adjust=False,ignore_na=False).mean()
    df["EWMA "+str(timeperiod2)+" Days"]=df["Adj Close"].ewm(span=timeperiod2,min_periods=timeperiod2,adjust=False,ignore_na=False).mean()
    df["MACD"]=df["EWMA "+str(timeperiod1)+" Days"]-df["EWMA "+str(timeperiod2)+" Days"]
    df["Signal Line"]=df["MACD"].ewm(span=timeperiod3,min_periods=timeperiod3,adjust=False,ignore_na=False).mean()
    
    
    df[['Adj Close',"EWMA "+str(timeperiod1)+" Days","EWMA "+str(timeperiod2)+" Days"]].plot(figsize=(15,5))
    plt.title(symbol+": MACD ("+str(timeperiod1)+","+str(timeperiod2)+","+str(timeperiod3)+") - EWMAs & Adj Close Price")
    plt.savefig(f'static/{targetfolder}/{imagecounter}_{symbol}_1_MACD.png')
    
    df[["MACD","Signal Line"]].plot(figsize=(15,5))
    plt.title(symbol+": MACD ("+str(timeperiod1)+","+str(timeperiod2)+","+str(timeperiod3)+") - MACD & Signal Line")
    plt.savefig(f'static/{targetfolder}/{imagecounter}_{symbol}_2_MACD.png')
    
    df["Histogram"]=df["MACD"]-df["Signal Line"]
    df.plot.bar(y='Histogram',figsize=(15,5))
    tick_spacing = 28
    plt.gca().xaxis.set_major_locator(plt.AutoLocator())
    plt.title(symbol+": MACD ("+str(timeperiod1)+","+str(timeperiod2)+","+str(timeperiod3)+") - MACD Histogram")
    
    #plt.savefig(f'static/{targetfolder}/{imagecounter}_{symbol}_3_MACD.png')
    
def plot_MACD(start_date,end_date,symbols,timeperiod1,timeperiod2,timeperiod3,imagecounter,targetfolder):
    for symbol in symbols:
        calcMACD(start_date,end_date,symbol,timeperiod1,timeperiod2,timeperiod3,imagecounter,targetfolder)

def calc_returns(dfprices,symbols):
    dfreturns=pd.DataFrame()
    columns = list(dfprices) 
    mean=[]
    stdev=[]
    for column in columns:
        dfreturns[f'Log Daily Returns {column}']=np.log(dfprices[column]).diff()
        mean.append(dfreturns[f'Log Daily Returns {column}'][1:].mean())
        stdev.append(dfreturns[f'Log Daily Returns {column}'][1:].std())
    dfreturns=dfreturns.dropna()
    
    if len(dfreturns.columns)==1:
        df_mean_stdev=pd.DataFrame(list(zip(symbols,mean,stdev)),columns =['Stock', 'Mean Log Daily Return','StdDev Log Daily Return']) 
    else:
        df_mean_stdev=pd.DataFrame(list(zip(symbols+["Portfolio"],mean,stdev)),columns =['Stock', 'Mean Log Daily Return','StdDev Log Daily Return'])
    
    return dfreturns ,df_mean_stdev    
    
def convertReturnsToPrices(dfreturns,startingrefprice):
    stockprices=pd.DataFrame()
    stockprices=np.exp(dfreturns)
    stockprices=stockprices.cumprod()
    stockprices=stockprices.mul(startingrefprice.values)
    stockprices.columns=stockprices.columns.str.lstrip('Log Daily Returns ')
    firstrow=pd.DataFrame(startingrefprice)
    stockprices=pd.concat([firstrow,stockprices])
    return stockprices    
    
def plotreturns(dfreturns,imagecounter,targetfolder):
    dfreturns.plot(subplots=True, figsize=(15,7.5*len(dfreturns.columns)))
    plt.savefig(f'static/{targetfolder}/{imagecounter}_03Dailyreturns.png')    
    
   #Correlations which are within the blue bands are not statistically significant

def plotACFPACF(series,imagecounter,targetfolder):
    
    for i in range(0,len(series.columns)):
        
        plt.figure(figsize=(15,5))
        Prices=series[series.columns[i]].plot()
        plt.title(series.columns[i])
        plt.savefig(f'static/{targetfolder}/{imagecounter}_{i}_Price.png')
        
        fig1, ax1 = plt.subplots(figsize=(15, 5))
        ACFplot=plot_acf(series[series.columns[i]], lags=30, ax=ax1)
        plt.title(series.columns[i]+' ACF')
        plt.savefig(f'static/{targetfolder}/{imagecounter}_{i}_PriceACF.png')
        
        fig2, ax2 = plt.subplots(figsize=(15, 5))
        PACFplot=plot_pacf(series[series.columns[i]], lags=30,ax=ax2)
        plt.title(series.columns[i]+' PACF')
        plt.savefig(f'static/{targetfolder}/{imagecounter}_{i}_PricePACF.png') 
    
def compareStartMidEnd(dfreturns,df_mean_stdev):

    #print ('Size of dataFrame=', len(dfreturns.index))
    desired_number_of_groups = 3
    group_size = int(len(dfreturns.index) / (desired_number_of_groups))
    #print("group_size=", group_size)
    remainder_size = len(dfreturns.index) % group_size
    #print("remainder_size=", remainder_size)
    df_split_list = [dfreturns.iloc[i:i + group_size] for i in range(0, len(dfreturns) - group_size + 1, group_size)]
    #print("Number of split_dataframes=", len(df_split_list))
    if remainder_size > 0:
        df_remainder = dfreturns.iloc[-remainder_size:len(dfreturns.index)]
        df_split_list.append(df_remainder)
    #print("Revised Number of split_dataframes=", len(df_split_list))
    #print("Splitting complete, verifying counts")

    count_all_rows_after_split = 0
    for index, split_df in enumerate(df_split_list):
        #print("split_df:", index, " size=", len(split_df.index))
        count_all_rows_after_split += len(split_df.index)

    if count_all_rows_after_split != len(dfreturns.index):
        raise Exception('rows_after_split = ', count_all_rows_after_split," but original CSV DataFrame has count =", len(dfreturns.index))

    columns =['Stock','Start Mean', 'Start StdDev','Middle Mean','Middle StdDev','End Mean','End StdDev']
    
    anothertable=[]
    for i in range(0,len(dfreturns.columns)):
        boxplotsplit=pd.DataFrame()
        boxplotsplit["Start"]=df_split_list[0].iloc[:,i].values
        boxplotsplit["Middle"]=df_split_list[1].iloc[:,i].values
        boxplotsplit["End"]=df_split_list[2].iloc[:,i].values
        #means = [boxplotsplit["Start"].mean(),boxplotsplit["Middle"].mean(),boxplotsplit["End"].mean()]
        #std =  [boxplotsplit["Start"].std(),boxplotsplit["Middle"].std(),boxplotsplit["End"].std()]
        meanstdev=[dfreturns.columns[i],boxplotsplit["Start"].mean(),boxplotsplit["Start"].std(),\
                   boxplotsplit["Middle"].mean(),boxplotsplit["Middle"].std(),\
                   boxplotsplit["End"].mean(),boxplotsplit["End"].std()]
        
        anothertable.append(meanstdev) 
        
    yetanothertable=pd.DataFrame(anothertable,columns=columns)
    
    yetanothertable["Overall Mean"]=df_mean_stdev["Mean Log Daily Return"]
    yetanothertable["Overall StdDev"]=df_mean_stdev["StdDev Log Daily Return"]
    yetanothertable=yetanothertable[['Stock','Overall Mean', 'Start Mean','Middle Mean','End Mean',\
                                     'Overall StdDev','Start StdDev','Middle StdDev','End StdDev']]
    
    return yetanothertable    
 
def fit_test_normal(dfreturns,symbols,imagecounter,targetfolder):
    
    columnlist=dfreturns.columns
    KSTestResults=[]
    KSPValResults=[]
    for column in columnlist:
        data=(dfreturns[column])
        normed_data=(data-dfreturns[column].mean())/dfreturns[column].std()
        KSTestResults.append(kstest(normed_data, 'norm')[0])
        KSPValResults.append(kstest(normed_data, 'norm')[1])
    if len(dfreturns.columns)==1:
        KSTestResultsDF=pd.DataFrame([dfreturns.columns,KSTestResults,KSPValResults]).T
    else:
        KSTestResultsDF=pd.DataFrame([dfreturns.columns,KSTestResults,KSPValResults]).T
    KSTestResultsDF.columns=["Share Name","KS Test Statistic","KS Test P-Value"]
    KSTestResultsDF["Accept/Reject At 5% Signif Lvl"]=np.where(KSTestResultsDF['KS Test P-Value']> 0.05, "Data looks normal (Fail to reject H0)", "Data does NOT look normal (Reject H0)")
    
    ShapiroWilkTestStats=[]
    ShapiroWilkTestPValue=[]
    for column in columnlist:
        data=(dfreturns[column])
        stat, pvalue = shapiro(data)
        ShapiroWilkTestStats.append(stat)
        ShapiroWilkTestPValue.append(round(pvalue,6))

    ShapiroWilkTestResultsDF=pd.DataFrame([ShapiroWilkTestStats,ShapiroWilkTestPValue]).T
    if len(dfreturns.columns)==1:
        ShapiroWilkTestResultsDF["Share Name"]=dfreturns.columns
    else:
        ShapiroWilkTestResultsDF["Share Name"]=dfreturns.columns
    ShapiroWilkTestResultsDF.columns=["SW Test Statistic","SW Test P-Value","Share Name"]
    ShapiroWilkTestResultsDF=ShapiroWilkTestResultsDF[["Share Name","SW Test Statistic","SW Test P-Value"]]
    ShapiroWilkTestResultsDF["Accept/Reject At 5% Signif Lvl"]=np.where(ShapiroWilkTestResultsDF['SW Test P-Value']> 0.05, "Data looks normal (Fail to reject H0)", "Data does NOT look normal (Reject H0)")

    
    ADFTestStats=[]
    ADFTestCritValue=[]
    for column in columnlist:
        data=(dfreturns[column])        
        ADFTest_result = adfuller(data)
        ADFTestStats.append(ADFTest_result[0])
        ADFTestCritValue.append(ADFTest_result[4]['5%'])

    ADFTestResultsDF=pd.DataFrame([ADFTestStats,ADFTestCritValue]).T
    ADFTestResultsDF["Share Name"]=dfreturns.columns
    ADFTestResultsDF.columns=["ADF Test Statistic","ADF Test Stat Crit Value At 5% Signif Lvl","Share Name"]
    ADFTestResultsDF=ADFTestResultsDF[["Share Name","ADF Test Statistic","ADF Test Stat Crit Value At 5% Signif Lvl"]]
    ADFTestResultsDF["Accept/Reject At 5% Signif Lvl"]=np.where(ADFTestResultsDF['ADF Test Statistic']> ADFTestResultsDF['ADF Test Stat Crit Value At 5% Signif Lvl'] , "Time series is NON-stationary (Reject H0)", "Time series is stationary (Fail To Reject H0)")  
        
    skewArray=[]
    kurtosisArray=[]
    for column in columnlist:
        data=(dfreturns[column])
        skewArray.append(skew(data))
        kurtosisArray.append(kurtosis(data))
        mu = np.mean(data)
        sigma = np.std(data)
        plt.figure(figsize = (15, 5))
        plt.hist(data, bins=50, density=True, alpha=0.6, color='g')
        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, sigma)
        plt.plot(x, p, 'k', linewidth=2)
        title = column+" Histogram vs Best Fit Normal Distribution Mu = %.3f,  Sigma = %.3f" % (mu, sigma)
        plt.title(title)
        plt.savefig(f'static/{targetfolder}/{imagecounter}_04histogram{column}.png')
    
    Kurtosis_Skew=pd.DataFrame(list(zip(skewArray, kurtosisArray)), 
               index=columnlist,columns =['Skew','Kurtosis']) 
    
#     If the skewness is between -0.5 and 0.5, the data are fairly symmetrical.
#     If the skewness is between -1 and – 0.5 or between 0.5 and 1, the data are moderately skewed.
#     If the skewness is less than -1 or greater than 1, the data are highly skewed.
    def skewconditions(s):
        if (s['Skew'] > 1) :
            return "Highly Positively Skewed"
        elif (s['Skew'] <= 1) and (s['Skew'] > 0.5) :
            return "Moderately Positively Skewed"
        elif (s['Skew'] <= 0.5) and (s['Skew'] > -0.5) :
            return "Fairly Symmetrical"
        elif (s['Skew'] <= -0.5) and (s['Skew'] > -1.0) :
            return "Moderately Negatively Skewed"
        elif (s['Skew'] <= -1.0) :
            return "Highly Negatively Skewed"
    
#     For kurtosis, the general guideline is that if the number is greater than +1, the distribution is too peaked. 
#     Likewise, a kurtosis of less than –1 indicates a distribution that is too flat
    def kurtosisconditions(s):
        if (s['Kurtosis'] > 3) :
            return "Leptokurtic:Peaked & Fat Tailed"
        elif (s['Kurtosis'] <-3):
            return "Platykurtic:Flat & Thin Tailed"
        else :
            return "Mesokurtic"
    
    Kurtosis_Skew['Skew Description'] = Kurtosis_Skew.apply(skewconditions, axis=1)
    Kurtosis_Skew['Kurtosis Description'] = Kurtosis_Skew.apply(kurtosisconditions, axis=1)
    
    for i in range(0,len(columnlist)):
        data=(dfreturns.iloc[:, i])
        res = stats.probplot(data, dist="norm")
        xxx=pd.DataFrame(list(zip(res[0][0],res[0][1])),columns =["Theoretical Quantiles","Ordered Values"]) 
        xxx.plot.scatter(x="Theoretical Quantiles",y="Ordered Values",title=columnlist[i],figsize = (7.5, 5))
        plt.plot(xxx["Theoretical Quantiles"], res[1][0]*xxx["Theoretical Quantiles"]+res[1][1])
        plt.savefig(f'static/{targetfolder}/{imagecounter}_05qqplot{i}.png')
    
    return KSTestResultsDF, ShapiroWilkTestResultsDF, Kurtosis_Skew, ADFTestResultsDF    
    
def bootstrap_w_replc_singleval(dfreturns):
    columns=dfreturns.columns
    singlesample=pd.DataFrame(dfreturns.values[np.random.randint(len(dfreturns), size=1)], columns=columns)
    return singlesample

def bootstrapforecast(dfreturns,T):
    columnlist=dfreturns.columns
    X=[]
    for i in range(0,T):
        X.append(bootstrap_w_replc_singleval(dfreturns).values.tolist()[0])
    Y=pd.DataFrame(X)
    Y.columns=columnlist
    Y.loc[-1] = [0]*len(columnlist)  # adding a row
    Y.index = Y.index + 1  # shifting index
    Y = Y.sort_index()  # sorting by index
    
    return Y

#Adapted from https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix

#import numpy as np,numpy.linalg

def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def nearPD(A, nit=10):
    n = A.shape[0]
    W = np.identity(n) 
# W is the matrix used for the norm (assumed to be Identity matrix here)
# the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk

def create_covar(dfreturns):  
    try:
        returns=[]
        arrOfReturns=[]
        columns = list(dfreturns)
        for column in columns:
            returns=dfreturns[column].values.tolist()
            arrOfReturns.append(returns)
        Cov = np.cov(np.array(arrOfReturns))    
        return Cov
    except LinAlgError :
        Cov = nearPD(np.array(arrOfReturns), nit=10)
        print("WARNING -Original Covariance Matrix is NOT Positive Semi Definite And Has Been Adjusted To Allow For Cholesky Decomposition ")
        return Cov

def GBMsimulatorMultiVar(So, mu, sigma, Cov, T, N):
    """
    Parameters

    seed:   seed of simulation
    So:     initial stocks' price
    mu:     expected return
    sigma:  volatility
    Cov:    covariance matrix
    T:      time period
    N:      number of increments
    """

    #np.random.seed(seed) turned off so Monte Carlo can be "randomised"
    dim = np.size(So)
    t = np.linspace(0., T, int(N))
    A = np.linalg.cholesky(Cov)
    S = np.zeros([dim, int(N)])
    S[:, 0] = So
    for i in range(1, int(N)):    
        drift = (mu - 0.5 * sigma**2) * (t[i] - t[i-1])
        Z = np.random.normal(0., 1., dim)
        diffusion = np.matmul(A, Z) * (np.sqrt(t[i] - t[i-1]))
        S[:, i] = S[:, i-1]*np.exp(drift + diffusion)
    return S, t

def GBMsimulatorUniVar(So, mu, sigma, T, N):
    """
    Parameters

    seed:   seed of simulation
    So:     initial stocks' price
    mu:     expected return
    sigma:  volatility
    Cov:    covariance matrix
    T:      time period
    N:      number of increments
    """

    #np.random.seed(seed) turned off so Monte Carlo can be "randomised"
    dim = np.size(So)
    t = np.linspace(0., T, int(N))
    S = np.zeros([dim, int(N)])
    S[:, 0] = So
    for i in range(1, int(N)):    
        drift = (mu - 0.5 * sigma**2) * (t[i] - t[i-1])
        Z = np.random.normal(0., 1., dim)
        diffusion = sigma* Z * (np.sqrt(t[i] - t[i-1]))
        S[:, i] = S[:, i-1]*np.exp(drift + diffusion)
    return S, t

def calculateRMSE(final,T,backtest_duration,dfprices):
    xyz=final.tail(T)
    xyz=xyz.head(backtest_duration)
    xyz
    qrs=pd.DataFrame(index=xyz.index)
    for i in range(0,len(dfprices.columns)):
        x=1+(len(dfprices.columns)-i)*-3
        qrs["Actual "+dfprices.columns[i]]=xyz.iloc[:,i]
        qrs["P50 "+dfprices.columns[i]]=xyz.iloc[:,x]
    for i in range(0,len(dfprices.columns)):
        x=i*2
        qrs["RMSE "+dfprices.columns[i]]=(qrs.iloc[:,x]-qrs.iloc[:,x+1])**2
    qrs
    RMSE=[]

    for i in range(0,len(dfprices.columns)):
        z=-(len(dfprices.columns)-i)
        RMSE.append((qrs.iloc[:,z].mean())**0.5)
    RMSE_DF=pd.DataFrame(RMSE,index=[dfprices.columns],columns=["RMSE For Backtest From "+qrs.index[0].strftime("%Y-%m-%d")+" To "+qrs.index[-1].strftime("%Y-%m-%d")\
                                                               +" ("+str(len(qrs))+" Days)"])
    return RMSE_DF

def MonteCarlo_GBM(start_date,end_date,backtest_duration,percentile_range,symbols,\
                       portfolioWeights,portfolioValue,T,N,NoOfIterationsMC,imagecounter,targetfolder):
    
    forecastresults=pd.DataFrame()
    percentiles=pd.DataFrame()
    
    extended_dates_future=[]
    lowerpercentile=int(percentile_range[1:3])
    upperpercentile=int(percentile_range[5:7])
 
    plotpiechart(symbols,portfolioWeights,imagecounter,targetfolder)

    if len(symbols)==1:
        dfpricesFULL, noOfSharesFULL, share_split_tableFULL = extract_prices(start_date,end_date,symbols,portfolioWeights,portfolioValue)
        backtest_end_date=dfpricesFULL.index[-(backtest_duration+1)].strftime("%Y-%m-%d")
        dfprices, noOfShares, share_split_table = extract_prices(start_date,backtest_end_date,symbols,portfolioWeights,portfolioValue)
        dfprices["Adj Close Portfolio"]=dfprices[list(dfprices.iloc[:,:-1].columns)].mul(noOfSharesFULL).sum(1)
        
    else:
        dfpricesFULL, noOfSharesFULL, share_split_tableFULL = extract_prices(start_date,end_date,symbols,portfolioWeights,portfolioValue)
        backtest_end_date=dfpricesFULL.index[-(backtest_duration+1)].strftime("%Y-%m-%d")
        dfprices, noOfShares, share_split_table = extract_prices(start_date,backtest_end_date,symbols,portfolioWeights,portfolioValue)
        dfprices["Adj Close Portfolio"]=dfprices[list(dfprices.iloc[:,:-1].columns)].mul(noOfSharesFULL).sum(1)
                  
    symbolsWPortfolio=symbols+["Portfolio"]

    dfreturns ,df_mean_stdev = calc_returns(dfprices,symbolsWPortfolio)
    
    S0=np.array(dfprices.tail(1).values.tolist()[0])
    mu=np.array(df_mean_stdev["Mean Log Daily Return"].values.tolist())
    sigma=np.array(df_mean_stdev["StdDev Log Daily Return"].values.tolist())   

    backtestdateslist=(list((dfpricesFULL.tail(backtest_duration+1).index)))
    backtestdates=[]
    for i in backtestdateslist:
        backtestdates.append(np.datetime64(datetime.strptime(str(i), '%Y-%m-%d %H:%M:%S').strftime("%Y-%m-%d")))
    
    for i in range(0,N-backtest_duration):
        extended_dates_future.append(np.busday_offset(end_date, i, roll='forward'))
        
    extended_dates=backtestdates[0:len(backtestdates)-1]+extended_dates_future
     
    if len(symbols)==1:
        for x in range(1,NoOfIterationsMC+1):
            stocks, time = GBMsimulatorUniVar(S0, mu, sigma, T, N)
            prediction=pd.DataFrame(stocks)
            prediction=prediction.T
            prediction.index=extended_dates
            prediction.columns=dfprices.columns
            prediction=prediction.add_prefix('Iter_'+str(x)+'_')
            forecastresults=pd.concat([forecastresults, prediction], axis=1)
        
        for x in range(1,NoOfIterationsMC+1):
            forecastresults["Iter_"+str(x)+"_Adj Close Portfolio"]=forecastresults["Iter_"+str(x)+"_Adj Close "+symbols[0]]*noOfSharesFULL

    else:
        Cov=create_covar(dfreturns)
        for x in range(1,NoOfIterationsMC+1):
            stocks, time = GBMsimulatorMultiVar(S0, mu, sigma, Cov, T, N)
            prediction=pd.DataFrame(stocks)
            prediction=prediction.T
            prediction.index=extended_dates
            prediction.columns=dfprices.columns
            prediction=prediction.add_prefix('Iter_'+str(x)+'_')
            forecastresults=pd.concat([forecastresults, prediction], axis=1)

    for y in range(0,len(symbolsWPortfolio)):
        percentiles["P"+str(lowerpercentile)+"_"+symbolsWPortfolio[y]]=forecastresults.filter(regex=symbolsWPortfolio[y]).quantile(float(lowerpercentile)/100,1)
        percentiles["P50_"+symbolsWPortfolio[y]]=forecastresults.filter(regex=symbolsWPortfolio[y]).quantile(0.5,1)
        percentiles["P"+str(upperpercentile)+"_"+symbolsWPortfolio[y]]=forecastresults.filter(regex=symbolsWPortfolio[y]).quantile(float(upperpercentile)/100,1)

        forecastresults=pd.concat([forecastresults,percentiles[["P"+str(lowerpercentile)+"_"+symbolsWPortfolio[y],"P50_"+symbolsWPortfolio[y],"P"+str(upperpercentile)+"_"+symbolsWPortfolio[y]]]], axis=1, sort=False)
    
    final=pd.concat([dfpricesFULL,forecastresults], axis=1, sort=False)
              
    for z in range(0,len(symbolsWPortfolio)):
        final.filter(regex="Adj Close "+symbolsWPortfolio[z]).tail(60).plot(legend=False,figsize = (20, 5),title=symbolsWPortfolio[z]+": Monte Carlo Simulations For "+str(NoOfIterationsMC)+" Iter-s")
        plt.axvline(x=end_date,linestyle='dashed')
        plt.savefig(f'static/{targetfolder}/{imagecounter}_totaliterations{z}.png')
        
        percentileplot=pd.DataFrame()
        percentileplot=pd.concat([final["Adj Close "+symbolsWPortfolio[z]],final.filter(regex="P??_"+symbolsWPortfolio[z])], axis=1, sort=False)
        percentileplot.tail(60).plot(legend=True,figsize = (20, 5),title=symbolsWPortfolio[z]+": Monte Carlo Simulations For "+percentile_range+" Range")
        plt.axvline(x=end_date,linestyle='dashed')
        if NoOfIterationsMC>0:
            plt.savefig(f'static/{targetfolder}/{imagecounter}_percentile{z}.png')
  
    ReturnsAtForecastEndDate=final.tail(1).iloc[:,-(len(symbolsWPortfolio))*3:].T
    HelperTable=pd.concat([dfpricesFULL.tail(1).round(3).T]*(3)) 
    HelperTable["Sym"]=HelperTable.index
    HelperTable['Sym'] =pd.Categorical(HelperTable["Sym"], list(dfprices.columns))
    HelperTable=HelperTable.sort_values(['Sym'])
    ReturnsAtForecastEndDate.insert(0, end_date, HelperTable.iloc[:,:-1].values)    
           
    ReturnsAtForecastEndDate["Returns Based On GBM"]=round((ReturnsAtForecastEndDate.iloc[:, 1]/ReturnsAtForecastEndDate.iloc[:, 0]-1)*100,2)
    
    return final, share_split_tableFULL , dfreturns , df_mean_stdev , ReturnsAtForecastEndDate, dfprices

def MonteCarlo_Bootstrap(start_date,end_date,backtest_duration,percentile_range,symbols,\
                       portfolioWeights,portfolioValue,T,N,NoOfIterationsMC,imagecounter,targetfolder):
    
    forecastresults=pd.DataFrame()
    percentiles=pd.DataFrame()
    
    extended_dates_future=[]
    lowerpercentile=int(percentile_range[1:3])
    upperpercentile=int(percentile_range[5:7])
    
    plotpiechart(symbols,portfolioWeights,imagecounter,targetfolder)

    if len(symbols)==1:
        dfpricesFULL, noOfSharesFULL, share_split_tableFULL = extract_prices(start_date,end_date,symbols,portfolioWeights,portfolioValue)
        backtest_end_date=dfpricesFULL.index[-(backtest_duration+1)].strftime("%Y-%m-%d")
        dfprices, noOfShares, share_split_table = extract_prices(start_date,backtest_end_date,symbols,portfolioWeights,portfolioValue)
        dfprices["Adj Close Portfolio"]=dfprices[list(dfprices.iloc[:,:-1].columns)].mul(noOfSharesFULL).sum(1)
        
    else:
        dfpricesFULL, noOfSharesFULL, share_split_tableFULL = extract_prices(start_date,end_date,symbols,portfolioWeights,portfolioValue)
        backtest_end_date=dfpricesFULL.index[-(backtest_duration+1)].strftime("%Y-%m-%d")
        dfprices, noOfShares, share_split_table = extract_prices(start_date,backtest_end_date,symbols,portfolioWeights,portfolioValue)
        dfprices["Adj Close Portfolio"]=dfprices[list(dfprices.iloc[:,:-1].columns)].mul(noOfSharesFULL).sum(1)
               
    symbolsWPortfolio=symbols+["Portfolio"]

    dfreturns ,df_mean_stdev = calc_returns(dfprices,symbolsWPortfolio)

    backtestdateslist=(list((dfpricesFULL.tail(backtest_duration+1).index)))
    backtestdates=[]
    for i in backtestdateslist:
        backtestdates.append(np.datetime64(datetime.strptime(str(i), '%Y-%m-%d %H:%M:%S').strftime("%Y-%m-%d")))
        
    for i in range(0,N-backtest_duration):
        extended_dates_future.append(np.busday_offset(end_date, i, roll='forward'))
        
    extended_dates=backtestdates[0:len(backtestdates)-1]+extended_dates_future

    for x in range(1,NoOfIterationsMC+1):      
        
        futurereturns=bootstrapforecast(dfreturns,T)
        futurereturns=np.exp(futurereturns)
        futurereturns=futurereturns.cumprod()
        stocks=pd.DataFrame()
        for i in range(0,len(symbolsWPortfolio)):
            futurereturns[str(i)+"Price"]=(futurereturns.iloc[:, i])*dfprices.tail(1).iloc[:, i][0]
        stocks=futurereturns[futurereturns.columns[-len(symbolsWPortfolio):]] 
        stocks.columns=list(dfreturns.columns)

        prediction=stocks
        prediction.index=extended_dates
        prediction.columns=dfprices.columns
        prediction=prediction.add_prefix('Iter_'+str(x)+'_')
        forecastresults=pd.concat([forecastresults,prediction], axis=1, sort=False)
    
    for y in range(0,len(symbolsWPortfolio)):
        percentiles["P"+str(lowerpercentile)+"_"+symbolsWPortfolio[y]]=forecastresults.filter(regex=symbolsWPortfolio[y]).quantile(float(lowerpercentile)/100,1)
        percentiles["P50_"+symbolsWPortfolio[y]]=forecastresults.filter(regex=symbolsWPortfolio[y]).quantile(0.5,1)
        percentiles["P"+str(upperpercentile)+"_"+symbolsWPortfolio[y]]=forecastresults.filter(regex=symbolsWPortfolio[y]).quantile(float(upperpercentile)/100,1)

        forecastresults=pd.concat([forecastresults,percentiles[["P"+str(lowerpercentile)+"_"+symbolsWPortfolio[y],"P50_"+symbolsWPortfolio[y],"P"+str(upperpercentile)+"_"+symbolsWPortfolio[y]]]], axis=1, sort=False)
    
    final=pd.concat([dfpricesFULL,forecastresults], axis=1, sort=False)
    
    for z in range(0,len(symbolsWPortfolio)):
        final.filter(regex="Adj Close "+symbolsWPortfolio[z]).tail(60).plot(legend=False,figsize = (20, 5),title=symbolsWPortfolio[z]+": Monte Carlo Simulations For "+str(NoOfIterationsMC)+" Iter-s")
        plt.axvline(x=end_date,linestyle='dashed')
        plt.savefig(f'static/{targetfolder}/{imagecounter}_totaliterations{z}.png')
        
        percentileplot=pd.DataFrame()
        percentileplot=pd.concat([final["Adj Close "+symbolsWPortfolio[z]],final.filter(regex="P??_"+symbolsWPortfolio[z])], axis=1, sort=False)
        percentileplot.tail(60).plot(legend=True,figsize = (20, 5),title=symbolsWPortfolio[z]+": Monte Carlo Simulations For "+percentile_range+" Range")
        plt.axvline(x=end_date,linestyle='dashed')
        if NoOfIterationsMC>1:
            plt.savefig(f'static/{targetfolder}/{imagecounter}_percentile{z}.png')
        
    if len(symbols)==1:
        ReturnsAtForecastEndDate=final.tail(1).iloc[:,-(len(symbolsWPortfolio))*3:].T
        HelperTable=pd.concat([dfpricesFULL.tail(1).round(3).T]*(3))
        HelperTable["Sym"]=HelperTable.index
        HelperTable['Sym'] =pd.Categorical(HelperTable["Sym"], list(dfprices.columns))
        HelperTable=HelperTable.sort_values(['Sym'])
        ReturnsAtForecastEndDate.insert(0, end_date, HelperTable.iloc[:,:-1].values) 
    else:
        ReturnsAtForecastEndDate=final.tail(1).iloc[:,-(len(symbolsWPortfolio))*3:].T
        HelperTable=pd.concat([dfpricesFULL.tail(1).round(3).T]*(3)) 
        HelperTable["Sym"]=HelperTable.index
        HelperTable['Sym'] =pd.Categorical(HelperTable["Sym"], list(dfprices.columns))
        HelperTable=HelperTable.sort_values(['Sym'])
        ReturnsAtForecastEndDate.insert(0, end_date, HelperTable.iloc[:,:-1].values)    
           
    ReturnsAtForecastEndDate["Returns Based On BStrp"]=round((ReturnsAtForecastEndDate.iloc[:, 1]/ReturnsAtForecastEndDate.iloc[:, 0]-1)*100,2)
    
    return final, share_split_tableFULL , dfreturns , df_mean_stdev, ReturnsAtForecastEndDate, dfprices

def clear_cache(subfolder):
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    mydir = os.path.join(fileDir, f'static/{subfolder}')
    filelist = [ f for f in os.listdir(mydir) if f.endswith(".csv") or f.endswith(".png") ]
    for f in filelist:
        os.remove(os.path.join(mydir, f))
        
#Adapted from https://intellipaat.com/community/34075/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm-mean

def numpy_ewma_vectorized_v2(data, window):
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]
    pows = alpha_rev**(np.arange(n+1))
    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)
    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def movingaverageforecast(start_date,end_date,backtest_duration,symbols,portfolioWeights,portfolioValue,T,N,averagetype,windowsize,imagecounter,targetfolder):

    dfprices, noOfShares, share_split_table = extract_prices(start_date,end_date,symbols,portfolioWeights,portfolioValue)
    dfreturns ,df_mean_stdev=calc_returns(dfprices,symbols)

    symbolsWPortfolio=symbols+["Portfolio"]

    resultantDF=[]
    
    if backtest_duration>0:
        backtestdateslist=(list((dfprices.tail(backtest_duration).index)))
    elif backtest_duration<=0:
        backtestdateslist=(list((dfprices.tail(backtest_duration+1).index)))
    backtestdates=[]
    for i in backtestdateslist:
        backtestdates.append(np.datetime64(datetime.strptime(str(i), '%Y-%m-%d %H:%M:%S').strftime("%Y-%m-%d")))

    extended_dates_future=[]
    if backtest_duration>0:
        for i in range(0,N-backtest_duration):
            extended_dates_future.append(np.busday_offset(end_date, i, roll='forward'))
    elif backtest_duration<=0:
        for i in range(1,N-backtest_duration):
            extended_dates_future.append(np.busday_offset(end_date, i, roll='forward'))
        
    extended_dates=backtestdates[0:len(backtestdates)-1]+extended_dates_future
        
    for i in range (0,len(dfreturns.columns)):
        train=dfreturns.iloc[:,i][:len(dfreturns)-backtest_duration].values
        dfpricestrain=dfprices.iloc[:,i][:len(dfreturns)-backtest_duration+1]

        predictions = list()
        history=train.tolist()

        for j in range(0,T):
        # make prediction
            if averagetype=="SMA" :
                yhat = mean(history[-windowsize:])
            elif averagetype=="EWMA":
                yhat=numpy_ewma_vectorized_v2(np.array(history), windowsize)[-1]
            history.append(yhat)
            predictions.append(yhat)

        predictions=np.exp(predictions)
        predictions=predictions.cumprod()*dfpricestrain.tail(1).values[0]
        stocks=pd.DataFrame(predictions,index=extended_dates,columns=[f"{averagetype} Forecast"])
        QQQ=pd.DataFrame(dfpricestrain.tail(1))
        QQQ.columns=[f"{averagetype} Forecast"]
        stocks=pd.concat([QQQ,stocks])

        stocks=pd.concat([dfprices.iloc[:,i],stocks],axis=1)
        stocks.tail(60).plot(figsize=(15,5))
        plt.title(f"{dfprices.columns[i]}: Forecast Via {averagetype} of {str(windowsize)} Days")
        plt.axvline(x=end_date,linestyle='dashed')
        plt.savefig(f'static/{targetfolder}/{imagecounter}_movingaverage_{i}.png')
        resultantDF.append(stocks)

    if backtest_duration>0:
        RMSE=[]
        anothertable=[]

        for i in range(0,len(dfprices.columns)):
            temptable=resultantDF[i].tail(T).head(backtest_duration)
            temptable["RMSE"]=(temptable.iloc[:,0]-temptable.iloc[:,1])**2
            RMSE.append((temptable.iloc[:,2].mean())**0.5)

        RMSE_DF=pd.DataFrame(RMSE,index=dfprices.columns,columns=["RMSE For Backtest From "+temptable.index[0].strftime("%Y-%m-%d")+" To "+temptable.index[-1].strftime("%Y-%m-%d")\
                                                               +" ("+str(len(temptable))+" Days)"])
            
    elif backtest_duration<=0:
        RMSE_DF=pd.DataFrame()

    return resultantDF, RMSE_DF, share_split_table
    
#Adapted from https://medium.com/analytics-vidhya/arima-garch-forecasting-with-python-7a3f797de3ff
# Use this code snippet to customize the auto-arima
# arima_model =  pmdarima.auto_arima(train,start_p=0, d=1, start_q=0, 
#                           max_p=5, max_d=5, max_q=5, seasonal=False, 
#                           error_action='warn',trace = True,
#                           supress_warnings=True,stepwise = True,
#                           random_state=20,n_fits = 50 )
# Use for seasonal effects
# seasonal=True,start_P=0,D=1, start_Q=0, max_P=0, max_D=0, max_Q=0, m=0,


def UnivarArimaGarchPredict(start_date,end_date,backtest_duration,symbols,portfolioWeights,portfolioValue,T,N,imagecounter,targetfolder):
    
    dfprices, noOfShares, share_split_table = extract_prices(start_date,end_date,symbols,portfolioWeights,portfolioValue)
    dfreturns ,df_mean_stdev=calc_returns(dfprices,symbols)
    symbolsWPortfolio=symbols+["Portfolio"]
    backtest_end_date=str(np.busday_offset(end_date, -backtest_duration, roll='backward'))
    extended_dates=[]
    resultantDF=[]
    
    backtestdateslist=(list((dfprices.tail(backtest_duration).index)))
    backtestdates=[]
    for i in backtestdateslist:
        backtestdates.append(np.datetime64(datetime.strptime(str(i), '%Y-%m-%d %H:%M:%S').strftime("%Y-%m-%d")))

    extended_dates_future=[]
    for i in range(0,N-backtest_duration):
        extended_dates_future.append(np.busday_offset(end_date, i, roll='forward'))

    extended_dates=backtestdates[0:len(backtestdates)-1]+extended_dates_future
    
    for i in range (0,len(dfreturns.columns)):
    
        returns=dfreturns.iloc[:,i][:len(dfreturns)-backtest_duration]*100 #*100 is for scaling purposes
        dfpricestrain=dfprices.iloc[:,i][:len(dfprices)-backtest_duration] #Used later to extract last row prices
    
        # fit ARIMA on returns 
        arima_model = pmdarima.auto_arima(returns,trace = True)
        p, d, q = arima_model.order
        arimaaicvalue=round(arima_model.aic())
        arima_residuals = arima_model.arima_res_.resid

        # fit a GARCH(1,1) model on the residuals of the ARIMA model
        garch = arch.arch_model(arima_residuals, p=1, q=1)
        garch_fitted = garch.fit()
        garchaicvalue=round(garch_fitted.aic)
        print(garch_fitted.summary())

        # Use ARIMA to predict mu mean term 
        # Use GARCH to predict the residual error term 

        predicted_mu = pd.DataFrame(arima_model.predict(n_periods=int(T)))

        garch_forecast = garch_fitted.forecast(horizon=int(T))
        predicted_et = garch_forecast.mean.iloc[-1:]
        predicted_et=predicted_et.T

        predictions=pd.DataFrame()
        predictions["ARIMA predicted mu"]=predicted_mu.iloc[:,0].values
        predictions["GARCH 1,1, predicted et"]=list(predicted_et.iloc[:,0])
        predictions["ARIMA+GARCH"]=predictions["ARIMA predicted mu"]+predictions["GARCH 1,1, predicted et"]
        predictions=predictions/100

        futurereturns=np.exp(predictions.iloc[:,2])
        futurereturns=futurereturns.cumprod()
        stocks=pd.DataFrame()
        stocks["ARIMAGarch Forecast "+symbolsWPortfolio[i]]=futurereturns*dfpricestrain.tail(1)[0]
        
        if backtest_duration>0:
            stocks.index= extended_dates
        elif backtest_duration<=0:
            stocks.index= extended_dates[1:]
        
        QQQ=pd.DataFrame(dfpricestrain.tail(1))
        QQQ.columns=["ARIMAGarch Forecast "+symbolsWPortfolio[i]]
        stocks=pd.concat([QQQ,stocks])

        XYZ=pd.concat([dfprices.iloc[:,i],stocks], axis=1, sort=False)
        resultantDF.append(XYZ)
        XYZ.tail(60).plot(figsize=(15,5))
        plt.title(f"{symbolsWPortfolio[i]}: Forecast Via ARIMA ({str(p)},{str(q)},{str(d)})+ Constant Mean GARCH(1,1) with AIC {str(arimaaicvalue)} (ARIMA) and AIC {str(garchaicvalue)} (GARCH)")
        plt.axvline(x=end_date,linestyle='dashed')
        plt.savefig(f'static/{targetfolder}/{imagecounter}_arimagarch{i}.png')
        
    if backtest_duration>0:
        RMSE=[]
        anothertable=[]

        for i in range(0,len(dfprices.columns)):
            temptable=resultantDF[i].tail(T).head(backtest_duration)
            temptable["RMSE"]=(temptable.iloc[:,0]-temptable.iloc[:,1])**2
            RMSE.append((temptable.iloc[:,2].mean())**0.5)

        RMSE_DF=pd.DataFrame(RMSE,index=dfprices.columns,columns=["RMSE For Backtest From "+temptable.index[0].strftime("%Y-%m-%d")+" To "+temptable.index[-1].strftime("%Y-%m-%d")\
                                                               +" ("+str(len(temptable))+" Days)"])
            
    elif backtest_duration<=0:
        RMSE_DF=pd.DataFrame()
    
    return resultantDF, RMSE_DF, share_split_table

# https://medium.com/datadriveninvestor/how-to-build-exponential-smoothing-models-using-python-simple-exponential-smoothing-holt-and-da371189e1a1
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


def exp_smoothing_configs(trend=['add', 'mul', None],seasonal=['add','mul',None],seasonal_period=[None]):
	config = list()
	# define config lists
	t_params = trend
	d_params = [True, False]
	s_params = seasonal
	p_params = seasonal_period
	b_params = [True, False]
	r_params = [True, False]
	# create config instances
	for t in t_params:
		for d in d_params:
			for s in s_params:
				for p in p_params:
					for b in b_params:
						for r in r_params:
							cfg = [t,d,s,p,b,r]
							config.append(cfg)
	return config

def AutoHoltWinters(start_date,end_date,forecast_end_date,backtestduration,symbols,portfolioWeights,portfolioValue,seasonalfreq,imagecounter,targetfolder):
    
    dfprices, noOfShares, share_split_table = extract_prices(start_date,end_date,symbols,portfolioWeights,portfolioValue)

    configs=exp_smoothing_configs(trend=['add','mul'],seasonal=['add','mul',None],seasonal_period=[seasonalfreq])
    
    if backtestduration==0:
        backtest_duration=1
    elif backtestduration>0:
        backtest_duration=backtestduration

    T=np.busday_count(end_date,forecast_end_date)+backtest_duration
    N=T+1
    
    OverallRMSEforConfigTable=[]
    FinalOutput=[]
    BestConfigSummary=[]
    OptimalRMSESummaryList=[]
    
    for j in range(0,len(dfprices.columns)):
        #split between the training and the test data sets. 
        df_test=dfprices.iloc[:,j][len(dfprices)-backtest_duration:]
        df_train=dfprices.iloc[:,j][:len(dfprices)-backtest_duration]
        
        extended_dates_future=[]
        for k in range(0,N-backtest_duration):
            extended_dates_future.append(pd.Timestamp(numpy.datetime64(np.busday_offset(end_date, k, roll='forward'))))

        extended_dates=df_test.index.tolist() +extended_dates_future[1:]

        RMSEforConfig=[]
        counter=0

        for i in range(0,len(configs)):

            try:
                #build and train the model on the training data
                #TESmodel = ExponentialSmoothing(df_train, trend=t, damped_trend=d, seasonal=s, seasonal_periods=p)
                TESmodel = ExponentialSmoothing(df_train, trend=configs[i][0], damped_trend=configs[i][1], seasonal=configs[i][2], seasonal_periods=configs[i][3])

                #TESmodel_fit = TESmodel.fit(optimized=True, use_boxcox=b, remove_bias=r)
                TESmodel_fit = TESmodel.fit(optimized=True, use_boxcox=configs[i][4], remove_bias=configs[i][5])

                forecastpast = TESmodel_fit.predict(0)
                forecastfuture =  TESmodel_fit.forecast(len(df_test))
                forecast=np.concatenate((forecastpast.values,forecastfuture.values))
                forecastlist=list(forecast)

                df_forecast_TES=pd.DataFrame(forecastlist,index=dfprices.index,columns=["TES Forecast"])
                TESoutput=pd.concat([dfprices.iloc[:,j],df_forecast_TES],axis=1)

                RMSEtraincalctable=TESoutput.head(len(df_train))

                RMSEtraincalctable["Sq Diff"]=(RMSEtraincalctable[RMSEtraincalctable.columns[0]]-RMSEtraincalctable[RMSEtraincalctable.columns[1]])**2
                RMSEtrain=(RMSEtraincalctable["Sq Diff"].mean())**0.5

#                 RMSEtestcalctable=TESoutput.tail(len(extended_dates))
#                 RMSEtestcalctable=RMSEtestcalctable.head(len(df_test))

#                 RMSEtestcalctable["Sq Diff"]=(RMSEtestcalctable[RMSEtestcalctable.columns[0]]-RMSEtestcalctable[RMSEtestcalctable.columns[1]])**2
#                 RMSEtest=(RMSEtestcalctable["Sq Diff"].mean())**0.5
                
                RMSEforConfig.append([counter,configs[i][0],configs[i][1],configs[i][2],configs[i][3],configs[i][4],configs[i][5],RMSEtrain])
                counter=counter+1
            #     print(f'Test Data RMSE over {str(len(df_test))} days: {str(RMSEtest)}')
            except ValueError:
                RMSEforConfig.append([counter,configs[i][0],configs[i][1],configs[i][2],configs[i][3],configs[i][4],configs[i][5],'Error'])
                counter=counter+1

        RMSEforConfigTable=pd.DataFrame(RMSEforConfig,columns=['No','trend t','damped trend d','seasonal s','seasonal periods p','use boxcox b','remove bias r','RMSE For Train Data'])
        
        OverallRMSEforConfigTable.append(RMSEforConfigTable)
        
        BestConfig=RMSEforConfigTable[RMSEforConfigTable['RMSE For Train Data']!='Error']
        BestConfig=BestConfig.dropna()
        BestConfig=BestConfig[BestConfig['RMSE For Train Data']==BestConfig['RMSE For Train Data'].min()]
        
        OptimalTESmodel = ExponentialSmoothing(df_train, trend=BestConfig.iloc[0]['trend t'], damped_trend=BestConfig.iloc[0]['damped trend d']\
                                        , seasonal=BestConfig.iloc[0]['seasonal s'], seasonal_periods=BestConfig.iloc[0]['seasonal periods p'])

        OptimalTESmodel_fit = OptimalTESmodel.fit(optimized=True, use_boxcox=BestConfig.iloc[0]['use boxcox b'], remove_bias=BestConfig.iloc[0]['remove bias r'])

        Optimalforecastpast = OptimalTESmodel_fit.predict(0)
        Optimalforecastfuture =  OptimalTESmodel_fit.forecast(len(extended_dates))
        Optimalforecast=np.concatenate((Optimalforecastpast.values,Optimalforecastfuture.values))
        Optimalforecastlist=list(Optimalforecast)
        

        Optimaldf_forecast_TES=pd.DataFrame(Optimalforecastlist,index=df_train.index.tolist()+extended_dates,columns=["TES Forecast"])
        OptimalTESoutput=pd.concat([dfprices.iloc[:,j],Optimaldf_forecast_TES],axis=1)

        OptimalRMSEtraincalctable=OptimalTESoutput.head(len(df_train))

        OptimalRMSEtraincalctable["Sq Diff"]=(OptimalRMSEtraincalctable[OptimalRMSEtraincalctable.columns[0]]-OptimalRMSEtraincalctable[OptimalRMSEtraincalctable.columns[1]])**2
        OptimalRMSEtrain=(OptimalRMSEtraincalctable["Sq Diff"].mean())**0.5

        OptimalRMSEtestcalctable=OptimalTESoutput.tail(len(extended_dates))
        OptimalRMSEtestcalctable=OptimalRMSEtestcalctable.head(len(df_test))

        OptimalRMSEtestcalctable["Sq Diff"]=(OptimalRMSEtestcalctable[OptimalRMSEtestcalctable.columns[0]]-OptimalRMSEtestcalctable[OptimalRMSEtestcalctable.columns[1]])**2
        OptimalRMSEtest=(OptimalRMSEtestcalctable["Sq Diff"].mean())**0.5

        OptimalTESoutput.tail(60).plot(figsize=(15,5))
        titlestring=f"{dfprices.columns[j]}: Holt Winters Best Fit= trend:{str(BestConfig.iloc[0]['trend t'])}, damped trend:{str(BestConfig.iloc[0]['damped trend d'])}, seasonal s:{str(BestConfig.iloc[0]['seasonal s'])}, seasonal periods:{str(BestConfig.iloc[0]['seasonal periods p'])}, use boxcox:{str(BestConfig.iloc[0]['use boxcox b'])}, remove bias:{str(BestConfig.iloc[0]['remove bias r'])}"
        titlestring=str(titlestring)
        plt.title(titlestring)
        if backtestduration>0:
            print("RMSE-Train Data:"+str(round(OptimalRMSEtrain,2))+" RMSE-Test Data From "+OptimalRMSEtestcalctable.index[0].strftime("%Y-%m-%d")+\
                                          " To "+OptimalRMSEtestcalctable.index[-1].strftime("%Y-%m-%d")+" ("+str(len(OptimalRMSEtestcalctable))+\
                                          " Days) :"+str(round(OptimalRMSEtest,2)))
        if backtestduration==0:
            print("RMSE-Train Data:"+str(round(OptimalRMSEtrain,2)))
            
        plt.axvline(x=end_date,linestyle='dashed')
        #plt.axvline(x=df_test.index[0],linestyle='dashed')
        plt.savefig(f'static/{targetfolder}/{imagecounter}_{j}_TESForecast.png')
        
        FinalOutput.append(OptimalTESoutput)
        BestConfigSummary.append(BestConfig)

        OptimalRMSESummaryList.append(OptimalRMSEtest)
        
    if backtestduration>0:
        OptimalRMSESummaryTable=pd.DataFrame([OptimalRMSESummaryList])
        OptimalRMSESummaryTable=OptimalRMSESummaryTable.T
        OptimalRMSESummaryTable.columns=["RMSE-Test Data From "+OptimalRMSEtestcalctable.index[0].strftime("%Y-%m-%d")+\
                                            " To "+OptimalRMSEtestcalctable.index[-1].strftime("%Y-%m-%d")+" ("+str(len(OptimalRMSEtestcalctable))+\
                                            " Days)"]
        OptimalRMSESummaryTable["Stocks"]=dfprices.columns
        OptimalRMSESummaryTable=OptimalRMSESummaryTable[["Stocks","RMSE-Test Data From "+OptimalRMSEtestcalctable.index[0].strftime("%Y-%m-%d")+\
                                            " To "+OptimalRMSEtestcalctable.index[-1].strftime("%Y-%m-%d")+" ("+str(len(OptimalRMSEtestcalctable))+\
                                            " Days)"]]
    elif backtestduration==0:
        OptimalRMSESummaryTable=pd.DataFrame()
        
        
    BestConfigSummaryTable=pd.DataFrame()
    for l in range(0,len(dfprices.columns)):
        BestConfigSummaryTable=pd.concat([BestConfigSummaryTable,BestConfigSummary[l]])

    BestConfigSummaryTable["Stocks"]=dfprices.columns
    BestConfigSummaryTable=BestConfigSummaryTable[['Stocks','trend t','damped trend d','seasonal s','seasonal periods p','use boxcox b','remove bias r','RMSE For Train Data']]
        
    FinalOutputTailTable=pd.DataFrame()

    for m in range(0,len(dfprices.columns)):
        FinalOutputTailTable=pd.concat([FinalOutputTailTable,FinalOutput[m].tail(60)],axis=1)

    FinalOutputTailTable

    return FinalOutput,FinalOutputTailTable,BestConfigSummaryTable,OptimalRMSESummaryTable,OverallRMSEforConfigTable

def EfficientPortfolioHistorical(start_date,end_date,symbols,portfolioValue,NoOfIterationsMC,AnnualRiskFreeRate,imagecounter,targetfolder):
    
    RiskFreeRate=(1+AnnualRiskFreeRate)**(1/252)-1
    #Effective rate for period = (1 + annual rate)**(1 / # of periods) – 1
    
    for symbol in symbols:
        dfprices = data.DataReader(symbols, start=start_date, end=end_date, data_source='yahoo')
        dfprices = dfprices[['Adj Close']]
    dfprices.columns=[' '.join(col).strip() for col in dfprices.columns.values]
    
    priceAtEndDate=[]
    for symbol in symbols:
        priceAtEndDate.append(dfprices[[f'Adj Close {symbol}']][-(1):].values[0][0])
    
    symbolsWPortfolio=symbols+["Portfolio"]
    
    ResultsTable=[]
    
    for i in range(0,NoOfIterationsMC):
        
        dfprices_inner=dfprices
        portfolioWeightsRandom=list(np.random.dirichlet(np.ones(len(symbols)),size=1)[0])
        
        noOfShares=[]
        portfolioValPerSymbol=[x * portfolioValue for x in portfolioWeightsRandom]
        for j in range(0,len(symbols)):
            noOfShares.append(portfolioValPerSymbol[j]/priceAtEndDate[j])
        noOfShares=[round(element, 5) for element in noOfShares]
        listOfColumns=dfprices_inner.columns.tolist()   
        dfprices_inner["Adj Close Portfolio"]=dfprices_inner[listOfColumns].mul(noOfShares).sum(1)

        dfreturns ,df_mean_stdev=calc_returns(dfprices_inner,symbols)
        
        mu=np.array(df_mean_stdev["Mean Log Daily Return"].values.tolist())
        sigma=np.array(df_mean_stdev["StdDev Log Daily Return"].values.tolist())
             
        IterationStdDev=df_mean_stdev.tail(1).values[0][2]
        IterationMean=df_mean_stdev.tail(1).values[0][1]
            
        negativereturnsonly=pd.DataFrame(dfreturns.iloc[:,len(dfreturns.columns)-1])
        negativereturnsonly=negativereturnsonly[negativereturnsonly['Log Daily Returns Adj Close Portfolio']<0]
        IterationNegativeReturnsStdDev=negativereturnsonly['Log Daily Returns Adj Close Portfolio'].std()
        
        # Note to go from LOG returns to Simple returns , I used simple returns =exp(log returns)−1 
        IterationSharpeRatio=round(((np.exp(IterationMean)-1)-RiskFreeRate)/(np.exp(IterationStdDev)-1),3)
        
        IterationSortinoRatio=round(((np.exp(IterationMean)-1)-RiskFreeRate)/(np.exp(IterationNegativeReturnsStdDev)-1),3)
        
        X=[portfolioWeightsRandom,IterationStdDev,IterationMean,IterationSharpeRatio,IterationSortinoRatio]
        
        ResultsTable.append(X)
        
        dfprices_inner.drop('Adj Close Portfolio',inplace=True, axis=1)
    
    FinalResultsTable=pd.DataFrame(ResultsTable,columns=["Weights","Std Dev","Mean","Sharpe Ratio","Sortino Ratio"])
    
    historical_dfreturns ,historical_df_mean_stdev=calc_returns(dfprices,symbols)
    
    historical_df_mean_stdev=historical_df_mean_stdev[['Stock','StdDev Log Daily Return','Mean Log Daily Return']]
    historical_df_mean_stdev.columns=['Stock','Std Dev','Mean']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    FinalResultsTable.plot.scatter(x="Std Dev",y='Mean',ax=ax)
    historical_df_mean_stdev.plot.scatter(x="Std Dev",y='Mean',c='r',marker='x',ax=ax)

    SharpeStdDev=FinalResultsTable.nlargest(1,['Sharpe Ratio'])['Std Dev'].values[0]
    SharpeMean=FinalResultsTable.nlargest(1,['Sharpe Ratio'])['Mean'].values[0]
    Sharperoundedweights=[round(num, 4) for num in FinalResultsTable.nlargest(1,['Sharpe Ratio'])['Weights'].values[0]]
    Sharpeweightstring=[]
    for i in range(0,len(symbols)):
        Sharpeweightstring.append([symbols[i]+":",Sharperoundedweights[i]])
    SharpeLabel="Optimal Sharpe Ratio"
    SharpeDetail='Optimal Sharpe Ratio: '+str(FinalResultsTable.nlargest(1,['Sharpe Ratio'])['Sharpe Ratio'].values[0])+" with Weights "+str(Sharpeweightstring)

    SortinoStdDev=FinalResultsTable.nlargest(1,['Sortino Ratio'])['Std Dev'].values[0]
    SortinoMean=FinalResultsTable.nlargest(1,['Sortino Ratio'])['Mean'].values[0]
    Sortinoroundedweights=[round(num, 4) for num in FinalResultsTable.nlargest(1,['Sortino Ratio'])['Weights'].values[0]]
    Sortinoweightstring=[]
    for i in range(0,len(symbols)):
        Sortinoweightstring.append([symbols[i]+":",Sortinoroundedweights[i]])
    SortinoLabel='Optimal Sortino Ratio'
    SortinoDetail='Optimal Sortino Ratio: '+str(FinalResultsTable.nlargest(1,['Sortino Ratio'])['Sortino Ratio'].values[0])+" with Weights "+str(Sortinoweightstring)
    
    SharpeSortino=pd.DataFrame(zip([SharpeStdDev,SortinoStdDev],[SharpeMean,SortinoMean]),index=['Optimal Sharpe','Optimal Sortino'],columns=['Std Dev','Mean'])
    SharpeSortino.plot.scatter(x="Std Dev",y='Mean',c='g',marker='x',ax=ax)
    
    txt=list(historical_df_mean_stdev['Stock'])+[SharpeLabel,SortinoLabel]
    z=list(historical_df_mean_stdev['Std Dev'])+[SharpeStdDev,SortinoStdDev]
    y=list(historical_df_mean_stdev['Mean'])+[SharpeMean,SortinoMean]

    for i, text in enumerate(txt):
        ax.annotate(text, (z[i], y[i]))
    
    plt.title("Mean vs Std Dev Of Log Returns For "+str(NoOfIterationsMC)+" Different Portfolio Weights")
    plt.savefig(f'static/{targetfolder}/{imagecounter}_efficientportfolio.png')
    print(SharpeDetail)
    print(SortinoDetail)
    
    FinalResultsTable['Log Returns Std Dev']=FinalResultsTable['Std Dev']
    FinalResultsTable['Log Returns Mean']=FinalResultsTable['Mean']
    FinalResultsTable=FinalResultsTable[['Weights','Log Returns Std Dev','Log Returns Mean','Sharpe Ratio','Sortino Ratio']]
    
    return FinalResultsTable, SharpeDetail, SortinoDetail

def EfficientPortfolioFuture(start_date,end_date,symbols,portfolioValue,T,N,NoOfIterationsMC,NoOfIterationsInnerLoop,AnnualRiskFreeRate,SimMethod,imagecounter,targetfolder):
    
    symbolsWPortfolio=symbols+["Portfolio"]

    RiskFreeRate=(1+AnnualRiskFreeRate)**(1/252)-1
    #Effective rate for period = (1 + annual rate)**(1 / # of periods) – 1
    
    for symbol in symbols:
        dfprices = data.DataReader(symbols, start=start_date, end=end_date, data_source='yahoo')
        dfprices = dfprices[['Adj Close']]
    dfprices.columns=[' '.join(col).strip() for col in dfprices.columns.values]
    
    priceAtEndDate=[]
    for symbol in symbols:
        priceAtEndDate.append(dfprices[[f'Adj Close {symbol}']][-(1):].values[0][0])
    
    symbolsWPortfolio=symbols+["Portfolio"]
    
    ResultsTable=[]
    
    for i in range(0,NoOfIterationsMC):
        
        dfprices_inner=dfprices
        portfolioWeightsRandom=list(np.random.dirichlet(np.ones(len(symbols)),size=1)[0])
        
        noOfShares=[]
        portfolioValPerSymbol=[x * portfolioValue for x in portfolioWeightsRandom]
        for j in range(0,len(symbols)):
            noOfShares.append(portfolioValPerSymbol[j]/priceAtEndDate[j])
        noOfShares=[round(element, 5) for element in noOfShares]
        listOfColumns=dfprices_inner.columns.tolist()   
        dfprices_inner["Adj Close Portfolio"]=dfprices_inner[listOfColumns].mul(noOfShares).sum(1)

        dfreturns ,df_mean_stdev=calc_returns(dfprices_inner,symbols)
        
        S0=np.array(dfprices.tail(1).values.tolist()[0])
        mu=np.array(df_mean_stdev["Mean Log Daily Return"].values.tolist())
        sigma=np.array(df_mean_stdev["StdDev Log Daily Return"].values.tolist())
        
        if SimMethod=="GBM":

            forecastresults=pd.DataFrame()
            percentiles=pd.DataFrame()
            
            for x in range(1,int(NoOfIterationsInnerLoop)):      
                
                Cov=create_covar(dfreturns)
                stocks, time = GBMsimulatorMultiVar(S0, mu, sigma, Cov, T, N)
                prediction=pd.DataFrame(stocks)
                prediction=prediction.T
                prediction.columns=dfprices.columns
                forecastresults=pd.concat([forecastresults,prediction], axis=1, sort=False)
    
            for y in range(0,len(symbolsWPortfolio)):
                percentiles["P50_"+symbolsWPortfolio[y]]=forecastresults.filter(regex=symbolsWPortfolio[y]).quantile(0.5,1) 
     
            IterationReturn,Iteration_Mean_Stdev=calc_returns(percentiles,symbols)
            IterationStdDev=Iteration_Mean_Stdev.tail(1).values[0][2]
            IterationMean=Iteration_Mean_Stdev.tail(1).values[0][1]
            IterationMeanComponentStocks=Iteration_Mean_Stdev.T.loc["Mean Log Daily Return"][:-1].values.tolist()
            IterationStdDevComponentStocks=Iteration_Mean_Stdev.T.loc["StdDev Log Daily Return"][:-1].values.tolist()
            
            negativereturnsonly=pd.DataFrame(IterationReturn.iloc[:,len(IterationReturn.columns)-1])
            negativereturnsonly=negativereturnsonly[negativereturnsonly[negativereturnsonly.columns[0]]<0]            
            IterationNegativeReturnsStdDev=negativereturnsonly[negativereturnsonly.columns[0]].std()
                        
        elif SimMethod=="Bootstrap":
            
            forecastresults=pd.DataFrame()
            returnspercentiles=pd.DataFrame()
            
            for x in range(1,int(NoOfIterationsInnerLoop)):  
            
                prediction=bootstrapforecast(dfreturns,T)
                prediction=prediction.add_prefix('Iter_'+str(x)+'_')
                forecastresults=pd.concat([forecastresults,prediction], axis=1, sort=False)
                    
            for y in range(0,len(symbolsWPortfolio)):
                returnspercentiles["P50_"+symbolsWPortfolio[y]]=forecastresults.filter(regex=symbolsWPortfolio[y]).quantile(0.5,1)
            
            IterationMeanComponentStocks=[]
            IterationStdDevComponentStocks=[]
            
            for y in range(0,int(len(returnspercentiles.columns)-1)):
                IterationMeanComponentStocks.append(returnspercentiles[returnspercentiles.columns[y]].mean())
                IterationStdDevComponentStocks.append(returnspercentiles[returnspercentiles.columns[y]].std())
            
            IterationStdDev=returnspercentiles[returnspercentiles.columns[-1]].std()
            IterationMean=returnspercentiles[returnspercentiles.columns[-1]].mean()
            
            negativereturnsonly=returnspercentiles[returnspercentiles[returnspercentiles.columns[-1]]<0]          
            IterationNegativeReturnsStdDev=negativereturnsonly[negativereturnsonly.columns[0]].std()
            
        # Note to go from LOG returns to Simple returns , I used simple returns =exp(log returns)−1 
        IterationSharpeRatio=round(((np.exp(IterationMean)-1)-RiskFreeRate)/(np.exp(IterationStdDev)-1),3)
        
        IterationSortinoRatio=round(((np.exp(IterationMean)-1)-RiskFreeRate)/(np.exp(IterationNegativeReturnsStdDev)-1),3)
        
        X=[portfolioWeightsRandom,IterationStdDev,IterationMean,IterationSharpeRatio,IterationSortinoRatio,\
           IterationStdDevComponentStocks,IterationMeanComponentStocks]
        
        ResultsTable.append(X)
        
        dfprices_inner.drop('Adj Close Portfolio',inplace=True, axis=1)
    
    FinalResultsTable=pd.DataFrame(ResultsTable,columns=["Weights","Std Dev","Mean",\
                                                         "Sharpe Ratio","Sortino Ratio","Components Log Returns Std Dev","Components Log Returns Mean"])

    SharpeStdDev=FinalResultsTable.nlargest(1,['Sharpe Ratio'])['Std Dev'].values[0]
    SharpeMean=FinalResultsTable.nlargest(1,['Sharpe Ratio'])['Mean'].values[0]
    Sharperoundedweights=[round(num, 4) for num in FinalResultsTable.nlargest(1,['Sharpe Ratio'])['Weights'].values[0]]
    Sharpeweightstring=[]
    for i in range(0,len(symbols)):
        Sharpeweightstring.append([symbols[i]+":",Sharperoundedweights[i]])
    SharpeLabel="Optimal Sharpe Ratio"
    SharpeDetail='Optimal Sharpe Ratio: '+str(FinalResultsTable.nlargest(1,['Sharpe Ratio'])['Sharpe Ratio'].values[0])+" with Weights "+str(Sharpeweightstring)
    
    SortinoStdDev=FinalResultsTable.nlargest(1,['Sortino Ratio'])['Std Dev'].values[0]
    SortinoMean=FinalResultsTable.nlargest(1,['Sortino Ratio'])['Mean'].values[0]
    Sortinoroundedweights=[round(num, 4) for num in FinalResultsTable.nlargest(1,['Sortino Ratio'])['Weights'].values[0]]
    Sortinoweightstring=[]
    for i in range(0,len(symbols)):
        Sortinoweightstring.append([symbols[i]+":",Sortinoroundedweights[i]])
    SortinoLabel='Optimal Sortino Ratio'
    SortinoDetail='Optimal Sortino Ratio: '+str(FinalResultsTable.nlargest(1,['Sortino Ratio'])['Sortino Ratio'].values[0])+" with Weights "+str(Sortinoweightstring)
       
    SharpeSortino=pd.DataFrame(zip([SharpeStdDev,SortinoStdDev],[SharpeMean,SortinoMean]),index=['Optimal Sharpe','Optimal Sortino'],columns=['Std Dev','Mean'])
  
    SharpeRatio_Best=pd.DataFrame(FinalResultsTable.nlargest(1,['Sharpe Ratio'])["Components Log Returns Mean"].values[0],index=symbols,columns=["Mean Log Daily Return"])
    SharpeRatio_Best["StdDev Log Daily Return"]=FinalResultsTable.nlargest(1,['Sharpe Ratio'])["Components Log Returns Std Dev"].values[0]
    SharpeRatio_Best
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    SharpeSortino.plot.scatter(x="Std Dev",y='Mean',c='g',marker='x',ax=ax)
  
    FinalResultsTable.plot.scatter(x="Std Dev",y='Mean',ax=ax)
    SharpeRatio_Best.plot.scatter(x="StdDev Log Daily Return",y='Mean Log Daily Return',c='r',marker='x',ax=ax)

    txt=list(SharpeRatio_Best.index)+[SharpeLabel,SortinoLabel]
    z=list(SharpeRatio_Best['StdDev Log Daily Return'])+[SharpeStdDev,SortinoStdDev]
    y=list(SharpeRatio_Best['Mean Log Daily Return'])+[SharpeMean,SortinoMean]

    for i, text in enumerate(txt):
        ax.annotate(text, (z[i], y[i]))

    plt.title("Mean vs Std Dev Of P50 Log Returns For "+str(NoOfIterationsMC)+" Different Portfolio Weights Simulated Using "+SimMethod+" over "+str(NoOfIterationsInnerLoop)+" Iters")
    plt.savefig(f'static/{targetfolder}/{imagecounter}_efficientportfolio.png')
    print(SharpeDetail)
    print(SortinoDetail)
    
    FinalResultsTable['Log Returns Std Dev']=FinalResultsTable['Std Dev']
    FinalResultsTable['Log Returns Mean']=FinalResultsTable['Mean']
    FinalResultsTable=FinalResultsTable[['Weights','Log Returns Std Dev','Log Returns Mean','Sharpe Ratio','Sortino Ratio',"Components Log Returns Std Dev","Components Log Returns Mean"]]
    
    return FinalResultsTable, SharpeDetail, SortinoDetail

#VECTOR AUTOREGRESSION

#https://www.analyticsvidhya.com/blog/2018/09/multivariate-time-series-guide-forecasting-modeling-python-codes/
#https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests


# Cointegration test helps to establish the presence of a statistically 
# significant connection between two or more time series.  
def cointegration_test(df,LagOrder_LowestAIC,alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,LagOrder_LowestAIC-1)
#     coint_johansen(df,det_order,Number of lagged differences in the model)    
#     det_order = order of time polynomial in the null-hypothesis
#     det_order = -1, no deterministic part
#     det_order =  0, for constant term
#     det_order =  1, for constant plus time-trend  
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    cointDFarray=[]
    for col, trace, cvt in zip(df.columns, traces, cvts):
        cointDFarray.append([col,round(trace,2),cvt, trace>cvt])
    cointDF=pd.DataFrame(cointDFarray,columns=["Name","Johanson Cointegration Test Stat","Critical Value 5% Signif Lvl","Accept(i.e True) / Reject(i.e False)"])
    cointDF=cointDF[["Name","Johanson Cointegration Test Stat","Critical Value 5% Signif Lvl","Accept(i.e True) / Reject(i.e False)"]]
    
    return cointDF


# Granger’s causality tests the null hypothesis that the coefficients of past values in the regression equation is zero.
# In simpler terms, the past values of time series (X) do not cause the other series (Y). 
# So, if the p-value obtained from the test is lesser than the significance level of 0.05, 
# then, you can safely reject the null hypothesis.

maxlag=21 #Reflects a month of lag if 1 year ~252 trading days
test = 'ssr_chi2test'

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


def vectorAutoRegression(start_date,end_date,symbols,portfolioWeights,portfolioValue,forecast_end_date,backtest_duration,imagecounter,targetfolder):
    
    if len(symbols)>1:
        dfprices, noOfShares, share_split_table = extract_prices(start_date,end_date,symbols,portfolioWeights,portfolioValue)
        dfreturns ,df_mean_stdev=calc_returns(dfprices,symbols)

        T=np.busday_count(end_date,forecast_end_date)+backtest_duration
        N=T+1

        #creating the train and validation set
        symbolsWPortfolio=symbols+["Portfolio"]
        train = dfreturns[:len(dfreturns)-backtest_duration]
        pricetrain = dfprices[:len(dfreturns)-backtest_duration+1]
        valid = dfreturns[len(dfreturns)-backtest_duration:]
        cols=train.columns

        #fit the model

        VARmodel = VAR(endog=train)

        bestAIC=[]
        for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]: #reflects test for lag order up till 1 mth ~21 days/mth
            result = VARmodel.fit(i)
            bestAIC.append([i, result.aic])
        bestAIC_DF=pd.DataFrame(bestAIC,columns=["Lag Order","AIC"])    
        LagOrder_LowestAIC=bestAIC_DF.nsmallest(1,['AIC'])['Lag Order'].values[0]
        
        #In a VAR(p) model, the first p lags of each variable in the system is used as regression predictors for EACH variable

        VARmodel_fit = VARmodel.fit(LagOrder_LowestAIC)
        VARmodel_fit.summary()

        from statsmodels.stats.stattools import durbin_watson
        out = durbin_watson(VARmodel_fit.resid)

        DWTableElements=[]
        for col, val in zip(cols, out):
            DWTableElements.append([col,val])

        DWTable=pd.DataFrame(DWTableElements,columns=["Stock Log Returns","Serial Corr"])

        extended_dates_future=[]
        for i in range(0,N-backtest_duration):
            extended_dates_future.append(pd.Timestamp(numpy.datetime64(np.busday_offset(end_date, i, roll='forward'))))

        extended_dates=valid.index.tolist() +extended_dates_future[1:]

        # make prediction on validation
        prediction = VARmodel_fit.forecast(VARmodel_fit.y, steps=len(extended_dates))

        #converting predictions to dataframe
        pred = pd.DataFrame(prediction,index=extended_dates,columns=[cols])

        pred
        startingrefprice=pricetrain.tail(1)

        stockprices=np.exp(pred)
        stockprices=stockprices.cumprod()
        stockprices=stockprices.mul(startingrefprice.values[0])
        stockprices.columns=dfprices.columns
        firstrow=pd.DataFrame(startingrefprice)
        stockprices=pd.concat([firstrow,stockprices])
        stockprices=stockprices.add_prefix("VAR_Forecast ")
        actual=dfprices
        stockprices=pd.concat([actual,stockprices],axis=1)
        for symbol in symbolsWPortfolio:
            stockprices[["Adj Close "+symbol,"VAR_Forecast Adj Close "+symbol]].tail(60).plot(figsize=(15,5))
            plt.title(symbol+": Vector Auto Regression Forecast With Optimal Lag Order="+str(bestAIC_DF.nsmallest(1,["AIC"])["Lag Order"].values[0])\
                     +" (AIC Value:"+str(bestAIC_DF.nsmallest(1,["AIC"])["AIC"].values[0])+" )")
            plt.axvline(x=end_date,linestyle='dashed')
            plt.savefig(f'static/{targetfolder}/{imagecounter}_{symbol}VectorAR.png')
            


        RMSECalcTable=stockprices.tail(len(extended_dates))
        RMSECalcTable=RMSECalcTable.head(len(valid))

        if backtest_duration>0:
            RMSE=[]
            anothertable=[]

            for i in range(0,len(dfprices.columns)):
                temptable=RMSECalcTable
                temptable["RMSE"]=(temptable.iloc[:,i]-temptable.iloc[:,i+len(dfprices.columns)])**2
                RMSE.append((temptable["RMSE"].mean())**0.5)

            RMSE_DF=pd.DataFrame(RMSE,index=dfprices.columns,columns=["RMSE For Backtest From "+temptable.index[0].strftime("%Y-%m-%d")+" To "+temptable.index[-1].strftime("%Y-%m-%d")\
                                                                       +" ("+str(len(temptable))+" Days)"])

        elif backtest_duration<=0:
            RMSE_DF=pd.DataFrame()

        print("Root Mean Square Error For Backtest Period")    
        #display(RMSE_DF)

        print('DURBIN WATSON TEST')
        print('Checks the serial correlation of the residuals errors if there is some pattern in the time series not explained by the Vector AutoRegression model')
        print('#Statistic ranges between 0 to 4. The closer to 2, then no significant serial correlation.The closer to 0, there is a positive serial correlation, closer to 4 implies negative serial correlation.')
        #display(DWTable)

        GrangersMatrix=grangers_causation_matrix(dfreturns, variables = dfreturns.columns)
        print('GRANGER CAUSALITY TEST (Using Sum Of Squared Residuals With Chi Square Test For Max Lag Of 1 Month (21 Trading Days)')
        print('Rows are the Response _y and columns are the predictor series _x so values refers to the p-value of log returns stock A_x causing log returns stock B_y')
        print('If the p-value obtained from the test is lesser than the significance level of 0.05, then, you can safely reject the null hypothesis that that the coefficients of past values in the regression equation is zero')
        #display(GrangersMatrix)

        JohansonCointTable=cointegration_test(dfreturns,LagOrder_LowestAIC)
        print('JOHANSON COINTEGRATION TEST (Assuming no deterministic part for the order of time polynomial)') 
        print('Establishes the presence of a statistically significant connection between two or more time series.')
        print('If Test Stat > Critical Value (TRUE) then we accept that there is a significant connection')
        #display(JohansonCointTable)
    
    if len(symbols)==1:
        stockprices=pd.DataFrame()
        RMSE_DF=pd.DataFrame()
        DWTable=pd.DataFrame()
        GrangersMatrix=pd.DataFrame()
        JohansonCointTable=pd.DataFrame()
        
        print('Error-Vector Auto Regression applicable only for Multivariate Time Series!')
    
    return stockprices,RMSE_DF,DWTable,GrangersMatrix,JohansonCointTable

#FACTOR ANALYSIS

#function to extract statistics for each stock
def get_key_stats(tgt_website):
 
    # The web page is make up of several html table. By calling read_html function.
    # all the tables are retrieved in dataframe format.
    # Next is to append all the table and transpose it to give a nice one row data.
    df_list = pd.read_html(tgt_website)
    df_statistics = df_list[0]
 
    for df in df_list[1:]:
        df_statistics = df_statistics.append(df)
 
    # Transpose the result to make all data in single row
    return df_statistics.set_index(0).T

#function to find the last recent weekday for a given date e.g. Fri if Sat or Sun 

def lastBusDay(enterdate):
    if enterdate.weekday()==6:
        lastBusDay = datetime(year=enterdate.year, month=enterdate.month, day=enterdate.day-2)
    elif enterdate.weekday()==5:
        lastBusDay = datetime(year=enterdate.year, month=enterdate.month, day=enterdate.day-1)
    else:
        lastBusDay = enterdate
    return lastBusDay

#See example https://finance.yahoo.com/quote/C52.SI/key-statistics?p=C52.SI
#This function extracts the relevant statistic and converts the unit into Millions and turns it into a float

def extract_statistic(df_statistics,metric,symbol):
    x=0
    if metric=="Market Cap (intraday) 5" or metric=="Enterprise Value 3" or metric=="Trailing P/E" or metric=="Forward P/E 1" or metric=="PEG Ratio (5 yr expected) 1" or metric=="Price/Sales (ttm)" or metric=="Price/Book (mrq)" or metric=="Enterprise Value/Revenue 3" or metric=="Enterprise Value/EBITDA 6":
      if metric=="Market Cap (intraday) 5":
        x=0
      elif metric=="Enterprise Value 3":
        x=1
      elif metric=="Trailing P/E" :
        x=2
      elif metric=="Forward P/E 1":
        x=3
      elif metric=="PEG Ratio (5 yr expected) 1":
        x=4 
      elif metric=="Price/Sales (ttm)":
        x=5
      elif metric=="Price/Book (mrq)":
        x=6 
      elif metric=="Enterprise Value/Revenue 3":
        x=7 
      elif metric=="Enterprise Value/EBITDA 6":
        x=8
      try:
        String=df_statistics.get(symbol).iloc[-2:-1,x][0]
      except:
        String=''
      Metric=''
      try:
          if 'T' in String:
              Metric=float(df_statistics.get(symbol).iloc[-2:-1,x][0].replace("T",""))*1e6
          elif 'B' in String:
              Metric=float(df_statistics.get(symbol).iloc[-2:-1,x][0].replace("B",""))*1e3
          elif 'M' in String:
              Metric=float(df_statistics.get(symbol).iloc[-2:-1,x][0].replace("M",""))
          elif '%' in String:
              Metric=float(df_statistics.get(symbol).iloc[-2:-1,x][0].replace("%",""))
          else :
              Metric=float(df_statistics.get(symbol).iloc[-2:-1,x][0])
      except:
          if String==np.nan:
            Metric=np.nan

    else:
      try:
        String=df_statistics.get(symbol).loc[1][metric]
      except:
          String=''
      Metric=''
      try:
          if 'T' in String:
              Metric=float(df_statistics.get(symbol).loc[1][metric].replace("T",""))*1e6
          elif 'B' in String:
              Metric=float(df_statistics.get(symbol).loc[1][metric].replace("B",""))*1e3
          elif 'M' in String:
              Metric=float(df_statistics.get(symbol).loc[1][metric].replace("M",""))
          elif '%' in String:
              Metric=float(df_statistics.get(symbol).loc[1][metric].replace("%",""))
          else :
              Metric=float(df_statistics.get(symbol).loc[1][metric])
      except:
          if String==np.nan:
              Metric=np.nan
    #print(symbol,":",metric,":",Metric)
    return metric,Metric

#Cronbach’s alpha is a convenient test used to estimate the reliability, or internal consistency, of a composite score
#Based on https://mathtuition88.com/2019/09/13/calculate-cronbach-alpha-using-python/

def CronbachAlpha(itemscores):
    itemscores = np.asarray(itemscores)
    itemvars = itemscores.var(axis=0, ddof=1)
    tscores = itemscores.sum(axis=1)
    nitems = itemscores.shape[1]

    return (nitems / (nitems-1)) * (1 - (itemvars.sum() / tscores.var(ddof=1)))

#based on https://stackoverflow.com/questions/8595973/truncate-to-three-decimals-in-python
#used later for geometric mean to avoid overflow error when exponentiating large figures
def trun_n_d(n,d):
    return int(n*10**d)/10**d

#This function normalises the factors and converts them into a single composite score that is then ranked 

def getStockReturns(symbols,current,currentMinus1Yr):
    prices_current=[]
    prices_lastyear=[]
    prices_return=[]

    for symbol in symbols:
        try:
            price_current=round(data.DataReader(symbol, start=lastBusDay(current), end=lastBusDay(current), data_source='yahoo')['Adj Close'][-1],4)
            price_lastyear=round(data.DataReader(symbol, start=lastBusDay(currentMinus1Yr), end=lastBusDay(currentMinus1Yr), data_source='yahoo')['Adj Close'][-1],4)
            price_return=round((price_current/price_lastyear-1)*100,1)
            print("Extracted Prices For ",symbol)
            time.sleep(random.randint(1,2))

        except:
            price_current=np.nan
            price_lastyear=np.nan
            price_return=np.nan
            print("Unable To Extract Prices For ",symbol)

        prices_current.append(price_current)
        prices_lastyear.append(price_lastyear)
        prices_return.append(price_return)

    stockprice_df=pd.DataFrame(zip(prices_current,prices_lastyear,prices_return),\
                                   columns=["Adj Close "+lastBusDay(current).strftime('%Y-%m-%d'),\
                                            "Adj Close "+lastBusDay(currentMinus1Yr).strftime('%Y-%m-%d'),"Stock Price Returns"],index=symbols)
    
    return stockprice_df

def rankstocks(symbols,df_statistics,listOfMetrics,select_order_by_metric,weights_by_metric,NormMethod,stockprice_df):
    
    raw_df=pd.DataFrame()
    for symbol in symbols:
        title=[]
        value=[]
        for item in listOfMetrics:
            itemtitle,itemvalue=extract_statistic(df_statistics,item,symbol)
            title.append(itemtitle)
            value.append(itemvalue)
        raw_df=raw_df.append(pd.DataFrame(value).T)
    raw_df.index=symbols
    raw_df.columns=listOfMetrics
    raw_df.replace('', np.nan, inplace=True)
 
    summary_df=pd.DataFrame()
    for symbol in symbols:
        title=[]
        value=[]
        for item in listOfMetrics:
            itemtitle,itemvalue=extract_statistic(df_statistics,item,symbol)
            title.append(itemtitle)
            value.append(itemvalue)
        summary_df=summary_df.append(pd.DataFrame(value).T)
    summary_df.index=symbols
    summary_df.columns=listOfMetrics
    summary_df.replace('', np.nan, inplace=True)
    summary_df.dropna(inplace=True)
    summary_df.replace(0.00,0.0001,inplace=True)
    summary_df=summary_df.convert_dtypes()
    
    summary_method=pd.DataFrame(zip(listOfMetrics,select_order_by_metric,weights_by_metric),columns=["Factor","Select Highest vs Lowest","Weightage"])

    
    if len(summary_df)<=1:
        
        return summary_method,raw_df,pd.DataFrame(),pd.DataFrame(),0.0,0.0 # calculation aborted due to insufficient stocks to compare

    elif len(summary_df)>1:    

        ranking_df=summary_df

        if NormMethod=="Z-Score Normalization + Additive Composite":
            for i in range(0,len(summary_df.columns)):
                if select_order_by_metric[i]=='Lowest':
                    ranking_df[summary_df.columns[i]]=1/ranking_df[summary_df.columns[i]]
                    ranking_df[summary_df.columns[i]]=ranking_df[[summary_df.columns[i]]].apply(zscore)
                elif select_order_by_metric[i]=='Highest':
                    ranking_df[summary_df.columns[i]]=summary_df[[summary_df.columns[i]]].apply(zscore)
            ranking_df['Composite Score']=ranking_df.dot(weights_by_metric)

        elif NormMethod=="MinMax Normalization + Additive Composite":
            for i in range(0,len(summary_df.columns)):
                xmin=ranking_df[summary_df.columns[i]].min()
                xmax=ranking_df[summary_df.columns[i]].max()
                maxminrange=xmax-xmin
                if select_order_by_metric[i]=='Lowest':
                    ranking_df[summary_df.columns[i]]=(xmax-ranking_df[summary_df.columns[i]])/maxminrange
                elif select_order_by_metric[i]=='Highest':
                    ranking_df[summary_df.columns[i]]=(ranking_df[summary_df.columns[i]]-xmin)/maxminrange
            ranking_df['Composite Score']=ranking_df.dot(weights_by_metric)

        elif NormMethod=="Percentile Normalization + Additive Composite":
            for i in range(0,len(summary_df.columns)):
                sz = ranking_df[summary_df.columns[i]].size-1
                if select_order_by_metric[i]=='Lowest':
                    ranking_df[summary_df.columns[i]] = 100-ranking_df[summary_df.columns[i]].rank(method='max').apply(lambda x: 100.0*(x-1)/sz)
                elif select_order_by_metric[i]=='Highest':
                    ranking_df[summary_df.columns[i]] = ranking_df[summary_df.columns[i]].rank(method='max').apply(lambda x: 100.0*(x-1)/sz)
            ranking_df['Composite Score']=ranking_df.dot(weights_by_metric)
                    
        elif NormMethod=="MinMax Normalization + Multiplicative Composite":
            #must normalize using min-max as Z scores will create zero and negative figures
            for i in range(0,len(summary_df.columns)):
                xmin=ranking_df[summary_df.columns[i]].min()
                xmax=ranking_df[summary_df.columns[i]].max()
                maxminrange=xmax-xmin
                if select_order_by_metric[i]=='Lowest':
                    ranking_df[summary_df.columns[i]]=(xmax-ranking_df[summary_df.columns[i]])/maxminrange
                elif select_order_by_metric[i]=='Highest':
                    ranking_df[summary_df.columns[i]]=(ranking_df[summary_df.columns[i]]-xmin)/maxminrange

            ranking_df=ranking_df.add(0.000001) # adding small value to all fields to avoid zero values

            WGMa=0
            WGMarray=[]
            for stock in ranking_df.index:
                for i in range(0,len(listOfMetrics)):
                    WGMa=WGMa+ranking_df.loc[stock][i]**weights_by_metric[i] 
                    #WGMa=WGMa+trun_n_d(ranking_df.loc[stock][i],5)**trun_n_d(weights_by_metric[i],5)
                    # Note : trun_n_d is a truncation function to 5 decimal points to avoid overflow error during expotentiation
                WGMb=WGMa**(1/sum(weights_by_metric))
                WGMarray.append(WGMb)

            ranking_df['Composite Score']=WGMarray        
        
        ranking_df['Rank By Composite Score']=ranking_df['Composite Score'].rank(ascending=False)
        ranking_df=ranking_df.sort_values(by=['Rank By Composite Score'])

        cronbachalpha=CronbachAlpha(ranking_df[ranking_df.columns[0:len(listOfMetrics)]])

        value=[]
        for symbol in ranking_df.index:
            itemtitle,itemvalue=extract_statistic(df_statistics,'Market Cap (intraday) 5',symbol)
            value.append(itemvalue)

        ranking_df['Market Cap (intraday) 5']=value

        ranking_df=pd.merge(ranking_df,stockprice_df,left_index=True,right_index=True,how='left')    
                        
        x=ranking_df['Composite Score'].values
        y=ranking_df['Stock Price Returns'].values
        correlcoeff=np.corrcoef(x,y)[0][1]
        
        #https://pbpython.com/pandas-qcut-cut.html
        ranking_df_w_quartiles=ranking_df
        ranking_df_w_quartiles['Quartiles By Composite Score'] = pd.qcut(ranking_df['Composite Score'], q=4, labels=['Bottom Quartile','3rd Quartile','2nd Quartile','Top Quartile'],precision=0)
        ranking_df_w_quartiles=ranking_df_w_quartiles[['Quartiles By Composite Score','Stock Price Returns']].groupby(['Quartiles By Composite Score']).median()
        
        
        value=[]
        for symbol in raw_df.index:
            itemtitle,itemvalue=extract_statistic(df_statistics,'Market Cap (intraday) 5',symbol)
            value.append(itemvalue)

        raw_df['Market Cap (intraday) 5']=value
        raw_df.replace('', np.nan, inplace=True)
        
        return summary_method,raw_df,ranking_df,ranking_df_w_quartiles,cronbachalpha,correlcoeff
    
    
#function to plot in a grid the correlation between each factor and the Stock Returns
def correlation(ranking_df,listOfMetrics, n_rows, n_cols,imagecounter):
    jet= plt.get_cmap('jet')
    colors = iter(jet(np.linspace(0,1,10)))
    fig = plt.figure(figsize=(5,15))
    #fig = plt.figure(figsize=(14,9))
    for i, var in enumerate(listOfMetrics):
        ax = fig.add_subplot(n_rows,n_cols,i+1)
        asset = ranking_df.loc[:,var]
        ax.scatter(ranking_df['Stock Price Returns'], asset, color = next(colors))
        ax.set_xlabel('Stock Price Returns')
        ax.set_ylabel("{}".format(var))
        ax.set_title(var +" vs Stock Price Returns")
    fig.tight_layout()
    plt.savefig(f'static/factoranalysis/{imagecounter}_02_CorrelByFactor.png')
    #plt.show()
    
from flask import Flask, render_template, url_for, request, session
import os

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def start():
    return render_template('index.html')

@app.route('/results', methods = ['POST'])
def results():
    imagecounter = str(random.randint(10000,99999))
    mainSelection=request.form['mainSelection']
    tickerPortfolio=request.form['tickerPortfolio']
    tickerPortfolio=tickerPortfolio.replace(" ","") 
    tickerPortfolio=tickerPortfolio.replace("'","")       
    
    portfolioWeights=request.form['portfolioWeights']
    portfolioValue=request.form.get('portfolioValue',type=float)
    StartDate = request.form['StartDate'] 
    EndDate=request.form['EndDate'] 
    ForecastDate=request.form['ForecastDate'] 
    BacktestDays=int(request.form['BacktestDays'])
    NoOfIter= int(request.form['NoOfIter'])
    PercentileRange= request.form['PercentileRange']
    averageType=request.form['AverageType']
    WindowSize1=int(request.form['WindowSize1'])
    WindowSize2=int(request.form['WindowSize2'])
    WindowSize3=int(request.form['WindowSize3'])
    noOfStdDev=int(request.form['noOfStdDev'])
    riskfreerate=request.form.get('riskfreerate',type=float)
    simMethod=request.form['simMethod']
    seasonality=int(request.form['seasonality'])

    if mainSelection=="candlestickChart":
        clear_cache('candlesticks')
        symbols=tickerPortfolio.split(",")
        plot_candlesticks(symbols,StartDate,EndDate,imagecounter,"candlesticks")
        
        hists = os.listdir('static/candlesticks')
        hists = ['candlesticks/' + file for file in hists]
        
        analysistype="Candlestick Chart"      
        
        return render_template('results.html',analysistype=analysistype, hists = hists, mainSelection=mainSelection,\
                               tickerPortfolio=tickerPortfolio,portfolioWeights=portfolioWeights,\
                               portfolioValue=portfolioValue,StartDate=StartDate,EndDate=EndDate,ForecastDate=ForecastDate,\
                               BacktestDays=BacktestDays,NoOfIter=NoOfIter,PercentileRange=PercentileRange,\
                               averageType=averageType,WindowSize1=WindowSize1,WindowSize2=WindowSize2,WindowSize3=WindowSize3,riskfreerate=riskfreerate,simMethod=simMethod\
                               )
    
    elif mainSelection=="RSIChart":
        clear_cache('relativestrengthindex')
        symbols=tickerPortfolio.split(",")
        plot_RSI(symbols,StartDate,EndDate,WindowSize1,averageType,imagecounter,"relativestrengthindex")
        
        hists = os.listdir('static/relativestrengthindex')
        hists = ['relativestrengthindex/' + file for file in hists]
        
        analysistype="Relative Strength Index Using "+averageType+" With Period Of "+str(WindowSize1)+" Days"      
        
        return render_template('results.html',analysistype=analysistype, hists = hists, mainSelection=mainSelection,\
                               tickerPortfolio=tickerPortfolio,portfolioWeights=portfolioWeights,\
                               portfolioValue=portfolioValue,StartDate=StartDate,EndDate=EndDate,ForecastDate=ForecastDate,\
                               BacktestDays=BacktestDays,NoOfIter=NoOfIter,PercentileRange=PercentileRange,\
                               averageType=averageType,WindowSize1=WindowSize1,WindowSize2=WindowSize2,WindowSize3=WindowSize3,riskfreerate=riskfreerate,simMethod=simMethod\
                               )
    
    elif mainSelection=="MACDChart":
        clear_cache('macd')
        symbols=tickerPortfolio.split(",")
        plot_MACD(StartDate,EndDate,symbols,WindowSize1,WindowSize2,WindowSize3,imagecounter,"macd")     
        
        hists = os.listdir('static/macd')
        hists = ['macd/' + file for file in hists]
        
        analysistype="Moving Average Convergence Divergence Using "+averageType+" With Periods Of "+str(WindowSize1)+" & "+str(WindowSize2)+" Days"      
        
        return render_template('results.html',analysistype=analysistype, hists = hists, mainSelection=mainSelection,\
                               tickerPortfolio=tickerPortfolio,portfolioWeights=portfolioWeights,\
                               portfolioValue=portfolioValue,StartDate=StartDate,EndDate=EndDate,ForecastDate=ForecastDate,\
                               BacktestDays=BacktestDays,NoOfIter=NoOfIter,PercentileRange=PercentileRange,\
                               averageType=averageType,WindowSize1=WindowSize1,WindowSize2=WindowSize2,WindowSize3=WindowSize3,riskfreerate=riskfreerate,simMethod=simMethod\
                               )
    
    elif mainSelection=="BBChart":
        clear_cache('bollingerbands')
        symbols=tickerPortfolio.split(",")
        plot_bollingerbands(symbols,StartDate,EndDate,WindowSize1,noOfStdDev,imagecounter,'bollingerbands')
        
        hists = os.listdir('static/bollingerbands')
        hists = ['bollingerbands/' + file for file in hists]
        
        analysistype="Bollinger Bands Using SMA Period Of "+str(WindowSize1)+" Days & Bands Reflecting "+str(noOfStdDev)+" Std Dev-s"      
        
        return render_template('results.html',analysistype=analysistype, hists = hists, mainSelection=mainSelection,\
                               tickerPortfolio=tickerPortfolio,portfolioWeights=portfolioWeights,\
                               portfolioValue=portfolioValue,StartDate=StartDate,EndDate=EndDate,ForecastDate=ForecastDate,\
                               BacktestDays=BacktestDays,NoOfIter=NoOfIter,PercentileRange=PercentileRange,\
                               averageType=averageType,WindowSize1=WindowSize1,WindowSize2=WindowSize2,WindowSize3=WindowSize3,riskfreerate=riskfreerate,simMethod=simMethod\
                               )
    
    elif mainSelection=="MovingAve":
        clear_cache('movingaverage')
        symbols=tickerPortfolio.split(",")
        portfolioWeights_func=portfolioWeights.split(",")
        portfolioWeights_func=[float(x) * 0.01 for x in portfolioWeights_func]
                
        T=np.busday_count(EndDate,ForecastDate)+BacktestDays
        N=T+1
        
        plotpiechart(symbols,portfolioWeights_func,imagecounter,"movingaverage")

        resultantDF, RMSE_DF, share_split_table=movingaverageforecast(StartDate,EndDate,BacktestDays,symbols,portfolioWeights_func,\
                                                   portfolioValue,T,N,averageType,WindowSize1,imagecounter,"movingaverage")        
        
        tableOne=RMSE_DF
        
        tailsDF=pd.DataFrame()
        for i in range(0,len(resultantDF)):
            tailsDF=pd.concat([tailsDF,resultantDF[i].tail(60)],axis=1)
        tableTwo=tailsDF
        tableThree=share_split_table
        
        hists = os.listdir('static/movingaverage')
        hists = ['movingaverage/' + file for file in hists]
        
        analysistype="Moving Average Forecast Using "+averageType+" With Period Of "+str(WindowSize1)+" Days"      
        
        return render_template('results.html',analysistype=analysistype, hists = hists, mainSelection=mainSelection,\
                               tickerPortfolio=tickerPortfolio,portfolioWeights=portfolioWeights,\
                               portfolioValue=portfolioValue,StartDate=StartDate,EndDate=EndDate,ForecastDate=ForecastDate,\
                               BacktestDays=BacktestDays,NoOfIter=NoOfIter,PercentileRange=PercentileRange,\
                               averageType=averageType,WindowSize1=WindowSize1,WindowSize2=WindowSize2,WindowSize3=WindowSize3,riskfreerate=riskfreerate,simMethod=simMethod,\
                              tableOne=[tableOne.to_html(classes='data', header="true")],\
                              tableTwo=[tableTwo.to_html(classes='data', header="true")],\
                              tableThree=[tableThree.to_html(classes='data', header="true")])

    
    elif mainSelection=="ARIMAGARCH":
        clear_cache('arimagarch')
        symbols=tickerPortfolio.split(",")
        portfolioWeights_func=portfolioWeights.split(",")
        portfolioWeights_func=[float(x) * 0.01 for x in portfolioWeights_func]
                
        T=np.busday_count(EndDate,ForecastDate)+BacktestDays
        N=T+1
        
        plotpiechart(symbols,portfolioWeights_func,imagecounter,"arimagarch")
        
        resultantDF, RMSE_DF, share_split_table =UnivarArimaGarchPredict(StartDate,EndDate,BacktestDays,\
                                                      symbols,portfolioWeights_func,portfolioValue,T,N,imagecounter,"arimagarch")
        
        
        tableOne=RMSE_DF
        tailsDF=pd.DataFrame()
        for i in range(0,len(resultantDF)):
            tailsDF=pd.concat([tailsDF,resultantDF[i].tail(60)],axis=1)
        tableTwo=tailsDF
        tableThree=share_split_table
        
        hists = os.listdir('static/arimagarch')
        hists = ['arimagarch/' + file for file in hists]
        
        analysistype="Univariate Auto-ARIMA + GARCH(1,1) Forecast"      
        
        return render_template('results.html',analysistype=analysistype, hists = hists, mainSelection=mainSelection,\
                               tickerPortfolio=tickerPortfolio,portfolioWeights=portfolioWeights,\
                               portfolioValue=portfolioValue,StartDate=StartDate,EndDate=EndDate,ForecastDate=ForecastDate,\
                               BacktestDays=BacktestDays,NoOfIter=NoOfIter,PercentileRange=PercentileRange,\
                               averageType=averageType,WindowSize1=WindowSize1,WindowSize2=WindowSize2,WindowSize3=WindowSize3,riskfreerate=riskfreerate,simMethod=simMethod,\
                              tableOne=[tableOne.to_html(classes='data', header="true")],\
                              tableTwo=[tableTwo.to_html(classes='data', header="true")],\
                              tableThree=[tableThree.to_html(classes='data', header="true")])


    elif mainSelection=="HoltWinters":
        clear_cache('holtwinters')
        symbols=tickerPortfolio.split(",")
        portfolioWeights_func=portfolioWeights.split(",")
        portfolioWeights_func=[float(x) * 0.01 for x in portfolioWeights_func]
                
        T=np.busday_count(EndDate,ForecastDate)+BacktestDays
        N=T+1
        
        plotpiechart(symbols,portfolioWeights_func,imagecounter,"holtwinters")  
        
        FinalOutput,FinalOutputTailTable,BestConfigSummaryTable,OptimalRMSESummaryTable,OverallRMSEforConfigTable  =AutoHoltWinters(StartDate,EndDate,ForecastDate,BacktestDays,\
                                                        symbols,portfolioWeights_func,portfolioValue,seasonality,imagecounter,'holtwinters')
        
        tableOne=OptimalRMSESummaryTable
        tableTwo=FinalOutputTailTable
        tableThree=BestConfigSummaryTable
        
        hists = os.listdir('static/holtwinters')
        hists = ['holtwinters/' + file for file in hists]
        
        analysistype="Triple Exponential Smoothing"      
        
        return render_template('results.html',analysistype=analysistype, hists = hists, mainSelection=mainSelection,\
                               tickerPortfolio=tickerPortfolio,portfolioWeights=portfolioWeights,\
                               portfolioValue=portfolioValue,StartDate=StartDate,EndDate=EndDate,ForecastDate=ForecastDate,\
                               BacktestDays=BacktestDays,NoOfIter=NoOfIter,PercentileRange=PercentileRange,\
                               averageType=averageType,WindowSize1=WindowSize1,WindowSize2=WindowSize2,WindowSize3=WindowSize3,riskfreerate=riskfreerate,simMethod=simMethod,\
                              tableOne=[tableOne.to_html(classes='data', header="true")],\
                              tableTwo=[tableTwo.to_html(classes='data', header="true")],\
                              tableThree=[tableThree.to_html(classes='data', header="true")])
    
    
    elif mainSelection=="VecAR":
        clear_cache('vectorautoreg')
        symbols=tickerPortfolio.split(",")
        portfolioWeights_func=portfolioWeights.split(",")
        portfolioWeights_func=[float(x) * 0.01 for x in portfolioWeights_func]
        
        plotpiechart(symbols,portfolioWeights_func,imagecounter,"vectorautoreg")
        
        stockprices,RMSE_DF,DWTable,GrangersMatrix,JohansonCointTable=vectorAutoRegression(StartDate,EndDate,\
                        symbols,portfolioWeights_func,portfolioValue,ForecastDate,BacktestDays,imagecounter,"vectorautoreg")


        
        tableOne=RMSE_DF
        tableTwo=stockprices.tail(60)
        tableThree=GrangersMatrix
        tableFour=JohansonCointTable
        tableFive=DWTable
        
        hists = os.listdir('static/vectorautoreg')
        hists = ['vectorautoreg/' + file for file in hists]
        
        if len(symbols)>1:
            analysistype="Multivariate Vector Auto Regression Forecast"
        elif len(symbols)==1:
            analysistype="ERROR ! Vector Auto Regression Forecast Requires Multi Variate Time Series Data"
        
        return render_template('resultsVecar.html',analysistype=analysistype, hists = hists, mainSelection=mainSelection,\
                               tickerPortfolio=tickerPortfolio,portfolioWeights=portfolioWeights,\
                               portfolioValue=portfolioValue,StartDate=StartDate,EndDate=EndDate,ForecastDate=ForecastDate,\
                               BacktestDays=BacktestDays,NoOfIter=NoOfIter,PercentileRange=PercentileRange,\
                               averageType=averageType,WindowSize1=WindowSize1,WindowSize2=WindowSize2,WindowSize3=WindowSize3,riskfreerate=riskfreerate,simMethod=simMethod,\
                              tableOne=[tableOne.to_html(classes='data', header="true")],\
                              tableTwo=[tableTwo.to_html(classes='data', header="true")],\
                              tableThree=[tableThree.to_html(classes='data', header="true")],\
                              tableFour=[tableFour.to_html(classes='data', header="true")],\
                              tableFive=[tableFive.to_html(classes='data', header="true")])
 
        
    elif mainSelection=="returnsDist":
        clear_cache('analysis')
        symbols=tickerPortfolio.split(",")
        portfolioWeights_func=portfolioWeights.split(",")
        portfolioWeights_func=[float(x) * 0.01 for x in portfolioWeights_func]
        
        dfprices, noOfShares, share_split_table = extract_prices(StartDate,EndDate,symbols,portfolioWeights_func,portfolioValue)
        
        dfreturns ,df_mean_stdev=calc_returns(dfprices,symbols)
        
        plotpiechart(symbols,portfolioWeights_func,imagecounter,"analysis")
        plotprices(dfprices,symbols,imagecounter,"analysis")
        plotreturns(dfreturns,imagecounter,"analysis")
        KSTestResultsDF, ShapiroWilkTestResultsDF , Kurtosis_Skew, ADFTestResultsDF = fit_test_normal(dfreturns,symbols,imagecounter,"analysis")
        StartMidEndDF =compareStartMidEnd(dfreturns,df_mean_stdev)
        
        hists = os.listdir('static/analysis')
        hists = ['analysis/' + file for file in hists]
        
        tableOne=df_mean_stdev
        tableTwo=KSTestResultsDF
        tableThree=Kurtosis_Skew
        tableFour=dfreturns.corr()
        tableFive=ADFTestResultsDF
        tableSix=StartMidEndDF
        analysistype="Detailed Analysis Of Log Returns"      
        
        
        return render_template('resultsLogreturns.html',analysistype=analysistype, hists = hists, mainSelection=mainSelection,\
                               tickerPortfolio=tickerPortfolio,portfolioWeights=portfolioWeights,\
                               portfolioValue=portfolioValue,StartDate=StartDate,EndDate=EndDate,ForecastDate=ForecastDate,\
                               BacktestDays=BacktestDays,NoOfIter=NoOfIter,PercentileRange=PercentileRange,\
                               averageType=averageType,WindowSize1=WindowSize1,WindowSize2=WindowSize2,WindowSize3=WindowSize3,riskfreerate=riskfreerate,simMethod=simMethod,\
                              tableOne=[tableOne.to_html(classes='data', header="true")],\
                              tableTwo=[tableTwo.to_html(classes='data', header="true")],\
                              tableThree=[tableThree.to_html(classes='data', header="true")],\
                              tableFour=[tableFour.to_html(classes='data', header="true")],\
                              tableFive=[tableFive.to_html(classes='data', header="true")],\
                              tableSix=[tableSix.to_html(classes='data', header="true")])
        
    elif mainSelection=="GBM":
        clear_cache('gbm_bootstrap')
        symbols=tickerPortfolio.split(",")
        portfolioWeights_func=portfolioWeights.split(",")
        portfolioWeights_func=[float(x) * 0.01 for x in portfolioWeights_func]
        
        plotpiechart(symbols,portfolioWeights_func,imagecounter,"gbm_bootstrap")
        
        T=np.busday_count(EndDate,ForecastDate)+BacktestDays
        N=T+1
        
        final, share_split_tableFULL , dfreturns , df_mean_stdev , ReturnsAtForecastEndDate, dfprices = MonteCarlo_GBM(StartDate,EndDate,BacktestDays,PercentileRange,symbols,\
                       portfolioWeights_func,portfolioValue,T,N,NoOfIter,imagecounter,"gbm_bootstrap")
        
        if BacktestDays>0:
            tableOne=calculateRMSE(final,T,BacktestDays,dfprices)
            
        elif BacktestDays<=0:
            tableOne=pd.DataFrame()
                    
        hists = os.listdir('static/gbm_bootstrap')
        hists = ['gbm_bootstrap/' + file for file in hists]
        
        GBMtail=pd.concat([final[final.columns[0:(len(symbols)+1)]].tail(60),final[final.columns[-3*(len(symbols)+1):]].tail(60)],axis=1)
        GBMtail=GBMtail.round(2) 
        
        tableTwo=GBMtail
        tableThree=ReturnsAtForecastEndDate
        tableFour=dfreturns.corr()
        tableFive=share_split_tableFULL
        
        analysistype="Forecast Of Future Price Values For Portfolio Based On Geometric Brownian Motion"
        
        return render_template('results.html',analysistype=analysistype, hists = hists, mainSelection=mainSelection,\
                               tickerPortfolio=tickerPortfolio,portfolioWeights=portfolioWeights,\
                               portfolioValue=portfolioValue,StartDate=StartDate,EndDate=EndDate,ForecastDate=ForecastDate,\
                               BacktestDays=BacktestDays,NoOfIter=NoOfIter,PercentileRange=PercentileRange,\
                               averageType=averageType,WindowSize1=WindowSize1,WindowSize2=WindowSize2,WindowSize3=WindowSize3,riskfreerate=riskfreerate,simMethod=simMethod,\
                              tableOne=[tableOne.to_html(classes='data', header="true")],\
                              tableTwo=[tableTwo.to_html(classes='data', header="true")],\
                              tableThree=[tableThree.to_html(classes='data', header="true")],\
                              tableFour=[tableFour.to_html(classes='data', header="true")],\
                              tableFive=[tableFive.to_html(classes='data', header="true")])
    
   
    elif mainSelection=="Bootstrap":
        clear_cache('gbm_bootstrap')
        symbols=tickerPortfolio.split(",")
        portfolioWeights_func=portfolioWeights.split(",")
        portfolioWeights_func=[float(x) * 0.01 for x in portfolioWeights_func]
        
        plotpiechart(symbols,portfolioWeights_func,imagecounter,"gbm_bootstrap")
                
        T=np.busday_count(EndDate,ForecastDate)+BacktestDays
        N=T+1
        
        final, share_split_tableFULL , dfreturns , df_mean_stdev , ReturnsAtForecastEndDate,dfprices = MonteCarlo_Bootstrap(StartDate,EndDate,BacktestDays,PercentileRange,symbols,\
                       portfolioWeights_func,portfolioValue,T,N,NoOfIter,imagecounter,"gbm_bootstrap")
        
        if BacktestDays>0:
            tableOne=calculateRMSE(final,T,BacktestDays,dfprices)
            
        elif BacktestDays<=0:
            tableOne=pd.DataFrame()
            
        hists = os.listdir('static/gbm_bootstrap')
        hists = ['gbm_bootstrap/' + file for file in hists]
        
        Bootstraptail=pd.concat([final[final.columns[0:(len(symbols)+1)]].tail(60),final[final.columns[-3*(len(symbols)+1):]].tail(60)],axis=1)
        Bootstraptail=Bootstraptail.round(2) 
        
        tableTwo=Bootstraptail
        tableThree=ReturnsAtForecastEndDate
        tableFour=dfreturns.corr()
        tableFive=share_split_tableFULL
        
        analysistype="Forecast Of Future Price Values For Portfolio Based On Bootstrap Sampling"
        
        return render_template('results.html',analysistype=analysistype, hists = hists, mainSelection=mainSelection,\
                               tickerPortfolio=tickerPortfolio,portfolioWeights=portfolioWeights,\
                               portfolioValue=portfolioValue,StartDate=StartDate,EndDate=EndDate,ForecastDate=ForecastDate,\
                               BacktestDays=BacktestDays,NoOfIter=NoOfIter,PercentileRange=PercentileRange,\
                               averageType=averageType,WindowSize1=WindowSize1,WindowSize2=WindowSize2,WindowSize3=WindowSize3,riskfreerate=riskfreerate,simMethod=simMethod,\
                              tableOne=[tableOne.to_html(classes='data', header="true")],\
                              tableTwo=[tableTwo.to_html(classes='data', header="true")],\
                              tableThree=[tableThree.to_html(classes='data', header="true")],\
                              tableFour=[tableFour.to_html(classes='data', header="true")],\
                              tableFive=[tableFive.to_html(classes='data', header="true")])

    elif mainSelection=="EfficientFrontierHist":
        clear_cache('efficientportfolio')
        symbols=tickerPortfolio.split(",")
        
        T=np.busday_count(EndDate,ForecastDate)+0
        N=T+1
                
        FinalResultsTable,SharpeDetail,SortinoDetail=EfficientPortfolioHistorical(StartDate,EndDate,symbols,portfolioValue,NoOfIter,riskfreerate,imagecounter,"efficientportfolio")
        
        HistSharpeWeights=FinalResultsTable.nlargest(1,['Sharpe Ratio'])['Weights'].tolist()

        HistSharpe_dfprices, HistSharpe_noOfShares, HistSharpe_share_split_table = extract_prices(StartDate,EndDate,symbols,HistSharpeWeights[0],portfolioValue)

        HistSharpe_dfreturns ,HistSharpe_df_mean_stdev=calc_returns(HistSharpe_dfprices,symbols)

        HistSharpe_df_mean_stdev["Mean Daily Return"]=np.exp(HistSharpe_df_mean_stdev["Mean Log Daily Return"])-1
        HistSharpe_df_mean_stdev["StdDev Daily Return"]=np.exp(HistSharpe_df_mean_stdev["StdDev Log Daily Return"])-1
        
        tableTwo=FinalResultsTable.nlargest(1,['Sharpe Ratio'])
        SharpeWeights=FinalResultsTable.nlargest(1,['Sharpe Ratio'])['Weights'].tolist()
        print(SharpeWeights[0])
      
        labels = symbols
        sizes = SharpeWeights[0]
        fig1, ax1 = plt.subplots()
        ax1.pie(SharpeWeights[0], labels=symbols, autopct='%1.1f%%',
            shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title("Optimal Portfolio Weights Based On Sharpe Ratio")
        plt.savefig(f'static/efficientportfolio/{imagecounter}_01SharpePortfolioweights.png')
            
        tableThree=FinalResultsTable.nlargest(1,['Sortino Ratio'])
        SortinoWeights=FinalResultsTable.nlargest(1,['Sortino Ratio'])['Weights'].tolist()
        print(SortinoWeights[0])
        
        sizes = SortinoWeights[0]
        fig1, ax1 = plt.subplots()
        ax1.pie(SortinoWeights[0], labels=symbols, autopct='%1.1f%%',
            shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title("Optimal Portfolio Weights Based On Sortino Ratio")
        plt.savefig(f'static/efficientportfolio/{imagecounter}_01SortinoPortfolioweights.png')
        
        tableFour=HistSharpe_df_mean_stdev
        
        hists = os.listdir('static/efficientportfolio')
        hists = ['efficientportfolio/' + file for file in hists]
        
        analysistype="Efficient Portfolio Weights Via Mean-Variance Analysis" 
        subheading="For Historical Period From "+StartDate+" To "+EndDate+" For Simulation Of "+str(NoOfIter)+" Iterations"
        
        return render_template('results.html',SharpeDetail=SharpeDetail,SortinoDetail=SortinoDetail,subheading=subheading,analysistype=analysistype, hists = hists, mainSelection=mainSelection,\
                               tickerPortfolio=tickerPortfolio,portfolioWeights=portfolioWeights,\
                               portfolioValue=portfolioValue,StartDate=StartDate,EndDate=EndDate,ForecastDate=ForecastDate,\
                               BacktestDays=BacktestDays,NoOfIter=NoOfIter,PercentileRange=PercentileRange,\
                               averageType=averageType,WindowSize1=WindowSize1,WindowSize2=WindowSize2,WindowSize3=WindowSize3,riskfreerate=riskfreerate,simMethod=simMethod,\
                              tableTwo=[tableTwo.to_html(classes='data', header="true")],\
                              tableThree=[tableThree.to_html(classes='data', header="true")],\
                              tableFour=[tableFour.to_html(classes='data', header="true")])    
    
    
    elif mainSelection=="EfficientFrontier":
        clear_cache('efficientportfolio')
        symbols=tickerPortfolio.split(",")
        
        T=np.busday_count(EndDate,ForecastDate)+0
        N=T+1
        
        NoOfIterationsInnerLoop=10 #Fixed to 10 to avoid time out error as algol is very time consuming
                
        FutureFinalResultsTable,FutureSharpeDetail,FutureSortinoDetail =EfficientPortfolioFuture(StartDate,EndDate,symbols,\
                                                                                                 portfolioValue,T,N,NoOfIter,\
                                                                                                 NoOfIterationsInnerLoop,\
                                                                                                 riskfreerate,simMethod,\
                                                                                                 imagecounter,\
                                                                                                 "efficientportfolio")
        
        FutureSharpeRatio_Best=pd.DataFrame(FutureFinalResultsTable.nlargest(1,['Sharpe Ratio'])["Components Log Returns Mean"].values[0],index=symbols,columns=["Mean Log Daily Return"])
        FutureSharpeRatio_Best["StdDev Log Daily Return"]=FutureFinalResultsTable.nlargest(1,['Sharpe Ratio'])["Components Log Returns Std Dev"].values[0]
        SharpeRatioFutureMEAN=FutureFinalResultsTable.nlargest(1,['Sharpe Ratio'])["Log Returns Mean"].values[0]
        SharpeRatioFutureSTDEV=FutureFinalResultsTable.nlargest(1,['Sharpe Ratio'])["Log Returns Std Dev"].values[0]

        FutureSharpeRatio_Best=FutureSharpeRatio_Best.append(pd.DataFrame([[SharpeRatioFutureMEAN,SharpeRatioFutureSTDEV]],index=["Portfolio"],columns=FutureSharpeRatio_Best.columns))

        FutureSharpeRatio_Best["Mean Daily Return"]=np.exp(FutureSharpeRatio_Best["Mean Log Daily Return"])-1
        FutureSharpeRatio_Best["StdDev Daily Return"]=np.exp(FutureSharpeRatio_Best["StdDev Log Daily Return"])-1

        
        tableTwo=FutureFinalResultsTable.nlargest(1,['Sharpe Ratio'])
        SharpeWeights=FutureFinalResultsTable.nlargest(1,['Sharpe Ratio'])['Weights'].tolist()
        print(SharpeWeights[0])
      
        labels = symbols
        sizes = SharpeWeights[0]
        fig1, ax1 = plt.subplots()
        ax1.pie(SharpeWeights[0], labels=symbols, autopct='%1.1f%%',
            shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title("Optimal Portfolio Weights Based On Sharpe Ratio")
        plt.savefig(f'static/efficientportfolio/{imagecounter}_01SharpePortfolioweights.png')
            
        tableThree=FutureFinalResultsTable.nlargest(1,['Sortino Ratio'])
        SortinoWeights=FutureFinalResultsTable.nlargest(1,['Sortino Ratio'])['Weights'].tolist()
        print(SortinoWeights[0])
        
        sizes = SortinoWeights[0]
        fig1, ax1 = plt.subplots()
        ax1.pie(SortinoWeights[0], labels=symbols, autopct='%1.1f%%',
            shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title("Optimal Portfolio Weights Based On Sortino Ratio")
        plt.savefig(f'static/efficientportfolio/{imagecounter}_01SortinoPortfolioweights.png')
        
        tableFour=FutureSharpeRatio_Best
        
        hists = os.listdir('static/efficientportfolio')
        hists = ['efficientportfolio/' + file for file in hists]
        
        analysistype="Efficient Portfolio Weights Via Mean-Variance Analysis" 
        subheading="For Forecast Period Of "+str(N)+" Days From "+EndDate+" Till "+ForecastDate+" Via "+simMethod+" Simulation Over "+str(NoOfIter)+" Iterations"
        
        return render_template('results.html',SharpeDetail=FutureSharpeDetail,SortinoDetail=FutureSortinoDetail,subheading=subheading,analysistype=analysistype, hists = hists, mainSelection=mainSelection,\
                               tickerPortfolio=tickerPortfolio,portfolioWeights=portfolioWeights,\
                               portfolioValue=portfolioValue,StartDate=StartDate,EndDate=EndDate,ForecastDate=ForecastDate,\
                               BacktestDays=BacktestDays,NoOfIter=NoOfIter,PercentileRange=PercentileRange,\
                               averageType=averageType,WindowSize1=WindowSize1,WindowSize2=WindowSize2,WindowSize3=WindowSize3,riskfreerate=riskfreerate,simMethod=simMethod,\
                              tableTwo=[tableTwo.to_html(classes='data', header="true")],\
                              tableThree=[tableThree.to_html(classes='data', header="true")],\
                              tableFour=[tableFour.to_html(classes='data', header="true")])  
    
    elif mainSelection=="factorAnalysis":
        clear_cache('factoranalysis')
        symbols=tickerPortfolio.split(",")
        #https://stackoverflow.com/questions/53344797/how-create-an-array-with-checkboxes-in-flask
        data = request.form.to_dict(flat=False)
        chosen=data.get('chosen[]')
        hiOrLo=data.get('hiOrLo[]')
        hiOrLo=list(filter(lambda a: a != 'NA', hiOrLo))
        factorweight=data.get('factorweight[]')
        factorweight=list(filter(lambda a: a != '0.00' and a != '0.0' and a!='0', factorweight))
        factorweight = [float(i)/100 for i in factorweight] 
        print(symbols)
        print(chosen)    
        print(hiOrLo)
        print(factorweight)
        checkArrayLengths=[len(chosen),len(hiOrLo),len(factorweight)]
        series = pd.Series(checkArrayLengths)

        errorMessage=""
        if sum(factorweight)!=1 or series.nunique() > 1:
            if sum(factorweight)!=1:
                errorMessage='ERROR ! : Factor Weights Do Not Sum'
                NormMethod=""
                quartileMessage=""
                cronbachalpha=""
                correlcoeff=""
                missingCount=""
                topQuartileListing=list(["NA"])
                tableOne=pd.DataFrame()
                tableTwo=pd.DataFrame()
                tableThree=pd.DataFrame()
                tableFour=pd.DataFrame()
                tableFive=pd.DataFrame()
                hists = os.listdir('static/factoranalysis')
                hists = ['factoranalysis/' + file for file in hists]

                return render_template('results_factoranalysis.html',errorMessage=errorMessage,hists=hists,cronbachalpha=cronbachalpha,\
                                  correlcoeff = correlcoeff,missingCount=missingCount,\
                                  NormMethod=NormMethod,quartileMessage=quartileMessage,\
                                  topQuartileListing=topQuartileListing,\
                                  tableOne=[tableOne.to_html(classes='data', header="true")],\
                                  tableTwo=[tableTwo.to_html(classes='data', header="true")],\
                                  tableThree=[tableThree.to_html(classes='data', header="true")],\
                                  tableFour=[tableFour.to_html(classes='data', header="true")],\
                                  tableFive=[tableFive.to_html(classes='data', header="true")])


            if series.nunique() > 1:
                errorMessage='Error ! : Array Lengths Mismatched - check if No of Factors selected, Hi/Lo and Weights are consistent'
                NormMethod=""
                quartileMessage=""
                cronbachalpha=""
                correlcoeff=""
                missingCount=""
                topQuartileListing=list(["NA"])
                tableOne=pd.DataFrame()
                tableTwo=pd.DataFrame()
                tableThree=pd.DataFrame()
                tableFour=pd.DataFrame()
                tableFive=pd.DataFrame()
                hists = os.listdir('static/factoranalysis')
                hists = ['factoranalysis/' + file for file in hists]

                return render_template('results_factoranalysis.html',errorMessage=errorMessage,hists=hists,cronbachalpha=cronbachalpha,\
                                  correlcoeff = correlcoeff,missingCount=missingCount,\
                                  NormMethod=NormMethod,quartileMessage=quartileMessage,\
                                  topQuartileListing=topQuartileListing,\
                                  tableOne=[tableOne.to_html(classes='data', header="true")],\
                                  tableTwo=[tableTwo.to_html(classes='data', header="true")],\
                                  tableThree=[tableThree.to_html(classes='data', header="true")],\
                                  tableFour=[tableFour.to_html(classes='data', header="true")],\
                                  tableFive=[tableFive.to_html(classes='data', header="true")])

        else:

            df_statistics = {}

            for symbol in symbols:
                try:
                    df_statistics[symbol] = get_key_stats(r'https://finance.yahoo.com/quote/' + symbol + '/key-statistics?p=' + symbol)
                    print('Extracted Stats For '+symbol)
                    time.sleep(random.randint(1,2))

                except:
                    print("Unable To Extract Statistics For ",symbol)    

            current = datetime.now() - timedelta(1) #current date set a day before as data may not be avail for today

            currentMinus1Yr=datetime(year=current.year-1, month=current.month, day=current.day)

            stockprice_df=getStockReturns(symbols,current,currentMinus1Yr)

            #display(stockprice_df)

            listOfMetrics=chosen
            select_order_by_metric=hiOrLo
            weights_by_metric=factorweight
            NormMethod=request.form['NormMethod']

            summary_method,raw_df,ranking_df,ranking_df_w_quartiles,cronbachalpha,correlcoeff=rankstocks(symbols,df_statistics,listOfMetrics,\
                                                                                          select_order_by_metric,\
                                                                                          weights_by_metric,\
                                                                                          NormMethod,\
                                                                                          stockprice_df)

            correlation(ranking_df,listOfMetrics, len(listOfMetrics),1,imagecounter)

            countNA=ranking_df.shape[0] - ranking_df.dropna().shape[0]
            if countNA!=0:
                quartileMessage="Note: There are "+str(countNA)+" stocks out of the total "+str(len(ranking_df))+" scored stocks without Stock Price info which were ignored in the aggregation"
            elif countNA==0:
                quartileMessage=""




            ranking_df_w_quartiles.plot.bar(figsize=(15,5))
            plt.title("Median Current Annual y.o.y Stock Price Returns By Quartile")
            plt.savefig(f'static/factoranalysis/{imagecounter}_01_Quartiles.png')    

            ranking_df.plot.scatter(x='Composite Score',y='Stock Price Returns',c='DarkBlue',figsize=(10,5))
            plt.title("Correlation Between Composite Score vs Current Annual y.o.y Stock Price Returns:"+str(round(correlcoeff,3)))
            plt.savefig(f'static/factoranalysis/{imagecounter}_03_CorrelOverall.png')

            tableOne=summary_method
            tableTwo=raw_df
            tableThree=ranking_df
            tableFour=pd.DataFrame(ranking_df.corr()['Stock Price Returns'].loc[listOfMetrics])
            tableFive=ranking_df_w_quartiles
            topQuartileListing=list(ranking_df.index[0:int(len(ranking_df.index)/4)])
            missingCount=len(raw_df)-len(ranking_df)
            hists = os.listdir('static/factoranalysis')
            hists = ['factoranalysis/' + file for file in hists]

            #return jsonify(data)

            return render_template('results_factoranalysis.html',errorMessage=errorMessage,hists=hists,cronbachalpha=round(cronbachalpha,3),\
                                      correlcoeff = round(correlcoeff,3),missingCount=missingCount,\
                                      NormMethod=NormMethod,quartileMessage=quartileMessage,\
                                      topQuartileListing=topQuartileListing,\
                                      tableOne=[tableOne.to_html(classes='data', header="true")],\
                                      tableTwo=[tableTwo.to_html(classes='data', header="true")],\
                                      tableThree=[tableThree.to_html(classes='data', header="true")],\
                                      tableFour=[tableFour.to_html(classes='data', header="true")],\
                                      tableFive=[tableFive.to_html(classes='data', header="true")])    
            
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)