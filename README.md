# StockAnalysisWebApp
This is a stock analysis Python Flask web app that allows you to forecast short term stock price movements using traditional statistical methods (e.g. ARIMA, HoltWinters and also GBM, etc)

The full list of features are:-

...Analysis > Candlestick Chart

...Analysis > Log Daily Returns

...Analysis > Relative Strength Index Chart (RSI) 

...Analysis > Moving Average Convergence Divergence (MACD) 

...Analysis > Bollinger Bands Chart 

...Analysis > Multi Factor Stock Screening & Ranking

...Forecast > Moving Average Forecast (Univariate)

...Forecast > Auto-ARIMA+GARCH Forecast (Univariate)

...Forecast > Auto-Holt Winters Forecast (Univariate)

...Forecast > Vector Auto Regression Forecast (Multivariate)  

...Forecast > Geometric Brownian Motion (GBM) Forecast (Multivariate)

...Forecast > Bootstrap Sampling Forecast (Multivariate)

...Forecast > Portfolio Weights For Optimal Risk-Returns

A working 'live' version of this app is hosted on Pythonanywhere https://www.stonksforecast.online

Unless otherwise stated, all calculations reference the Adjusted Closing Prices which are extracted from Yahoo Finance through Python's PandasDataReader library and processed in the backend by a Python script.

By default this web-app only plots and tabulates the last 60 "tail" values of any actual vs predicted stock prices. However the full table is available in the Jupyter Notebook iPYNB version of this file which is also in this repo

Be aware that some selections need more time for the results to be calculated (E.g. Auto-Holt Winters , Efficient Portfolio Weights)

The code used is a combination of original work and modified versions of code written by others (References directly listed in the flask_app.py file and also listed in the webapp itself) 

https://medium.com/swlh/generating-candlestick-charts-from-scratch-ef6e1d3cf0e9

https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas

https://tcoil.info/compute-bollinger-bands-for-stocks-with-python-and-pandas/

https://intellipaat.com/community/34075/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm-mean/

https://medium.com/analytics-vidhya/arima-garch-forecasting-with-python-7a3f797de3ff

https://machinelearningmastery.com/how-to-grid-search-triple-exponential-smoothing-for-time-series-forecasting-in-python/

https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python

https://towardsdatascience.com/how-to-simulate-financial-portfolios-with-python-d0dc4b52a278

Please Note: I am not a financial expert of any sort so the use of this tool for any investment decisions will be at your own risk !  
