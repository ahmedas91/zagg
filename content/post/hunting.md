+++
categories = []
date = "2016-06-26T20:55:56+03:00"
description = "Each year, a bunch of news sites and organizations publish lists of the most innovative companies. Can we use those lists to hunt down growth companies? To answer the question, I aggregated those lists from the sites and compared the holding period return for the listed companies from 2012 to 2015."
keywords = []
title = "Hunting Down Growth Stocks"

+++
Growth companies are those with huge potential returns and are often found in the technology sector. Here is the official definition from [investing.com](http://www.investopedia.com/terms/g/growthcompany.asp):

>A growth company is any firm whose business generates significant positive cash flows or earnings, which increase at significantly faster rates than the overall economy. 

So how can we spot those kind of companies? We can screan stocks based on annual earning growth, revenues growth, return on equity...etc. We can also look for companies developing disruptive technologies. But could we just let the experts do it for us? for free? Well this is what I am exploring in this post. 

Each year, a bunch of news sites and organizations publish lists of the most innovative companies. In this post, I aggregated those lists from the sites and compared the holding period return for the listed companies from 2012 to 2015. Not all companies in the lists were included. Only those with that are listed on either NASDAQ, NYSE and AMEX, and Trading in the stock market during the whole holding period. You can download the aggregated list from the [repo](https://github.com/ahmedas91/Hunting_growth_stocks) on github. The data are collected from the below soursec. Note that for Forves list, I could not find the whole list for 2012, only the top ten. 

- [Forbes](http://www.forbes.com/sites/samanthasharf/2012/09/05/the-ten-most-innovative-companies-in-america/#3f28c5aa23d3) 
- [Barclays](http://www.businessinsider.com/presenting-the-39-companies-that-will-win-through-innovation-2012-4?op=1)
- [Thomson Reuters](http://top100innovators.stateofinnovation.thomsonreuters.com/)
- [PWC](http://www.strategyand.pwc.com/global/home/what-we-think/innovation1000/top-innovators-spenders#/tab-2012)
- [MIT](http://www2.technologyreview.com/tr50/2012/?_ga=1.224498527.453581319.1458158445)
- [BCG](https://www.bcgperspectives.com/content/interactive/innovation_growth_most_innovative_companies_interactive_guide/)
- [Fast Company](http://www.fastcompany.com/section/most-innovative-companies-2012)

## loading the data
```python
# first we import the required libraries
import pandas as pd
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
%matplotlib inline  
```
```python
# Load the aggregated list of companies
data = pd.read_csv("TOP_COMP_2012.csv")
start = datetime.datetime(2013, 1, 1)
end = datetime.datetime(2015, 12, 31)
tickers = list(data['ticker'])
```
```python
# importing the stock prices from yahoo finance
ls_key = 'Adj Close'
f = web.DataReader(tickers, 'yahoo',start,end)
cleanData = f.ix[ls_key]
stock_data = pd.DataFrame(cleanData)
```
## Results
```python
# Calulating the holding period returns 
returns = (stock_data.iloc[-1]/stock_data.iloc[0] - 1)
returns = pd.DataFrame(returns)
returns = returns.sort_values(by = [0], ascending=False)
returns['ticker'] = list(returns.index)
returns.columns = ['HPR','ticker']
```
```python
#Average return
avg_return = returns['HPR'].mean(axis=0)*100
# percentage of positive returns
percentage_positive = len(returns[returns['HPR']>0])/float(len(returns))*100 
```
```python
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
init_notebook_mode()
iplot({
"data": [
    go.Bar(
        x=returns['ticker'],
        y=returns['HPR']
    )
        ],
'layout': go.Layout(yaxis=go.YAxis(title='Holding Period Return', tickformat='%'), 
                    autosize=False,
                    width=850,
                    height=600)
    })
```
<iframe width="950" height="450" frameborder="0" scrolling="no" src="https://plot.ly/~ahmedas91/0.embed"></iframe>

The above chart shows an astonishing results with an average return of 66%. Around 87% of the companies showed a positive holding period return. So can we conclude that we can just rely on the experts for hunting big growth companies? Let's not get our hopes up yet. Let's first check if we just invested in the S&P 500 and compare its cumulative returns with a portfolio of equal weights of the stocks above.
```pyhon
# Importing the s&p price index
ff = web.DataReader("^GSPC", 'yahoo',start,end)
s_p = pd.DataFrame(ff['Adj Close'])

# calculating the daily cumulative returns during the period
sp = pd.DataFrame([0])
portfolio = pd.DataFrame([0])
for i in range(1,len(stock_data)):
    sp_returns = (s_p.iloc[i]/s_p.iloc[0] - 1)[0]
    portfolio_returns = (stock_data.iloc[i]/stock_data.iloc[0] - 1).mean(axis=0)
    sp = sp.append([sp_returns])
    portfolio = portfolio.append([portfolio_returns])    
cum_returns = pd.concat([portfolio, sp], axis=1)
cum_returns.columns = ['Portfolio','S&P']
cum_returns = cum_returns.set_index(stock_data.index)

iplot({
'data':[
    go.Scatter(
            x = cum_returns.index,
            y = cum_returns[col],
            name = col) for col in cum_returns.columns],
'layout': go.Layout(yaxis=go.YAxis(title='Holding Period Return', tickformat='%'), 
            autosize=False,
            width=950,
            height=600)
})
```
<iframe width="950" height="450" frameborder="0" scrolling="no" src="https://plot.ly/~ahmedas91/4.embed"></iframe>

Well, we're still beating the market by about 30%. So can we really just let the experts do it for us? Maybe use their lists as preliminary screener only.

