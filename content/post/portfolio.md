+++
categories = []
date = "2016-03-01T20:55:56+03:00"
description = "n a previous post, we naively selected growth companies and constructed a uniform-weigh portfolio out of them. In this post, we are going to use the same list of companies to construct a minimum-variance portfolio based on Harry Markowitz portfolio theory."
keywords = []
title = "Efficient Frontier with Python"

+++

In a previous post, we naively selected growth companies and constructed a uniform-weigh portfolio out of them. In this post, we are going to use the same list of companies to construct a minimum-vaiance portfolios based on Harry Markowitz's ['Portfolio Selection' paper](http://www.efalken.com/LowVolClassics/markowitz_JF1952.pdf) published 1952. Portfolio theory in a nutshell is finding the optimal wights that maximizes the return given a level of risk (variance or standard deviation) or the other way around, minimize the risk given an expected return. We are only going to need some matrix algebra and quadratic programming to explain the mathematics behind the theory. Dr. Eric Zivot's [notes](http://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf) provides a great detailed explanation using matrix algebra and most below is taken from his notes.

Let's say we have N companies and $x$ is a vector of initial random weighs for the portfolio, `$\mu$` is a vector of the average returns and `${\sigma  }^{ 2 } $` is an `$N\times N$` covariance matrix where:

<div>$$\mu =\begin{bmatrix} { \mu  }_{ 1 } \\ { \mu  }_{ 2 } \\ \vdots  \\ { \mu  }_{ N } \end{bmatrix}\quad x=\begin{bmatrix} { x }_{ 1 } \\ { x }_{ 2 } \\ \vdots  \\ { x }_{ N } \end{bmatrix}\quad \Sigma  =\begin{bmatrix} { \sigma  }_{ 11 } & { \sigma  }_{ 12 } & \dots  & { \sigma  }_{ 1N } \\ { \sigma  }_{ 21 } & { \sigma  }_{ 22 } & \cdots  & { \sigma  }_{ 11 } \\ \vdots  & \vdots  & \ddots  & \vdots  \\ { \sigma  }_{ N1 } & { \sigma  }_{ 11 } & \dots  & { \sigma  }_{ NN } \end{bmatrix}$$</div>

The portfolio return $R$ and variance `${\sigma  }^{ 2 }$` are calculated by:

<div>$$R = x'\mu$$</div>

<div>$$\sigma = x'\Sigma x $$</div>

The main problem here is finding the optimal weigh vector `$x$` that satisfies either one of the conditions below depending on whether the investor requires a minimum level of risk or maximum return:

<div>$$\underset { x }{ min } \quad { \sigma  }^{ 2 }=x'\Sigma x\quad s.t.\quad \mu =x'{ \mu  }_{ required }\quad and\quad x'1=1$$</div>

or

<div>$$\underset { x }{ max } \quad R =x'{ \mu  }\quad s.t.\quad \Sigma =x'{ \Sigma  }_{ required }x\quad and\quad x'1=1$$</div>

Solving either of them will give a portfolio that's on the efficient frontier which is, according to [investopedia](http://www.investopedia.com/terms/e/efficientfrontier.asp) explanation, a set of optimal portfolios that offers the highest expected return for a defined level of risk or the lowest risk for a given level of expected return. And here is where we're going to start working on python (repo on [github](https://github.com/ahmedas91/efficient_frontier)). We're going to draw all the possible portfolios that satisfies the conditions above. The examples [here](https://plot.ly/ipython-notebooks/markowitz-portfolio-optimization/) and [here](https://wellecks.wordpress.com/2014/03/23/portfolio-optimization-with-python/) helped me a lot writing the code. Actually, they provides better explanation of the codes than me. The hardest part was trying to solve the optimization problem using `cvxopt` library. 

```python
#import required libraries
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import cufflinks as cf
init_notebook_mode()
import cvxopt as opt
from cvxopt import solvers
```
First let's import the daily prices of the companies using the same period from the previous post from 1/1/2013 to 12/31/2015.

```python
data = pd.read_csv("TOP_COMP_2012.csv")
start = datetime.datetime(2013, 1, 1)
end = datetime.datetime(2015, 12, 31)
tickers = list(data['ticker'])
```
```python
ls_key = 'Adj Close'
f = web.DataReader(tickers, 'yahoo',start,end)
cleanData = f.ix[ls_key]
stock_data = pd.DataFrame(cleanData)
```
Now we'll calculate the monthly returns of the stocks. Fortunately pandas library has a function resumaple that can perform operations on a group of specified periods (in our case by month).

```python
monthly_returns = stock_data.resample('BM', how=lambda x: (x[-1]/x[0])-1)
```
To get beautiful graphs, we are only going to choose five random companies from the list. 

```python
companies = list(monthly_returns.sample(5,axis=1, random_state=4).columns)
companies #prints ['TECH', 'WWD', 'CMPR', 'GIS', 'EW']
```
```python
stock_data[companies].iplot(yTitle='Daily Price',dimensions=(950, 500))
```
<iframe width="900" height="400" frameborder="0" scrolling="no" src="https://plot.ly/~ahmedas91/8.embed"></iframe>

```python
monthly_returns[companies].iplot(yTitle='Monthly Returns',dimensions=(950, 500))
```
<iframe width="900" height="400" frameborder="0" scrolling="no" src="https://plot.ly/~ahmedas91/10.embed"></iframe>

```python
#Function to get evctor $x$ of random portfolio weighs that sums to 1:
def random_wieghts(n):
    a = np.random.rand(n)
    return a/a.sum()
```
```python
def initial_portfolio(monthly_returns):
    #monthly_returns = data.resample('BM', how=lambda x: (x[-1]/x[0])-1)
    
    cov = np.matrix(monthly_returns.cov())
    expected_returns = np.matrix(monthly_returns.mean())
    wieghs = np.matrix(random_wieghts(expected_returns.shape[1]))
    
    mu = wieghs.dot(expected_returns.T)
    sigma = np.sqrt(wieghs * cov.dot(wieghs.T))
    
    return mu[0,0],sigma[0,0]
```
```python
n_portfolios = 1000
means, stds = np.column_stack([
    initial_portfolio(monthly_returns[companies]) 
    for _ in xrange(n_portfolios)
])
```
```python
iplot({
'data':[
    go.Scatter(
                x = stds,
                y = means,
                mode = 'markers',
                name = 'Random Portfolio')
        ],
'layout': go.Layout(yaxis=go.YAxis(title='Means'),
                    xaxis=go.XAxis(title='stdv'))
})
```
<iframe width="900" height="400" frameborder="0" scrolling="no" src="https://plot.ly/~ahmedas91/12.embed"></iframe>

```python
import cvxopt as opt
from cvxopt import solvers

def frontier(monthly_returns):
    cov = np.matrix(monthly_returns.cov())
    n = monthly_returns.shape[1]
    avg_ret = np.matrix(monthly_returns.mean()).T
    r_min = 0.01
    mus = []
    for i in range(120):
        r_min += 0.0001
        mus.append(r_min)
    P = opt.matrix(cov)
    q = opt.matrix(np.zeros((n, 1)))
    G = opt.matrix(np.concatenate((
                -np.transpose(np.array(avg_ret)), 
                -np.identity(n)), 0))
    A = opt.matrix(1.0, (1,n))
    b = opt.matrix(1.0)
    opt.solvers.options['show_progress'] = False
    portfolio_weights = [solvers.qp(P, q, G,
                                    opt.matrix(np.concatenate((-np.ones((1,1))*yy,
                                                               np.zeros((n,1))), 0)), 
                                    A, b)['x'] for yy in mus]
    portfolio_returns = [(np.matrix(x).T * avg_ret)[0,0] for x in portfolio_weights]
    portfolio_stdvs = [np.sqrt(np.matrix(x).T * cov.T.dot(np.matrix(x)))[0,0] for x in portfolio_weights]
    return portfolio_weights, portfolio_returns, portfolio_stdvs
```
```python
w_f, mu_f, sigma_f = frontier(monthly_returns[companies])
```
```python
iplot({
'data':[
    go.Scatter(
                x = stds,
                y = means,
                mode = 'markers',
                name = 'Random Portfolio'),
    go.Scatter(
                x = sigma_f,
                y = mu_f,
                mode = 'lines+markers',
                name = 'Efficient Frontier')
            
        ],
'layout': go.Layout(yaxis=go.YAxis(title='Means'),
                    xaxis=go.XAxis(title='stdv'))
})
```
<iframe width="900" height="400" frameborder="0" scrolling="no" src="https://plot.ly/~ahmedas91/14.embed"></iframe>

We can see from the above graph that the efficient frontier take a shape of a parabola, and it represent all the efficient portfolios that can be constructed using the given companies. However, this is not it. We can also add a risk-free asset and find the Tangency Portfolio. But we'll leave it for another post. 






References:

1. http://www.efalken.com/LowVolClassics/markowitz_JF1952.pdf

2. http://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf

3. https://plot.ly/ipython-notebooks/markowitz-portfolio-optimization/

4. https://wellecks.wordpress.com/2014/03/23/portfolio-optimization-with-python/



