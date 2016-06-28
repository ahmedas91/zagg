+++
categories = []
date = "2016-04-05T20:55:56+03:00"
description = "On my test post, we'll solve  Kaggle's Digit Recognizer competition using python's machine learning library `sklearn`. It a really simple problem and used as a starting point (along with the Titanic one). "
keywords = []
title = "Kaggle Digits Recognition"

+++

On my test post, we'll solve  Kaggle's [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) competition using python's machine learning library `sklearn`. It a really simple problem and used as a starting point (along with the [Titanic](https://www.kaggle.com/c/titanic) one) Kaggle competitions. 
## Exploring the data 


```python
import numpy as np

X =  np.genfromtxt('train.csv',dtype='int_', 
                   delimiter=',', skip_header=1)
x_test =  np.genfromtxt('test.csv',dtype='int_', 
                        delimiter=',', skip_header=1)
x_train = X[:,1:]
y_train = X[:,0]
```

Each row in the `x_train` and `x_test` data is a 28x28 pixels image with a total of 784 pixels. Therefore, we will write a simple function takes randomly selected rows, reshapes them into 28x28 matrices and display them using `matplotlib.image.mpimg`. 


```python
import matplotlib.pyplot as plt
%matplotlib inline

def display(n):    
    for i in range(1,(n**2)+1):
        plt.subplot(n,n,i)
        plt.axis('off')
        pic = np.reshape(x_train[np.random.randint(1,42000)],(28,28))
        imgplot = plt.imshow(pic, cmap='Greys')
display(5)
```
{{< figure src="/images/digits.png">}}

## Training the data

#### Logistic regression
```python
from sklearn.linear_model import LogisticRegressionCV
model_log = LogisticRegressionCV(multi_class='multinomial') 
model_log.fit(x_train, y_train)
```
#### Random forests
```python
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(1000)
model_rf = ran.fit(x_train,y_train)
```
#### Neural Networks
```python
import import tensorflow.contrib.learn as skflow
#Tensorflow library only accept numbers with type float
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_test = x_test.astype('float32')
model_nn = skflow.TensorFlowDNNClassifier(hidden_units=[100,50], n_classes=10, steps=100000)
model_nn.fit(x_train, y_train)
```
Now let's check the accuracy of the training data. 
```python
print 'Logistic regression', model_log.score(x_train, y_train)
print 'Random forests', model_rf.score(x_train, y_train)
print 'Neural networks', model_nn.score(x_train, y_train)
```
prints:
`Logistic regression 0.941357142857`
`Random forests 1.0`
`Neural networks 0.99328571428571433`

# Submitting the data

We're gonna do the same thing with but with the x_test data and do some data cleaning for submission. 


```python
def submit(model,x_test, name):
    # predict y using the test set
    y_test = model.predict(x_test) 
    # kaggles requires the submission to include an index column         
    index = np.arange(1,len(x_test)+1, dtype=int)  
    # merging the y_test and index columns        
    y_test = np.column_stack((index,y_test))
    # convert to pandas dataframe
    y_test = pd.DataFrame(y_test)
    # headers required for the submission                          
    y_test.columns = ['ImageId','Label']    
    # write the data to csv file in the directory    
    y_test = y_test.astype('int')
    y_test.to_csv("".join([name,'.csv']), index=False) 
```
```python
submit(model_log,x_test,'model_log')
submit(model_rf,x_test,'model_rf')
submit(model_nn,x_test,'model_nn')
```
### Results

| Model               | Test Accuracy Score |
|---------------------|---------------------|
| Logistic Regression | 0.91771             |
| Random Forests      | 0.96800             |
| Neural Networks     | 0.94900             |