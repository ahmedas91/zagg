+++
categories = []
date = "2016-04-05"
description = "hopefully this works out but still every thing needs some work"
keywords = []
title = "Kaggle Digits Recognition"

+++
Dr. Anderw Ng's [machine leanring class](https://www.coursera.org/learn/machine-learning) excercises provide detailed walkthrough for solving a simple digit recognition problem, A LOT beter then than my humble attempt below on kaggle's [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) competition. Here I'm just applying what i've learned on a Kaggle competition using python's `sklearn` library. 

## Libraries
- [Numpy](http://www.numpy.org/)
- [Matplotlib](matplotlib.org/)
- [sklearn](scikit-learn.org/)
- [pandas](http://pandas.pydata.org/)

## Exploring the data 
```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

X =  np.genfromtxt('train.csv',dtype='int_', 
                   delimiter=',', skip_header=1)
x_test =  np.genfromtxt('test.csv',dtype='int_', 
                        delimiter=',', skip_header=1)
x_train = X[:,1:]
y_train = X[:,0]
```

Each row in the `x_train` and `x_test` data is a 28x28 pixels image with a total of 784 pixels. Therefore, we will write a simple function takes randomly selected rows, reshapes them into 28x28 matrices and desplay them using `matplotlib.image.mpimg`. 
```python
def display(n):    
    for i in range(1,(n**2)+1):
        plt.subplot(n,n,i)
        plt.axis('off')
        pic = np.reshape(x_train[np.random.randint(1,42000)],(28,28))
        imgplot = plt.imshow(pic, cmap='Greys')
display(5)
```
{{< figure src="/images/digits.png"
>}}

## Training the data
We're going to use a cross validated logistic linear regression function with an l2 regulaization using sklearn's `linear_model.LogisticRegressionCV`. Since we have 10 classes 0 to 9, we'll also need the `multiclass.OneVsRestClassifier`.
```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegressionCV
classifier = OneVsRestClassifier(LogisticRegressionCV(penalty='l2', 
                                                      n_jobs = -1)) 
classifier.fit(x_train, y_train)
```
Now let's check the accuracy of the training data. 
```python
# predict y using the train set
y_predicted = classifier.predict(x_train)             
accuracy = np.mean((y_predicted == y_train) * 100)
print "Training set accuracy: {0:.4f}%".format(accuracy)
```
`Training set accuracy: 93.1619%`

## Submitting the data
We're gonna do the same thing with but with the x_test data and do some data cleaning for submission. 
```python
# predict y using the test set
y_test = classifier.predict(x_test) 
# kaggles requires the submission to include an index column         
index = np.arange(1,len(x_test)+1, dtype=int)  
# merging the y_test and index columns        
y_test = np.column_stack((index,y_test))
# convert to pandas dataframe
y_test = pd.DataFrame(y_test)
# headers required for the submission                          
y_test.columns = ['ImageId','Label']    
# write the data to csv file in the directory               
y_test.to_csv('y_test_kaggle_digits.csv', index=False) 
```
After submitting the csv file we get an accuracy of 0.91100 which is not that bad (unless you check the rank and realize we're at the buttom!!). 