+++
categories = []
date = "2016-04-05T20:55:56+03:00"
description = " "
keywords = []
title = "Kaggle Digits Recognition"

+++

In this post, I'll solve  Kaggle's [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) competition using python's machine learning library `sklearn`. However, if you want to check out an implementation from scratch, I have uploaded one on this repo on github. 

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

We're going to use a cross validated logistic linear regression function with an l2 regularization using sklearn's `linear_model.LogisticRegressionCV`. Since we have 10 classes 0 to 9, we'll also need the `multiclass.OneVsRestClassifier`.


```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegressionCV

classifier = OneVsRestClassifier(LogisticRegressionCV(penalty='l2', n_jobs = -1)) 
classifier.fit(x_train, y_train)
```

Now let's check the accuracy of the training data. 


```python
# predict y using the train set
y_predicted = classifier.predict(x_train)             
accuracy = np.mean((y_predicted == y_train) * 100)
print "Training set accuracy: {0:.4f}%".format(accuracy)
```

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
