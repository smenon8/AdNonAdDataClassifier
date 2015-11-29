
# coding: utf-8

# In[1]:

import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import pandas as pd
import statsmodels.api as sm
import pylab as pl
from sklearn.linear_model import LogisticRegression


# In[ ]:

def logistic_regression(percent):
    dta = pd.read_csv('../data/ready_for_logistic_clean.csv')
    
    #printing a few statistics
    #print(dta.std())
    div_percentage = percent/100
    train_test_boundary = math.floor(div_percentage*len(dta))
    end_boundary = len(dta)
    train_data = dta[:train_test_boundary]
    test_data = dta[train_test_boundary:]
    
    
    full_fl = csv.reader(open("../data/ready_for_logistic_clean.csv","r"))
    headers = full_fl.__next__()
    data = train_data[headers]
    test_data = test_data[headers]
    data['intercept'] = 1.0
    test_data['intercept'] = 1.0
    #print(data.head())
    
    headers_1 = headers
    headers_1.remove('class')
    headers_1.append('intercept')
    
    
    # ## Learning using logistic regression
    
    logistic = LogisticRegression()
    y = data['class']
    X = data[headers_1]
    logistic.fit(X,y)
    
    
    # ## Predicition made on the test data
    
    y = test_data['class']
    actual_class_val = []
    for i in range(train_test_boundary,end_boundary):
        actual_class_val.append(y[i])
    
    # Predicted value
    predictions = logistic.predict(test_data[headers_1])   
    
    ## Calculating accuracy of training data set
    
    count = 0
    
    for i in range(0,(end_boundary-train_test_boundary)):
        if predictions[i] == actual_class_val[i]:
            count += 1
            
    return count*100/(end_boundary-train_test_boundary)


# In[ ]:

x = []
y = []
for i in range(40,100):
    x.append(i)
    y.append(logistic_regression(i))
    print(i)
    
plt.plot(x,y)
plt.show()

