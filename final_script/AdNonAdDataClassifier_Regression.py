
# coding: utf-8

# ### Script Name : AdNonAdDataClassfier_Regression.ipynb
# #### Author : Sreejith Menon, Jairaj Shaktawat, Surbhi Arora, Pooja Donekal, Shvetha Suvarna
# #### Date : 11/07/2015
# #### Version: 1.0 Initial Draft
# ####                 2.0 Added Logic for Discretization

# In[ ]:

import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import pandas as pd
import statsmodels.api as sm
import pylab as pl
from sklearn.linear_model import LogisticRegression
import random
#%matplotlib inline


# In[ ]:

# Randomizing the CSV file

f = open('../data/ready_for_logistic_clean.csv')
reader = csv.reader(f)
headers = reader.__next__()

dt = []
for row in reader:
    dt.append(row)

f.close()

rand = random.sample(range(0, len(dt)), len(dt))

rand_data = []
print(max(rand))

for i in rand:
    print(rand[i])
    rand_data.append(dt[rand[i]])
    
fl = open("../data/ready_for_logistic_clean_1.csv","w")
ready_full_data = csv.writer(fl,dialect = 'excel',lineterminator='\n')
ready_full_data.writerow(headers)
for row in rand_data:
    ready_full_data.writerow(row)
    
fl.close()     


# ## Getting the data ready for logistic regression

# In[ ]:

dta = pd.read_csv('../data/ready_for_logistic_clean_1.csv')

#printing a few statistics
#print(dta.std())

div_percentage = 0.80
train_test_boundary = math.floor(div_percentage*len(dta))
end_boundary = len(dta)
train_data = dta[:train_test_boundary]
test_data = dta[train_test_boundary:]


# In[ ]:

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

# In[ ]:

logistic = LogisticRegression()
y = data['class']
X = data[headers_1]
logistic.fit(X,y)


# ## Predicition made on the test data

# In[ ]:

y = test_data['class']
actual_class_val = []
for i in range(train_test_boundary,end_boundary):
    actual_class_val.append(y[i])

# Predicted value
predictions = logistic.predict(test_data[headers_1])   


# In[ ]:

## Calculating accuracy of training data set

print(len(predictions))
print(len(actual_class_val))

count = 0

for i in range(0,(end_boundary-train_test_boundary)):
    if predictions[i] == actual_class_val[i]:
        count += 1
        
print(count)
print(count*100/(end_boundary-train_test_boundary))


# In[ ]:

ans = logistic.predict_proba(test_data[headers_1])


# In[ ]:

x = []
y = []
z = []
for row in ans:
    x.append(row[0])
    y.append(row[1])
    z.append(row[0]+row[1])


# In[ ]:

pl.plot(x)
pl.plot(y)
pl.plot(z)
pl.xlabel('Testing data indices', fontsize=18)
pl.ylabel('Probability of class', fontsize=16)
plt.show()


# In[ ]:



