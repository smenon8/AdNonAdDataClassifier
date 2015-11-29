
# coding: utf-8

# ### Script Name : AdNonAdDataClassfier_DataCleanse.ipynb
# #### Author : Sreejith Menon, Jairaj Shaktawat, Surbhi Arora, Pooja Donekal, Shvetha Suvarna
# #### Date : 11/07/2015
# #### Version: 1.0 Initial Draft
# ####                 2.0 Added Logic for Discretization

# In[8]:

import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import pandas as pd
import statsmodels.api as sm
import pylab as pl
from sklearn.linear_model import LogisticRegression

#%matplotlib inline


# In[ ]:

adDataCsv = csv.reader(open("../data/ad.data","r"))

headers = adDataCsv.__next__()
len(headers)


# ## Seperate training and test data
# ### Training data is the data will no missing data
# ### Test data is the data with missing data

# In[ ]:


similar_index = []
train_data = []
test_data = []

for row in adDataCsv:
    if row[0].strip() == '?' or row[1].strip() == '?' or row[2].strip() == '?':
        test_data.append(row)
    else:
        train_data.append(row)        

print("done")    


# ## Imputing the missing data using k-NN

# In[ ]:

## Calculating score matrix for each test data
for row_test in test_data:
    score = [0]*len(train_data)
    row_num = 0
    for row_train in train_data:
        for attrib in range(3,len(headers)):
            if row_test[attrib] == row_train[attrib]:
                score[row_num] += 1
        row_num += 1
        
        ## Score matrix for one test example available
    similar_index.append(score.index(max(score)))


# In[ ]:

print(score.index((max(score))))

print(similar_index)


# In[ ]:

## Assigning Height, Width and aratio same as that of the training data with the highest match

for i in range(0,len(test_data)):
    ind = similar_index[i]
    test_data[i][0] = train_data[ind][0] # Height
    test_data[i][1] = train_data[ind][1] # Width
    test_data[i][2] = train_data[ind][2] # aratio
    


# In[ ]:

## Append the test data with the training data
full_data = []
for row in train_data:
    full_data.append(row)
    
for row in test_data:
    full_data.append(row)
    



# ## Logic for discretizing the continous field values.
# ### Each of the real values divided into 14 

# In[ ]:

for i in range(0,len(full_data)):
    temp1=[0]*14
    temp2=[0]*14
    temp3=[0]*14

    val1=math.floor(int(full_data[i][0])/50)
    temp1[val1]=1
   
    val2=math.floor(int(full_data[i][1])/50)
    temp2[val2]=1
   
    val3=math.floor(float(full_data[i][2])/5)
    temp3[val3]=1
    
    full_data[i]=temp3+full_data[i][3:]
    full_data[i]=temp2+full_data[i]
    full_data[i]=temp1+full_data[i]


# In[ ]:

print(full_data[1][0])


# In[ ]:

### logic for removing the first 3 columns and ad = 1 and non_ad = 0

i = 0
full_bin_data = []
for row in full_data:
    row_dash = []
    for i in range(0,len(row)):
        if row[i] == "ad.":
            row_dash.append(1)
        else:
            if row[i] == "nonad.":
                row_dash.append(0)
            else:
                row_dash.append(row[i])
    full_bin_data.append(row_dash)


# In[ ]:

headers_back = headers

headers_back.remove('height')
headers_back.remove('width')
headers_back.remove('aratio')

headers_new = []

for i in range(0,14):
    headers_new.append('height%d'%i)

for i in range(0,14):
    headers_new.append('width%d'%i)    
    
for i in range(0,14):
    headers_new.append('aratio%d'%i) 
    
for row in headers_back:
    headers_new.append(row)



# In[ ]:

## Writing the data to a csv file

print(len(full_bin_data))

fl = open("../data/ready_for_logistic.csv","w")
ready_full_data = csv.writer(fl,dialect = 'excel',lineterminator='\n')
ready_full_data.writerow(headers_new)
for row in full_bin_data:
    ready_full_data.writerow(row)
    
fl.close()    


# #### Cleansing data to convert everything to float values 

# In[ ]:

full_fl = csv.reader(open("../data/ready_for_logistic.csv","r"))
headers = full_fl.__next__()

fl = open("../data/ready_for_logistic_clean.csv","w")
ready_full_data = csv.writer(fl,dialect = 'excel',lineterminator='\n')
ready_full_data.writerow(headers)
count = 0
for row in full_fl:
    clean_row = []
    for col in row:
        if(col == '?'):
            clean_row.append(0)
            count += 1
        else:
            clean_row.append(int(col.strip()))
        
    ready_full_data.writerow(clean_row)

fl.close()

print(count)

