
# coding: utf-8

# In[1]:

"""
    Script Name : AdNonAdDataClassfier.py
    Author : Sreejith Menon
    Date : 11/07/2015
    
    Version: 1.0 Initial Draft
"""


# In[2]:

import numpy as np
import csv
import matplotlib.pyplot as plt
import math
get_ipython().magic('matplotlib inline')


# In[3]:

adDataCsv = csv.reader(open("../data/ad.data","r"))

headers = adDataCsv.__next__()
len(headers)


# In[4]:

## Seperate training and test data
## Training data is the data will no missing data
## Test data is the data with missing data

train_data = []
test_data = []
similar_index = []

for row in adDataCsv:
    if row[0].strip() == '?' or row[1].strip() == '?' or row[2].strip() == '?':
        test_data.append(row)
    else:
        train_data.append(row)        
        
csv_train_data = csv.writer(open("../data/train_ad.csv","w"),dialect = 'excel')
csv_test_data = csv.writer(open("../data/test_ad.csv","w"),dialect = 'excel')

for row in test_data:
    csv_test_data.writerow(row)
    
    
for row in train_data:
    csv_train_data.writerow(row)   
    
print("done")    


# In[5]:

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


# In[6]:

print(similar_index)


# In[7]:

## Assigning Height, Width and aratio same as that of the training data with the highest match

for i in range(0,len(test_data)):
    ind = similar_index[i]
    test_data[i][0] = train_data[ind][0] # Height
    test_data[i][1] = train_data[ind][1] # Width
    test_data[i][2] = train_data[ind][2] # aratio
    


# In[8]:

## Append the test data with the training data
full_data = []
for row in train_data:
    full_data.append(row)
    
for row in test_data:
    full_data.append(row)
    



# In[9]:

print(len(full_data))


# In[11]:

## Discretizing the real values
## Finding max height and width
max = [0,0,0]
n_new_feature = [0,0,0]
for row in full_data:
    if max[0] < float(row[0].strip()):
        max[0] = float(row[0].strip())
        n_new_feature[0] = math.ceil(max[0]/10)
    if max[1] < float(row[1].strip()):
        max[1] = float(row[1].strip())
        n_new_feature[1] = math.ceil(max[1]/10)
    if max[2] < float(row[2].strip()):
        max[2] = float(row[2].strip())
        n_new_feature[2] = math.ceil(max[2]/10)
        
## Max will decide the number of new features to be added


# In[13]:

print(max)
print(n_new_feature)


# In[14]:

disc_full_data = []
single_row = []
for row in full_data:
    ind = math.floor(float(row[0].strip()))
    for j in range(0,3):
        for i in range(0,n_new_feature[j]):
            if i == ind:
                single_row.append(1)
            else:
                single_row.append(0)
    single_row.append(row[2:len(row)])
    disc_full_data.append(single_row)
    


# In[15]:

print(disc_full_data[0])


# In[ ]:



