
# coding: utf-8

# In[ ]:

"""
    Script Name : AdNonAdDataClassfier.py
    Author : Sreejith Menon
    Date : 11/07/2015
    
    Version: 1.0 Initial Draft
"""


# In[ ]:

import numpy as np
import csv
import matplotlib.pyplot as plt
import math
get_ipython().magic('matplotlib inline')


# In[ ]:

adDataCsv = csv.reader(open("../data/ad.data","r"))

headers = adDataCsv.__next__()
len(headers)


# In[ ]:

## Seperate training and test data
## Training data is the data will no missing data
## Test data is the data with missing data
similar_index = []
train_data = []
test_data = []

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
    
print("done1")    


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
    



# In[ ]:

print(len(full_data))


# In[ ]:

## Discretizing the real values ** Start coding from here  ** Seems wrong to me
## Finding max height and width
maxim = [0,0,0]
n_new_feature = [0,0,0]
for row in full_data:
    if maxim[0] < float(row[0].strip()):
        maxim[0] = float(row[0].strip())
        n_new_feature[0] = math.ceil(maxim[0]/10)
    if maxim[1] < float(row[1].strip()):
        maxim[1] = float(row[1].strip())
        n_new_feature[1] = math.ceil(maxim[1]/10)
    if maxim[2] < float(row[2].strip()):
        maxim[2] = float(row[2].strip())
        n_new_feature[2] = math.ceil(maxim[2]/10)
        
## Maxim will decide the number of new features to be added


# In[ ]:

"""print(maxim)
print(n_new_feature)"""


# In[ ]:

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
    


# In[ ]:

print(disc_full_data[0])


# In[ ]:




# In[ ]:



