
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
        
"""csv_train_data = csv.writer(open("../data/train_ad.csv","w"),dialect = 'excel')
csv_test_data = csv.writer(open("../data/test_ad.csv","w"),dialect = 'excel')

for row in test_data:
    csv_test_data.writerow(row)
    
    
for row in train_data:
    csv_train_data.writerow(row)
"""   
print("done")    


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




# In[ ]:

## Discretizing the real values ** Start coding from here  ** Seems wrong to me
## Finding max height and width
"""for i in range(0,len(full_data)):
   temp1=[0]*14
   temp2=[0]*14
   temp3=[0]*14

   val1=math.floor(int(full_data[i][0])/50)
   temp1[val1]=1
   full_data[i]=full_data[i]+temp1
   val2=math.floor(int(full_data[i][1])/50)
   temp2[val2]=1
   full_data[i]=full_data[i]+temp2
   val3=math.floor(float(full_data[i][2])/5)
   temp3[val3]=1
   full_data[i]=full_data[i]+temp3"""


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

print(full_data[0])


# In[ ]:

### logic for removing the first 3 columns and ad = 5 and non_ad = -5
## since it is Jai's lucky number
i = 0
full_bin_data = []

for row in full_data:
    row_dash = []
    for i in range(0,len(row)):
        if row[i] == "ad.":
            row_dash.append('5')
        else:
            if row[i] == "nonad." :
                row_dash.append('-5')
            else:
                row_dash.append(row[i])
    full_bin_data.append(row_dash)


# In[ ]:

print(full_bin_data[13])


# In[ ]:




# In[ ]:



