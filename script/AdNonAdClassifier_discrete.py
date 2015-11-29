
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
import pandas as pd
import scipy.optimize as opt
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
for i in range(0,len(full_data)):
   temp1=[0]
   temp2=[0]
   temp3=[0]
   for j in range(1,13):
       temp1.append('0');
       temp2.append('0');
       temp3.append('0');
   val1=math.floor(int(full_data[i][0])/50)
   temp1[val1]=1
   full_data[i]=full_data[i]+temp1
   val2=math.floor(int(full_data[i][1])/50)
   temp2[val2]=1
   full_data[i]=full_data[i]+temp2
   val3=math.floor(float(full_data[i][2])/5)
   temp3[val3]=1
   full_data[i]=full_data[i]+temp3
   # In[ ]:
   csv_full_data = csv.writer(open("../data/full_ad.csv","w"),dialect = 'excel')
   
for row in full_data:
    csv_full_data.writerow(row)



# In[ ]:


 
"""def logisticRegressionReg(filePath, method):
     
    df = pd.read_csv(filePath, header = None)
     
    y = np.array(df[len(df.columns)-1])
     
    m = len(y)
 
    X = np.array(np.array(df.values[:,:-1]))
        
    init_val = 0.1*np.random.randn(X.shape[1]) #it doesn't converge with any value; if it fails run it again!
    theta, j_min = opt.fmin_bfgs(compute_cost, init_val, fprime=compute_grad, args=(X, y), full_output=True, disp=False)[0:2]
    p = predict(theta, X, y)
     
    if method == "l":
        plotResultLinear(df, theta)
    elif method == 'p':
        plotResultPolynomial(df, theta, X, y)
     
    return theta, j_min, p
 
 
def sigmoid(z):
    return 1/(1+np.exp(-z))
 
 
def compute_cost(theta, X, y):
    m = len(y)
     
    h = sigmoid(np.dot(X,theta))
 
    J = (-1/m)*(np.dot(y.T, np.log(h))+np.dot((1-y).T, np.log(1-h))) + (1/(2*m))*(np.dot(theta.T,theta) - theta[0]**2)
    return J
 
def compute_grad(theta, X, y):
    m = len(y)
     
    h = sigmoid(np.dot(X,int(theta)))
     
    grad = ((1/m)*np.dot(h-y, X)) + (1/m)*theta
    grad[0] = grad[0] - (1/m)*theta[0]
     
    return grad
 
 
def predict(theta, X, y):
 
    p = theta.dot(X.T)
    p = np.where(p >= 0.5, 1, 0)
    p = sum(np.where(p==y, 1, 0))/100
     
    return p
 
     
 
def plotResultLinear(df, theta):
     
    x0_min, x0_max = np.array(df[0]).min(), np.array(df[0]).max()
    x1_min, x1_max = np.array(df[1]).min(), np.array(df[1]).max()
    xmin, xmax = np.min([x0_min, x1_min]), np.max([x0_max, x1_max])
     
    ax = plt.subplot(111)
     
    ax.scatter(df[df[2]==0][0], df[df[2]==0][1], c='r', marker='o', label='Not Admitted')
    ax.scatter(df[df[2]==1][0], df[df[2]==1][1], c='g', marker='+', label='Admitted')
    ax.plot(-(theta[0]+theta[1]*np.arange(xmin, xmax))/theta[2], np.arange(xmin, xmax), label='Decision Boundary')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel("Exam 1 Score")
    plt.ylabel("Exam 2 Score")
    plt.xlim(xmin, xmax)
    plt.ylim(xmin, xmax)
     
    return
 
def plotResultPolynomial(df, theta, X, y):
    pointsForGrid = 50
     
    x0_min, x0_max = np.array(df[0]).min(), np.array(df[0]).max()
    x1_min, x1_max = np.array(df[1]).min(), np.array(df[1]).max()
    xmin, xmax = np.min([x0_min, x1_min]), np.max([x0_max, x1_max])
     
    ax = plt.subplot(111)
     
    ax.scatter(df[df[2]==0][0], df[df[2]==0][1], c='r', marker='o', label='Not Passed')
    ax.scatter(df[df[2]==1][0], df[df[2]==1][1], c='g', marker='+', label='Passed')
 
         
    u, v = np.linspace(xmin, xmax, pointsForGrid), np.linspace(xmin, xmax, pointsForGrid)
    #Build all the possible u[i],v[j] couples
    pairs = np.dstack(np.meshgrid(u,v)).reshape(-1,2)
    #For each u[i],v[j] couple calculate the polynomial and evaluate the output of the model (with our previously calculated theta)
    z = np.apply_along_axis(lambda x: mapFeature(np.array([x[0]]), np.array([x[1]])).dot(theta), 1, pairs)
    #z must have shape (u,v)
    z = z.reshape(pointsForGrid, pointsForGrid)
    #Plot only the contour line where the function is (0.0), i.e. the decision boundary. In Python it is not possible
    #to plot a single contour line so here two very very near lines are draw
    cs = ax.contour(u, v, z, levels=[0.0, 0.001], colors='k')
    cs.collections[0].set_label('Decision Boundary')
     
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
     
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    plt.xlim(xmin, xmax)
    plt.ylim(xmin, xmax)
     
    return
 
if __name__ == '__main__':
     
    np.set_printoptions(precision=3)
    print(logisticRegressionReg("../data/full_data.csv", "p"))
    plt.show()"""
