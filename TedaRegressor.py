"""
The __init__ method initializes a new instance of the DataCloud class with the given values of nf, mu, w, and x. It sets the values of n, nf, mean, mu, w, variance, pertinency, and faux. The N class variable is also incremented.

The addDataCloud method adds a new data point to the cloud. It updates the values of n, mean, and variance.

The updateDataCloud method updates the values of n, mean, variance, and faux for the cloud. It takes these values as arguments and updates the corresponding instance variables.

"""

from padasip.filters.base_filter import AdaptiveFilter

import pandas as pd
import numpy as np
import padasip as pa

np.random.seed(0)

class DataCloud:
  N=0
  def __init__(self, nf, mu, w, x):
      """
      Initializes a new DataCloud object.
      
      Args:
      nf (int): The filter order.
      mu (float): The step size for the filter.
      w (list of float): The initial weight vector for the filter.
      x (float): The initial data point.
      """
      self.n=1
      self.nf=nf
      self.mean=x
      self.mu = mu
      self.w = w
      self.variance=0
      self.pertinency=1
      self.faux = pa.filters.FilterRLS(n=self.nf, mu=self.mu, w=self.w)
      DataCloud.N+=1
      
  def addDataCloud(self,x):
      """
      Adds a new data point to the DataCloud object.
      
      Args:
      x (float): The new data point.
      """
      self.n=2
      self.mean=(self.mean+x)/2
      self.variance=((np.linalg.norm(self.mean-x))**2)
      
  def updateDataCloud(self,n,mean,variance, faux):
      """
      Updates the DataCloud object with new values.
      
      Args:
      n (int): The new value for the n parameter.
      mean (float): The new value for the mean.
      variance (float): The new value for the variance.
      faux (AdaptiveFilter): The new filter object.
      """
      self.n=n
      self.mean=mean
      self.variance=variance
      self.faux = faux

"""
This class is a Time-varying Evolving Data Analysis (TEDA) Regressor.

Attributes:
c (numpy.array): array of DataClouds with shape (1,) and dtype DataCloud.
alfa (numpy.array): array of floats with shape (1,).
intersection (numpy.array): array of ints with shape (1,1).
listIntersection (numpy.array): array of ints with shape (1,).
matrixIntersection (numpy.array): array of ints with shape (1,1).
relevanceList (numpy.array): array of ints with shape (1,).
k (int): integer representing the number of filters.

Methods:
init(self, m, mu, activation_function, threshold): initializes TEDARegressor object.
"""

  
class TEDARegressor:
  c= np.array([DataCloud(nf=2, mu=0.9, w=[0,0], x=0)],dtype=DataCloud)
  alfa= np.array([0.0],dtype=float)
  intersection = np.zeros((1,1),dtype=int)
  listIntersection = np.zeros((1),dtype=int)
  matrixIntersection = np.zeros((1,1),dtype=int)
  relevanceList = np.zeros((1),dtype=int)
  k=1

  def __init__(self, m, mu, activation_function, threshold):

    """
    Initializes a TEDARegressor object.

    Args:
        m (int): the number of features.
        mu (float): forgetting factor.
        activation_function (str): activation function used to initialize the weights.
        threshold (float): convergence threshold.

    Returns:
        None.
    """

    TEDARegressor.m = m
    TEDARegressor.mu = mu
    TEDARegressor.activation_function = activation_function
    TEDARegressor.threshold = threshold
    TEDARegressor.alfa= np.array([0.0],dtype=float)
    TEDARegressor.intersection = np.zeros((1,1),dtype=int)
    TEDARegressor.listIntersection = np.zeros((1),dtype=int)
    TEDARegressor.relevanceList = np.zeros((1),dtype=int)
    TEDARegressor.matrixIntersection = np.zeros((1,1),dtype=int)
    TEDARegressor.k=1
    TEDARegressor.classIndex = [[1.0],[1.0]]
    TEDARegressor.argMax = []
    TEDARegressor.RLSF_Index = []
    TEDARegressor.Ypred = []
    TEDARegressor.X_ant = np.zeros((1, TEDARegressor.m), dtype=float)
    TEDARegressor.NumberOfFilters = []
    TEDARegressor.NumberOfDataClouds = []
    
    np.random.seed(0)
    TEDARegressor.random_factor = np.random.rand(TEDARegressor.m-1, TEDARegressor.m)

    if (TEDARegressor.activation_function == "relu"): #He
      factor = np.sqrt(2/(TEDARegressor.m-1))
    elif (TEDARegressor.activation_function == "tanh1"): #Xavier
      factor = np.sqrt(1/(TEDARegressor.m-1))
    elif (TEDARegressor.activation_function == "tanh2"): #Yoshua
      factor = np.sqrt(2/((2*TEDARegressor.m)-1))
    elif (TEDARegressor.activation_function == "zero"):
      factor = 0
    else: #Utiliza a Formula do "He" como DEFAULT
      factor = np.sqrt(2/TEDARegressor.m-1)
           
    TEDARegressor.w_init = TEDARegressor.random_factor*factor 
    TEDARegressor.w_init = TEDARegressor.w_init[0].tolist()
    TEDARegressor.c = np.array([DataCloud(nf=TEDARegressor.m, mu=TEDARegressor.mu, w=TEDARegressor.w_init, x=0)],dtype=DataCloud)
    TEDARegressor.f0 = pa.filters.FilterRLS(TEDARegressor.m, mu=TEDARegressor.mu, w=TEDARegressor.w_init)

  """
  Merge clusters in the TEDARegressor's listIntersection and update the matrixIntersection.

  This method iterates through the listIntersection and merge clusters when they have intersection. The merging process is done by updating the corresponding values in the matrixIntersection. Additionally, the statistics of the merged clusters are updated according to the weighted mean formula.

  Args:
  self (TEDARegressor): An instance of TEDARegressor class.

  Returns:
  None

  """
  def mergeClouds(self):
    """
    Merges the clouds that have an intersection with each other, updating the weight parameters.
    
    Args:
    - self: the object itself.
    
    Returns:
    - None
    
    """  
    i=0
    while(i<len(TEDARegressor.listIntersection)-1):
      merge = False
      j=i+1
      while(j<len(TEDARegressor.listIntersection)):

        if(TEDARegressor.listIntersection[i] == 1 and TEDARegressor.listIntersection[j] == 1):
          TEDARegressor.matrixIntersection[i,j] = TEDARegressor.matrixIntersection[i,j] + 1;

        nI = TEDARegressor.c[i].n
        nJ = TEDARegressor.c[j].n

        meanI = TEDARegressor.c[i].mean
        meanJ = TEDARegressor.c[j].mean
        varianceI = TEDARegressor.c[i].variance
        varianceJ = TEDARegressor.c[j].variance
        nIntersc = TEDARegressor.matrixIntersection[i,j]
        fauxI = TEDARegressor.c[i].faux
        fauxJ = TEDARegressor.c[j].faux
        
        wI = fauxI.getW()
        wJ = fauxJ.getW()
        
        dwI = fauxI.getdW()
        dwJ = fauxJ.getdW()

        # update the weight parameter W        
        W = (nI*wI)/(nI + nJ) + (nJ*wJ)/(nI + nJ)

        # Check if the intersection value is greater than the difference between n and intersection.
        if(nIntersc > (nI - nIntersc) or nIntersc > (nJ - nIntersc)):
            merge = True

            # update values for the new cloud
            n = nI + nJ - nIntersc
            mean = ((nI * meanI) + (nJ * meanJ))/(nI + nJ)
            variance = ((nI - 1) * varianceI + (nJ - 1) * varianceJ)/(nI + nJ - 2)

            # merge faux clouds
            if(nI >= nJ):
                fauxI.mergeWith(fauxJ, nI, nJ)
                faux = fauxI
            else:
                fauxJ.mergeWith(fauxI, nI, nJ)
                faux = fauxJ

            # create and update new data cloud
            newCloud = DataCloud(nf = TEDARegressor.m, mu = TEDARegressor.mu, w = TEDARegressor.w_init, x = mean)
            newCloud.updateDataCloud(n, mean, variance, faux)

            # update intersection list and data cloud list
            TEDARegressor.listIntersection = np.concatenate((TEDARegressor.listIntersection[0:i], np.array([1]), TEDARegressor.listIntersection[i+1:j], TEDARegressor.listIntersection[j+1:np.size(TEDARegressor.listIntersection)]), axis=None)
            TEDARegressor.c = np.concatenate((TEDARegressor.c[0:i], np.array([newCloud]), TEDARegressor.c[i+1:j], TEDARegressor.c[j+1:np.size(TEDARegressor.c)]), axis=None)

            # update intersection matrix
            M0 = TEDARegressor.matrixIntersection

            # remove rows
            M1 = np.concatenate((M0[0:i, :], np.zeros((1, len(M0))), M0[i+1:j, :], M0[j+1:len(M0), :]))
            # remove columns
            M1 = np.concatenate((M1[:, 0:i], np.zeros((len(M1), 1)), M1[:, i+1:j], M1[:, j+1:len(M0)]), axis=1)
            # calculate new column
            col = (M0[:, i] + M0[:, j]) * (M0[:, i] * M0[:, j] != 0)
            col = np.concatenate((col[0:j], col[j+1:np.size(col)]))
            # calculate new row
            lin = (M0[i, :] + M0[j, :]) * (M0[i, :] * M0[j, :] != 0)
            lin = np.concatenate((lin[0:j], lin[j+1:np.size(lin)]))
            # update column
            M1[:, i] = col
            # update row
            M1[i, :] = lin
            M1[i, i+1:j] = M0[i, i+1:j] + M0[i+1:j, j].T

            TEDARegressor.matrixIntersection = M1
        j += 1
      if (merge):
        i = 0
      else:
        i += 1
				
  def run(self, X):
    """Executes the TEDARegressor algorithm for the given input data X.

    Args:
    - self: an instance of the TEDARegressor class.
    - X: a numpy array representing the input data.

    Returns:
    - None.

    """
    # Initialize listIntersection with zeros.
    TEDARegressor.listIntersection = np.zeros((np.size(TEDARegressor.c)), dtype=int)

    if TEDARegressor.k == 1:
        # Create a new DataCloud instance for the first data point.
        TEDARegressor.c[0] = DataCloud(nf=TEDARegressor.m, mu=TEDARegressor.mu, w=TEDARegressor.w_init, x=X)
        TEDARegressor.argMax.append(0)
        TEDARegressor.c[0].faux = TEDARegressor.f0 
        TEDARegressor.RLSF_Index.append(0)
        TEDARegressor.X_ant = X

    elif TEDARegressor.k == 2:
        # Add data point to the existing DataCloud.
        TEDARegressor.c[0].addDataCloud(X)
        TEDARegressor.argMax.append(0)
        TEDARegressor.RLSF_Index.append(0)
        TEDARegressor.X_ant = X
    
    elif TEDARegressor.k >= 3:
        i = 0
        createCloud = True
        TEDARegressor.alfa = np.zeros((np.size(TEDARegressor.c)), dtype=float)

        # Iterate over existing DataCloud instances.
        for data in TEDARegressor.c:
            n = data.n + 1
            mean = ((n-1)/n) * data.mean + (1/n) * X
            variance = ((n-1)/n) * data.variance + (1/n) * ((np.linalg.norm(X-mean))**2)
            eccentricity = (1/n) + ((mean-X).T.dot(mean-X)) / (n*variance)
            typicality = 1 - eccentricity
            norm_eccentricity = eccentricity / 2
            norm_typicality = typicality / (TEDARegressor.k - 2)
            faux_ = data.faux

            if (norm_eccentricity <= (TEDARegressor.threshold**2 + 1) / (2*n)):
                # If the data point fits inside the DataCloud, update it and set createCloud to False.
                data.updateDataCloud(n, mean, variance, faux_)
                TEDARegressor.alfa[i] = norm_typicality
                createCloud = False
                TEDARegressor.listIntersection.itemset(i, 1)

                # Adapt the faux function for the DataCloud.
                faux_.adapt(X[-1], TEDARegressor.X_ant)

            else:
                # If the data point doesn't fit inside the DataCloud, set listIntersection for this index to 0.
                TEDARegressor.alfa[i] = norm_typicality
                TEDARegressor.listIntersection.itemset(i, 0)

            i += 1

        if (createCloud):
            # If none of the existing DataClouds can accommodate the data point, create a new DataCloud instance.
            TEDARegressor.c = np.append(TEDARegressor.c, DataCloud(nf=TEDARegressor.m, mu=TEDARegressor.mu, w=TEDARegressor.w_init, x=X))
            TEDARegressor.listIntersection = np.insert(TEDARegressor.listIntersection, i, 1)
            TEDARegressor.matrixIntersection = np.pad(TEDARegressor.matrixIntersection, ((0,1),(0,1)), 'constant', constant_values=(0))

        #print("TEDARegressor dps do Merge:", TEDARegressor.c)
        TEDARegressor.NumberOfFilters.append(len(TEDARegressor.c))
        TEDARegressor.relevanceList = TEDARegressor.alfa /np.sum(TEDARegressor.alfa)
        TEDARegressor.argMax.append(np.argmax(TEDARegressor.relevanceList))
        TEDARegressor.classIndex.append(TEDARegressor.alfa)
            
        filter_used = TEDARegressor.c[np.argmax(TEDARegressor.relevanceList)].faux
        
        self.mergeClouds()
          
        y_pred = filter_used.predict(X)

        TEDARegressor.Ypred.append(y_pred)
        TEDARegressor.RLSF_Index.append(np.argmax(TEDARegressor.relevanceList))
        TEDARegressor.X_ant = X
      
    TEDARegressor.k=TEDARegressor.k+1