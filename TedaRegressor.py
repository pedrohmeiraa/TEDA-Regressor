from padasip.filters.base_filter import AdaptiveFilter

import pandas as pd
import numpy as np
import padasip as pa

np.random.seed(0)

class DataCloud:
  N=0
  def __init__(self, nf, mu, w, x):
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
      self.n=2
      self.mean=(self.mean+x)/2
      self.variance=((np.linalg.norm(self.mean-x))**2)
  def updateDataCloud(self,n,mean,variance, faux):
      self.n=n
      self.mean=mean
      self.variance=variance
      self.faux = faux
  
 
class TEDARegressor:
  c= np.array([DataCloud(nf=2, mu=0.9, w=[0,0], x=0)],dtype=DataCloud)
  alfa= np.array([0.0],dtype=float)
  intersection = np.zeros((1,1),dtype=int)
  listIntersection = np.zeros((1),dtype=int)
  matrixIntersection = np.zeros((1,1),dtype=int)
  relevanceList = np.zeros((1),dtype=int)
  k=1

  def __init__(self, m, mu, activation_function, threshold):
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
    TEDARegressor.classIndex = [[1.0],[1.0]] #<========== try in another moment: [np.array(1.0),np.array(1.0)]
    TEDARegressor.argMax = []
    TEDARegressor.RLSF_Index = []
    TEDARegressor.Ypred = []
    TEDARegressor.X_ant = np.zeros((1, TEDARegressor.m), dtype=float)
    TEDARegressor.NumberOfFilters = []
    TEDARegressor.NumberOfDataClouds = []
    
    np.random.seed(0)
    TEDARegressor.random_factor = np.random.rand(TEDARegressor.m-1, TEDARegressor.m)
    #TEDARegressor.random_factor = np.array([[0.00504779, 0.99709118]])

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
    #print("w_init do TEDA: ", TEDARegressor.w_init)
    TEDARegressor.c = np.array([DataCloud(nf=TEDARegressor.m, mu=TEDARegressor.mu, w=TEDARegressor.w_init, x=0)],dtype=DataCloud)
    TEDARegressor.f0 = pa.filters.FilterRLS(TEDARegressor.m, mu=TEDARegressor.mu, w=TEDARegressor.w_init)


  def mergeClouds(self):
    i=0
    while(i<len(TEDARegressor.listIntersection)-1):
      #print("i do merge",i)
      #print(TEDARegressor.listIntersection)
      #print(TEDARegressor.matrixIntersection)
      merge = False
      j=i+1
      while(j<len(TEDARegressor.listIntersection)):
        #print("j do merge",j)
        #print("i",i,"j",j,"l",np.size(TEDARegressor.listIntersection),"m",np.size(TEDARegressor.matrixIntersection),"c",np.size(TEDARegressor.c))
        if(TEDARegressor.listIntersection[i] == 1 and TEDARegressor.listIntersection[j] == 1):
          TEDARegressor.matrixIntersection[i,j] = TEDARegressor.matrixIntersection[i,j] + 1;
          #print(TEDARegressor.matrixIntersection)
        nI = TEDARegressor.c[i].n
        nJ = TEDARegressor.c[j].n
        #print("I: ",list(TEDARegressor.c).index(TEDARegressor.c[i]), ". J: ",list(TEDARegressor.c).index(TEDARegressor.c[j]))
        meanI = TEDARegressor.c[i].mean
        meanJ = TEDARegressor.c[j].mean
        varianceI = TEDARegressor.c[i].variance
        varianceJ = TEDARegressor.c[j].variance
        nIntersc = TEDARegressor.matrixIntersection[i,j]
        fauxI = TEDARegressor.c[i].faux    #fauxI = TEDARegressor.RLS_Filters[i]
        fauxJ = TEDARegressor.c[j].faux    #fauxJ = TEDARegressor.RLS_Filters[j]
        
        wI = fauxI.getW()
        wJ = fauxJ.getW()
        #print("wJ: ", wJ)
        
        dwI = fauxI.getdW()
        dwJ = fauxJ.getdW()
                
        W = (nI*wI)/(nI + nJ) + (nJ*wJ)/(nI + nJ)

        if (nIntersc > (nI - nIntersc) or nIntersc > (nJ - nIntersc)):
          #print(nIntersc, "(nIntersc) >", nI, "-", nIntersc, "=", nI - nIntersc, "(nI - nIntersc) OR", nIntersc, "(nIntersc) >", nJ, "-", nIntersc, "=", nJ - nIntersc, "(nJ - nIntersc)")
          
          #print("(nIntersc) =",nIntersc, ",nI =", nI, ",nJ =", nJ)
          #print("Juntou!")

          merge = True
          #update values
          n = nI + nJ - nIntersc
          mean = ((nI * meanI) + (nJ * meanJ))/(nI + nJ)
          variance = ((nI - 1) * varianceI + (nJ - 1) * varianceJ)/(nI + nJ - 2)
          #faux = fauxI #Considerando o de maior experiencia (mais amostras)
          #faux = pa.filters.FilterRLS(2, mu=mu_, w = W) #Considerando a inicialização dos pesos (W)
                       
          if (nI >= nJ):
            fauxI.mergeWith(fauxJ, nI, nJ)
            faux = fauxI

          else:
            fauxJ.mergeWith(fauxI, nI, nJ)
            faux = fauxJ
          
          #TEDARegressor.RLS_Filters.pop()

          newCloud = DataCloud(nf=TEDARegressor.m, mu=TEDARegressor.mu, w=TEDARegressor.w_init, x=mean)
          newCloud.updateDataCloud(n,mean,variance, faux)
          
          #atualizando lista de interseção
          TEDARegressor.listIntersection = np.concatenate((TEDARegressor.listIntersection[0 : i], np.array([1]), TEDARegressor.listIntersection[i + 1 : j],TEDARegressor.listIntersection[j + 1 : np.size(TEDARegressor.listIntersection)]),axis=None)
          #print("listInters dps de att:", TEDARegressor.listIntersection)
          #atualizando lista de data clouds
          #print("dentro do if do merge antes", TEDARegressor.c)
          TEDARegressor.c = np.concatenate((TEDARegressor.c[0 : i ], np.array([newCloud]), TEDARegressor.c[i + 1 : j],TEDARegressor.c[j + 1 : np.size(TEDARegressor.c)]),axis=None)
          #print("dentro do if do merge dps do concate", TEDARegressor.c)

          #update  intersection matrix
          M0 = TEDARegressor.matrixIntersection
          #Remover linhas 
          M1=np.concatenate((M0[0 : i , :],np.zeros((1,len(M0))),M0[i + 1 : j, :],M0[j + 1 : len(M0), :]))
          #remover colunas
          M1=np.concatenate((M1[:, 0 : i ],np.zeros((len(M1),1)),M1[:, i+1 : j],M1[:, j+1 : len(M0)]),axis=1)
          #calculando nova coluna
          col = (M0[:, i] + M0[:, j])*(M0[: , i]*M0[:, j] != 0)
          col = np.concatenate((col[0 : j], col[j + 1 : np.size(col)]))
          #calculando nova linha
          lin = (M0[i, :]+M0[j, :])*(M0[i, :]*M0[j, :] != 0)
          lin = np.concatenate((lin[ 0 : j], lin[j + 1 : np.size(lin)]))
          #atualizando coluna
          M1[:,i]=col
          #atualizando linha
          M1[i,:]=lin
          M1[i, i + 1 : j] = M0[i, i + 1 : j] + M0[i + 1 : j, j].T;   
          TEDARegressor.matrixIntersection = M1
          #print(TEDARegressor.matrixIntersection)
        j += 1
      if (merge):
        i = 0
      else:
        i += 1
				
  def run(self,X):
    TEDARegressor.listIntersection = np.zeros((np.size(TEDARegressor.c)),dtype=int)
    #print("k=", TEDARegressor.k)
    if TEDARegressor.k==1:
      TEDARegressor.c[0]=DataCloud(nf=TEDARegressor.m, mu=TEDARegressor.mu, w=TEDARegressor.w_init, x=X)
      TEDARegressor.argMax.append(0)
      TEDARegressor.c[0].faux = TEDARegressor.f0 #TEDARegressor.RLS_Filters = [TEDARegressor.f0] 
      TEDARegressor.RLSF_Index.append(0)
      TEDARegressor.X_ant = X
      #TEDARegressor.c[0].faux.adapt(X[1], TEDARegressor.X_ant)

    elif TEDARegressor.k==2:
      TEDARegressor.c[0].addDataCloud(X)
      TEDARegressor.argMax.append(0)
      TEDARegressor.RLSF_Index.append(0)
      TEDARegressor.X_ant = X
      #TEDARegressor.c[0].faux.adapt(X[1], TEDARegressor.X_ant)
    
    elif TEDARegressor.k>=3:
      i=0
      createCloud = True
      TEDARegressor.alfa = np.zeros((np.size(TEDARegressor.c)),dtype=float)

      for data in TEDARegressor.c:
        n= data.n + 1
        mean = ((n-1)/n)*data.mean + (1/n)*X
        variance = ((n-1)/n)*data.variance +(1/n)*((np.linalg.norm(X-mean))**2)
        eccentricity=(1/n)+((mean-X).T.dot(mean-X))/(n*variance)
        typicality = 1 - eccentricity
        norm_eccentricity = eccentricity/2
        norm_typicality = typicality/(TEDARegressor.k-2)
        faux_ = data.faux
        #faux_.adapt(X[-1], TEDARegressor.X_ant)
        
        if (norm_eccentricity<=(TEDARegressor.threshold**2 +1)/(2*n)): #Se couber dentro da Cloud
          
          data.updateDataCloud(n,mean,variance, faux_)
          TEDARegressor.alfa[i] = norm_typicality
          createCloud= False
          TEDARegressor.listIntersection.itemset(i,1)
          #print("pesos=", data.faux.w)
          #print("x_ant=", TEDARegressor.X_ant)
          #print("dentro da cloud")
          faux_.adapt(X[-1], TEDARegressor.X_ant)
        #TEDARegressor.c[i].faux.adapt(X[-1], TEDARegressor.X_ant) #data.faux.adapt(X[-1], TEDARegressor.X_ant) #data.faux.adapt(X[-1], TEDARegressor.X_ant)
              
        else: #Se nao couber
          TEDARegressor.alfa[i] = norm_typicality
          TEDARegressor.listIntersection.itemset(i,0)
          #print("fora da cloud")
          #print("fora -> i:", i, " - Filtro: ", faux_)
          #faux_.adapt(X[-1], TEDARegressor.X_ant)
          #TEDARegressor.c[i].faux.adapt(X[-1], TEDARegressor.X_ant) #data.faux.adapt(X[-1], TEDARegressor.X_ant)
        i+=1

      if (createCloud):
        #print("no if de criar TEDARegressor:", TEDARegressor.c)
        TEDARegressor.c = np.append(TEDARegressor.c,DataCloud(nf=TEDARegressor.m, mu=TEDARegressor.mu, w=TEDARegressor.w_init, x=X))
        #print("dps do if TEDARegressor:", TEDARegressor.c)
        TEDARegressor.listIntersection = np.insert(TEDARegressor.listIntersection,i,1)
        TEDARegressor.matrixIntersection = np.pad(TEDARegressor.matrixIntersection, ((0,1),(0,1)), 'constant', constant_values=(0))
        #print("DataCloud Created!")
        #TEDARegressor.RLS_Filters.append(pa.filters.FilterRLS(TEDARegressor.m, mu=TEDARegressor.mu, w=TEDARegressor.w_init))


      #print("TEDARegressor antes do Merge:", TEDARegressor.c)
      
      #print("TEDARegressor dps do Merge:", TEDARegressor.c)
      TEDARegressor.NumberOfFilters.append(len(TEDARegressor.c))
      TEDARegressor.relevanceList = TEDARegressor.alfa /np.sum(TEDARegressor.alfa)
      TEDARegressor.argMax.append(np.argmax(TEDARegressor.relevanceList))
      TEDARegressor.classIndex.append(TEDARegressor.alfa)
      #print("Alfa", TEDARegressor.alfa)
      #print("argmax", np.argmax(TEDARegressor.relevanceList))
      #print("relevance list: ", TEDARegressor.relevanceList)
          
      filter_used = TEDARegressor.c[np.argmax(TEDARegressor.relevanceList)].faux
      
      self.mergeClouds()
         
      
      #Best of all
      y_pred = filter_used.predict(X)
      #print("ypred_major", y_pred_major)        
      #print("___________")

      TEDARegressor.Ypred.append(y_pred)
      TEDARegressor.RLSF_Index.append(np.argmax(TEDARegressor.relevanceList))
      TEDARegressor.X_ant = X
      
    TEDARegressor.k=TEDARegressor.k+1