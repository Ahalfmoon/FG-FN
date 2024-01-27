import numpy as np

import sys

sys.path.append('/home/zmx/ClinicalNotesICU-master/')

sys.path.append('/home/zmx/PyPOTS-main/')

from sklearn import metrics

from sklearn.preprocessing import StandardScaler


from pypots.data import  mcar, masked_fill #load_specific_dataset,

from pypots.imputation import SAITS

from pypots.imputation import BRITS

from pypots.imputation import LOCF

from pypots.imputation import LSTM

from pypots.imputation import MRNN

from pypots.imputation import Transformer

from pypots.classification import GRUD

import torch

from pypots.utils.metrics import cal_mae

from pypots.utils.metrics import cal_mse

from pypots.utils.metrics import cal_mre

from pypots.utils.metrics import cal_rmse

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def pyimpute(X,test_X,x_vectors,text_emb,labels,test_labels): #,text_emb,labels

  num_samples = len(X[0])

  X_intact, X, missing_mask, indicating_mask = mcar(X, 0.15) # hold out 10% observed values as ground truth     
 

  X = masked_fill(X, 1 - missing_mask, np.nan)

  
  test_X_intact, test_X, test_X_missing_mask, test_X_indicating_mask = mcar(test_X, 0.1) # hold out 10% observed values as ground truth 


  test_X = masked_fill(test_X, 1 - test_X_missing_mask, np.nan) 


  #print("X",np.array(X).shape,type(X))   #(11988,48,37)

 

  



  # Model training. This is PyPOTS showtime. ?? 

 

  



  #print("SATIs  0.15 x3 base 1impu+0pre loss  impu ")     

  #saits = SAITS(n_steps=48, n_features=59, n_layers=2, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=3)


  #saits.fit(X,x_vectors,text_emb,labels)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model


  #X_imputation = saits.impute(X,x_vectors,text_emb,labels)  
  
  #print("SATIs  impu")       
 

  #test_X_imputation = saits.impute(test_X,x_vectors,text_emb,test_labels)  

  


  
  #lstm = LSTM(n_steps=48, n_features=59,text_size=768, rnn_hidden_size=128,epochs=2)#n_steps, n_features,text_size 

 

  #lstm.fit(X,x_vectors,text_emb,labels)  #(X,text_emb,labels)  



  #X_imputation = lstm.impute(X,x_vectors,text_emb,labels)   #(X,text_emb,labels)  



  #test_X_imputation = lstm.impute(test_X,x_vectors,text_emb,test_labels)   #(X,text_emb,labels) 

   

  #print("bri 0.05 base var sim fusion loss 1+1+0  imp  ")  

  #brits = BRITS(n_steps=48, n_features=59,text_size=768, rnn_hidden_size=256,epochs=3)#n_steps, n_features,text_size



  #brits.fit(X,x_vectors,text_emb,labels)  #(X,text_emb,labels) 



  #X_imputation = brits.impute(X,x_vectors,text_emb,labels)   #(X,text_emb,labels)  

  

  #test_X_imputation = brits.impute(test_X,x_vectors,text_emb,test_labels)     

  

  

 # print("mrnn  0.05  base impu? ")           

 # mrnn = MRNN(n_steps=48, n_features=59,text_size=768, rnn_hidden_size=256,epochs=4)#n_steps, n_features,text_size 



  #mrnn.fit(X,x_vectors,text_emb,labels)  #(X,text_emb,labels) 



  #X_imputation = mrnn.impute(X,x_vectors,text_emb,labels)   #(X,text_emb,labels)  

  

  #test_X_imputation = mrnn.impute(test_X,x_vectors,text_emb,test_labels)

  

 # print("mrnn   imputation ") 

 

  

  #print("grud 0.05 1 imp+ 0pre baselines impu")   

  #grud = GRUD(n_steps=48, n_features=59, rnn_hidden_size=128,n_classes=2, epochs=2)#n_steps, n_features,text_size



  #grud.fit(X,x_vectors,text_emb,labels)  #(X,text_emb,labels)  



  #X_imputation,y = grud.classify(X,x_vectors,text_emb,labels)  

  

  #test_X_imputation,y2 = grud.classify(test_X,x_vectors,text_emb,test_labels) 



 # print("grud  impu ",imputation.shape)   

   

  print("transformer  0.05 encoder 128 var concat text biobert base 1*imp + loss")                              
 
  transformer = Transformer(n_steps=48,n_features=59,n_layers=2, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.4,epochs=3)                               #rnn_hidden_size=128  
    
  transformer.fit(X,x_vectors,text_emb,labels) 
    #print("input",X.shape,x_vectors.shape) #16574  11181
  X_imputation,prob,pred = transformer.impute(X,x_vectors,text_emb,labels)  
  
  '''
  train_activations = np.zeros(shape=labels.shape, dtype=float) 
  predlabels = np.zeros(shape=labels.shape, dtype=float) 
  test_activations = np.zeros(shape=test_labels.shape, dtype=float)
  torch.cuda.empty_cache()
  for task_id in range(4):      
    #print("labels??",labels.shape,task_id,labels[:, 0]) 
    label=labels[:, task_id]   
    transformer.fit(X,x_vectors,text_emb,label) 
    #print("input",X.shape,x_vectors.shape) #16574  11181
    X_imputation,prob,pred = transformer.impute(X,x_vectors,text_emb,label) #,prob  ,predlabel  
    train_activations[:, task_id] = prob[:]   
    #predlabels[:, task_id]=predlabel[:]  
    #test_label=test_labels[:, task_id] #16741
    #print("labels??",test_X.shape,labels.shape,test_label.shape) 
    #transformer.fit(test_X,x_vectors,text_emb,test_labels)
    #test_X_imputation,test_prob = brits.impute(test_X,x_vectors,text_emb,test_label)  # impute the originally-missing values and artificially-missing values
    #test_activations[:, task_id] = test_prob[:]
    
    print("transformer only imp loss ") 
    
  delta_data = np.abs(X_imputation - X_intact)
  np.savetxt("delta_data.csv", delta_data[0,:,:], delimiter=",") 
  
  #train_activations=torch.Tensor(train_activations).to(device)
  #test_activations=torch.Tensor(test_activations).to(device)
  
  print(type(labels), type(train_activations)) 
  labels=labels.cpu()
  #test_labels=test_labels.cpu()
  auc_scores = metrics.roc_auc_score(labels, train_activations, average=None) 
  
  #test_auc_scores = metrics.roc_auc_score(test_labels, test_activations, average=None) 
  
  ave_auc_micro = metrics.roc_auc_score(labels, train_activations,
                                          average="micro")
  apr= metrics.average_precision_score(labels, train_activations)
  print("aucpr",apr)
  
  #f1 = metrics.f1_score(labels, predlabels, average= "weighted")
  #f2 = metrics.f1_score(labels, predlabels, average= "micro")
  #f3 = metrics.f1_score(labels, predlabels, average= "macro")
  #print("f1 weighted",f1 ,f2,f3) 
  
  
  ave_auc_macro = metrics.roc_auc_score(labels, train_activations,
                                          average="macro")
  ave_auc_weighted = metrics.roc_auc_score(labels, train_activations,
                                             average="weighted")
  print("ROC AUC scores for labels:", auc_scores)
  #print("test_ ROC AUC scores for labels:", test_auc_scores)
  print("ave_auc_micro = {}".format(ave_auc_micro))
  print("ave_auc_macro = {}".format(ave_auc_macro))
  print("ave_auc_weighted = {}".format(ave_auc_weighted)) 
  
  '''

 # print("LOCF imp loss ") 

 # locf = LOCF() 

 

 # locf.fit(X) 



 # imputation = locf.impute(X)    



  #print("lstm bri 0.05 base var sim fusion loss 1+1+0  imp  ")     


  mae = cal_mae(X_imputation, X_intact, indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)



  mse = cal_mse(X_imputation, X_intact, indicating_mask)



  mre = cal_mre(X_imputation, X_intact, indicating_mask)



  rmse = cal_rmse(X_imputation, X_intact, indicating_mask)



  print("train impuations")



  print("mae",mae)



  print("mse",mse)



  print("mre",mre)



  print("rmse",rmse)

  

  

  mae = cal_mae(test_X_imputation, test_X_intact, test_X_indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)



  mse = cal_mse(test_X_imputation, test_X_intact, test_X_indicating_mask)



  mre = cal_mre(test_X_imputation, test_X_intact, test_X_indicating_mask)



  rmse = cal_rmse(test_X_imputation, test_X_intact, test_X_indicating_mask)



  print("test impuations")



  print("mae",mae)



  print("mse",mse)



  print("mre",mre)



  print("rmse",rmse)



  



  return X_imputation







