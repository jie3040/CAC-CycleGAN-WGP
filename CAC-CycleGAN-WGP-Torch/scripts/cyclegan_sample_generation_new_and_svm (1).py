
import torch
import numpy as np
import random
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def to_numpy(data):
    """将 tensor 或 tensor 列表转换为 numpy 数组"""
    if isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()
    elif isinstance(data, list):
        return np.array([to_numpy(x) for x in data])
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)

def scalar_stand(Train_X, Test_X):
    # 用训练集标准差标准化训练集以及测试集
    scalar_train = preprocessing.StandardScaler().fit(Train_X)
    #scalar_test = preprocessing.StandardScaler().fit(Test_X)
    Train_X = scalar_train.transform(Train_X)
    Test_X = scalar_train.transform(Test_X)
    return Train_X, Test_X

def samlpe_generation_feed_svm(add_quantity,test_x,test_y,generator,domain_A_train_x, domain_A_train_y,domain_B_train_x,domain_B_train_y, c=0.2, g=0.001, ):
    
    Generated_samples=[]
    Labels=[]
    
    if add_quantity==0:
      
      Generated_samples=[]
      Labels=[]
      
    else: 
      
      for i in range(9):

        labels = np.full((add_quantity, 1), i)
        selected_indices = list(range(add_quantity))
        indices_array = np.array(selected_indices)

        domain_A_batch = torch.from_numpy(domain_A_train_x[indices_array]).float()
        labels_batch = torch.from_numpy(labels).long()
        
        # 如果使用 GPU
        if torch.cuda.is_available():
            domain_A_batch = domain_A_batch.cuda()
            labels_batch = labels_batch.cuda()
            
        generated_sample=generator(domain_A_batch,labels_batch)
        Generated_samples.append(generated_sample)
        Labels.append(labels)
     

    Generated_samples=to_numpy(Generated_samples).reshape(-1, 512)
    Labels=to_numpy(Labels).reshape(-1, 1)

    domain_B_train_x_feed=domain_B_train_x

    domain_B_train_y_feed=domain_B_train_y                   

    systhesis_samples=np.vstack((domain_B_train_x_feed,Generated_samples))
    systhesis_labels=np.vstack((domain_B_train_y_feed,Labels))
    
    Systhesis_total_samples=np.vstack((systhesis_samples,domain_A_train_x))
    Systhesis_total_labels=np.vstack((systhesis_labels,domain_A_train_y))
    
    train_x=Systhesis_total_samples
    train_y=Systhesis_total_labels
    
    train_X,test_X=scalar_stand(train_x, test_x)
    train_Y, test_Y=train_y,test_y
    
    estimator = SVR(kernel='linear')
    classifier = SVC(C=c, gamma=g,kernel = 'rbf', random_state =0)
    
    classifier.fit(train_X, train_Y)
    
    Y_pred = classifier.predict(test_X)
    
    cm = confusion_matrix(test_Y, Y_pred)
    
    accuracy = accuracy_score(test_Y, Y_pred)
    
    print(cm)
    return accuracy
  
  
  


    
