from keras.models import load_model

import numpy as np
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import random
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score








def scalar_stand(Train_X, Test_X):
    # 用训练集标准差标准化训练集以及测试集
    scalar_train = preprocessing.StandardScaler().fit(Train_X)
    #scalar_test = preprocessing.StandardScaler().fit(Test_X)
    Train_X = scalar_train.transform(Train_X)
    Test_X = scalar_train.transform(Test_X)
    return Train_X, Test_X

def samlpe_generation_feed_svm(add_quantity,test_x,test_y,generator,domain_A_train_x, domain_A_train_y,domain_B_train_x,domain_B_train_y):
    
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
        generated_sample=generator.predict([domain_A_train_x[indices_array],labels])
        Generated_samples.append(generated_sample)
        Labels.append(labels)
     

    Generated_samples=np.array(Generated_samples).reshape(-1, 512)
    Labels=np.array(Labels).reshape(-1, 1)

    domain_B_train_x_feed=np.concatenate([domain_B_train_x[0:5],
                       domain_B_train_x[100:105],
                       domain_B_train_x[200:205],
                       domain_B_train_x[300:305],
                       domain_B_train_x[400:405],
                       domain_B_train_x[500:505],
                       domain_B_train_x[600:605],
                       domain_B_train_x[700:705],
                       domain_B_train_x[800:805]],axis=0)

    domain_B_train_y_feed=np.concatenate([domain_B_train_y[0:5],
                       domain_B_train_y[100:105],
                       domain_B_train_y[200:205],
                       domain_B_train_y[300:305],
                       domain_B_train_y[400:405],
                       domain_B_train_y[500:505],
                       domain_B_train_y[600:605],
                       domain_B_train_y[700:705],
                       domain_B_train_y[800:805]],axis=0)                   

    systhesis_samples=np.vstack((domain_B_train_x_feed,Generated_samples))
    systhesis_labels=np.vstack((domain_B_train_y_feed,Labels))
    
    Systhesis_total_samples=np.vstack((systhesis_samples,domain_A_train_x))
    Systhesis_total_labels=np.vstack((systhesis_labels,domain_A_train_y))
    
    train_x=Systhesis_total_samples
    train_y=Systhesis_total_labels
    
    train_X,test_X=scalar_stand(train_x, test_x)
    train_Y, test_Y=train_y,test_y
    
    estimator = SVR(kernel='linear')
    classifier = SVC(C=0.2, gamma=0.001,kernel = 'rbf', random_state =0)
    
    classifier.fit(train_X, train_Y)
    
    Y_pred = classifier.predict(test_X)
    
    cm = confusion_matrix(test_Y, Y_pred)
    
    accuracy = accuracy_score(test_Y, Y_pred)
    
    print(cm)
    return accuracy
  
  
  



PATH='/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD_new/dataset_fft_for_cyclegan_case1_512.npz'
data = np.load(PATH)

domain_A_train_X=data['domain_A_train_X'][0:1000]
domain_A_train_Y=data['domain_A_train_Y'][0:1000]

domain_B_train_X_0=data['domain_B_train_X_0']
domain_B_train_Y_0=data['domain_B_train_Y_0']

domain_B_train_X_1=data['domain_B_train_X_1']
domain_B_train_Y_1=data['domain_B_train_Y_1']

domain_B_train_X_2=data['domain_B_train_X_2']
domain_B_train_Y_2=data['domain_B_train_Y_2']

domain_B_train_X_3=data['domain_B_train_X_3']
domain_B_train_Y_3=data['domain_B_train_Y_3']

domain_B_train_X_4=data['domain_B_train_X_4']
domain_B_train_Y_4=data['domain_B_train_Y_4']

domain_B_train_X_5=data['domain_B_train_X_5']
domain_B_train_Y_5=data['domain_B_train_Y_5']

domain_B_train_X_6=data['domain_B_train_X_6']
domain_B_train_Y_6=data['domain_B_train_Y_6']

domain_B_train_X_7=data['domain_B_train_X_7']
domain_B_train_Y_7=data['domain_B_train_Y_7']

domain_B_train_X_8=data['domain_B_train_X_8']
domain_B_train_Y_8=data['domain_B_train_Y_8']

test_X=data['test_X']
test_Y=data['test_Y']

domain_B_train_X=np.concatenate([domain_B_train_X_0, 
                        domain_B_train_X_1, 
                        domain_B_train_X_2,
                        domain_B_train_X_3,
                        domain_B_train_X_4,
                        domain_B_train_X_5,
                        domain_B_train_X_6,
                        domain_B_train_X_7,
                        domain_B_train_X_8
                        ], axis=0)

domain_B_train_Y=np.concatenate([domain_B_train_Y_0, 
                        domain_B_train_Y_1, 
                        domain_B_train_Y_2,
                        domain_B_train_Y_3,
                        domain_B_train_Y_4,
                        domain_B_train_Y_5,
                        domain_B_train_Y_6,
                        domain_B_train_Y_7,
                        domain_B_train_Y_8
                        ], axis=0)

generator = load_model('/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD_new/CycleGAN_g_AB_model_1100_case1.h5')

print(samlpe_generation_feed_svm(0 ,test_X,test_Y,generator,domain_A_train_X,domain_A_train_Y,domain_B_train_X,domain_B_train_Y))
    