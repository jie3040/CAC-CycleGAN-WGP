import numpy as np

PATH='/home/liaowenjie/anaconda3//myfolder/GAN_for_UFD_new/dataset_case1_fft_512.npz'

data = np.load(PATH)

train_X = data['train_X']
train_Y = data['train_Y']
test_X = data['test_X']
test_Y = data['test_Y']

domain_A_train_X=train_X[9000:10000]
domain_A_train_Y=train_Y[9000:10000]

domain_B_train_X_0=train_X[0:1000]
domain_B_train_Y_0=train_Y[0:1000]

domain_B_train_X_1=train_X[1000:2000]
domain_B_train_Y_1=train_Y[1000:2000]

domain_B_train_X_2=train_X[2000:3000]
domain_B_train_Y_2=train_Y[2000:3000]

domain_B_train_X_3=train_X[3000:4000]
domain_B_train_Y_3=train_Y[3000:4000]

domain_B_train_X_4=train_X[4000:5000]
domain_B_train_Y_4=train_Y[4000:5000]

domain_B_train_X_5=train_X[5000:6000]
domain_B_train_Y_5=train_Y[5000:6000]

domain_B_train_X_6=train_X[6000:7000]
domain_B_train_Y_6=train_Y[6000:7000]

domain_B_train_X_7=train_X[7000:8000]
domain_B_train_Y_7=train_Y[7000:8000]

domain_B_train_X_8=train_X[8000:9000]
domain_B_train_Y_8=train_Y[8000:9000]


np.savez('/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD_new/dataset_fft_for_cyclegan_case1_512_test.npz', domain_A_train_X=domain_A_train_X,
                                                                           domain_A_train_Y=domain_A_train_Y,
                                                                           domain_B_train_X_0=domain_B_train_X_0,
                                                                           domain_B_train_Y_0=domain_B_train_Y_0,
                                                                           domain_B_train_X_1=domain_B_train_X_1,
                                                                           domain_B_train_Y_1=domain_B_train_Y_1,
                                                                           domain_B_train_X_2=domain_B_train_X_2,
                                                                           domain_B_train_Y_2=domain_B_train_Y_2,
                                                                           domain_B_train_X_3=domain_B_train_X_3,
                                                                           domain_B_train_Y_3=domain_B_train_Y_3,
                                                                           domain_B_train_X_4=domain_B_train_X_4,
                                                                           domain_B_train_Y_4=domain_B_train_Y_4,
                                                                           domain_B_train_X_5=domain_B_train_X_5,
                                                                           domain_B_train_Y_5=domain_B_train_Y_5,
                                                                           domain_B_train_X_6=domain_B_train_X_6,
                                                                           domain_B_train_Y_6=domain_B_train_Y_6,
                                                                           domain_B_train_X_7=domain_B_train_X_7,
                                                                           domain_B_train_Y_7=domain_B_train_Y_7,
                                                                           domain_B_train_X_8=domain_B_train_X_8,
                                                                           domain_B_train_Y_8=domain_B_train_Y_8,
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
                                                                         
                                                                          test_X=test_X, test_Y=test_Y)