from __future__ import print_function, division
from pickle import FALSE

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate,multiply
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D,Embedding,concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.losses import mean_absolute_error

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers.legacy import Adam
import datetime
#import matplotlib.pyplot as plt
import sys
#from data_loader import DataLoader
import numpy as np
import os
#import preprocess
import cyclegan_sample_generation_new_and_svm
from sklearn.utils import shuffle
from tensorflow.keras.models import save_model
import pandas as pd
from tensorflow.keras.optimizers.legacy import RMSprop
from tensorflow.keras import backend as K
from functools import partial
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Concatenate
from tensorflow.python.framework.ops import disable_eager_execution




class RandomWeightedAverage(Concatenate):
    """Provides a (random) weighted average between real and generated samples"""
    def call(self, inputs):
        alpha = K.random_uniform((45, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


    
    


class CycleGAN():
    def __init__(self):
        self.data_lenth=512
        self.sample_shape=(self.data_lenth,)
        self.num_classes=10
        self.latent_dim = 100
        self.batch_size=45
        self.dataset_name='fualt_diagnosis'
        self.n_critic = 1
        
        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 32

        # Loss weights
        self.lambda_adv = 1
        self.lambda_cycle = 10                 # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle       # Identity loss
        self.lambda_fake_A_classification=1
        self.lambda_fake_B_classification=1

        #self.d_optimizer_1 = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.5)
        #self.d_optimizer_2 = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.5)

        self.d_optimizer_1 = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        self.d_optimizer_2 = tf.keras.optimizers.RMSprop(learning_rate=0.001)
 
        #optimizer = Adam(0.0005, 0.5)
        optimizer = RMSprop(lr=0.001)
        # Build and compile the discriminators
        
        self.d_1 = self.build_discriminator() # d_1 is for normal/healthy samples
        self.d_2 = self.build_discriminator() # d_2 is for fault samples

        self.LAMBDA_GP=10
        self.LABEL_LABEL=1
              
        
        # Build the generators
        
        self.g_AB = self.build_generator() #g_AB is for generating fault samples
        self.g_BA = self.build_generator() #g_BA is for generating healthy samples
        
        # Freeze generator's layers while training critic
        
        # Input samples from both domains
        # A is healthy domain, B is faulty domain
        
        sample_A=Input(shape=self.sample_shape)
        sample_B=Input(shape=self.sample_shape)

        label_B = Input(shape=(1,))
        label_A= Input(shape=(1,))
       
        # Discriminator determines validity of the real and fake samples
    
        #d1
        
        
        
        
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------
        
        
        
        # For the combined model we will only train the generators
        
        self.d_1.trainable = False
        self.d_2.trainable = False
        
        self.g_AB.trainable = True
        self.g_BA.trainable = True


        
        
        
        
        # Translate samples to the other domain
        
        fake_B=self.g_AB([sample_A,label_B])
        
        
        fake_A=self.g_BA([sample_B,label_A])
        
        # Translate samples back to original domain
        reconstr_A=self.g_BA([fake_B,label_A])
        reconstr_B=self.g_AB([fake_A,label_B])
        
        # Identity mapping of samples
        
        sample_A_id=self.g_BA([sample_A,label_A])
        sample_B_id=self.g_AB([sample_B,label_B])
        
        
        
        # Discriminators determines validity of translated samples
        
        valid_A_for_fake_A,label_A_for_fake_A = self.d_1(fake_A)
        valid_B_for_fake_B,label_B_for_fake_B = self.d_2(fake_B)

        # Compute L1 cycle-consistency losses
        cycle_loss_A = mean_absolute_error(sample_A, reconstr_A)
        cycle_loss_B = mean_absolute_error(sample_B, reconstr_B)

        # Compute L1 identity losses
        id_loss_A = mean_absolute_error(sample_A, sample_A_id)
        id_loss_B = mean_absolute_error(sample_B, sample_B_id)

        def zero_loss(y_true, y_pred):
          return y_pred


      
        
        # Combined model trains generators to fool discriminators
        

        

        self.combined = Model(inputs=[sample_A, label_A,sample_B,label_B],
                              outputs=[valid_A_for_fake_A, valid_B_for_fake_B,
                              cycle_loss_A, cycle_loss_B, id_loss_A, id_loss_B,
                              label_A_for_fake_A,label_B_for_fake_B])
        
        self.combined.compile(loss=[self.wasserstein_loss, self.wasserstein_loss,
                                    zero_loss, zero_loss, zero_loss, zero_loss,
                                    'sparse_categorical_crossentropy','sparse_categorical_crossentropy']
                                    ,
                            loss_weights=[  self.lambda_adv, self.lambda_adv,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ,
                                            self.lambda_fake_A_classification, self.lambda_fake_B_classification],
                            optimizer=optimizer)
        
    def wasserstein_loss(self, y_true, y_pred):
            return K.mean(y_true * y_pred)



    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=5, normalization=True, zeroPadding=False):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            
            if zeroPadding:

              d = ZeroPadding2D(padding=((0,1),(0,1)))(d)
            if normalization:

              d = BatchNormalization(momentum=0.8)(d)

            d = LeakyReLU(alpha=0.2)(d)
            d = Dropout(0.25)(d)
            
            
            return d
        
        
        sample = Input(shape=self.sample_shape)
        

        # Reshape the sample
        reshaped_sample = Reshape((32, 16, 1))(sample)
        
        
        
        d1 = d_layer(reshaped_sample, self.df, normalization=False)

        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8, normalization=False)

        x = Flatten()(d4)
        validity = Dense(1)(x)
        label = Dense(self.num_classes, activation="softmax")(x)
        
        
        return Model(sample, [validity, label])
    
    def build_generator(self):
        """U-Net Generator"""
        
        
        def conv2d(layer_input,filters, f_size=5):

            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = BatchNormalization()(d)
            return d
           
        def deconv2d(layer_input, skip_input, filters, f_size=5, dropout_rate=0):
            
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u
        

        sample = Input(shape=self.sample_shape)
        label = Input(shape=(1,), dtype='int32')

        # Reshape the sample
        reshaped_sample = Reshape((32, 16, 1))(sample)
        embedded_label = Embedding(self.num_classes, 32*16)(label)
        
        flattened_label = Reshape((32, 16, 1))(embedded_label)
        
        concatenated = concatenate([reshaped_sample, flattened_label], axis=-1)

        d0=concatenated
        
        #Downsampling
        d1=conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        
        #Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        u4 = Conv2D(1, kernel_size=5, strides=1, padding='same')(u4)

        output_sample=Flatten()(u4)

        output_sample = Dense(units=512)(output_sample)

        #output_sample=BatchNormalization()(output_sample)

        

        
        return Model([sample,label], output_sample)


    
    
    def gradient_penalty_loss(self, gradients):
      
      gradients_sqr = tf.square(gradients)
      gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=tf.range(1, len(gradients_sqr.shape)))
      gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
      gradient_penalty = tf.square(1 - gradient_l2_norm)
      return tf.reduce_mean(gradient_penalty)



    
    def train(self, epochs, batch_size): 
        
        start_time = datetime.datetime.now()
        
        # Adversarial loss ground truths
        valid = -np.ones((batch_size,1) )
        fake = np.ones((batch_size,1) )
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        
        #path='/home/liaowenjie/myfolder/GAN_for_UFD/dataset/'
        
        
        #train_X, train_Y, valid_X, valid_Y, test_X, test_Y = preprocess.prepro(d_path=path,
                                                                #length=1024,
                                                                #number=1000,
                                                                #normal=False,
                                                                #rate=[0.5, 0.25, 0.25],
                                                                #enc=False,
                                                                #enc_step=28)
        

        PATH='/content/drive/MyDrive/Colab_Notebooks/myfolder/GAN_for_UFD/dataset_fft_for_cyclegan_case1_512.npz'
        data = np.load(PATH)

        domain_A_train_X=data['domain_A_train_X']
        domain_A_train_Y=data['domain_A_train_Y']

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

        domain_B_train_X=np.concatenate([domain_B_train_X_0[0:5], 
                        domain_B_train_X_1[0:5], 
                        domain_B_train_X_2[0:5],
                        domain_B_train_X_3[0:5],
                        domain_B_train_X_4[0:5],
                        domain_B_train_X_5[0:5],
                        domain_B_train_X_6[0:5],
                        domain_B_train_X_7[0:5],
                        domain_B_train_X_8[0:5]
                        ], axis=0)

        domain_B_train_Y=np.concatenate([domain_B_train_Y_0[0:5], 
                        domain_B_train_Y_1[0:5], 
                        domain_B_train_Y_2[0:5],
                        domain_B_train_Y_3[0:5],
                        domain_B_train_Y_4[0:5],
                        domain_B_train_Y_5[0:5],
                        domain_B_train_Y_6[0:5],
                        domain_B_train_Y_7[0:5],
                        domain_B_train_Y_8[0:5]]
                        , axis=0)






        
        domain_A_train_X_900=domain_A_train_X[0:900]
        domain_A_train_Y_900=domain_A_train_Y[0:900]
       
                
             
        num_batches=int(domain_A_train_X_900.shape[0]/45)
        
      

        accuracy_list=[]


        for epoch in range(epochs):
            
            for batch_i in range(num_batches):
                
                # Select a batch of samples from domian A
                
                start_i =batch_i * 45
                end_i=(batch_i + 1) * 45


                batch_samples_A=domain_A_train_X_900[start_i:end_i]
                batch_labels_A=domain_A_train_Y_900[start_i:end_i]
                batch_samples_B=domain_B_train_X
                batch_labels_B=domain_B_train_Y

                #self.g_AB.trainable = False
                #self.g_BA.trainable = False

                self.d_1.trainable = True
                self.d_2.trainable = True
        

                for _ in range(self.n_critic):

                  #train d_1

                  #fake_A=self.g_BA([batch_samples_B,batch_labels_A])
                  #interpolated_sample_1 = RandomWeightedAverage()([batch_samples_A, fake_A])

                  with tf.GradientTape(persistent=True) as tape1:

                    fake_A=self.g_BA([batch_samples_B,batch_labels_A])
                    interpolated_sample_1 = RandomWeightedAverage()([batch_samples_A, fake_A])

                    d_pred_real,d_label_real = self.d_1(batch_samples_A)
                    d_pred_fake,d_label_fake = self.d_1(fake_A)
                    d_pred_inter,d_label_inter =self.d_1(interpolated_sample_1)

                    d_loss_real = self.wasserstein_loss(valid, d_pred_real)
                    d_loss_fake = self.wasserstein_loss(fake, d_pred_fake)

                    gradients = tape1.gradient(d_pred_inter, interpolated_sample_1)
            
                    gradient_penalty= self.gradient_penalty_loss(gradients)

                    label_loss_real = tf.keras.losses.sparse_categorical_crossentropy(batch_labels_A, d_label_real)
                    label_loss_fake = tf.keras.losses.sparse_categorical_crossentropy(batch_labels_A, d_label_fake)

                    d_loss_1 = d_loss_real + d_loss_fake + self.LAMBDA_GP * gradient_penalty + self.LABEL_LABEL * (label_loss_real + label_loss_fake)
                                    
                  #print(d_loss_1)
                  average_d_loss_1 = tf.reduce_mean(d_loss_1)

                  grads_1 = tape1.gradient(d_loss_1, self.d_1.trainable_weights)

                  #for g, v in zip(grads_1, self.d_1.trainable_weights):
                    #print(f"Gradient for {v.name}: {g}")
                  
                  self.d_optimizer_1.apply_gradients(zip(grads_1, self.d_1.trainable_weights))

                  del tape1


                  #train d_2

                  with tf.GradientTape(persistent=True) as tape2:

                    fake_B=self.g_AB([batch_samples_A,batch_labels_B])
                    interpolated_sample_2 = RandomWeightedAverage()([batch_samples_B, fake_B])

                    d_pred_real,d_label_real = self.d_2(batch_samples_B)
                    d_pred_fake,d_label_fake = self.d_2(fake_B)
                    d_pred_inter,d_label_inter =self.d_2(interpolated_sample_2)

                    d_loss_real = self.wasserstein_loss(valid, d_pred_real)
                    d_loss_fake = self.wasserstein_loss(fake, d_pred_fake)

                    gradients = tape2.gradient(d_pred_inter, interpolated_sample_2)
            
                    gradient_penalty= self.gradient_penalty_loss(gradients)
            

                    label_loss_real = tf.keras.losses.sparse_categorical_crossentropy(batch_labels_B, d_label_real)
                    label_loss_fake = tf.keras.losses.sparse_categorical_crossentropy(batch_labels_B, d_label_fake)

                    d_loss_2 = d_loss_real + d_loss_fake + self.LAMBDA_GP * gradient_penalty + self.LABEL_LABEL * (label_loss_real + label_loss_fake)

                  average_d_loss_2 = tf.reduce_mean(d_loss_2)

                  grads_2 = tape2.gradient(d_loss_2, self.d_2.trainable_weights)

                  self.d_optimizer_2.apply_gradients(zip(grads_2, self.d_2.trainable_weights))

                  del tape2







              
                
            
                
                  #d_1_loss = self.d_1_model.train_on_batch([batch_samples_A, batch_samples_B, batch_labels_A], [valid, fake, dummy,batch_labels_A,batch_labels_A])

                
                  #d_2_loss = self.d_2_model.train_on_batch([batch_samples_B, batch_samples_A, batch_labels_B], [valid, fake, dummy,batch_labels_B,batch_labels_B])

                


                

                
                # Total disciminator loss
                #d_loss = 0.5 * np.add(dA_loss, dB_loss)

                self.d_1.trainable = False
                self.d_2.trainable = False
                

                self.g_AB.trainable = True
                self.g_BA.trainable = True

              

                # ------------------
                #  Train Generators
                # ------------------
                
                g_loss = self.combined.train_on_batch([batch_samples_A,batch_labels_A,batch_samples_B,batch_labels_B],
                                                        [valid, valid,
                                                        dummy, dummy,
                                                        dummy, dummy,
                                                        batch_labels_A,batch_labels_B])
                #print(g_loss)
                elapsed_time = datetime.datetime.now() - start_time

                #self.d_1.trainable = True
                #self.d_2.trainable = True
                
                # Plot the progress
                
                print ("[Epoch %d/%d] [Batch %d/%d] [D_1 loss: %f][D_2 loss: %f][G loss: %05f, adv: %05f, recon: %05f, id: %05f,classA:%05f,classB:%05f] time: %s " \
                                                                            % ( epoch, epochs,
                                                                                batch_i, num_batches,
                                                                                average_d_loss_1,
                                                                                average_d_loss_2,
                                                                                g_loss[0],
                                                                                np.mean(g_loss[1:3]),
                                                                                np.mean(g_loss[3:5]),
                                                                                np.mean(g_loss[5:7]),
                                                                                g_loss[7],g_loss[8],
                                                                                elapsed_time))
            #val_d_loss = 0
            #val_g_loss = 0
            #num_val_batches = int(domain_A_train_X.shape[0]/batch_size)

            #D_loss_data_point.append(d_loss)
            #G_loss_data_point.append(g_loss[0])
            #adv_loss_data_point.append(np.mean(g_loss[1:3]))
            #recon_loss_data_point.append(np.mean(g_loss[3:5]))
            #id_loss_data_point.append(np.mean(g_loss[5:7]))



            #accuracy_for_svm=cyclegan_sample_generation_new_and_svm.samlpe_generation_feed_svm(test_X,test_Y,gan.g_AB,domain_A_train_X,domain_A_train_Y,domain_B_train_X,domain_B_train_Y)

            #print("[Epoch %d/%d] [Accuracy_for_SVM: %f]"\
              #%(epoch, epochs,accuracy_for_svm))
            
            #accuracy_list.append(accuracy_for_svm)
            
            if epoch % 100 == 0:

                #data = {
                    #"Epoch": [epoch] * len(D_loss_data_point),
                    #"D_loss": D_loss_data_point,
                    #"G_loss": G_loss_data_point,
                    #"Adv_loss": adv_loss_data_point,
                    #"Recon_loss": recon_loss_data_point,
                    #"ID_loss": id_loss_data_point
                #}
                #df_to_append = pd.DataFrame(data)
                #df = df.append(df_to_append, ignore_index=True)

                #file_path = f"/content/drive/MyDrive/Colab_Notebooks/myfolder/GAN_for_UFD/data_points_{epoch}.csv"
                
                #df.to_csv(file_path, index=False) 

                # Save the discriminator model
                #save_model(gan.d_1, f"/content/drive/MyDrive/Colab_Notebooks/myfolder/GAN_for_UFD/CycleGAN_d_1_model_{epoch}.h5")
    
                #save_model(gan.d_2, f"/content/drive/MyDrive/Colab_Notebooks/myfolder/GAN_for_UFD/CycleGAN_d_2_model_{epoch}.h5")
            
                # Save the generator model  

                accuracy_for_svm=cyclegan_sample_generation_new_and_svm.samlpe_generation_feed_svm(995,test_X,test_Y,gan.g_AB,domain_A_train_X,domain_A_train_Y,domain_B_train_X,domain_B_train_Y) 

                print("[Epoch %d/%d] [Accuracy_for_SVM: %f]"\
                  %(epoch, epochs,accuracy_for_svm))
            
                accuracy_list.append(accuracy_for_svm) 

                print(accuracy_list)  
      
                save_model(gan.g_AB, f"/content/drive/MyDrive/Colab_Notebooks/myfolder/GAN_for_UFD/CycleGAN_g_AB_model_{epoch}_case1.h5")
    
                #save_model(gan.g_BA, f"/content/drive/MyDrive/Colab_Notebooks/myfolder/GAN_for_UFD/CycleGAN_g_BA_model_{epoch}.h5")

                #np.savetxt(f"/content/drive/MyDrive/Colab_Notebooks/myfolder/GAN_for_UFD/accuracy_list_for_SVM_{epoch}.txt", accuracy_list)



                

                    



if __name__ == '__main__':

    
    import tensorflow as tf
    tf.compat.v1.experimental.output_all_intermediates(True)
    
    
    
    gan = CycleGAN()
    #batch_size 单个domain是5 实际上 是5*9=45
    gan.train(epochs=5000, batch_size=45)
    
    # Save the discriminator model
    #save_model(gan.d_1, '/content/drive/MyDrive/Colab_Notebooks/myfolder/GAN_for_UFD/CycleGAN_d_1_model.h5')
    
    #save_model(gan.d_2, '/content/drive/MyDrive/Colab_Notebooks/myfolder/GAN_for_UFD/CycleGAN_d_2_model.h5')
            
    # Save the generator model      
      
    save_model(gan.g_AB, '/content/drive/MyDrive/Colab_Notebooks/myfolder/GAN_for_UFD/CycleGAN_g_AB_case1_model.h5')
    
    #save_model(gan.g_BA, '/content/drive/MyDrive/Colab_Notebooks/myfolder/GAN_for_UFD/CycleGAN_g_BA_model.h5')       
    
        
    
        
        
            
        
        
        
    
    
    
    
    
        
        
        