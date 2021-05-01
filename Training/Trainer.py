#!/usr/bin/env python
# coding: utf-8

# In[21]:


import sys,os
sys.path.append(os.path.abspath(r"E:\Study Material\Python_Machine_AI\Deep Learning_Lessons\Praktisch\Tensorflow\Projects\Semantic Segmentation FCN"))

# In[22]:


from Loss_Function.Custom_Loss_Function import Categorical_Cross_Entropy
import tensorflow as tf
import os


# In[23]:


K = tf.keras.backend
class OneCycleSchedule(tf.keras.callbacks.Callback):
    def __init__(self,iterations,max_rate,start_rate = None,last_iterations = None,last_rate = None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or max_rate / 1000
        self.iteration = 0
    def _interpolate(self,I1,I2,R1,R2):
        return ((R2 - R1)*(self.iteration - I1)) / (I2-I1+R1)
    def on_batch_begin(self,batch,logs):
        if self.iteration < self.half_iteration:     # lINEAR RATE INCREASE
            rate = self._interpolate(0,self.half_iteration,self.start_rate,self.max_rate)
        elif self.iteration < 2*self.half_iteration:
            rate = self._interpolate(self.half_iteration,2*self.half_iteration,self.max_rate,self.start_rate)
        else:  # Last few Iterations
            rate = self._interpolate(2*self.half_iteration,self.last_iterations,self.start_rate,self.last_rate)
            rate = max(rate,self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr,rate)


# In[24]:


def Get_id(path):
    import time
    fpath = os.path.join(path,"My_logs")
    id_ = time.strftime("run_%Y_%m_%D_%H_%M_%S")
    return os.path.join(fpath,id_)


# In[25]:


class Train_FCN:
    def __init__(self,config,model,Train_Generator,Val_Generator,Train_Size):
        self.config = config
        self.model = model
        self.Train_Generator = Train_Generator
        self.Val_Generator = Val_Generator
        self.epoch = self.config["train"]["epoch"]
        self.graph_path = self.config["Network"]["graph_path"]
        self.Train_Size = Train_Size
        self.Out_Weight_path = self.config["train"]["Output"]["weight"]
        self.callbacks = self.Give_Callbacks()
    
    def Give_Callbacks(self):
        Callbacks = []
        if self.config["Callbacks"]["Earlystop"]["Use_Early_Stop"]:
            earlystop = tf.keras.callbacks.EarlyStopping(patience=self.config["Callbacks"]["Earlystop"]["patience"],
                                                        monitor=self.config["Callbacks"]["Earlystop"]["monitor"],mode='auto')
            Callbacks.append(earlystop)
            
        if self.config["Callbacks"]["Model_Checkpoint_Best"]["enabled"]:  # Saving only best checkpoint
            path = self.config["Callbacks"]["Model_Checkpoint_Best"]["out_file"]
            monitor = self.config["Callbacks"]["Model_Checkpoint_Best"]["monitor"]
            Best_Checkpoint = tf.keras.callbacks.ModelCheckpoint(path,monitor=monitor,save_best_only=True
                                                                 ,save_weights_only=True,mode = "min",verbose = 1)
            Callbacks.append(Best_Checkpoint)
        
        if self.config["Callbacks"]["Model_Checkpoint_last"]["enabled"]:  # Saving only best checkpoint
            path = self.config["Callbacks"]["Model_Checkpoint_last"]["Out_file"]
            Last_Checkpoint = tf.keras.callbacks.ModelCheckpoint(path,monitor=monitor,save_best_only=False
                                                                 ,save_weights_only=True)
            Callbacks.append(Last_Checkpoint)
            
        if self.config["Callbacks"]["Tensorboard"]["enabled"]:
            Event_Out = Get_id(self.config["Callbacks"]["Tensorboard"]["event_file"])
            Callbacks.append(tf.keras.callbacks.TensorBoard(log_dir = Event_Out))
        
        if self.config["Callbacks"]["OneCycle"]["enabled"]:
            rate = self.config["Callbacks"]["OneCycle"]["max_rate"]
            Iterations = self.Train_Size // self.config["train"]["batch_size"] * self.epoch
            Callbacks.append(OneCycleSchedule(iterations=Iterations,max_rate=rate))
        return Callbacks
    
    def save_graph(self, model, graph_path):
        model_json = model.to_json()
        with open(graph_path, "w") as json_file:
            json_file.write(model_json)
            
    
    def Train(self):
        
        # Use Pretrained Weights
        if self.config["train"]["weight_initialization"]["use_pretrained"]:
            read_from = self.config["train"]["weight_initialization"]["restore_from"]
            print("Restoring Weights From: ",read_from)
            self.model.load_weights(read_from)
        else:
            print("Saving Weights",self.graph_path)
            self.save_graph(self.model,self.graph_path)
            
        # Compiling the model
        opt = self.config["train"]["optimizer"]
        lr = self.config["train"]["learning_rate"]
       
        if opt == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)        
        elif opt == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
        elif opt == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
        elif opt == 'adagrad':
            optimizer = tf.keras.optimizers.Adagrad(lr=lr, epsilon=None, decay=0.0)
        elif opt == 'adadelta':
            optimizer = tf.keras.optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0)
        else:
            raise Exception('Optimizer unknown')
        
        
        self.model.compile(loss = Categorical_Cross_Entropy(),optimizer = optimizer)
        
        # Fitting the Model
        Use_MultiProcessing = self.config["train"]["use_multiprocessing"]
        self.model.fit_generator(generator=self.Train_Generator, validation_data=self.Val_Generator, 
                                 epochs=self.epoch, verbose=1, max_queue_size=10, 
                                 workers=tf.data.AUTOTUNE, use_multiprocessing=Use_MultiProcessing, shuffle=False, 
                                 callbacks=self.callbacks)
        #save weights
        print("Saving weights in", self.Out_Weight_path)
        self.model.save(self.Out_Weight_path)
        


# In[ ]:




