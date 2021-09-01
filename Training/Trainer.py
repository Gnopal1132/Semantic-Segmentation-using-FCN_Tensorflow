from Loss_Function.Custom_Loss_Function import categorical_cross_entropy
import tensorflow as tf
import os,numpy as np


K = tf.keras.backend


class one_cycle(tf.keras.callbacks.Callback):
    def __init__(self, iterations, max_rate,start_rate=None, last_iteration=None, last_rate=None,):
        super().__init__()
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iteration = last_iteration or iterations // 10 + 1
        self.half_iteration = (self.iterations - self.last_iteration) // 2
        self.last_rate = last_rate or max_rate / 1000
        self.iteration = 0
        self.loss = []
        self.rate = []

    def __interpolate(self, start_iteration, final_iteration, start_rate, final_rate):
        return ((final_rate - start_rate)*(self.iteration - start_iteration))/((final_iteration-start_iteration) + start_rate)

    def on_batch_begin(self, batch, logs=None):
        if self.iteration < self.half_iteration:
            rate = self.__interpolate(0, self.half_iteration,self.start_rate,self.max_rate)
        elif self.iteration < 2*self.half_iteration:
            rate = self.__interpolate(self.half_iteration, 2*self.half_iteration, self.max_rate, self.start_rate)
        else:
            rate = self.__interpolate(2*self.half_iteration, self.iterations,self.start_rate, self.last_rate)
            rate = max(rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, rate)

    def on_epoch_end(self, epoch, logs=None):
        self.loss.append(logs['loss'])
        self.rate.append(K.get_value(self.model.optimizer.lr))


class TrainFcn:
    def __init__(self, config, trainsize, model, Train_Generator, Val_Generator):
        self.config = config
        self.model = model
        self.Train_Generator = Train_Generator
        self.Val_Generator = Val_Generator
        self.train_size = trainsize
        self.epoch = self.config["train"]["epoch"]
        self.graph_path = self.config["Network"]["graph_path"]
        self.batch_size = self.config["train"]["batch_size"]
        self.Out_Weight_path = self.config["train"]["Output"]["weight"]
        self.callbacks = self.give_callbacks()

    def get_id(self, path):
        import time
        fpath = os.path.join(path, "My_logs")
        id_ = time.strftime("run_%Y_%m_%D_%H_%M_%S")
        return os.path.join(fpath, id_)

    def give_callbacks(self):

        callbacks = []
        if self.config["Callbacks"]["Earlystop"]["Use_Early_Stop"]:
            patience = self.config["Callbacks"]["Earlystop"]["patience"]
            monitor = self.config["Callbacks"]["Earlystop"]["monitor"]
            earlystop = tf.keras.callbacks.EarlyStopping(patience=patience, monitor=monitor)
            callbacks.append(earlystop)
            
        if self.config["Callbacks"]["Model_Checkpoint_Best"]["enabled"]:  # Saving only best checkpoint
            path = self.config["Callbacks"]["Model_Checkpoint_Best"]["out_file"]
            monitor = self.config["Callbacks"]["Model_Checkpoint_Best"]["monitor"]
            best_checkpoint = tf.keras.callbacks.ModelCheckpoint(path, monitor=monitor, save_best_only=True
                                                                 , save_weights_only=True)
            callbacks.append(best_checkpoint)
        
        if self.config["Callbacks"]["Model_Checkpoint_last"]["enabled"]:  # Saving only best checkpoint
            path = self.config["Callbacks"]["Model_Checkpoint_last"]["Out_file"]
            monitor = self.config["Callbacks"]["Model_Checkpoint_last"]["monitor"]
            last_checkpoint = tf.keras.callbacks.ModelCheckpoint(path, monitor=monitor, save_best_only=False
                                                                 , save_weights_only=True)
            callbacks.append(last_checkpoint)
            
        if self.config["Callbacks"]["Tensorboard"]["enabled"]:
            event_out = self.get_id(self.config["Callbacks"]["Tensorboard"]["event_file"])
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=event_out))
        
        if self.config["Callbacks"]["OneCycle"]["enabled"]:
            rate = self.config["Callbacks"]["OneCycle"]["max_rate"]
            iterations = np.ceil(self.train_size/self.batch_size) * self.epoch
            callbacks.append(one_cycle(iterations=iterations, max_rate=rate))
        return callbacks

    def save_graph(self, model, graph_path):
        model_json = model.to_json()
        with open(graph_path, "w") as json_file:
            json_file.write(model_json)
            
    
    def train(self):
        
        # Use Pretrained Weights
        if self.config["train"]["weight_initialization"]["use_pretrained"]:
            read_from = self.config["train"]["weight_initialization"]["restore_from"]
            print("Restoring Weights From: ", read_from)
            self.model.load_weights(read_from)
        else:
            print("Saving Weights", self.graph_path)
            self.save_graph(self.model, self.graph_path)
            
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

        self.model.compile(loss=categorical_cross_entropy(), optimizer=optimizer)
        
        # Fitting the Model
        use_multi_processing = self.config["train"]["use_multiprocessing"]
        self.model.fit_generator(generator=self.Train_Generator, validation_data=self.Val_Generator, 
                                 epochs=self.epoch, verbose=1, max_queue_size=10, 
                                 workers=tf.data.AUTOTUNE, use_multiprocessing=use_multi_processing, shuffle=False,
                                 callbacks=self.callbacks)
        #save weights
        print("Saving weights in", self.Out_Weight_path)
        self.model.save(self.Out_Weight_path)



