import keras
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import Concatenate, Add, Dot, concatenate, add, dot, multiply
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from keras import initializers
from keras import backend as K
import numpy as np
import random
import sys, argparse
import pandas as pd
from data_interpreter_Keras_aux import dataInterpreter, metaDataEndomondo
import pickle
from math import floor
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import datetime
from tqdm import tqdm
import os 
from keras.optimizers import Adagrad, Adam, SGD, RMSprop

def parse():
    parser = argparse.ArgumentParser(description='context2seq-NAT')
    parser.add_argument('--patience', default=10, type=int, help='patience for early stop') # [3,5,10,20]
    parser.add_argument('--epoch', default=50, type=int, help='max epoch') # [50,100]
    parser.add_argument('--attributes', default="userId,sport,gender", help='input attributes')
    parser.add_argument('--input_attributes', default="distance,altitude,time_elapsed", help='input attributes')
    parser.add_argument('--pretrain', action='store_true', help='use pretrain model')
    parser.add_argument('--temporal', action='store_true', help='use temporal input')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size') # 
    parser.add_argument('--attr_dim', default=5, type=int, help='attribute dimension') # 
    parser.add_argument('--hidden_dim', default=64, type=int, help='rnn hidden dimension') # 
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate') # 0.001 for fine tune; 0.005 for general
    parser.add_argument('--user_reg', default=0.0, type=float, help='user attribute reg') 
    parser.add_argument('--sport_reg', default=0.01, type=float, help='sport attribute reg') 
    parser.add_argument('--gender_reg', default=0.05, type=float, help='gender attribute reg') 
    parser.add_argument('--out_reg', default=0.0, type=float, help='final output layer reg') 
    parser.add_argument('--pretrain_file', default="", help='pretrain file') 

    args = parser.parse_args()
    return args

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 


class keras_endoLSTM(object):
    def __init__(self, args, newModel):

        if newModel:
            self.model_save_location = "./model_states/"
            self.summaries_dir = path + "./logs/"
            self.data_path = "endomondoHR_proper.json"
            self.trainValidTestFN = self.data_path.split(".")[0] + "_temporal_dataset.pkl"

            self.patience = args.patience # [3,5,10]
            self.max_epochs = args.epoch # [50,100]
            print("patience={}".format(self.patience))
            print("max_epochs={}".format(self.max_epochs))

            self.zMultiple = 5
            self.attrFeatures = args.attributes.split(',')
            self.user_dim = args.attribute_dim
            self.sport_dim = args.attribute_dim
            self.gender_dim = args.attribute_dim
            self.includeUser = 'userId' in self.attrFeatures
            self.includeSport = 'sport' in self.attrFeatures
            self.includeGender = 'gender' in self.attrFeatures

            self.pretrain = args.pretrain
            self.includeTemporal = args.temporal
            self.pretrain_model_file_name = args.pretrain_file
            
            self.lr = args.lr
            print("rmsprop lr = {}".format(self.lr))
            print("include pretrain/user/sport/gender/temporal = {}/{}/{}/{}/{}".format(
                   self.pretrain, self.includeUser, self.includeSport, self.includeGender, self.includeTemporal))

            self.model_file_name = []
            self.model_file_name.extend(self.attrFeatures)
            if self.includeTemporal:
                self.model_file_name.append("context")
            print(self.model_file_name)            

            self.trainValidTestSplit = [0.8, 0.1, 0.1]
            self.targetAtts = ['derived_speed']
            self.inputAtts = ['distance', 'altitude', 'time_elapsed'] # only feed in the total time to finish the workout
                       
            self.trimmed_workout_len = 450
            self.num_steps = self.trimmed_workout_len
            self.batch_size_m = args.batch_size
            # Should the data values be scaled to their z-scores with the z-multiple?
            self.scale = True
            self.scaleTargets = False 

            self.endo_reader = dataInterpreter(self.inputAtts, self.targetAtts, self.includeUser, self.includeSport, self.includeGender, self.includeTemporal,  
                                               fn=self.data_path, scaleVals=self.scale, trimmed_workout_len=self.trimmed_workout_len, 
                                               scaleTargets=self.scaleTargets, trainValidTestSplit=self.trainValidTestSplit, 
                                               zMultiple = self.zMultiple, trainValidTestFN=self.trainValidTestFN)

            # preprocess data: scale
            self.endo_reader.preprocess_data()
            self.input_dim = self.endo_reader.input_dim 
            self.output_dim = self.endo_reader.output_dim 
            self.train_size = len(self.endo_reader.trainingSet)
            self.valid_size = len(self.endo_reader.validationSet)
            self.test_size = len(self.endo_reader.testSet)
            # build model
            self.model = self.build_model(args)


    def build_model(self, args):
        print('Build model...')
        self.num_users = len(self.endo_reader.oneHotMap['userId'])
        self.num_sports = len(self.endo_reader.oneHotMap['sport'])

        self.hidden_dim = args.hidden_dim
        user_reg = args.user_reg
        sport_reg = args.sport_reg
        gender_reg = args.gender_reg
        output_reg = args.output_reg
        print("user/sport/output regularizer = {}/{}/{}".format(user_reg, sport_reg, output_reg))

        inputs = Input(shape=(self.num_steps,self.input_dim), name='input')
        self.layer1_dim = self.input_dim

        if self.includeUser:
            user_inputs = Input(shape=(self.num_steps,1), name='user_input')
            User_Embedding = Embedding(input_dim=self.num_users, output_dim=self.user_dim, name='user_embedding', 
                                       embeddings_initializer=initializers.random_normal(stddev=0.01), embeddings_regularizer=l2(user_reg))
            user_embedding = User_Embedding(user_inputs)
            user_embedding = Lambda(lambda y: K.squeeze(y, 2))(user_embedding) 
            self.layer1_dim += self.user_dim

        if self.includeSport:
            sport_inputs = Input(shape=(self.num_steps,1), name='sport_input')
            Sport_Embedding = Embedding(input_dim=self.num_sports, output_dim=self.sport_dim, name='sport_embedding', 
                                   embeddings_initializer=initializers.random_normal(stddev=0.01), embeddings_regularizer=l2(sport_reg))
            sport_embedding = Sport_Embedding(sport_inputs)
            sport_embedding = Lambda(lambda y: K.squeeze(y, 2))(sport_embedding) 
            self.layer1_dim += self.sport_dim

        if self.includeGender:
            gender_inputs = Input(shape=(self.num_steps,1), name='gender_input')
            Gender_Embedding = Embedding(input_dim=self.num_users, output_dim=self.gender_dim, name='gender_embedding', 
                                       embeddings_initializer=initializers.random_normal(stddev=0.01), embeddings_regularizer=l2(gender_reg))
            gender_embedding = Gender_Embedding(gender_inputs)
            gender_embedding = Lambda(lambda y: K.squeeze(y, 2))(gender_embedding) 
            self.layer1_dim += self.gender_dim

        if self.includeTemporal:
            context_input_1 = Input(shape=(self.num_steps,self.input_dim + 1), name='context_input_1') # add 1 for since_last
            context_input_2 = Input(shape=(self.num_steps,self.output_dim), name='context_input_2')

        predict_vector = inputs
        if self.includeUser:
            predict_vector = concatenate([predict_vector, user_embedding])

        if self.includeSport:
            predict_vector = concatenate([predict_vector, sport_embedding])

        if self.includeGender:
            predict_vector = concatenate([predict_vector, gender_embedding]) 
            
        if self.includeTemporal:
            self.context_dim = self.hidden_dim
            context_layer_1 = LSTM(self.hidden_dim, return_sequences=True, input_shape=(self.num_steps, self.input_dim), name='context_layer_1')
            context_layer_2 = LSTM(self.hidden_dim, return_sequences=True, input_shape=(self.num_steps, self.output_dim), name='context_layer_2')
            context_embedding_1 = context_layer_1(context_input_1)
            context_embedding_2 = context_layer_2(context_input_2)
            context_embedding_1 = Dropout(0.1, name='context_dropout_1')(context_embedding_1)
            context_embedding_2 = Dropout(0.1, name='context_dropout_2')(context_embedding_2)
            context_embedding = Dense(self.context_dim, activation='selu', name='context_projection')(concatenate([context_embedding_1, context_embedding_2]))
            predict_vector = concatenate([context_embedding, predict_vector]) 
            self.layer1_dim += self.context_dim
        
        layer1 = LSTM(self.hidden_dim, return_sequences=True, input_shape=(self.num_steps, self.layer1_dim), name='layer1')(predict_vector)
        dropout1 = Dropout(0.2, name='dropout1')(layer1)
        layer2 = LSTM(self.hidden_dim, return_sequences=True, name='layer2')(dropout1)
        dropout2 = Dropout(0.2, name='dropout2')(layer2)
        output = Dense(self.output_dim, name='output', kernel_regularizer=l2(output_reg))(dropout2)
        predict = Activation('selu', name='selu_activation')(output)
        #predict = Activation('linear', name='linear_activation')(output)

        inputs_array = [inputs]
        if self.includeUser:
            inputs_array.append(user_inputs)
        if self.includeSport:
            inputs_array.append(sport_inputs)
        if self.includeGender:
            inputs_array.append(gender_inputs)
        if self.includeTemporal:
            inputs_array.extend([context_input_1, context_input_2])
        model = Model(inputs=inputs_array, outputs=[predict])


        #model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=self.lr), metrics=['mae', 'mse'])
        model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=self.lr), metrics=['mae', root_mean_squared_error])
        #model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.01))

        print("Endomodel Built!")
        model.summary()

        if self.pretrain == True:
            print("pretrain model: {}".format(self.pretrain_model_file_name))

            filepath = "./"+self.pretrain_model_file_name+"_bestValidScore"

            custom_ob = {'root_mean_squared_error':root_mean_squared_error}
            pretrain_model = keras.models.load_model(self.model_save_location+self.pretrain_model_file_name+"/"+self.pretrain_model_file_name+"_bestValidScore", custom_objects=custom_ob) 

            layer_dict = dict([(layer.name, layer) for layer in pretrain_model.layers])
            for layer_name in layer_dict:
                weights = layer_dict[layer_name].get_weights()
    
                if layer_name=='layer1':
                    weights[0] = np.vstack([weights[0],
                                            np.zeros((self.layer1_dim - self.input_dim - self.user_dim - self.sport_dim, self.hidden_dim * 4)).astype(np.float32)])

                model.get_layer(layer_name).set_weights(weights)
            del pretrain_model
        
        return model

    def run_model(self, model):

        modelRunIdentifier = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.model_file_name.append(modelRunIdentifier) # Applend a unique identifier to the filenames
        self.model_file_name = "_".join(self.model_file_name)

        self.model_save_location += self.model_file_name + "/"
        self.summaries_dir += self.model_file_name + "/"
        os.mkdir(self.model_save_location) 
        os.mkdir(self.summaries_dir) 

        best_valid_score = 9999999999
        best_epoch = 0
      
        train_steps_per_epoch = int(self.train_size * self.trimmed_workout_len / (self.num_steps * self.batch_size_m))
        valid_steps_per_epoch = int(self.valid_size * self.trimmed_workout_len / (self.num_steps * self.batch_size_m))
        test_steps_per_epoch = int(self.test_size * self.trimmed_workout_len / (self.num_steps * self.batch_size_m))

        # avoid process data in each iterator?
        for iteration in range(1, self.max_epochs):
            print()
            print('-' * 50)
            print('Iteration', iteration)

            trainDataGen = self.endo_reader.generator_for_autotrain(self.batch_size_m, self.num_steps, "train")  
            
            model_save_fn = self.model_save_location + self.model_file_name + "_epoch_"+str(iteration)
            checkpoint = ModelCheckpoint(model_save_fn, verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')

            history = model.fit_generator(trainDataGen, train_steps_per_epoch, epochs=1, verbose=1, callbacks=[checkpoint])
            try:
                del history.model
                with open(self.summaries_dir+"model_history_"+self.model_file_name+"_epoch_"+str(iteration), "wb") as f:
                    pickle.dump(history, f)
                print("Model history saved")
            except:
                pass

            validDataGen = self.endo_reader.generator_for_autotrain(self.batch_size_m, self.num_steps, "valid")
            valid_score = model.evaluate_generator(validDataGen, valid_steps_per_epoch)
            print("Valid score: ", valid_score)
            try:
                with open(self.summaries_dir+"model_valid_score_"+self.model_file_name+"_epoch_"+str(iteration), "wb") as f:
                    pickle.dump(valid_score, f)
                print("Model validation score saved")
            except:
                pass

            if valid_score[0] <= best_valid_score:
                best_valid_score = valid_score[0]
                best_epoch = iteration
            elif (iteration-best_epoch < self.patience):
                pass
            else:
                print("Stopped early at epoch: " + str(iteration))
                break
        
        # load best model
        custom_ob = {'root_mean_squared_error':root_mean_squared_error}
        best_model = keras.models.load_model(self.model_save_location+self.model_file_name+"_epoch_"+str(best_epoch), custom_objects=custom_ob)
        best_model.save(self.model_save_location+self.model_file_name+"_bestValidScore")

        print("Best epoch: " + str(best_epoch) + " validation score: " + str(best_valid_score))

        print("Testing")
        testDataGen = self.endo_reader.generator_for_autotrain(self.batch_size_m, self.num_steps, "test")
        test_score = best_model.evaluate_generator(testDataGen, test_steps_per_epoch)
        print("Test score: " + str(test_score))
        print("Done!!!")

def main(argv):
    newModel = True
    my_lstm = keras_endoLSTM(argv, newModel)
    my_lstm.run_model(my_lstm.model)

if __name__ == "__main__":
    args = parse()
    main(args)
