import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from data_interpolate import dataInterpreter, metaDataEndomondo

import matplotlib
import datetime as dt, itertools, pandas as pd, matplotlib.pyplot as plt, numpy as np
import datetime, time
import os
import util as util

global logger

util.setup_log()
logger = util.logger

use_cuda = torch.cuda.is_available()
logger.info("Is CUDA available? %s.", use_cuda)


class contextEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(contextEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_dim = self.output_size
        self.context_layer_1 = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_dim, batch_first=True)
        self.context_layer_2 = nn.LSTM(input_size = 1, hidden_size = self.hidden_dim, batch_first=True)
        self.dropout_rate = 0.2
        print("context encoder dropout: {}".format(self.dropout_rate))
        self.dropout = nn.Dropout(self.dropout_rate)
        self.project = nn.Linear(self.hidden_dim * 2, self.context_dim)

    def forward(self, context_input_1, context_input_2):
        context_input_1 = self.dropout(context_input_1)
        context_input_2 = self.dropout(context_input_2)
         
        hidden_1 = self.init_hidden(context_input_1) # 1 * batch_size * hidden_size
        cell_1 = self.init_hidden(context_input_1)
        hidden_2 = self.init_hidden(context_input_2) # 1 * batch_size * hidden_size
        cell_2 = self.init_hidden(context_input_2)

        #print("context_input_1: ", context_input_1.shape)
        #print("context_input_2: ", context_input_2.shape)

        self.context_layer_1.flatten_parameters()
        outputs_1, lstm_states_1 = self.context_layer_1(context_input_1, (hidden_1, cell_1))
        #context_embedding_1 = lstm_states_1[0]
        context_embedding_1 = outputs_1
        self.context_layer_2.flatten_parameters()
        outputs_2, lstm_states_2 = self.context_layer_2(context_input_2, (hidden_2, cell_2))
        #context_embedding_2 = lstm_states_1[0]
        context_embedding_2 = outputs_2

        #context_embedding_1 = self.dropout(context_embedding_1)
        #context_embedding_2 = self.dropout(context_embedding_2)
        context_embedding = self.project(torch.cat([context_embedding_1, context_embedding_2], dim=-1))

        '''print(context_embedding_1.shape)
        print(context_embedding_2.shape)
        print(context_embedding.shape)'''

        return context_embedding

    def init_hidden(self, x):
        # No matter whether CUDA is used, the returned variable will have the same type as x.
        return Variable(x.data.new(1, x.size(0), self.hidden_size).zero_()) # dimension 0 is the batch dimension

class encoder(nn.Module):
    def __init__(self, input_size, hidden_size, T, attr_embeddings, logger, dropout=0.1):
        # input size: number of underlying factors (81)
        # T: number of time steps (10)
        # hidden_size: dimension of the hidden state
        super(encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.user_embedding = attr_embeddings[0]
        self.sport_embedding = attr_embeddings[1]
        self.logger = logger
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        print("encoder dropout: {}".format(self.dropout_rate))

        #self.lstm_layer = nn.LSTM(input_size = input_size, hidden_size = hidden_size, batch_first=True)
        self.lstm_layer = nn.LSTM(input_size = input_size, hidden_size = hidden_size)
        self.attn_linear = nn.Linear(in_features = 2 * hidden_size + T, out_features = 1)

    def forward(self, attr_inputs, context_embedding, input_variable):
        for attr in attr_inputs:
            attr_input = attr_inputs[attr]
            if attr == "user_input":
                attr_embed = self.user_embedding(attr_input)
            if attr == "sport_input":
                attr_embed = self.sport_embedding(attr_input)
            input_variable = torch.cat([attr_embed, input_variable], dim=-1)

        input_variable = torch.cat([context_embedding, input_variable], dim=-1)

        input_data = input_variable
        # input_data: batch_size * T * input_size
        input_weighted = Variable(input_data.data.new(input_data.size(0), self.T, self.input_size).zero_())
        input_encoded = Variable(input_data.data.new(input_data.size(0), self.T, self.hidden_size).zero_())
        # hidden, cell: initial states with dimention hidden_size
        hidden = self.init_hidden(input_data) # 1 * batch_size * hidden_size
        cell = self.init_hidden(input_data)
        # hidden.requires_grad = False
        # cell.requires_grad = False
  
        for t in range(self.T):
            #print("time step {}".format(t))
            # Eqn. 8: concatenate the hidden states with each predictor
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim = 2) # batch_size * input_size * (2*hidden_size + T)
            # Eqn. 9: Get attention weights
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T)) # (batch_size * input_size) * 1
            attn_weights = F.softmax(x.view(-1, self.input_size), dim = -1) # batch_size * input_size, attn weights with values sum up to 1.
            # Eqn. 10: LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :]) # batch_size * input_size

            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282

            #print("weighted_input: ", weighted_input.shape)
            #print("hidden: ", hidden.shape)
            #print("cell: ", cell.shape)

            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden
        return input_weighted, input_encoded

    def init_hidden(self, x):
        # No matter whether CUDA is used, the returned variable will have the same type as x.
        return Variable(x.data.new(1, x.size(0), self.hidden_size).zero_()) # dimension 0 is the batch dimension

class decoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, T, logger):
        super(decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.logger = logger

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
                                         nn.Tanh(), nn.Linear(encoder_hidden_size, 1))
        #self.lstm_layer = nn.LSTM(input_size = 1, hidden_size = decoder_hidden_size, batch_first=True)
        self.lstm_layer = nn.LSTM(input_size = 1, hidden_size = decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + 1, 1)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, 1)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        # input_encoded: batch_size * T * encoder_hidden_size
        # y_history: batch_size * (T-1)
        # Initialize hidden and cell, 1 * batch_size * decoder_hidden_size
        hidden = self.init_hidden(input_encoded)
        cell = self.init_hidden(input_encoded)
        #print("input_encoded: ", input_encoded.shape)

        # hidden.requires_grad = False
        # cell.requires_grad = False
        for t in range(self.T):
            # Eqn. 12-13: compute attention weights
            ## batch_size * T * (2*decoder_hidden_size + encoder_hidden_size)
            x = torch.cat((hidden.repeat(self.T, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T, 1, 1).permute(1, 0, 2), input_encoded), dim = 2)
            x = F.softmax(self.attn_layer(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size
                                                )).view(-1, self.T), dim = -1) # batch_size * T, row sum up to 1
            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :] # batch_size * encoder_hidden_size

            #print("time step {}".format(t))

            if t < self.T - 1:
                # Eqn. 15
                y_tilde = self.fc(torch.cat((context, y_history[:, t].unsqueeze(1)), dim = 1)) # batch_size * 1
                #print("y_tilde: ", y_tilde.shape)
                #print("hidden: ", hidden.shape)
                #print("cell: ", cell.shape)

                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
                hidden = lstm_output[0] # 1 * batch_size * decoder_hidden_size
                cell = lstm_output[1] # 1 * batch_size * decoder_hidden_size

        # Eqn. 22: final output
        y_pred = self.fc_final(torch.cat((hidden[0], context), dim = 1))
        # self.logger.info("hidden %s context %s y_pred: %s", hidden[0][0][:10], context[0][:10], y_pred[:10])
        return y_pred.view(y_pred.size(0))

    def init_hidden(self, x):
        return Variable(x.data.new(1, x.size(0), self.decoder_hidden_size).zero_())


# In[ ]:

# Train the model
class da_rnn:
    def __init__(self, logger, encoder_hidden_size = 64, decoder_hidden_size = 64, T = 10,
                 learning_rate = 0.01, batch_size = 5120, parallel = True, debug = False, test_model_path = None):
        self.T = T
        self.logger = logger
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        path = "data"
        self.model_save_location = path + "/fitrec-attn/model_states/"
        self.summaries_dir = path + "/fitrec-attn/logs/"
        self.data_path = "endomondoHR_proper.json"
        self.patience = 3 # [3,5,10]
        self.max_epochs = 50
        self.zMultiple = 5

        self.pretrain, self.includeUser, self.includeSport, self.includeTemporal = False, True, False, True

        print("include pretrain/user/sport/temporal = {}/{}/{}/{}".format(self.pretrain,self.includeUser,self.includeSport,self.includeTemporal))

        self.model_file_name = []
        if self.includeUser:
            self.model_file_name.append("userId")
        if self.includeSport:
            self.model_file_name.append("sport")
        if self.includeTemporal:
            self.model_file_name.append("context")
        print(self.model_file_name)

        self.user_dim = 5
        self.sport_dim = 5

        self.trainValidTestSplit = [0.8, 0.1, 0.1]
        self.targetAtts = ['heart_rate']
        self.inputAtts = ['derived_speed', 'altitude']

        self.trimmed_workout_len = 300
        self.num_steps = self.trimmed_workout_len

        # Should the data values be scaled to their z-scores with the z-multiple?
        self.scale_toggle = True
        self.scaleTargets = False 

        self.trainValidTestFN = self.data_path.split(".")[0] + "_temporal_dataset.pkl"

        self.endo_reader = dataInterpreter(self.T, self.inputAtts, self.includeUser, self.includeSport, 
                                           self.includeTemporal, self.targetAtts, fn=self.data_path,
                                           scaleVals=self.scale_toggle, trimmed_workout_len=self.trimmed_workout_len, 
                                           scaleTargets=self.scaleTargets, trainValidTestSplit=self.trainValidTestSplit, 
                                           zMultiple = self.zMultiple, trainValidTestFN=self.trainValidTestFN)

        self.endo_reader.preprocess_data()

        self.input_dim = self.endo_reader.input_dim 
        self.output_dim = self.endo_reader.output_dim 

        self.train_size = len(self.endo_reader.trainingSet)
        self.valid_size = len(self.endo_reader.validationSet)
        self.test_size = len(self.endo_reader.testSet)

        modelRunIdentifier = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.model_file_name.append(modelRunIdentifier) # Applend a unique identifier to the filenames
        self.model_file_name = "_".join(self.model_file_name)

        self.model_save_location += self.model_file_name + "/"
        print(self.model_save_location)
        
        # build model
        # model
        self.num_users = len(self.endo_reader.oneHotMap['userId'])
        self.num_sports = len(self.endo_reader.oneHotMap['sport'])
        self.num_genders = len(self.endo_reader.oneHotMap['gender'])

        self.input_size = self.input_dim
        self.attr_num = 0
        self.attr_embeddings = []
        user_embedding = nn.Embedding(self.num_users, self.user_dim)
        torch.nn.init.xavier_uniform(user_embedding.weight.data)
        self.attr_embeddings.append(user_embedding)
        sport_embedding = nn.Embedding(self.num_sports, self.sport_dim)
        self.attr_embeddings.append(sport_embedding) 

        if self.includeUser:
            self.attr_num += 1
            self.input_size += self.user_dim
        if self.includeSport:
            self.attr_num += 1
            self.input_size += self.sport_dim
       
        if self.includeTemporal:
            # self.context_dim = self.user_dim
            # self.context_dim = encoder_hidden_size
            self.context_dim = int(encoder_hidden_size / 2)
            self.input_size += self.context_dim
            self.context_encoder = contextEncoder(input_size = self.input_dim + 1, hidden_size = encoder_hidden_size, output_size = self.context_dim).cuda()

        if use_cuda:
            for attr_embedding in self.attr_embeddings:
                attr_embedding = attr_embedding.cuda()       
 
        self.encoder = encoder(input_size = self.input_size, hidden_size = encoder_hidden_size, T = T, 
                               attr_embeddings = self.attr_embeddings, 
                              logger = logger).cuda()
        self.decoder = decoder(encoder_hidden_size = encoder_hidden_size,
                               decoder_hidden_size = decoder_hidden_size,
                               T = T, logger = logger).cuda()

        if parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.context_encoder = nn.DataParallel(self.context_encoder)
            self.decoder = nn.DataParallel(self.decoder)
 
        wd1 = 0.002
        #wd1 = 0.003
        wd2 = 0.005
        if self.includeUser:
            print("user weight decay: {}".format(wd1))
        if self.includeSport:
            print("sport weight decay: {}".format(wd2))
        #self.encoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.encoder.parameters()),
        #                                   lr = learning_rate, weight_decay=wd)

        self.encoder_optimizer = optim.Adam([
                {'params': [param for name, param in self.encoder.named_parameters() if 'user_embedding' in name], 'weight_decay':wd1},
                {'params': [param for name, param in self.encoder.named_parameters() if 'sport_embedding' in name], 'weight_decay':wd2},
                {'params': [param for name, param in self.encoder.named_parameters() if 'embedding' not in name]}
            ], lr=learning_rate)

        self.context_encoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.context_encoder.parameters()),
                                           lr = learning_rate)
        self.decoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.decoder.parameters()),
                                           lr = learning_rate)
        self.loss_func = nn.MSELoss(size_average=True)
        
        if test_model_path:
            checkpoint = torch.load(test_model_path)
            self.encoder.load_state_dict(checkpoint['en'])
            self.context_encoder.load_state_dict(checkpoint['context_en'])
            self.decoder.load_state_dict(checkpoint['de'])
            print("test model: {}".format(test_model_path))


    def get_batch(self, batch):
        
        attr_inputs = {}
        if self.includeUser:
            user_input = batch[0]['user_input']
            attr_inputs['user_input'] = user_input
        if self.includeSport:
            sport_input = batch[0]['sport_input']
            attr_inputs['sport_input'] = sport_input
        
        for attr in attr_inputs:
            attr_input = attr_inputs[attr]     
            attr_input = Variable(torch.from_numpy(attr_input).long())
            if use_cuda:
                attr_input = attr_input.cuda()
            attr_inputs[attr] = attr_input
        
        context_input_1 = batch[0]['context_input_1']
        context_input_2 = batch[0]['context_input_2']
        context_input_1 = Variable(torch.from_numpy(context_input_1).float())
        context_input_2 = Variable(torch.from_numpy(context_input_2).float())

        input_variable = batch[0]['input'] 
        target_variable = batch[1]
        input_variable = Variable(torch.from_numpy(input_variable).float())
        target_variable = Variable(torch.from_numpy(target_variable).float())
        if use_cuda:
            input_variable = input_variable.cuda()
            target_variable = target_variable.cuda()
            context_input_1 = context_input_1.cuda()
            context_input_2 = context_input_2.cuda()
        
        y_history = target_variable[:, :self.T - 1, :].squeeze(-1)
        y_target = target_variable[:, -1, :].squeeze(-1)      
        
        return attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target 

    def train(self, n_epochs = 30, print_every=400):
        
        # initialize
        print('Initializing ...')
        start_epoch = 0
        best_val_loss = None
        best_epoch_path = None
        best_valid_score = 9999999999
        best_epoch = 0

        for iteration in range(n_epochs):

            print()
            print('-' * 50)
            print('Iteration', iteration)

            epoch_start_time = time.time()
            start_time = time.time()
            
            # train
            trainDataGen = self.endo_reader.generator_for_autotrain(self.batch_size, self.num_steps, "train")
            print_loss = 0
            for batch, training_batch in enumerate(trainDataGen):
              
                 
                attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target = self.get_batch(training_batch) 
                loss = self.train_iteration(attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target)
                
                print_loss += loss
                if batch % print_every == 0 and batch > 0:
                    cur_loss = print_loss / print_every
                    elapsed = time.time() - start_time

                    print('| epoch {:3d} | {:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                          'loss {:5.3f}'.format(
                          iteration, batch, self.learning_rate,
                          elapsed * 1000 / print_every, cur_loss))

                    print_loss = 0
                    start_time = time.time()
            
            # evaluate    
            validDataGen = self.endo_reader.generator_for_autotrain(self.batch_size, self.num_steps, "valid")
            val_loss = 0
            val_batch_num = 0
            for val_batch in validDataGen:
                val_batch_num += 1

                attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target = self.get_batch(val_batch) 
                loss = self.evaluate(attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target)
                
                val_loss += loss
            val_loss /= val_batch_num
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.3f}'.format(iteration, (time.time() - epoch_start_time),
                                               val_loss))
            print('-' * 89)
            if not best_val_loss or val_loss <= best_val_loss:
                
                best_val_loss = val_loss
                best_epoch = iteration
                best_epoch_path = self.model_save_location + self.model_file_name + "_epoch_"+str(iteration)

                if not os.path.exists(self.model_save_location):
                    os.makedirs(self.model_save_location)
                torch.save({
                    'epoch': iteration,
                    'en': self.encoder.state_dict(),
                    'context_en': self.context_encoder.state_dict(),
                    'de': self.decoder.state_dict(),
                    'en_opt': self.encoder_optimizer.state_dict(),
                    'context_en_opt': self.context_encoder_optimizer.state_dict(),
                    'de_opt': self.decoder_optimizer.state_dict(),
                    'loss': loss
                }, best_epoch_path)

            elif (iteration - best_epoch < self.patience):
                pass
            else:
                print("Stopped early at epoch: " + str(iteration))
                break        
                
        # load best to test
        if best_epoch_path:
            checkpoint = torch.load(best_epoch_path)
            self.encoder.load_state_dict(checkpoint['en'])  
            self.context_encoder.load_state_dict(checkpoint['context_en'])  
            self.decoder.load_state_dict(checkpoint['de'])                
        print("best model: {}".format(best_epoch_path))
                
        # test
        testDataGen = self.endo_reader.generator_for_autotrain(self.batch_size, self.num_steps, "test")
        test_loss = 0
        test_batch_num = 0
        
        for test_batch in testDataGen:
            test_batch_num += 1
             
            attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target = self.get_batch(test_batch) 
            loss = self.evaluate(attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target)
                
            test_loss += loss
        test_loss /= test_batch_num
        print('-' * 89)
        print('| test loss {:5.3f}'.format(test_loss))
        print('-' * 89)                
                
                
    def train_iteration(self, attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target):
        self.encoder.train()
        self.context_encoder.train()
        self.decoder.train()
        
        self.encoder_optimizer.zero_grad()
        self.context_encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        context_embedding = self.context_encoder(context_input_1, context_input_2)
        input_weighted, input_encoded = self.encoder(attr_inputs, context_embedding, input_variable)
        y_pred = self.decoder(input_encoded, y_history)
        
        loss = self.loss_func(y_pred, y_target)
        loss.backward()

        self.encoder_optimizer.step()
        self.context_encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data.item()

    def evaluate(self, attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target):
        self.encoder.eval()
        self.context_encoder.eval()
        self.decoder.eval()

        context_embedding = self.context_encoder(context_input_1, context_input_2)
        input_weighted, input_encoded = self.encoder(attr_inputs, context_embedding, input_variable)
        y_pred = self.decoder(input_encoded, y_history)
        
        loss = self.loss_func(y_pred, y_target)

        return loss.data.item()
    
    
    def predict(self, on_train = False):
        if on_train:
            y_pred = np.zeros(self.train_size - self.T + 1)
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_size)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i : (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T - 1))
            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(batch_idx[j], batch_idx[j] + self.T - 1), :]
                    y_history[j, :] = self.y[range(batch_idx[j],  batch_idx[j]+ self.T - 1)]
                else:
                    X[j, :, :] = self.X[range(batch_idx[j] + self.train_size - self.T, batch_idx[j] + self.train_size - 1), :]
                    y_history[j, :] = self.y[range(batch_idx[j] + self.train_size - self.T,  batch_idx[j]+ self.train_size - 1)]

            y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cuda())
            _, input_encoded = self.encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda()))
            y_pred[i:(i + self.batch_size)] = self.decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]
            i += self.batch_size
        return y_pred


learning_rate = 0.005
#batch_size = 5120
#batch_size = 10240
batch_size = 12800
hidden_size = 64
#T=20
T=10
print("learning rate = {}, batch_size = {}, hidden_size = {}, T = {}".format(learning_rate, batch_size, hidden_size, T))
model = da_rnn(logger = logger, parallel = False, T = T, encoder_hidden_size=hidden_size, decoder_hidden_size=hidden_size, learning_rate = learning_rate, batch_size=batch_size)

model.train(n_epochs = 50)

'''
plt.figure()
plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
plt.show()

plt.figure()
plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
plt.show()

plt.figure()
plt.plot(y_pred, label = 'Predicted')
plt.plot(model.y[model.train_size:], label = "True")
plt.legend(loc = 'upper left')
plt.show()
'''
