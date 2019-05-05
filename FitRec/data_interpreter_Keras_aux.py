import numpy as np
import pickle
import os
from haversine import haversine
from math import floor
from collections import defaultdict
import random
import gzip
from tqdm import tqdm 
import pandas as pd
import time
import multiprocessing
from multiprocessing import Pool

# dataset already been preprocessed
def parse(path):
    if 'gz' in path:
        f = gzip.open(path, 'rb')
        for l in f.readlines():
            yield(eval(l.decode('ascii')))
    else:
        f = open(path, 'rb')
        for l in f.readlines():
            yield(eval(l))

def process(line):
    return eval(line)

class dataInterpreter(object):
    def __init__(self, inputAtts, targetAtts=['derived_speed'], includeUser=True, includeSport=False, includeGender=False, includeTemporal=False, fn="endomondoHR_proper.json", scaleVals=True, trimmed_workout_len=450, scaleTargets="scaleVals", trainValidTestSplit=[.8,.1,.1], zMultiple=5, trainValidTestFN=None):
        self.filename = fn
        self.data_path = "./data"
        self.metaDataFn = fn.split(".")[0] + "_metaData.pkl"

        self.scaleVals = scaleVals
        self.trimmed_workout_len = trimmed_workout_len
        if scaleTargets == "scaleVals":
            scaleTargets = scaleVals
        self.scale_targets = scaleTargets # set to false when scale only inputs
        self.smooth_window = 1 # window size = 1 means no smoothing
        self.perform_target_smoothing = True

        self.isNominal = ['gender', 'sport']
        self.isDerived = ['time_elapsed', 'distance', 'derived_speed', 'since_begin', 'since_last']
        self.isSequence = ['altitude', 'heart_rate', 'latitude', 'longitude'] + self.isDerived

        self.inputAtts = inputAtts
        self.includeUser = includeUser
        self.includeSport = includeSport
        self.includeGender = includeGender
        self.includeTemporal = includeTemporal

        self.targetAtts = ["tar_" + tAtt for tAtt in targetAtts]

        print("input attributes: ", self.inputAtts)
        print("target attributes: ", self.targetAtts)

        self.trainValidTestSplit = trainValidTestSplit
        self.trainValidTestFN = trainValidTestFN
        self.zMultiple = zMultiple

    def preprocess_data(self):
 
        self.original_data_path = self.data_path + "/" + self.filename 
        self.processed_path = self.data_path + "/processed_" + self.filename.split(".")[0] + ".npy"

        # load index for train/valid/test
        self.loadTrainValidTest()

        if os.path.exists(self.processed_path):
            # preprocessed data already exist
            print("{} exists".format(self.processed_path))
            self.original_data = np.load(self.processed_path)[0]
            self.map_workout_id()
        else:
            # not preprocessed yet, load raw data and preprocess
            print("load original data")
            pool = Pool(5) 
            with open(self.original_data_path, 'r') as f:
                self.original_data =pool.map(process, f)
            pool.close()
            pool.join()
            self.map_workout_id()
            # derive data
            self.derive_data()
            # build meta
            self.buildMetaData()
            # scale data
            self.scale_data()
        
        self.load_meta()
        self.input_dim = len(self.inputAtts)
        self.output_dim = len(self.targetAtts) # each continuous target has dimension 1, so total length = total dimension
      
    def map_workout_id(self):
        # convert workout id to original data id
        self.idxMap = defaultdict(int)
        for idx, d in enumerate(self.original_data):  
            self.idxMap[d['id']] = idx

        self.trainingSet = [self.idxMap[wid] for wid in self.trainingSet]
        self.validationSet = [self.idxMap[wid] for wid in self.validationSet]
        self.testSet = [self.idxMap[wid] for wid in self.testSet]
        
        # update workout id to index in original_data
        contextMap2 = {} 
        for wid in self.contextMap:
            context = self.contextMap[wid]
            contextMap2[self.idxMap[wid]] = (context[0], context[1], [self.idxMap[wid] for wid in context[2]])
        self.contextMap = contextMap2 
    
    
    def load_meta(self): 
        self.buildMetaData() 

    def randomizeDataOrder(self, dataIndices):
        return np.random.permutation(dataIndices)

    
    def generateByIdx(self, index):
        targetAtts = self.targetAtts
        inputAtts = self.inputAtts

        inputDataDim = self.input_dim
        targetDataDim = self.output_dim
        
        current_input = self.original_data[index] 
        workoutid = current_input['id']

        inputs = np.zeros([inputDataDim, self.trimmed_workout_len])
        outputs = np.zeros([targetDataDim, self.trimmed_workout_len])
        for idx, att in enumerate(inputAtts):
            if att == 'time_elapsed':
                inputs[idx, :] = np.ones([1, self.trimmed_workout_len]) * current_input[att][self.trimmed_workout_len-1] # given the total workout length
            else:
                inputs[idx, :] = current_input[att][:self.trimmed_workout_len]
        for att in targetAtts:
            outputs[0, :] = current_input[att][:self.trimmed_workout_len]
        inputs = np.transpose(inputs)
        outputs = np.transpose(outputs)

        if self.includeUser:
            user_inputs = np.ones([self.trimmed_workout_len, 1]) * self.oneHotMap['userId'][current_input['userId']]
        if self.includeSport:
            sport_inputs = np.ones([self.trimmed_workout_len, 1]) * self.oneHotMap['sport'][current_input['sport']]
        if self.includeGender:
            gender_inputs = np.ones([self.trimmed_workout_len, 1]) * self.oneHotMap['gender'][current_input['gender']]

        # build context input    
        if self.includeTemporal:
            context_idx = self.contextMap[idx][2][-1] # index of previous workouts
            context_input = self.original_data[context_idx]

            context_since_last = np.ones([1, self.trimmed_workout_len]) * self.contextMap[idx][0]
            # consider what context?
            context_inputs = np.zeros([inputDataDim, self.trimmed_workout_len])
            context_outputs = np.zeros([targetDataDim, self.trimmed_workout_len])
            for idx, att in enumerate(inputAtts):
                if att == 'time_elapsed':
                    context_inputs[idx, :] = np.ones([1, self.trimmed_workout_len]) * context_input[att][self.trimmed_workout_len-1]
                else:
                    context_inputs[idx, :] = context_input[att][:self.trimmed_workout_len]
            for att in targetAtts:
                context_outputs[0, :] = context_input[att][:self.trimmed_workout_len]
            context_input_1 = np.transpose(np.concatenate([context_inputs, context_since_last], axis=0))
            context_input_2 = np.transpose(context_outputs)

        inputs_dict = {'input':inputs}
        if self.includeUser:       
            inputs_dict['user_input'] = user_inputs
        if self.includeSport:       
            inputs_dict['sport_input'] = sport_inputs
        if self.includeGender:
            inputs_dict['gender_input'] = gender_inputs
        if self.includeTemporal:
            inputs_dict['context_input_1'] = context_input_1
            inputs_dict['context_input_2'] = context_input_2

        return (inputs_dict, outputs, workoutid)
    
    # yield input and target data
    def dataIteratorSupervised(self, trainValidTest):
        targetAtts = self.targetAtts
        inputAtts = self.inputAtts

        inputDataDim = self.input_dim
        targetDataDim = self.output_dim

        # run on train, valid or test?
        if trainValidTest == 'train':
            indices = self.trainingSet
        elif trainValidTest == 'valid':
            indices = self.validationSet
        elif trainValidTest == 'test':
            indices = self.testSet
        else:
            raise (Exception("invalid dataset type: must be 'train', 'valid', or 'test'"))

        # loop each data point
        for idx in indices:
            current_input = self.original_data[idx] 
            workoutid = current_input['id']
 
            inputs = np.zeros([inputDataDim, self.trimmed_workout_len])
            outputs = np.zeros([targetDataDim, self.trimmed_workout_len])
            for i, att in enumerate(inputAtts):
                if att == 'time_elapsed':
                    inputs[i, :] = np.ones([1, self.trimmed_workout_len]) * current_input[att][self.trimmed_workout_len-1] # given the total workout length
                else:
                    inputs[i, :] = current_input[att][:self.trimmed_workout_len]
            for att in targetAtts:
                outputs[0, :] = current_input[att][:self.trimmed_workout_len]
            inputs = np.transpose(inputs)
            outputs = np.transpose(outputs)

            if self.includeUser:
                user_inputs = np.ones([self.trimmed_workout_len, 1]) * self.oneHotMap['userId'][current_input['userId']]
            if self.includeSport:
                sport_inputs = np.ones([self.trimmed_workout_len, 1]) * self.oneHotMap['sport'][current_input['sport']]
            if self.includeGender:
                gender_inputs = np.ones([self.trimmed_workout_len, 1]) * self.oneHotMap['gender'][current_input['gender']]
   
            # build context input    
            if self.includeTemporal:
                context_idx = self.contextMap[idx][2][-1] # index of previous workouts
                context_input = self.original_data[context_idx]

                context_since_last = np.ones([1, self.trimmed_workout_len]) * self.contextMap[idx][0]
                # consider what context?
                context_inputs = np.zeros([inputDataDim, self.trimmed_workout_len])
                context_outputs = np.zeros([targetDataDim, self.trimmed_workout_len])
                for i, att in enumerate(inputAtts):
                    if att == 'time_elapsed':
                        context_inputs[i, :] = np.ones([1, self.trimmed_workout_len]) * context_input[att][self.trimmed_workout_len-1]
                    else:
                        context_inputs[i, :] = context_input[att][:self.trimmed_workout_len]
                for att in targetAtts:
                    context_outputs[0, :] = context_input[att][:self.trimmed_workout_len]
                context_input_1 = np.transpose(np.concatenate([context_inputs, context_since_last], axis=0))
                context_input_2 = np.transpose(context_outputs)
            
            inputs_dict = {'input':inputs}
            if self.includeUser:       
                inputs_dict['user_input'] = user_inputs
            if self.includeSport:       
                inputs_dict['sport_input'] = sport_inputs
            if self.includeGender:
                inputs_dict['gender_input'] = gender_inputs
            if self.includeTemporal:
                inputs_dict['context_input_1'] = context_input_1
                inputs_dict['context_input_2'] = context_input_2
                
            yield (inputs_dict, outputs, workoutid)


    # feed into Keras' fit_generator (automatically resets)
    def generator_for_autotrain(self, batch_size, num_steps, trainValidTest):
        print("batch size = {}, num steps = {}".format(batch_size, num_steps))
        print("start new generator epoch: " + trainValidTest)

        # get the batch generator based on mode: train/valid/test
        if trainValidTest=="train":
            data_len = len(self.trainingSet)
        elif trainValidTest=="valid":
            data_len = len(self.validationSet)
        elif trainValidTest=="test":
            data_len = len(self.testSet)
        else:
            raise(ValueError("trainValidTest is not a valid value"))
        batchGen = self.dataIteratorSupervised(trainValidTest)
        epoch_size = int(data_len / batch_size)
        inputDataDim = self.input_dim
        targetDataDim = self.output_dim

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
            
        for i in range(epoch_size):
            inputs = np.zeros([batch_size, num_steps, inputDataDim])
            outputs = np.zeros([batch_size, num_steps, targetDataDim])
            workoutids = np.zeros([batch_size])

            if self.includeUser:
                user_inputs = np.zeros([batch_size, num_steps, 1])
            if self.includeSport:
                sport_inputs = np.zeros([batch_size, num_steps, 1])
            if self.includeGender:
                gender_inputs = np.zeros([batch_size, num_steps, 1])
            if self.includeTemporal:
                context_input_1 = np.zeros([batch_size, num_steps, inputDataDim + 1])
                context_input_2 = np.zeros([batch_size, num_steps, targetDataDim])

            # inputs_dict = {'input':inputs}
            inputs_dict = {'input':inputs, 'workoutid':workoutids}
            for j in range(batch_size):
                current = next(batchGen)
                inputs[j,:,:] = current[0]['input']
                outputs[j,:,:] = current[1]
                workoutids[j] = current[2]

                if self.includeUser:
                    user_inputs[j,:,:] = current[0]['user_input']
                    inputs_dict['user_input'] = user_inputs
                if self.includeSport:
                    sport_inputs[j,:,:] = current[0]['sport_input']
                    inputs_dict['sport_input'] = sport_inputs
                if self.includeGender:
                    gender_inputs[j,:,:] = current[0]['gender_input']
                    inputs_dict['gender_input'] = gender_inputs
                if self.includeTemporal:
                    context_input_1[j,:,:] = current[0]['context_input_1']
                    context_input_2[j,:,:] = current[0]['context_input_2']
                    inputs_dict['context_input_1'] = context_input_1
                    inputs_dict['context_input_2'] = context_input_2
            # yield one batch
            yield (inputs_dict, outputs)

    def loadTrainValidTest(self):
        with open(self.trainValidTestFN, "rb") as f:
            self.trainingSet, self.validationSet, self.testSet, self.contextMap = pickle.load(f)
            print("train/valid/test set size = {}/{}/{}".format(len(self.trainingSet), len(self.validationSet), len(self.testSet)))
            print("dataset split loaded")       

    # derive 'time_elapsed', 'distance', 'new_workout', 'derived_speed'
    def deriveData(self, att, currentDataPoint, idx):
        if att == 'time_elapsed':
            # Derive the time elapsed from the start
            timestamps = currentDataPoint['timestamp']
            initialTime = timestamps[0]
            return [x - initialTime for x in timestamps]
        elif att == 'distance':
            # Derive the distance
            lats = currentDataPoint['latitude']
            longs = currentDataPoint['longitude']
            indices = range(1, len(lats)) 
            distances = [0]
            # Gets distance traveled since last time point in kilometers
            distances.extend([haversine([lats[i-1],longs[i-1]], [lats[i],longs[i]]) for i in indices]) 
            return distances
        # derive the new_workout list
        elif att == 'new_workout': 
            workoutLength = self.trimmed_workout_len
            newWorkout = np.zeros(workoutLength)
            # Add the signal at start
            newWorkout[0] = 1 
            return newWorkout
        elif att == 'derived_speed':
            distances = self.deriveData('distance', currentDataPoint, idx)
            timestamps = currentDataPoint['timestamp']
            indices = range(1, len(timestamps))
            times = [0]
            times.extend([timestamps[i] - timestamps[i-1] for i in indices])
            derivedSpeeds = [0]
            for i in indices:
                try:
                    curr_speed = 3600 * distances[i] / times[i]
                    derivedSpeeds.append(curr_speed)
                except:
                    derivedSpeeds.append(derivedSpeeds[i-1])
            return derivedSpeeds
        elif att == 'since_last':
            if idx in self.contextMap:
                total_time = self.contextMap[idx][0]
            else:
                total_time = 0
            return np.ones(self.trimmed_workout_len) * total_time
        elif att == 'since_begin':
            if idx in self.contextMap:
                total_time = self.contextMap[idx][1]
            else:
                total_time = 0
            return np.ones(self.trimmed_workout_len) * total_time
        else:
            raise(Exception("No such derived data attribute"))

        
    # computing z-scores and multiplying them based on a scaling paramater
    # produces zero-centered data, which is important for the drop-in procedure
    def scaleData(self, data, att, zMultiple=2):
        mean, std = self.variableMeans[att], self.variableStds[att]
        diff = [d - mean for d in data]
        zScore = [d / std for d in diff] 
        return [x * zMultiple for x in zScore]

    # perform fixed-window median smoothing on a sequence
    def median_smoothing(self, seq, context_size):
        # seq is a list
        if context_size == 1: # if the window is 1, no smoothing should be applied
            return seq
        seq_len = len(seq)
        if context_size % 2 == f0:
            raise(exception("Context size must be odd for median smoothing"))

        smoothed_seq = []
        # loop through sequence and smooth each position
        for i in range(seq_len): 
            cont_diff = (context_size - 1) / 2
            context_min = int(max(0, i-cont_diff))
            context_max = int(min(seq_len, i+cont_diff))
            median_val = np.median(seq[context_min:context_max])
            smoothed_seq.append(median_val)

        return smoothed_seq
    
    def buildEncoder(self, classLabels):
        # Constructs a dictionary that maps each class label to a list 
        # where one entry in the list is 1 and the remainder are 0
        encodingLength = len(classLabels)
        encoder = {}
        mapper = {}
        for i, label in enumerate(classLabels):
            encoding = [0] * encodingLength
            encoding[i] = 1
            encoder[label] = encoding
            mapper[label] = i
        return encoder, mapper
    
    
    def writeSummaryFile(self):
        metaDataForWriting=metaDataEndomondo(self.numDataPoints, self.encodingLengths, self.oneHotEncoders,  
                                             self.oneHotMap, self.isSequence, self.isNominal, self.isDerived, 
                                             self.variableMeans, self.variableStds)
        with open(self.metaDataFn, "wb") as f:
            pickle.dump(metaDataForWriting, f)
        print("Summary file written")
        
    def loadSummaryFile(self):
        try:
            print("Loading metadata")
            with open(self.metaDataFn, "rb") as f:
                metaData = pickle.load(f)
        except:
            raise(IOError("Metadata file: " + self.metaDataFn + " not in valid pickle format"))
        self.numDataPoints = metaData.numDataPoints
        self.encodingLengths = metaData.encodingLengths
        self.oneHotEncoders = metaData.oneHotEncoders
        self.oneHotMap = metaData.oneHotMap
        self.isSequence = metaData.isSequence 
        self.isNominal = metaData.isNominal
        self.variableMeans = metaData.variableMeans
        self.variableStds = metaData.variableStds
        print("Metadata loaded")

        
    def derive_data(self):
        print("derive data")
        # derive based on original data
        for idx, d in enumerate(self.original_data):
            for att in self.isDerived:
                self.original_data[idx][att] = self.deriveData(att, d, idx) # add derived attribute
            
        
    # Generate meta information about data
    def buildMetaData(self):
        if os.path.isfile(self.metaDataFn):
            self.loadSummaryFile()
        else:
            print("Building data schema")
            # other than categoriacl, all are continuous
            # categorical to one-hot: gender, sport
            # categorical to embedding: userId  
            
            # continuous attributes
            print("is sequence: {}".format(self.isSequence))  
            # sum of variables? 
            variableSums = defaultdict(float)
            
            # number of categories for each categorical variable
            classLabels = defaultdict(set)
        
            # consider all data to first get the max, min, etc...      
            for currData in self.original_data:
                # update number of users
                att = 'userId'
                user = currData[att]
                classLabels[att].add(user)
                
                # update categorical attribute
                for att in self.isNominal:
                    val  = currData[att]
                    classLabels[att].add(val)
                    
                # update continuous attribute
                for att in self.isSequence: 
                    variableSums[att] += sum(currData[att])

            oneHotEncoders = {}
            oneHotMap = {}
            encodingLengths = {}
            for att in self.isNominal:
                oneHotEncoders[att], oneHotMap[att] = self.buildEncoder(classLabels[att]) 
                encodingLengths[att] = len(classLabels[att])
            
            att = 'userId'
            oneHotEncoders[att], oneHotMap[att] = self.buildEncoder(classLabels[att]) 
            encodingLengths[att] = 1
            
            for att in self.isSequence:
                encodingLengths[att] = 1
            
            # summary information
            self.numDataPoints=len(self.original_data)
            
            # normalize continuous: altitude, heart_rate, latitude, longitude, speed and all derives            
            self.computeMeanStd(variableSums, self.numDataPoints, self.isSequence)
    
            self.oneHotEncoders=oneHotEncoders
            self.oneHotMap = oneHotMap
            self.encodingLengths = encodingLengths
            #Save that summary file so that it can be used next time
            self.writeSummaryFile()

 
    def computeMeanStd(self, varSums, numDataPoints, attributes):
        print("Computing variable means and standard deviations")
        
        # assume each data point has 500 time step?! is it correct?
        numSequencePoints = numDataPoints * 500 
        
        variableMeans = {}
        for att in varSums:
            variableMeans[att] = varSums[att] / numSequencePoints
        
        varResidualSums = defaultdict(float)
        
        for numDataPoints, currData in enumerate(self.original_data):
            # loop each continuous attribute
            for att in attributes:
                dataPointArray = np.array(currData[att])
                # add to the variable running sum of squared residuals
                diff = np.subtract(dataPointArray, variableMeans[att])
                sq = np.square(diff)
                varResidualSums[att] += np.sum(sq)

        variableStds = {}
        for att in varResidualSums:
            variableStds[att] = np.sqrt(varResidualSums[att] / numSequencePoints)
            
        self.variableMeans = variableMeans
        self.variableStds = variableStds
        
        
    # scale continuous data
    def scale_data(self, scaling=True): 
        print("scale data")
        targetAtts = ['heart_rate', 'derived_speed']

        for idx, currentDataPoint in enumerate(self.original_data):
            # target attribute, add to dict 
            for tAtt in targetAtts:         
                if self.perform_target_smoothing:
                    tar_data = self.median_smoothing(currentDataPoint[tAtt], self.smooth_window)
                else:
                    tar_data = currentDataPoint[tAtt]
                if self.scale_targets:
                    tar_data = self.scaleData(tar_data, tAtt, self.zMultiple) 
                self.original_data[idx]["tar_" + tAtt] = tar_data
                    
            # continuous input attribute, update dict
            for att in self.isSequence: 
                if scaling:
                    in_data = currentDataPoint[att]
                    self.original_data[idx][att] = self.scaleData(in_data, att, self.zMultiple) 
        for d in self.original_data:
            key = 'url'
            del d[key]
            key = 'speed'
            if key in d:
                del d[key]
        
        # write to disk
        np.save([self.original_data], self.processed_path)


class metaDataEndomondo(object):
    def __init__(self, numDataPoints, encodingLengths, oneHotEncoders, oneHotMap, isSequence, isNominal, isDerived,
                 variableMeans, variableStds):
        self.numDataPoints = numDataPoints
        self.encodingLengths = encodingLengths
        self.oneHotEncoders = oneHotEncoders
        self.oneHotMap = oneHotMap
        self.isSequence = isSequence
        self.isNominal = isNominal
        self.isDerived = isDerived
        self.variableMeans = variableMeans
        self.variableStds = variableStds


if __name__ == "__main__":

    data_path = "endomondoHR_proper.json"
    attrFeatures = ['userId', 'sport', 'gender']
    trainValidTestSplit = [0.8, 0.1, 0.1]
    targetAtts = ["derived_speed"]
    inputAtts = ["distance", "altitude", "time_elapsed"]
    endo_reader = dataInterpreter(inputAtts)
    endo_reader.preprocess_data()

