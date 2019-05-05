## split data into train,valid, test
from pathlib import Path
import os
from tqdm import tqdm_notebook as tqdm
import json
import gzip
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import random
import collections
import pickle
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import time, datetime

import multiprocessing
from multiprocessing import Pool

def calc_mse(avg, hrs):
    dif = (np.array(hrs) - avg) ** 2
    return np.mean(dif)
def convert2datetime(unix_timestamp):
    utc_time = time.gmtime(unix_timestamp)
    l = time.localtime(unix_timestamp)
    dt = datetime.datetime(*l[:6])
    return dt
def parse(path):
    if 'gz' in path:
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')
    for l in f.readlines():
        yield(eval(l.decode('ascii')))
        
# load original data, filter invalid data point
# split into train,valid,test
# build context
path = Path("data/")
in_path = str(path / "endomondoHR_proper.json")
def process(line):
    return eval(line)
pool = Pool(5) 
with open(in_path, 'r') as f:
    data = pool.map(process, f)
pool.close()
pool.join()
print(len(data))


##################
###### process data - keep 10 core

data2 = data
print(len(data2))
# aggregate workout id for each user
user2workout = defaultdict(list) 
idxMap = {}
for idx in range(len(data2)):
    d = data2[idx]
    wid = d['id']
    uid = d['userId']
    user2workout[uid].append(wid)
    idxMap[wid] = idx
print(len(user2workout)) # how many user

# sort user's workout based on time
for u in user2workout:
    workout = user2workout[u]
    dts = [(convert2datetime(data2[idxMap[wid]]['timestamp'][0]), wid) for wid in workout]
    dts = sorted(dts, key=lambda x:x[0])
    new_workout = [x[1] for x in dts] # ascending
    user2workout[u] = new_workout

# keep 10 core
user2workout_core = defaultdict(list)
for u in user2workout:
    workout = user2workout[u]
    if len(workout) >= 10:
        user2workout_core[u] = workout
print(len(user2workout_core))
# total workouts in 10 core subset
count = 0
for u in user2workout_core:
    count += len(user2workout_core[u])
print(count)

# build time lines
times = defaultdict(float)
user_times = defaultdict(float)
for u in user2workout_core:
    workout = user2workout_core[u]
    tt = []
    for wid in workout:
        idx = idxMap[wid]
        d = data2[idx]
        ts = d['timestamp']
        times[wid] = (ts[-1] - ts[0]) / 3600
        tt.append((ts[-1] - ts[0]) / 3600)
    tt = np.array(tt).mean()
    user_times[u] = tt
        
vals = np.array(list(times.values()))
print(vals.mean())

# contextMap stores all workouts previous to current
contextMap= {}
for u in user2workout_core:
    wids = user2workout_core[u]
    indices = [idxMap[wid] for wid in wids]
    
    # build start time
    start_times = []
    for idx in indices:
        start_time = data[idx]['timestamp'][0]
        start_times.append(start_time)
    
    # build context
    for i in range(1, len(wids)):
        wid = wids[i]
        since_last = (start_times[i] - start_times[i-1]) / (3600*24)
        since_begin = (start_times[i] - start_times[0]) / (3600*24)
        contextMap[wid] = (since_last, since_begin, wids[:i]) 
    
print(len(contextMap))
        
# normalize since last and begin
since_last_array = []
since_begin_array = []
for wid in contextMap:
    t = contextMap[wid]
    since_last_array.append(t[0])
    since_begin_array.append(t[1])

since_last_array  = np.array(since_last_array)
since_begin_array  = np.array(since_begin_array)

def normalize(inp, zMultiple=5):
    mean, std = inp.mean(), inp.std()
    diff = inp - mean
    zScore = diff / std 
    return zScore * zMultiple

since_last_array2 = normalize(since_last_array)
since_begin_array2 = normalize(since_begin_array)

# put nomalized since last and begin into contextMap
i = 0
contextMap2 = {}
for wid in contextMap:
    t = contextMap[wid]
    t0 = since_last_array2[i]
    t1 = since_begin_array2[i]
    i += 1
    contextMap2[wid] = (t0, t1, t[2])
    
    
# split whole dataset, leave latest into valid and test
train,valid,test = [],[],[]
for u in user2workout_core:
    indices = user2workout_core[u][1:] # remove the first workout since it has no context
    l = len(indices)
    # split in ascending order
    train.extend(indices[:int(0.8*l)])
    valid.extend(indices[int(0.8*l):int(0.9*l)])
    test.extend(indices[int(0.9*l):])
    
print("train/valid/test = {}/{}/{}".format(len(train), len(valid), len(test)))

with open('endomondoHR_proper_temporal_dataset.pkl', 'wb') as f:
    pickle.dump((train,valid,test,contextMap2), f)
    
    
    


