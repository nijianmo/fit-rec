# split data into train,valid,test

from collections import defaultdict
import time
import datetime

def convert2datetime(unix_timestamp):
    utc_time = time.gmtime(unix_timestamp)
    l = time.localtime(unix_timestamp)
    dt = datetime.datetime(*l[:6])
    return dt

path = "data/processed_endomondoHR_proper_interpolate.npy"
data = np.load(path)[0]

# rebuild dataset

user2workout = defaultdict(list) # append workout id
idxMap = {}

for idx in range(len(data2)):
    d = data2[idx]
    wid = d['id']
    uid = d['userId']
    user2workout[uid].append(wid)
    idxMap[wid] = idx
    
print(len(data2))
len(user2workout)

for u in user2workout:
    workout = user2workout[u]
    dts = [(convert2datetime(data2[idxMap[wid]]['timestamp'][0]), wid) for wid in workout]
    dts = sorted(dts, key=lambda x:x[0])
    new_workout = [x[1] for x in dts] # ascending
    user2workout[u] = new_workout
    
user2workout_core = defaultdict(list)
for u in user2workout:
    workout = user2workout[u]
    if len(workout) >= 10:
        user2workout_core[u] = workout
        
print(len(user2workout_core))
count = 0
for u in user2workout_core:
    count += len(user2workout_core[u])
print(count)


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


# all workouts previous to current
contextMap= {}

for u in user2workout_core:
    wids = user2workout_core[u]
    indices = [idxMap[wid] for wid in wids]
    
    # build start time
    start_times = []
    for idx in indices:
        start_time = data2[idx]['timestamp'][0]
        start_times.append(start_time)
    
    # build context
    for i in range(1, len(wids)):
        wid = wids[i]
        since_last = (start_times[i] - start_times[i-1]) / (3600*24)
        since_begin = (start_times[i] - start_times[0]) / (3600*24)
        contextMap[wid] = (since_last, since_begin, wids[:i]) 
    
len(contextMap)
        
def normalize(inp, zMultiple=5):
    mean, std = inp.mean(), inp.std()
    diff = inp - mean
    zScore = diff / std 
    return zScore * zMultiple

# normalize since last and begin
wid2idx = {}

since_last_array = []
since_begin_array = []
for i,wid in enumerate(contextMap):
    t = contextMap[wid]
    wid2idx[wid] = i
    since_last_array.append(t[0])
    since_begin_array.append(t[1])

since_last_array  = np.array(since_last_array)
since_begin_array  = np.array(since_begin_array)

since_last_array2 = normalize(since_last_array)
since_begin_array2 = normalize(since_begin_array)

contextMap2 = {}
for wid in contextMap:
    t = contextMap[wid]
    i = wid2idx[wid]
    t0 = since_last_array2[i]
    t1 = since_begin_array2[i]
    contextMap2[wid] = (t0, t1, t[2])


# split
train,valid,test = [],[],[]
for u in user2workout_core:
    indices = user2workout_core[u][1:] # remove the first workout
    l = len(indices)
    # split in ascending order
    train.extend(indices[:int(0.8*l)])
    valid.extend(indices[int(0.8*l):int(0.9*l)])
    test.extend(indices[int(0.9*l):])
    
print("train/valid/test = {}/{}/{}".format(len(train), len(valid), len(test)))

with open('endomondoHR_proper_temporal_dataset.pkl', 'wb') as f:
    pickle.dump((train,valid,test,contextMap2), f)



