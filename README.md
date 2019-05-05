### Code for our WWW 19' paper 
- Modeling Heart Rate and Activity Data for Personalized Fitness Recommendation

You can find the data at here: https://sites.google.com/eng.ucsd.edu/fitrec-project/home. To run the code below, you need to first download the data files.

### FitRec
This is the code for the workout profile prediction task. Data file you need: processed_endomondHR_proper.npy[https://drive.google.com/open?id=12ymlWEcKhVuQ3syNb92zVMmowAsZwSZ4].

```data_split.py``` - first run this to split the dataset into train/valid/test.  
```heart_rate_aux.py``` - run this file to predict the heart rate given the route and target time.
```speed_aux.py``` - run this file to predict the speed given the route and target time.


### FitRec-Attn
This is the attention-based model for the short-term prediction task. Data file you need: processed_endomondoHR_proper_interpolate.npy[https://drive.google.com/file/d/1L0BqpXtYrLyrG7A9JP7w0ACvTuRTXhxT/view?usp=sharing].



### Requirement
Tensorflow==1.4
Keras==2.0
PyTorch==2.0

