### Code for our WWW 19' paper 
 - Jianmo Ni, Larry Muhlstein, Julian McAuley, "Modeling heart rate and activity data for personalized fitness recommendation", in Proc. of the 2019 World Wide Web Conference (WWW'19), San Francisco, US, May. 2019.

You can find the data at here: https://sites.google.com/eng.ucsd.edu/fitrec-project/home. To run the code below, you need to first download the data files. Please cite our paper if you find the code and data helpful.

### FitRec
This is the code for the workout profile prediction task. Data file you need: [processed_endomondHR_proper.npy](https://drive.google.com/open?id=12ymlWEcKhVuQ3syNb92zVMmowAsZwSZ4).

 - `data_split.py` - First run this to split the dataset into train/valid/test. Or you can directly download the files here [endomondoHR_proper_temporal_dataset.pkl](https://drive.google.com/file/d/1GEUaNp04Yz0uUpTWjJPlO7tbA7SA_zu_/view?usp=sharing) and [endomondoHR_proper_metaData.pkl](https://drive.google.com/file/d/1Q8UYbDcKi_gHXwIuWyUbR0kfAqe4fAkS/view?usp=sharing).
 - `heart_rate_aux.py` - Run this file to predict the heart rate given the route and target time.
 - `speed_aux.py` - Run this file to predict the speed given the route and target time.
 - `data_interpreter_Keras_aux.py` - This is the dataloader file. 

### FitRec-Attn
This is the attention-based model for the short-term prediction task. Data file you need: [processed_endomondoHR_proper_interpolate.npy](https://drive.google.com/file/d/1L0BqpXtYrLyrG7A9JP7w0ACvTuRTXhxT/view?usp=sharing).

 - `data_split.py` - Similarlry, first run this to split file. Or you can directly download the files here [endomondoHR_proper_temporal_dataset.pkl](https://drive.google.com/file/d/1F8bzJrZUZ-3vpxs5RkJCHZ2kiDQPOC13/view?usp=sharing).
 - `fitrect-attn.py` - Run this to predict short-term prediction using our attention-based model.
 - `data_interpolate.py` - This is the dataloader file.

### Requirement
 - Tensorflow>=1.4
 - Keras=2.0
 - PyTorch>=0.4

