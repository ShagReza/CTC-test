# -*- coding: utf-8 -*-
"""
5 Esfand 99
Shaghayegh reza
"""

import librosa
import matplotlib.pyplot as plt
from librosa import display



#----------------------------------------------------------
#MFCC transformation
y, sr = librosa.load('D:\\Shapar\\ShaghayeghUni\\AfterPropozal\\Phase3-SpeechRecognition\\CTCtest\\free-spoken-digit-dataset-master\\free-spoken-digit-dataset-master\\recordings\\0_george_0.wav')
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
print(mfcc.shape)
plt.figure(figsize=(6, 4))
librosa.display.specshow(mfcc, x_axis='time')
#----------------------------------------------------------





#----------------------------------------------------------
# Data preperation:
'''
train_data - A (sample size, timesteps, n_mfcc) size array
train_labels = A (sample size, timesteps, num_classes) size array
train_inp_lengths - A (sample size,)` size array (for CTC loss)
train_seq_lengths - A (sample size,)` size array (for CTC loss)

test_data - A (sample size, timesteps, n_mfcc) size array
test_labels = A (sample size, timesteps, num_classes+1) size array
test_inp_lengths - A (sample size,)` size array (for CTC loss)
test_seq_lengths - A (sample size,)` size array (for CTC loss)
'''
use globe
for i in enumerate:
    # extract mfcc
    # concatenate mfcc
    # use first character as class label
    # train_inp_lengths
    

# use 2500 first data as train and the rest 500 data for test
    
#----------------------------------------------------------





#----------------------------------------------------------
#Convert chars to numbers
alphabet = 'abcdefghijklmnopqrstuvwxyz '
a_map = {} # map letter to number
rev_a_map = {} # map number to letter
for i, a in enumerate(alphabet):
  a_map[a] = i
  rev_a_map[i] = a

label_map = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7: 'seven', 8: 'eight', 9:'nine'}
#----------------------------------------------------------




#----------------------------------------------------------
#Defining the model
def ctc_loss(inp_lengths, seq_lengths):
    def loss(y_true, y_pred):
        l = tf.reduce_mean(K.ctc_batch_cost(tf.argmax(y_true, axis=-1), y_pred, inp_lengths, seq_lengths))        
        return l            
    return loss

K.clear_session()
inp = tfk.Input(shape=(10,50))
inp_len = tfk.Input(shape=(1))
seq_len = tfk.Input(shape=(1))
out = tfkl.Conv1D(filters= 128, kernel_size= 5, padding='same', activation='relu')(inp)
out = tfkl.BatchNormalization()(out)
out = tfkl.Bidirectional(tfkl.GRU(128, return_sequences=True, implementation=0))(out)
out = tfkl.Dropout(0.2)(out)
out = tfkl.BatchNormalization()(out)
out = tfkl.TimeDistributed(tfkl.Dense(27, activation='softmax'))(out)
cnn_model = tfk.models.Model(inputs=[inp, inp_len, seq_len], outputs=out)
cnn_model.compile(loss=ctc_loss(inp_lengths=inp_len , seq_lengths=seq_len), optimizer='Adam', metrics=['mae'])
#----------------------------------------------------------



#----------------------------------------------------------
# training the model
cnn_model.fit([train_data, train_inp_lengths, train_seq_lengths], train_labels, batch_size=64, epochs=20)
#----------------------------------------------------------




#----------------------------------------------------------

# predicting with the model:
y = cnn_model.predict([test_data, test_inp_lengths, test_seq_lengths])

n_ids = 5

for pred, true in zip(y[:n_ids,:,:], test_labels[:n_ids,:,:]):
  pred_ids = np.argmax(pred,axis=-1)
  true_ids = np.argmax(true, axis=-1)
  print('pred > ',[rev_a_map[tid] for tid in pred_ids])
  print('true > ',[rev_a_map[tid] for tid in true_ids])
#----------------------------------------------------------
