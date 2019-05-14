import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.contrib import seq2seq
import numpy as np

idx2char = ['h', 'i', 'e', 'l', 'o']
# Teach hello: hihell -> ihello
x_data = [[0, 1, 0, 2, 3, 3]]  # hihell
x_one_hot = [[[1, 0, 0, 0, 0],  # h 0
              [0, 1, 0, 0, 0],  # i 1
              [1, 0, 0, 0, 0],  # h 0
              [0, 0, 1, 0, 0],  # e 2
              [0, 0, 0, 1, 0],  # l 3
              [0, 0, 0, 1, 0]]]  # l 3
y_data = [[1, 0, 2, 3, 3, 4]]  # ihello

#参数
num_classes=5
input_dim=5
hidden_size=5
batch_size=1
sequence=6 #len(ihello)==6
learning_rate=0.1

X=tf.placeholder(dtype=tf.float32,shape=([None,sequence,input_dim]))
Y=tf.placeholder(dtype=tf.int32,shape=([None,sequence]))

cell=rnn.BasicLSTMCell(num_units=hidden_size,state_is_tuple=True)
initial_state=cell.zero_state(batch_size,tf.float32)
outputs,_states=tf.nn.dynamic_rnn(cell,X,initial_state=initial_state,dtype=tf.float32)

#fc
X_for_fc=tf.reshape(outputs,[-1,hidden_size])
outputs=layers.fully_connected(inputs=X_for_fc,num_outputs=num_classes,activation_fn=None)

outputs=tf.reshape(outputs,[batch_size,sequence,num_classes])#shape=(批次，输出个数，词库内的单词个数)

weights=tf.ones([batch_size,sequence])
sequence_loss=seq2seq.sequence_loss(logits=outputs,targets=Y,weights=weights)
loss=tf.reduce_mean(sequence_loss)
train=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

pred=tf.argmax(outputs,axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        loss_val,_=sess.run([loss,train],feed_dict={X:x_one_hot,Y:y_data})
        result=sess.run(pred,feed_dict={X:x_one_hot})
        print(i,'loss:',loss_val,'prediction:',result,'true Y:',y_data)
        print(result.shape)
        result_str=[idx2char[c] for c in np.squeeze(result)]#np.squeeze 去掉shape中维度为1的
        print('\tprediction str:',''.join(result_str))