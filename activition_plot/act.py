import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10,10,100)
y1 = tf.nn.sigmoid(x)
y2 = tf.nn.elu(x)
y3 = tf.nn.relu(x)
y4 = tf.nn.tanh(x)
y5 = tf.nn.softsign(x)
y6 = tf.nn.relu6(x)
y7 = tf.nn.softplus(x)
def drop_plot(sess,x,y,title,location):
   
    ax1 = plt.subplot2grid((6,2),location)
    ax1.plot(x,sess.run(y))
    ax1.set_title(title)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    fig = plt.figure(figsize=(12.0, 8.0))
    drop_plot(sess,x,y1,'sigmoid',(0,0))
    drop_plot(sess,x,y2,'elu',(0,1)) 
    drop_plot(sess,x,y3,'relu',(2,0)) 
    drop_plot(sess,x,y4,'tanh',(2,1))
    drop_plot(sess,x,y5,'softsign',(4,0))
    drop_plot(sess,x,y6,'relu6',(4,1))
    drop_plot(sess,x,y7,'softplus',(5,1))
    plt.show()


