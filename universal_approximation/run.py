import time
import os
import argparse
import io
dir = os.path.dirname(os.path.realpath(__file__))
print(dir)



import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def univApprox(x,hidden_dim=50):
    input_dim = 1
    output_dim = 1

    with tf.variable_scope('UniversalApproximator'):
        ua_w = tf.get_variable(name='ua_w',shape=[input_dim,hidden_dim],initializer=tf.random_normal_initializer(stddev=0.1))
        ua_b = tf.get_variable(name='ua_b',shape=[hidden_dim],initializer=tf.constant_initializer(0.))

        z = tf.matmul(x,ua_w)+ua_b
        a = tf.nn.relu(z)

        ua_v = tf.get_variable(name = 'ua_v',shape=[hidden_dim,output_dim],initializer=tf.random_normal_initializer(stddev=0.1))

        z = tf.matmul(a,ua_v)

    return z

def function_to_appox(x):
    return tf.sin(x)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_neurons',default=50,type=int,help="Number of Neurons ")
    args = parser.parse_args()

    with tf.variable_scope('Graph') as scope:

        x = tf.placeholder(tf.float32,shape=[None,1],name='x')

        y_true = function_to_appox(x)

        y = univApprox(x,args.nb_neurons)


        with tf.variable_scope('Loss'):
            loss = tf.reduce_mean(tf.square(y-y_true))
            loss_summary_t = tf.summary.scalar('loss',loss)

        adam = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = adam.minimize(loss)

    
    with tf.variable_scope('TensorboardMatPlotInput') as scope:
        image_strbuf_plh = tf.placeholder(tf.string,shape=[])
        my_img = tf.image.decode_png(image_strbuf_plh,4)
        image_summary  = tf.summary.image(
            'matplotlib_graph',
            tf.expand_dims(my_img,0)
        )        
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        result_folder = dir +'/results/' + str(int(time.time()))
        sw = tf.summary.FileWriter(result_folder,sess.graph)

        sess.run(tf.global_variables_initializer())
        for i in range(3000):
            x_in = np.random.uniform(-10,10,[100000,1])
            current_loss,loss_summary,_ = sess.run([loss,loss_summary_t,train_op],feed_dict={
                 x:x_in
            })
            sw.add_summary(loss_summary,i+1)

            if (i+1)%100 == 0:
                print('batch :{},loss:{}'.format(i+1,current_loss))

        print('Plotting graphs')

        inputs = np.array([[(i-100)/100] for i in range(2000)])
        y_true_res,y_res = sess.run([y_true,y],feed_dict={x:inputs})

        plt.figure(1)
        plt.subplot(211)
        plt.plot(inputs,y_true_res.flatten())
        plt.subplot(212)
        plt.plot(inputs,y_res) 

        imgdata = io.BytesIO()

        plt.savefig(imgdata,format='png')

        imgdata.seek(0)

        plot_img_summary=sess.run(image_summary,feed_dict={image_strbuf_plh:imgdata.getvalue()})

        sw.add_summary(plot_img_summary,i+1)
        plt.clf()

        saver.save(sess,result_folder+"/data.chkp")

