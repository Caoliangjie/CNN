import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *
np.random.seed(1)
X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes=load_dataset()
index=6
plt.imshow(X_train_orig[index])
print("y="+str(np.squeeze(Y_train_orig[:,index])))
plt.show()##不是ipythonnotebook加这句才能把图片显示出来哦
X_train=X_train_orig/255
X_test=X_test_orig/255
Y_train=convert_to_one_hot(Y_train_orig,6).T
Y_test=convert_to_one_hot(Y_test_orig,6).T
print("number of training examples="+str(X_train.shape[0]))
print("number of test examples="+str(X_test.shape[0]))
print("X_train shape:"+str(X_train.shape))
print("Y_train shape:"+str(Y_train.shape))
print("X_test shape:"+str(X_test.shape))
print("Y_test shape:"+str(Y_test.shape))
conv_layers={}
def create_placeholders(n_H0,n_W0,n_C0,n_y):##存放输入的函数
    X=tf.placeholder(tf.float32,shape=(None,n_H0,n_W0,n_C0))
    Y=tf.placeholder(tf.float32,shape=(None,n_y))
    return X,Y
X,Y=create_placeholders(64,64,3,6)
print("X="+str(X))
print("Y="+str(Y))
def initialize_parameters():
    tf.set_random_seed(1)
    W1=tf.get_variable("W1",[4,4,3,8],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2=tf.get_variable("W2",[2,2,8,16],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters={"W1":W1,
                "W2":W2}
    return parameters
tf.reset_default_graph()
with tf.Session() as sess_test:
    parameters=initialize_parameters()
    init=tf.global_variables_initializer()
    sess_test.run(init)
    print("W1="+str(parameters["W1"].eval()[1,1,1]))
    print("W2="+str(parameters["W2"].eval()[1,1,1]))
def forward_propagation(X,parameters):#前向过程
    W1=parameters['W1']
    W2=parameters['W2']
    Z1=tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')##卷积层
    A1=tf.nn.relu(Z1)#relu层
    P1=tf.nn.max_pool(A1,ksize=[1,8,8,1],strides=[1,8,8,1],padding='SAME')#池化层，用的是max_pooling
    Z2=tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='SAME')#第二个卷积层
    A2=tf.nn.relu(Z2)#第二个relu
    P2=tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')#第二个conv层
    P2=tf.contrib.layers.flatten(P2)#flatten层
    Z3=tf.contrib.layers.fully_connected(P2,num_outputs=6,activation_fn=None)#全连接层
    return Z3
tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(64, 64, 3, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
    print("Z3 = " + str(a))
def compute_cost(Z3,Y):
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))
    return cost
tf.reset_default_graph()
with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(64, 64, 3, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(cost, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})
    print("cost = " + str(a))

def model(X_train,Y_train,X_test,Y_test,learning_rate=0.009,num_epochs=100,minibatch_size=64,print_cost=True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed=3
    (m,n_H0,n_W0,n_C0)=X_train.shape
    n_y=Y_train.shape[1]
    costs=[]
    X,Y=create_placeholders(n_H0,n_W0,n_C0,n_y)
    parameters=initialize_parameters()
    Z3=forward_propagation(X,parameters)
    cost=compute_cost(Z3,Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost=0
            num_minibatches=int(m/minibatch_size)
            seed=seed+1
            minibatches=random_mini_batches(X_train,Y_train,minibatch_size,seed)
            for minibatch in minibatches:
                (minibatch_X,minibatch_Y)=minibatch
                _,temp_cost=sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
                minibatch_cost+=temp_cost/num_minibatches
            if print_cost==True and epoch % 5 ==0:
                print("Cost after peoch % i:%f" % (epoch,minibatch_cost))
            if print_cost==True and epoch % 1==0:
                costs.append(minibatch_cost)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learing rate ="+str(learning_rate))
        plt.show()
        predict_op=tf.argmax(Z3,1)
        correct_prediction=tf.equal(predict_op,tf.argmax(Y,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
        print(accuracy)
        train_accuracy=accuracy.eval({X:X_train,Y:Y_train})
        test_accuracy=accuracy.eval({X:X_test,Y:Y_test})
        print("Train Accuracy:",train_accuracy)
        print("Test Accuracy:", test_accuracy)
        return train_accuracy,test_accuracy,parameters
_, _, parameters = model(X_train, Y_train, X_test, Y_test)
fname = "images/87816.jpg"
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(1024,1600))
plt.imshow(my_image)
plt.show()
