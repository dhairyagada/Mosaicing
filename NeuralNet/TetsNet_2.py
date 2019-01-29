import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pprint

pp = pprint.PrettyPrinter()

## Training Dataset 

Train_x = np.load('RandomDataX.npy')
Train_y = np.load('RandomDataY.npy')

print(Train_x.shape)
print(Train_y.shape)

learning_rate = 0.001
training_epochs = 500

cost_history = np.empty(shape=[1],dtype=float)

N_dim = Train_x.shape[1]                                  ## Input Nodes
n_class = Train_y.shape[1]                                ## Output Nodes

print(N_dim)
print(n_class)

model_path = "/home/dhairya/Desktop/FYP/Mosaicing/NeuralNet/TetsNet_2.py"

n_hidden_1 = 16
n_hidden_2 = 8
n_hidden_3 = 4

x = tf.placeholder(tf.float32,[None,N_dim])
w = tf.Variable(tf.zeros([N_dim,n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32,[None,n_class])



def MultiLayerPerceptron(x,weights,biases):

    layer1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer1 = tf.nn.tanh(layer1)

    layer2 = tf.add(tf.matmul(layer1,weights['h2']),biases['b2'])
    layer2 = tf.nn.tanh(layer2)

    layer3 = tf.add(tf.matmul(layer2,weights['h3']),biases['b3'])
    layer3 = tf.nn.tanh(layer3)

    out_layer =  tf.add(tf.matmul(layer3,weights['out']),biases['out'])

    return out_layer

weights = {
    'h1' : tf.Variable(tf.truncated_normal([N_dim,n_hidden_1])),
    'h2' : tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
    'h3' : tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_3])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_3,n_class]))
}

biases = {
    'b1' : tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3' : tf.Variable(tf.truncated_normal([n_hidden_3])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
}

init = tf.global_variables_initializer()

saver = tf.train.Saver()

y = MultiLayerPerceptron(x,weights,biases)


cost_function = tf.reduce_mean(tf.losses.absolute_difference(y_,y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess =tf.Session()
sess.run(init)

mse_history = []
accuracy_history =[]

for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={x: Train_x,y_:Train_y})
    cost = sess.run(cost_function,feed_dict={x: Train_x,y_:Train_y})
    
    cost_history = np.append(cost_history,cost)
    
    """  pred_y =  sess.run(y,feed_dict={x:Test_x})
    mse = tf.reduce_mean(tf.square(pred_y-Test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)  """
    
    print('epoch :',epoch,'   cost : ',cost)

print(sess.run(y,feed_dict={x:xT}))
print(3,2,5)
plt.plot(cost_history,'r')
plt.show()
save_path = saver.save(sess,model_path)
sess.close()
print('Done')