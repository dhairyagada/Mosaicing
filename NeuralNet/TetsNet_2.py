import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pprint

pp = pprint.PrettyPrinter()

## Training Dataset 

DataSetSize = 200
Train_x = np.zeros((DataSetSize,48))
Train_x[:,0:24] = np.random.rand(DataSetSize,24)
Train_y = np.random.randint(0,high=10,size=(DataSetSize,3))
for i in range(DataSetSize):
    Train_x[i,24:32]= Train_y[i,0]*Train_x[i,0:8]
    Train_x[i,32:40]= Train_y[i,1]*Train_x[i,8:16]
    Train_x[i,40:48]= Train_y[i,2]*Train_x[i,16:24]

Test_x = np.zeros((10,48))
Test_x[:,0:24] = np.random.rand(10,24)
Test_y = np.random.randint(0,high=10,size=(10,3))
for i in range(10):
    Test_x[i,24:32]= Test_y[i,0]*Test_x[i,0:8]
    Test_x[i,32:40]= Test_y[i,1]*Test_x[i,8:16]
    Test_x[i,40:48]= Test_y[i,2]*Test_x[i,16:24]

xT = np.zeros((1,48))
for k in range(8):
    xT[0,k] = 0.1
    xT[0,k+8] = 0.2
    xT[0,k+16] = 0.4
    xT[0,k+24] = 0.4
    xT[0,k+32] = 0.4
    xT[0,k+40] = 1.2 

pp.pprint(xT[0,0:8])
pp.pprint(xT[0,8:16])
pp.pprint(xT[0,16:24])
pp.pprint(xT[0,24:32])
pp.pprint(xT[0,32:40])
pp.pprint(xT[0,40:48])

print(Train_x.shape)
print(Train_y.shape)

learning_rate = 0.01
training_epochs = 10000

cost_history = np.empty(shape=[1],dtype=float)

N_dim = Train_x.shape[1]                                  ## Input Nodes
n_class = Train_y.shape[1]                                ## Output Nodes

print(N_dim)
print(n_class)

model_path = "/home/dhairya/Desktop/FYP/Mosaicing/NeuralNet/TetsNet_2.py"

n_hidden_1 = 8
n_hidden_2 = 4

x = tf.placeholder(tf.float32,[None,N_dim])
w = tf.Variable(tf.zeros([N_dim,n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32,[None,n_class])



def MultiLayerPerceptron(x,weights,biases):

    layer1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer1 = tf.nn.sigmoid(layer1)

    layer2 = tf.add(tf.matmul(layer1,weights['h2']),biases['b2'])
    layer2 = tf.nn.sigmoid(layer2)

    out_layer =  tf.add(tf.matmul(layer2,weights['out']),biases['out'])

    return out_layer

weights = {
    'h1' : tf.Variable(tf.truncated_normal([N_dim,n_hidden_1])),
    'h2' : tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_2,n_class]))
}

biases = {
    'b1' : tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.truncated_normal([n_hidden_2])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
}

init = tf.global_variables_initializer()

saver = tf.train.Saver()

y = MultiLayerPerceptron(x,weights,biases)

cost_function = tf.reduce_mean(tf.losses.mean_squared_error(y_,y,scope=None))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess =tf.Session()
sess.run(init)

mse_history = []
accuracy_history =[]

for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={x: Train_x,y_:Train_y})
    cost = sess.run(cost_function,feed_dict={x: Train_x,y_:Train_y})
     
    cost_history = np.append(cost_history,cost)
    
    pred_y =  sess.run(y,feed_dict={x:Test_x})
    mse = tf.reduce_mean(tf.square(pred_y-Test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_) 
    
    print('epoch :',epoch,'   cost : ',cost,'    mse : ',mse_)

print(sess.run(y,feed_dict={x:xT}))
plt.plot(mse_history,'r')
plt.show()
save_path = saver.save(sess,model_path)
print('Done')