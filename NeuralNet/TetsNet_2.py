import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pprint


def PCARed(X):
    U, Sigma, VT = np.linalg.svd(X, full_matrices=True)
    num_components = 8 # Number of principal components
    Y = np.matmul(X, VT[:num_components,:].T)
    return Y

pp = pprint.PrettyPrinter()

## Training Dataset 

Train_x = np.load('RandomDataX.npy')
Train_y = np.load('RandomDataY.npy')

Train_x = PCARed(Train_x)
#Train_y = PCARed(Train_y)

print(Train_x.shape)
print(Train_y.shape)

learning_rate = tf.placeholder(tf.float32)
training_epochs = 1000

cost_history = np.empty(shape=[1],dtype=float)

N_dim = Train_x.shape[1]                                  ## Input Nodes
n_class = Train_y.shape[1]                                ## Output Nodes

print(N_dim)
print(n_class)

model_path = "/home/dhairya/Desktop/FYP/Mosaicing/NeuralNet/TetsNet_2.py"

n_hidden_1 = 7
n_hidden_2 = 6
#n_hidden_3 = 4

x = tf.placeholder(tf.float32,[None,N_dim])
w = tf.Variable(tf.zeros([N_dim,n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32,[None,n_class])

xmean, xvar = tf.nn.moments(x,[0])
xn = tf.nn.batch_normalization(x,xmean,xvar,0,0.01,0.001)
def MultiLayerPerceptron(x,weights,biases):

    layer1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer1 = tf.nn.tanh(layer1)
    layer1mean,layer1var = tf.nn.moments(layer1,[0])
    layer1 = tf.nn.batch_normalization(layer1,layer1mean,layer1var,0,0.01,0.001)


    layer2 = tf.add(tf.matmul(layer1,weights['h2']),biases['b2'])
    layer2 = tf.nn.tanh(layer2)
    layer2mean,layer2var = tf.nn.moments(layer2,[0])
    layer2 = tf.nn.batch_normalization(layer2,layer2mean,layer2var,0,0.01,0.001)


    #layer3 = tf.add(tf.matmul(layer2,weights['h3']),biases['b3'])
    #layer3 = tf.nn.tanh(layer3)layer2

    out_layer =  tf.add(tf.matmul(layer2,weights['out']),biases['out'])

    return out_layer

weights = {
    'h1' : tf.Variable(tf.truncated_normal([N_dim,n_hidden_1])),
    'h2' : tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
    #'h3' : tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_3])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_2,n_class]))
}

biases = {
    'b1' : tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.truncated_normal([n_hidden_2])),
    #'b3' : tf.Variable(tf.truncated_normal([n_hidden_3])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
}



y = MultiLayerPerceptron(xn,weights,biases)


#cost_function = tf.reduce_mean(tf.losses.absolute_difference(y_,y))
cost_function = tf.reduce_mean(tf.square(y-y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
tf.losses.mean_pairwise_squared_error
init = tf.global_variables_initializer()

saver = tf.train.Saver()
sess =tf.Session()
sess.run(init)

mse_history = []
accuracy_history =[]
## Batch1
for B in range(4):
    trange = B*2500
    print(B)
    for lr in [0.5]:
        for epoch in range(training_epochs):
            sess.run(training_step,feed_dict={x: Train_x[0:trange],y_:Train_y[0:trange],learning_rate:lr})
            cost = sess.run(cost_function,feed_dict={x: Train_x[0:trange],y_:Train_y[0:trange]})
            cost_history = np.append(cost_history,cost)

            print('Cost :[%f%%] \r'%cost,end="")


print('Cost :',cost)
xTestOld = np.load('TestDataX.npy')
yTest = np.load('TestDataY.npy')

xTest = PCARed(xTestOld)

yResult = sess.run(y,feed_dict={x:xTest})
pp.pprint(yResult)
pp.pprint(yTest)
pp.pprint(yTest-yResult)
plt.plot(cost_history,'r')
plt.show()
save_path = saver.save(sess,model_path)
sess.close()
print('Done')