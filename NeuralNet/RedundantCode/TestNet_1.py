import tensorflow as tf


## Build a graph

#node1 = tf.constant(3.0)
#node2 = tf.constant(4.0)
""" node1 = tf.placeholder(tf.float32)
node2 = tf.placeholder(tf.float32)
c = node1*node2 """

W = tf.Variable([0.0],tf.float32)
b = tf.Variable([0.0],tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W*x +b

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_deltas)
#loss = tf.losses.mean_squared_error(y,linear_model)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()


## Run the graph

sess = tf.Session()
sess.run(init)


#File_Writer = tf.summary.FileWriter('NeuralNet/graph1',sess.graph)

#print(sess.run([c]))
#print(sess.run(c,{node1:[3.0,5.0],node2:[4.0,5.0]}))

for i in range(10):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
    print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))
print(sess.run([W,b]))
print(sess.run(linear_model,{x:[5,10,20]}))
sess.close()