from __future__ import print_function
import numpy

filename = 'car_training.csv'
raw_data = open(filename, 'rt')
data = numpy.loadtxt(raw_data, delimiter=",")		

'''
input > weight > hidden layer 1 (activation function) > weight > hidden layer 2 (activation function) > weight > output layer

compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer.... SGD, AdaGrad)

backpropagation

feed forward + backprop = epoch

'''




import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 10
#batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 4 # 1st layer number of features
n_hidden_2 = 4 # 2nd layer number of features
n_input = 6    #  data input 
n_classes = 1 #  total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float",[None, n_input]) #[None, n_input]
y = tf.placeholder("float",[None, n_classes]) #[None, n_classes]


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(data)/6) #int((8*len(data))/batch_size)
        # Loop over all batches
        for j in range(total_batch):
	    i = (j+1) * 6
	    if i < len(data)-6:
	    	tempx = [[data[i,0],data[i,1],data[i,2],data[i,3],data[i,4],data[i,5]],[data[i+1,0],data[i+1,1],data[i+1,2],data[i+1,3],data[i+1,4],data[i+1,5]],[data[i+2,0],data[i+2,1],data[i+2,2],data[i+2,3],data[i+2,4],data[i+2,5]],[data[i+3,0],data[i+3,1],data[i+3,2],data[i+3,3],data[i+3,4],data[i+3,5]],[data[i+4,0],data[i+4,1],data[i+4,2],data[i+4,3],data[i+4,4],data[i+4,5]],[data[i+5,0],data[i+5,1],data[i+5,2],data[i+5,3],data[i+5,4],data[i+5,5]]]


	    	tempy = [[data[i,6]],[data[i+1,6]],[data[i+2,6]],[data[i+3,6]],[data[i+4,6]],[data[i+5,6]]]
	    
            batch_x, batch_y = tempx,tempy#mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
	
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    filename = 'validation_set.csv'
    raw_data = open(filename, 'rt')
    testdata = numpy.loadtxt(raw_data, delimiter=",")
    x_test=[]
    y_test=[]
    for i in range(8):
      x_test.append([])
      #y_test.append([])
      for j in range(6):
        if j < 6:
          x_test[i].append(testdata[i][j])
        #else:
          #y_test[i].append(testdata[i][j])
    y_test = [[testdata[0][6]],[testdata[1][6]],[testdata[2][6]],[testdata[3][6]],[testdata[4][6]],[testdata[5][6]],[testdata[0][6]],[testdata[6][6]],[testdata[7][6]]]

#print(testdata)
    

    print("Accuracy:", accuracy.eval(feed_dict={x: x_test, y: y_test})) #testdata,testlabels


