from __future__ import print_function
import numpy
from tensorflow.python.client import timeline

filename = 'yeast.data'
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
training_epochs = 200
#batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 4 # 1st layer number of features
# n_hidden_2 =  # 2nd layer number of features
n_input = 8 #  data input 
n_classes = 10 #  total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float",[None, n_input]) #[None, n_input]
y = tf.placeholder("float",[None, n_classes]) #[None, n_classes]


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
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
    #sess.run(init)

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(init, options=run_options, run_metadata=run_metadata)

    # Create the Timeline object, and write it to a json
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
        f.write(ctf)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(data)/8) #int((8*len(data))/batch_size)
        # Loop over all batches
        for j in range(total_batch):
	    i = (j+1) * 8
	    if i < len(data)-8:
	    	tempx = [[data[i,0],data[i,1],data[i,2],data[i,3],data[i,4],data[i,5],data[i,6],data[i,7]],[data[i+1,0],data[i+1,1],data[i+1,2],data[i+1,3],data[i+1,4],data[i+1,5],data[i+1,6],data[i+1,7]],[data[i+2,0],data[i+2,1],data[i+2,2],data[i+2,3],data[i+2,4],data[i+2,5],data[i+2,6],data[i+2,7]],[data[i+3,0],data[i+3,1],data[i+3,2],data[i+3,3],data[i+3,4],data[i+3,5],data[i+3,6],data[i+3,7]],[data[i+4,0],data[i+4,1],data[i+4,2],data[i+4,3],data[i+4,4],data[i+4,5],data[i+4,6],data[i+4,7]],[data[i+5,0],data[i+5,1],data[i+5,2],data[i+5,3],data[i+5,4],data[i+5,5],data[i+5,6],data[i+5,7]],[data[i+6,0],data[i+6,1],data[i+6,2],data[i+6,3],data[i+6,4],data[i+6,5],data[i+6,6],data[i+6,7]],[data[i+7,0],data[i+7,1],data[i+7,2],data[i+7,3],data[i+7,4],data[i+7,5],data[i+7,6],data[i+7,7]]]


	    	tempy = [[data[i,8],data[i,9],data[i,10],data[i,11],data[i,12],data[i,13],data[i,14],data[i,15],data[i,16],data[i,17]],[data[i+1,8],data[i+1,9],data[i+1,10],data[i+1,11],data[i+1,12],data[i+1,13],data[i+1,14],data[i+1,15],data[i+1,16],data[i+1,17]],[data[i+2,8],data[i+2,9],data[i+2,10],data[i+2,11],data[i+2,12],data[i+2,13],data[i+2,14],data[i+2,15],data[i+2,16],data[i+2,17]],[data[i+3,8],data[i+3,9],data[i+3,10],data[i+3,11],data[i+3,12],data[i+3,13],data[i+3,14],data[i+3,15],data[i+3,16],data[i+3,17]],[data[i+4,8],data[i+4,9],data[i+4,10],data[i+4,11],data[i+4,12],data[i+4,13],data[i+4,14],data[i+4,15],data[i+4,16],data[i+4,17]],[data[i+5,8],data[i+5,9],data[i+5,10],data[i+5,11],data[i+5,12],data[i+5,13],data[i+5,14],data[i+5,15],data[i+5,16],data[i+5,17]],[data[i+6,8],data[i+6,9],data[i+6,10],data[i+6,11],data[i+6,12],data[i+6,13],data[i+6,14],data[i+6,15],data[i+6,16],data[i+6,17]],[data[i+7,8],data[i+7,9],data[i+7,10],data[i+7,11],data[i+7,12],data[i+7,13],data[i+7,14],data[i+7,15],data[i+7,16],data[i+7,17]]]
	    
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
    filename = 'ecoli.data'
    raw_data = open(filename, 'rt')
    testdata = numpy.loadtxt(raw_data, delimiter=",")
    x_test=[]
    y_test=[]
    for i in range(8):
      x_test.append([])
      y_test.append([])
      for j in range(17):
        if j < 7:
          x_test[i].append(testdata[i][j])
          if j == 6:
            x_test[i].append(0.1)
        else:
          y_test[i].append(testdata[i][j])

#print(testdata)
    

    print("Accuracy:", accuracy.eval(feed_dict={x: x_test, y: y_test})) #testdata,testlabels
    


