#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from functools import reduce
import time
from tqdm import tqdm

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

NUM_FEATURES = 21
NUM_CLASSES = 3

learning_rate = 0.01
epochs = 5000
list_batch_size =[4, 8, 16, 32, 64]
batch_size = 32
num_neurons = 10
seed = 10
beta = 1e-6
no_folds = 5
np.random.seed(seed)

#read train data
train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter= ',')
trainX, train_Y = train_input[1:, :21], train_input[1:,-1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix


print('Sample size: %d'%(trainX.shape[0]))
n = trainX.shape[0]
a = np.arange(n)
np.random.shuffle(a)
trainX, trainY = trainX[a], trainY[a]

# Split into 70:30 (train:test)
trainX, testX, trainY, testY = train_test_split(trainX, trainY, test_size=0.30)

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

# Build the graph for the deep net
	
weights1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neurons], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weights1')
biases1  = tf.Variable(tf.zeros([num_neurons]), name='biases1')
hiddenlayer1 = tf.nn.relu(tf.matmul(x, weights1) + biases1)

weights2 = tf.Variable(tf.truncated_normal([num_neurons, NUM_CLASSES], stddev=1.0 / math.sqrt(float(num_neurons))), name='weights2')
biases2 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases2')
#logits = y
logits  = tf.matmul(hiddenlayer1, weights2) + biases2

#softmax activation
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
#J
#loss = tf.reduce_mean(cross_entropy)
regularizer = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)
l2_loss = tf.reduce_mean(cross_entropy + beta*regularizer)

# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(l2_loss)

correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

fold_size = int(trainX.shape[0]/ no_folds)
all_testmean_acc = []
all_testmean_loss = []
list_time_taken = []

for batch in list_batch_size:
    print('Training batch_size {}'.format(batch))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_acc = []
        test_acc = []
        temp_time_taken = []
        num_batches = int(fold_size/batch)
        print('Number of batches: %g'%(num_batches))

        # Arrays to store 5000 epochs' accuracies of each fold
        fold_0_acc = []
        fold_1_acc = []
        fold_2_acc = []
        fold_3_acc = []
        fold_4_acc = []

        # Arrays to store 5000 epochs' losses of each fold
        fold_0_loss = []
        fold_1_loss = []
        fold_2_loss = []
        fold_3_loss = []
        fold_4_loss = []

        fold_avg_arr = []
        fold_test_acc = []
        fold_test_loss = []

        for fold in range(no_folds):
            fold_start = fold*fold_size
            fold_end = fold_start + fold_size - 1
            x_test = trainX[fold_start: fold_end]
            y_test = trainY[fold_start: fold_end]
            x_train  = np.append(trainX[:fold_start], trainX[fold_end:], axis=0)
            y_train = np.append(trainY[:fold_start], trainY[fold_end:], axis=0)
            start_time = time.time()
            
            for i in tqdm(range(epochs)): #iteration for GD
                temp_train_accuracy = 0 #to store average accuracy
                temp_train_loss = 0
                for i in range(num_batches): #i starts from 0
                    start = i * batch_size
                    end = start + batch_size - 1 
                    batchX = x_train[start:end]
                    batchY = y_train[start:end]
                    train_op.run(feed_dict={x:batchX, y_:batchY})
            
                    temp_train_accuracy += accuracy.eval(feed_dict={x:x_train, y_:y_train})
                    temp_train_loss += l2_loss.eval(feed_dict={x:x_train, y_:y_train})
                        
                temp_time_taken.append(time.time() - start_time)
                train_acc.append(temp_train_accuracy/num_batches) # Mean training accuracy of each batch
 
                if(fold == 0):
                    fold_0_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))
                    fold_0_loss.append(l2_loss.eval(feed_dict={x: x_test, y_: y_test}))
                elif(fold == 1):
                    fold_1_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))
                    fold_1_loss.append(l2_loss.eval(feed_dict={x: x_test, y_: y_test}))
                elif(fold == 2):
                    fold_2_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))
                    fold_2_loss.append(l2_loss.eval(feed_dict={x: x_test, y_: y_test}))
                elif(fold == 3):
                    fold_3_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))
                    fold_3_loss.append(l2_loss.eval(feed_dict={x: x_test, y_: y_test}))
                elif(fold == 4):
                    fold_4_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))
                    fold_4_loss.append(l2_loss.eval(feed_dict={x: x_test, y_: y_test}))
   

        for f in range(epochs):
            # Avg accuracy of each epoch across diff folds
            test_avg_acc = ((fold_0_acc[f] + fold_1_acc[f] + fold_2_acc[f] + fold_3_acc[f] + fold_4_acc[f])/5)
            fold_test_acc.append(test_avg_acc)
            test_avg_loss = ((fold_0_loss[f] + fold_1_loss[f] + fold_2_loss[f] + fold_3_loss[f] + fold_4_loss[f])/5)
            fold_test_loss.append(test_avg_acc)

        

        list_time_taken.append(np.array(sum(map(np.array, temp_time_taken))) /no_folds)
        all_testmean_acc.append(fold_test_acc)
        all_testmean_loss.append(fold_test_loss)

        sess.close()

            
plt.figure()
for fold_test_acc in all_testmean_acc:
#plt.plot(range(epochs), train_mean_acc, label= 'train accuracy')
    plt.plot(range(epochs), fold_test_acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('CV Accuracy')
legend = []
for i in list_batch_size:
    legend.append("Batch Size " + str(i))
plt.legend(legend)
plt.show()
plt.savefig('Question_2a_accuracy.png')

plt.figure()
plt.title('Time taken for different batch sizes')
for x in range (len(list_time_taken)):
    plt.plot(range(epochs), list_time_taken[x], label='Batch sizes {}'.format(str(list_batch_size[x])))
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Time taken')
plt.legend()
plt.show()
plt.savefig('Question_2a_timetaken.png')
