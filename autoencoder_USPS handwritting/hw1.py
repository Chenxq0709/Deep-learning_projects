import tensorflow as tf
import numpy as np
from mat4py import loadmat
import matplotlib.pyplot as plt

def load_data(mode='train'):
    if mode == 'train':
        load_data = loadmat('usps.mat')
        x_train = np.array(load_data['usps_train_input'])
        y_train = np.array(load_data['usps_train_target'])
        for i in range(0, len(x_train)):
            x_train[i] = np.rot90(np.flip(x_train[i].reshape(16, 16), 0),3).reshape(1, 256)

        return x_train, y_train
    elif mode == 'test':
        load_data = loadmat('usps.mat')
        x_test = np.array(load_data['usps_test_input'])
        y_test = np.array(load_data['usps_test_target'])
        for i in range(0, len(x_test)):
            x_test[i]= np.rot90(np.flip(x_test[i].reshape(16, 16), 0),3).reshape(1, 256)

        return x_test, y_test
    elif mode == 'all':
        load_data = loadmat('usps.mat')
        x_all = np.array(load_data['usps_all_input'])
        y_all = np.array(load_data['usps_all_target'])
        for i in range(0, len(x_all)):
            x_all[i]= np.rot90(np.flip(x_all[i].reshape(16, 16), 0),3).reshape(1, 256)

    return x_all, y_all

def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W_' + name,
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)


def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name,
                           dtype=tf.float32,
                           initializer=initial)

def fc_layer(x, num_units, name, use_relu=True):
    """
    Create a fully-connected layer
    :param x: input from previous layer
    :param num_units: number of hidden units in the fully-connected layer
    :param name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    with tf.variable_scope(name):
        in_dim = x.get_shape()[1]
        W = weight_variable(name, shape=[in_dim, num_units])
        tf.summary.histogram('W', W)
        b = bias_variable(name, [num_units])
        tf.summary.histogram('b', b)
        layer = tf.matmul(x, W)
        layer += b
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer


#main function______________________________________________________________
# Data Dimensions
img_h = img_w = 16  # USPS handwritten images are 16x16
img_size_flat = img_h * img_w  # 16x16=256, the total number of pixels
n_classes = 10  # Number of classes (0~9), one class per digit

# Load USPS data, in the encoder ,target is input
x_train, y_train= load_data(mode='train')
y_train = x_train
x_test, y_test = load_data(mode='test')
y_test = x_test

# Hyper-parameters
h = 101  # Number of units in the first hidden layer, eg 101,185,1420,4260
epochs = 491  # Total number of training epochs
batch_size = 80  # Training batch size
learning_rate = 0.001  # learning rate or gradient descent step size
do_early_stopping = True # if apply early stopping learning
N_weights = 2*img_size_flat*h + h+ img_size_flat

# Create the graph for the linear model
# Placeholders for inputs (x) and outputs(y)
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X')
y = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='Y')

# Create the network layers
fc1 = fc_layer(x, h, 'FC1', use_relu=True)
output = fc_layer(fc1, img_size_flat, 'OUT', use_relu=True)

# Define the loss function, optimizer, and gradient
loss = tf.losses.mean_squared_error(labels=output, predictions=x)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
grad_and_var = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').compute_gradients(loss)

train_RMSE = []
test_RMSE = []
train_G = []
test_G = []
train_IG = []
test_IG = []
if (do_early_stopping == True):
    best_so_far_loss = float("inf")
    best_so_far_epoch = 0


# Create the op for initializing all variables
init = tf.global_variables_initializer()
# Launch the graph (session)
with tf.Session() as sess:
    sess.run(init)
    global_step = 0
    # Number of training iterations in each epoch
    num_tr_iter = int(len(x_train) / batch_size)
    for epoch in range(epochs):
        #print('Training epoch: {}'.format(epoch + 1))
        x_train, y_train = randomize(x_train, y_train)
        for iteration in range(num_tr_iter):
            global_step += 1
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch = get_next_batch(x_train, y_train, start, end)

            # Run optimization op (backprop)
            feed_dict_batch = {x: x_batch, y: y_batch}
            sess.run(optimizer, feed_dict=feed_dict_batch)

        #after each epoch, record the three indicators
        feed_dict_valid = {x: x_train, y: y_train}
        loss_train, grad_and_var_train = sess.run([loss, grad_and_var], feed_dict=feed_dict_valid)
        RMSE = np.sqrt(loss_train)
        train_RMSE.append(RMSE)
        gradients = [x[0] for x in grad_and_var_train]
        G = tf.global_norm(gradients).eval()
        train_G.append(G)
        train_IG.append(G/(learning_rate * N_weights))

        # Test the network after training
        feed_dict_valid = {x: x_test, y: y_test}
        loss_test, grad_and_var_test = sess.run([loss, grad_and_var], feed_dict=feed_dict_valid)
        RMSE = np.sqrt(loss_test)
        test_RMSE.append(RMSE)
        gradients = [x[0] for x in grad_and_var_test]
        G = tf.global_norm(gradients).eval()
        test_G.append(G)
        test_IG.append(G/(learning_rate * N_weights))

        if do_early_stopping == True :
           if loss_test<best_so_far_loss:
                best_so_far_loss = loss_test
                best_so_far_epoch = epoch
                best_so_far_var = [x[1] for x in grad_and_var_train]

    print('Early stopping:test loss was lowest after', best_so_far_epoch ,'epoch. We chose the model that we had then.')

    plt.plot(train_RMSE,'g-',label= 'training')
    plt.plot(test_RMSE,'b--',label= 'test')
    plt.xlabel('epoch')
    plt.ylabel('Root Mean Squared Error')
    plt.legend()
    plt.show()

    plt.plot(train_G[1:epochs],'g-',label='training')
    plt.plot(test_G[1:epochs],'b--',label='test')
    plt.xlabel('epoch')
    plt.ylabel('Gradient of Mean Square Error')
    plt.legend()
    plt.show()

    plt.plot(train_IG[1:epochs],'g-',label='training')
    plt.plot(test_IG[1:epochs],'b--',label='test')
    plt.xlabel('epoch')
    plt.ylabel('Weighted Gradient')
    plt.legend()
    plt.show()
