import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import datetime


def hw2_load_data(mode='train'):
    # Data Dimensions
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    img_h = img_w = 28  # USPS handwritten images are 16x16
    img_size_flat = img_h * img_w  # the total number of pixels

    n_shoes = 3
    class_names = ['Sandal', 'Sneaker', 'Ankle boot']
    class_shoes = [5, 7, 9]

    if mode == 'train':
        train_images = train_images.reshape(60000, img_size_flat)
        x_train = []
        y_train = []
        for i in range(60000):
            if train_labels[i] in class_shoes:
                x_train.append(train_images[i])
                y_train.append(np.eye(n_shoes)[int((train_labels[i] - 5) / 2)])
        return np.asarray(x_train), np.asarray(y_train)
    elif mode == 'test':
        test_images = test_images.reshape(10000, img_size_flat)
        x_test = []
        y_test = []
        for i in range(10000):
            if test_labels[i] in class_shoes:
                x_test.append(test_images[i])
                y_test.append(np.eye(n_shoes)[int((test_labels[i] - 5) / 2)])
        return np.asarray(x_test), np.asarray(y_test)


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation, :]
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


def fc_layer(x, num_units, name, use_sigmoid=True):
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
        if use_sigmoid:
            layer = tf.nn.sigmoid(layer)
        return layer


def Cal_hL(X_train, Y_train, n_classes):
    pca = PCA(0.90)
    s = []
    for i in range(n_classes):
        one_class = []
        for j in range(18000):
            if (Y_train[j] == np.eye(3)[i]).all():
                one_class.append(X_train[j])
        pca_result = pca.fit_transform(one_class)
        s.append(pca_result.shape[1])
    print(s)
    return (sum(s))


def Cal_h90(X_train, dim):
    pca = PCA(0.90)
    pca_result = pca.fit_transform(X_train)
    h_90 = pca_result.shape[1]

    pca = PCA(dim)
    pca_full = pca.fit(X_train)

    plt.plot(pca_full.explained_variance_ratio_)
    plt.ylabel('Eigenvalues in Decreasing order')
    plt.show()

    plt.plot(np.cumsum(pca_full.explained_variance_ratio_), 'k')
    h90_dot = plt.plot([h_90], [0.9], 'ro', label='90% explained')

    plt.xlabel('# of components')
    plt.ylabel('Cumulative explained the sum of eigenvalues')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    plt.show()
    return h_90


# main function______________________________________________________________
# Data Dimensions
img_h = img_w = 28  # USPS handwritten images are 28*28
img_size_flat = img_h * img_w  # the total number of pixels

n_classes = 3
class_names = ['Sandal', 'Sneaker', 'Ankle boot']

# Load USPS data, in the encoder ,target is input
x_train, y_train = hw2_load_data(mode='train')
x_test, y_test = hw2_load_data(mode='test')
print("Size of:")
print("- Training-set-input:\t\t{}".format(x_train.shape))
print("- Training-set-target:\t\t{}".format(y_train.shape))
print("- test-set-input:\t\t{}".format(x_test.shape))
print("- test-set-target:\t\t{}".format(y_test.shape))

# h_90 = Cal_h90(x_train, img_size_flat)
# hL = Cal_hL(x_train,y_train, n_classes)
# print("The first alternative hidden layer size: h90=\t",h_90)
# print("The second alternative hidden layer size: hL=\t",hL)
# Hyper-parameters
h = 296  # Number of units in the first hidden layer, eg 101,185,1420,4260
epochs = 600  # Total number of training epochs
batch_size = 200  # Training batch size
learning_rate = 0.001
do_early_stopping = True  # if apply early stopping learning
D = np.sqrt(img_size_flat * h + h + h * n_classes + n_classes)

# Create graph

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')

fc1 = fc_layer(x, h, 'Hidden_layer', use_sigmoid=True)
output = fc_layer(fc1, n_classes, 'Output_layer', use_sigmoid=False)

# Define the loss function, optimizer, gradients and variable ,and accuracy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output), name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
# grad_and_var = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').compute_gradients(loss)
grad_and_var = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).compute_gradients(loss)
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
prediction = tf.argmax(output, axis=1, name='prediction')
true_label = tf.argmax(y, axis=1, name='true_label')
confusion_matrix = tf.confusion_matrix(labels=true_label, predictions=prediction, num_classes=n_classes)

train_ACRE = []
train_G = []
train_G2 = []
train_GW = []
train_accu = []
test_accu = []
if do_early_stopping == True:
    best_so_far_acc = 0
    best_so_far_epoch = 0
    best_so_far_confusion_test = []
    best_so_far_confusion_train = []
    Hidden = []

# Create the op for initializing all variables
init = tf.global_variables_initializer()

# Launch the graph (session)
with tf.Session() as sess:
    sess.run(init)
    global_step = 0
    # Number of training iterations in each epoch
    num_tr_iter = int(len(x_train) / batch_size)

    start_time = datetime.now()

    for epoch in range(epochs):
        x_train, y_train = randomize(x_train, y_train)
        for iteration in range(num_tr_iter):
            global_step += 1
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch = get_next_batch(x_train, y_train, start, end)

            # Run optimization op (backprop)
            feed_dict_batch = {x: x_batch, y: y_batch}
            sess.run(optimizer, feed_dict=feed_dict_batch)

        # after each epochs, record the accuracy
        feed_dict_train = {x: x_train, y: y_train}
        # after each BATCH, record indicators Acre,G,G2,GW for training set
        #  loss_batch, grad_and_var_batch= sess.run([loss, grad_and_var], feed_dict=feed_dict_train)
        #
        #  gradients = [x[0] for x in grad_and_var_batch]
        #  grad_norm = tf.global_norm(gradients).eval()
        #  weights = [x[1] for x in grad_and_var_batch]
        #  w_norm = tf.global_norm(weights).eval()
        #  GW_norm = grad_norm / w_norm
        # # G = grad_norm / learning_rate
        #  G2 = grad_norm / (learning_rate * D)
        #
        #  train_ACRE.append(loss_batch)
        # # train_G.append(G)
        #  train_G2.append(G2)
        #  train_GW.append(GW_norm)

        acc_train = sess.run(accuracy, feed_dict=feed_dict_train)
        train_accu.append(acc_train)

        feed_dict_test = {x: x_test, y: y_test}
        acc_test = sess.run(accuracy, feed_dict=feed_dict_test)
        test_accu.append(acc_test)
        print('---------------------------------------------------------')
        print("Epoch: {0}, training accuracy: {1:.01%}, test accuracy: {2:.01%}".
              format(epoch + 1, acc_train, acc_test))
        print('---------------------------------------------------------')

        if do_early_stopping == True:
            if acc_test > best_so_far_acc:
                best_so_far_acc = acc_test
                best_so_far_epoch = epoch
                best_so_far_confusion_train = sess.run(confusion_matrix, feed_dict=feed_dict_train)
                best_so_far_confusion_test = sess.run(confusion_matrix, feed_dict=feed_dict_test)
                Hidden = sess.run(fc1, feed_dict=feed_dict_train)

    time_elapsed = datetime.now() - start_time
    #  print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    #
    #  print('Early stopping:test accuracy is highest after', best_so_far_epoch ,'epoch. We chose the model that we had then.')
    #  print('The confusion matrix for training set:', best_so_far_confusion_train)
    #  print('The confusion matrix for test set:', best_so_far_confusion_test)
    #  print('The loss:', train_ACRE[best_so_far_epoch])
    # # print('The G:', train_G[best_so_far_epoch])
    #  print('The G2:', train_G2[best_so_far_epoch])
    #  print('The GW:', train_GW[best_so_far_epoch])
    #  print('The best accuracy:', best_so_far_acc)

    np.save('./data/train_ACRE', train_ACRE)
    # np.save('./data/train_G', train_G)
    np.save('./data/train_G2', train_G2)
    np.save('./data/train_GW', train_GW)
    np.save('./data/train_acc', train_accu)
    np.save('./data/test_acc', test_accu)

    # plt.plot(train_ACRE, 'g-',label= 'training')
    # plt.xlabel('epoch')
    # plt.ylabel('Batch Average Cross Entropy error')
    # plt.legend()
    # plt.show()

    # plt.plot(train_G, 'g-',label='training')
    # plt.xlabel('epoch')
    # plt.ylabel('Gradient norm/ learning rate')
    # plt.legend()
    # plt.show()

    # plt.plot(train_G2, 'g-', label='training')
    # plt.xlabel('epoch')
    # plt.ylabel('Gradient norm/ (learning rate * d)')
    # plt.legend()
    # plt.show()
    #
    # plt.plot(train_GW, 'g-',label='training')
    # plt.xlabel('epoch')
    # plt.ylabel('Gradient norm / Weight norm')
    # plt.legend()
    # plt.show()
    #
    # plt.plot(train_accu, 'g-',label= 'training')
    # plt.plot(test_accu, 'b--',label= 'test')
    # plt.plot([best_so_far_epoch], [best_so_far_acc], 'ro', label='early stopping')
    # plt.xlabel('epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()

    np.save('./data/Hidden', Hidden)
    print('type:', type(Hidden))
    print(Hidden.shape)
    pca = PCA(0.99)
    pca_result = pca.fit_transform(Hidden)
    h_99 = pca_result.shape[1]
    print(h_99)
    pca = PCA(h)
    pca_full = pca.fit(Hidden)
    plt.plot(pca_full.explained_variance_)
    plt.ylabel('Eigenvalues in Decreasing order')
    plt.show()

    plt.plot(np.cumsum(pca_full.explained_variance_ratio_), 'k')
    h99_dot = plt.plot([h_99], [0.99], 'ro', label='99% explained')
    plt.xlabel('# of components')
    plt.ylabel('Cumulative explained the sum of eigenvalues')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    plt.show()

    one_class = []
    for j in range(18000):
        if (y_train[j] == np.eye(3)[0]).all():
            one_class.append(Hidden[j])
    prof1 = np.asarray(one_class).mean(0)

    one_class = []
    for j in range(18000):
        if (y_train[j] == np.eye(3)[1]).all():
            one_class.append(Hidden[j])
    prof2 = np.asarray(one_class).mean(0)

    one_class = []
    for j in range(18000):
        if (y_train[j] == np.eye(3)[2]).all():
            one_class.append(Hidden[j])
    prof3 = np.asarray(one_class).mean(0)

    plt.plot(prof1, 'g-', label='prof1')
    plt.show()
    plt.plot(prof2, 'b-', label='prof2')
    plt.show()
    plt.plot(prof3, 'r-', label='prof3')
    plt.show()

    print(' the hidden neurons which achieve best DIFFERENTIATION between class C1 versus C2:',
          np.argmax(np.abs(prof1 - prof2)))
    plt.plot(np.abs(prof1 - prof2), 'g-', label='|prof1-prof2|')
    plt.show()
    print(' the hidden neurons which achieve best DIFFERENTIATION between class C1 versus C3:',
          np.argmax(np.abs(prof1 - prof3)))
    plt.plot(np.abs(prof1 - prof3), 'g-', label='|prof1-prof3|')
    plt.show()
    print(' the hidden neurons which achieve best DIFFERENTIATION between class C2 versus C3:',
          np.argmax(np.abs(prof2 - prof3)))
    plt.plot(np.abs(prof2 - prof3), 'g-', label='|prof2-prof3|')
    plt.show()
