import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.decomposition import PCA


def load_data(mode='train'):
    """
    Function to (download and) load the MNIST data
    :param mode: train or test
    :return: images and the corresponding labels
    """
    fashion_mnist = keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    pca = PCA(0.95)
    pca_result = pca.fit_transform(x_train)
    h_95 = pca_result.shape[1]
    print('h95=', h_95)
    pca = PCA(len(y_train))
    pca_full = pca.fit(x_train)
    plt.plot(pca_full.explained_variance_)
    plt.ylabel('Eigenvalues in Decreasing order')
    plt.show()

    if mode == 'train':
        x_train, y_train = reformat(x_train, y_train)
        return x_train, y_train
    elif mode == 'test':
        x_test, y_test = reformat(x_test, y_test)
    return x_test, y_test


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def reformat(x, y):
    """
    Reformats the data to the format acceptable for convolutional layers
    :param x: input array
    :param y: corresponding labels
    :return: reshaped input and labels
    """
    img_size, num_ch, num_class = 28, 1, 10
    dataset = x.reshape((-1, img_size, img_size, num_ch)).astype(np.float32)
    labels = (np.arange(num_class) == y[:, None]).astype(np.float32)
    return dataset, labels


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch

def plot_images(images, cls_true, cls_pred=None, title=None):
    """
    Create figure with 3x3 sub-plots.
    :param images: array of images to be plotted, (9, img_h*img_w)
    :param cls_true: corresponding true labels (9,)
    :param cls_pred: corresponding true labels (9,)
    """
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(np.squeeze(images[i]), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            ax_title = "True: {0}".format(cls_true[i])
        else:
            ax_title = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_title(ax_title)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        plt.suptitle(title, size=20)
    plt.show(block=False)


def plot_example_errors(images, cls_true, cls_pred, title=None):
    """
    Function for plotting examples of images that have been mis-classified
    :param images: array of all images, (#imgs, img_h*img_w)
    :param cls_true: corresponding true labels, (#imgs,)
    :param cls_pred: corresponding predicted labels, (#imgs,)
    """
    # Negate the boolean array.
    incorrect = np.logical_not(np.equal(cls_pred, cls_true))

    # Get the images from the test-set that have been
    # incorrectly classified.
    incorrect_images = images[incorrect]

    # Get the true and predicted classes for those images.
    cls_pred = cls_pred[incorrect]
    cls_true = cls_true[incorrect]

    # Plot the first 9 images.
    plot_images(images=incorrect_images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9],
                title=title)

# weight and bais wrappers
def weight_variable(shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W',
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)


def bias_variable(shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b',
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
        W = weight_variable(shape=[in_dim, num_units])
        tf.summary.histogram('weight', W)
        b = bias_variable(shape=[num_units])
        tf.summary.histogram('bias', b)
        layer = tf.matmul(x, W)
        layer += b
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer


def conv_layer(x, filter_size, num_filters, stride, name):
    """
    Create a 2D convolution layer
    :param x: input from previous layer
    :param filter_size: size of each filter
    :param num_filters: number of filters (or output feature maps)
    :param stride: filter stride
    :param name: layer name
    :return: The output array
    """
    with tf.variable_scope(name):
        num_in_channel = x.get_shape().as_list()[-1]
        shape = [filter_size, filter_size, num_in_channel, num_filters]
        W = weight_variable(shape=shape)
        tf.summary.histogram('weight', W)
        b = bias_variable(shape=[num_filters])
        tf.summary.histogram('bias', b)
        layer = tf.nn.conv2d(x, W,
                             strides=[1, stride, stride, 1],
                             padding="SAME")
        layer += b
        return tf.nn.relu(layer)


def flatten_layer(layer):
    """
    Flattens the output of the convolutional layer to be fed into fully-connected layer
    :param layer: input array
    :return: flattened array
    """
    with tf.variable_scope('Flatten_layer'):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat


def max_pool(x, ksize, stride, name):
    """
    Create a max pooling layer
    :param x: input to max-pooling layer
    :param ksize: size of the max-pooling filter
    :param stride: stride of the max-pooling filter
    :param name: layer name
    :return: The output array
    """
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding="SAME",
                          name=name)


# Data Dimensions
img_h = img_w = 28  # MNIST images are 28x28
img_size_flat = img_h * img_w  # 28x28=784, the total number of pixels
n_classes = 10  # Number of classes, one class per digit
n_channels = 1

x_train, y_train = load_data(mode='train')
x_test, y_test = load_data(mode='test')

print("Size of:")
print("- Training-set:\t\t{}".format(len(y_train)))
print("- Test-set:\t\t{}".format(len(y_test)))

pca = PCA(0.95)
pca_result = pca.fit_transform(x_train)
h_95 = pca_result.shape[1]
print('h95=',h_95)
pca = PCA(len(y_train))
pca_full = pca.fit(x_train)
plt.plot(pca_full.explained_variance_)
plt.ylabel('Eigenvalues in Decreasing order')
plt.show()
#
# plt.plot(np.cumsum(pca_full.explained_variance_ratio_), 'k')
# h95_dot = plt.plot([h_95], [0.95], 'ro', label='95% explained')
# plt.xlabel('# of components')
# plt.ylabel('Cumulative explained the sum of eigenvalues')
# plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
# plt.show()


# Hyper-parameters
lr_0 = 0.001  # The optimization initial learning rate
epochs = 50  # Total number of training epochs
batch_size = 500  # Training batch size
do_early_stopping = True  # if apply early stopping learning

train_lr = []
train_BACRE = []
train_BG = []
train_accu = []
test_accu = []
if do_early_stopping == True :
    best_so_far_acc = 0
    best_so_far_epoch = 0
    best_so_far_confusion_test = []
    best_so_far_confusion_train = []
#    Hidden = []

# Network Configuration
# 1st Convolutional Layer
filter_size1 = 5  # Convolution filters are 5 x 5 pixels.
num_filters1 = 16  # There are 16 of these filters.
stride1 = 1  # The stride of the sliding window

# 2nd Convolutional Layer
filter_size2 = 5  # Convolution filters are 5 x 5 pixels.
num_filters2 = 32  # There are 32 of these filters.
stride2 = 1  # The stride of the sliding window

# Fully-connected layer.
h1 = 151  # Number of neurons in hidden layer.
lr = lr_0
# Create the network graph
# Placeholders for inputs (x), outputs(y)
with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, shape=[None, img_h, img_w, n_channels], name='X')
    y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')

conv1 = conv_layer(x, filter_size1, num_filters1, stride1, name='conv1')
pool1 = max_pool(conv1, ksize=2, stride=2, name='pool1')
conv2 = conv_layer(pool1, filter_size2, num_filters2, stride2, name='conv2')
pool2 = max_pool(conv2, ksize=2, stride=2, name='pool2')
layer_flat = flatten_layer(pool2)
fc1 = fc_layer(layer_flat, h1, 'FC1', use_relu=True)
#output_logits = fc_layer(layer_flat, n_classes, 'OUT', use_relu=False)
output_logits = fc_layer(fc1, n_classes, 'OUT', use_relu=False)

# Define the loss function, optimizer, and accuracy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='Adam-op').minimize(loss)
grad_and_var = tf.train.GradientDescentOptimizer(learning_rate=lr).compute_gradients(loss)
correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')
true_label = tf.argmax(y, axis=1, name='true_label')
confusion_matrix = tf.confusion_matrix(labels=true_label, predictions=cls_prediction, num_classes=n_classes)

# Creating the op for initializing all variables
init = tf.global_variables_initializer()


# Launch the graph (session)
with tf.Session() as sess:
    sess.run(init)
    global_step = 0

    # Number of training iterations in each epoch
    num_tr_iter = int(len(y_train) / batch_size)
    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch + 1))
        x_train, y_train = randomize(x_train, y_train)
        lr = lr_0/(epoch+1)
        confusion_train = np.zeros((n_classes, n_classes))
        for iteration in range(num_tr_iter):
            global_step += 1
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch = get_next_batch(x_train, y_train, start, end)

            # Run optimization op (backprop)
            feed_dict_batch = {x: x_batch, y: y_batch}
            sess.run(optimizer, feed_dict=feed_dict_batch)
            confusion_batch = sess.run(confusion_matrix, feed_dict=feed_dict_batch)
            confusion_train = confusion_train + confusion_batch

 #           if (iteration+1) % display_freq == 0:
            # Calculate and display the batch loss and accuracy
            loss_batch, acc_batch, grad_and_var_batch = sess.run([loss, accuracy, grad_and_var],feed_dict=feed_dict_batch)
            gradients = [x[0] for x in grad_and_var_batch]
            grad_norm = tf.global_norm(gradients).eval()
            G = grad_norm
            #print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}". format(iteration+1, loss_batch, acc_batch))
            train_BACRE.append(loss_batch)
            train_BG.append(G)
            train_lr.append(lr)

        train_accu.append(acc_batch)
        # Test the network when training is done after each epoch
        feed_dict_test = {x: x_test, y: y_test}
        loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
        test_accu.append(acc_test)
        print('---------------------------------------------------------')
        print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
        print('---------------------------------------------------------')

        if do_early_stopping == True:
                if acc_test > best_so_far_acc:
                    best_so_far_acc = acc_test
                    best_so_far_epoch = epoch
                    best_so_far_confusion_train = confusion_train
                    best_so_far_confusion_test = sess.run(confusion_matrix, feed_dict=feed_dict_test)


print('Early stopping:test accuracy is highest after', best_so_far_epoch ,'epoch. We chose the model that we had then.')
print('The confusion matrix for training set:', best_so_far_confusion_train)
print('The confusion matrix for test set:', best_so_far_confusion_test)
print('The best accuracy:', best_so_far_acc)

plt.plot(train_BACRE, 'g-', label='training')
plt.xlabel('batch')
plt.ylabel('Average Cross Entropy error')
plt.legend()
plt.show()

plt.plot(train_BG, 'g-', label='training')
plt.xlabel('batch')
plt.ylabel('Gradient norm')
plt.legend()
plt.show()

plt.plot(train_lr, 'g-', label='training')
plt.xlabel('batch')
plt.ylabel('learning rate')
plt.legend()
plt.show()

plt.plot(train_accu, 'g-', label='training')
plt.plot(test_accu, 'b--', label='test')
plt.plot([best_so_far_epoch], [best_so_far_acc], 'ro', label='early stopping')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# np.save('./data/train_ACRE', train_ACRE)
# np.save('./data/train_acc', train_accu)
# np.save('./data/test_acc', test_accu)

# print('type:', type(Hidden))
# print(Hidden.shape)
# pca = PCA(0.95)
# pca_result = pca.fit_transform(Hidden)
# h_95 = pca_result.shape[1]
# print('h95=',h_95)
# pca = PCA(Hidden.shape[1])
# pca_full = pca.fit(Hidden)
# plt.plot(pca_full.explained_variance_)
# plt.ylabel('Eigenvalues in Decreasing order')
# plt.show()
#
# plt.plot(np.cumsum(pca_full.explained_variance_ratio_), 'k')
# h95_dot = plt.plot([h_95], [0.95], 'ro', label='95% explained')
# plt.xlabel('# of components')
# plt.ylabel('Cumulative explained the sum of eigenvalues')
# plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
# plt.show()