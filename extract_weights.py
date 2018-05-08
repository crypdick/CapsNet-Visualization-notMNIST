from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tensorflow as tf
import numpy as np
import os

PATH_TO_CKPT = '/tmp'
MODEL_VERSION = 'keras_model.ckpt'
PATH_TO_MODEL = os.path.join(PATH_TO_CKPT, MODEL_VERSION)

PATH_TO_WEIGHTS = 'numpy_weights'
PATH_TO_CONV1 = os.path.join(PATH_TO_WEIGHTS, 'conv1.weights.npz')
PATH_TO_CONV1_BIAS = os.path.join(PATH_TO_WEIGHTS, 'conv1.bias.npz')
PATH_TO_PRIMARY_CAPS = os.path.join(PATH_TO_WEIGHTS, 'primary_caps.weights.npz')
PATH_TO_PRIMARY_CAPS_BIAS = os.path.join(PATH_TO_WEIGHTS, 'primary_caps.bias.npz')
PATH_TO_DIGIT_CAPS = os.path.join(PATH_TO_WEIGHTS, 'digit_caps.weights.npz')
PATH_TO_FULLY_CONNECTED1 = os.path.join(PATH_TO_WEIGHTS, 'fully_connected1.weights.npz')
PATH_TO_FULLY_CONNECTED2 = os.path.join(PATH_TO_WEIGHTS, 'fully_connected2.weights.npz')
PATH_TO_FULLY_CONNECTED3 = os.path.join(PATH_TO_WEIGHTS, 'fully_connected3.weights.npz')
PATH_TO_FULLY_CONNECTED1_BIAS = os.path.join(PATH_TO_WEIGHTS, 'fully_connected1.bias.npz')
PATH_TO_FULLY_CONNECTED2_BIAS = os.path.join(PATH_TO_WEIGHTS, 'fully_connected2.bias.npz')
PATH_TO_FULLY_CONNECTED3_BIAS = os.path.join(PATH_TO_WEIGHTS, 'fully_connected3.bias.npz')

print_tensors_in_checkpoint_file(file_name=PATH_TO_MODEL, tensor_name='', all_tensors=False, all_tensor_names=False)

sess = tf.Session()
new_saver = tf.train.import_meta_graph(PATH_TO_MODEL + '.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(PATH_TO_CKPT))

# conv1/kernel (DT_FLOAT) [9,9,1,256]
weights = sess.run('conv1/kernel:0')
with open(PATH_TO_CONV1, 'wb') as outfile:
    np.save(outfile, weights)

# conv1/kernel (DT_FLOAT) [9,9,1,256]
bias = sess.run('conv1/kernel 0')
with open(PATH_TO_CONV1_BIAS, 'wb') as outfile:
    np.save(outfile, bias)

# primarycap_conv2d/kernel (DT_FLOAT) [9,9,256,256]
weights = sess.run('primarycap_conv2d/kernel 0')
with open(PATH_TO_PRIMARY_CAPS, 'wb') as outfile:
    np.save(outfile, weights)

# primarycap_conv2d/bias (DT_FLOAT) [256]
bias = sess.run('primarycap_conv2d/bias 0')
with open(PATH_TO_PRIMARY_CAPS_BIAS, 'wb') as outfile:
    np.save(outfile, bias)

# DigitCaps_layer/routing/Weight (DT_FLOAT) [1,1152,10,8,16]
# new Keras: digitcaps/W (DT_FLOAT) [10,1152,16,8]
weights = sess.run('digitcaps/W:0')
with open(PATH_TO_DIGIT_CAPS, 'wb') as outfile:
    np.save(outfile, weights)

# dense_1/kernel (DT_FLOAT) [160,512]
weights = sess.run('dense_1/kernel:0')
with open(PATH_TO_FULLY_CONNECTED1, 'wb') as outfile:
    np.save(outfile, weights)

# dense_1/bias (DT_FLOAT) [512]
bias = sess.run('dense_1/bias0')
with open(PATH_TO_FULLY_CONNECTED1_BIAS, 'wb') as outfile:
    np.save(outfile, bias)

# dense_1/kernel (DT_FLOAT) [160,512]
weights = sess.run('dense_1/kernel:0')
with open(PATH_TO_FULLY_CONNECTED2, 'wb') as outfile:
    np.save(outfile, weights)


# dense_2/bias (DT_FLOAT) [1024]
bias = sess.run('dense_2/bias:0')
with open(PATH_TO_FULLY_CONNECTED2_BIAS, 'wb') as outfile:
    np.save(outfile, bias)


# dense_2/kernel (DT_FLOAT) [512,1024]
weights = sess.run('dense_2/kernel:0')
with open(PATH_TO_FULLY_CONNECTED3, 'wb') as outfile:
    np.save(outfile, weights)


# dense_3/bias (DT_FLOAT) [784]
bias = sess.run('dense_3/bias:0')
with open(PATH_TO_FULLY_CONNECTED3_BIAS, 'wb') as outfile:
    np.save(outfile, bias)
