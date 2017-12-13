# Importing the libraries
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Importing the MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)

# Visualizing the data
plt.imshow(mnist.train.images[1].reshape(28, 28)) # visualizing one of the handwritten digits
plt.show()

class GAN():
    def __init__(self):
        self.alpha = 0.01
        
    def __lrelu(self, x): # leaky relu
        '''
        Leaky Relu:
        -----------
        alpha : leakiness
        for alpha < 1, tf.maximum(x, alpha * x)
        
        Note: tf.nn.leaky_relu is available on Tensorflow 1.4 onwards
        
        '''
        
        return tf.maximum(self.alpha * x, x)
    
    def generator(self, inp, reuse = None):
        # Input here is the random inoput at the start of the process
        with tf.variable_scope('gen', reuse = reuse):
            # first hidden layer
            # applying leaky-relu activation function in the hidden layers
            hidden1 = self.__lrelu(tf.layers.dense(inputs = inp, units = 128))
            # 2nd hidden layer
            hidden2 = self.__lrelu(tf.layers.dense(inputs = hidden1, units = 128)) 
            # output layer
            # applying hyperbolic tangent activation function in the output layer
            output = tf.layers.dense(hidden2, units = 784, activation = tf.nn.tanh)
            # since this is the output layer and we are generating a new image, the pixel size should be 784
            
            return output
    
    def discriminator(self, inp, reuse=None): 
        # This will tell if the output from generator is real or fake
        with tf.variable_scope('dis', reuse = reuse):
            # first hidden layer
            hidden1 = self.__lrelu(tf.layers.dense(inputs=inp, units = 128))
            # 2nd hidden layer
            hidden2 = self.__lrelu(tf.layers.dense(inputs=hidden1, units = 128))
            # output layer
            logits = tf.layers.dense(hidden2, units=1) # probability of being real or fake
            
            # returning the output before and after applying the activation function
            return tf.sigmoid(logits), logits        

# Placeholders for our actual images in order to creater a tensor graph
actual_image = tf.placeholder(tf.float32, shape = [None, 784]) # None is reserved for the actual batch size
# 784 is the number of pixels in our input data
noise = tf.placeholder(tf.float32, shape = [None, 100]) # random noise at the start of the process
# vector of 100 random points

# Generator
g = GAN()
G = g.generator(noise) # passing the noise thru the generator

# Discriminator
# Actual discriminator output(relevent to the actual images we have)
D_output_actual , D_logits_actual = g.discriminator(actual_image)
# Fake image that was generated from the Generator(with random initial noise)
D_output_fake, D_logits_fake = g.discriminator(G, reuse = True)
# reuse = 'True' to make sure the variables in the discriminator is reusable. Thats is why we had to create the variable scope

def loss_func(logit, label):
    # Loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = logit))
    # 'softmax' activation is used because of multiclass-classification problem
    
    return cross_entropy

# Calculating the losses
# Discriminator loss when its trained on actual data
D_real_loss = loss_func(D_logits_actual, tf.ones_like(D_output_actual)* (0.9))
# we want all the labels to be true when we are trainign on real data. Hence using 'ones_like'
# 0.9 is used for smoothing
# Discriminator loss when its trained on fake data
D_fake_loss = loss_func(D_logits_fake, tf.zeros_like(D_logits_fake))
# we want all the labels to be false when we are trainign on fake data data. Hence using 'zeros_like'

D_loss = D_real_loss + D_fake_loss  # total discriminator loss
G_loss = loss_func(D_logits_fake, tf.ones_like(D_logits_fake)) # total generator loss

tvars = tf.trainable_variables()

# distinguishing the generator and discriminator variables based on variable scope
g_vars = [var for var in tvars if 'gen' in var.name]
d_vars = [var for var in tvars if 'dis' in var.name]

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
# an algorithm to find optimal set of weights for NN by varying the weights during stichastic gradient descent process

# Training the models
G_trainer = optimizer.minimize(G_loss, var_list = g_vars)
D_trainer = optimizer.minimize(D_loss, var_list = d_vars)

# Initializing the global variables
init = tf.global_variables_initializer()

# Executing the graph
batch_size = 100
epochs = 500
saver = tf.train.Saver(var_list = g_vars) # to save the trained model for future reference

# Save a sample per epoch
samples = []

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    
    # Recall an epoch is an entire run through the training data
    for e in range(epochs):
        num_batches = mnist.train.num_examples // batch_size # to calculate number of batches required
        
        for _ in range(num_batches):            
            # Grab batch of images
            batch = mnist.train.next_batch(batch_size)
            
            # Get images, reshape and rescale to pass to the discriminator
            batch_images = batch[0].reshape((batch_size, 784))
            batch_images = batch_images*2 - 1
            
            # noise (random latent noise data for Generator)
            # -1 to 1 because of tanh activation
            batch_noise = np.random.uniform(-1, 1, size = (batch_size, 100))
            
            # Run optimizers, no need to save outputs, we won't use them
            _ = sess.run(D_trainer, feed_dict={actual_image: batch_images, noise: batch_noise})
            _ = sess.run(G_trainer, feed_dict={noise: batch_noise})
        
            
        print("Currently on Epoch {} of {} total...".format(e + 1, epochs))
        saver.save(sess, './models/gan.ckpt') # saving the trained model

# Generating a new digit using the trained GAN
# saver = tf.train.Saver(var_list = g_vars)
new_samples = []
with tf.Session() as sess:    
    # Restoring the trained model
    saver.restore(sess, './models/gan.ckpt'
    
    g = GAN()    
    for _ in range(5):
        sample_noise = np.random.uniform(-1, 1, size = (1,100))
        gen_sample = sess.run(g.generator(noise, reuse = True), feed_dict = {noise: sample_noise})        
        new_samples.append(gen_sample)

# Visualizing the newly generated image
plt.imshow(new_samples[0].reshape(28,28), cmap='Greys')
plt.show()