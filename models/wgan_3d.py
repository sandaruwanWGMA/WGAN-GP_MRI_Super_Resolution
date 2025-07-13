from os import path
from numpy import squeeze
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv3DTranspose, Reshape, LeakyReLU, Flatten, Conv3D
from tensorflow.keras.optimizers import Adam
from utils.training import DataGenerator, save_models
from utils.imutils import process_mnc
from utils.minc_viewer import Viewer


class DCWGAN(Model):

        """
        Keras Model that takes Generator and Critic Graphs as input
        and trains them adversarially
        """

        def __init__(self,
                     generator,
                     critic,
                     g_opt=Adam(lr=5e-5, beta_1=0.5),
                     c_opt=Adam(lr=5e-5, beta_1=0.5),
                     gradient_penalty_weight=10):

                super(DCWGAN, self).__init__()

                self.z_units = generator.input.shape[1]  # Size of Gen Input Latent Space
                self.gradient_penalty_weight = gradient_penalty_weight  # interpolated image loss weight

                # Generator and Critic Optimizers
                self.generator_optimizer = g_opt
                self.critic_optimizer = c_opt

                # Build Generator
                self.generator = generator

                # Build Discriminator
                self.critic = critic

        def set_trainable(self, model, val):

                """
                Switchs a model to trainable
                :param model: keras model
                :param val: bool, trainable / not trainable
                """

                model.trainable = val
                for layer in model.layers:
                        layer.trainable = val

        def gradient_penalty_loss(self, real_data, generated_data):

                """
                Computes Interpolated Loss of WGAN training
                :param real_data: batch of real data to be interpolated
                :param generated_data: batch of generated data to be interpolated
                :return: interpolated loss
                """

                # Get Number of instances of real data == Batch size
                batch_size = real_data.shape[0]
                alpha = tf.random.uniform((batch_size, 1, 1, 1, 1))  # alpha shape matches 3d data
                inter_data = (alpha * real_data) + ((1 - alpha) * generated_data)

                with tf.GradientTape() as g:
                        g.watch(inter_data)
                        logits_inter_data = self.critic(inter_data)

                gradients = g.gradient(logits_inter_data, inter_data)

                # compute the euclidean norm by squaring ...
                gradients_sqr = tf.square(gradients)
                #   ... summing over the rows ...
                gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=0)
                #   ... and sqrt
                gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
                # compute lambda * (1 - ||grad||)^2 still for each single sample
                gradient_penalty = tf.square(1 - gradient_l2_norm)
                # return the mean as loss over all the batch samples
                return tf.reduce_mean(gradient_penalty)

        def generate_data(self, n):
                """
                Creates a batch of fake data
                :param n: int, size of batch
                :return: batch of generated data
                """

                # Create random z vectors to feed generator
                random_z_vectors = tf.random.normal(shape=(n, self.z_units))
                generated_data = self.generator(random_z_vectors)
                return generated_data

        def compute_generator_loss(self, batch_size):

                """
                passes through the network and computes generator loss
                :param batch_size: int, size of training batch
                :return generator loss
                """

                # Get fake data to feed generator
                generated_data = self.generate_data(batch_size)
                # feed generator and get logits
                logits_generated_data = self.critic(generated_data)
                # losses of fake with label "1"
                generator_loss = tf.reduce_mean(logits_generated_data)

                return generator_loss

        def compute_generator_gradients(self, batch_size):

                """
                Compute Gradients to update generator
                :param batch_size: int, size of training batch
                :return: generator gradients and training loss
                """

                with tf.GradientTape() as gen_tape:
                        gen_loss = self.compute_generator_loss(batch_size)
                # compute gradients
                gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                return gen_gradients, gen_loss

        def apply_generator_gradients(self, gradients):

                """
                Apply calculated gradients to update generator
                :param gradients: generator gradients
                """

                # Optimizer applies gradients on trainable weights
                self.generator_optimizer.apply_gradients(
                        zip(gradients, self.generator.trainable_variables)
                )

        def compute_critic_loss(self, real_data):

                """
                passes through the network and computes loss
                :param real_data: batch of original images
                :return critic loss
                """

                # Get Number of instances of real data == Batch size
                batch_size = real_data.shape[0]

                # Convert numpy data data to tensor
                real_data = tf.convert_to_tensor(real_data, dtype=tf.float32)

                # Create random z vectors to feed generator
                random_z_vectors = tf.random.normal(shape=(batch_size, self.z_units))
                generated_data = self.generator(random_z_vectors)

                # discriminate x and x_gen
                logits_real_data = self.critic(real_data)
                logits_generated_data = self.critic(generated_data)

                # gradient penalty
                critic_regularizer = self.gradient_penalty_loss(real_data, generated_data)

                # losses
                critic_loss = (
                                tf.reduce_mean(logits_real_data)
                                - tf.reduce_mean(logits_generated_data)
                                + critic_regularizer
                                * self.gradient_penalty_weight
                )

                return critic_loss

        def compute_critic_gradients(self, real_data):
                """
                Compute Gradients to update critic
                :param real_data: batch of original images
                :return: critic gradients and critic loss
                """

                # pass through network
                with tf.GradientTape() as critic_tape:
                        critic_loss = self.compute_critic_loss(real_data)

                # compute gradients
                critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)

                return critic_gradients, critic_loss

        def apply_critic_gradients(self, gradients):
                """
                Apply calculated gradients to update critic
                :param gradients: critic gradients
                """

                # Optimizer applies gradients on trainable weights
                self.critic_optimizer.apply_gradients(
                        zip(gradients, self.critic.trainable_variables)
                )

        @tf.function
        def train_generator(self, batch_size):

                """
                Pass forward and apply gradients to generator
                :param batch_size: int, size of training batch
                :return: loss computed in training
                """

                gen_gradients, gen_loss = self.compute_generator_gradients(batch_size)
                self.apply_generator_gradients(gen_gradients)
                return gen_loss

        @tf.function
        def train_critic(self, real_data):

                """
                Pass forward and apply gradients to critic
                :param real_data: batch of original images
                :return: loss computed in training
                """

                critic_gradients, critic_loss = self.compute_critic_gradients(real_data)
                self.apply_critic_gradients(critic_gradients)
                return critic_loss


if __name__ == '__main__':

        # Initialize NN weights
        init = tf.initializers.RandomNormal(stddev=0.02)

        # Create generator Graph
        generator = Sequential(
                [       # Input
                        Input(shape=(10,), name='z_input'),
                        # 1st Deconvolution
                        Dense(4 * 4 * 4 * 128),
                        LeakyReLU(alpha=0.2, name='lrelu_1'),
                        Reshape((4, 4, 4, 128), name="conv_1"),
                        # 2nd Deconvolution
                        Conv3DTranspose(
                                filters=64,
                                kernel_size=5,
                                strides=(4, 4, 4),
                                kernel_initializer=init,
                                use_bias=True,
                                padding="same",
                                name="conv_2"),
                        LeakyReLU(alpha=0.2, name='lrelu_2'),
                        # 3rd Deconvolution
                        Conv3DTranspose(
                                filters=32,
                                kernel_size=5,
                                strides=(4, 4, 4),
                                kernel_initializer=init,
                                use_bias=True,
                                padding='same',
                                name="conv_3"),
                        LeakyReLU(alpha=0.2, name='lrelu_3'),
                        # Output
                        Conv3DTranspose(
                                filters=1,
                                kernel_size=5,
                                strides=(3, 4, 3),
                                kernel_initializer=init,
                                activation='linear',
                                use_bias=True,
                                padding='same',
                                name='output')
                ],
                name="generator",
        )

        # Create critic Graph
        critic = Sequential(
                [       # Input
                        Input(shape=(192, 256, 192, 1), name='input'),
                        # 1st Convolution
                        Conv3D(
                                filters=32,
                                kernel_size=5,
                                strides=(3, 4, 3),
                                kernel_initializer=init,
                                use_bias=True,
                                padding="same",
                                name="conv_1"),
                        LeakyReLU(alpha=0.2, name='lrelu_1'),
                        # 2nd Convolution
                        Conv3D(
                                filters=64,
                                kernel_size=5,
                                strides=(4, 4, 4),
                                kernel_initializer=init,
                                use_bias=True,
                                padding='same',
                                name="conv_2"),
                        LeakyReLU(alpha=0.2, name='lrelu_2'),
                        # 3rd Convolution
                        Conv3D(
                                filters=128,
                                kernel_size=5,
                                strides=(4, 4, 4),
                                kernel_initializer=init,
                                use_bias=True,
                                padding='same',
                                name='conv_3'),
                        LeakyReLU(alpha=0.2, name='lrelu_3'),
                        # Output
                        Flatten(),
                        Dense(1, activation=None, name='output', kernel_initializer=init)
                ],
                name="critic",
        )

        # Create adversarial graph
        wgan = DCWGAN(generator=generator, critic=critic)

        # Path to mnc files
        IMAGES_PATH = path.join('..', 'resources', 'mri')
        data_generator = DataGenerator(IMAGES_PATH, process_mnc)  # Collects real images for training

        # --------------------
        #  PARAMETER INIT
        # --------------------

        batch_size = 4  # Samples every epoch
        n_epochs = 10  # Training Epochs
        plot_interval = 10  # Every plot_interval create a graph with real and generated data distribution
        c_loops = 5  # number of loops to train critic every epoch
        z_control = tf.random.normal((batch_size, wgan.z_units))  # Vector to feed gen and control training evolution

        # --------------------
        #  TENSORBOARD SETUP
        # --------------------
        generator_train_loss = tf.keras.metrics.Mean('generator_train_loss', dtype=tf.float32)
        critic_train_loss = tf.keras.metrics.Mean('critic_train_loss', dtype=tf.float32)

        for epoch in range(n_epochs):

                # --------------------
                #     TRAINING
                # --------------------

                # Train Critic
                for _ in range(c_loops):

                        batch = data_generator.get_batch(batch_size)
                        c_loss = wgan.train_critic(batch)

                # Train Generator
                g_loss = wgan.train_generator(batch_size)  # Train our model on real distribution points

                # -----------------------
                #  TENSORBOARD TRACKING
                # ------------------------

                # Save generator and critic losses
                generator_train_loss(g_loss)
                critic_train_loss(c_loss)

                # track data through console
                template = 'Epoch {}, Gen Loss: {}, Dis Loss {}'
                print(template.format(epoch + 1,
                                      generator_train_loss.result(),
                                      critic_train_loss.result()))

        # save models after training
        save_models(critic, generator, None, "dc_wgan")

        # generate fake sample to visualize
        fake = generator(z_control)[0]
        fake = squeeze(fake, 3)
        print(fake.min(), fake.max())
        Viewer(fake)
