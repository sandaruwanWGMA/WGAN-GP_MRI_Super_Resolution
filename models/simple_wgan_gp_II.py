from os import path
from time import strftime, localtime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from models.simple_gan import build_simple_nn
from utils.simple_data_generator import get_quadratic_data_sample, plot_quadratic_data
from utils.plot_animator import PlotAnimator
import numpy as np
import matplotlib.pyplot as plt


class Simple_WGAN(Model):

        def __init__(self, layers_n, hidden_units, gout_units=2, z_units=10, g_lr=5e-5, c_lr=5e-5, gradient_penalty_weight = 10):
                super(Simple_WGAN, self).__init__()


                # Set Parameters for data
                self.z_units = z_units  # Generator Input units
                self.layers_n = layers_n  # Number of hidden layers
                self.g_out_dim = gout_units  # Generator Output Units == Discriminator Input Units
                self.hidden_units = hidden_units  # units on hidden layers
                self.g_lr = g_lr  # generator learning rate
                self.c_lr = c_lr  # discriminator learning rate
                self.gradient_penalty_weight = gradient_penalty_weight  # interpolated image loss weight
                # critic_loops = 5  # number of iterations to train discriminator on every epoch

                self.generator_optimizer = Adam(lr=self.g_lr, beta_1=0.5)
                self.critic_optimizer = Adam(lr=self.c_lr, beta_1=0.5)

                # Build Generator
                self.generator = build_simple_nn(
                        input_units=self.z_units,
                        output_units=self.g_out_dim,
                        layer_number=self.layers_n,
                        units_per_layer=self.hidden_units,
                        activation='linear',
                        model_name='generator'
                )

                # Build Discriminator
                self.critic = build_simple_nn(
                        input_units=self.g_out_dim,
                        output_units=1,
                        layer_number=self.layers_n,
                        units_per_layer=self.hidden_units,
                        activation=None,
                        model_name='critic'
                )

        def set_trainable(self, model, val):
                model.trainable = val
                for layer in model.layers:
                        layer.trainable = val

        def gradient_penalty_loss(self, real_data, generated_data):

                # Get Number of instances of real data == Batch size
                batch_size = real_data.shape[0]
                alpha = tf.random.uniform((batch_size, 1))
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

                # Create random z vectors to feed generator
                random_z_vectors = tf.random.normal(shape=(n, self.z_units))
                generated_data = self.generator(random_z_vectors)
                return generated_data

        def compute_generator_loss(self, batch_size):

                # Get fake data to feed generator
                generated_data = self.generate_data(batch_size)
                # feed generator and get logits
                logits_generated_data = self.critic(generated_data)
                # losses of fake with label "1"
                generator_loss = tf.reduce_mean(logits_generated_data)

                return generator_loss

        def compute_generator_gradients(self, batch_size):

                with tf.GradientTape() as gen_tape:
                        gen_loss = self.compute_generator_loss(batch_size)
                # compute gradients
                gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                return gen_gradients

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
                Compute Gradients to update generator and discriminator
                :param real_data:
                :return:
                """

                # pass through network
                with tf.GradientTape() as critic_tape:
                        critic_loss = self.compute_critic_loss(real_data)

                # compute gradients
                critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)

                return critic_gradients

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
                gen_gradients = self.compute_generator_gradients(batch_size)
                self.apply_generator_gradients(gen_gradients)

        @tf.function
        def train_critic(self, real_data):
                critic_gradients = self.compute_critic_gradients(real_data)
                self.apply_critic_gradients(critic_gradients)



if __name__ == '__main__':

        # --------------------
        #  PARAMETER INIT
        # --------------------

        wgan = Simple_WGAN(
                layers_n=4,
                hidden_units=16,
        )
        batch_size = 64  # Samples every epoch
        n_epochs = 1001  # Training Epochs
        plot_interval = 10  # Every plot_interval create a graph with real and generated data distribution
        c_loops = 5  # number of loops to train critic every epoch

        # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8.5, 4))
        # fig.suptitle("Simple Wasserstein GP GAN Training Evolution", size=20)
        # animator = PlotAnimator(fig)

        # --------------------
        #  TENSORBOARD SETUP
        # --------------------
        generator_train_loss = tf.keras.metrics.Mean('generator_train_loss', dtype=tf.float32)
        critic_train_loss = tf.keras.metrics.Mean('critic_train_loss', dtype=tf.float32)
        # Set Tensorboard Directory to track data
        time = strftime("%d-%b-%H%M", localtime())
        log_dir = path.join('..', 'logs', 'simple_wgan', time)
        # Start model data tracing (logs)
        summary_writer = tf.summary.create_file_writer(log_dir)
        # tf.summary.trace_on(graph=True, profiler=True)

        z_control = tf.random.normal((batch_size, wgan.z_units))
        real_distribution = get_quadratic_data_sample(batch_size)

        for epoch in range(n_epochs):

                # --------------------
                #     TRAINING
                # --------------------

                # Train Critic
                for _ in range(c_loops):
                        training_data = get_quadratic_data_sample(batch_size)  # Get points from real distribution
                        wgan.train_critic(training_data)

                # Train Generator
                wgan.train_generator(batch_size)  # Train our model on real distribution points
                c_loss= wgan.compute_critic_loss(training_data)  # Get batch loss to track data
                g_loss = wgan.compute_generator_loss(batch_size)

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

                if epoch % plot_interval == 0:

                        # -----------------------
                        #  TENSORBOARD PLOTTING
                        # ------------------------

                        with summary_writer.as_default():

                                # Write losses
                                tf.summary.scalar('Generator Loss',
                                                  generator_train_loss.result(),
                                                  step=epoch)

                                tf.summary.scalar('Discriminator Loss',
                                                  critic_train_loss.result(),
                                                  step=epoch)

                        # animator.update_distribution_plot(
                        #         ax1, training_data, fake.numpy(), epoch)
                        # animator.update_training_plot(
                        #         ax2,
                        #         g_loss.numpy(),
                        #         c_loss.numpy(),
                        #         epoch)
                        #
                        # animator.epoch_end()

        # animator.close(3)
        # Plot
        fake = wgan.generator(z_control)
        plot_quadratic_data(real_distribution, show=False)
        plot_quadratic_data(fake.numpy())
