from os import path
from numpy import squeeze
from time import strftime, localtime, time
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from utils.super_res_data_generator import SuperResDataGenerator
from utils.minc_viewer import Viewer
from models.wgan_3d_low import critic, generator
from models.wgan_3d import DCWGAN


# Function to save loss plots
def save_loss_plots(g_losses, c_losses, save_path):
    """
    Save plot of generator and critic losses
    
    Args:
        g_losses (list): Generator losses
        c_losses (list): Critic losses
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 5))
    
    # Plot generator loss
    plt.subplot(1, 2, 1)
    plt.plot(g_losses)
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot critic loss
    plt.subplot(1, 2, 2)
    plt.plot(c_losses)
    plt.title('Critic Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
        
        # load weights
        critic.load_weights("/kaggle/input/model/tensorflow2/default/1/critic_dc_wgan_low.h5")
        generator.load_weights("/kaggle/input/model/tensorflow2/default/1/generator_dc_wgan_low.h5")
        
        # Create adversarial graph
        gen_opt = Adam()
        critic_opt = Adam()
        wgan = DCWGAN(generator=generator, critic=critic, g_opt=gen_opt, c_opt=critic_opt)

        # Path to MRI files
        HIGH_RES_PATH = "/kaggle/input/high-res-and-low-res-mri/Refined-MRI-dataset/High-Res"
        LOW_RES_PATH = "/kaggle/input/high-res-and-low-res-mri/Refined-MRI-dataset/Low-Res"
        
        # Create super-resolution data generator
        data_generator = SuperResDataGenerator(
            high_res_path=HIGH_RES_PATH, 
            low_res_path=LOW_RES_PATH,
            patch_size=16,  # Match the input size of the critic model (16x16x16)
            normalize=True  # Normalize images to [-1, 1]
        )

        # --------------------
        #  PARAMETER INIT
        # --------------------

        batch_size = 4  # Samples every epoch
        n_epochs = 200  # Training Epochs (changed to 200)
        plot_interval = 20  # Every plot_interval create a graph with real and generated data distribution
        c_loops = 5  # number of loops to train critic every epoch
        z_control = tf.random.normal((1, wgan.z_units))  # Vector to feed gen and control training evolution

        # --------------------
        #  TENSORBOARD SETUP
        # --------------------

        generator_train_loss = tf.keras.metrics.Mean('generator_train_loss', dtype=tf.float32)
        critic_train_loss = tf.keras.metrics.Mean('critic_train_loss', dtype=tf.float32)

        # Set Tensorboard Directory to track data
        time_now = strftime("%d-%b-%H%M", localtime())
        log_dir = path.join('logs', 'dc_wgan_low', time_now)

        # Create plots directory
        plots_dir = path.join('/kaggle/working', 'loss_plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Start model data tracing (logs)
        summary_writer = tf.summary.create_file_writer(log_dir)
        tf.summary.trace_on()

        # Lists to store losses for plotting
        g_loss_list, c_loss_list = [], []

        print("START TRAINING")
        for epoch in range(n_epochs):

                start_time = time()

                # --------------------
                #     TRAINING
                # --------------------

                # Get a batch of high-res and low-res patches
                hr_batch, lr_batch = data_generator.get_batch(batch_size)

                # Train Critic using high-res images as real data
                for _ in tf.range(c_loops):
                        c_loss = wgan.train_critic(hr_batch)  # Train and get critic loss

                # Train Generator
                g_loss = wgan.train_generator(batch_size)  # Train our model on real distribution points

                # -----------------------
                #  TENSORBOARD TRACKING
                # ------------------------

                # Save generator and critic losses
                generator_train_loss(g_loss)
                critic_train_loss(c_loss)
                
                # Store losses for plotting
                g_loss_list.append(float(g_loss))
                c_loss_list.append(float(c_loss))

                # track data through console
                template = 'Epoch {}/{}, Gen Loss: {:.4f}, Dis Loss {:.4f}'
                print(template.format(epoch + 1, n_epochs,
                                      generator_train_loss.result(),
                                      critic_train_loss.result()))

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
                
                # Save loss plots periodically
                if (epoch + 1) % plot_interval == 0 or epoch == n_epochs - 1:
                    loss_plot_path = path.join(plots_dir, f'losses_epoch_{epoch+1}.png')
                    save_loss_plots(g_loss_list, c_loss_list, loss_plot_path)
                    print(f"Saved loss plot at epoch {epoch+1}")

                print("Epoch took {} seconds".format(round(time() - start_time, 2)))

        # ---------------
        #  SAVE WEIGHTS
        # ---------------

        # save models after training
        folder_path = path.join('models', 'weights', 'dc_wgan_low')
        os.makedirs(folder_path, exist_ok=True)  # Create directory if it doesn't exist
        generator_name = 'generator_dc_wgan_low_sr.h5'
        critic_name = 'critic_dc_wgan_low_sr.h5'
        generator_path = path.join(folder_path, generator_name)
        critic_path = path.join(folder_path, critic_name)
        generator.save(generator_path)
        critic.save(critic_path)

        # -----------------------
        #  INSPECT GENERATED MRI
        # -----------------------

        # generate fake sample to visualize
        fake = generator(z_control)[0]  # Generate fake MRI
        fake = squeeze(fake, 3)  # Convert (16x16x16x1) into (16x16x16)
        print("Min and Max values of fake image are:", fake.min(), fake.max())
        Viewer(fake)  # Inspect Generated MRI
        
        # Save final loss plot
        final_loss_plot_path = path.join(plots_dir, 'final_losses.png')
        save_loss_plots(g_loss_list, c_loss_list, final_loss_plot_path)
        print(f"Final loss plot saved to {final_loss_plot_path}")
