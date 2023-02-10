import numpy as np
import os
import datetime
import time

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

from network import *

#--------------------------------------------------
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#--------------------------------------------------

import easydict

FLAGS = easydict.EasyDict({
                        #    "mode": "Train",
                           "mode": "Test",
                           "train_labels": "/mnt/ssd/data/PAL/mask/train1_ori.txt",
                           "train_images": 435,

                           "test_labels": "/mnt/ssd/data/PAL/mask/train1_mask.txt",
                           "test_images": 435,

                           "test_dir": "./train_100/",

                           "samples": "./samples/",

                           "logs": "./logs/gradient_tape/",
                           "checkpoint_dir": "/mnt/disk2/experiment3/PAL/prop/checkpoint",

                           "batch_size": 4,
                           "epochs": 400,

                        #    "lr": 0.000005,
                           "lr": 0.00001,
                        #    "lr": 0.0001,
                        #    "lr": 0.0002,

                           "buffer_size": 500,
                           "gen_lambda": 1,
                        #    "l2_lambda": 100,
                        #    "edge_lambda": 100,
                           "l2_lambda": 10,
                           "edge_lambda": 10,
                           "image_size": 256,
                           "channel": 3
                           })

# util function -- start
def load_label_txt(label_file):
    target_images = np.loadtxt(label_file, dtype='<U100', skiprows=0, usecols=0)

    input_images = np.char.replace(target_images, '_ori', '_mask')
    return input_images, target_images

def load(input_image, target_image):
    input_image = tf.io.read_file(input_image)
    input_image = tf.image.decode_jpeg(input_image)
    input_image = tf.cast(input_image, tf.float32)

    target_image = tf.io.read_file(target_image)
    target_image = tf.image.decode_jpeg(target_image)
    target_image = tf.cast(target_image, tf.float32)

    return input_image, target_image

def resize(input_image, target_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    target_image = tf.image.resize(target_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, target_image

def random_crop(input_image, target_image):
    stacked_image = tf.stack([input_image, target_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, FLAGS.image_size, FLAGS.image_size, FLAGS.channel])

    return cropped_image[0], cropped_image[1]

# normalizing the images to [-1, 1]
def normalize(input_image, target_image):
    input_image = (input_image / 127.5) - 1
    target_image = (target_image / 127.5) - 1
    # input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    # target_image = tf.image.convert_image_dtype(target_image, tf.float32)

    return input_image, target_image

@tf.function()
def random_jitter(input_image, target_image):
    if tf.random.uniform(()) > 0.5:
        # resizing to 286 x 286 x 3
        input_image, target_image = resize(input_image, target_image, FLAGS.image_size + 30, FLAGS.image_size + 30)

        # randomly cropping to 256 x 256 x 3
        input_image, target_image = random_crop(input_image, target_image)

    if tf.random.uniform(()) > 0.5:
        # random flipping
        input_image = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)

    return input_image, target_image

def load_image(input_image, target_image):
    input_image, target_image = load(input_image, target_image)

    if FLAGS.mode == 'Train':
        input_image, target_image = random_jitter(input_image, target_image)
    
    input_image, target_image = resize(input_image, target_image,
                                        FLAGS.image_size, FLAGS.image_size)
    input_image, target_image = normalize(input_image, target_image)

    return input_image, target_image
# util function -- end

# loss & optimizer & metrics -- start
def CharbonnierLoss(target, predict):
    diff = target - predict
    loss = tf.reduce_mean(tf.math.sqrt((diff * diff) + (0.001 * 0.001)))
    return loss

def EdgeLoss(target, predict):
    k = tf.constant([[0.05, 0.25, 0.4, 0.25, 0.05]], tf.float32)
    k_t = tf.linalg.matrix_transpose(k)
    kernel = tf.linalg.matmul(k_t, k)
    kernel = tf.expand_dims(kernel, -1)
    kernel = tf.expand_dims(kernel, -1)
    kernel = tf.tile(kernel, [1, 1, 3, 1])

    target_img = tf.pad(target, [[0,0], [2,2], [2,2], [0,0]], "REFLECT")
    target_filtered = tf.nn.depthwise_conv2d(target_img, kernel, strides=[1,1,1,1], padding="VALID")
    target_down = target_filtered[:, ::2, ::2, :]
    target_new_filter = tf.zeros_like(target_filtered).numpy()
    target_new_filter[:, ::2, ::2, :] = target_down*4
    target_new_filter = tf.pad(target_new_filter, [[0,0], [2,2], [2,2], [0,0]], "REFLECT")
    target_filtered = tf.nn.depthwise_conv2d(target_new_filter, kernel, strides=[1,1,1,1], padding="VALID")
    target_diff = target - target_filtered 

    predict_img = tf.pad(predict, [[0,0], [2,2], [2,2], [0,0]], "REFLECT")
    predict_filtered = tf.nn.depthwise_conv2d(predict_img, kernel, strides=[1,1,1,1], padding="VALID")
    predict_down = predict_filtered[:, ::2, ::2, :]
    predict_new_filter = tf.zeros_like(predict_filtered).numpy()
    predict_new_filter[:, ::2, ::2, :] = predict_down*4
    predict_new_filter = tf.pad(predict_new_filter, [[0,0], [2,2], [2,2], [0,0]], "REFLECT")
    predict_filtered = tf.nn.depthwise_conv2d(predict_new_filter, kernel, strides=[1,1,1,1], padding="VALID")
    predict_diff = predict - predict_filtered 

    loss = CharbonnierLoss(target_diff, predict_diff)
    return loss

# optimizer = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)

loss_object_gan = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_optimizer = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)

train_gen_gan_loss = tf.keras.metrics.Mean('train_gen_gan_loss', dtype=tf.float32)
train_gen_l2_loss = tf.keras.metrics.Mean('train_gen_l2_loss', dtype=tf.float32)
train_gen_total_loss = tf.keras.metrics.Mean('train_gen_total_loss', dtype=tf.float32)
train_edge_loss = tf.keras.metrics.Mean('train edge loss', dtype=tf.float32)

train_disc_gan_loss = tf.keras.metrics.Mean('train_disc_gan_loss', dtype=tf.float32)

test_gen_gan_loss = tf.keras.metrics.Mean('test_gen_gan_loss', dtype=tf.float32)
test_gen_l2_loss = tf.keras.metrics.Mean('test_gen_l2_loss', dtype=tf.float32)
test_gen_total_loss = tf.keras.metrics.Mean('test_gen_total_loss', dtype=tf.float32)
test_edge_loss = tf.keras.metrics.Mean('test edge loss', dtype=tf.float32)

test_disc_gan_loss = tf.keras.metrics.Mean('test_disc_gan_loss', dtype=tf.float32)

train_psnr = tf.keras.metrics.Mean('train psnr', dtype=tf.float32)
train_ssim = tf.keras.metrics.Mean('train ssim', dtype=tf.float32)

test_psnr = tf.keras.metrics.Mean('test psnr', dtype=tf.float32)
test_ssim = tf.keras.metrics.Mean('test ssim', dtype=tf.float32)

def cal_generator_loss(gen_image, target_image, disc_generated_output):
    gan_loss = loss_object_gan(tf.ones_like(disc_generated_output), disc_generated_output)

    l2_loss = tf.reduce_mean((target_image - gen_image)**2)

    edge_loss = EdgeLoss(target_image, gen_image)

    total_gan_loss = (FLAGS.gen_lambda * gan_loss) + (FLAGS.l2_lambda * l2_loss) + (FLAGS.edge_lambda * edge_loss)

    return gan_loss, l2_loss, edge_loss, total_gan_loss

def cal_discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object_gan(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object_gan(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

# loss & optimizer & metrics -- end

# train & test log -- start
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

train_log_dir = FLAGS.logs + current_time + '/train'
test_log_dir = FLAGS.logs + current_time + '/test'

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
# train & test log -- end

# model -- start
generator = Unet(input_shape=(FLAGS.image_size, FLAGS.image_size, FLAGS.channel))
discriminator = Discriminator()
generator.summary()
discriminator.summary()
# model -- end

# check point -- start
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 generator=generator,
                                 discriminator_optimizer=discriminator_optimizer,
                                 discriminator=discriminator)
ckpt_manager = tf.train.CheckpointManager(checkpoint, FLAGS.checkpoint_dir, max_to_keep=None)
# check point -- end

# train & test step -- start
def train_step(input_image, target_image):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_image = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target_image], training=True)
        disc_gen_output = discriminator([input_image, gen_image], training=True)

        gen_gan_loss, gen_l2_loss, edge_loss, gen_total_loss = cal_generator_loss(gen_image, target_image, disc_gen_output)
        disc_gan_loss = cal_discriminator_loss(disc_real_output, disc_gen_output)
    
    gen_gradient = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    disc_gradient = disc_tape.gradient(disc_gan_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gen_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradient, discriminator.trainable_variables))
    

    train_gen_gan_loss(gen_gan_loss)
    train_gen_l2_loss(gen_l2_loss)
    train_gen_total_loss(gen_total_loss)
    train_edge_loss(edge_loss)

    train_disc_gan_loss(disc_gan_loss)
    return gen_image[0], target_image[0]

def test_step(input_image, target_image):
    gen_image = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target_image], training=False)
    disc_gen_output = discriminator([input_image, gen_image], training=False)

    gen_gan_loss, gen_l2_loss, edge_loss, gen_total_loss = cal_generator_loss(gen_image, target_image, input_image)
    disc_gan_loss = cal_discriminator_loss(disc_real_output, disc_gen_output)

    test_gen_gan_loss(gen_gan_loss)
    test_gen_l2_loss(gen_l2_loss)
    test_edge_loss(edge_loss)
    test_gen_total_loss(gen_total_loss)

    test_disc_gan_loss(disc_gan_loss)
# train & test step -- end

# fit -- start
def fit(train_data, test_data, epochs):
    template1 = '{}/{} epochs | progress time : {} sec'
    template2 = 'Train loss : G Total: {}, GAN: {}, L2: {}, edge: {}, D Total: {}'
    template3 = 'Test  loss : G Total: {}, GAN: {}, L2: {}, edge: {}, D Total: {}'

    train_edge_loss.reset_states()
    test_edge_loss.reset_states()

    train_iter = 0
    test_iter = 0

    for epochs in range(epochs):
        start = time.time()

        for input_image, target_image in train_data:
            gen, ori = train_step(input_image, target_image)
            with train_summary_writer.as_default():
                tf.summary.scalar('train_gen_gan_loss', train_gen_gan_loss.result(), step=train_iter)
                tf.summary.scalar('train_gen_l2_loss', train_gen_l2_loss.result(), step=train_iter)
                tf.summary.scalar('train_gen_total_loss', train_gen_total_loss.result(), step=train_iter)
                tf.summary.scalar('train edge loss', train_edge_loss.result(), step=train_iter)
                tf.summary.scalar('train_disc_gan_loss', train_disc_gan_loss.result(), step=train_iter)
            train_iter += 1

        tf.keras.preprocessing.image.save_img(FLAGS.samples + '{}_gen.png'.format(epochs), gen)
        tf.keras.preprocessing.image.save_img(FLAGS.samples + '{}_ori.png'.format(epochs), ori)

        for input_image, target_image in test_data:
            test_step(input_image, target_image)

            with test_summary_writer.as_default():
                tf.summary.scalar('test_gen_gan_loss', test_gen_gan_loss.result(), step=test_iter)
                tf.summary.scalar('test_gen_l2_loss', test_gen_l2_loss.result(), step=test_iter)
                tf.summary.scalar('test_gen_total_loss', test_gen_total_loss.result(), step=test_iter)
                tf.summary.scalar('test edge loss', test_edge_loss.result(), step=test_iter)
                tf.summary.scalar('test_disc_gan_loss', test_disc_gan_loss.result(), step=test_iter)
            test_iter += 1

        print(template1.format(epochs, FLAGS.epochs, time.time() - start))
        print(template2.format(train_gen_total_loss.result(), train_gen_gan_loss.result(), train_gen_l2_loss.result(), train_edge_loss.result(), train_disc_gan_loss.result()))
        print(template3.format(test_gen_total_loss.result(), test_gen_gan_loss.result(), test_gen_l2_loss.result(), test_edge_loss.result(), test_disc_gan_loss.result()))
        ckpt_manager.save()
    ckpt_manager.save()
# fit -- end

# main -- start
def main():
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.mkdir(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.logs):
        os.mkdir(FLAGS.logs)
    if not os.path.exists(FLAGS.samples):
        os.mkdir(FLAGS.samples)
    
    if FLAGS.mode == 'Train':
        tr_input_images, tr_target_images = load_label_txt(FLAGS.train_labels)
        te_input_images, te_target_images = load_label_txt(FLAGS.test_labels)

        train_dataset = tf.data.Dataset.from_tensor_slices((tr_input_images, tr_target_images))
        train_dataset = train_dataset.shuffle(FLAGS.buffer_size)
        train_dataset = train_dataset.map(load_image)
        train_dataset = train_dataset.batch(FLAGS.batch_size)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((te_input_images, te_target_images))
        test_dataset = test_dataset.shuffle(FLAGS.buffer_size)
        test_dataset = test_dataset.map(load_image)
        test_dataset = test_dataset.batch(FLAGS.batch_size)
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        fit(train_dataset, test_dataset, FLAGS.epochs)
    
    elif FLAGS.mode == 'Test':
        if not os.path.exists(FLAGS.test_dir):
            os.mkdir(FLAGS.test_dir)
        te_input_images, te_target_images = load_label_txt(FLAGS.train_labels)

        test_dataset = tf.data.Dataset.from_tensor_slices((te_input_images, te_target_images))
        test_dataset = test_dataset.map(load_image)
        test_dataset = test_dataset.batch(1)

        checkpoint.restore(tf.train.latest_checkpoint(FLAGS.checkpoint_dir))\

        count = 0

        for input_image, target_image in test_dataset:
            gen_image = generator(input_image, training=False)
            gen_image = gen_image[0]
            image_name = te_input_images[count].split('/')[-1]

            tf.keras.preprocessing.image.save_img(FLAGS.test_dir + image_name, gen_image)

            count += 1

    else:
        print('check mode')

# define main -- end

if __name__ == "__main__":
    main()
