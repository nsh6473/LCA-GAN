import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import datetime
import matplotlib.cm as cm

from tensorflow.keras.preprocessing import image_dataset_from_directory

import argparse

parser = argparse.ArgumentParser(description='')

#--------------------------------------------------
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#--------------------------------------------------

# parser.add_argument('--mode', dest='mode', default='Train', help='train or test mode')
parser.add_argument('--mode', dest='mode', default='Test', help='train or test mode')
# parser.add_argument('--mode', dest='mode', default='Feature', help='feature map and class activation map')

# parser.add_argument('--train_labels', dest='train_labels', default='/mnt/ssd/data/MORPH/mask/data1_ori.txt', help='train data labels')
# parser.add_argument('--test_labels', dest='test_labels', default='/mnt/ssd/data/MORPH/mask/data2_ori.txt', help='test data labels')
# parser.add_argument('--classes', dest='classes', type=int, default=50, help='num of classes')
parser.add_argument('--train_labels', dest='train_labels', default='/mnt/ssd/data/PAL/mask/train1_ori.txt', help='train data labels')
parser.add_argument('--test_labels', dest='test_labels', default='/mnt/ssd/data/PAL/mask/test1_ori.txt', help='test data labels')
parser.add_argument('--classes', dest='classes', type=int, default=73, help='num of classes')

parser.add_argument('--feature_image', dest='feature_image', default='/mnt/ssd/network/prop2/train_58/335021_00M17.JPG', help='test data labels')
parser.add_argument('--pred_index', dest='pred_index', default=None, help='grad CAM pre index')
parser.add_argument('--feature_dir', dest='feature_dir', default='./feature', help='feature dir')
parser.add_argument('--cam_dir', dest='cam_dir', default='/cam', help='cam dir')
parser.add_argument('--grad_cam_dir', dest='grad_cam_dir', default='/grad_cam', help='grad cam dir')

parser.add_argument('--feature_alpha', dest='feature_alpha', type=float, default=0.4, help='feature alpha')

parser.add_argument('--image_size', dest='image_size', type=int, default=224, help='image size')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=256, help='crop size')
parser.add_argument('--channel', dest='channel', type=int, default=3, help='channel')

parser.add_argument('--batch_size', dest = 'batch_size', type=int, default=20, help='num of batch size')
parser.add_argument('--buffer_size', dest = 'buffer_size', type=int, default=500, help='num of buffer_size')
parser.add_argument('--epochs', dest = 'epochs', type=int, default=200, help='num of epochs')
parser.add_argument('--save_frq', dest = 'save_frq', type=int, default=1, help='checkpoint save frequence')

parser.add_argument('--test_dir', dest='test_dir', default='./test1', help='test dir')

# parser.add_argument('--checkpoint', dest = 'checkpoint', default='/mnt/disk2/experiment3/dex/train1/cp-{epoch:03d}.ckpt', help='checkopint')
parser.add_argument('--checkpoint', dest = 'checkpoint', default='/mnt/disk2/experiment3/PAL/dex/train1/cp-{epoch:03d}.ckpt', help='checkopint')
# parser.add_argument('--checkpoint', dest = 'checkpoint', default='/mnt/disk2/experiment3/train1_high/', help='checkopint')

args = parser.parse_args()

# util function -- start
def load_label_txt(label_file):
    images = np.loadtxt(label_file, dtype='<U100', skiprows=0, usecols=0)
    labels = np.loadtxt(label_file, dtype=np.int32, skiprows=0, usecols=1)
    return images, labels

def load(images):
    input_image = tf.io.read_file(images)
    input_image = tf.image.decode_jpeg(input_image)
    input_image = tf.cast(input_image, tf.float32)
    return input_image

def resize(images, height, width):
    images = tf.image.resize(images, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return images

def normalize(image):
    image = (image / 127.5) - 1
    return image

def load_image_train(images, labels):
    image = load(images)    
    image = resize(image, args.crop_size, args.crop_size)
    image = tf.image.random_crop(image, size=[args.image_size, args.image_size, args.channel])
    image = tf.image.random_flip_left_right(image)
    image = normalize(image)

    label = tf.one_hot(labels, args.classes)
    return image, label

def load_image_test(images, labels):
    image = load(images)
    image = resize(image, args.image_size, args.image_size)
    image = normalize(image)

    label = tf.one_hot(labels, args.classes)
    return image, label
# util function -- end

# model build -- start
vgg16_conv = tf.keras.applications.VGG16(input_shape = (args.image_size, args.image_size, args.channel), 
                                        include_top = False, 
                                        weights = 'imagenet')
vgg16_conv.trainable = True

feature = vgg16_conv.output
GAP = tf.keras.layers.GlobalAveragePooling2D()(feature)
dense1 = tf.keras.layers.Dense(4096, activation='relu')(GAP)
drop1 = tf.keras.layers.Dropout(0.5)(dense1)
dense2 = tf.keras.layers.Dense(4096, activation='relu')(drop1)
drop2 = tf.keras.layers.Dropout(0.5)(dense2)
outputs = tf.keras.layers.Dense(args.classes)(drop2)
model = tf.keras.Model(inputs = vgg16_conv.input, outputs=outputs)

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.Adam(lr=2e-5),
              metrics=['accuracy'])

model.summary()
# model build -- end

# check point -- start
checkpoint_path = args.checkpoint
checkpoint_dir = os.path.dirname(checkpoint_path)
# check point -- end

##### callback --start #####
# tensorboard callback -- start
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                        histogram_freq=1)
# tensorboard callback -- end

# checkpoint callback -- start
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=args.batch_size)
# checkpoint callback -- end

vgg_callback = [tensorboard_callback, checkpoint_callback]
##### callback --end #####

def main():
    # if not os.path.exists(args.checkpoint_dir):
    #     os.makedirs(args.checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    if args.mode == 'Train':
        tr_images, tr_labels = load_label_txt(args.train_labels)
        te_images, te_labels = load_label_txt(args.test_labels)

        train_dataset = tf.data.Dataset.from_tensor_slices((tr_images, tr_labels))
        train_dataset = train_dataset.shuffle(args.buffer_size)
        train_dataset = train_dataset.map(load_image_train)
        train_dataset = train_dataset.batch(args.batch_size)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((te_images, te_labels))
        test_dataset = test_dataset.shuffle(args.buffer_size)
        test_dataset = test_dataset.map(load_image_test)
        test_dataset = test_dataset.batch(args.batch_size)
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        model.save_weights(checkpoint_path.format(epoch=0))

        history = model.fit(train_dataset,
                            epochs =args.epochs,
                            validation_data = test_dataset,
                            callbacks = vgg_callback)

    elif args.mode == 'Test':
        te_low_images, te_labels = load_label_txt(args.test_labels)

        test_dataset = tf.data.Dataset.from_tensor_slices((te_low_images, te_labels))
        test_dataset = test_dataset.map(load_image_test)
        test_dataset = test_dataset.batch(1)

        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

        age_MAE = 0.0
        dex_MAE = 0.0

        images = len(test_dataset)

        count = 0

        for image, label in test_dataset:
            pred = model.predict(image)
            soft = tf.nn.softmax(pred)

            pred_age = np.where(np.max(soft)==soft)[1][0]
            pred_dex = np.sum(np.arange(args.classes) * soft)

            real_age = np.where(np.max(label)==label)[1][0]

            age_AE = np.abs(pred_age - real_age)
            dex_AE = np.abs(pred_dex - real_age)

            age_MAE += age_AE
            dex_MAE += dex_AE

            print("------------------------------------------------------")
            print("{}'s age AE : {}".format(te_low_images[count], age_AE))
            print("{}'s dex AE : {}".format(te_low_images[count], dex_AE))
            print("------------------------------------------------------")

            count += 1

        total_age = age_MAE
        total_dex = dex_MAE

        age_MAE /= images
        dex_MAE /= images
        print("Total AE: {}, age MAE : {}".format(total_age, age_MAE))
        print("Total AE: {}, dex MAE : {}".format(total_dex, dex_MAE))
    
    elif args.mode == 'Feature':
        feature_dir = args.feature_dir
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)

        cam_dir = feature_dir + args.cam_dir
        if not os.path.exists(cam_dir):
            os.makedirs(cam_dir)

        grad_cam_dir = feature_dir + args.grad_cam_dir
        if not os.path.exists(grad_cam_dir):
            os.makedirs(grad_cam_dir)

        ori_image = tf.io.read_file(args.feature_image)
        ori_image = tf.image.decode_jpeg(ori_image)
        plt.imsave(cam_dir + '/ori_image.JPG', ori_image.numpy())
        ori_image = tf.image.resize(ori_image, [args.image_size, args.image_size])
        image = ori_image / 127.5 - 1
        image = tf.expand_dims(image, 0)

        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

        layer_outputs = [layer.output for layer in model.layers]
        extract_model = tf.keras.Model(inputs = model.input, outputs = layer_outputs)
        extract_model.summary()

        with tf.GradientTape(persistent=True) as tape:
            feature = extract_model(image)

            if args.pred_index is None:
                pred_index = tf.argmax(feature[-1][0])
            class_channel = feature[-1][:, pred_index]

        for conv_layer in range(1, len(feature) - 6):
            gradient = tape.gradient(class_channel, feature[conv_layer])
            pooled_gradient = tf.reduce_mean(gradient, axis=(0, 1, 2))
            conv_output = feature[conv_layer]
            heatmap = conv_output @ pooled_gradient[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = (tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)).numpy()

            grad_cam_image_name = '/gradCAM_' + str(conv_layer)
            plt.imsave(grad_cam_dir + grad_cam_image_name + '.JPG', heatmap)

            heatmap = np.uint8(255 * heatmap)
            jet = cm.get_cmap('jet')
            jet_color = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_color[heatmap]

            print(np.shape(heatmap), np.shape(jet_heatmap))

            jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
            jet_heatmap = jet_heatmap.resize((args.image_size, args.image_size))
            jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap) / 255.0
            heatmap_alpha = jet_heatmap * args.feature_alpha

            grad_cam_jet_name = '/gradCAM_jet_' + str(conv_layer)

            grad_cam_alpha_name = '/gradCAM_alpha_' + str(conv_layer)

            plt.imsave(grad_cam_dir + grad_cam_jet_name + '.JPG', jet_heatmap)
            plt.imsave(grad_cam_dir + grad_cam_alpha_name + '.JPG', heatmap_alpha)

            ori_image = tf.keras.preprocessing.image.img_to_array(ori_image)
            superimosed_image = heatmap_alpha + ((ori_image / 255.0) * (1 - args.feature_alpha))

            superimosed_image_name = '/gradCAM_superimosed_' + str(conv_layer)

            plt.imsave(grad_cam_dir + superimosed_image_name + '.JPG', superimosed_image)

# range(1, len(feature - 6)) -> 1 is input layer feature - 6 -> dense and activation layer
        for conv_layer in range(1, len(feature)-6):
            conv_layer_file = '/conv' + str(conv_layer)

            if not os.path.exists(feature_dir + conv_layer_file):
                os.makedirs(feature_dir + conv_layer_file)

            class_activation_map = 0.

            for channel in range(feature[conv_layer].shape[-1]):
                image_name = '/conv' + str(conv_layer) + '_' + str(channel)
                feature_map = feature[conv_layer][0,:,:,channel]
                feature_map = tf.expand_dims(feature_map, -1)
                class_activation_map += tf.image.resize(feature_map, [args.image_size, args.image_size])
                plt.imsave(feature_dir + conv_layer_file + image_name + '.JPG', feature_map[:,:,0])

            class_activation_map /= channel
            class_activation_map = tf.expand_dims(class_activation_map, -1)
            plt.imsave(cam_dir + '/conv' + str(conv_layer) + '.JPG', class_activation_map)
    
#     elif args.mode == 'Feature':
#         feature_dir = args.feature_dir
#         if not os.path.exists(feature_dir):
#             os.makedirs(feature_dir)

#         cam_dir = feature_dir + args.cam_dir
#         if not os.path.exists(cam_dir):
#             os.makedirs(cam_dir)

#         ori_image = tf.io.read_file(args.feature_image)
#         ori_image = tf.image.decode_jpeg(ori_image)
#         plt.imsave(cam_dir + '/ori_image.JPG', ori_image.numpy())
#         ori_image = tf.image.resize(ori_image, [args.image_size, args.image_size])
#         image = ori_image / 127.5 - 1
#         image = tf.expand_dims(image, 0)

#         model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

#         layer_outputs = [layer.output for layer in model.layers]
#         extract_model = tf.keras.Model(inputs = model.input, outputs = layer_outputs)
#         extract_model.summary()

#         print(layer_outputs[-7])

#         with tf.GradientTape() as tape:
#             feature = extract_model(image)

#             if args.pred_index is None:
#                 pred_index = tf.argmax(feature[-1][0])
#             class_channel = feature[-1][:, pred_index]

#         gradient = tape.gradient(class_channel, feature[-7])
#         pooled_gradient = tf.reduce_mean(gradient, axis=(0, 1, 2))
#         last_conv_outputs = feature[-7]
#         heatmap = last_conv_outputs @ pooled_gradient[..., tf.newaxis]
#         heatmap = tf.squeeze(heatmap)
#         heatmap = (tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)).numpy()

#         plt.imsave(cam_dir + '/gradCAM.JPG', heatmap)

#         heatmap = np.uint8(255 * heatmap)
#         jet = cm.get_cmap('jet')
#         jet_color = jet(np.arange(256))[:, :3]
#         jet_heatmap = jet_color[heatmap]

#         jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
#         jet_heatmap = jet_heatmap.resize((args.image_size, args.image_size))
#         jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap) / 255.0
#         heatmap_alpha = jet_heatmap * args.feature_alpha

#         plt.imsave(cam_dir + '/gradCAM_jet.JPG', jet_heatmap)
#         plt.imsave(cam_dir + '/gradCAM_alpha.JPG', heatmap_alpha)

#         ori_image = tf.keras.preprocessing.image.img_to_array(ori_image)
#         superimosed_image = heatmap_alpha + ((ori_image / 255.0) * (1 - args.feature_alpha))

#         plt.imsave(cam_dir + '/superimosed_image.JPG', superimosed_image)

# # range(1, len(feature - 6)) -> 1 is input layer feature - 6 -> dense and activation layer
#         for conv_layer in range(1, len(feature)-6):
#             conv_layer_file = '/conv' + str(conv_layer)

#             if not os.path.exists(feature_dir + conv_layer_file):
#                 os.makedirs(feature_dir + conv_layer_file)

#             class_activation_map = 0.

#             for channel in range(feature[conv_layer].shape[-1]):
#                 image_name = '/conv' + str(conv_layer) + '_' + str(channel)
#                 feature_map = feature[conv_layer][0,:,:,channel]
#                 feature_map = tf.expand_dims(feature_map, -1)
#                 class_activation_map += tf.image.resize(feature_map, [args.image_size, args.image_size])
#                 plt.imsave(feature_dir + conv_layer_file + image_name + '.JPG', feature_map[:,:,0])

#             class_activation_map /= channel
#             class_activation_map = tf.expand_dims(class_activation_map, -1)
#             plt.imsave(cam_dir + '/conv' + str(conv_layer) + '.JPG', class_activation_map)

if __name__ == "__main__":
    main()
