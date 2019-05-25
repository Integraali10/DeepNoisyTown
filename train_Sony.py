from __future__ import division
import os, time, scipy.io
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


import numpy as np
import rawpy
import glob
from network_Sony import network_RED30,network_UnetCrop

name_of = 'result_RED30_withval'
input_dir = '/workspace/SID/Sony/short/'
gt_dir = '/workspace/SID/Sony/long/'
checkpoint_dir = os.path.join('./checkpoint/', name_of)
result_dir = os.path.join('/workspace/SID/Sony/garbage/', name_of)
val_results = os.path.join(result_dir, 'validation')
train_logs = os.path.join(result_dir, 'train_logs')
# get train IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]
val_fns = glob.glob(gt_dir + '2*.ARW')
val_ids = [int(os.path.basename(val_fn)[0:5]) for val_fn in val_fns]

ps = 512  # patch size for training
save_freq = 200
val_freq = 10
DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    train_ids = train_ids[0:5]
    val_ids  = train_ids[0:1]


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def PSNR_val (a, b, max_val):
    max_val = math_ops.cast(max_val, a.dtype)
    max_val = tf.image.convert_image_dtype(max_val, tf.float32)
    a = tf.image.convert_image_dtype(a, tf.float32)
    b = tf.image.convert_image_dtype(b, tf.float32)
    mse = math_ops.reduce_mean(math_ops.squared_difference(a, b), [-3, -2, -1])
    psnr_val = math_ops.subtract(
    20 * math_ops.log(max_val) / math_ops.log(10.0),
    np.float32(10 / np.log(10)) * math_ops.log(mse),
    name='psnr')
    return array_ops.identity(psnr_val)

sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
out_image = network_RED30(in_image,ps,ps)
#out_image = network(in_image)

G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))
PSNR = PSNR_val(out_image, gt_image, max_val=1.0)

t_vars = tf.trainable_variables()
lr = tf.placeholder(tf.float32)
G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)
saver = tf.train.Saver()

with tf.name_scope('performance'):
    # Summaries need to be displayed
    # Whenever you need to record the loss, feed the mean loss to this placeholder
    tf_glos_ph = tf.placeholder(tf.float32,shape=None,name='loss_summary')
    # Create a scalar summary object for the loss so it can be displayed
    tf_glos_summary = tf.summary.scalar('loss', tf_glos_ph)

    # Whenever you need to record the loss, feed the mean test accuracy to this placeholder
    tf_psnr_ph = tf.placeholder(tf.float32,shape=None, name='psnr_summary')
    # Create a scalar summary object for the accuracy so it can be displayed
    tf_psnr_summary = tf.summary.scalar('PSNR', tf_psnr_ph)

performance_summaries = tf.summary.merge([tf_glos_summary,tf_psnr_summary])
#pisatel into tensorboard
val_writer = tf.summary.FileWriter(val_results, sess.graph)
#train_writer = tf.summary.FileWriter(train_logs, sess.graph)

sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

# Raw data takes long time to load. Keep them in memory after loaded.
gt_images = [None] * 6000
input_images = {}
input_images['300'] = [None] * len(train_ids)
input_images['250'] = [None] * len(train_ids)
input_images['100'] = [None] * len(train_ids)

val_gt = [None] * 6000
val_images = {}
val_images['300'] = [None] * len(val_ids)
val_images['250'] = [None] * len(val_ids)
val_images['100'] = [None] * len(val_ids)
g_loss = np.zeros((5000, 1))
psnr_arr = np.zeros((5000, 1))
val_g_loss = np.zeros((5000, 1))
val_psnr_arr = np.zeros((5000, 1))

#
# allfolders = glob.glob('./result/*0')
# lastepoch = 0
# for folder in allfolders:
#     lastepoch = np.maximum(lastepoch, int(folder[-4:]))

def validation_sessions(epoch):
    cnt = 0
    for ind in np.random.permutation(len(val_ids)):
        # get the path from image id
        val_id = val_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % val_id)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % val_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()
        cnt+=1
        if val_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            val_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            val_gt[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # crop
        H = val_images[str(ratio)[0:3]][ind].shape[1]
        W = val_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = val_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = val_gt[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)

        G_current, PSNR_current, output = sess.run([G_loss, PSNR, out_image],
                                        feed_dict={in_image: input_patch, gt_image: gt_patch})
        output = np.minimum(np.maximum(output, 0), 1)
        val_psnr_arr[ind] = PSNR_current
        val_g_loss[ind] = G_current
        avgPSNR = np.mean(val_psnr_arr[np.where(val_psnr_arr)])
        avgGloss = np.mean(val_g_loss[np.where(val_g_loss)])

        print("%d val %d Loss=%.3f Time=%.3f PSNR = %.3f" % (epoch, cnt, avgGloss, time.time() - st, avgPSNR))

        nam = os.path.join(val_results,'%04d' % epoch)
        if not os.path.isdir(nam):
            os.makedirs(nam)

        temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
        scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
            nam + '/%05d_00_val_%d.jpg' % (val_id, ratio))

    avgPSNR = np.mean(val_psnr_arr[np.where(val_psnr_arr)])
    avgGloss = np.mean(val_g_loss[np.where(val_g_loss)])
    summ = sess.run(performance_summaries, feed_dict={tf_psnr_ph: avgPSNR, tf_glos_ph: avgGloss})
    # Write the obtained summaries to the file, so it can be displayed in the TensorBoard
    val_writer.add_summary(summ, epoch)

learning_rate = 1e-3
for epoch in range(0, 5001):
    if os.path.isdir("result/%04d" % epoch):
        continue
    cnt = 0
    if epoch > 20 and epoch <=2000:
        learning_rate = 1e-4
    elif epoch> 2000 and epoch <=4000:
        learning_rate = 1.5e-5
    else:
        learning_rate = 1e-5

    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()
        cnt += 1

        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)

        _, G_current, PSNR_current, output = sess.run([G_opt, G_loss, PSNR, out_image],
                                        feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate})
        output = np.minimum(np.maximum(output, 0), 1)
        psnr_arr[ind] = PSNR_current
        g_loss[ind] = G_current

        print("%d %d Loss=%.3f Time=%.3f PSNR = %.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st,
                                             np.mean(psnr_arr[np.where(psnr_arr)])))
        #train_writer.add_summary(summ, epoch)
        if epoch % save_freq == 0:
            nam = os.path.join(result_dir,'%04d' % epoch)
            if not os.path.isdir(nam):
                os.makedirs(nam)

            temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
            scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                nam + '/%05d_00_train_%d.jpg' % (train_id, ratio))
    if epoch % val_freq == 0:
        validation_sessions(epoch)
    namecheckpoint = os.path.join(checkpoint_dir , 'model.ckpt')
    saver.save(sess, namecheckpoint)

sess.close()