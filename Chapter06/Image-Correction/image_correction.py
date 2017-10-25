import os
import time
import numpy as np
import tensorflow as tf
from glob import glob
from dcgan import DCGAN
from utils import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_dir',       'checkpoints',           """Path to write logs and checkpoints""")
tf.app.flags.DEFINE_string('complete_src',  'complete_src',          """Path to images for completion""")
tf.app.flags.DEFINE_string('complete_dir',  'complete',              """Path to save completed images""")
tf.app.flags.DEFINE_string('masktype',      'center',                """Mask types: center, random""")
tf.app.flags.DEFINE_integer('batch_size',   128,                     """Batch size""")
tf.app.flags.DEFINE_integer('latest_ckpt',  0,                       """Latest checkpoint timestamp to load""")
tf.app.flags.DEFINE_integer('nb_channels',  3,                       """Number of color channels""")
tf.app.flags.DEFINE_boolean('is_complete',  False,                   """True for completion only""")




def main(_):
    dcgan = DCGAN(batch_size=FLAGS.batch_size, s_size=6, nb_channels=FLAGS.nb_channels)

    g_saver = tf.train.Saver(dcgan.g.variables, max_to_keep=15)
    d_saver = tf.train.Saver(dcgan.d.variables, max_to_keep=15)
    g_checkpoint_path = os.path.join(FLAGS.log_dir, 'g.ckpt')
    d_checkpoint_path = os.path.join(FLAGS.log_dir, 'd.ckpt')
    g_checkpoint_restore_path = os.path.join(FLAGS.log_dir, 'g.ckpt-'+str(FLAGS.latest_ckpt))
    d_checkpoint_restore_path = os.path.join(FLAGS.log_dir, 'd.ckpt-'+str(FLAGS.latest_ckpt))

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, graph=sess.graph)

        sess.run(tf.global_variables_initializer())
        # restore or initialize generator
        if os.path.exists(g_checkpoint_restore_path+'.meta'):
            print('Restoring variables:')
            for v in dcgan.g.variables:
                print(' ' + v.name)
            g_saver.restore(sess, g_checkpoint_restore_path)

        if FLAGS.is_complete:
            # restore discriminator
            if os.path.exists(d_checkpoint_restore_path+'.meta'):
                print('Restoring variables:')
                for v in dcgan.d.variables:
                    print(' ' + v.name)
                d_saver.restore(sess, d_checkpoint_restore_path)

                # Directory to save completed images
                if not os.path.exists(FLAGS.complete_dir):
                    os.makedirs(FLAGS.complete_dir)

                # Create mask
                if FLAGS.masktype == 'center':
                    scale = 0.25
                    mask = np.ones(dcgan.image_shape)
                    sz = dcgan.image_size
                    l = int(sz*scale)
                    u = int(sz*(1.0-scale))
                    mask[l:u, l:u, :] = 0.0
                if FLAGS.masktype == 'random':
                    fraction_masked = 0.8
                    mask = np.ones(dcgan.image_shape)
                    mask[np.random.random(dcgan.image_shape[:2]) < fraction_masked] = 0.0

                # Read actual images
                originals = glob(os.path.join(FLAGS.complete_src, '*.jpg'))
                batch_mask = np.expand_dims(mask, axis=0)

                for idx in range(len(originals)):
                    image_src = get_image(originals[idx], dcgan.image_size, nb_channels=FLAGS.nb_channels)
                    if FLAGS.nb_channels == 3:
                        image = np.expand_dims(image_src, axis=0)
                    elif FLAGS.nb_channels == 1:
                        image = np.expand_dims(np.expand_dims(image_src, axis=3), axis=0)

                    # Save original image (y)
                    filename = os.path.join(FLAGS.complete_dir, 'original_image_{:02d}.jpg'.format(idx))
                    imsave(image_src, filename)

                    # Save corrupted image (y . M)
                    filename = os.path.join(FLAGS.complete_dir, 'corrupted_image_{:02d}.jpg'.format(idx))
                    if FLAGS.nb_channels == 3:
                        masked_image = np.multiply(image_src, mask)
                        imsave(masked_image, filename)
                    elif FLAGS.nb_channels == 1:
                        masked_image = np.multiply(np.expand_dims(image_src, axis=3), mask)
                        imsave(masked_image[:, :, 0], filename)

                    zhat = np.random.uniform(-1, 1, size=(1, dcgan.z_dim))
                    v = 0
                    momentum = 0.9
                    lr = 0.01

                    for i in range(0, 1001):
                        fd = {dcgan.zhat: zhat, dcgan.mask: batch_mask, dcgan.image: image}
                        run = [dcgan.complete_loss, dcgan.grad_complete_loss, dcgan.G]
                        loss, g, G_imgs = sess.run(run, feed_dict=fd)

                        v_prev = np.copy(v)
                        v = momentum*v - lr*g[0]
                        zhat += -momentum * v_prev + (1+momentum)*v
                        zhat = np.clip(zhat, -1, 1)

                        if i % 100 == 0:
                            filename = os.path.join(FLAGS.complete_dir,
                                'hats_img_{:02d}_{:04d}.jpg'.format(idx, i))
                            if FLAGS.nb_channels == 3:
                                save_images(G_imgs[0, :, :, :], filename)
                            if FLAGS.nb_channels == 1:
                                save_images(G_imgs[0, :, :, 0], filename)

                            inv_masked_hat_image = np.multiply(G_imgs, 1.0-batch_mask)
                            completed = masked_image + inv_masked_hat_image
                            filename = os.path.join(FLAGS.complete_dir,
                                'completed_{:02d}_{:04d}.jpg'.format(idx, i))
                            if FLAGS.nb_channels == 3:
                                save_images(completed[0, :, :, :], filename)
                            if FLAGS.nb_channels == 1:
                                save_images(completed[0, :, :, 0], filename)



if __name__ == '__main__':
    tf.app.run()

