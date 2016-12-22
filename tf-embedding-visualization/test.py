import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.models.image.imagenet import classify_image
from tensorflow.contrib.tensorboard.plugins import projector

classify_image.FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_dir', '/tmp/imagenet',
                           """Path to model.""")


def main(argv=None):
    classify_image.maybe_download_and_extract()
    classify_image.create_graph()
    basedir = os.path.dirname(__file__)
    with tf.Session() as sess:
        data = np.loadtxt('embedding.tsv')
        embedding_var = tf.Variable(data, name = 'embedding_var')
        # prepare projector config
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        summary_writer = tf.train.SummaryWriter(os.path.join(basedir, 'logdir'))
        # link metadata
        #metadata_path = os.path.join(basedir, 'logdir', 'metadata.tsv')
        #with open(metadata_path, 'w') as f:
        #    for name in files:
        #        f.write('%s\n' % name)
        #embedding.metadata_path = metadata_path

        # write to sprite image file
        image_path = os.path.join(basedir, 'logdir', 'sprite.jpg')
        #size = int(math.sqrt(len(images))) + 1
        #while len(images) < size * size:
        #    images.append(np.zeros((100, 100, 3), dtype=np.uint8))
        #rows = []
        #for i in range(size):
        #    rows.append(tf.concat(1, images[i*size:(i+1)*size]))
        #jpeg = tf.image.encode_jpeg(tf.concat(0, rows))
        #with open(image_path, 'wb') as f:
        #    f.write(sess.run(jpeg))
        embedding.sprite.image_path = image_path
        embedding.sprite.single_image_dim.extend([64, 64])
        # save embedding_var
        projector.visualize_embeddings(summary_writer, config)
        sess.run(tf.variables_initializer([embedding_var]))
        saver = tf.train.Saver([embedding_var])
        saver.save(sess, os.path.join(basedir, 'logdir', 'model.ckpt'))


if __name__ == '__main__':
    tf.app.run()
