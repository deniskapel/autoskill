import configparser
import os

import tensorflow as tf
import tensorflow_hub as tfhub

os.environ['TFHUB_CACHE_DIR'] = '../models/tf_cache'

config = configparser.ConfigParser()
config.read('config.ini')
g = tf.Graph()

with g.as_default():
    # feeding 1D tensors of text into the graph.
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    encoder = tfhub.load(config["URL"]["use"])
    embedded_text = encoder(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])

g.finalize()

session = tf.Session(graph=g)
session.run(init_op)

def embed(texts):
    return session.run(embedded_text, feed_dict={text_input: texts})
