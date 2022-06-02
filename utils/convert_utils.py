# tensorflow==1.14.0
# tensorflow_text==0.1.0
# tensorflow-hub==0.7.0

import numpy as np
import tensorflow_hub as tfhub
import tensorflow as tf
import tensorflow_text

from tqdm import tqdm

tensorflow_text.__name__

sess = tf.InteractiveSession(graph=tf.Graph())
module = tfhub.Module("./models/convert_data/convert")

text_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
extra_text_placeholder = tf.placeholder(dtype=tf.string, shape=[None])

# The encode_context signature now also takes the extra context.
context_encoding_tensor = module(
    {"context": text_placeholder, "extra_context": extra_text_placeholder}, signature="encode_context"
)


responce_text_placeholder = tf.placeholder(dtype=tf.string, shape=[None])

response_encoding_tensor = module(responce_text_placeholder, signature="encode_response")

sess.run(tf.tables_initializer())
sess.run(tf.global_variables_initializer())


def encode_context(dialogue_history):
    """Encode the dialogue context to the response ranking vector space.

    Args:
        dialogue_history: a list of strings, the dialogue history, in
            chronological order.
    """

    # The context is the most recent message in the history.
    context = dialogue_history[-1]

    extra_context = list(dialogue_history[:-1])
    extra_context.reverse()
    extra_context_feature = " ".join(extra_context)

    return sess.run(
        context_encoding_tensor,
        feed_dict={text_placeholder: [context], extra_text_placeholder: [extra_context_feature]},
    )[0]


def encode_responses(texts: list):
    """use convert model to encode all the responses """
    return sess.run(response_encoding_tensor, feed_dict={responce_text_placeholder: texts})


def eval_responses(encoded_context, encoded_responses) -> np.array:
    """calc scores of each response for given encoded context"""
    return encoded_context.dot(encoded_responses.T)
