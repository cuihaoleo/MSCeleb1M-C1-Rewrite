#!/usr/bin/env python3

import os
import base64

import tensorflow as tf
import numpy as np
import cv2

tf.flags.DEFINE_string(
    'input', '', 'Input (TSV or directory with JPEG/PNG images).')

tf.flags.DEFINE_string(
    'output_file', 'output.csv', 'Output CSV file.')

tf.flags.DEFINE_string(
    'meta_graph', '', 'Exported MetaGraph file.')

tf.flags.DEFINE_string(
    'model', '', 'Exported model file.')

tf.flags.DEFINE_integer(
    'batch_size', 32, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def aligned_tsv_loader(input_tsv):
    with open(input_tsv) as fin:
        for line in fin:
            freebase_mid, search_rank, image_url, page_url, \
                face_id, box_b64, image_b64 = line.strip().split("\t")

            fullpath = "{},{},{}".format(freebase_mid, search_rank, face_id)
            buf = base64.decodebytes(image_b64.encode())
            buf = np.frombuffer(buf, dtype=np.uint8)
            raw = cv2.imdecode(buf, cv2.IMREAD_COLOR)

            yield fullpath, raw


def directory_image_loader(input_dir):
    for fname in os.listdir(input_dir):
        if os.path.splitext(fname)[-1].lower() not in [".jpg", ".png"]:
            continue

        fullpath = os.path.join(input_dir, fname)
        raw = cv2.imread(fullpath, cv2.IMREAD_COLOR)
        yield fullpath, raw


def load_images(input_path, target_size, batch_size):
    if os.path.isdir(input_path):
        loader = directory_image_loader(input_path)
    elif os.path.isfile(input_path) and input_path.lower().endswith(".tsv"):
        loader = aligned_tsv_loader(input_path)
    else:
        raise Exception("Unknown input!")

    batch_shape = [batch_size] + list(target_size) + [3]
    images = np.zeros(batch_shape, dtype=np.uint8)
    filenames = []
    idx = 0

    for fullpath, raw in loader:
        resized = cv2.resize(raw, target_size[::-1])

        images[idx, ...] = resized
        filenames.append(fullpath)

        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape, dtype=np.uint8)
            idx = 0

    if idx > 0:
        yield filenames, images


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    saver = tf.train.import_meta_graph(FLAGS.meta_graph, clear_devices=True)
    graph = tf.get_default_graph()
    input_tf = graph.get_tensor_by_name("input:0")
    output_tf = graph.get_tensor_by_name("towerp0/linear/output:0")
    prob_tf = tf.nn.softmax(output_tf)

    image_height = input_tf.shape[1].value
    image_width = input_tf.shape[2].value

    with tf.Session() as sess:
        saver.restore(sess, FLAGS.model)
        with open(FLAGS.output_file, "w") as fout:
            for filenames, images in load_images(FLAGS.input,
                                                 (image_height, image_width),
                                                 FLAGS.batch_size):
                prob = sess.run(prob_tf, feed_dict={input_tf: images})
                for idx, path in enumerate(filenames):
                    label = prob[idx, :].argmax()
                    confidence = prob[idx, label]
                    print(path, label, confidence, sep=",", file=fout)
                    fout.flush()


if __name__ == "__main__":
    tf.app.run()
