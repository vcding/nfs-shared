from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import math
import numpy as np

import tensorflow as tf

import resnet_model

#cifar-10
FLAGS = None
batch_size=128
def _my_input_fn(filenames, num_epochs):
    image_bytes = 32 * 32 * 3
    label_bytes = 1
    
    def parser(serialized_example):
        train_bytes = tf.decode_raw(serialized_example, tf.uint8)
        train_label_uint8 = tf.strided_slice(
            train_bytes,
            [0],
            [label_bytes])
        train_image_uint8 = tf.strided_slice(
            train_bytes,
            [label_bytes],
            [label_bytes + image_bytes])
        train_label = tf.cast(train_label_uint8, tf.int32)
        train_label.set_shape([1])
        train_image_pre1 = tf.reshape(
            train_image_uint8,
            [3, 32, 32])
        """ [depth, height, width] -> [height, width, depth] """
        train_image_pre2 = tf.transpose(
            train_image_pre1,
            [1, 2, 0])
        train_image_pre3 = tf.cast(
            train_image_pre2,
            tf.float32)
        train_image = tf.image.per_image_standardization(train_image_pre3)
        train_image.set_shape([32, 32, 3])
        """ convert label : (ex) 2 -> (0.0, 0.0, 1.0, 0.0, ...) """
        train_label = tf.sparse_to_dense(train_label, [10], 1.0, 0.0)
        return train_image, train_label

    dataset = tf.data.FixedLengthRecordDataset(
        filenames,
        image_bytes + label_bytes)
    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(
        parser,
        num_parallel_calls=batch_size)
    dataset = dataset.prefetch(4 * batch_size).cache().repeat()
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size)
    )
    dataset = dataset.prefetch(1)
    return dataset

def _get_train_input_fn(traindir, num_epochs):
    filenames = [os.path.join(FLAGS.train_dir, 'data_batch_%d.bin' % i)
        for i in range(1, 6)]
    return lambda: _my_input_fn(filenames, num_epochs)

def _get_eval_input_fn(filename, num_epochs):
    filenames = [filename]
    return lambda: _my_input_fn(filenames, num_epochs)

def _my_model_fn(features, labels, mode):
    """ device is automatically detected and assigned """
    #device = '/job:localhost/replica:0/task:0/device:GPU:0'
    #with tf.device(device):

    #
    # Model - Here we use pre-built 'resnet_model'
    #
    params = resnet_model.HParams(
        batch_size=batch_size,
        num_classes=10,
        min_lrn_rate=0.0001,
        lrn_rate=0.1,
        num_residual_units=5, # 5 x (3 x sub 2) + 2 = 32 layers
        use_bottleneck=False,
        weight_decay_rate=0.0002,
        relu_leakiness=0.1,
        optimizer='mom')
    train_model = resnet_model.ResNet(
        params,
        features,
        labels,
        'train')
    train_model.build_graph()

    # create evaluation metrices
    """ Please umcomment """
    """ when you output precision and accuracy to TensorBoard or use INFER """
    #truth = tf.argmax(train_model.labels, axis=1)
    #predictions = tf.argmax(train_model.predictions, axis=1)
    #precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))
    #accuracy = tf.metrics.accuracy(truth, predictions)
    #tf.summary.scalar('precision', precision) # output to TensorBoard
    #tf.summary.scalar('accuracy', accuracy[1]) # output to TensorBoard

    # define operations
    if mode == tf.estimator.ModeKeys.TRAIN:
        """ We don't use tf.train.LoggingTensorHook because it doesn't work when distributed tensorflow. """
        #logging_hook = tf.train.LoggingTensorHook(
        #    tensors={
        #        'step': train_model.global_step,
        #        'loss': train_model.cost,
        #        'lrn_rate': train_model.lrn_rate,
        #        'precision': precision
        #    },
        #    every_n_iter=10) # log output every 10 steps
        class _CustomLogHook(tf.train.SessionRunHook):
            def before_run(self, run_context):
                return tf.train.SessionRunArgs(
                    fetches = [train_model.global_step, train_model.cost])
            def after_run(self, run_context, run_values):
                if run_values.results[0] % 10 == 0: # log output every 10 steps
                    print('step:%d  loss:%.2f' % (run_values.results[0], run_values.results[1]))
        return tf.estimator.EstimatorSpec(
            mode,
            loss=train_model.cost,
            train_op=train_model.train_op,
            #training_chief_hooks=[logging_hook])
            training_chief_hooks=[_CustomLogHook()])
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
#            'accuracy': accuracy
        }
        return tf.estimator.EstimatorSpec(
            mode,
            loss=train_model.cost,
            eval_metric_ops=eval_metric_ops)
    """ Please umcomment when you use INFER """
    #if mode == tf.estimator.ModeKeys.INFER:
    #    probabilities = tf.nn.softmax(train_model.predictions, name='softmax_tensor')
    #    predict_outputs = {
    #        'classes': predictions,
    #        'probabilities': probabilities
    #    }
    #    export_outputs = {
    #        'prediction': tf.estimator.export.PredictOutput(predict_outputs)
    #    }
    #    return tf.estimator.EstimatorSpec(
    #        mode,
    #        predictions=predict_outputs,
    #        export_outputs=export_outputs)

def main(_):
    #distribute
    run_config = tf.contrib.learn.RunConfig()
 
    # define
    cifar10_resnet_classifier = tf.estimator.Estimator(
        model_fn=_my_model_fn,
        model_dir=FLAGS.out_dir,
        config=run_config)
    train_spec = tf.estimator.TrainSpec(
        input_fn=_get_train_input_fn(FLAGS.train_dir, 10),
        #max_steps=50000 * 10 // batch_size) # Full spec
        max_steps=5000) # For benchmarking
        #max_steps=1000) # For seminar
    eval_spec = tf.estimator.EvalSpec(
        input_fn=_get_eval_input_fn(FLAGS.test_file, 1),
        steps=10000 * 1 / batch_size,
        start_delay_secs=0)
        
    # run !
    tf.estimator.train_and_evaluate(
        cifar10_resnet_classifier,
        train_spec,
        eval_spec
    )
    #cifar10_resnet_classifier.train(
    #    input_fn=_get_train_input_fn(FLAGS.train_dir, 10),
    #    hooks=[_LearningRateSetterHook()]
    #)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir',
        type=str,
        default=r'/tf/distribute/cifar-10-batches-bin/train_batches_bin',
        help='Dir path for the training data.')
    parser.add_argument(
        '--test_file',
        type=str,
        default=r'/tf/distribute/cifar-10-batches-bin/test_batch.bin',
        help='Dir path for the training data.')
    parser.add_argument(
        '--out_dir',
        type=str,
        default=r'/tf/distribute/out',
        help='Dir path for model output.')    
    #parser.add_argument(
    #    '--use_gpu',
    #    action='store_true')
    FLAGS, unparsed = parser.parse_known_args()
 
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
