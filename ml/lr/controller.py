# -*- coding: utf-8 -*-

import tensorflow as tf
from util.DataUtil import DataUtil


flags=tf.app.flags
FLAGS=flags.FLAGS
flags.DEFINE_float('learning_rate', 1, 'The learning rate.')
flags.DEFINE_string('sample_file', None, 'Sample file.')
flags.DEFINE_string('mapping_file', None, 'Feature mapping file.')
flags.DEFINE_string('model_load_path', None, 'Modelloadpath.')
flags.DEFINE_string('model_save_path', None, 'Model save path.')
flags.DEFINE_integer('performance', 1, '0 for predict, 1 for training, 2 for estimate, 3 for print weight, and 4 for converting weight into liblinear form.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('epochs', 10, 'Training epochs.')
flags.DEFINE_integer('verbose', -1, 'How many batches to print information.')
flags.DEFINE_float('w1', 200, 'The weight for positive samples.')
flags.DEFINE_float('l2_reg', 0, 'The l2 regularization coefficient.')
flags.DEFINE_float('l1_reg', 0, 'The l1 regularization coefficient.')
flags.DEFINE_float('max_grad_norm', 10, 'The maximum permissible norm of the gradient.')
flags.DEFINE_float('input_shape', 10, 'Input units.')
flags.DEFINE_integer('feature_idx', 2, 'Feature string index.')
flags.DEFINE_integer('label_idx', 0, 'Label index.')
flags.DEFINE_boolean('load_once', False, 'Indicates whether load sample data once at all.')

def main(_):
    mapping_dict, max_idx, feature_size = DataUtil.load_mapping(FLAGS.mapping_file)
    FLAGS.input_shape = [feature_size]
    if FLAGS.performance==1:

        pass
    pass



if __name__ == '__main__':
    tf.app.run()
    pass

