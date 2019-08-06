# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This code implements a Feed forward neural network using Keras API."""
import pickle



import argparse
import glob
import os
import keras.backend.tensorflow_backend as K
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model
import numpy as np
from tensorflow.python.lib.io import file_io

import trainer.model as model

# INPUT_SIZE = 288
ONE_HOUR=12
bins = np.array([50,100,150,200,250,500,1100])
# CLASS_SIZE = len(bins)

# CHUNK_SIZE specifies the number of lines
# to read in case the file is very large
CHUNK_SIZE = 5000
CHECKPOINT_FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
DISK_MODEL = 'disk_model.hdf5'




def train_and_evaluate(args):
  INPUT_DIM = args.training_history*ONE_HOUR
  CLASS_SIZE = len(bins)+1
  hidden_units = args.hidden_units
  # hidden_units = [int(units) for units in args.hidden_units.split(',')]
  learning_rate = args.learning_rate
  disk_model = model.model_fn(INPUT_DIM, CLASS_SIZE,
                              hidden_units,learning_rate
                              )
  try:
    os.makedirs(args.job_dir)
  except:
    pass

  # Unhappy hack to workaround h5py not being able to write to GCS.
  # Force snapshots and saves to local filesystem, then copy them over to GCS.
  checkpoint_path = CHECKPOINT_FILE_PATH
  if not args.job_dir.startswith('gs://'):
    checkpoint_path = os.path.join(args.job_dir, checkpoint_path)

  # Model checkpoint callback.
  checkpoint = ModelCheckpoint(
      checkpoint_path,
      monitor='val_loss',
      verbose=1,
      period=args.checkpoint_epochs,
      mode='min')

  # Continuous eval callback.
  # evaluation = ContinuousEval(args.eval_frequency, args.eval_files,
	# 														args.learning_rate, args.job_dir)

  # Tensorboard logs callback.
  tb_log = TensorBoard(
      log_dir=os.path.join(args.job_dir, 'logs'),
      histogram_freq=0,
      write_graph=True,
      embeddings_freq=0)

  callbacks = [checkpoint,tb_log]

  history = disk_model.fit_generator(
      model.generator_input(args.train_files, args.training_history,args.train_batch_size),
      validation_data=model.generator_input(args.eval_files, args.training_history,args.eval_batch_size),
      steps_per_epoch=args.train_steps,validation_steps = 10,
      epochs=args.num_epochs,
      callbacks=callbacks)

  # Unhappy hack to workaround h5py not being able to write to GCS.
  # Force snapshots and saves to local filesystem, then copy them over to GCS.
  if args.job_dir.startswith('gs://'):
    disk_model.save(DISK_MODEL)
    copy_file_to_gcs(args.job_dir, DISK_MODEL)
  else:
    disk_model.save(os.path.join(args.job_dir, DISK_MODEL))
  with file_io.FileIO(
          os.path.join(args.job_dir, 'history'), mode='w+') as output_f:
      pickle.dump(history.history, output_f)
  # with open('/Users/xuerongwan/Documents/keras_job/history_1', 'wb') as fp:
  #     pickle.dump(history.history, fp)
  # Convert the Keras model to TensorFlow SavedModel.
  # model.to_savedmodel(disk_model, os.path.join(args.job_dir, 'export'))


# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
  with file_io.FileIO(file_path, mode='rb') as input_f:
    with file_io.FileIO(
        os.path.join(job_dir, file_path), mode='w+') as output_f:
      output_f.write(input_f.read())


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train-files',
      nargs='+',
      help='Training file local or GCS',
      default='/Users/xuerongwan/Documents/keras_job/train/*.npz')
  parser.add_argument(
      '--eval-files',
      nargs='+',
      help='Evaluation file local or GCS',
      default='/Users/xuerongwan/Documents/keras_job/eval/*.npz')
  parser.add_argument(
      '--job-dir',
      type=str,
      help='GCS or local dir to write checkpoints and export model',
      default='/Users/xuerongwan/Documents/keras_job')
  parser.add_argument(
      '--training-history',
      type=int,
      default=24,
      help='number of hours of input')
  parser.add_argument(
      '--hidden_units',
      nargs='*',
      help='List of hidden units per layer. All layers are fully connected. Ex.'
           '`64 32` means first layer has 64 nodes and second one has 32.',
      default='400,200',
      )
  parser.add_argument(
      '--train-steps',
      type=int,
      default=30,
      help="""\
        Maximum number of training steps to perform
        Training steps are in the units of training-batch-size.
        So if train-steps is 500 and train-batch-size if 100 then
        at most 500 * 100 training instances will be used to train.""")
  parser.add_argument(
      '--eval-steps',
      help='Number of steps to run evalution for at each checkpoint',
      default=7,
      type=int)
  parser.add_argument(
      '--train-batch-size',
      type=int,
      default=30000,
      help='Batch size for training steps')
  parser.add_argument(
      '--eval-batch-size',
      type=int,
      default=30000,
      help='Batch size for evaluation steps')
  parser.add_argument(
      '--learning-rate',
      type=float,
      default=0.001,
      help='Learning rate for SGD')
  parser.add_argument(
      '--eval-frequency',
      default=10,
      help='Perform one evaluation per n epochs')
  parser.add_argument(
      '--first-layer-size',
      type=int,
      default=256,
      help='Number of nodes in the first layer of DNN')
  parser.add_argument(
      '--num-layers',
      type=int,
      default=2,
      help='Number of layers in DNN')
  parser.add_argument(
      '--scale-factor',
      type=float,
      default=0.25,
      help="""Rate of decay size of layer for Deep Neural Net.
        max(2, int(first_layer_size * scale_factor**i))""")
  parser.add_argument(
      '--eval-num-epochs',
      type=int,
      default=1,
      help='Number of epochs during evaluation')
  parser.add_argument(
      '--num-epochs',
      type=int,
      default=2,
      help='Maximum number of epochs on which to train')
  parser.add_argument(
      '--checkpoint-epochs',
      type=int,
      default=10,
      help='Checkpoint per n training epochs')

  args, _ = parser.parse_known_args()
  train_and_evaluate(args)


######################################google code call back########################################

class ContinuousEval(Callback):
  """Continuous eval callback to evaluate the checkpoint once

     every so many epochs.
  """

  def __init__(self,
               eval_frequency,
               eval_files,
               learning_rate,
               job_dir,
               steps=1000):
    self.eval_files = eval_files
    self.eval_frequency = eval_frequency
    self.learning_rate = learning_rate
    self.job_dir = job_dir
    self.steps = steps

  def on_epoch_begin(self, epoch, logs={}):
    """Compile and save model."""
    if epoch > 0 and epoch % self.eval_frequency == 0:
      # Unhappy hack to work around h5py not being able to write to GCS.
      # Force snapshots and saves to local filesystem, then copy them over to GCS.
      model_path_glob = 'checkpoint.*'
      if not self.job_dir.startswith('gs://'):
        model_path_glob = os.path.join(self.job_dir, model_path_glob)
      checkpoints = glob.glob(model_path_glob)
      if len(checkpoints) > 0:
        checkpoints.sort()
        disk_model = load_model(checkpoints[-1])
        disk_model = model.compile_model(disk_model, self.learning_rate)
        loss, acc = disk_model.evaluate_generator(
            model.generator_input(self.eval_files, chunk_size=CHUNK_SIZE),
            steps=self.steps)
        print('\nEvaluation epoch[{}] metrics[{:.2f}, {:.2f}] {}'.format(
            epoch, loss, acc, disk_model.metrics_names))
        if self.job_dir.startswith('gs://'):
          copy_file_to_gcs(self.job_dir, checkpoints[-1])
      else:
        print('\nEvaluation epoch[{}] (no checkpoints found)'.format(epoch))