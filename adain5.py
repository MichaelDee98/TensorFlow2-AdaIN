# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import PIL
import time
import os
import random
"""#Global"""

BATCH_SIZE = 8
IMAGE_SIZE = [256, 256, 3]
ENCODER_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1'
)
ENCODER_WEIGHTS_PATH = './vgg19_normalised.npz'
LEARNING_RATE = 1e-4
LR_DECAY_RATE = 5e-5
DECAY_STEPS = 1.0

"""#Utils

"""

def preprocess(image, mode='BGR'):
    if mode == 'BGR':
        return image - np.array([103.939, 116.779, 123.68])
    else:
        return image - np.array([123.68, 116.779, 103.939])

def deprocess(image, mode='BGR'):
    if mode == 'BGR':
        return image + np.array([103.939, 116.779, 123.68])
    else:
        return image + np.array([123.68, 116.779, 103.939])

def preprocess_img(img):
  """Preprocess image."""
  crop_size = 256
  img = tf.image.random_crop(img, (crop_size, crop_size, 3))

  return img


def process_path(path_to_img):
  img = load_and_process_img(path_to_img)
  img = preprocess_img(img)
  return img

def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    min_dim = 512

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    small_dim = tf.reduce_min(shape)
    scale = min_dim / small_dim
    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = tf.image.random_crop(img, [256,256,3])
    #img = img[tf.newaxis, :]

    return img
    
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def prepare_dataset(path_to_imgs):
  dataset = tf.data.Dataset.list_files(path_to_imgs)
  dataset = dataset.map(load_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset

def load_img_Inference(path_to_img, max_dim=None, resize=True, frame = False):

    img = tf.io.read_file(path_to_img)     
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if resize:
        if frame:
            new_shape = tf.cast([218, 512], tf.int32)
            img = tf.image.resize(img, new_shape)
        else:  
            new_shape = tf.cast([256, 256], tf.int32)
            img = tf.image.resize(img, new_shape)

    if max_dim:
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        
    img = img[tf.newaxis, :]


    return img


def decode_img(img, reverse_channels=False):
  """Decodes preprocessed images."""

  # perform the inverse of the preprocessiing step
  img *= 255.
  if reverse_channels:
    img = img[..., ::-1]

  img = tf.cast(img, dtype=tf.uint8)
  return img

#style_path = enter your style path
#content_path = enter your content path

"""##Dataset"""

style_train_ds = prepare_dataset(style_path + '/**/*.jpg')

content_train_ds = prepare_dataset(content_path + '/*.jpg')

print(f"Style images {len(style_train_ds)}")

print(f"Contain images {len(content_train_ds)}")

train_ds = tf.data.Dataset.zip((style_train_ds, content_train_ds))
train_ds = train_ds.shuffle(BATCH_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

print(f"Final train dataset {len(train_ds)}")

"""#Networks"""

class Encoder(tf.keras.Model):

  def __init__(self, weights_path):
    super().__init__()
    # load weights (kernel and bias) from npz file
    weights = np.load(weights_path)

    idx = 0
    self.weight_vars = []

    # create the TensorFlow variables
    for layer in ENCODER_LAYERS:
      kind = layer[:4]

      if kind == 'conv':
        kernel = weights['arr_%d' % idx].transpose([2, 3, 1, 0])
        bias   = weights['arr_%d' % (idx + 1)]
        kernel = kernel.astype(np.float32)
        bias   = bias.astype(np.float32)
        idx += 2

        
        W = tf.Variable(kernel, trainable=False, name='kernel')
        b = tf.Variable(bias,   trainable=False, name='bias')

        self.weight_vars.append((W, b))

  def get_model(self):
    # create the computational graph
    idx = 0
    style_outputs = []
    content_outputs = []
    inputs = tf.keras.layers.Input(shape=(None, None, 3), batch_size=None, name="input") 
    x = inputs

    for layer in ENCODER_LAYERS:
      kind = layer[:4]

      if kind == 'conv':
        x = tf.keras.layers.Lambda(
        lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
        mode='REFLECT'))(x)  

        kernel, bias = self.weight_vars[idx]
        idx += 1
        filters = kernel.get_shape()[-1]
        kernel_size = [kernel.get_shape()[0],kernel.get_shape()[1]]

        # conv and add bias
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, 
                                    kernel_initializer=tf.keras.initializers.Constant(kernel),
                                    bias_initializer=tf.keras.initializers.Constant(bias),
                                    trainable=False, name = layer)(x)

      elif kind == 'relu':
        x = tf.keras.layers.ReLU(name=layer)(x)

        if layer in style_layers:
          style_outputs.append(x)
        if layer in content_layers:
          content_outputs.append(x) 

      elif kind == 'pool':
        x = tf.keras.layers.MaxPool2D(pool_size=[2, 2],strides=[2, 2], padding='SAME', name=layer)(x)
      
    model_outputs = style_outputs + content_outputs
    models = [tf.keras.models.Model(inputs=inputs, outputs=outputs) for outputs in model_outputs]
    return models

class Decoder(tf.keras.Model):

  def __init__(self):
    super().__init__()
    self.weight_vars = []
    self._scale = 2
      
    self.weight_vars.append(self._create_variables(512, 256, 3, scope='conv4_1'))

    self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_4'))
    self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_3'))
    self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_2'))
    self.weight_vars.append(self._create_variables(256, 128, 3, scope='conv3_1'))

    self.weight_vars.append(self._create_variables(128, 128, 3, scope='conv2_2'))
    self.weight_vars.append(self._create_variables(128,  64, 3, scope='conv2_1'))

    self.weight_vars.append(self._create_variables( 64,  64, 3, scope='conv1_2'))
    self.weight_vars.append(self._create_variables( 64,   3, 3, scope='conv1_1'))

  def _create_variables(self, input_filters, output_filters, kernel_size, scope):
    shape  = [kernel_size, kernel_size, input_filters, output_filters]
    kernel = tf.Variable(tf.initializers.GlorotUniform()(shape=shape), name = 'kernel')
    bias = tf.Variable(tf.initializers.GlorotUniform()(shape=[output_filters]), name='bias')
    #print(kernel)
    return (kernel, bias)

  def get_model(self):
    # upsampling after 'conv4_1', 'conv3_1', 'conv2_1'
    upsample_indices = (0, 4, 6)
    final_layer_idx  = len(self.weight_vars) - 1

    inputs = tf.keras.layers.Input(shape=(None, None, 512), batch_size=None)
    x = inputs
    for i in range(len(self.weight_vars)):
      kernel, bias = self.weight_vars[i]

      if i == final_layer_idx:
        x = tf.keras.layers.Lambda(
        lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
        mode='REFLECT'))(x)  


        filters = kernel.get_shape()[-1]
        kernel_size = [kernel.get_shape()[0],kernel.get_shape()[1]]

        # conv and add bias
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, 
                                    kernel_initializer=tf.keras.initializers.Constant(kernel),
                                    bias_initializer=tf.keras.initializers.Constant(bias),
                                    trainable=True)(x)
      else:
        x = tf.keras.layers.Lambda(
        lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
        mode='REFLECT'))(x)  
        filters = kernel.get_shape()[-1]
        kernel_size = [kernel.get_shape()[0],kernel.get_shape()[1]]

        # conv and add bias
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, 
                                    kernel_initializer=tf.keras.initializers.Constant(kernel),
                                    bias_initializer=tf.keras.initializers.Constant(bias),
                                    trainable=True, activation=tf.keras.activations.relu)(x)
      
      if i in upsample_indices:
        height = tf.shape(x)[1] * self._scale
        width  = tf.shape(x)[2] * self._scale
        x = tf.image.resize(x, [height, width], 
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
    outputs = x
    return tf.keras.models.Model(inputs=inputs, outputs=outputs)

def adaptive_instance_normalization(style, content, epsilon=1e-5):
  style_mean, style_var = tf.nn.moments(style, [1,2], keepdims=True)
  style_std = tf.sqrt(style_var + epsilon)

  content_mean, content_var = tf.nn.moments(content, [1,2], keepdims=True)
  content_std = tf.sqrt(content_var + epsilon)

  adain = (style_std*(content - content_mean)/content_std) + style_mean  
  return adain

"""#Losses"""

def get_content_loss(adain_output, target_encoded):
  return tf.reduce_sum(tf.reduce_mean(tf.square(adain_output - target_encoded),axis=[1,2]))

def get_style_loss(base_style_encoded, target_encoded):
  eps = 1e-5
  
  base_style_mean, base_style_var = tf.nn.moments(base_style_encoded, 
                                                  axes=[1,2])
  # Add epsilon for numerical stability for gradients close to zero
  base_style_std = tf.math.sqrt(base_style_var + eps)

  target_mean, target_var = tf.nn.moments(target_encoded,
                                          axes=[1,2])
  # Add epsilon for numerical stability for gradients close to zero
  target_std = tf.math.sqrt(target_var + eps)

  mean_diff = tf.reduce_sum(tf.square(base_style_mean - target_mean))
  std_diff = tf.reduce_sum(tf.square(base_style_std - target_std))
  return tf.reduce_sum(mean_diff + std_diff)

STYLE_LOSS_WEIGHT = 2

def get_loss(adain_output, style, target_encoded):
  # Content loss
  content_loss = get_content_loss(adain_output, encoder[-1](target_encoded))
  
  # Style loss
  style_loss = 0
  for i in range(num_style_layers):
    style_loss += get_style_loss(encoder[i](style), encoder[i](target_encoded))

  return content_loss + STYLE_LOSS_WEIGHT * style_loss

"""#Train Step"""

def train_step(content_img, style_img):
  with tf.GradientTape() as tape:
    
    
    encoded_content = encoder[-1](content_img)
    encoded_style = encoder[-1](style_img)

    adain_output = adaptive_instance_normalization(encoded_style, encoded_content)

    target_img = decoder(adain_output)

    target_img = deprocess(target_img)
    target_img = tf.reverse(target_img, axis=[-1])

    target_img = tf.clip_by_value(target_img, 0.0, 255.0)

    target_img = tf.reverse(target_img, axis=[-1])
    target_img = preprocess(target_img)
    
    
    loss = get_loss(adain_output, style_img, target_img)

    

  gradients = tape.gradient(loss, decoder.trainable_variables)
  optimizer.apply_gradients(zip(gradients, decoder.trainable_variables))

  train_loss(loss)

# Content layer where will pull our feature maps
content_layers = ['relu4_1'] 

# Style layer we are interested in
style_layers = ['relu1_1',
                'relu2_1',
                'relu3_1', 
                'relu4_1' 
               ]
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

encoder_model = Encoder(ENCODER_WEIGHTS_PATH)
encoder = encoder_model.get_model()
decoder_model = Decoder()
decoder = decoder_model.get_model()

learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(LEARNING_RATE, DECAY_STEPS, LR_DECAY_RATE)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
train_loss = tf.keras.metrics.Mean(name='train_loss')

EPOCHS = 4
PROGBAR = tf.keras.utils.Progbar(len(train_ds))
for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()

  step = 0
  start_time = time.perf_counter()
  for (style_tr, content_tr) in train_ds.as_numpy_iterator():
      start_time = time.perf_counter()

      style_tr = style_tr*255
      style_tr = tf.reverse(style_tr, axis=[-1])
      style_tr = preprocess(style_tr)

      content_tr = content_tr*255
      content_tr = tf.reverse(content_tr, axis=[-1])
      content_tr = preprocess(content_tr)
      
      train_step(content_tr, style_tr)
      print(f"Train step: {time.perf_counter() - start_time}")
      # start_time = time.perf_counter()
      step += 1
      PROGBAR.update(step)

  template = 'Epoch {}, Loss: {}'
  print(template.format(epoch+1,
                        train_loss.result()))

decoder.save_weights("./weights/git/decoder")
