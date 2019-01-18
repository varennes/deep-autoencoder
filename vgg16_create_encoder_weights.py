from keras.layers import Input
from keras.applications.vgg16 import VGG16
from module_autoencoder import Encoder

vgg16_weights = 'model_weights/vgg16_initial_encoder_weights.h5'

input_shape = ( 48, 48, 3)
vgg16 = VGG16( input_shape=input_shape, include_top=False, weights='imagenet')
vgg16.layers.pop()
vgg16.save_weights( vgg16_weights)

ec_input = Input( input_shape, name='ec_input')
encoder_vgg16 = Encoder( ec_input)
encoder_vgg16.model.load_weights( vgg16_weights)
encoder_vgg16.model.save_weights( vgg16_weights)
