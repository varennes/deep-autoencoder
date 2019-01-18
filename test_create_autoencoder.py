from module_autoencoder import Autoencoder
from keras.layers import Input

vgg16_weights = 'model_weights/vgg16_initial_encoder_weights.h5'

input_shape = ( 48, 48, 3)
ec_input = Input( input_shape, name='ec_input')
ae = Autoencoder( ec_input)
ae.model.load_weights( vgg16_weights, by_name=True)

print ae.model.summary()
