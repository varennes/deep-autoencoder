from keras.models import Model
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, UpSampling2D


class Encoder:
    def __init__(self, input_tensor):
        self.model = self.get_model( input_tensor)

    def get_model(self, input_tensor):
        ec_block1 = Conv2D( 64, (3, 3), padding='same',
                            activation='relu', name='ec_b1_conv1')(input_tensor)
        ec_block1 = Conv2D( 64, (3, 3), padding='same',
                            activation='relu', name='ec_b1_conv2')(ec_block1)
        ec_block1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                            name='ec_b1_pool')(ec_block1)

        ec_block2 = Conv2D(128, (3, 3), padding='same',
                            activation='relu', name='ec_b2_conv1')(ec_block1)
        ec_block2 = Conv2D(128, (3, 3), padding='same',
                            activation='relu', name='ec_b2_conv2')(ec_block2)
        ec_block2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                            name='ec_b2_pool')(ec_block2)

        ec_block3 = Conv2D(256, (3, 3), padding='same',
                            activation='relu', name='ec_b3_conv1')(ec_block2)
        ec_block3 = Conv2D(256, (3, 3), padding='same',
                            activation='relu', name='ec_b3_conv2')(ec_block3)
        ec_block3 = Conv2D(256, (3, 3), padding='same',
                            activation='relu', name='ec_b3_conv3')(ec_block3)
        ec_block3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                            name='ec_b3_pool')(ec_block3)

        ec_block4 = Conv2D(512, (3, 3), padding='same',
                            activation='relu', name='ec_b4_conv1')(ec_block3)
        ec_block4 = Conv2D(512, (3, 3), padding='same',
                            activation='relu', name='ec_b4_conv2')(ec_block4)
        ec_block4 = Conv2D(512, (3, 3), padding='same',
                            activation='relu', name='ec_b4_conv3')(ec_block4)
        ec_block4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                            name='ec_b4_pool')(ec_block4)

        ec_block5 = Conv2D(512, (3, 3), padding='same',
                            activation='relu', name='ec_b5_conv1')(ec_block4)
        ec_block5 = Conv2D(512, (3, 3), padding='same',
                            activation='relu', name='ec_b5_conv2')(ec_block5)
        ec_block5 = Conv2D(512, (3, 3), padding='same',
                            activation='relu', name='ec_b5_conv3')(ec_block5)

        return Model( inputs=input_tensor, outputs=ec_block5)


class Decoder:
    def __init__(self, input_tensor):
        self.model = self.get_model( input_tensor)

    def get_model(self, input_tensor):
        dc_block1 = Dense(512, activation='relu', name='dc_b1_dense1')(input_tensor)
        dc_block1 = Dense(784, activation='relu', name='dc_b1_dense2')(dc_block1)

        dc_block2 = Conv2D( 16, (3, 3), padding='same',
                            activation='relu', name='dc_b2_conv1')(dc_block1)
        dc_block2 = UpSampling2D(size=(2, 2), name='dc_b2_upsample')(dc_block2)

        dc_block3 = Conv2D( 32, (3, 3), padding='same',
                            activation='relu', name='dc_b3_conv1')(dc_block2)
        dc_block3 = UpSampling2D(size=(2, 2), name='dc_b3_upsample')(dc_block3)

        dc_block4 = Conv2D( 64, (3, 3), padding='same',
                            activation='relu', name='dc_b4_conv1')(dc_block3)
        dc_block4 = UpSampling2D(size=(2, 2), name='dc_b4_upsample')(dc_block4)

        dc_block5 = Conv2D(128, (3, 3), padding='same',
                            activation='relu', name='dc_b5_conv1')(dc_block4)
        dc_block5 = UpSampling2D(size=(2, 2), name='dc_b5_upsample')(dc_block5)

        dc_block6 = Conv2D( 64, (3, 3), padding='same',
                            activation='relu', name='dc_b6_conv1')(dc_block5)
        dc_block6 = Conv2D(  3, (3, 3), padding='same',
                            activation='relu', name='dc_b6_conv2')(dc_block6)

        return Model( inputs=input_tensor, outputs=dc_block6)


class Autoencoder:
    def __init__(self, input_tensor):
        self.model = self.get_model( input_tensor)
        self.layer_names = [ layer.name for layer in self.model.layers]

    def get_model(self, input_tensor):
        ec_block1 = Conv2D( 64, (3, 3), padding='same',
                            activation='relu', name='ec_b1_conv1')(input_tensor)
        ec_block1 = Conv2D( 64, (3, 3), padding='same',
                            activation='relu', name='ec_b1_conv2')(ec_block1)
        ec_block1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                            name='ec_b1_pool')(ec_block1)

        ec_block2 = Conv2D(128, (3, 3), padding='same',
                            activation='relu', name='ec_b2_conv1')(ec_block1)
        ec_block2 = Conv2D(128, (3, 3), padding='same',
                            activation='relu', name='ec_b2_conv2')(ec_block2)
        ec_block2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                            name='ec_b2_pool')(ec_block2)

        ec_block3 = Conv2D(256, (3, 3), padding='same',
                            activation='relu', name='ec_b3_conv1')(ec_block2)
        ec_block3 = Conv2D(256, (3, 3), padding='same',
                            activation='relu', name='ec_b3_conv2')(ec_block3)
        ec_block3 = Conv2D(256, (3, 3), padding='same',
                            activation='relu', name='ec_b3_conv3')(ec_block3)
        ec_block3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                            name='ec_b3_pool')(ec_block3)

        ec_block4 = Conv2D(512, (3, 3), padding='same',
                            activation='relu', name='ec_b4_conv1')(ec_block3)
        ec_block4 = Conv2D(512, (3, 3), padding='same',
                            activation='relu', name='ec_b4_conv2')(ec_block4)
        ec_block4 = Conv2D(512, (3, 3), padding='same',
                            activation='relu', name='ec_b4_conv3')(ec_block4)
        ec_block4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                            name='ec_b4_pool')(ec_block4)

        ec_block5 = Conv2D(512, (3, 3), padding='same',
                            activation='relu', name='ec_b5_conv1')(ec_block4)
        ec_block5 = Conv2D(512, (3, 3), padding='same',
                            activation='relu', name='ec_b5_conv2')(ec_block5)
        ec_block5 = Conv2D(512, (3, 3), padding='same',
                            activation='relu', name='ec_b5_conv3')(ec_block5)

        dc_block1 = Dense(512, activation='relu', name='dc_b1_dense1')(ec_block5)
        dc_block1 = Dense(784, activation='relu', name='dc_b1_dense2')(dc_block1)

        dc_block2 = Conv2D( 16, (3, 3), padding='same',
                            activation='relu', name='dc_b2_conv1')(dc_block1)
        dc_block2 = UpSampling2D(size=(2, 2), name='dc_b2_upsample')(dc_block2)

        dc_block3 = Conv2D( 32, (3, 3), padding='same',
                            activation='relu', name='dc_b3_conv1')(dc_block2)
        dc_block3 = UpSampling2D(size=(2, 2), name='dc_b3_upsample')(dc_block3)

        dc_block4 = Conv2D( 64, (3, 3), padding='same',
                            activation='relu', name='dc_b4_conv1')(dc_block3)
        dc_block4 = UpSampling2D(size=(2, 2), name='dc_b4_upsample')(dc_block4)

        dc_block5 = Conv2D(128, (3, 3), padding='same',
                            activation='relu', name='dc_b5_conv1')(dc_block4)
        dc_block5 = UpSampling2D(size=(2, 2), name='dc_b5_upsample')(dc_block5)

        dc_block6 = Conv2D( 64, (3, 3), padding='same',
                            activation='relu', name='dc_b6_conv1')(dc_block5)
        dc_block6 = Conv2D(  3, (3, 3), padding='same',
                            activation='relu', name='dc_b6_conv2')(dc_block6)

        return Model( inputs=input_tensor, outputs=dc_block6)

    def freeze_encoder(self):
        for name in self.layer_names:
            if name.startswith('ec'):
                self.model.get_layer( name).trainable = False

    def thaw_encoder(self):
        for name in self.layer_names:
            if name.startswith('ec'):
                self.model.get_layer( name).trainable = Train

    def freeze_decoder(self, freeze_list=[]):
        # if no list / empty list provided, freeze all decoder layers
        if len(freeze_list)==0:
            for name in self.layer_names:
                if name.startswith('dc'):
                    self.model.get_layer( name).trainable = False
        else:
            for name in freeze_list:
                self.model.get_layer( name).trainable = False

    def freeze_status(self):
        print '\n AE MODEL LAYER TRAINING STATUS'
        for name in self.layer_names:
            print ' layer %s - Trainable = %s' % ( name, self.model.get_layer(name).trainable)
