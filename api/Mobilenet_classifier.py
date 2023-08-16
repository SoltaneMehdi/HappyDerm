from keras import Model, models  # for model manipulation
from keras.applications import MobileNet
from keras.layers import (
    concatenate,
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Activation,
    BatchNormalization,
    Dropout,
    Layer,
)
from keras import backend as K
import keras.layers as kl
import tensorflow as tf


class SoftAttention(Layer):
    def __init__(self, ch, m, concat_with_x=False, aggregate=False, **kwargs):
        self.channels = int(ch)
        self.multiheads = m
        self.aggregate_channels = aggregate
        self.concat_input_with_scaled = concat_with_x

        super(SoftAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.i_shape = input_shape

        kernel_shape_conv3d = (self.channels, 3, 3) + (1, self.multiheads)  # DHWC

        self.out_attention_maps_shape = (
            input_shape[0:1] + (self.multiheads,) + input_shape[1:-1]
        )

        if self.aggregate_channels == False:
            self.out_features_shape = input_shape[:-1] + (
                input_shape[-1] + (input_shape[-1] * self.multiheads),
            )
        else:
            if self.concat_input_with_scaled:
                self.out_features_shape = input_shape[:-1] + (input_shape[-1] * 2,)
            else:
                self.out_features_shape = input_shape

        self.kernel_conv3d = self.add_weight(
            shape=kernel_shape_conv3d, initializer="he_uniform", name="kernel_conv3d"
        )
        self.bias_conv3d = self.add_weight(
            shape=(self.multiheads,), initializer="zeros", name="bias_conv3d"
        )

        super(SoftAttention, self).build(input_shape)

    def call(self, x):
        exp_x = K.expand_dims(x, axis=-1)

        c3d = K.conv3d(
            exp_x,
            kernel=self.kernel_conv3d,
            strides=(1, 1, self.i_shape[-1]),
            padding="same",
            data_format="channels_last",
        )
        conv3d = K.bias_add(c3d, self.bias_conv3d)
        conv3d = kl.Activation("relu")(conv3d)

        conv3d = K.permute_dimensions(conv3d, pattern=(0, 4, 1, 2, 3))

        conv3d = K.squeeze(conv3d, axis=-1)
        conv3d = K.reshape(
            conv3d, shape=(-1, self.multiheads, self.i_shape[1] * self.i_shape[2])
        )

        softmax_alpha = K.softmax(conv3d, axis=-1)
        softmax_alpha = kl.Reshape(
            target_shape=(self.multiheads, self.i_shape[1], self.i_shape[2])
        )(softmax_alpha)

        if self.aggregate_channels == False:
            exp_softmax_alpha = K.expand_dims(softmax_alpha, axis=-1)
            exp_softmax_alpha = K.permute_dimensions(
                exp_softmax_alpha, pattern=(0, 2, 3, 1, 4)
            )

            x_exp = K.expand_dims(x, axis=-2)

            u = kl.Multiply()([exp_softmax_alpha, x_exp])

            u = kl.Reshape(
                target_shape=(
                    self.i_shape[1],
                    self.i_shape[2],
                    u.shape[-1] * u.shape[-2],
                )
            )(u)

        else:
            exp_softmax_alpha = K.permute_dimensions(
                softmax_alpha, pattern=(0, 2, 3, 1)
            )

            exp_softmax_alpha = K.sum(exp_softmax_alpha, axis=-1)

            exp_softmax_alpha = K.expand_dims(exp_softmax_alpha, axis=-1)

            u = kl.Multiply()([exp_softmax_alpha, x])

        if self.concat_input_with_scaled:
            o = kl.Concatenate(axis=-1)([u, x])
        else:
            o = u

        return [o, softmax_alpha]

    def compute_output_shape(self, input_shape):
        return [self.out_features_shape, self.out_attention_maps_shape]

    def get_config(self):
        return super(SoftAttention, self).get_config()


def build_model(num_classes=7, loss_function="categorical_crossentropy"):
    mobile_net = MobileNet(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )

    mobile_net.trainable = False
    conv = mobile_net.layers[-6].output

    # Add the Soft Attention Layer
    attention_layer, map2 = SoftAttention(
        aggregate=True,
        m=16,
        concat_with_x=False,
        ch=int(conv.shape[-1]),
        name="soft_attention",
    )(conv)
    attention_layer = MaxPooling2D(pool_size=(2, 2), padding="same")(attention_layer)
    conv = MaxPooling2D(pool_size=(2, 2), padding="same")(conv)

    conv = concatenate([conv, attention_layer])
    conv = Activation("relu")(conv)
    conv = Dropout(0.2)(conv)
    conv = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D(pool_size=(4, 4), padding="same")(conv)
    conv = Flatten()(conv)
    conv = Dense(1024, activation="relu")(conv)
    conv = Dense(num_classes, activation="softmax")(conv)

    model = Model(inputs=mobile_net.inputs, outputs=conv)
    model.compile(optimizer="adam", loss=loss_function, metrics=["accuracy"])
    return model


model_binary = build_model(num_classes=2, loss_function="binary_crossentropy")
model_binary.load_weights("./MobileNet+SA/Binary_fmobilenetSA.h5")
model_malignant = build_model(num_classes=4, loss_function="categorical_crossentropy")
model_malignant.load_weights("./MobileNet+SA/4Classes_fmobilelnetSA.h5")
model_benign = build_model(num_classes=3, loss_function="categorical_crossentropy")
model_benign.load_weights("./MobileNet+SA/3Classes_fmobilenetSA.h5")


def classify(generator, mode="classic"):
    binary_pred = (model_binary.predict(generator))[0] * 100
    if binary_pred[0] > binary_pred[1]:  # case the image is judged benign:
        benign_pred = (model_benign.predict(generator))[0] * 100
        prediction = {
            "AKIEC": 0,
            "BCC": 0,
            "BKL": benign_pred[0],
            "DF": benign_pred[1],
            "MEL": 0,
            "NV": benign_pred[2],
            "VASC": 0,
        }
    else:  # case the image is judged malignant:
        malignant_pred = (model_malignant.predict(generator))[0] * 100
        prediction = {
            "AKIEC": malignant_pred[0],
            "BCC": malignant_pred[1],
            "BKL": 0,
            "DF": 0,
            "MEL": malignant_pred[2],
            "NV": 0,
            "VASC": malignant_pred[3],
        }

    return prediction
