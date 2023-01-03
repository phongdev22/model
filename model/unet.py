import tensorflow as tf

image_size = 512
channels = 3

def down_block(x, filters, kernal_size=(3,3), padding="same", strides=(1,1)):
    c = tf.keras.layers.Conv2D(filters , kernal_size , padding=padding , strides=strides , activation='relu')(x)
    c = tf.keras.layers.Dropout(0.2,)(c)
    c = tf.keras.layers.Conv2D(filters , kernal_size , padding=padding , strides=strides , activation='relu')(c)
    p = tf.keras.layers.MaxPool2D(pool_size = (2,2), strides=(2,2))(c)
    return c, p

def up_block(x, skip, filters, kernal_size=(3,3), padding="same", strides=(1,1)):
    us = tf.keras.layers.UpSampling2D((2,2))(x)
    concat = tf.keras.layers.Concatenate(axis=-1)([us,skip])
    c = tf.keras.layers.Conv2D(filters , kernal_size , padding=padding, strides=strides, activation='relu')(concat)
    c = tf.keras.layers.Conv2D(filters , kernal_size , padding=padding, strides=strides, activation='relu')(c)
    return c

def bottleneck(x, filters, kernal_size=(3,3), padding="same", strides=(1,1)):
    c = tf.keras.layers.Conv2D(filters , kernal_size ,  padding=padding , strides=strides , activation='relu')(x)
    c = tf.keras.layers.Conv2D(filters , kernal_size ,  padding=padding , strides=strides , activation='relu')(c)
    return c

def model():
    f = [2,4,8,16,32]
    
    inputs = tf.keras.layers.Input((image_size, image_size, channels))

    p0 = inputs

    c1, p1 = down_block(p0,f[0])
    c2, p2 = down_block(p1,f[1])
    c3, p3 = down_block(p2,f[2])
    c4, p4 = down_block(p3,f[3])

    bn = bottleneck(p4, f[4])

    u1 = up_block(bn, c4, f[3])
    u2 = up_block(u1, c3, f[2])
    u3 = up_block(u2, c2, f[1])
    u4 = up_block(u3, c1, f[0])

    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(u4)

    model = tf.keras.models.Model(inputs, outputs)
    model.summary()
    return model
