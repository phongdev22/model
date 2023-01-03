from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D

def denoise(input_shape = (64,64,1)):
    model = Sequential([
        #encode
        Conv2D(64, (3,3), activation='relu', padding='same',input_shape=input_shape),
        MaxPooling2D(2,padding='same'),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,padding='same'),
        Conv2D(16, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,padding='same'),
        #decode
        Conv2D(16, (3,3), activation='relu', padding='same'),
        UpSampling2D(2),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        UpSampling2D(2),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        UpSampling2D(2),
        Conv2D(1, (3,3), activation='sigmoid', padding='same'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    # model.summary()

    return model