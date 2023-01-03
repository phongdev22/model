from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from utils.DataLoader import DataLoader
from model.denoising import denoise
import numpy as np
import os
import pickle

n_epochs = 5
batch_size = 64

# dataset = DataLoader("./data/VINAI_Chest_Xray", 64, 64).load_data()
model = denoise()

def make_noise(dataset):   
    #contain noise image
    noise_images = []
    # generate noise image from dataset
    for normal_image in dataset:
        #noise coefficient
        w, h, c = normal_image.shape
        mean = 0
        sigma = 1
        gauss = np.random.normal(mean, sigma, (w, h, c))
        gauss = gauss.reshape(w, h, c)
        #------------------
        noise_image = normal_image + gauss * 0.08
        noise_images.append(noise_image)

    noise_images = np.array(noise_images)
    return noise_images;

if not os.path.exists('data.dat'):

    dataset = DataLoader("./data/VINAI_Chest_Xray", 64, 64).load_data()

    noise_images = make_noise(dataset=dataset)

    noise_train, noise_test, normal_train, normal_test = train_test_split(noise_images, dataset, test_size=0.2)

    with open("data.dat", "wb") as f:
        pickle.dump([noise_train, noise_test, normal_train, normal_test], f)
else:
    with open("data.dat", "rb") as f:
        arr = pickle.load(f)
        noise_train, noise_test, normal_train, normal_test = arr[0], arr[1], arr[2], arr[3]

early_callback = EarlyStopping(monitor="val_loss", min_delta= 0 , patience=10, verbose=1, mode="auto")
model.fit(
    noise_train, normal_train,
    epochs=n_epochs ,
    batch_size=batch_size,
    verbose= 1,
    validation_data=(noise_test, normal_test)
    )

model.save("./h5/denoise_model.h5")