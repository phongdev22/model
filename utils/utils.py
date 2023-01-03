import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#----------------------------VISUALIZATION------------------------------------
        
def draw_boxes(img , boxes, thickness = 5, color=(255, 0, 0)):
        image = img
        for box in boxes:
            image = cv2.rectangle(img, (box[0],box[1]),(box[2],box[3]),color,thickness)
        return image

def data_aug():
    return

def overlay_plot(img, mask):
    plt.figure()
    plt.imshow(img.T, 'gray', interpolation='none')
    plt.imshow(mask.T, 'jet', interpolation='none', alpha=0.5)

def display_training_curves(training, validation, title, subplot):
    ax = plt.subplot(subplot)
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    ax.set_xlabel('epoch')
    ax.legend(['training', 'validation'])

def predict_segmentation(model_path, path):

    model = tf.keras.models.load_model(model_path)

    img = cv2.imread(path)  

    img = cv2.resize(img, (512, 512), cv2.INTER_LINEAR)

    img_expand = img[np.newaxis, ...]

    img_pred = model.predict(img_expand).reshape(512, 512)

    img_pred[img_pred < 0.5] = 0
    img_pred[img_pred >= 0.5] = 1

    return img_pred
