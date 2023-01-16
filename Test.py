import os
import cv2
import numpy as np
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, confusion_matrix
import numpy as np
from keras.callbacks import  ModelCheckpoint
from util import *
import scipy.misc as mc
import  math
import time
import datetime
from skimage.io import imread
import cv2
start_time = time.time()
print ("Start: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
import os
from keras.models import model_from_json
from keras.models import Model
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import configparser
from keras_drop_block import DropBlock2D
from keras import backend as K




config = configparser.RawConfigParser()
config.read('configuration.txt')
name_experiment = config.get('experiment name', 'name')
path_experiment = './' +name_experiment +'/'




data_location = ''

testing_images_loc = data_location + 'DRIVE/test/images/'
testing_label_loc = data_location + 'DRIVE/test/labels/'
test_files = os.listdir(testing_images_loc)
test_data = []
test_label = []
desired_size = 592
for i in test_files:
    im = imread(testing_images_loc + i)
    label = imread(testing_label_loc + i.split('_')[0] + '_manual1.png',pilmode="L")
    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color2)

    test_data.append(cv2.resize(new_im, (desired_size, desired_size)))

    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    test_label.append(temp)

test_data = np.array(test_data)
test_label = np.array(test_label)
x_test = test_data.astype('float32') / 255.

x_test = np.reshape(x_test, (len(x_test), desired_size, desired_size, 3))
y_test = test_label.astype('float32') / 255.
y_test = np.reshape(y_test, (len(y_test), desired_size, desired_size, 1))  # adapt this if using `channels_first`

y_test=crop_to_shape(y_test,(len(y_test), 584, 565, 1))


best_last = config.get('testing settings', 'best_last')
#Load the saved model
#model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read(), custom_objects={'DropBlock2D': DropBlock2D})
model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')
#Calculate the predictions
y_pred = model.predict(x_test)
y_pred= crop_to_shape(y_pred,(20,584,565,1))
y_pred_threshold = []
i=0
for y in y_pred:
    _, temp = cv2.threshold(y, 0.5, 1, cv2.THRESH_BINARY)
    y_pred_threshold.append(temp)
    y = y * 255
    cv2.imwrite('./DRIVE/test/result/%d.png' % i, y)
    i+=1

y_test = list(np.ravel(y_test))
y_pred_threshold = list(np.ravel(y_pred_threshold))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()

# distances
def iou_accuracy(y_true, y_pred):
    intersection = y_true * y_pred
    union = y_true + ((1. - y_true) * y_pred)
    return K.sum(intersection) / K.sum(union)


def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return 1-((1 - jac) * smooth)


def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection +smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) +smooth)


#losses
def iou_loss(y_true, y_pred):

    return 1 - iou_accuracy(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):

    return 1 - dice_coef(y_true, y_pred)


print('Accuracy:', accuracy_score(y_test, y_pred_threshold))
print("IOU", iou_accuracy(y_test, y_pred_threshold))
print("Jacc", jaccard_distance(y_test, y_pred_threshold))
print("Dice", dice_coef(y_test, y_pred_threshold))
print('Sensitivity:', recall_score(y_test, y_pred_threshold))
print('Specificity:', tn / (tn + fp))
print('NPV:', tn / (tn + fn))
#print("JS:", jaccard_score(y_test, list(np.ravel(y_pred))))
print('PPV', tp / (tp + fp))
print('AUC:', roc_auc_score(y_test, list(np.ravel(y_pred))))
print("F1:",2*tp/(2*tp+fn+fp))
N=tn+tp+fn+fp
S=(tp+fn)/N
P=(tp+fp)/N
print("MCC:",(tp/N-S*P)/math.sqrt(P*S*(1-S)*(1-P)))


file_perf = open(path_experiment+'performances.txt', 'w')
file_perf.write("Area under the ROC curve: "+str(roc_auc_score)
                + "\nF1 score (F-measure): " +str(2*tp/(2*tp+fn+fp))
                +"\nACCURACY: " +str(accuracy_score)
                +"\nSENSITIVITY: " +str(recall_score)
                +"\nSPECIFICITY: " +str(tn / (tn + fp))
                +"\nMCC" +str((tp/N-S*P)/math.sqrt(P*S*(1-S)*(1-P)))
                )
file_perf.close()

