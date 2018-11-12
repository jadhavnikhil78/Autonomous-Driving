import numpy as np
import cv2
from tqdm import tqdm
import time
from scipy import signal as signal_sci, optimize
import signal
import neural_network as ann

## Pre-process input images
def train(path_to_images, csv_file):
    '''s
        Method to perform preprocessing on input images.
        Args:
        path_to_images = path to jpg image files
        csv_file = path and filename to csv file containing frame numbers and steering angles.
        Returns:
        NN = Trained Neural Network object
        '''
    
    # Import Steering Angles CSV
    data = np.genfromtxt(csv_file, delimiter = ',')
    frame_nums = data[:,0]
    steering_angles = data[:,1]
    
    im_train_X = []
    for frame_number in range(len(frame_nums)):
        temp = cv2.imread(path_to_images + '/' + str(int(frame_number)).zfill(4) + '.jpg')
        temp_resized = cv2.resize(temp, (60, 64))
        temp_crop = crop_center(temp_resized)
        temp_lane = select_rgb_white_yellow(temp_crop)
        temp_gray = convert_to_grayscale(temp_lane)
        im_train_X.append(list(temp_gray.ravel()))

    rawangles = steering_angles
    steering_angles = signal_sci.savgol_filter(steering_angles, 3, 2)
    encoded_angles = encoder(steering_angles)

    trainX = np.array(im_train_X)
    trainY = encoded_angles
    trainX = trainX / 255
    
    NN = ann.Neural_Network(rawangles)
    T = ann.trainer(NN)
    
    T.train(trainX,trainY)
    
    return NN

##Turning angle prediction
def predict(NN, image_file):
    '''
        Method to predict the turning angle.
        Given an image filename, load image, make and return predicted steering angle in degrees.
        '''
    im_full = cv2.imread(image_file)
    
    temp_resized = cv2.resize(im_full, (60, 64))
    temp_crop = crop_center(temp_resized)
    temp_lane = select_rgb_white_yellow(temp_crop)
    temp_gray = convert_to_grayscale(temp_lane)
    
    temp_gray = temp_gray / 255
    testX = temp_gray.ravel()
    testY = NN.angles
    final_angle = decoder(NN.forward(testX), testY)
    
    return final_angle

def convert_to_grayscale(im):
    '''
        Convert color image to grayscale.
        Args: im = (nxmx3) floating point color image scaled between 0 and 1
        Returns: (nxm) floating point grayscale image scaled between 0 and 1
        '''
    return np.mean(im, axis = 2)

# image is expected be in RGB color space
def select_rgb_white_yellow(image):
    # white color mask
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([190, 190,   0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

def crop_center(img):
    return img[25:,:]

## Normalize input angles
def encoder(steering_angles):
    encoded_angles = []
    my_bins = np.histogram(steering_angles, bins=64)[1]
    for i in range(len(steering_angles)):
        encode = np.zeros(64)
        j = int(np.digitize(steering_angles[i], bins=my_bins))
        if j > 1:
            j = j - 2
        else:
            j = j - 1
        encode[j] = 0.5
        padding = [0.05, 0.2]
        if j == 0:
            encode[j+1:j+3] = padding[::-1]
        elif j == 1:
            encode[0] = padding[-1]
        elif j == 63:
            encode[j-2:j] = padding
        elif j == 62:
            encode[j-2:j] = padding
            encode[j+1] = padding[-1]
        else:
            encode[j-2:j] = padding
            encode[j+1:j+3] = padding[::-1]
        encoded_angles.append(encode)
    return encoded_angles

## De-normalize the predicted angles
def decoder(image, angles):
    if type(image) is list:
        result = []
        for img in image:
            j = np.argmax(img) + 1
            my_bins = np.histogram(angles, bins=64)[1]
            angle = my_bins[j]
            result.append(angle)
        return result
    else:
        j = np.argmax(image) + 1
        my_bins = np.histogram(angles, bins=64)[1]
        angle = my_bins[j]
        return angle
