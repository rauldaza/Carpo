from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
import os
import cv2
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

class ROI_image_creation:
    """
    A class for processing and creating Region of Interest (ROI) images.
    ROIs are specific areas of interest in images that need to be processed separately.
    """
    
    ROI_shape: int = 60

    @staticmethod
    def sort_ROIs(crops: List[Tuple[int, NDArray[np.uint8]]]) -> List[NDArray[np.uint8]]:
        """
        Sorts and organizes ROI crops into a standardized order of 18 positions.
        
        Args:
            crops: List of tuples containing (class_number, image_crop)
        
        Returns:
            List of 18 images, with empty (1,1) arrays for missing ROIs
        """
        sorted_crops = sorted(crops, key=lambda crops: crops[0])
        ROI_images: List[NDArray[np.uint8]] = list()
        pointer = 0
        crop_class = -1
        ROI_order = [i for i in range(18)]

        for ROI_number in ROI_order:
            if pointer < len(sorted_crops):
                crop_class = sorted_crops[pointer][0]
            if ROI_number != crop_class:
                ROI_images.append(np.zeros((1,1), dtype=np.uint8))
            else:
                ROI_images.append(sorted_crops[pointer][1])
                pointer += 1
        return ROI_images

    @staticmethod
    def one_channel_ROI_images(ROI_images: List[NDArray[np.uint8]]) -> List[NDArray[np.uint8]]:
        """
        Converts all RGB ROI images to grayscale (single channel).
        
        Args:
            ROI_images: List of ROI images
            
        Returns:
            List of grayscale ROI images
        """
        for i in range(0,18):
            if ROI_images[i].ndim == 3:
                grayscale_img = cv2.cvtColor(ROI_images[i], cv2.COLOR_BGR2GRAY)
                ROI_images[i] = grayscale_img
        return ROI_images

    @classmethod
    def image_ROIs(cls, crops: List[Tuple[int, NDArray[np.uint8]]]) -> NDArray[np.uint8]:
        """
        Creates a single image containing all ROIs arranged in a 6x3 grid.
        Each ROI is resized to standard dimensions and padded if necessary.
        
        Args:
            crops: List of tuples containing (class_number, image_crop)
            
        Returns:
            A single numpy array containing all ROIs arranged in a grid
        """
        ROI_images = cls.sort_ROIs(crops)
        ROI_images = cls.one_channel_ROI_images(ROI_images)

        ROI_concatenated = np.zeros((0,3*cls.ROI_shape), dtype=np.uint8)
        for i in range(6):
            row = np.zeros((cls.ROI_shape,0), dtype=np.uint8)
            for j in range(3):
                ROI_image = ROI_images[3*i + j]
                height, width = ROI_image.shape
                if height > width:
                    width = height
                else:
                    height = width
                black_image = np.zeros((height,width), dtype=np.uint8)
                height, width = ROI_image.shape
                black_image[0:height, 0:width] = ROI_image
                black_image_resized = cv2.resize(black_image, (cls.ROI_shape, cls.ROI_shape), interpolation = cv2.INTER_AREA)
                row = np.concatenate((row,black_image_resized), axis=1)
            ROI_concatenated = np.concatenate((ROI_concatenated,row), axis=0)
        return ROI_concatenated

    @classmethod
    def stack_ROIs(cls, crops: List[Tuple[int, NDArray[np.uint8]]]) -> NDArray[np.uint8]:
        """
        Creates a 3D array by stacking all ROIs along the third dimension.
        Each ROI is resized to standard dimensions and padded if necessary.
        
        Args:
            crops: List of tuples containing (class_number, image_crop)
            
        Returns:
            A 3D numpy array with shape (ROI_shape, ROI_shape, 18)
        """
        ROI_images = cls.sort_ROIs(crops)
        ROI_images = cls.one_channel_ROI_images(ROI_images)
       
        ROI_concatenated: List[NDArray[np.uint8]] = list()
        for i in range(18):
            ROI_image = ROI_images[i]
            height, width = ROI_image.shape
            if height > width:
                width = height
            else:
                height = width
            black_image = np.zeros((height,width), dtype=np.uint8)

            height, width = ROI_image.shape
            black_image[0:height, 0:width] = ROI_image
            black_image_resized = cv2.resize(black_image, (cls.ROI_shape, cls.ROI_shape), interpolation = cv2.INTER_AREA)
            ROI_concatenated.append(black_image_resized)
        return np.stack(ROI_concatenated, axis=2)
    

@register_keras_serializable()
def identity(x): # A custom layer that does nothing, used for deployment to replace the custom layer used in training
    return x

class BAA_prediction:
    """
    A class for making Bone Age Assessment (BAA) predictions using a pre-trained model.
    """
    
    @staticmethod
    def load_image_from_array(image, sex_label):
        """
        Converts a NumPy array (e.g., in uint8 [0,255]) to a normalized tf.Tensor in [0,1] range,
        and returns it along with the provided sex_label.
        """

        if len(image.shape) == 2:  # If shape is (H, W)
            image = tf.expand_dims(image, axis=-1)  # Convert to (H, W, 1)
        # Convert the numpy array to a tensor
        image = tf.convert_to_tensor(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.grayscale_to_rgb(image)
        image = image / 255.0  # Normalize the image to [0, 1] range
        return image, sex_label

    model: tf.keras.Model = tf.keras.models.load_model(
    r"Model_weights/deployed_model.keras",
    custom_objects={"identity": identity}) # Load the model, for training it has a custom layer, in deployment it is not needed so is
    # replaced by the identity function

    @classmethod
    def predict_BAA(cls, image: NDArray[np.uint8], sex_list: list) -> NDArray[np.int64]:
        """
        Makes a bone age prediction using the loaded model.
        
        Args:
            image: Input image to be classified
            
        Returns:
            Predicted class index for bone age assessment
        """

        image, sex_list = cls.load_image_from_array(image, sex_list)

        image = np.expand_dims(image, axis=0)  # Now shape: (1, H, W, C)

        sex_np = np.array([sex_list])

        # Now pass both inputs as a list to predict:
        prediction = cls.model.predict([image, sex_np])
        prediction = int(prediction[0,0]) 

        # limits the prediction to the range of 0 to 228
        if prediction < 0:
            prediction = 0
        elif prediction > 228:
            prediction = 228

        return prediction