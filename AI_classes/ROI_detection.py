import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, List, Set, Optional
from ultralytics import YOLO
import ultralytics
import math
from PIL import Image
from skimage import exposure

# Type alias for RGB color tuples
ColorRGB = Tuple[int, int, int]

class ROI_detection:
    """
    A class for detecting and processing Regions of Interest (ROIs) using YOLO model.
    
    Class Attributes:
        model_weights_path (str): Path to the trained YOLO model weights
        model (YOLO): Initialized YOLO model for ROI detection
    """
    model_weights_path: str = r'Model_weights\YOLO_ROIs_detection.pt'
    model: YOLO = YOLO(model_weights_path)
        
    def __init__(self) -> None:
        """Initialize ROI detection instance."""
        ...
    
    @staticmethod
    def contrast_adjust(im: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Adjust image contrast using percentile-based intensity rescaling.
        
        Args:
            im: Input image array
            
        Returns:
            Contrast-adjusted image array
        """
        p2, p98 = np.percentile(im, (1, 99))
        image_rescale = exposure.rescale_intensity(im, in_range=(p2, p98))
        return image_rescale

    @classmethod
    def preprocessing(cls, im: NDArray[np.uint8], threshold: float = 5) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """
        Preprocess image by resizing and applying blur if needed.
        
        Args:
            im: Original grayscale image
            threshold: Blur threshold value
            
        Returns:
            Tuple containing:
                - Resized image converted to BGR
                - Blurred version of resized image converted to BGR
        """
        im = cls.contrast_adjust(im)
        dim = cls.resize_shape_calculation(im)

        resized_im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
        blured_im = cls.blur_image(resized_im, threshold=threshold)
        
        return cv2.cvtColor(resized_im,cv2.COLOR_GRAY2BGR), cv2.cvtColor(blured_im,cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def blur_image(resized_im: NDArray[np.uint8], threshold: float) -> NDArray[np.uint8]:
        """
        Apply adaptive blur to image until blur level is below threshold.
        
        Args:
            resized_im: Input image to blur
            threshold: Target blur level threshold
            
        Returns:
            Blurred image meeting the threshold requirement
        """
        blur_level: float = cv2.Laplacian(resized_im, cv2.CV_64F).var()
        n: int = 3  # Initial kernel size
        blured_im = resized_im
        if threshold is not None:
            while blur_level >= threshold:
                blured_im = cv2.blur(blured_im,(n,n))
                blur_level = cv2.Laplacian(blured_im,cv2.CV_64F).var()
                n += 2
        return blured_im

    @staticmethod
    def resize_shape_calculation(im: NDArray[np.uint8]) -> Tuple[int, int]:
        """
        Calculate new dimensions while preserving aspect ratio, scaling to 640px max dimension.
        
        Args:
            im: Input image array
            
        Returns:
            Tuple of (width, height) for resized image
        """
        larger_dimension = max(im.shape[0], im.shape[1])
        scale_percent = 640/larger_dimension
        width = int(im.shape[1] * scale_percent)
        height = int(im.shape[0] * scale_percent)
        return (width, height)
    
    @staticmethod
    def square_size_calculation(
        top_left_x: int, 
        top_left_y: int, 
        botom_right_x: int, 
        botom_right_y: int, 
        Class: int
    ) -> Tuple[int, int, int, int]:
        """
        Calculate square bounding box dimensions based on ROI class.
        
        Args:
            top_left_x: X coordinate of top-left corner
            top_left_y: Y coordinate of top-left corner
            botom_right_x: X coordinate of bottom-right corner
            botom_right_y: Y coordinate of bottom-right corner
            Class: ROI class number
            
        Returns:
            Tuple of adjusted coordinates (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
            
        Raises:
            Exception: If Class is not in valid range (0-17)
        """
        open: Set[int] = {i for i in range(9)}
        close: Set[int] = {i for i in range(9,18)}
        height = botom_right_y - top_left_y
        width = botom_right_x - top_left_x
        multiplier = 1.2

        if Class in open:
            if height > width:
                new_width = height
                delta = new_width - width
                left_over = delta % 2
                delta = delta // 2
                top_left_x = top_left_x - delta
                botom_right_x = botom_right_x + delta + left_over
            else:
                new_height = width
                delta = new_height - height
                left_over = delta % 2
                delta = delta // 2
                top_left_y = top_left_y - delta
                botom_right_y = botom_right_y + delta + left_over
        elif Class in close:
            if height > width:
                new_height = int(width * multiplier)
                delta = height - new_height
                left_over = delta % 2
                delta = delta // 2
                top_left_y = top_left_y + delta
                botom_right_y = botom_right_y - delta - left_over

                new_width = int(width * multiplier)
                delta = new_width - width
                left_over = delta % 2
                delta = delta // 2
                top_left_x = top_left_x - delta
                botom_right_x = botom_right_x + delta + left_over
            else:
                new_width = int(height * multiplier)
                delta = width - new_width
                left_over = delta % 2
                delta = delta // 2
                top_left_x = top_left_x + delta
                botom_right_x = botom_right_x - delta - left_over

                new_height = int(height * multiplier)
                delta = new_height - height
                left_over = delta % 2
                delta = delta // 2
                top_left_y = top_left_y - delta
                botom_right_y = botom_right_y + delta + left_over
        else:
            raise Exception(f'The class {Class} should not exist')
        return top_left_x, top_left_y, botom_right_x, botom_right_y
    
    @staticmethod
    def square_size_validation(
        top_left_x: int,
        top_left_y: int,
        botom_right_x: int,
        botom_right_y: int,
        im_shape: Tuple[int, int, int]
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Validate and adjust ROI coordinates to fit within image boundaries.
        
        Args:
            top_left_x: X coordinate of top-left corner
            top_left_y: Y coordinate of top-left corner
            botom_right_x: X coordinate of bottom-right corner
            botom_right_y: Y coordinate of bottom-right corner
            im_shape: Image dimensions (height, width, channels)
            
        Returns:
            Tuple containing:
                - Validated coordinates (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
                - Desired shape (height, width)
                - Actual shape (height, width)
        """
        im_height, im_width, _ = im_shape
        ROI_desired_width = botom_right_x - top_left_x
        ROI_desired_heigth = botom_right_y - top_left_y
        
        # Clamp coordinates to image boundaries
        top_left_x = max(0, top_left_x)
        top_left_y = max(0, top_left_y)
        botom_right_x = min(im_width, botom_right_x)
        botom_right_y = min(im_height, botom_right_y)
        
        ROI_width = botom_right_x - top_left_x
        ROI_heigth = botom_right_y - top_left_y
        
        coordinates = (top_left_x, top_left_y, botom_right_x, botom_right_y)
        desired_shape = (ROI_desired_heigth, ROI_desired_width)
        real_shape = (ROI_heigth, ROI_width)
        return coordinates, desired_shape, real_shape

    @classmethod
    def crop_ROIs(
        cls,
        im: NDArray[np.uint8],
        data_dict: Dict[int, Tuple[float, NDArray[np.uint16]]]
    ) -> Tuple[List[Tuple[int, NDArray[np.uint8]]], Dict[int, Tuple[Tuple[int, int, int, int], Tuple[int, int], Tuple[int, int]]]]:
        """
        Extract ROI crops from the image based on detection results.
        
        Args:
            im: Input image array
            data_dict: Dictionary mapping class IDs to detection data
            
        Returns:
            Tuple containing:
                - List of (class_id, cropped_image) pairs
                - Dictionary mapping class IDs to ROI information
        """
        crops = len(data_dict)*[None]
        crops_info: Dict[int, Tuple[Tuple[int, int, int, int], Tuple[int, int], Tuple[int, int]]] = dict()
        im_shape = im.shape
        i = 0
        for Class, datum in data_dict.items():
            top_left_x, top_left_y, botom_right_x, botom_right_y = datum[1]
            top_left_x, top_left_y, botom_right_x, botom_right_y = cls.square_size_calculation(
                top_left_x, top_left_y, botom_right_x, botom_right_y, Class)
            coordinates, desired_shape, real_shape = cls.square_size_validation(
                top_left_x, top_left_y, botom_right_x, botom_right_y, im_shape)
            top_left_x, top_left_y, botom_right_x, botom_right_y = coordinates
            crops[i] = (Class, im[top_left_y:botom_right_y, top_left_x:botom_right_x])
            crops_info[Class] = (coordinates, desired_shape, real_shape)
            i += 1
        return crops, crops_info
    
    @classmethod
    def crops_prepocessing(
        cls,
        crops: List[Tuple[int, NDArray[np.uint8]]]
    ) -> List[Tuple[int, NDArray[np.uint8]]]:
        """
        Apply contrast adjustment to all ROI crops.
        
        Args:
            crops: List of (class_id, image) pairs
            
        Returns:
            List of (class_id, processed_image) pairs
        """
        preprocessed_crops = len(crops)*[None]
        for i, crop in enumerate(crops):
            Class, im = crop
            preprocessed_crops[i] = (Class, cls.contrast_adjust(im))
        return preprocessed_crops

    @classmethod
    def show_predicts_process(cls, im: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Process image and display detection results.
        
        Args:
            im: Input image array
            
        Returns:
            Image array with detection visualizations
        """
        resized_im, preprocessed_im = cls.preprocessing(im)
        ROI_result.result = cls.model(preprocessed_im)
        ROI_result.update()
        return ROI_result.show_predicts()

    @classmethod
    def total_process(
        cls,
        im: NDArray[np.uint8],
        do_angle_aligment: bool = False,
        threshold: float = 5,
        verbose: bool = True
    ) -> List[Tuple[int, NDArray[np.uint8]]]:
        """
        Complete ROI detection and processing pipeline.
        
        Args:
            im: Input image array
            do_angle_aligment: Whether to perform angle alignment
            threshold: Blur threshold value
            verbose: Whether to print processing details
            
        Returns:
            List of (class_id, processed_roi_image) pairs
        """
        resized_im, preprocessed_im = cls.preprocessing(im, threshold=threshold)
        ROI_result.result = cls.model(preprocessed_im, verbose=verbose)
        ROI_result.update()
        
        if do_angle_aligment:
            angle = ROI_result.angle_aligment_calculation()
            rotated_im = ROI_result.rotation_aligment(im, angle)
            resized_im, preprocessed_im = cls.preprocessing(rotated_im, threshold=threshold)
            ROI_result.result = cls.model(preprocessed_im)
            ROI_result.update()
            
        crops, crops_info = cls.crop_ROIs(resized_im, ROI_result.data_dict)
        preprocessed_crops = cls.crops_prepocessing(crops)
        return preprocessed_crops
    
class ROI_result:
    """
    A class for handling and visualizing YOLO model detection results.
    
    Class Attributes:
        result: Raw detection results from YOLO model
        original_image: Preprocessed input image
        names: Mapping of numeric labels to original class names
        shape: Image dimensions
        boxes: Bounding box information
        data: Detection data matrix
        data_dict: Processed detection results by class
        colors: Color mapping for visualization
    """
    result: Optional[ultralytics.engine.results.Results] = None
    original_image: Optional[NDArray[np.uint8]] = None
    names: Optional[Dict[int, str]] = None
    shape: Optional[Tuple[int, int]] = None
    boxes: Optional[ultralytics.engine.results.Boxes] = None
    data: Optional[NDArray[np.float32]] = None
    data_dict: Optional[Dict[int, Tuple[float, NDArray[np.uint16]]]] = None

    colors: Dict[int, ColorRGB] = {
        0:(0x38,0x38,0xFF),
        1:(0x97,0x9D,0xFF),
        2:(0xA8,0x99,0x2C),
        3:(0xFF,0xC2,0x00),
        4:(0x93,0x45,0x34),
        5:(0xFF,0x73,0x64),
        6:(0xEC,0x18,0x00),
        7:(0x10,0x38,0x84),
        8:(0x85,0x00,0x52),
        9:(0xFF,0x38,0xCB),
        10:(0x1F,0x70,0xFF),
        11:(0x1D,0xB2,0xFF),
        12:(0x31,0xD2,0xCF),
        13:(0x0A,0xF9,0x48),
        14:(0x17,0xCC,0x92),
        15:(0x86,0xDB,0x3D),
        16:(0x34,0x93,0x1A),
        17:(0xBB,0xD4,0x00)
    }

    def __init__(self) -> None:
        """Initialize ROI result instance."""
        ...

    @classmethod
    def update(cls) -> None:
        """
        Update class attributes based on latest detection results.
        
        Processes raw YOLO detection results and updates all relevant class attributes
        including original image, class names, boxes, and processed detection data.
        """
        cls.result = cls.result[0]
        cls.original_image = cls.result.orig_img
        cls.names = cls.result.names
        cls.shape = cls.result.orig_shape
        cls.boxes = cls.result.boxes.numpy()
        cls.data = cls.boxes.data
        cls.data = cls.original_enumeration()
        cls.data_dict = cls.bounding_box_filter()

    @classmethod
    def bounding_box_filter(cls) -> Dict[int, Tuple[float, NDArray[np.uint16]]]:
        """
        Filter multiple detections to keep only highest confidence detection per class.
        
        For each ROI class, retains only the bounding box with the highest confidence
        score, as there should be at most one instance of each ROI.
        
        Returns:
            Dictionary mapping class IDs to tuples of (confidence_score, bounding_box_coordinates)
        """
        data = cls.data
        data_dict: Dict[int, Tuple[float, NDArray[np.uint16]]] = dict()
        
        for datum in data:
            Class = int(datum[-1])
            conf = float(datum[-2])
            bounding_box = datum[:4].astype(np.uint16)
            if Class not in data_dict or data_dict[Class][0] < conf:
                data_dict[Class] = (conf, bounding_box)
        return data_dict

    @classmethod
    def original_enumeration(cls) -> NDArray[np.float32]:
        """
        Convert numeric labels back to original class names.
        
        Returns:
            Detection data array with original class names
        """
        data = cls.data
        names = cls.names
        for box_data in data:
            box_data[-1] = names[int(box_data[-1])]  # change to the original enumeration
        return data

    @classmethod
    def angle_aligment_calculation(cls) -> float:
        """
        Calculate angle needed to align hand vertically based on key ROI positions.
        
        Uses ROIs 14 and 0 as reference points to determine the rotation angle
        needed for vertical alignment.
        
        Returns:
            Calculated rotation angle in degrees, or 0 if reference ROIs are not found
        """
        data_dict = cls.data_dict
        if 14 in data_dict and 0 in data_dict:
            bounding_box_1 = data_dict[1][1]
            bounding_box_14 = data_dict[14][1]
            centroid_x_1, centroid_y_1 = cls.centroid(bounding_box_1)
            centroid_x_14, centroid_y_14 = cls.centroid(bounding_box_14)
            angle = math.atan2(-(centroid_y_14-centroid_y_1), centroid_x_14-centroid_x_1)*180/math.pi
            return angle
        return 0
        
    @staticmethod
    def rotation_aligment(im: NDArray[np.uint8], angle: float) -> NDArray[np.uint8]:
        """
        Rotate image to align with vertical axis.
        
        Args:
            im: Input image array
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image array
        """
        return np.array(Image.fromarray(im).rotate(angle=-angle+90, resample=Image.BICUBIC, expand=True))

    @staticmethod
    def centroid(coordinates: NDArray[np.uint16]) -> Tuple[int, int]:
        """
        Calculate centroid of a bounding box.
        
        Args:
            coordinates: Array of box coordinates [x1, y1, x2, y2]
            
        Returns:
            Tuple of (x, y) coordinates of the centroid
        """
        x = (coordinates[0] + coordinates[2]) // 2
        y = (coordinates[1] + coordinates[3]) // 2
        return (x, y)
    
    @classmethod
    def show_predicts(cls, image: Optional[NDArray[np.uint8]] = None) -> NDArray[np.uint8]:
        """
        Visualize detection results on image.
        
        Draws bounding boxes, class labels, and confidence scores for all detected ROIs.
        
        Args:
            image: Optional input image array. If None, uses original image.
            
        Returns:
            Image array with visualization overlays
        """
        colors = cls.colors
        if image is None:
            predicts_im = cls.original_image.copy()
        else:
            predicts_im = image.copy()

        for Class, datum in cls.data_dict.items():
            top_left = datum[1][:2]
            botom_right = datum[1][2:]
            # Draw bounding box
            cv2.rectangle(predicts_im, 
                        tuple(top_left), 
                        tuple(botom_right),
                        colors[Class],
                        thickness=1,
                        lineType=cv2.LINE_AA)
            
            # Add text label
            text = f'{Class} {datum[0]:.2f}'
            fondScale = 0.6
            size, _ = cv2.getTextSize(text, fontFace=0, fontScale=fondScale, thickness=1)
            
            # Draw text background
            cv2.rectangle(predicts_im,
                        tuple(top_left - np.array((0, size[1]+2))),
                        tuple(top_left + np.array((size[0], 0))),
                        colors[Class],
                        thickness=-1,
                        lineType=cv2.LINE_AA)
            
            # Draw text
            cv2.putText(predicts_im,
                       text,
                       (top_left[0], top_left[1]-2),
                       fontFace=0,
                       fontScale=fondScale,
                       color=(255,255,255),
                       thickness=1,
                       lineType=cv2.LINE_AA)
            
        return predicts_im