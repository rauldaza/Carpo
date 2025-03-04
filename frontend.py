import tensorflow as tf
import customtkinter as ctk
import CTkMenuBar as CTkMenu
from tkcalendar import DateEntry
from PIL import Image
from customtkinter import filedialog
import os
from AI_classes.ROI_detection import ROI_detection
from AI_classes.BAA_prediction import BAA_prediction, ROI_image_creation
import datetime
import numpy as np
import re
import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import generate_uid
import sys

sys.stderr = open("error_log.txt", "w")
sys.stdout = open("output_log.txt", "w")


names = None # Array of the names of the patients
heights = None # Array of the heights of the patients
weights = None # Array of the weights of the patients
sexes = None # Array of the sexes of the patients
ethnicities = None # Array of the ethnicities of the patients
birthdays = None # Array of the birthdays of the patients
corrected_BAA = None # Array of the corrected Bone Age Assessment
BAA_predictions = None # Array of the Bone Age Assessment predictions
images_crops = None # Array of the crops of ROIs found in the images
images = list() # Array of the carpograms
ROI_images = list() # Array of the ROIs of the images
image_counter = 0 # Counter of the current image
is_images_loaded = False # Flag to know if the images are loaded
is_ROIs_calculated = False # Flag to know if the ROIs are calculated
is_BAAs_calculated = False # Flag to know if the BAAs are calculated
total_image_paths = None # Array of the paths of the images
today_year, today_month, today_day = datetime.date.today().year, datetime.date.today().month, datetime.date.today().day # today's date
year_offset = 4*5 # every 4 years is a leap year
default_date = datetime.date(today_year+year_offset, today_month, today_day) 
frame_counter = 0 # represents the view. 0: Cargar carpeta/imagenes, 1: imagenes cargadas, 2: TW ROIs, 3: Prediccion BAA
ROI_counter = 0 # Counter of the current ROI
ROI_names = ['Radio', 'Grande', 'Granchoso', 'Pisiforme-Piramidal', 'Semilunar', 'Escafoides', 'Trapecio', 'Trapezoide',     
             'Cúbito', 'Epifisis Carpo-Metacarpo I', 'Epifisis Metacarpo-Falange proximal I',
               'Epifisis Falange proximal-Falange distal I', 'Epifisis Metacarpo-Falange proximal III', 
               'Epifisis Falange proximal-Falange media III', 'Epifisis Falange media-Falange distal III', 
               'Epifisis Metacarpo-Falange proximal V', 'Epifisis Falange proximal-Falange media V', 
               'Epifisis Falange media-Falange distal V'] # Names of the ROIs

def load_images_view_func():
    """
    Switches the view to the initial 'load images' screen.
    - Clears any currently displayed frames based on frame_counter
    - Resets to frame_counter 0 (initial view)
    - Deletes all loaded images and patient info
    - Sets up the main left and right frames with load buttons
    """
    global frame_counter

    # Deletes every frame already displayed based on current view
    if frame_counter == 1:
        images_loaded_right_frame.grid_forget()
        images_loaded_left_frame.grid_forget()
    elif frame_counter == 2:
        TW_ROIs_right_frame.grid_forget()
        TW_ROIs_left_frame.grid_forget()
    elif frame_counter == 3:
        Predictions_right_frame.grid_forget()
        Predictions_left_frame.grid_forget()

    # Reset to initial view state
    frame_counter = 0
    delete_images_func()
    delete_patients_info()

    # Configure and display main frames
    left_arguments = dict(row=0,column=0,sticky='NSWE',padx=5)
    right_arguments = dict(row=0,column=1,sticky='NSWE',padx=5)
    main_left_frame.grid(**left_arguments)
    main_right_frame.grid(**right_arguments)

    main_left_frame.rowconfigure((0,3),weight=1)
    main_left_frame.columnconfigure(0,weight=1)

    main_right_frame.columnconfigure(0,weight=1)
    load_images.grid(row=1,column=0,pady=10)
    load_folder.grid(row=2,column=0,pady=10)

def load_images_func(is_folder, is_append):
    """
    Handles loading images from either individual files or a folder.
    
    Args:
        is_folder (bool): True if loading from folder, False if loading individual files
        is_append (bool): True to append to existing images, False to replace
    
    This function:
    - Manages image loading based on current view state
    - Handles file/folder selection via dialog
    - Updates global image lists and patient data arrays
    - Updates UI to show loaded images
    - Initializes patient data structures if needed
    """
    global is_images_loaded, left_arguments, right_arguments, images, frame_counter, image_counter
    global names, heights, weights, sexes, ethnicities, birthdays, total_image_paths
    
    images_path = []
    # Only allow loading in views 0 or 1
    if frame_counter == 0 or frame_counter == 1:
        if not is_append:
            images.clear()
            image_counter = 0
            delete_patients_info()
            
        # Handle folder selection
        if is_folder:
            folder_name = filedialog.askdirectory(title='Seleccione la carpeta')
            if folder_name:
                listed_files = os.listdir(folder_name)
                listed_files = [os.path.join(folder_name,listed_file) for listed_file in listed_files]
                images_path = listed_files
        # Handle individual file selection
        else:
            files_names = filedialog.askopenfilenames(title='Seleccione las imagenes')
            if files_names:
                images_path = files_names

        # Process selected images
        if images_path:
            if total_image_paths is None:
                total_image_paths = list(images_path)
            else:
                total_image_paths.extend(images_path)

            # Load images and create CTkImage objects
            images.extend([ctk.CTkImage(light_image=Image.open(path).convert('L'),
                                      dark_image=Image.open(path).convert('L'),
                                      size=(350,450)) for path in images_path])

            # Extend patient data arrays if needed
            if names is not None:
                names += [None]*(len(images)-len(names))
                heights += [None]*(len(images)-len(heights))
                weights += [None]*(len(images)-len(weights))
                sexes += [None]*(len(images)-len(sexes))
                ethnicities += [None]*(len(images)-len(ethnicities))
                birthdays += [None]*(len(images)-len(birthdays))

            # Update label if in images loaded view
            if frame_counter == 1:
                radiograph_label.configure(text=f'{image_counter+1} de {len(images)}')

    # Switch to images loaded view if in initial view and images exist
    if frame_counter == 0 and images:
        is_images_loaded = True
        frame_counter = 1
        
        # Initialize patient data arrays
        names = [None]*len(images)
        heights = [None]*len(images)
        weights = [None]*len(images)
        sexes = [None]*len(images)
        ethnicities = [None]*len(images)
        birthdays = [None]*len(images)

        # Update UI layout
        main_right_frame.grid_forget()
        main_left_frame.grid_forget()

        images_loaded_left_frame.grid(**left_arguments)
        images_loaded_right_frame.grid(**right_arguments)

        images_loaded_right_frame.columnconfigure((0,1),weight=1)
        images_loaded_left_frame.columnconfigure((0,1),weight=1)

        # Configure initial image display
        hand_radiograph.configure(image=images[0],text='')
        radiograph_label.configure(text=f'{image_counter+1} de {len(images)}')

        # Grid all UI elements
        TW_ROIs.grid(row=0,column=0,pady=10,columnspan=2)
        baa_prediction.grid(row=1,column=0,pady=10,columnspan=2)
        name.grid(row=2,column=0,pady=10,columnspan=2)
        height.grid(row=3,column=0,pady=10,columnspan=2)
        weight.grid(row=4,column=0,pady=10,columnspan=2)
        male.grid(row=5,column=0,pady=10,sticky='E')
        female.grid(row=5,column=1,pady=10,sticky='W')
        ethnicity.grid(row=6,column=0,pady=10,columnspan=2)
        age.grid(row=7,column=0,pady=10,columnspan=2)
        
        ethnicity.set('Etnia')

        hand_radiograph.grid(row=0,column=0,columnspan=2)
        previous_image.grid(row=1,column=0,padx=5,pady=5,sticky='E')
        next_image.grid(row=1,column=1,padx=5,pady=5,sticky='W')
        radiograph_label.grid(row=2,column=0,columnspan=2)

        # Reset entry fields
        patient_name, patient_height, patient_weight, _, _, _ = get_entries() 
        reset_entries(patient_name, patient_height, patient_weight)

def delete_images_func():
    """
    Clears all loaded images and related data.
    - Clears image lists
    - Resets flags for ROI and BAA calculations
    - Clears path information
    """
    global is_images_loaded, frame_counter, is_ROIs_calculated, total_image_paths, corrected_BAA, is_BAAs_calculated
    if frame_counter == 0:
        images.clear()
        ROI_images.clear()
        if frame_counter == 1:
            frame_counter = 0
        is_images_loaded = False
        is_ROIs_calculated = False
        is_BAAs_calculated = False
        total_image_paths = None
        corrected_BAA = None

def delete_patients_info():
    """
    Resets all patient-related information arrays to None.
    Includes demographic data and BAA predictions.
    """
    global names, heights, weights, sexes, ethnicities, birthdays, corrected_BAA, BAA_predictions
    names = None
    heights = None
    weights = None
    sexes = None
    ethnicities = None
    birthdays = None
    corrected_BAA = None
    BAA_predictions = None

def change_image_func(is_next):
    """
    Handles navigation between images in the dataset.
    
    Args:
        is_next (bool): True to move to next image, False for previous
        
    - Stores current patient data before changing images
    - Updates image counter
    - Updates UI with new image and associated data
    - Handles different views (patient info, ROI, predictions)
    """
    global image_counter, images, hand_radiograph, radiograph_label, ROI_counter

    # Store current data if in patient info view
    if frame_counter == 1:
        patient_name, patient_height, patient_weight, patient_sex, patient_ethnicity, patient_birthday = get_entries()
        reset_entries(patient_name, patient_height, patient_weight)
        store_entries(patient_name, patient_height, patient_weight, patient_sex, patient_ethnicity, patient_birthday)
    
    # Store BAA data if in predictions view
    if frame_counter == 3:
        patient_BAA = get_BAA_entry()
        reset_BAA_entry(patient_BAA)
        store_BAA_entry(patient_BAA)
    
    # Update image counter with bounds checking
    image_counter += 1 if is_next else -1
    if image_counter == len(images):
        image_counter -= 1
    elif image_counter == -1:
        image_counter += 1

    # Update UI based on current view
    if frame_counter == 1: 
        display_entries_info(image_counter)

    if frame_counter == 2:
        ROI_counter = 0
        ROI_label.configure(text=f'{ROI_names[ROI_counter]}')
        ROI_radiograph.configure(image=ROI_images[image_counter][ROI_counter])
        ROI_radiograph_label.configure(text=f'{ROI_counter+1} de {len(ROI_images[image_counter])}')

    if frame_counter == 3:
        display_BAA_entry_info(image_counter)
        name_label.configure(text=f'Nombre: {names[image_counter]}')
        height_label.configure(text=f'Altura: {heights[image_counter]} cm')
        weight_label.configure(text=f'Peso: {weights[image_counter]} kg')
        sex_label.configure(text=f'Sexo: {sexes[image_counter]}')
        ethnicity_label.configure(text=f'Etnia: {ethnicities[image_counter]}')
        age_label.configure(text=f'Fecha de nacimiento: {birthdays[image_counter]}')
        BAA_label.configure(text=f'Edad ósea: {BAA_predictions[image_counter]}')

    # Update image display
    radiograph_label.configure(text=f'{image_counter+1} de {len(images)}')
    hand_radiograph.configure(image=images[image_counter])

def change_ROI_func(is_next):
    """
    Navigates between different ROIs (Regions of Interest) for the current image.
    
    Args:
        is_next (bool): True to move to next ROI, False for previous
        
    Updates ROI display and labels while maintaining bounds checking.
    """
    global ROI_counter, image_counter
    ROI_counter += 1 if is_next else -1
    if ROI_counter == len(ROI_images[image_counter]):
        ROI_counter -= 1
    elif ROI_counter == -1:
        ROI_counter += 1

    ROI_label.configure(text=f'{ROI_names[ROI_counter]}')
    ROI_radiograph.configure(image=ROI_images[image_counter][ROI_counter])
    ROI_radiograph_label.configure(text=f'{ROI_counter+1} de {len(ROI_images[image_counter])}')    

def TW_ROIs_func():
    """
    Switches to the TW ROIs (Tanner-Whitehouse Regions of Interest) view.
    - Calculates ROIs if not already done
    - Stores current patient data
    - Updates UI layout for ROI display
    - Initializes ROI navigation
    """
    global frame_counter, image_counter, is_images_loaded
    
    ROIs_array()

    if is_images_loaded:
        if frame_counter == 1:
            # Store current patient data
            patient_name, patient_height, patient_weight, patient_sex, patient_ethnicity, patient_birthday = get_entries()
            if patient_birthday == default_date:
                patient_birthday = None
            if patient_ethnicity == 'Etnia':
                patient_ethnicity = None
            store_entries(patient_name, patient_height, patient_weight, patient_sex, patient_ethnicity, patient_birthday)

            images_loaded_left_frame.grid_forget()
            images_loaded_right_frame.grid_forget()
        elif frame_counter == 3:
            Predictions_left_frame.grid_forget()
            Predictions_right_frame.grid_forget()

        # Initialize ROI view
        frame_counter = 2
        image_counter = 0
        ROI_counter = 0

        # Update UI layout
        TW_ROIs_left_frame.grid(**left_arguments)
        TW_ROIs_right_frame.grid(**right_arguments)

        # Configure initial display
        hand_radiograph.configure(image=images[0],text='')
        radiograph_label.configure(text=f'{image_counter+1} de {len(images)}')

        ROI_label.configure(text=f'{ROI_names[ROI_counter]}')
        ROI_radiograph.configure(image=ROI_images[image_counter][ROI_counter],text='')
        ROI_radiograph_label.configure(text=f'{ROI_counter+1} de {len(ROI_images[image_counter])}')

        # Configure grid layout
        TW_ROIs_left_frame.columnconfigure((0,1),weight=1)
        TW_ROIs_right_frame.columnconfigure((0,1),weight=1)

        # Place ROI UI elements
        ROI_label.grid(row=0,column=0,columnspan=2)
        ROI_radiograph.grid(row=1,column=0,columnspan=2)
        previous_ROI.grid(row=2,column=0,padx=5,pady=5,sticky='E')
        next_ROI.grid(row=2,column=1,padx=5,pady=5,sticky='W')
        ROI_radiograph_label.grid(row=3,column=0,columnspan=2)
        save_ROIs.grid(row=4,column=0,columnspan=2)

def baa_prediction_func():
    global frame_counter, is_images_loaded, corrected_BAA, image_counter

    if corrected_BAA is None:
        corrected_BAA = [None]*len(images)

    ROIs_array()
    BAAs_array()

    if is_images_loaded:
        if frame_counter == 1:
            # gets the last data in the entries
            patient_name, patient_height, patient_weight, patient_sex, patient_ethnicity, patient_birthday = get_entries()
            # prevents to catch the default info
            if patient_birthday == default_date:
                patient_birthday = None
            if patient_ethnicity == 'Etnia':
                patient_ethnicity = None
            store_entries(patient_name, patient_height, patient_weight, patient_sex, patient_ethnicity, patient_birthday)

            images_loaded_left_frame.grid_forget()
            images_loaded_right_frame.grid_forget()
        elif frame_counter == 2:
            TW_ROIs_left_frame.grid_forget()
            TW_ROIs_right_frame.grid_forget()
        
        frame_counter = 3
        image_counter = 0

        Predictions_left_frame.grid(**left_arguments)
        Predictions_right_frame.grid(**right_arguments)

        Predictions_right_frame.columnconfigure(0,weight=1)

        hand_radiograph.configure(image=images[0],text='')
        radiograph_label.configure(text=f'{image_counter+1} de {len(images)}')

        ctk.CTkLabel(master=Predictions_right_frame,text='RESULTADOS',height=40,font=('Segoe UI',20)).grid(row=0,column=0)
        name_label.grid(row=1,column=0)
        height_label.grid(row=2,column=0)
        weight_label.grid(row=3,column=0)
        sex_label.grid(row=4,column=0)
        ethnicity_label.grid(row=5,column=0)
        age_label.grid(row=6,column=0)
        BAA_label.grid(row=7,column=0)
        BAA_entry.grid(row=8,column=0,pady=5)
        save_results.grid(row=9,column=0,pady=10)

        name_label.configure(text=f'Nombre: {names[image_counter]}')
        height_label.configure(text=f'Altura: {heights[image_counter]} cm')
        weight_label.configure(text=f'Peso: {weights[image_counter]} kg')
        sex_label.configure(text=f'Sexo: {sexes[image_counter]}')
        ethnicity_label.configure(text=f'Etnia: {ethnicities[image_counter]}')
        age_label.configure(text=f'Fecha de nacimiento: {birthdays[image_counter]}')
        BAA_label.configure(text=f'Edad ósea: {BAA_predictions[image_counter]}')
        
def ROIs_array():
    """
    Processes and creates ROI images if not already calculated.
    - Uses ROI_detection for processing
    - Creates CTkImage objects for each ROI
    - Updates global ROI-related variables
    """
    global ROI_images, total_image_paths, images_crops, is_ROIs_calculated
    if not is_ROIs_calculated:
        if images_crops is None:
            images_crops = []
        for image in images:
            im = image.cget('light_image')
            im = np.array(im)
            crops = ROI_detection.total_process(im, verbose=False)
            images_crops.append(crops)
            ROIs = ROI_image_creation.stack_ROIs(crops)
            ROI_images.append([ctk.CTkImage(light_image=Image.fromarray(ROIs[:,:,i]),
                                          dark_image=Image.fromarray(ROIs[:,:,i]),
                                          size=(100,100)) for i in range(18)])
        is_ROIs_calculated = True

def BAAs_array():
    """
    Calculates Bone Age Assessment (BAA) predictions for all images if not already done.
    - Processes ROI images through BAA prediction model
    - Updates global BAA prediction array
    - Sets calculation flag
    """
    global BAA_predictions, is_BAAs_calculated, sexes

    one_hot_encoded_sex = {
        'Female':[1, 0],
        'Male':[0, 1],
        None:[1, 0]
    } # One-hot encoding of the sex, the default value is female encoding

    if not is_BAAs_calculated:
        if BAA_predictions is None:
            BAA_predictions = []
        for crops, sex in zip(images_crops, sexes):
            ROI_image = ROI_image_creation.image_ROIs(crops)
            age = BAA_prediction.predict_BAA(ROI_image, one_hot_encoded_sex[sex])
            BAA_predictions.append(age)
        is_BAAs_calculated = True

def save_ROIs_func():
    """
    Saves detected ROIs to individual image files in a user-selected directory.
    - Creates a subfolder for each image
    - Saves non-empty ROIs as separate PNG files
    - Names files according to ROI type
    - Shows confirmation message when complete
    """
    folder_name = filedialog.askdirectory(title='Seleccione la carpeta')
    for ROIs, path in zip(ROI_images, total_image_paths):
        name = os.path.splitext(os.path.basename(path))[0]  # Extracts filename without extension
        os.mkdir(f'{folder_name}/{name}')
        
        for i, ROI in enumerate(ROIs):
            im = ROI.cget('light_image')
            # Only save if the ROI contains actual image data
            if im.getbbox() is not None:
                im.save(f'{folder_name}/{name}/{ROI_names[i]}.png')
    show_message('ROIs guardadas')

def save_results_func():
    """
    Saves complete analysis results as DICOM files.
    - Stores patient information, measurements, and analysis results in DICOM format
    - Includes demographic data, bone age assessments, and image data
    - Creates standardized DICOM metadata
    - Shows confirmation message when complete
    """
    global images, names, birthdays, sexes, weights, heights, ethnicities, corrected_BAA, BAA_predictions, total_image_paths

    # Get final BAA entry before saving
    patient_BAA = get_BAA_entry()
    store_BAA_entry(patient_BAA)

    folder_name = filedialog.askdirectory(title='Seleccione la carpeta')

    # Standard DICOM sex codes
    sex_map = {
        "Male": "M",
        "Female": "F",
        "Other": "O"
    }
    
    for i, (image, path) in enumerate(zip(images, total_image_paths)):
        # Create output filename
        # Obtén el nombre base del archivo (p.ej. "1386.png")
        base_filename = os.path.basename(path)
        # Extrae solo el nombre sin extensión (p.ej. "1386")
        name = os.path.splitext(base_filename)[0]
        # Construye la ruta completa de salida
        output_dicom = os.path.join(folder_name, f'{name}.dcm')

        # Initialize DICOM dataset
        ds = FileDataset(output_dicom, {}, 
                        file_meta=pydicom.dataset.FileMetaDataset(),
                        preamble=b"\0" * 128)

        # Set required DICOM metadata
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage
        ds.SOPInstanceUID = generate_uid()
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.Modality = "OT"  # Other
        
        # Add patient information if available
        if names[i] is not None:
            ds.PatientName = names[i]
        
        if birthdays[i] is not None:
            ds.PatientBirthDate = birthdays[i].strftime("%Y%m%d")
        
        ds.PatientSex = sex_map.get(sexes[i], "O")
        
        if weights[i] is not None:
            try:
                ds.PatientWeight = float(weights[i])
            except ValueError:
                ds.PatientWeight = None
        
        if heights[i] is not None:
            try:
                ds.PatientSize = float(heights[i]) / 100  # Convert to meters
            except ValueError:
                ds.PatientSize = None
        
        if ethnicities[i] is not None:
            ds.PatientComments = f"Ethnicity: {ethnicities[i]}"
        
        # Add bone age assessment
        bone_age = corrected_BAA[i] if corrected_BAA[i] is not None else BAA_predictions[i]
        if bone_age is not None:
            ds.add_new((0x0012, 0x0020), "LO", f"Bone Age: {bone_age}")

        # Add image data
        im = image.cget('light_image')
        im = np.array(im)
        
        # Set image-specific DICOM attributes
        ds.Rows, ds.Columns = im.shape
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.SamplesPerPixel = 1
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = im.tobytes()

        # Save DICOM file
        ds.save_as(output_dicom)

    show_message('Imagenes guardadas')

def show_message(text):
    """
    Displays a modal message box with the given text.
    
    Args:
        text (str): Message to display
    """
    message_window = ctk.CTkToplevel(root)
    message_window.title("Message")
    message_window.geometry("300x150")
    message_window.grab_set()  # Make window modal

    message_label = ctk.CTkLabel(message_window, text=text)
    message_label.pack(pady=20)

    ok_button = ctk.CTkButton(message_window, text="OK", command=message_window.destroy)
    ok_button.pack(pady=10)

def reset_entries(patient_name, patient_height, patient_weight):
    """
    Clears all patient data entry fields and resets to default values.
    
    Args:
        patient_name (str): Current name to clear
        patient_height (str): Current height to clear
        patient_weight (str): Current weight to clear
    """
    name.delete(0, len(patient_name))
    height.delete(0, len(patient_height))
    weight.delete(0, len(patient_weight))
    sex_variable.set('None')
    ethnicity.set('Etnia')
    age.set_date(default_date)

def get_entries():
    """
    Retrieves all current values from patient data entry fields.
    
    Returns:
        tuple: (name, height, weight, sex, ethnicity, birthday)
    """
    patient_name = name.get()
    patient_height = height.get()
    patient_weight = weight.get()
    patient_sex = sex_variable.get()
    patient_ethnicity = ethnicity.get()
    patient_birthday = age.get_date()
    return patient_name, patient_height, patient_weight, patient_sex, patient_ethnicity, patient_birthday

def store_entries(patient_name, patient_height, patient_weight, patient_sex, patient_ethnicity, patient_birthday):
    """
    Stores patient data in global arrays for the current image.
    Handles empty/default values by storing None instead.
    """
    global image_counter
    names[image_counter] = patient_name if patient_name != '' else None
    heights[image_counter] = patient_height if patient_height != '' else None
    weights[image_counter] = patient_weight if patient_weight != '' else None 
    sexes[image_counter] = patient_sex if patient_sex != 'None' else None
    ethnicities[image_counter] = patient_ethnicity if patient_ethnicity != 'Etnia' else None
    birthdays[image_counter] = patient_birthday if patient_birthday != default_date else None

def display_entries_info(image_counter):
    """
    Populates entry fields with stored patient data for the current image.
    Only fills fields that have stored values (not None).
    
    Args:
        image_counter (int): Index of current image
    """
    if names[image_counter] is not None:
        name.insert(0, names[image_counter])
    if heights[image_counter] is not None:
        height.insert(0, heights[image_counter])
    if weights[image_counter] is not None:
        weight.insert(0, weights[image_counter])
    if sexes[image_counter] is not None:
        sex_variable.set(sexes[image_counter])
    if ethnicities[image_counter] is not None:
        ethnicity.set(ethnicities[image_counter])
    if birthdays[image_counter] is not None:
        age.set_date(birthdays[image_counter])

def reset_BAA_entry(patient_BAA):
    """
    Clears the bone age assessment entry field.
    
    Args:
        patient_BAA (str): Current BAA value to clear
    """
    BAA_entry.delete(0, len(patient_BAA))

def get_BAA_entry():
    """
    Gets the current value from the bone age assessment entry field.
    
    Returns:
        str: Current BAA entry value
    """
    return BAA_entry.get()

def store_BAA_entry(patient_BAA):
    """
    Stores the bone age assessment value for the current image.
    
    Args:
        patient_BAA (str): BAA value to store
    """
    global image_counter, corrected_BAA
    corrected_BAA[image_counter] = patient_BAA if patient_BAA != '' else None

def display_BAA_entry_info(image_counter):
    """
    Displays stored bone age assessment value for the current image.
    
    Args:
        image_counter (int): Index of current image
    """
    global corrected_BAA
    if corrected_BAA[image_counter] is not None:
        BAA_entry.insert(0, corrected_BAA[image_counter])

# window
root = ctk.CTk()
root.title('Carpocef')
root.geometry('960x540')

# Menu
menu = CTkMenu.CTkMenuBar(root)
file = menu.add_cascade("Archivos")

dropdownfile = CTkMenu.CustomDropdownMenu(widget=file)
dropdownfile.add_option('Añadir imagenes',command=lambda : load_images_func(False,True))
dropdownfile.add_option('Añadir carpeta',command=lambda : load_images_func(True,True))
dropdownfile.add_option('Eliminar imagenes',command=load_images_view_func)
dropdownfile.add_separator()
dropdownfile.add_option('Cargar carpeta/imagenes',command=load_images_view_func)
dropdownfile.add_option('TW ROIs',command=TW_ROIs_func)
dropdownfile.add_option('Predicción BAA',command=baa_prediction_func)
dropdownfile.add_separator()
dropdownfile.add_option('Salir',command=root.quit)

# Global Frame
global_frame = ctk.CTkFrame(master=root)

global_frame.columnconfigure((0,1),weight=1,uniform='fred')
global_frame.rowconfigure(0,weight=1)
global_frame.pack(fill='both',expand=True)

# ----Main----
# Frames
main_right_frame = ctk.CTkFrame(master=global_frame)
main_left_frame = ctk.CTkFrame(master=global_frame)
# Buttons
load_images = ctk.CTkButton(master=main_left_frame,text='Cargar Imagenes',command=lambda : load_images_func(False,False))
load_folder = ctk.CTkButton(master=main_left_frame,text='Cargar Carpeta',command=lambda : load_images_func(True,False))

# ----Images Loaded----
ethnicities_options = ['Asiatico', 'Africano', 'Caucasico', 'Hispano', 'Mixto', 'No aplica']
# Frames
images_loaded_right_frame = ctk.CTkFrame(master=global_frame)
images_loaded_left_frame = ctk.CTkFrame(master=global_frame)
# Buttons
previous_image = ctk.CTkButton(master=images_loaded_left_frame,text='<<',width=20,command=lambda : change_image_func(False))
next_image = ctk.CTkButton(master=images_loaded_left_frame,text='>>',width=20,command=lambda : change_image_func(True))
TW_ROIs = ctk.CTkButton(master=images_loaded_right_frame,text='TW ROIs',command=TW_ROIs_func)
baa_prediction = ctk.CTkButton(master=images_loaded_right_frame,text='Predicción BAA',command=baa_prediction_func)
# Labels
hand_radiograph = ctk.CTkLabel(master=images_loaded_left_frame,text='')
radiograph_label = ctk.CTkLabel(master=images_loaded_left_frame,text='')
# Entries
name = ctk.CTkEntry(master=images_loaded_right_frame,placeholder_text='Nombre')
height = ctk.CTkEntry(master=images_loaded_right_frame,placeholder_text='Estatura')
weight = ctk.CTkEntry(master=images_loaded_right_frame,placeholder_text='Peso')
# RadidoButtons
sex_variable = ctk.StringVar(master=images_loaded_right_frame,value='None')
male = ctk.CTkRadioButton(master=images_loaded_right_frame,text='Masculino',value='Male',variable=sex_variable)
female = ctk.CTkRadioButton(master=images_loaded_right_frame,text='Fememino',value='Female',variable=sex_variable)
# Calendars
age = DateEntry(master=images_loaded_right_frame,font=50,background='darkblue',foreground='white',borderwidth=2,year=today_year+year_offset,date_pattern='dd/MM/yyyy')
# Combo boxes
ethnicity = ctk.CTkComboBox(master=images_loaded_right_frame,values=ethnicities_options)

# ----TW ROIs----
# Frames
TW_ROIs_right_frame = ctk.CTkFrame(master=global_frame)
TW_ROIs_left_frame = images_loaded_left_frame
# Buttons
save_ROIs = ctk.CTkButton(master=TW_ROIs_right_frame,text='Guardar ROIs',command=save_ROIs_func)
previous_ROI = ctk.CTkButton(master=TW_ROIs_right_frame,text='<<',width=20,command=lambda : change_ROI_func(False))
next_ROI = ctk.CTkButton(master=TW_ROIs_right_frame,text='>>',width=20,command=lambda : change_ROI_func(True))
# Labels
ROI_label = ctk.CTkLabel(master=TW_ROIs_right_frame,text='')
ROI_radiograph =ctk.CTkLabel(master=TW_ROIs_right_frame,text='')
ROI_radiograph_label = ctk.CTkLabel(master=TW_ROIs_right_frame,text='')

# ----Predictions----
# Frames
Predictions_right_frame = ctk.CTkFrame(master=global_frame)
Predictions_left_frame = images_loaded_left_frame
# Buttons
save_results = ctk.CTkButton(master=Predictions_right_frame,text='Guardar Resultados',command=save_results_func)
# Labels
font_size = 15
font_family = 'Segoe UI'
name_label = ctk.CTkLabel(master=Predictions_right_frame,text='',font=(font_family,font_size))
height_label = ctk.CTkLabel(master=Predictions_right_frame,text='',font=(font_family,font_size))
weight_label = ctk.CTkLabel(master=Predictions_right_frame,text='',font=(font_family,font_size))
sex_label = ctk.CTkLabel(master=Predictions_right_frame,text='',font=(font_family,font_size))
ethnicity_label = ctk.CTkLabel(master=Predictions_right_frame,text='',font=(font_family,font_size))
age_label = ctk.CTkLabel(master=Predictions_right_frame,text='',font=(font_family,font_size))
BAA_label = ctk.CTkLabel(master=Predictions_right_frame,text='',font=(font_family,font_size))
# Entries
BAA_entry = ctk.CTkEntry(master=Predictions_right_frame,placeholder_text='Edad Osea final')

# Initial widgets
left_arguments = dict(row=0,column=0,sticky='NSWE',padx=5)
right_arguments = dict(row=0,column=1,sticky='NSWE',padx=5)
main_left_frame.grid(**left_arguments)
main_right_frame.grid(**right_arguments)

main_left_frame.rowconfigure((0,3),weight=1)
main_left_frame.columnconfigure(0,weight=1)

main_right_frame.columnconfigure(0,weight=1)
load_images.grid(row=1,column=0,pady=10)
load_folder.grid(row=2,column=0,pady=10)


root.mainloop()

