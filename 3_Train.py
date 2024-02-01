import streamlit as st
import io

import random
import numpy as np
import tensorflow_hub as hub
import os
import shutil
import keras.backend as K
import tensorflow as tf
from tensorflow import keras
import contextlib
from sklearn.utils import compute_class_weight
def get_classes_0():
    if not (os.path.isdir("data/images/images")):
        #st.warning("You have to Uploid data properly first " ,icon="⚠️")
        return None
    else:
        l = os.listdir("data/images/images")
        if len(l) == 0:
            #st.warning("You have to Uploid data properly first ", icon="⚠️")
            return None
        else:
            return l
# path = "C:/Users/youss/Pictures/icons/ds-2.png"
# add_logo(path)
classes_0=get_classes_0()
if classes_0!=None:
    num_labels=len(classes_0)
else:
    num_labels=5
features={
    "validate":{
        "transfer_learning_1":False,
        "transfer_learning_2":False,
        "upload_data": False,
        "confirm_data_splitting": False,
        "image_dataset_from_directory": False,
        "data_augmentation_1":False,
        "data_augmentation_2":False,
        "input_layer": False,
        "model": False,
        "compile_1": False,
        "compile_2": False,
        "fit": False
    },
    "choise_data_form":"labeled Folder",
    "change_classes_names":False,
    "all_classes":"classe_1,classe_2,classe_3",
    "classe_column":" ",
    "images_file_path":" ",
    "all_parameters":None,
    "all_layers_names":None,
 "used_test_dataset":False,
    "use_test_data":False,
    "use_existing_model":False,
    "use_last_model":False,
    "rescale_0": 'None',
    'label_mode_0': 'binary',
    'color_mode_0': 'rgb',
    'batch_size_0': 32,
    'image_size_0': (256, 256),
'interpolation_0': 'bilinear',
'crop_to_aspect_ratio_0': False,
  "rescale_1": 'None',
    'label_mode_1': 'binary',
    'color_mode_1': 'rgb',
    'batch_size_1': 32,
    'image_size_1': (256, 256),
'interpolation_1': 'bilinear',
'crop_to_aspect_ratio_1': False,
  "rescale_2": 'None',
    'label_mode_2': 'binary',
    'color_mode_2': 'rgb',
    'batch_size_2': 32,
    'image_size_2': (256, 256),
'interpolation_2': 'bilinear',
'crop_to_aspect_ratio_2': False,
  "rescale_3": 'None',
    'label_mode_3': 'binary',
    'color_mode_3': 'rgb',
    'batch_size_3': 32,
    'image_size_3': (256, 256),
'interpolation_3': 'bilinear',
'crop_to_aspect_ratio_3': False,
  "rescale_4": 'None',
    'label_mode_4': 'binary',
    'color_mode_4': 'rgb',
    'batch_size_4': 32,
    'image_size_4': (256, 256),
'interpolation_4': 'bilinear',
'crop_to_aspect_ratio_4': False,
    "data_augmentation_over_sampling_dict":None,
    "show_data_over_sampling_data_augmentation_option":False,
    "fixinig_inbalence":" ",
    "use_data_augmentation":False,
    "data_augmentation_dict":None,
    "validation_size_org":0.01,
    "rescale":'None',
    "flow_from_directory":None,
    "image_dataset_from_directory":None,
    'label_mode': 'binary',
    'color_mode': 'rgb',
    'batch_size': 32,
    'image_size': (256, 256),
    'target_size': (256, 256),
    'shuffle': True,
    'seed': 10,
    'interpolation': 'bilinear',
    'crop_to_aspect_ratio': False,
    'class_mode': 'categorical',
    'keep_aspect_ratio': False,
    "featurewise_center": False,
    "samplewise_center": False,
    "featurewise_std_normalization": False,
    "samplewise_std_normalization": False,
    "zca_whitening": False,
    "zca_epsilon": "1e-06",
    "rotation_range": 0,
    "width_shift_range": 0.0,
    "height_shift_range": 0.0,
    "brightness_range": None,
    "shear_range": 0.0,
    "zoom_range": 0.0,
    "channel_shift_range": 0.0,
    "fill_mode": "nearest",
    "cval": 0.0,
    "horizontal_flip": False,
    "vertical_flip": False,
    "preprocessing_function": False,
    "data_format": None,
    "validation_split": 0.0,
    "interpolation_order": 1,
    "dtype": None,
    "data_aug": False,
    "ReduceLROnPlateau": None,
    "use_step_per_epoch": "None",
    "use_validation_step": 'None',
    "start_from_last_epoch": False,
    "model": None,
    "load_weights": False,
    "max": None,
    "k": 5,
    "AUC_dict": None,
    "from_logits": False,
    "num_labels": num_labels,
    "multi_label": False,
    "summation_method": 'interpolation',
    'curve': 'ROC',
    'num_thresholds': 200,
    "F1Score_mode": 1,
    "F1Score": False,
    "average": "None",
    "your_callback_list": [],
    "EarlyStopping_": False, "ModelCheckpoint_": False, "ReduceLROnPlateau_": False,
    'jit_compile_c': False,
    "metrics": [],
    "Accuracy": False, "AUC": False, "Precision": False, "Recall": False, "TopKCategoricalAccuracy": False,
    "MeanAbsoluteError": False, "MeanSquaredError": False, "RootMeanSquaredError": False,
    "the_optimizer": None,
    'optimizer': "Adam",
    'learning_rate': "0.001",
    "use_weight_decay": False,
    'weight_decay': "0.004",
    'beta_1': "0.9",
    'beta_2': "0.999",
    'epsilon': "1e-07",
    'amsgrad': False,
    "use_clipnorm": False,
    'clipnorm': 0.1,
    "use_clipvalue": False,
    'clipvalue': 0.5,
    'use_global_clipnorm': False,
    'global_clipnorm': 1.0,
    'use_ema': False,
    'ema_momentum': 0.99,
    'ema_overwrite_frequency': 100,
    'rho': 0.95,
    'momentum': 0.0,
    'centered': False,
    'initial_accumulator_value': 0.1,
    'nesterov': False,
    'calllback': False,
    "callbacks": "None",
    "monitor": "val_loss",
    "min_delta": 0.0,
    "verbose": 1,
    "EarlyStopping": "None",
    "ModelCheckpoint": "None",
    "start_callback": False,
    "optimize": "adam",
    "loss": "BinaryCrossentropy",
    "your_callback_selected": "None",
    "delete": False,
    "epochs": 10,
    "steps_per_epoch": 10,
    "validation_steps": 10,
    "validation_batch_size": 32,
    "patience": 0,
    "mode": "auto",
    "baseline": 10.0,
    "restore_best_weights": True,
    "start_from_epoch": 0,
    "save_best_only": False,
    "save_weights_only": False,
    "file_name": "model",
    "initial_value_threshold_use": False,
    "initial_value_threshold": 0.5,
    "save_freq": 5,
    "save_freq_selected": "epoch",
    "baseline_use": False,
    "factor": 0.1,
    "cooldown": 0,
    "min_lr": 0.0,
    "steps_per_execution_use": False,
    "steps_per_execution": 1,
    "jit_compile": True,
"not_showing_other_expander":False,
"show_base_model_summary":False,
"trainable":False,
"image_shape":(256,256,3),
"number_layers_trainable_false":2,
"trainable_options":"all layers Trainable False",
"last_layer":"",
'base_model':None,
"pre_trained_model_dict":None,
"width" : 256,
"height": 256,
"channels": 3,
"include_top" : True,
"weights" :'imagenet',
"pooling": "None",
"classes" :1000,
"classifier_activation" : 'softmax',
"transfer_learning_methode" :"URL from  Tensorflow hub",
"URL": "",
"use_transfer_learning" :False,
"model_names":"Vgg19"
}
if "validate" not in st.session_state:
    for feature, default_value in features.items():
        if feature not in st.session_state:
            st.session_state[feature] = default_value

###############################"
from contextlib import contextmanager
from io import StringIO
# from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread

import sys


def get_y():
    root_directory = "data"
    training = os.path.join(os.path.join(root_directory, "data"), "training")
    y = []
    i = 0
    classes = os.listdir(training)
    for folder in classes:
        y = y + [i] * len(os.listdir(os.path.join(training, folder)))
        i = i + 1
    return np.array(y), classes
def delete_files(path):
    l=os.listdir(path)
    for file in l:
        if len(file)>19:
         if file[:18]=="data__augmented__":
             os.remove(os.path.join(path,file))
def delete_augmented_images():
    training_dir = 'data/data/training'
    l = os.listdir(training_dir)
    if st.session_state.max!=None:
        for dir in l:
            dir_path=os.path.join(training_dir,dir)
            if dir_path==st.session_state.max["name"]:
                continue
            else:
                delete_files(dir_path)
def generate_augmented_images(num_images, path):
    # Create an ImageDataGenerator object with specified augmentations
    if type(st.session_state.data_augmentation_over_sampling_dict)==dict:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
           **st.session_state.data_augmentation_over_sampling_dict
        )

    else:
        st.warning('We use the default data augmentation for the generated images to balance the data ')
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    # Get a list of all image file names in the specified directory
    file_names = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    # Use the ImageDataGenerator to generate augmented images
    for i in range(num_images):
        # Choose a random image file name from the list
        img_path = file_names[random.randint(0, len(file_names) - 1)]

        # Load the image and reshape it to match the input shape expected by the model
        img = tf.keras.utils.load_img(img_path, target_size=st.session_state.image_shape)
        x = tf.keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Generate an augmented image using the datagen object
        aug_iter = datagen.flow(x, batch_size=1)
        aug_img = aug_iter.next()[0].astype('uint8')

        # Save the augmented image to disk with a new file name
        file_name, ext = os.path.splitext(os.path.basename(img_path))
        aug_file_name = f"data__augmented__{file_name}_aug_{i}{ext}"
        aug_file_path = os.path.join(path, aug_file_name)
        tf.keras.utils.save_img(aug_file_path, aug_img)
def over_sampling_with_data_augmentation():
    training_dir = 'data/data/training'
    l = os.listdir(training_dir)
    classes = {}
    max = {"value": 0}
    for dir in l:
        if len(os.listdir(os.path.join(training_dir, dir))) > max["value"]:
            max["value"] = len(os.listdir(os.path.join(training_dir, dir)))
            max["name"] = os.path.join(training_dir, dir)
        classes[os.path.join(training_dir, dir)] = len(os.listdir(os.path.join(training_dir, dir)))
    # now we will create a function that add images to inbalence classe
    classes.pop(max["name"])
    for path, value in classes.items():
        generate_augmented_images(max["value"]-value, path)
    return max
def get_class_weights():
    y_train, classes = get_y()
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(zip(np.unique(y_train), class_weights))
    return class_weights
###############################""
def get_classes():
    if not (os.path.isdir("data/images/images")):
        st.warning("You have to Uploid data properly first " ,icon="⚠️")
        return None
    else:
        l = os.listdir("data/images/images")
        if len(l) == 0:
            st.warning("You have to Uploid data properly first ", icon="⚠️")
            return None
        else:
            return l
def Undersampling():
    training_dir = 'data/data/training'
    l = os.listdir(training_dir)
    classes = {}
    min = {"value": len(os.listdir(os.path.join(training_dir, l[0])))}
    for dir in l:
        if len(os.listdir(os.path.join(training_dir, dir))) < min["value"]:
            min["value"] = len(os.listdir(os.path.join(training_dir, dir)))
    return min
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    l = os.listdir(SOURCE)

    s = 0
    if st.session_state.fixinig_inbalence=="Undersampling":
        min=Undersampling()
        l = random.sample(l, len(l))
        t=int(SPLIT_SIZE*min["value"])
    else:
        t = int(SPLIT_SIZE * (len(l)))
    for img in l:
        s = s + 1
        image_source_path = os.path.join(SOURCE, img)
        if s <= t:
            shutil.copyfile(image_source_path, os.path.join(TRAINING, img))
        else:
            shutil.copyfile(image_source_path, os.path.join(TESTING, img))
def split_all_data(test_size, classes):
    root_dir = os.path.join("data", "data")
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    else:
        os.mkdir(root_dir)
    test_size = float(test_size)
    training_folder = os.path.join(root_dir, "training")
    testing_folder = os.path.join(root_dir, "testing")
    os.makedirs(training_folder)
    os.makedirs(testing_folder)
    classes_folder = {}
    for classe in classes:
        classes_folder[classe + "_training"] = os.path.join(training_folder, classe)
        classes_folder[classe + "_testing"] = os.path.join(testing_folder, classe)
        os.makedirs(classes_folder[classe + "_training"])
        os.makedirs(classes_folder[classe + "_testing"])
        split_data(os.path.join("data/images/images", classe), classes_folder[classe + "_training"],
                   classes_folder[classe + "_testing"], 1 - test_size)
    return "done"
def train_val_generators(dict_data_augmentation_train, dict_flow_from_directory_train, classes):
    # Instantiate the ImageDataGenerator class (don't forget to set the arguments to augment the images)
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**dict_data_augmentation_train)

    # Pass in the appropriate arguments to the flow_from_directory method
    training_folder = os.path.join(os.path.join("data", "data"), "training")
    testing_directory = os.path.join(os.path.join("data", "data"), "testing")
    dict_flow_from_directory_train["classes"] = classes
    dict_flow_from_directory_validation = dict_flow_from_directory_train
    dict_flow_from_directory_train["directory"] = training_folder
    train_generator = train_datagen.flow_from_directory(**dict_flow_from_directory_train)

    # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
    if st.session_state.validation_size_org>0:
        validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=dict_data_augmentation_train["rescale"])
        dict_flow_from_directory_validation["directory"] = testing_directory
        # Pass in the appropriate arguments to the flow_from_directory method
        validation_generator = validation_datagen.flow_from_directory(**dict_flow_from_directory_validation)
    else:
        validation_generator=None

    return train_generator, validation_generator

def image_dataset_from_directory(image_dataset_from_directory_dict, classes):
    image_dataset_from_directory_dict['directory'] = os.path.join(os.path.join("data", "data"), "training")
    image_dataset_from_directory_dict['class_names'] = classes
    if st.session_state.rescale == "1/255.0":
        rescale_factor=tf.keras.layers.Rescaling(1./255)
    elif st.session_state.rescale == "1/127.5 - 1":
        rescale_factor = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)
    else:
        rescale_factor=1
    train_datst = tf.keras.utils.image_dataset_from_directory(**image_dataset_from_directory_dict)
    if rescale_factor != 1:
        train_datst = train_datst.map(lambda x, y: (rescale_factor(x), y))

    image_dataset_from_directory_dict['directory'] = os.path.join(os.path.join("data", "data"), "testing")
    if st.session_state.validation_size_org>0:
        val_dats = tf.keras.utils.image_dataset_from_directory(**image_dataset_from_directory_dict)
        if rescale_factor!=1:
            val_dats = val_dats.map(lambda x, y: (rescale_factor(x), y))
    else:
        val_dats=None
    return train_datst, val_dats
if get_classes()!=None:
    num_labels=len(get_classes())
else:
    num_labels=5
    st.warning("Uploid data properly", icon="⚠️")

features_help = {
    "monitor": "Quantity to be monitored.",
    "min_delta": "Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.",
    "patience": "Number of epochs with no improvement after which training will be stopped.",
    "verbose": "Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays messages when the callback takes an action",
    "mode": "In 'min' mode, training will stop when the quantity monitored has stopped decreasing; in 'max' mode it will stop when the quantity monitored has stopped increasing; in 'auto' mode, the direction is automatically inferred from the name of the monitored quantity.",
    "baseline": "Baseline value for the monitored quantity. Training will stop if the model doesn't show improvement over the baseline.",
    "restore_best_weights": "Whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used. An epoch will be restored regardless of the performance relative to the baseline. If no epoch improves on baseline, training will run for patience epochs and restore weights from the best epoch in that set.",
    "start_from_epoch": "Number of epochs to wait before starting to monitor improvement. This allows for a warm-up period in which no improvement is expected and thus training will not be stopped.",
    "save_best_only": "if save_best_only=True, it only saves when the model is considered the 'best' and the latest best model according to the quantity monitored will not be overwritten",
    "save_weights_only": "if True, then only the model's weights will be saved",
    "save_freq": "'epoch' or integer. When using 'epoch', the callback saves the model after each epoch. When using integer, the callback saves the model at end of this many batches",
    "initial_value_threshold": "A threshold value for the monitored metric, below which the model will not be saved during the first epoch. This can be useful for avoiding overfitting or instability during the initial training epochs.",
    "factor": "actor by which the learning rate will be reduced. new_lr = lr * factor.",
    "cooldown": "number of epochs to wait before resuming normal operation after lr has been reduced",
    "min_lr": "lower bound on the learning rate.",
    "learning_rate": "The learning rate determines the step size used during optimization",
    "beta_1": "A float between 0 and 1. The exponential decay rate for the first moment estimates",
    "beta_2": "A float between 0 and 1. The exponential decay rate for the second moment estimates",
    "epsilon": " A small float value used to prevent division by zero. The default value is 1e-7.",
    "amsgrad": " A boolean indicating whether to use the AMSGrad variant of the algorithm. The default value is False.",
    "weight_decay": "Weight decay is a regularization technique used to prevent overfitting in neural networks. It adds a penalty term to the loss function to encourage the model to learn simpler weights. The normal value to give for weight decay is between 0.0001 to 0.001",
    "clipnorm": "Clipnorm is a technique used to prevent the gradient from exploding during backpropagation. It clips the norm of the gradient vector to a maximum value, preventing it from exceeding a certain threshold. The normal value to give for clipnorm is between 0.1 to 10.0.",
    "clipvalue": "Clipvalue is another technique used to prevent the gradient from exploding. Instead of clipping the norm of the gradient vector, clipvalue clips the values of the gradient tensor to a maximum and minimum value. The normal value to give for clipvalue is between 0.5 to 5.0.",
    "global_clipnorm": "Global_clipnorm is a variation of clipnorm where the gradient is clipped based on the norm of all the gradients in the network, instead of just the norm of each gradient vector. This can be useful in large-scale distributed training when gradients are aggregated from multiple workers. The normal value to give for global_clipnorm is between 1.0 to 10.0.",
    "use_ema": "If True, exponential moving average (EMA) is applied. EMA consists of computing an exponential moving average of the weights of the model (as the weight",
    "ema_overwrite_frequency": "Exponential moving average (EMA) is a technique used to smooth out the training process and prevent oscillations in the loss function. EMA_overwrite_frequency specifies how often the EMA is updated during training. The normal value to give for ema_overwrite_frequency is between 100 to 1000.",
    "ema_momentum": "the momentum to use when computing the EMA of the model's weights: new_average = ema_momentum * old_average + (1 - ema_momentum) * current_variable_value.",
    "jit_compile": "This parameter is a boolean indicating whether to use TensorFlow's XLA (Accelerated Linear Algebra) compiler to optimize the execution of the model during training.",
    "momentum": " This parameter is used to accelerate the gradient descent algorithm in the relevant direction and dampen the oscillations. It is a float value between 0 and 1. If set to 0, momentum is not used.",
    "nesterov": "This parameter is a boolean indicating whether to use Nesterov momentum or not. Nesterov momentum is a modification of the classic momentum algorithm that calculates the gradient not at the current weights but at a hypothetical point in the direction of the momentum. The default value is False",
    "rho": "Discounting factor for the old gradients.",
    "centered": "If True, gradients are normalized by the estimated variance of the gradient; if False, by the uncentered second moment. Setting this to True may help with training, but is slightly more expensive in terms of computation and memory. Defaults to False",
    "initial_accumulator_value": "Floating point value. Starting value for the accumulators (per-parameter momentum values). Must be non-negative.",
    "F1Score": "micro: True positivies, false positives and false negatives are computed globally. macro: True positivies, false positives and false negatives are computed for each classand their unweighted mean is returned. weighted: Metrics are computed for each class    and returns the mean weighted by the number of true instances in each class.",
    "from_logits": "ndicating whether the predictions (y_pred in update_state) are probabilities or sigmoid logits. As a rule of thumb, when using a keras loss, the from_logits constructor argument of the loss should match the AUC from_logits constructor argument.",
    "num_labels": "ndicating whether the predictions (y_pred in update_state) are probabilities or sigmoid logits. As a rule of thumb, when using a keras loss, the from_logits constructor argument of the loss should match the AUC from_logits constructor argument.",
    "multi_label": "indicating whether multilabel data should be treated as such, wherein AUC is computed separately for each label and then averaged across labels, or (when False) if the data should be flattened into a single label before AUC computation. In the latter case, when multilabel data is passed to AUC, each label-prediction pair is treated as an individual data point. Should be set to False for multi-class data",
    "summation_method": "Specifies the Riemann summation method used. 'interpolation' (default) ",
    'curve': "Specifies the name of the curve to be computed, 'ROC' [default] or 'PR' for the Precision-Recall-curve.",
    'num_thresholds': "Defaults to 200. The number of thresholds to use when discretizing the roc curve. Values must be > 1."
}
def create_callback(callback):
    # "EarlyStopping", "ModelCheckpoint","ReduceLROnPlateau"]
    if callback == "EarlyStopping":
        print(type(st.session_state["EarlyStopping"])==dict,"test",st.session_state["EarlyStopping"])
        if type(st.session_state["EarlyStopping"])==dict:
            if st.session_state["EarlyStopping"]["monitor"][:4]=="val_" and st.session_state.validation_size_org==0:
                st.warning("You don't have validation data Please conform validation split >0", icon="⚠️")
                return None,False
            else:
                print("st.session_state[EarlyStopping][verbose]",st.session_state["EarlyStopping"]["verbose"])
                the_callback = tf.keras.callbacks.EarlyStopping(monitor=st.session_state["EarlyStopping"]["monitor"],
                                                                min_delta=st.session_state["EarlyStopping"]["min_delta"],
                                                                patience=st.session_state["EarlyStopping"]["patience"],
                                                                verbose=st.session_state["EarlyStopping"]["verbose"],
                                                                mode=st.session_state["EarlyStopping"]["mode"],
                                                                baseline=st.session_state["EarlyStopping"]["baseline"],
                                                                restore_best_weights=st.session_state["EarlyStopping"][
                                                                    "restore_best_weights"])
        else:
            st.warning("Please Confirm EarlyStopping",icon="⚠️")
            return None,False
    elif callback == "ModelCheckpoint":
        if type(st.session_state["ModelCheckpoint"] )== dict:
            if st.session_state["ModelCheckpoint"]["monitor"][:4]=="val_" and st.session_state.validation_size_org==0:
                st.warning("You don't have validation data Please conform validation split >0", icon="⚠️")
                return None,False
            else:
                the_callback = tf.keras.callbacks.ModelCheckpoint(
                    monitor=st.session_state.ModelCheckpoint["monitor"],
                    verbose=st.session_state.ModelCheckpoint["verbose"],
                    filepath=st.session_state.ModelCheckpoint["filepath"],
                    save_best_only=st.session_state.ModelCheckpoint["save_best_only"],
                    initial_value_threshold=st.session_state.ModelCheckpoint["initial_value_threshold"],
                    save_weights_only=st.session_state.ModelCheckpoint["save_weights_only"],
                    mode=st.session_state.ModelCheckpoint["mode"],
                    save_freq=st.session_state.ModelCheckpoint["save_freq"])
        else:
            st.warning("Please Confirm ModelCheckpoint", icon="⚠️")
            return None, False
    elif callback == "ReduceLROnPlateau":
        if type(st.session_state["ReduceLROnPlateau"] )== dict:
            print(st.session_state["ReduceLROnPlateau"]["monitor"][:4])
            if st.session_state["ReduceLROnPlateau"]["monitor"][:4]=="val_" and st.session_state.validation_size_org==0:
                st.warning("You don't have validation data Please conform validation split >0", icon="⚠️")
                return None,False
            else:
                the_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor=st.session_state.ReduceLROnPlateau["monitor"],
                                                                    factor=st.session_state.ReduceLROnPlateau["factor"],
                                                                    patience=st.session_state.ReduceLROnPlateau["patience"],
                                                                    verbose=st.session_state.ReduceLROnPlateau["verbose"],
                                                                    mode=st.session_state.ReduceLROnPlateau["mode"],
                                                                    min_delta=st.session_state.ReduceLROnPlateau["min_delta"],
                                                                    cooldown=st.session_state.ReduceLROnPlateau["cooldown"],
                                                                    min_lr=st.session_state.ReduceLROnPlateau["min_lr"])
        else:
            st.warning("Please Confirm ReduceLROnPlateau", icon="⚠️")
            return None, False
    return the_callback,True


def create_all_callbacks():
    callbacks = []
    statu=True
    if len(st.session_state['your_callback_list']) > 0:
        for i in range(0, len(st.session_state['your_callback_list'])):
            callack,statue=create_callback(st.session_state['your_callback_list'][i])
            if statue:
                callbacks.append(callack)
            else:
                statu=False
    return callbacks,statu
def create_optimazer():
  #  ["RMSprop", "Adam", "SGD", "Adagrad", "Adadelta", "AdamW"]
    try:
        if 'jit_compile' in st.session_state.the_optimizer:
            st.session_state.the_optimizer.pop('jit_compile')
        if 'use_ema' in st.session_state.the_optimizer:
            st.session_state.the_optimizer.pop('use_ema')
    except:
        print('some problem hapen in optimazer  ...')
    if st.session_state.the_optimizer==None:
        return tf.keras.optimizers.Adam()
    if st.session_state.the_optimizer["name"] == "RMSprop":
        optimazer = tf.keras.optimizers.RMSprop(**st.session_state.the_optimizer)
    elif st.session_state.the_optimizer["name"] == "Adam":
        optimazer = tf.keras.optimizers.Adam(**st.session_state.the_optimizer)
    elif st.session_state.the_optimizer["name"] == "SGD":
        optimazer = tf.keras.optimizers.SGD(**st.session_state.the_optimizer)
    elif st.session_state.the_optimizer["name"] == "Adagrad":
        optimazer = tf.keras.optimizers.Adagrad(**st.session_state.the_optimizer)
    elif st.session_state.the_optimizer["name"] == "Adadelta":
        optimazer = tf.keras.optimizers.Adadelta(**st.session_state.the_optimizer)
    elif st.session_state.the_optimizer["name"]=="AdamW":
        optimazer = tf.keras.optimizers.experimental.AdamW(**st.session_state.the_optimizer)
    else:
        optimazer=tf.keras.optimizers.Adam()
    return optimazer
def f1_score_binary(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val
def f1_score_categrecal(y_true, y_pred):
    # Calculate true positives, false positives and false negatives for each class
    tp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)), axis=0)
    fp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred - y_true, 0, 1)), axis=0)
    fn = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true - y_pred, 0, 1)), axis=0)

    # Calculate precision and recall for each class
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    # Calculate F1 score for each class and average over all classes
    f1_score = tf.keras.backend.mean(2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
    return f1_score
def get_metrics():
    metrrics = []

    if len(st.session_state["metrics"]) > 0:
        for metric in st.session_state["metrics"]:
            print(metric)
            if metric == "Accuracy":
                metrrics.append("accuracy")
            elif metric == "AUC":

                metrrics.append(
                    tf.keras.metrics.AUC(name="AUC", num_thresholds=st.session_state.AUC_dict["num_thresholds"],
                                         curve=st.session_state.AUC_dict["curve"],
                                         summation_method=st.session_state.AUC_dict["summation_method"]
                                         , multi_label=st.session_state.AUC_dict["multi_label"],
                                         from_logits=st.session_state.AUC_dict["from_logits"],
                                         num_labels=st.session_state.AUC_dict["num_labels"]))
            elif metric == "Precision":
                metrrics.append(tf.keras.metrics.Precision(name="Precision"))
            elif metric == "Recall":
                metrrics.append(tf.keras.metrics.Recall(name="Recall"))
            elif metric == "TopKCategoricalAccuracy":
                metrrics.append(
                    tf.keras.metrics.TopKCategoricalAccuracy(name="TopKCategoricalAccuracy", k=st.session_state.k))
            elif metric == "MeanAbsoluteError":
                metrrics.append(tf.keras.metrics.MeanAbsoluteError(name="MeanAbsoluteError"))
            elif metric == "MeanSquaredError":
                metrrics.append(tf.keras.metrics.MeanSquaredError(name="MeanSquaredError"))
            elif metric == "RootMeanSquaredError":
                metrrics.append(tf.keras.metrics.RootMeanSquaredError(name="RootMeanSquaredError"))
            elif metric == "F1Score":
                if get_classes==None:
                    st.warning('Upload data properly', icon="⚠️")
                else:
                    if len(get_classes()) > 2:
                        metrrics.append(f1_score_categrecal)
                        st.session_state.F1Score_mode = 2
                    else:
                        if st.session_state.use_data_augmentation:
                            if st.session_state.flow_from_directory['class_mode'] == "binary":
                                metrrics.append(f1_score_binary)
                                st.session_state.F1Score_mode = 1
                            else:
                                metrrics.append(f1_score_categrecal)
                                st.session_state.F1Score_mode = 2
                        else:
                            if st.session_state.image_dataset_from_directory['label_mode'] == "binary":
                                metrrics.append(f1_score_binary)
                                st.session_state.F1Score_mode = 1
                            else:
                                metrrics.append(f1_score_categrecal)
                                st.session_state.F1Score_mode = 2

        return metrrics
def create_model():
    if st.session_state.model!=None:
     if st.session_state.use_transfer_learning and st.session_state.trainable_options == "chose number of layers" and st.session_state.load_weights:
        model=st.session_state.model
        st.session_state.base_model.trainable = True
        for layer in st.session_state.base_model.layers[:st.session_state.number_layers_trainable_false + 1]:
            layer.trainable = False
        # st.session_state.base_model.summary(print_fn=lambda x: st.text(x))
        return model
    model = tf.keras.Sequential()
    if not (st.session_state.validate["transfer_learning_1"] and st.session_state.validate["transfer_learning_2"]) or not (st.session_state.validate["transfer_learning_1"] and st.session_state.transfer_learning_methode=="Load model"):
        model.add(tf.keras.layers.InputLayer(input_shape=st.session_state.image_shape))
    if st.session_state.use_transfer_learning and st.session_state.validate["transfer_learning_1"]:
        if st.session_state.transfer_learning_methode == "URL from  Tensorflow hub":
            model.add(hub.KerasLayer(st.session_state.URL, trainable=st.session_state.trainable))
        elif st.session_state.transfer_learning_methode == "Pre trained model from tensorflow" and \
                st.session_state.validate["transfer_learning_2"]:
            if st.session_state.base_model != None:
                if st.session_state.trainable_options == "all layers Trainable False":
                    st.session_state.base_model.trainable = False
                    base_model = st.session_state.base_model
                elif st.session_state.trainable_options == "all layers Trainable True":
                    st.session_state.base_model.trainable = True
                    base_model = st.session_state.base_model
                else:
                    st.session_state.base_model.trainable = True
                    for layer in st.session_state.base_model.layers[
                                 :st.session_state.number_layers_trainable_false + 1]:
                        layer.trainable = False
                    base_model = st.session_state.base_model
                # base_model.summary(print_fn=lambda x: st.text(x))
                model.add(base_model)
        elif st.session_state.transfer_learning_methode == "Pre trained model from tensorflow" and not \
        st.session_state.validate["transfer_learning_2"]:
            st.warning("Please Confirm the chossing Last layer ")
            return None
        else:
            if st.session_state.model_load == None:
                st.warning("You didin't load your model properly", icon="⚠️")
                return None
            else:
                model = st.session_state.model_load
            if "load_weight" in st.session_state:
                if st.session_state.load_weights:
                    file, methode = find_weight_file()
                    try:
                        st.write(methode)
                        model.load_weights(file)
                    except:
                        st.write("You need to use the same model archetecture to use load model")
            return model
    elif st.session_state.use_transfer_learning and not st.session_state.validate["transfer_learning_1"]:
        st.warning("Please Confirm Transfer learning use ")
        return None
    all_parameters=[]
    all_layer_name=[]
    i = 0
    for layer in st.session_state.layers:
        i = i + 1
        if layer.get_parameters() is not None:
            parameters = dict(layer.get_parameters())
        else:
            parameters=None

        if parameters is not None:
            if "kernel_regularizer" in parameters:
                if parameters["kernel_regularizer"]!=None:

                    if type(parameters["kernel_regularizer"])==dict:
                        if parameters["kernel_regularizer"]["regularizer"]=="l1":

                            parameters["kernel_regularizer"]=tf.keras.regularizers.L1(l1=layer.parameters["kernel_regularizer"]["penalty_l1"])
                        elif parameters["kernel_regularizer"]["regularizer"]=="l2":
                            parameters["kernel_regularizer"]=tf.keras.regularizers.L2(l2=layer.parameters["kernel_regularizer"]["penalty_l2"])
                        elif parameters["kernel_regularizer"]["regularizer"]=="L1_L2":
                            parameters["kernel_regularizer"] = tf.keras.regularizers.L1L2(l2=layer.parameters["kernel_regularizer"]["penalty_l2"],l1=layer.parameters["kernel_regularizer"]["penalty_l1"])
        all_parameters.append(parameters)
        all_layer_name.append(layer.type)
        if layer.type == "Conv2D":
            model.add(tf.keras.layers.Conv2D(**parameters))
        elif layer.type == "BatchNormalization":
            model.add(tf.keras.layers.BatchNormalization(**parameters))
        elif layer.type == "DepthwiseConv2D":
            model.add(tf.keras.layers.DepthwiseConv2D(**parameters))
        elif layer.type == "SeparableConv2D":
            model.add(tf.keras.layers.SeparableConv2D(**parameters))
        elif layer.type == "MaxPooling2D":
            model.add(tf.keras.layers.MaxPooling2D(**parameters))
        elif layer.type == "AveragePooling2D":
            model.add(tf.keras.layers.AveragePooling2D(**parameters))
        elif layer.type == "GlobalMaxPooling2D":
            model.add(tf.keras.layers.GlobalMaxPooling2D())
        elif layer.type == "GlobalAveragePooling2D":
            model.add(tf.keras.layers.GlobalAvgPool2D())
        elif layer.type == "Dropout":
            model.add(tf.keras.layers.Dropout(**parameters))
        elif layer.type == "Flatten":
            model.add(tf.keras.layers.Flatten())
        elif layer.type == "Dense":
            model.add(tf.keras.layers.Dense(**parameters))
    if st.session_state.load_weights:

        file, methode = find_weight_file()
        try:
            model.load_weights(file)
            st.write(methode)
        except :
            st.write("You need to use the same model archetecture to use load model")
    st.session_state.all_layers_names=all_layer_name
    st.session_state.all_parameters=all_parameters
    return model

def find_weight_file():
    if type(st.session_state.ModelCheckpoint)==dict:
        file_path=st.session_state.ModelCheckpoint["filepath"]
        methode="Using ModelCheckpoint file"

    elif st.session_state.model!=None:
        model=st.session_state.model
        if not os.path.exists("data/saved_model"):
            os.makedirs("data/saved_model")
        model.save("data/saved_model/model.h5")
        file_path="data/saved_model/model.h5"
        methode="Use the last model.save"
    return file_path, methode
def add_calbacks(callback):
    with st.expander(callback):
        st.header(callback)
        if callback == "EarlyStopping":
            values = ["val_loss", "loss"]
            val_values=[]
            for val in st.session_state["metrics"]:
                val_values.append("val_"+val)
            values = list(set(values + st.session_state["metrics"] + val_values))
            if "F1Score" in st.session_state["metrics"]:
                values.remove('F1Score')
                values.remove('val_F1Score')
                if get_classes==None:
                    st.warning('Upload data properly', icon="⚠️")
                else:
                    if len(get_classes()) > 2:
                        values.append("f1_score_categrecal")
                        values.append("val_f1_score_categrecal")
                        st.session_state.F1Score_mode = 2
                    else:
                        if st.session_state.use_data_augmentation:
                            if st.session_state.flow_from_directory['class_mode'] == "binary":
                                values.append("f1_score_binary")
                                values.append("val_f1_score_binary")
                                st.session_state.F1Score_mode = 1
                            else:
                                values.append("f1_score_categrecal")
                                values.append("val_f1_score_categrecal")
                                st.session_state.F1Score_mode = 2
                        else:
                            if st.session_state.image_dataset_from_directory['label_mode'] == "binary":
                                values.append("f1_score_binary")
                                values.append("val_f1_score_binary")
                                st.session_state.F1Score_mode = 1
                            else:
                                values.append("f1_score_categrecal")
                                values.append("val_f1_score_categrecal")
                                st.session_state.F1Score_mode = 2

            values_ex = values.index(st.session_state.monitor)
            monitor = st.selectbox("monitor", values, index=values_ex, help=features_help["monitor"])
            min_delta = st.number_input("min_delta", min_value=0.0, max_value=1.0, value=st.session_state.min_delta,
                                        help=features_help["min_delta"])
            patience = st.slider("patience", 1, 50, value=st.session_state.patience, help=features_help["patience"])
            value = [0, 1]
            verbose = st.radio("verbose", value, index=value.index(st.session_state.verbose),
                               help=features_help["verbose"])
            mode_liste = ["auto", "min", "max"]
            mode = st.selectbox("mode", mode_liste, index=mode_liste.index(st.session_state.mode),
                                help=features_help["mode"])
            baseline_use = st.checkbox('use baseline', value=st.session_state.baseline_use,
                                       help=features_help["baseline"])
            if baseline_use:
                baseliene = st.slider("baseline", 0.0, 50.0, value=st.session_state.baseline, step=0.1,
                                      help=features_help["baseline"])
            else:
                baseliene = st.session_state.baseline
            restore_best_weights = st.checkbox("restore_best_weights", value=st.session_state.restore_best_weights,
                                               help=features_help["restore_best_weights"])
            start_from_epoch = st.slider("start_from_epoch", 0, 100, value=st.session_state.start_from_epoch,
                                         help=features_help["start_from_epoch"])


        elif callback == "ModelCheckpoint":
            values = ["val_loss", "loss"]
            val_values = []
            for val in st.session_state["metrics"]:
                val_values.append("val_" + val)

            values = list(set(values + st.session_state["metrics"] + val_values))
            values_ex = values.index(st.session_state.monitor)
            monitor = st.selectbox("monitor :", values, index=values_ex, help=features_help["monitor"])
            value = [0, 1]
            verbose = st.radio("verbose :", value, index=value.index(st.session_state.verbose),
                               help=features_help["verbose"])
            file_name = st.text_input("file_name", value=st.session_state.file_name)
            save_best_only = st.checkbox("save_best_only :", value=st.session_state.save_best_only,
                                         help=features_help["save_best_only"])
            if save_best_only:
                initial_value_threshold_use = st.checkbox("use initial_value_threshold_use :",
                                                          value=st.session_state.initial_value_threshold_use,
                                                          help=features_help["initial_value_threshold"])
                if initial_value_threshold_use:
                    initial_value_threshold = st.slider("initial_value_threshold :", 0.01, 10.0, step=0.01,
                                                        value=st.session_state.initial_value_threshold,
                                                        help=features_help["initial_value_threshold"])
                else:
                    initial_value_threshold = st.session_state.initial_value_threshold
            else:
                initial_value_threshold_use = False
            save_weights_only = st.checkbox("save_weights_only", value=st.session_state.save_weights_only,
                                            help=features_help["save_weights_only"])
            mode_liste = ["auto", "min", "max"]
            mode = st.selectbox("mode :", mode_liste, index=mode_liste.index(st.session_state.mode),
                                help=features_help["mode"])
            save_freq_list = ['epoch', "chose number of batches"]
            save_freq_selected = st.radio("save_freq", save_freq_list,
                                          index=save_freq_list.index(st.session_state.save_freq_selected),
                                          help=features_help["save_freq"])
            if save_freq_selected == "chose number of batches":
                save_freq = st.slider("save_freq", 1, 500, step=1, value=st.session_state.save_freq,
                                      help=features_help["save_freq"])
            else:
                save_freq = st.session_state.save_freq

        elif callback == "ReduceLROnPlateau":
            values = ["val_loss", "loss"]
            val_values = []
            for val in st.session_state["metrics"]:
                val_values.append("val_" + val)

            values = list(set(values + st.session_state["metrics"] + val_values))
            values_ex = values.index(st.session_state.monitor)
            monitor = st.selectbox("monitor : ", values, index=values_ex, help=features_help["monitor"])
            factor = st.number_input("factor : ", min_value=0.0, max_value=1.0, value=st.session_state.factor,
                                     help=features_help["factor"])
            patience = st.slider("patience : ", 1, 50, value=st.session_state.patience, help=features_help["patience"])
            value = [0, 1]
            verbose = st.radio("verbose : ", value, index=value.index(st.session_state.verbose),
                               help=features_help["verbose"])
            mode_liste = ["auto", "min", "max"]
            mode = st.selectbox("mode : ", mode_liste, index=mode_liste.index(st.session_state.mode),
                                help=features_help["mode"])
            min_delta = st.number_input("min_delta : ", min_value=0.0, max_value=1.0, value=st.session_state.min_delta,
                                        help=features_help["min_delta"])
            cooldown = st.slider("cooldown : ", 0, 100, step=1, value=st.session_state.cooldown,
                                 help=features_help["cooldown"])
            min_lr = st.text_input("min_lr : ", value=str(st.session_state.min_lr), help=features_help["min_lr"])

        submet = st.button("Confirm " + callback)
        if submet:
            if callback == "EarlyStopping":
                st.session_state.monitor = monitor
                st.session_state.min_delta = min_delta
                st.session_state.patience = patience
                st.session_state.verbose = verbose
                st.session_state.mode = mode
                st.session_state.baseline_use = baseline_use
                st.session_state.baseline = baseliene
                st.session_state.restore_best_weights = restore_best_weights
                st.session_state.start_from_epoch = start_from_epoch
                EarlyStopping = {"monitor": st.session_state.monitor, "min_delta": st.session_state.min_delta,
                                 "patience": st.session_state.patience,
                                 "verbose": st.session_state.verbose, "mode": st.session_state.mode,
                                 "restore_best_weights": st.session_state.restore_best_weights,
                                 "start_from_epoch": st.session_state.start_from_epoch}
                if baseline_use:
                    EarlyStopping["baseline"] = st.session_state.baseline
                else:
                    EarlyStopping["baseline"] = None
                st.session_state.EarlyStopping = EarlyStopping
                if st.session_state.your_callback_list.count("EarlyStopping") == 0:
                    st.session_state.your_callback_list.append("EarlyStopping")
            elif callback == "ModelCheckpoint":
                st.session_state.monitor = monitor
                st.session_state.verbose = verbose
                st.session_state.file_name = file_name
                st.session_state.save_best_only = save_best_only
                st.session_state.initial_value_threshold_use = initial_value_threshold_use

                st.session_state.save_weights_only = save_weights_only
                st.session_state.mode = mode
                st.session_state.save_freq_selected = save_freq_selected
                ModelCheckpoint = {"monitor": st.session_state.monitor, "verbose": st.session_state.verbose,
                                   "filepath": os.path.join("data", st.session_state.file_name + ".h5"),
                                   "save_best_only": st.session_state.save_best_only,
                                   "save_weights_only": st.session_state.save_weights_only,
                                   "mode": st.session_state.mode}
                if st.session_state.initial_value_threshold_use == True:
                    st.session_state.initial_value_threshold = initial_value_threshold
                    ModelCheckpoint["initial_value_threshold"] = st.session_state.initial_value_threshold
                else:
                    ModelCheckpoint["initial_value_threshold"] = None
                if st.session_state.save_freq_selected == "epoch":
                    ModelCheckpoint["save_freq"] = "epoch"
                else:
                    st.session_state.save_freq = save_freq
                    ModelCheckpoint["save_freq"] = st.session_state.save_freq
                st.session_state.ModelCheckpoint = ModelCheckpoint
                if st.session_state.your_callback_list.count("ModelCheckpoint") == 0:
                    st.session_state.your_callback_list.append("ModelCheckpoint")
            elif callback == "ReduceLROnPlateau":
                st.session_state.monitor = monitor
                st.session_state.factor = factor
                st.session_state.patience = patience

                st.session_state.verbose = verbose
                st.session_state.mode = mode
                st.session_state.min_delta = min_delta
                st.session_state.cooldown = cooldown
                st.session_state.min_lr = float(min_lr)
                ReduceLROnPlateau = {"monitor": st.session_state.monitor,
                                     "factor": st.session_state.factor,
                                     "patience": st.session_state.patience,
                                     "verbose": st.session_state.verbose,
                                     "mode": st.session_state.mode,
                                     "min_delta": st.session_state.min_delta,
                                     "cooldown": st.session_state.cooldown,
                                     "min_lr": st.session_state.min_lr}
                st.session_state.ReduceLROnPlateau = ReduceLROnPlateau
                if st.session_state.your_callback_list.count("ReduceLROnPlateau") == 0:
                    st.session_state.your_callback_list.append("ReduceLROnPlateau")

            st.write("callback added!")
def optimizers(optimazer):
    st.header(optimazer)
    learning_rate = st.text_input("learning_rate", value=str(st.session_state.learning_rate),
                                  help=features_help["learning_rate"])
    if optimazer == "Adam" or optimazer == "AdamW":
        beta_1 = st.text_input("beta_1", value=str(st.session_state.beta_1), help=features_help["beta_1"])
        beta_2 = st.text_input("beta_2", value=str(st.session_state.beta_2), help=features_help["beta_1"])
        epsilon = st.text_input("epsilon", value=str(st.session_state.epsilon), help=features_help["epsilon"])
        amsgrad = st.checkbox("amsgrad", value=st.session_state.amsgrad, help=features_help["amsgrad"])
    if optimazer != "AdamW":
        weight_decay_help = "Weight decay is a regularization technique used to prevent overfitting in neural networks. It adds a penalty term to the loss function to encourage the model to learn simpler weights. The normal value to give for weight decay is between 0.0001 to 0.001"
        use_weight_decay = st.checkbox("use weight decay ", value=st.session_state.use_weight_decay,
                                       help=weight_decay_help)
        if use_weight_decay:
            weight_decay = st.text_input("weight_decay", value=str(st.session_state.weight_decay))
        else:
            weight_decay = None
    if optimazer == "RMSprop" or optimazer == "Adadelta":
        rho = st.text_input("rho", value=str(st.session_state.rho), help=features_help["rho"])
    elif optimazer == "Adadelta" or optimazer == "Adagrad" or optimazer == "RMSprop":
        epsilon = st.text_input("epsilon", value=str(st.session_state.epsilon), help=features_help["epsilon"])
    elif optimazer == "SGD":
        momentum = st.text_input("momentum", value=str(st.session_state.momentum), help=features_help["momentum"])
        nesterov = st.checkbox("nesterov", value=st.session_state.nesterov, help=features_help["nesterov"])
        amsgrad = st.checkbox("amsgrad", value=st.session_state.amsgrad, help=features_help["amsgrad"])
    if optimazer == "RMSprop":
        momentum = st.text_input("momentum", value=str(st.session_state.momentum), help=features_help["momentum"])
        centered = st.checkbox("centered", value=st.session_state.centered, help=features_help["centered"])
    elif optimazer == "Adagrad":
        initial_accumulator_value = st.text_input("initial_accumulator_value",
                                                  value=str(st.session_state.initial_accumulator_value),
                                                  help=features_help["initial_accumulator_value"])

    elif optimazer == "AdamW":
        weight_decay_help = "Weight decay is a regularization technique used to prevent overfitting in neural networks. It adds a penalty term to the loss function to encourage the model to learn simpler weights. The normal value to give for weight decay is between 0.0001 to 0.001"
        weight_decay = st.text_input("weight_decay", value=str(st.session_state.weight_decay), help=weight_decay_help)
        name = 'AdamW',
    use_clipnorm = st.checkbox("use clipnorm ", value=st.session_state.use_clipnorm, help=features_help["clipnorm"])
    if use_clipnorm:
        clipnorm = st.text_input("clipnorm", value=str(st.session_state.clipnorm))
    else:
        clipnorm = None
    use_clipvalue = st.checkbox("use_clipvalue", value=st.session_state.use_clipvalue, help=features_help["clipvalue"])
    if use_clipvalue:
        clipvalue = st.slider("clipvalue", 0.5, 5.0, value=st.session_state.clipvalue, step=0.1,
                              help=features_help["clipvalue"])
    else:
        clipvalue = None
    use_global_clipnorm = st.checkbox("use_global_clipnorm", value=st.session_state.use_global_clipnorm,
                                      help=features_help["global_clipnorm"])
    if use_global_clipnorm:
        global_clipnorm = st.slider("global_clipnorm", 1.0, 10.0, step=0.1, value=st.session_state.global_clipnorm,
                                    help=features_help["global_clipnorm"])
    else:
        global_clipnorm = None
    # use_ema = st.checkbox("use_ema", value=st.session_state.use_ema, help=features_help["use_ema"])
    # if use_ema:
    #     ema_momentum = st.text_input("ema_momentum", value=str(st.session_state.ema_momentum),
    #                                  help=features_help["ema_momentum"])
    #     ema_overwrite_frequency = st.text_input("ema_overwrite_frequency",
    #                                             value=str(st.session_state.ema_overwrite_frequency),
    #                                             help=features_help["ema_overwrite_frequency"])
    # else:
    #     ema_momentum = None
    #     ema_overwrite_frequency = None
    # jit_compile = st.checkbox("jit_compile", value=st.session_state.jit_compile, help=features_help["jit_compile"])
    confirm = st.button("confirm " + optimazer)
    if confirm:
        st.session_state.validate["compile_1"] = True
        st.session_state.learning_rate = learning_rate
        # st.session_state.jit_compile = jit_compile
        # st.session_state.use_ema = use_ema
        base = {"learning_rate": float(learning_rate)}#, "jit_compile": jit_compile, "use_ema": use_ema}
        st.session_state.use_clipnorm = use_clipnorm
        if use_clipnorm:
            st.session_state.clipnorm = clipnorm
            base["clipnorm"] = float(clipnorm)
        st.session_state.use_clipvalue = use_clipvalue
        if use_clipvalue:
            base["clipvalue"] = float(clipvalue)
            st.session_state.clipvalue = clipvalue
        if use_global_clipnorm:
            base["global_clipnorm"] = float(global_clipnorm)
            st.session_state.global_clipnorm = global_clipnorm
        # if use_ema:
        #     base["ema_momentum"] = float(ema_momentum)
        #     base["ema_overwrite_frequency"] = int(ema_overwrite_frequency)
        #     st.session_state.ema_momentum = ema_momentum
        #     st.session_state.ema_overwrite_frequency = int(ema_overwrite_frequency)
        if optimazer == "Adam" or optimazer == "AdamW":
            st.session_state.beta_1 = beta_1
            base["beta_1"] = float(beta_1)
            st.session_state.beta_2 = beta_2
            base["beta_2"] = float(beta_2)
            st.session_state.epsilon = epsilon
            base["epsilon"] = float(epsilon)
            st.session_state.amsgrad = amsgrad
            base["amsgrad"] = amsgrad
        if optimazer != "AdamW":
            st.session_state.use_weight_decay = use_weight_decay
            if use_weight_decay:
                st.session_state.weight_decay = weight_decay
                base["weight_decay"] = float(weight_decay)
        if optimazer == "Adam":
            base["name"] = "Adam"
        elif optimazer == "AdamW":
            st.session_state.weight_decay = weight_decay
            base["weight_decay"] = float(weight_decay)
            base["name"] = "AdamW"
        elif optimazer == "SGD":
            st.session_state.momentum = momentum
            st.session_state.nesterov = nesterov
            st.session_state.amsgrad = amsgrad
            base["momentum"] = float(momentum)
            base["nesterov"] = nesterov
            base["amsgrad"] = amsgrad
            base["name"] = "SGD"
        elif optimazer == "RMSprop":
            st.session_state.momentum = momentum
            st.session_state.centered = centered
            st.session_state.rho = rho
            st.session_state.epsilon = epsilon
            base["momentum"] = float(momentum)
            base["rho"] = float(rho)
            base["epsilon"] = float(epsilon)
            base["centered"] = centered
            base["name"] = "RMSprop"
        elif optimazer == "Adagrad":
            st.session_state.initial_accumulator_value = initial_accumulator_value
            st.session_state.epsilon = epsilon
            base["epsilon"] = float(epsilon)
            base["initial_accumulator_value"] = float(initial_accumulator_value)
            base["name"] = "Adagrad"
        elif optimazer == "Adadelta":
            base["rho"] = float(rho)
            st.session_state.rho = rho
            st.session_state.epsilon = epsilon
            base["epsilon"] = float(epsilon)
            base["name"] = "Adadelta"
        st.session_state["the_optimizer"] = base
        st.experimental_rerun()
with st.expander("Compile", expanded=True):
    st.subheader("Compile")
    optimazer_list = ["RMSprop", "Adam", "SGD", "Adagrad", "Adadelta", "AdamW"]
    optimizer = st.selectbox("Optimier", optimazer_list, index=optimazer_list.index(st.session_state.optimizer))
    if optimizer:
        optimizers(optimizer)
    loss_list = ["BinaryCrossentropy", "CategoricalCrossentropy", "MeanSquaredError", "Hinge",
                 "SparseCategoricalCrossentropy", "KLDivergence","BinaryFocalCrossentropy"]
    loss = st.selectbox("loss", loss_list, index=loss_list.index(st.session_state.loss))
    metrics_list = ["Accuracy", "AUC", "Precision", "Recall", "TopKCategoricalAccuracy", "MeanAbsoluteError",
                    "MeanSquaredError", "RootMeanSquaredError", "F1Score"]
    metric_values = {}
    st.subheader("Choose your metrics:")
    for metric in metrics_list:
        metric_values[metric] = st.checkbox(metric, value=st.session_state[metric])
        if metric == "AUC":
            if metric_values[metric]:
                num_thresholds = st.number_input("num_thresholds", value=st.session_state.num_thresholds, step=1,
                                                 min_value=2, help=features_help["num_thresholds"])
                curve_option = ['ROC', "PR"]
                curve = st.selectbox("curve", curve_option, index=curve_option.index(st.session_state.curve),
                                     help=features_help["curve"])
                summation_method_list = ['interpolation', 'minoring', 'majoring']
                summation_method = st.selectbox("summation_method", summation_method_list,
                                                index=summation_method_list.index(st.session_state.summation_method),
                                                help=features_help["summation_method"])
                multi_label = st.checkbox('multi_label', value=st.session_state.multi_label,
                                          help=features_help["multi_label"])
                if multi_label:
                    num_labels = st.number_input("num_labels", value=st.session_state.num_labels,
                                                 help=features_help["num_labels"])
                values = [True, False]
                from_logits = st.radio("from_logits", values, index=values.index(st.session_state.from_logits),
                                       help=features_help["from_logits"])
        if metric == "TopKCategoricalAccuracy":
            if metric_values[metric]:
                k = st.slider("k :", min_value=3, max_value=len(get_classes()), value=st.session_state.k)

    # in summary, for a CNN model used for image classification with multiple classes, set from_logits to False. For a binary classification problem, set from_logits to True because the sigmoid function is applied to the logits to compute the binary cross-entropy loss."""

    values = [True, False]
    steps_per_execution_use = st.radio("steps_per_execution_use", values,
                                       index=values.index(st.session_state.steps_per_execution_use))
    if steps_per_execution_use:
        steps_per_execution = st.slider("steps_per_execution", 1, 100, value=st.session_state.steps_per_execution,
                                        step=1)
    else:
        steps_per_execution = st.session_state.steps_per_execution
    jit_compile = st.checkbox("jit_compile : ", value=st.session_state.jit_compile_c, help=features_help["jit_compile"])
    submittedd = st.button("Save")
    if submittedd:
        st.session_state.validate["compile_2"] = True
        if metric_values["AUC"]:
            st.session_state.num_thresholds = num_thresholds
            st.session_state.curve = curve
            st.session_state.summation_method = summation_method
            st.session_state.multi_label = multi_label
            if multi_label:
                st.session_state.num_labels = num_labels
            else:
                num_labels = None
            st.session_state.from_logits = from_logits

            st.session_state.AUC_dict = {"num_thresholds": num_thresholds, "curve": curve,
                                         "summation_method": summation_method, "multi_label": multi_label,
                                         "from_logits": "from_logits", "num_labels": num_labels}
        if metric_values["TopKCategoricalAccuracy"]:
            st.session_state.k = k

        st.session_state.optimizer = optimizer
        st.session_state.loss = loss
        st.session_state.steps_per_execution_use = steps_per_execution_use
        st.session_state.steps_per_execution = steps_per_execution
        st.session_state.jit_compile_c = jit_compile
        list_metrics = []
        for metric, value in metric_values.items():
            st.session_state[metric] = value
            if value == True:
                list_metrics.append(metric)
        st.session_state["metrics"] = list_metrics
        st.write("Done!")
with st.form("my_form_Callbacks"):
    st.subheader("Callbacks")

    callback_list = ["EarlyStopping", "ModelCheckpoint", "ReduceLROnPlateau"]
    callback_values = {}
    for calback in callback_list:
        print(st.session_state[calback + "_"], calback)
        callback_values[calback] = st.checkbox(calback, value=st.session_state[calback + "_"])
    submitted = st.form_submit_button("Confirm")
    if submitted:
        st.write("Done!")
        the_callback_selected = []
        for callback, values in callback_values.items():
            if values:
                the_callback_selected.append(callback)
            else:
                st.session_state[callback] = None
            st.session_state[callback + "_"] = values
            print(st.session_state[calback + "_"],calback,callback_values)
        st.session_state.your_callback_list = the_callback_selected

        st.experimental_rerun()
        st.write("Done!")
for callback in st.session_state.your_callback_list:
    add_calbacks(callback)
with st.expander("fit",expanded=True):
    st.subheader("Model.fit")
    batch_size = st.slider("batch_size", 1, 400, value=st.session_state.batch_size, step=1)
    epochs = st.slider("epochs", 1, 500, value=st.session_state.epochs, step=1)
    verbose = st.slider("verbose", 1, 2, value=st.session_state.verbose, step=1)
    validation_split = st.slider("validation_split", 0.0, 1.0, value=st.session_state.validation_split, step=0.01)
    value = [False, True]
    shuffle = st.radio("shuffle", value, index=value.index(st.session_state.shuffle))
    list_choise=["None","give it a value"]
    use_step_per_epoch=st.selectbox("steps_per_epoch : ",list_choise,index=list_choise.index(st.session_state.use_step_per_epoch),help="Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch")
    if use_step_per_epoch=="give it a value":
        steps_per_epoch = st.slider("steps_per_epoch", 1, 500, value=st.session_state.steps_per_epoch, step=1,help="Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch")
    use_validation_step=st.selectbox("validation step : ",list_choise,index=list_choise.index(st.session_state.use_validation_step),help="Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch.")
    if use_validation_step=="give it a value":
        validation_steps = st.slider("validation_steps", 1, 500, value=st.session_state.validation_steps, step=1,help="Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch.")
    validation_batch_size = st.slider("validation_batch_size", 1, 500, value=st.session_state.validation_batch_size,
                                      step=1)
    validation_freq = 1
    max_queue_size = 10
    workers = 1
    use_multiprocessing = False
    confirm = st.button("confirm")
    if confirm:
        st.session_state.use_step_per_epoch=use_step_per_epoch
        if use_step_per_epoch!="None":
            st.session_state.steps_per_epoch=steps_per_epoch
        st.session_state.use_validation_step=use_validation_step
        if use_validation_step!="None":
            st.session_state.validation_steps=validation_steps
        st.session_state.validate["fit"] = True
        st.session_state.batch_size = batch_size
        st.session_state.epochs = epochs
        st.session_state.verbose = verbose
        st.session_state.validation_split = validation_split
        st.session_state.shuffle = shuffle
        st.session_state.validation_batch_size = validation_batch_size
        st.write('training ....')

def load_weights_fn():
    if st.session_state.load_weight or not st.session_state.load_weight :
        st.session_state.load_weights=st.session_state.load_weight
def start_from_last_epochs_fn():
    if st.session_state.start_from_last_epochs or not st.session_state.start_from_last_epochs :
        st.session_state.start_from_last_epoch=st.session_state.start_from_last_epochs
load_weights=st.checkbox("Load weights",on_change=load_weights_fn,key="load_weight",value=st.session_state.load_weights)
if load_weights:
    start_from_last_epoch=st.checkbox("start from last epoch",value= st.session_state.start_from_last_epoch,key="start_from_last_epochs",on_change=start_from_last_epochs_fn)

training = st.button("start training ")
if training:


    callbacks,statue = create_all_callbacks()
    if not st.session_state.validate["upload_data"]:
        st.warning("Upload data before training ", icon="⚠️")
    elif ((not st.session_state.validate["data_augmentation_1"] )and (not st.session_state.validate["data_augmentation_2"] ))and  (not st.session_state.validate["image_dataset_from_directory"]):
        st.warning("Confirm Pre-Processing before training ", icon="⚠️")
    elif not st.session_state.validate["confirm_data_splitting"]:
        st.warning('Confirm data spliting', icon="⚠️")
    elif not statue:
        st.warning("Confirm Callbacks before training ", icon="⚠️")
    elif not st.session_state.validate["compile_2"]:
        st.warning("Confirm Compile before training ", icon="⚠️")
    elif not st.session_state.validate["fit"]:
        st.warning("Confirm Fit before training ", icon="⚠️")
    elif not st.session_state.validate["model"]:
        st.warning("Create you model before training ", icon="⚠️")

    else:
        if st.session_state.use_step_per_epoch!="None":
            steps_per_epoch=st.session_state.steps_per_epoch
        else:
            steps_per_epoch=None
        if st.session_state.use_validation_step!="None":
            validation_steps=st.session_state.validation_steps
        else:
            validation_steps=None
        st.session_state.validation_steps_used=validation_steps
        st.session_state.steps_per_epoch_used=steps_per_epoch
        model = create_model()
        print(get_metrics())
        model.compile(optimizer=create_optimazer(),
                      loss=st.session_state.loss,
                      metrics=get_metrics(),
                      jit_compile=st.session_state.jit_compile_c)
        model.build(input_shape=[None] + list(st.session_state.image_shape))
        model.summary(print_fn=lambda x: st.text(x))
        st.write("done")
        if st.session_state.use_data_augmentation == False:
            classes = get_classes()
            with st.spinner("Spliting Data to train and test"):
                done = split_all_data(st.session_state.validation_size_org, classes)
            if st.session_state["fixinig_inbalence"]=="Oversampling":
                st.session_state.max=over_sampling_with_data_augmentation()
            else:
                if st.session_state.max != None:
                    delete_augmented_images()
            with st.spinner("Creating train and test generator"):
                train_datset, val_dataset = image_dataset_from_directory(st.session_state.image_dataset_from_directory, classes)
            st.session_state.val_dataset = val_dataset
        else:
            st.write('spliting all data')
            classes = get_classes()
            with st.spinner("Spliting Data to train and test"):
                done = split_all_data(st.session_state.validation_size_org, classes)

            if st.session_state["fixinig_inbalence"]=="Oversampling":
                st.session_state.max=over_sampling_with_data_augmentation()
            else:
                if st.session_state.max != None:
                    delete_augmented_images()
            with st.spinner("Spliting Data to train and test"):
                train_datset, val_dataset = train_val_generators(st.session_state.data_augmentation_dict,
                                                             st.session_state.flow_from_directory, classes)
            st.session_state.val_dataset = val_dataset
            # from tqdm.keras import TqdmCallback

        oa = st.container()
        oa2 = st.empty()
        if "cap" not in st.session_state:
            st.session_state.cap = io.StringIO()

        st.session_state.ii = 0


        class CustomCallback(keras.callbacks.Callback):
            # def on_epoch_end(self, epoch, logs=None):
            #     pass
            #     capp = st.session_state.cap.getvalue()
            #     st.code(capp)
            #     st.session_state.cap.truncate(0)
            #     st.session_state.cap.seek(0)
            # def on_train_batch_end(self, batch, logs=None):
            # capp = st.session_state.cap.getvalue()
            # st.code(capp)
            # st.session_state.cap.truncate(0)
            # st.session_state.cap.seek(0)
            # def on_predict_end(self, logs=None):

            # def on_test_end(self, logs=None):

            # def on_train_end(self, logs=None):

            def on_train_begin(self, logs=None):
                capp = st.session_state.cap.getvalue()
                oa.text(capp)
                oa.success("Training started")
                st.session_state.cap.truncate(0)
                st.session_state.cap.seek(0)

            def on_train_end(self, logs=None):
                capp = st.session_state.cap.getvalue()
                oa.text(capp)
                oa.success("Finished training.")
                st.session_state.cap.truncate(0)
                st.session_state.cap.seek(0)

            def on_epoch_begin(self, epoch, logs=None):
                capp = st.session_state.cap.getvalue()
                oa.text(capp)
                st.session_state.ii += 1
                oa.text(f"Epoch: {st.session_state.ii}/{st.session_state.epochs}")
                st.session_state.cap.truncate(0)
                st.session_state.cap.seek(0)

            def on_epoch_end(self, epoch, logs=None):
                capp = st.session_state.cap.getvalue()
                oa.text(capp)
                st.session_state.cap.truncate(0)
                st.session_state.cap.seek(0)

            # def on_test_begin(self, logs=None):
            #     capp = st.session_state.cap.getvalue()
            #     st.write(capp)
            #     st.session_state.cap.truncate(0)
            #     st.session_state.cap.seek(0)
            def on_test_end(self, logs=None):
                capp = st.session_state.cap.getvalue()
                oa.text(capp)
                st.session_state.cap.truncate(0)
                st.session_state.cap.seek(0)

            # def on_predict_begin(self, logs=None):
            #     capp = st.session_state.cap.getvalue()
            #     st.write(capp)
            #     st.session_state.cap.truncate(0)
            #     st.session_state.cap.seek(0)
            def on_predict_end(self, logs=None):
                capp = st.session_state.cap.getvalue()
                oa.text(capp)
                st.session_state.cap.truncate(0)
                st.session_state.cap.seek(0)

            # def on_train_batch_begin(self, batch, logs=None):
            #     capp = st.session_state.cap.getvalue()
            #     st.write(capp)
            #     st.session_state.cap.truncate(0)
            #     st.session_state.cap.seek(0)
            def on_train_batch_end(self, batch, logs=None):
                capp = st.session_state.cap.getvalue()
                oa2.text(capp)
                st.session_state.cap.truncate(0)
                st.session_state.cap.seek(0)

            # def on_test_batch_begin(self, batch, logs=None):
            #     capp = st.session_state.cap.getvalue()
            #     st.write(capp)
            #     st.session_state.cap.truncate(0)
            #     st.session_state.cap.seek(0)
            def on_test_batch_end(self, batch, logs=None):
                capp = st.session_state.cap.getvalue()
                oa2.text(capp)
                st.session_state.cap.truncate(0)
                st.session_state.cap.seek(0)

            # def on_predict_batch_begin(self, batch, logs=None):
            #     capp = st.session_state.cap.getvalue()
            #     st.write(capp)
            #     st.session_state.cap.truncate(0)
            #     st.session_state.cap.seek(0)
            def on_predict_batch_end(self, batch, logs=None):
                capp = st.session_state.cap.getvalue()
                oa2.text(capp)
                st.session_state.cap.truncate(0)
                st.session_state.cap.seek(0)


        # with st_stdout("code"):

        callbacks.append(CustomCallback())

        st.session_state.cap.truncate(0)
        st.session_state.cap.seek(0)
        if st.session_state["fixinig_inbalence"] == "add Class weights to loss function":
            if st.session_state.start_from_last_epoch and st.session_state.load_weights:
                st.session_state.prev_history=st.session_state.history
                with contextlib.redirect_stdout(st.session_state.cap):
                    history = model.fit(
                        train_datset,
                        initial_epoch=st.session_state.history.epoch[-1],
                        validation_data=val_dataset,
                        batch_size=st.session_state.batch_size,
                        epochs=st.session_state.epochs+len(st.session_state.history.epoch),
                        verbose=st.session_state.verbose,
                        callbacks=callbacks,
                        class_weight=get_class_weights(),
                        validation_steps=validation_steps,
                        steps_per_epoch=steps_per_epoch
                    )
            else:
                with contextlib.redirect_stdout(st.session_state.cap):
                    history = model.fit(
                        train_datset,
                        validation_data=val_dataset,
                        batch_size=st.session_state.batch_size,
                        epochs=st.session_state.epochs,
                        verbose=st.session_state.verbose,
                        callbacks=callbacks,
                        class_weight=get_class_weights(),
                        validation_steps=validation_steps,
                        steps_per_epoch=steps_per_epoch
                    )
        else :

            if st.session_state.start_from_last_epoch and st.session_state.load_weights:
                st.session_state.prev_history = st.session_state.history
                with contextlib.redirect_stdout(st.session_state.cap):
                    history = model.fit(
                        train_datset,
                        initial_epoch=st.session_state.history.epoch[-1],
                        validation_data=val_dataset,
                        batch_size=st.session_state.batch_size,
                        epochs=st.session_state.epochs + len(st.session_state.history.epoch),
                        verbose=st.session_state.verbose,
                        callbacks=callbacks,
                        validation_steps=validation_steps,
                        steps_per_epoch=steps_per_epoch
                    )
            else:
                with contextlib.redirect_stdout(st.session_state.cap):
                    history = model.fit(
                        train_datset,
                        validation_data=val_dataset,
                        batch_size=st.session_state.batch_size,
                        epochs=st.session_state.epochs,
                        verbose=st.session_state.verbose,
                        callbacks=callbacks,
                        validation_steps=validation_steps,
                        steps_per_epoch=steps_per_epoch
                    )

        st.session_state.history = history
        st.session_state.model = model
        model.save("data/data/training/model.h5")







    # history = model.fit(
    # train_datset,
    # validation_data=val_dataset,
    # batch_size=st.session_state.batch_size,
    # epochs=st.session_state.epochs,
    # verbose=st.session_state.verbose,
    # callbacks=callbacks
    # )
    # st.write("hereeeeeeeeee")

    # #st.line_chart(history.history['loss'])
    # #st.line_chart(history.history['val_loss'])
    # #custom_fit(model, train_datset, val_dataset, epochs, callbacks)
    # st.write("Starting training with {} epochs...".format(epochs))
    # # epochs = 2
    # for epoch in range(epochs):
    #     print("\nStart of epoch %d" % (epoch,))
    #     st.write("Epoch {}".format(epoch + 1))
    #     start_time = time.time()
    #     progress_bar = st.progress(0.0)
    #     percent_complete = 0
    #     epoch_time = 0
    #     # Creating empty placeholder to update each step result in epoch.
    #     st_t = st.empty()

    #     train_loss_list = []
    #     # Iterate over the batches of the dataset.
    #     for step, (x_batch_train, y_batch_train) in enumerate(train_datset):
    #         start_step = time.time()
    #         loss_value = train_step(x_batch_train, y_batch_train)
    #         end_step = time.time()
    #         epoch_time += (end_step - start_step)
    #         train_loss_list.append(float(loss_value))

    #         # Log every 200 batches.
    #         if step % 1 == 0:
    #             print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
    #             print("Seen so far: %d samples" % ((step + 1) * batch_size))
    #             step_acc = float(train_acc_metric.result())#i will add the auc w nchalah timchi min marra loula
    #             step_auc=float(train_auc_metric.result())
    #             percent_complete = ((step/len(train_datset)))
    #             progress_bar.progress(percent_complete)
    #             st_t.write("Duration : {0:.2f}s, Training acc. : {1:.4f}   ,Training AUC : {3:4f}  Loss : {2:4f}".format((epoch_time), float(step_acc),float(loss_value),float(step_auc)))

    #     progress_bar.progress(1.0)
    #     # Display metrics at the end of each epoch.
    #     train_acc = train_acc_metric.result()
    #     train_auc=train_auc_metric.result()
    #     print("Training acc over epoch: %.4f" % (float(train_acc),))
    #     # Reset training metrics at the end of each epoch
    #     train_acc_metric.reset_states()
    #     train_auc_metric.reset_states()
    #     # Find epoch training loss.
    #     print(train_loss_list)
    #     train_loss = round((sum(train_loss_list) / len(train_loss_list)), 5)#5 the number of degit for the loss
    #     val_loss_list = []
    #     # Run a validation loop at the end of each epoch.
    #     for x_batch_val, y_batch_val in val_dataset:
    #         val_loss_list.append(float(test_step(x_batch_val, y_batch_val)))
    #     # Find epoch validation loss.
    #     val_loss = round((sum(val_loss_list) / len(val_loss_list)), 5)
    #     val_acc = val_acc_metric.result()
    #     val_auc=val_auc_metric.result()
    #     val_acc_metric.reset_states()
    #     val_auc_metric.reset_states()
    #     print("Validation acc: %.4f" % (float(val_acc),))
    #     print("Time taken: %.2fs" % (time.time() - start_time))
    #     st_t.write("Duration : {0:.2f}s, Training acc. : {1:.4f},training auc : {5:4f}   Training loss {4:4f} Validation acc.:{2:.4f}   Validation Loss : {3:4f} Validation auc : {6:4f}".format(
    #         (time.time() - start_time), float(train_acc), float(val_acc),float(val_loss),float(train_loss),float(train_auc),float(val_auc)))

