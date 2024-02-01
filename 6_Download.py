import streamlit as st
import os

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


import json
# def upload_data():
#     root = "data/images/images/"
#     number_of_classes = len(os.listdir(os.path.join(root)))
#
#     model =  f"""
# import os
# import zipfile
#
# #Choose your data location (change the name)
# test_local_zip = '{st.session_state.data_name.name}'
#
# #Unzip
# zip_ref = zipfile.ZipFile(test_local_zip, 'r')
# zip_ref.extractall('./training')
# zip_ref.close()
#
# # Define the training base directorie
# train_dir = './training'
#
# # Print classes names and how many images they contain
# number_of_classes = len(os.listdir(train_dir))
# for i in range(number_of_classes):
#     name = os.listdir(train_dir)[i]
#     count = len(os.listdir(os.path.join(train_dir, name)))
#     print ("Name: ", name, ", Count: ", count)
#
#
#     """
#     return model
#
# def PreProcessing():
#     pass
#
# def Model():
#     model = f"""
# #Building the Model
# import keras
# model = keras.Sequential()"""
#     k = 4
#     for i in range(len(st.session_state.layers)):
#         layer = st.session_state.layers[i]
#         if layer.type == "Conv2D":
#             filters = layer.parameters["filters"]
#             kernel = layer.parameters["kernel_size"]
#             strides = layer.parameters["strides"]
#             padding = layer.parameters["padding"]
#             activation = layer.parameters["activation"]
#             kernel_regularizer = layer.parameters["kernel_regularizer"]
#             model += f"""
# model.add(keras.layers.Conv2D({filters}, {kernel}, strides = {strides}, padding = "{padding}", activation = "{activation}", kernel_regularizer = {kernel_regularizer}))"""
#     return model






if st.button("Download Model Code"):
    import json
    json_data={}
    # Sample dictionaries
    json_data["split_size"] =st.session_state.validation_size_org
    json_data["use_data_augmentation"]=st.session_state.use_data_augmentation
    if st.session_state.use_data_augmentation == False:
        json_data["image_dataset_from_directory_dict"] =st.session_state.image_dataset_from_directory
        json_data["rescale"]=st.session_state.rescale
    else:
        json_data["train_data_augmentation"]= st.session_state.data_augmentation_dict
        json_data["flow_from_directory"]=st.session_state.flow_from_directory
    json_data["image_shape"]=st.session_state.image_shape
    json_data["all_layers_names"]=st.session_state.all_layers_names
    json_data["all_parameters"]=st.session_state.all_parameters
    json_data['the_optimizer']=st.session_state.the_optimizer
    json_data['metrics']=st.session_state["metrics"]
    json_data["AUC_dict"]=st.session_state.AUC_dict
    json_data["k_top"]=st.session_state.k
    json_data['loss']=st.session_state.loss
    json_data['jit_compile_c']=st.session_state.jit_compile_c
    json_data["batch_size"]=st.session_state.batch_size
    json_data["epochs"]=st.session_state.epochs
    json_data['verbose']=st.session_state.verbose
    json_data['transfer_learning_methode']=st.session_state.transfer_learning_methode
    json_data['URL']=st.session_state.URL
    json_data['trainable']=st.session_state.trainable
    json_data['pre_trained_model_dict']=st.session_state.pre_trained_model_dict
    json_data["EarlyStopping"]=st.session_state["EarlyStopping"]
    json_data["ModelCheckpoint"]=st.session_state["ModelCheckpoint"]
    json_data["ReduceLROnPlateau"]=st.session_state["ReduceLROnPlateau"]
    json_data["validation_steps"]=st.session_state.validation_steps_used
    json_data["fixinig_inbalence"]=st.session_state["fixinig_inbalence"]
    json_data['steps_per_epoch']=st.session_state.steps_per_epoch_used
    if type(json_data['pre_trained_model_dict'])==dict:
        json_data['pre_trained_model_dict'].pop("model_classe")

    if 'layer_name' not in st.session_state:
        json_data["chose_layer_output"] = None
    else:
        json_data["chose_layer_output"]=st.session_state.layer_name
    json_data['trainable_options']=st.session_state.trainable_options
    json_data['number_layers_trainable_false']=st.session_state.number_layers_trainable_false

    json_data['use_transfer_learning']=st.session_state.use_transfer_learning
    with open("data.json", "w") as json_file:
        json.dump(json_data, json_file)
    st.download_button(label="Download json Code",
                       data="data.json",
                        file_name="data.json",
                        )
    st.download_button(label="Download model code ",data="code_to_download.py",file_name="code_to_download.py")
    # Creating a dictionary to store the individual dictionaries
    # # Generate the model code
    # model_code = upload_data()
    # model_code += Model()
    # # Set the download filename
    # filename = "my_model.py"
    #
    # # Set the content type
    # content_type = "text/x-python"
    #
    # # Create the download link
    # st.download_button(label="Download Model Code",
    #                    data=model_code,
    #                    file_name=filename,
    #                    mime=content_type)
# def get_base64_encoded_file(file_path):
#     chunk_size = 1192  # Adjust the chunk size according to your needs
#     encoded_string = ""
#     with open(file_path, 'rb') as file:
#         while True:
#             chunk = file.read(chunk_size)
#             if not chunk:
#                 break
#             encoded_string += base64.b64encode(chunk).decode('utf-8')
#     return encoded_string
#
# if os.path.exists("data/data/training/model.h5") and st.session_state.model is not None:
#     if st.button("Save Model"):
#         try:
#             file_path = "data/data/training/model.h5"
#             encoded_file = get_base64_encoded_file(file_path)
#             href = f'<a href="data:application/octet-stream;base64,{encoded_file}" download="model.h5">Click here to download the model</a>'
#             st.markdown(href, unsafe_allow_html=True)
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")





def download_model():
    if type(st.session_state.ModelCheckpoint)==dict:
        file_path=st.session_state.ModelCheckpoint["filepath"]
    else:
        file_path = "data/data/training/model.h5"

    # Check if the download button is clicked
    if st.button("Download Model"):
        st.markdown("Downloading model... Please wait!")

        # Read the file and prepare for download
        with open(file_path, "rb") as file:
            model_data = file.read()

        # Provide the file data for download
        st.download_button(
            label="Click to download",
            data=model_data,
            file_name="model.h5",
            mime="application/octet-stream"
        )
if os.path.exists("data/data/training/model.h5") :#and st.session_state.model is not None:
    download_model()