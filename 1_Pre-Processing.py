import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt

# st.sidebar.success("Data augmentation is optional you can leave it empy ")
st.title("Pre-Processing")
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



#####tf.keras.preprocessing.image.ImageDataGenerator is a data generator that reads images and applies data augmentation techniques.

###1

feature_help={
    "featurewise_center":"If True, the mean pixel value of the entire dataset is subtracted from each image. If False, no mean subtraction is performed",
    "samplewise_center": " If True, the mean pixel value of each individual image is subtracted from that image. If False, no mean subtraction is performed",
    "featurewise_std_normalization": "if True, the pixel values of the entire dataset are divided by the standard deviation of the pixel values of the entire dataset. If False, no normalization is performed.",
    "samplewise_std_normalization":"set to True to divide the pixel values of each input image by the standard deviation of that image only. This can help to scale the pixel values to a similar range, which can be useful if different images have different brightness or contrast levels",
    "zca_whitening": "Boolean, set to True to apply ZCA whitening to the input data. This can help to decorrelate the pixel values and reduce redundancy in the input data.",
    "zca_epsilon": "Float, a small positive constant to avoid division by zero in ZCA whitening.",
    "rotation_range":"nt, range (in degrees) for random rotations applied to the images during training",
    "width_shift_range": "range for random horizontal shifts applied to the images during training",
    "height_shift_range": "range for random vertical shifts applied to the images during training",
    "brightness_range": "Tuple of two floats, range for random brightness adjustments applied to the images during training",
    "shear_range": "Float, range (in degrees) for random shearing transformations applied to the images during training",
    "zoom_range": "range for random zooming applied to the images during training",
    "channel_shift_range": "Float, range for random channel shifts applied to the images during training.",
    "fill_mode": "Method for filling in newly created pixels during image transformations",
    "cval":"Float or Int, value used for filling in newly created pixels during image transformations when fill_mode is 'constant'",
    "horizontal_flip": " Boolean, set to True to randomly flip images horizontally during training",
    "vertical_flip": "Boolean, set to True to randomly flip images vertically during training.",
    "validation_split": " Float, fraction of the data to reserve for validation",
    "interpolation_order": 1,
}
def get_all_y():
    classes=get_classes()
    if classes==None:
        st.session_state.validate["upload_data"]=False
        return None,None
    else:
        root_directory="data/images/images"
        y = []
        i = 0
        for folder in classes:
            y = y + [i] * len(os.listdir(os.path.join(root_directory, folder)))
            i = i + 1
        return np.array(y), classes
def show_histogram_classes_frequency():
    y,classes=get_all_y()
    if classes!=None:
        fig, ax = plt.subplots()
        plt.hist(y)
        plt.xticks(range(len(classes)), classes)
        plt.figure(figsize=(6, 4))
        # Set the x-axis label
        plt.xlabel('Label')

        # Set the y-axis label
        plt.ylabel('Frequency')
        # st.plot(fig)
        return fig
    else:

        return None
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
            print(l)
            return l

def fixe_glitch_problem():
    if st.session_state.data_augmnt or not st.session_state.data_augmnt:
        st.session_state.use_data_augmentation=st.session_state.data_augmnt
def data_augmentation(type,i):
    with st.expander(type,expanded=True):
        if type=="data_augmentation":

            st.subheader("Data augmentation")
            rescale_liste = ['None', '1/255.0']
            rescale = st.selectbox("rescale", rescale_liste, index=rescale_liste.index(st.session_state.rescale))
            ###1
            values = [True, False]
            default_ix = values.index(st.session_state["featurewise_center"])
            featurewise_center = st.radio(
                "featurewise_center", values, index=default_ix, horizontal=True, help=feature_help['featurewise_center']
            )
            values = [True, False]
            default_ix = values.index(st.session_state["samplewise_center"])
            samplewise_center = st.radio(
                "samplewise_center", values, index=default_ix, horizontal=True, help=feature_help['samplewise_center']
            )

            ###3 featurewise_std_normalization
            values = [True, False]
            default_ix = values.index(st.session_state["featurewise_std_normalization"])
            featurewise_std_normalization = st.radio(
                "featurewise_std_normalization", values, index=default_ix, horizontal=True,
                help=feature_help['featurewise_std_normalization']
            )

            ###4 samplewise_std_normalization
            values = [True, False]
            default_ix = values.index(st.session_state["samplewise_std_normalization"])
            samplewise_std_normalization = st.radio(
                "samplewise_std_normalization", values, index=default_ix, horizontal=True,
                help=feature_help['samplewise_std_normalization']
            )

            ###5 zca_whitening
            values = [True, False]
            default_ix = values.index(st.session_state["zca_whitening"])
            zca_whitening = st.radio("zca_whitening", values, index=default_ix, horizontal=True,
                                     help=feature_help['zca_whitening'])

            ###6 zca_epsilon
        else:
            st.subheader("Data augmentation for OverSampling images")

        # IDK what is this param values

        ###7 rotation_range
        default_ix = st.session_state["rotation_range"]
        rotation_range = st.slider(
            "rotation_range", 0, 180, value=default_ix, step=5, help=feature_help['rotation_range'],key=i
        )

        ###8 width_shift_range
        default_ix = st.session_state["width_shift_range"]
        width_shift_range = st.slider(
            "width_shift_range",
            0.0,
            1.0,
            value=default_ix,
            step=0.1, help=feature_help['width_shift_range'],key=i+1
        )

        ###9 height_shift_range
        default_ix = st.session_state["height_shift_range"]
        height_shift_range = st.slider(
            "height_shift_range",
            0.0,
            1.0,
            value=default_ix,
            step=0.1, help=feature_help['height_shift_range'],key=i+2
        )

        ###10 brightness_range
        default_ix = st.session_state["brightness_range"]
        if default_ix == None:
            default_ix = (0.0, 0.0)
        help = "The brightness_range parameter allows you to specify a random range of scalar factors to use for brightness adjustments. For example, setting brightness_range=(0.5, 1.5) means that the brightness of the input images can be randomly adjusted by a scalar factor between 0.5 and 1.5."
        brightness_range = st.slider(
            "brightness_range", 0.0, 5.0, default_ix, step=0.1, help=help,key=i+3
        )

        ###11 shear_range
        default_ix = st.session_state["shear_range"]
        shear_range = st.slider("shear_range", 0.0, 0.5, value=default_ix, step=0.1, help=feature_help['shear_range'],key=i+4)

        ###12 zoom_range
        default_ix = st.session_state["zoom_range"]
        help = "If zoom_range represents the maximum amount of zoom as a fraction of the original size. For example, setting zoom_range=0.2 means that the input images can be zoomed in or out by a random factor between 0.8 and 1.2 times their original size."
        zoom_range = st.slider(
            "zoom_range", 0.0, 0.5, value=default_ix, step=0.1, help=help,key=i+5
        )

        # 13 channel_shift_range
        default_ix = st.session_state["channel_shift_range"]
        help = "it represents the maximum amount of channel shift as a fraction of the maximum intensity value. For example, setting channel_shift_range=0.2 means that the intensity values of the red, green, and blue channels of the input images can be randomly shifted by a maximum of 0.2 times the maximum intensity value."
        channel_shift_range = st.slider(
            "channel_shift_range", 0.0, 50.0, value=default_ix, step=0.1, help=help,key=i+6
        )

        ###14 fill_mode
        values = ["nearest", "constant", "reflect", "wrap"]
        default_ix = values.index(st.session_state["fill_mode"])
        fill_mode = st.radio("fill_mode", values, index=default_ix, horizontal=True, help=feature_help['fill_mode'],key=i+7)

        ###15 cval
        default_ix = st.session_state["cval"]
        cval = st.number_input("cval", value=default_ix, help=feature_help['cval'],key=i+8)

        ###16 horizontal_flip
        values = [True, False]
        default_ix = values.index(st.session_state["horizontal_flip"])
        horizontal_flip = st.radio(
            "horizontal_flip", values, index=default_ix, horizontal=True, help=feature_help['horizontal_flip'],key=i+9
        )

        ###17 vertical_flip
        values = [True, False]
        default_ix = values.index(st.session_state["vertical_flip"])
        vertical_flip = st.radio("vertical_flip", values, index=default_ix, horizontal=True,
                                 help=feature_help['vertical_flip'],key=i+10)





        ###22 interpolation_order
        # no

        ###23 dtype
        # no

        ###24 save_to_dir
        ###25 save_prefix
        ### (maybe)we can add button to preview the new images and option to download them

        # The form button
        submitted = st.button("Save",key=i+12)
        if submitted:

            st.session_state["horizontal_flip"] = horizontal_flip
            st.session_state["vertical_flip"] = vertical_flip
            st.session_state["cval"] = cval
            st.session_state["fill_mode"] = fill_mode
            st.session_state["channel_shift_range"] = channel_shift_range
            st.session_state["zoom_range"] = zoom_range
            st.session_state["shear_range"] = shear_range
            st.session_state["brightness_range"] = brightness_range
            st.session_state["width_shift_range"] = width_shift_range
            st.session_state["height_shift_range"] = height_shift_range
            st.session_state["rotation_range"] = rotation_range
            if type == "data_augmentation":
                st.session_state.validate["data_augmentation_1"]=True
                st.session_state["zca_whitening"] = zca_whitening
                st.session_state.rescale = rescale
                st.session_state["featurewise_center"] = featurewise_center
                st.session_state["samplewise_center"] = samplewise_center
                st.session_state["samplewise_std_normalization"] = samplewise_std_normalization
                st.session_state[
                    "featurewise_std_normalization"
                ] = featurewise_std_normalization
                if st.session_state.rescale == "1/255.0":
                    rescale = 1 / 255.0
                else:
                    rescale = None
                st.session_state.data_augmentation_dict={"featurewise_center":st.session_state["featurewise_center"],
                                                     "samplewise_center" :st.session_state["samplewise_center"],
                                                     "featurewise_std_normalization" :st.session_state["featurewise_std_normalization"],
                                                     "samplewise_std_normalization" :st.session_state["samplewise_std_normalization"],
                                                     "zca_whitening" : st.session_state["zca_whitening"],
                                                     "rotation_range": st.session_state["rotation_range"],
                                                     "width_shift_range": st.session_state["width_shift_range"],
                                                     "height_shift_range" :st.session_state["height_shift_range"],
                                                     "rescale":rescale,
                                                     "brightness_range":st.session_state["brightness_range"],
                                                     "shear_range":st.session_state["shear_range"],
                                                     "zoom_range": st.session_state["zoom_range"],
                                                     "channel_shift_range":st.session_state["channel_shift_range"],
                                                     "fill_mode":  st.session_state["fill_mode"],
                                                     "cval":st.session_state["cval"],
                                                     "horizontal_flip": st.session_state["horizontal_flip"],
                                                     "vertical_flip": st.session_state["vertical_flip"]

            }
                st.write("Done!")
            else:
                st.session_state.data_augmentation_over_sampling_dict={
                                                     "rotation_range": st.session_state["rotation_range"],
                                                     "width_shift_range": st.session_state["width_shift_range"],
                                                     "height_shift_range" :st.session_state["height_shift_range"],

                                                     "brightness_range":st.session_state["brightness_range"],
                                                     "shear_range":st.session_state["shear_range"],
                                                     "zoom_range": st.session_state["zoom_range"],
                                                     "channel_shift_range":st.session_state["channel_shift_range"],
                                                     "fill_mode":  st.session_state["fill_mode"],
                                                     "cval":st.session_state["cval"],
                                                     "horizontal_flip": st.session_state["horizontal_flip"],
                                                     "vertical_flip": st.session_state["vertical_flip"]
            }
                st.write("done!")
################################
################################
###starting the form for data augmentation
################################
################################
show_histogramme=st.checkbox("Show histogramme for data distrubtion")
if show_histogramme:
    try:
        fig=show_histogram_classes_frequency()
        st.pyplot(fig)
    except:
        st.error("some error happen can you uploid data preperly!")
def fixe_inbalence_bug_solution():
    if st.session_state.fixe_inbalence_option:
        st.session_state.fixinig_inbalence=st.session_state.fixe_inbalence_option
fixinig_inbalence_data_option=[" ","Undersampling","Oversampling","add Class weights to loss function"]

fixinig_inbalence=st.selectbox("chose methode for fixing inbalence data", fixinig_inbalence_data_option,index=fixinig_inbalence_data_option.index(st.session_state["fixinig_inbalence"]),key="fixe_inbalence_option",on_change=fixe_inbalence_bug_solution)
if st.session_state.fixinig_inbalence=="Oversampling":
    st.session_state.show_data_over_sampling_data_augmentation_option=True
else:
    st.session_state.show_data_over_sampling_data_augmentation_option=False
if st.session_state.show_data_over_sampling_data_augmentation_option:
    data_augmentation("over_sampling",i=100)
st.subheader('spliting data to training and testing')
validation_size_org=st.slider("chose testing size",0.00,1.0,step=0.05,value=st.session_state.validation_size_org)
confirm=st.button("confirm data spliting")
if confirm:
    st.session_state.validate["confirm_data_splitting"] = True
    st.session_state.validation_size_org=validation_size_org
data_augmentationn=st.checkbox("use data augmentation :",value=st.session_state.use_data_augmentation,on_change=fixe_glitch_problem,key="data_augmnt")

if data_augmentationn:
    data_augmentation("data_augmentation",i=1)
    with st.expander("flow from directory ",expanded=True):
        st.subheader("target_size")
        col1, buff, col2 = st.columns([2, 1, 2])
        with col1:
            target_size_height = st.number_input("target size height : ", format="%d",
                                                 min_value=5, value=st.session_state.target_size[0])
        with col2:
            target_size_width = st.number_input("target size width : ", format="%d", min_value=5,
                                                value=st.session_state.target_size[1])
        color_mode_liste= ["rgb", "grayscale", "rgba"]
        color_mode = st.selectbox("color mode : ", color_mode_liste,index=color_mode_liste.index(st.session_state.color_mode))
        class_mode_liste=['categorical', "binary"]
        class_mode = st.selectbox("class mode : ", class_mode_liste,index=class_mode_liste.index(st.session_state['class_mode']))
        batch_size = st.number_input('batch size : ', format="%d", min_value=1,value=st.session_state.batch_size)
        shuffle = st.checkbox("shuffle",value=st.session_state.shuffle)
        if shuffle:
            seed = st.number_input('seed',value=st.session_state.seed)
        else:
            seed=None
        interpolation_liste=["nearest", "bilinear","bicubic"]
        interpolation=st.selectbox("interpolation",interpolation_liste)
        keep_aspect_ratio=st.checkbox(" keep_aspect_ratio",value=st.session_state.keep_aspect_ratio)
        confirm=st.button("confirm")
        if confirm:
            st.session_state.validate["data_augmentation_2"]=True
            st.session_state.target_size = (target_size_height, target_size_width)
            st.session_state.color_mode=color_mode
            st.session_state['class_mode']=class_mode
            st.session_state.batch_size=batch_size
            st.session_state.validation_batch_size=batch_size
            st.session_state.shuffle=shuffle
            if shuffle:
                st.session_state.seed=seed
            st.session_state.interpolation = interpolation
            st.session_state.keep_aspect_ratio=keep_aspect_ratio
            st.session_state.flow_from_directory={"class_mode":st.session_state.class_mode,"color_mode":st.session_state.color_mode,"batch_size":st.session_state.batch_size,"interpolation":st.session_state.interpolation,
                                                  "target_size":st.session_state.target_size,"keep_aspect_ratio":keep_aspect_ratio,'seed':seed,"shuffle":st.session_state.shuffle}
            st.write("done")



else:
    with st.expander("image_dataset_from_directory",expanded=True):
        rescale_liste = ['None', '1/255.0', '1/127.5 - 1']
        rescale = st.selectbox("rescale", rescale_liste, index=rescale_liste.index(st.session_state.rescale),help="rescale using recale layer from Tensorflow : To rescale an input in the [0, 255] range to be in the [0, 1] range, you would pass scale=1./255. To rescale an input in the [0, 255] range to be in the [-1, 1] range, you would pass scale=1./127.5, offset=-1.")
        ###1
        label_mode_liste = ['categorical', "binary"]
        label_mode = st.selectbox("class mode : ", label_mode_liste,index=label_mode_liste.index(st.session_state['label_mode']))
        color_mode_liste = ["rgb", "grayscale", "rgba"]
        color_mode = st.selectbox("color mode : ", color_mode_liste,index=color_mode_liste.index(st.session_state.color_mode))
        batch_size = st.number_input('batch size : ', format="%d", min_value=1,value=st.session_state.batch_size)
        col1, buff, col2 = st.columns([2, 1, 2])
        with col1:
            target_size_height = st.number_input("target size height : ", format="%d",
                                                 min_value=5,value=st.session_state.image_size[0])
        with col2:
            target_size_width = st.number_input("target size width : ", format="%d", min_value=5,value=st.session_state.image_size[1])
        shuffle = st.checkbox("shuffle",value=st.session_state.shuffle)
        if shuffle:
            seed = st.number_input('seed',value=st.session_state.seed)

        interpolation_liste = ["nearest", "bilinear", "bicubic"]
        interpolation = st.selectbox("interpolation", interpolation_liste,index=interpolation_liste.index(st.session_state.interpolation))
        crop_to_aspect_ratio=st.checkbox("crop_to_aspect_ratio",st.session_state.crop_to_aspect_ratio)
        confirm=st.button("confirm")
        if confirm:
            st.session_state.rescale = rescale
            st.session_state.validate["image_dataset_from_directory"] = True
            st.session_state['label_mode']=label_mode
            st.session_state.color_mode=color_mode
            st.session_state.batch_size=batch_size
            st.session_state.validation_batch_size=batch_size
            st.session_state.image_size=(target_size_height,target_size_width)
            if shuffle:
                st.session_state.seed=seed
            st.session_state.interpolation=interpolation
            st.session_state.crop_to_aspect_ratio=crop_to_aspect_ratio
            st.session_state.image_dataset_from_directory={"label_mode":st.session_state.label_mode,"color_mode":st.session_state.color_mode,"batch_size":st.session_state.batch_size,"interpolation":st.session_state.interpolation,
                                                          "image_size":st.session_state.image_size,"crop_to_aspect_ratio":crop_to_aspect_ratio,'seed':st.session_state.seed}
            st.write("done")


