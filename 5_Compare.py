import os
import zipfile
import numpy as np
import shutil
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import keras.backend as K
from sklearn.metrics import roc_curve, roc_auc_score, auc

import matplotlib.pyplot as plt
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

execute_comp=False
st.set_page_config(page_title="Compaire", page_icon=":guardsman:", layout="wide")
if not os.path.exists('test_data'):
    os.makedirs('test_data')
st.set_option('deprecation.showfileUploaderEncoding', False)

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

def load_model(model_path):
    model = tf.keras.models.load_model(model_path,custom_objects={"f1_score_binary":f1_score_binary,"f1_score_categrecal":f1_score_categrecal})
    return model

def extract_zip(file, path_to_extract):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(path_to_extract)

def save_uploaded_file(uploaded_file,i):
    distination_path='model_loid'+str(i)
    if not os.path.exists(distination_path):
        os.makedirs(distination_path)
    else:
        shutil.rmtree(distination_path)
        os.makedirs(distination_path)
    file_path = os.path.join(distination_path, "model.h5")

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

from PIL import ImageOps


def preprocess_images(image_folder,image_dataset_from_directory_dict,rescale,classes):
    image_dataset_from_directory_dict['directory'] = image_folder
    image_dataset_from_directory_dict['class_names'] = classes
    if rescale == "1/255.0":
        rescale_factor = tf.keras.layers.Rescaling(1. / 255)
    elif st.session_state.rescale == "1/127.5":
        rescale_factor = tf.keras.layers.Rescaling(1. / 127.5)
    else:
        rescale_factor = 1

    test_datst = tf.keras.utils.image_dataset_from_directory(**image_dataset_from_directory_dict)
    if rescale_factor != 1:
        test_datst = test_datst.map(lambda x, y: (rescale_factor(x), y))
    return test_datst
def preprocessing_ui(c,i):

    rescale_liste = ['None', '1/255.0', '1/127.5']
    rescale = c.selectbox("rescale", rescale_liste, index=rescale_liste.index(st.session_state["rescale_"+str(i)]),key=i+1)
    ###1
    label_mode_liste = ['categorical', "binary"]
    label_mode = c.selectbox("class mode : ", label_mode_liste,index=label_mode_liste.index(st.session_state['label_mode_'+str(i)]),key=i+100)
    color_mode_liste = ["rgb", "grayscale", "rgba"]
    color_mode = c.selectbox("color mode : ", color_mode_liste,index=color_mode_liste.index(st.session_state["color_mode_"+str(i)]),key=i+200)
    batch_size = c.number_input('batch size : ', format="%d", min_value=1,value=st.session_state["batch_size_"+str(i)],key=i+300)

    target_size_height = c.number_input("target size height : ", format="%d",
                                         min_value=5,value=st.session_state["image_size_"+str(i)][0],key=i+400)

    target_size_width = c.number_input("target size width : ", format="%d", min_value=5,value=st.session_state["image_size_"+str(i)][1],key=i+500)

    interpolation_liste = ["nearest", "bilinear", "bicubic"]
    interpolation = c.selectbox("interpolation", interpolation_liste,index=interpolation_liste.index(st.session_state["interpolation_"+str(i)]),key=i+600)
    crop_to_aspect_ratio=c.checkbox("crop_to_aspect_ratio",st.session_state["crop_to_aspect_ratio_"+str(i)],key=i+700)
    confirm=c.button("confirm",key=i+800)
    if confirm:
        st.session_state["rescale_"+str(i)] = rescale

        st.session_state['label_mode_'+str(i)]=label_mode
        st.session_state["color_mode_"+str(i)]=color_mode
        st.session_state["batch_size_"+str(i)]=batch_size
        st.session_state["image_size_"+str(i)]=(target_size_height,target_size_width)
        st.session_state["interpolation_"+str(i)]=interpolation
        st.session_state["crop_to_aspect_ratio_"+str(i)]=crop_to_aspect_ratio
        #st.write("done!")
        st.session_state.compaire_variable[i]["preprocessing_ui"]={"label_mode":label_mode,"color_mode":color_mode,"batch_size":batch_size,"interpolation":interpolation,
                                                      "image_size":st.session_state["image_size_"+str(i)],"crop_to_aspect_ratio":crop_to_aspect_ratio}
def get_classes(test_dir):
    if len(os.listdir(test_dir))==0:
        if execute_comp:
            st.warning("Uploid test data")
        return []
    else:
        return os.listdir(test_dir)

def get_confusion_metrics(c,model,test_data,label_mode,indice,classes):
    val_images=[]
    val_labels=[]
    val_dataset=test_data
    for images, labels in val_dataset:
        val_images.append(images)
        val_labels.append(labels)
        test_images = np.concatenate(val_images, axis=0)
        test_labels = np.concatenate(val_labels, axis=0)
    test_labels_roc=test_labels
    y_pred = model.predict(test_images)

    if len(classes) > 2:
        y_pred_classes = np.argmax(y_pred, axis=1)
        test_labels=np.argmax(test_labels, axis=1)
    else:
        if label_mode=="binary":
            y_pred_classes = (y_pred > 0.5).astype(int)
        else:
            y_pred_classes = np.argmax(y_pred, axis=1)
            test_labels = np.argmax(test_labels, axis=1)
    report = classification_report(test_labels, y_pred_classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    matrix = confusion_matrix(test_labels, y_pred_classes)

    # Plot the confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='coolwarm')
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(classes)):
        for j in range(len(classes)):

            text = ax.text(j, i, matrix[i, j],
                           ha="center", va="center", color="w")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im)
    c.pyplot(fig)
    c.table(report_df)
    if label_mode == "binary":
        fpr, tpr, _ = roc_curve(test_labels, y_pred)
        roc_auc = roc_auc_score(test_labels, y_pred)
        st.session_state.compaire_variable[indice]["fpr"],st.session_state.compaire_variable[indice]["tpr"],st.session_state.compaire_variable[indice]["roc_auc"]=fpr,tpr,roc_auc
        # Plot ROC curve
        plt.figure(figsize=(5, 3))
        plt.plot(fpr, tpr, color='blue', lw=2,
                 label='ROC curve (area = {0:0.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        c.pyplot(plt.gcf())
    else:
        n_classes=len(classes)
        y_pred_proba = tf.nn.softmax(y_pred, axis=-1).numpy()
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for k in range(n_classes):
            fpr[k], tpr[k], _ = roc_curve(test_labels_roc[:, k], y_pred_proba[:, k])
            roc_auc[k] = auc(fpr[k], tpr[k])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_roc.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        st.session_state.compaire_variable[indice]["fpr"],st.session_state.compaire_variable[indice]["tpr"],st.session_state.compaire_variable[indice]["roc_auc"]=fpr["micro"],tpr["micro"],roc_auc["micro"]
        # Plot ROC curves for each class
        plt.figure(figsize=(5, 3))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))

        # Plot micro-average ROC curve
        plt.plot(fpr["micro"], tpr["micro"], color='blue', lw=2,
                 label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        c.pyplot(plt.gcf())

def roc_curve_for_all_models(c):
    text=""
    if st.session_state.compaire_variable[0]["label_mode"]=="binary":
        text=""
    else:
        text="Micro-average "
    plt.figure(figsize=(5, 3))

    for i in range(len(c)):
        plt.plot(st.session_state.compaire_variable[i]["fpr"], st.session_state.compaire_variable[i]["tpr"], lw=2,
                 label=text+f'ROC curve model {i+1} (area = {st.session_state.compaire_variable[i]["roc_auc"]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    st.pyplot(plt.gcf())
def get_file():

    if st.session_state.ModelCheckpoint_ :
        file_path = st.session_state.ModelCheckpoint["filepath"]
        methode = "Using ModelCheckpoint file"

    elif type(st.session_state.model)is not  None:
        model = st.session_state.model
        if not os.path.exists("data/saved_model"):
            os.makedirs("data/saved_model")
        model.save("data/saved_model/model.h5")
        file_path = "data/saved_model/model.h5"
        methode = "Use the last model.save"
    return file_path, methode
def fixe_glitch_test_data():
    if st.session_state.used_test_dataset or not st.session_state.used_test_dataset:
        st.session_state.use_test_data=st.session_state.used_test_dataset
def use_val_dataset():
    if os.path.exists("data/data/testing"):
        if len(os.listdir("data/data/testing"))>1:
            return "data/data/testing"
        else:
            st.warning("You don't have testing data ")
            return "test_data"
    else:
        st.warning("You don't have testing data ")
        return None

a1,a2,a3 = st.columns([1,1,1])
a2.title("Model Comparison")
use_test_data=st.checkbox("Use Test Data used to validate trained model",value=st.session_state.use_test_data,key="used_test_dataset",on_change=fixe_glitch_test_data)
# use_test_data=st_toggle_switch(
#     label="Use Test Data used to validate trained model",
#     key="used_test_dataset",
#     default_value=st.session_state.use_test_data,
#     label_after=st.session_state.use_test_data,
#     inactive_color="#D3D3D3",  # optional
#     active_color="#11567f",  # optional
#     track_color="#29B5E8",  # optional
#
# )
# if st.session_state.used_test_dataset or not st.session_state.used_test_dataset:
#     st.session_state.use_test_data = st.session_state.used_test_dataset
uploaded_dataset = st.file_uploader("Upload your zipped test dataset", type=["zip"])
button_cmp=st.button("Compaire")
if button_cmp:
    execute_comp=True

b1,b2,b3 = st.columns([4,1,5])
n = a2.number_input("N°  Models", 1, 5, value=2)
if uploaded_dataset is  None :
    if os.path.exists("test_data"):
        shutil.rmtree("test_data")
        os.makedirs('test_data')
    else:
        os.makedirs('test_data')
c = [st]
if n:
    match n:
        case 1:
            c = [st]
        case 2:
            c1, c2 = st.columns([1,1])
            c = [c1, c2]
        case 3:
            c1, c2, c3= st.columns([1,1,1])
            c = [c1, c2, c3]
        case 4:
            c1, c2, c3, c4 = st.columns([1,1,1,1])
            c = [c1, c2, c3, c4]
        case 5:
            c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
            c = [c1, c2, c3, c4, c5 ]



# c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
# c = [c1, c2, c3, c4, c5 ]

if "compaire_variable" not in st.session_state:
    st.session_state.compaire_variable=[{"column":0,"preprocessing_ui":0,"model_path":0,"test_dataset":0,"label_mode":0,"test_labels":"0","y_pred":"0"},
                                        {"column":1,"preprocessing_ui":0,"model_path":0,"test_dataset":0,"label_mode":0,"test_labels":"0","y_pred":"0"},
                                        {"column":2,"preprocessing_ui":0,"model_path":0,"test_dataset":0,"label_mode":0,"test_labels":"0","y_pred":"0"},
                                        {"column":3,"preprocessing_ui":0,"model_path":0,"test_dataset":0,"label_mode":0,"test_labels":"0","y_pred":"0"},
                                        {"column":4,"preprocessing_ui":0,"model_path":0,"test_dataset":0,"label_mode":0,"test_labels":"0","y_pred":"0"}]
dataset_path = "test_data"
if st.session_state.use_test_data:
    dataset_path=use_val_dataset()
    # dataset_path
else:
    dataset_path="test_data"
if uploaded_dataset is not None :
    if not os.path.exists("test_data"):
        os.makedirs('test_data')
    extract_zip(uploaded_dataset, dataset_path)

def fixe_glitch():
    if st.session_state["use_existing_model"] or not st.session_state["use_existing_model"]:
        st.session_state.use_last_model=st.session_state["use_existing_model"]
classes = get_classes(dataset_path)
# dataset_path
for i in range(len(c)):

    name = "Model " + str(i+1)
    c[i].subheader(name)
    test=False

    if i==0:
        use_existing_model=c[0].checkbox("Use the last model created",value=st.session_state.use_last_model,key="use_existing_model",on_change=fixe_glitch)
    uploaded_model = c[i].file_uploader("Upload your .h5 model file", type=["h5"], key=i+1500)

    preprocessing_ui(c[i],i)
    if i==0 and st.session_state.use_last_model and len(classes)!=0 and execute_comp and (type(st.session_state.compaire_variable[i]["preprocessing_ui"])!=int or type(st.session_state.compaire_variable[i]["model_path"])!=int and type(st.session_state.compaire_variable[i]["test_dataset"])!=int):
        with st.spinner("Saving and loading model and dataset..."):
            test = True
            model_file,methode = get_file()
            st.session_state.compaire_variable[i]["model_path"] = model_file
            model = load_model(model_file)


            st.session_state.compaire_variable[i]["label_mode"] = st.session_state.compaire_variable[i]["preprocessing_ui"]["label_mode"]
            test_dataset = preprocess_images(dataset_path, st.session_state.compaire_variable[i]["preprocessing_ui"],st.session_state["rescale_" + str(i)], classes)
            st.session_state.compaire_variable[i]["test_dataset"] = test_dataset
    if uploaded_model is not None  and st.session_state.compaire_variable[i]["preprocessing_ui"] is not int and execute_comp  and len(classes)!=0 :
        with st.spinner("Saving and loading model and dataset..."):
            test=True
            model_file = save_uploaded_file(uploaded_model,i)
            st.session_state.compaire_variable[i]["model_path"]=model_file
            model = load_model(model_file)

            # if not os.path.exists("test_data"):
            #     os.makedirs('test_data')
            # extract_zip(uploaded_dataset, dataset_path)


            st.session_state.compaire_variable[i]["label_mode"]=st.session_state.compaire_variable[i]["preprocessing_ui"]["label_mode"]
            #c[i].write(st.session_state.compaire_variable[i]["preprocessing_ui"])
            test_dataset = preprocess_images(dataset_path,st.session_state.compaire_variable[i]["preprocessing_ui"],st.session_state["rescale_"+str(i)],classes)
            st.session_state.compaire_variable[i]["test_dataset"]=test_dataset
    elif st.session_state.compaire_variable[i]["model_path"] is not int and st.session_state.compaire_variable[i]["test_dataset"] is not int and len(classes)!=0 and execute_comp and type(st.session_state.compaire_variable[i]["model_path"])!=int :
        test=True
        model = load_model(st.session_state.compaire_variable[i]["model_path"])
        # dataset_path = "test_data"
        # if not os.path.exists("test_data"):
        #     os.makedirs('test_data')
        # extract_zip(uploaded_dataset, dataset_path)

    if test and execute_comp:
        with st.spinner("Evaluating model..."):
            get_confusion_metrics(c[i], model, st.session_state.compaire_variable[i]["test_dataset"],
                                  st.session_state.compaire_variable[i]["label_mode"], i, classes)
button=st.button("Get ROC graph")
if button:
    roc_curve_for_all_models(c)
# st.session_state