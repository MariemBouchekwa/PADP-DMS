import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
from PIL import Image
import shutil
import zipfile
import pandas as pd
import keras.backend as K
import base64

def get_classes_0():
  if not (os.path.isdir("data/images/images")):
    # st.warning("You have to Uploid data properly first " ,icon="⚠️")
    return None
  else:
    l = os.listdir("data/images/images")
    if len(l) == 0:
      # st.warning("You have to Uploid data properly first ", icon="⚠️")
      return None
    else:
      return l


# path = "C:/Users/youss/Pictures/icons/ds-2.png"
# add_logo(path)
classes_0 = get_classes_0()
if classes_0 != None:
  num_labels = len(classes_0)
else:
  num_labels = 5
features = {
  "validate": {
    "transfer_learning_1": False,
    "transfer_learning_2": False,
    "upload_data": False,
    "confirm_data_splitting": False,
    "image_dataset_from_directory": False,
    "data_augmentation_1": False,
    "data_augmentation_2": False,
    "input_layer": False,
    "model": False,
    "compile_1": False,
    "compile_2": False,
    "fit": False
  },
  "choise_data_form": "labeled Folder",
  "change_classes_names": False,
  "all_classes": "classe_1,classe_2,classe_3",
  "classe_column": " ",
  "images_file_path": " ",
  "all_parameters": None,
  "all_layers_names": None,
  "used_test_dataset": False,
  "use_test_data": False,
  "use_existing_model": False,
  "use_last_model": False,
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
  "data_augmentation_over_sampling_dict": None,
  "show_data_over_sampling_data_augmentation_option": False,
  "fixinig_inbalence": " ",
  "use_data_augmentation": False,
  "data_augmentation_dict": None,
  "validation_size_org": 0.01,
  "rescale": 'None',
  "flow_from_directory": None,
  "image_dataset_from_directory": None,
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
  "not_showing_other_expander": False,
  "show_base_model_summary": False,
  "trainable": False,
  "image_shape": (256, 256, 3),
  "number_layers_trainable_false": 2,
  "trainable_options": "all layers Trainable False",
  "last_layer": "",
  'base_model': None,
  "pre_trained_model_dict": None,
  "width": 256,
  "height": 256,
  "channels": 3,
  "include_top": True,
  "weights": 'imagenet',
  "pooling": "None",
  "classes": 1000,
  "classifier_activation": 'softmax',
  "transfer_learning_methode": "URL from  Tensorflow hub",
  "URL": "",
  "use_transfer_learning": False,
  "model_names": "Vgg19"
}
if "validate" not in st.session_state:
  for feature, default_value in features.items():
    if feature not in st.session_state:
      st.session_state[feature] = default_value


def plot_graphs(history, string):
  fig, ax = plt.subplots()
  if st.session_state.start_from_last_epoch:
    plt.plot(st.session_state.prev_history.history[string]+ history.history[string])
    if 'val_' + string in history.history and  'val_' + string in st.session_state.prev_history.history:
      plt.plot(st.session_state.prev_history.history["val_"+string]+ history.history["val_"+string])
      plt.legend([string, 'val_' + string])
    else:
      plt.legend([string])
    plt.plot([len(st.session_state.prev_history.epoch), len(st.session_state.prev_history.epoch)],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend([string, 'val_' + string])
  else:
    plt.plot(history.history[string])
    if 'val_'+string in history.history:
      plt.plot(history.history['val_'+string])
      plt.legend([string, 'val_' + string])
    else:
      plt.legend([string])
  plt.xlabel("Epochs")
  plt.ylabel(string)

  return fig

def f1_score_binary(y_true, y_pred):  # taken from old keras source code
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  recall = true_positives / (possible_positives + K.epsilon())
  f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
  return f1_val
def get_classes():
  root_directory = "data/data/testing"
  if os.path.exists(root_directory):
    return os.listdir(root_directory)
  else:
    return os.listdir("data/data/images")

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
def get_confusion_metrics():

  if st.session_state.ModelCheckpoint_:
    if not(st.session_state.ModelCheckpoint["save_weights_only"]):
      if st.session_state.F1Score:
        if st.session_state.F1Score_mode>1:
           model=tf.keras.models.load_model(filepath=st.session_state.ModelCheckpoint['filepath'],custom_objects={"f1_score_categrecal":f1_score_categrecal})#os.path.join('PFE_CODE',st.session_state.ModelCheckpoint['filepath']))
        else:
          model = tf.keras.models.load_model(filepath=st.session_state.ModelCheckpoint['filepath'], custom_objects={
            "f1_score_binary": f1_score_binary})  # os.path.join('PFE_CODE',st.session_state.ModelCheckpoint['filepath']))
      else:
        model = tf.keras.models.load_model(filepath=st.session_state.ModelCheckpoint['filepath'])
    else:
      model=st.session_state.model
      model.load_weights(filepath=st.session_state.ModelCheckpoint['filepath'])
  else:
      model=st.session_state.model

  val_images=[]
  val_labels=[]
  if st.session_state.use_data_augmentation:
    val_generator=st.session_state.val_dataset
    for i in range(len(val_generator)):
      batch = val_generator[i]
      images, labels = batch
      val_images.append(images)
      val_labels.append(labels)
  else:
    val_dataset=st.session_state.val_dataset
    for images, labels in val_dataset:
      val_images.append(images)
      val_labels.append(labels)
  test_images = np.concatenate(val_images, axis=0)
  test_labels = np.concatenate(val_labels, axis=0)
  y_pred = model.predict(test_images)
  if len(get_classes())>2:
    y_pred_classes = np.argmax(y_pred, axis=1)
    test_labels = np.argmax(test_labels, axis=1)
  else:
    if st.session_state.use_data_augmentation and st.session_state.flow_from_directory['class_mode']=="binary" or (not (st.session_state.use_data_augmentation) and st.session_state.image_dataset_from_directory['label_mode']=="binary"):
      y_pred_classes = (y_pred > 0.5).astype(int)
    else:
      test_labels = np.argmax(test_labels, axis=1)
      y_pred_classes = np.argmax(y_pred, axis=1)

  report = classification_report(test_labels, y_pred_classes, output_dict=True)
  report_df = pd.DataFrame(report).transpose()
  matrix = confusion_matrix(test_labels, y_pred_classes)

  classes=get_classes()
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
  st.pyplot(fig)
  st.table(report_df)

def zip_folder(folder_path, output_path):
  """
  Compresses the folder at folder_path and saves it to output_path
  """
  with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_obj:
    # Iterate over all the files in the folder
    for foldername, subfolders, filenames in os.walk(folder_path):
      for filename in filenames:
        # Get the full path of the file
        file_path = os.path.join(foldername, filename)
        # Add the file to the zip folder
        zip_obj.write(file_path)


# Example usage

  # Print the classification report
def prepare_test_directory():
  root_dir = "data/testing_with_classes"
  if os.path.exists(root_dir):
    shutil.rmtree(root_dir)
  os.mkdir(root_dir)

  classes=get_classes()
  for classe in classes:
    os.mkdir(os.path.join(root_dir,classe))

def test():
  prepare_test_directory()
  testing_dir="data/testing_with_classes"
  if st.session_state.ModelCheckpoint_:
    st.write(st.session_state.ModelCheckpoint["filepath"])
    if not(st.session_state.ModelCheckpoint["save_weights_only"]):
      if st.session_state.F1Score:
        if st.session_state.F1Score_mode>1:
           model=tf.keras.models.load_model(filepath=st.session_state.ModelCheckpoint['filepath'],custom_objects={"f1_score_categrecal":f1_score_categrecal})#os.path.join('PFE_CODE',st.session_state.ModelCheckpoint['filepath']))
        else:
          model = tf.keras.models.load_model(filepath=st.session_state.ModelCheckpoint['filepath'], custom_objects={
            "f1_score_binary": f1_score_binary})  # os.path.join('PFE_CODE',st.session_state.ModelCheckpoint['filepath']))
      else:
        model = tf.keras.models.load_model(filepath=st.session_state.ModelCheckpoint['filepath'])
    else:
      model=st.session_state.model
      model.load_weights(filepath=st.session_state.ModelCheckpoint['filepath'])
  else:
      model=st.session_state.model
  if st.session_state.use_data_augmentation:
    target_size=st.session_state.target_size
  else:
    target_size=st.session_state.image_size
  test=False
  if st.session_state.color_mode=="grayscale":
    test=True
  if st.session_state.rescale=="1/255.0":
    x=1/255.0
  elif st.session_state.rescale=='1/127.5':
    x=1/127.5
  else:
    x=1
  for fn in os.listdir('data/testing/'):
    if test:
      image = Image.open('data/testing/' + fn)

      # Convert the image to grayscale
      grayscale_image = image.convert("L")
      grayscale_image=np.asarray(grayscale_image)
      img=tf.image.resize(image, target_size)
    else:
      path = 'data/testing/' + fn
      img = tf.keras.utils.load_img(path, target_size=target_size)
      img = tf.keras.utils.img_to_array(img)

    img=img*x
    img = np.expand_dims(img, axis=0)

    prediction_class = model.predict(img, batch_size=10)
    if len(get_classes()) > 2:
      y_pred_classes = np.argmax(prediction_class, axis=1)
    else:
      if st.session_state.use_data_augmentation and st.session_state.flow_from_directory['class_mode'] == "binary" or (
              not (st.session_state.use_data_augmentation) and st.session_state.image_dataset_from_directory[
        'label_mode'] == "binary"):
        y_pred_classes = (prediction_class> 0.5).astype(int)
      else:
        y_pred_classes = np.argmax(prediction_class, axis=1)
    classes = get_classes()
    classe_predected=classes[y_pred_classes[0]]
    shutil.copyfile('data/testing/' + fn,os.path.join(testing_dir,os.path.join(classe_predected,fn)))
  zip_folder(testing_dir,'data/testing_result.zip')
  with open('data/testing_result.zip', 'rb') as fp:
    btn = st.download_button(
      label="Download testing result ZIP",
      data=fp,
      file_name="testing_reesult.zip",
      mime="application/zip"
    )

st.header("Chose metrics")
metrics_values={}
for metric in st.session_state.metrics:
  metrics_values[metric]=st.checkbox(metric)
if st.button("confirm"):
  for metric,value in metrics_values.items():
    if st.session_state.model==None:
      st.warning("You don't have model to evaluate", icon="⚠️")
      break
    if value:
      if metric=="Accuracy":
        st.pyplot(plot_graphs(st.session_state.history,"accuracy"))
      elif metric=="F1Score":
        if st.session_state.F1Score_mode>1:
          st.pyplot(plot_graphs(st.session_state.history, "f1_score_categrecal"))
        else:
          st.pyplot(plot_graphs(st.session_state.history, "f1_score_binary"))
      else:
        st.pyplot(plot_graphs(st.session_state.history, metric))
  if st.session_state.model == None:
    st.warning("You don't have model to evaluate", icon="⚠️")
  else:
    st.pyplot(plot_graphs(st.session_state.history, "loss"))
if st.button("get confusion metrixs "):
  if st.session_state.model == None:
    st.warning("You don't have model to evaluate", icon="⚠️")
  else:
    get_confusion_metrics()
data = st.file_uploader("Upload test images ", type=["ZIP", "TAR","RAR","ARJ","TGZ"])

testing_root = "data/testing/"


# Preprocess the image
if data is not None:
  if not os.path.exists(testing_root):
    os.mkdir(testing_root)
  else:
    shutil.rmtree('data/testing/')
    os.makedirs("data/testing/")

  with zipfile.ZipFile(data, "r") as z:
      z.extractall(testing_root)
  if st.session_state.model == None:
    st.warning("You don't have model to evaluate", icon="⚠️")
  else:
    test()


def download_model():
    if type(st.session_state.ModelCheckpoint) == dict:
      file_path = st.session_state.ModelCheckpoint["filepath"]
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
if os.path.exists("data/data/training/model.h5"):  # and st.session_state.model is not None:
  download_model()