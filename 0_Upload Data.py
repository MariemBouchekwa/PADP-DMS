# st.title("Upload Data")
# st.sidebar.info("Upload Data")

import os
import zipfile
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import shutil
import random
import imghdr

if "validate" not in st.session_state:
    st.session_state.validate = {
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
    }


def count_corapted_images(source):
    images_source = os.listdir(source)
    s = 0
    c = 0
    for image in images_source:
        if os.path.getsize(os.path.join(source, image)) == 0:
            s = s + 1
            os.remove(os.path.join(source, image))
        elif imghdr.what(os.path.join(source, image)) not in ['jpeg', 'jpg', 'png', 'bmp', 'gif']:
            c = c + 1
            os.remove(os.path.join(source, image))
    return s, c


# import tensorflow as tf
# from tensorflow import keras

st.set_page_config(page_title="Upload data", page_icon=":guardsman:")


# , layout="wide"
# Define a function to create the GUI

def create_gui():
    root = "data/images/images/"
    if not (os.path.isdir(root)):
        st.session_state.validate["upload_data"] = False
        st.title("Upload Data")

        # Choose the CNN architecture
        # 'labeled Folder', 'labeled images (with the class names )', 'all images in one folder with csv file'
        architectures = ['labeled Folder']

        choise_data_form = st.radio("chose the form of your data", architectures)

        # Show information about the selected architecture
        if choise_data_form == 'labeled Folder':
            chose = 0
        elif choise_data_form == 'labeled images (with the class names )':
            chose = 1
        elif choise_data_form == 'all images in one folder with csv file':
            chose = 2
        st.info("Your zip should contain a folder of each class")

        # Allow the user to upload an image
        data = st.file_uploader("Upload dataset", type=["ZIP", "TAR", "RAR", "ARJ", "TGZ"])

        root = "data/images/images/"
        # Preprocess the image
        if data is not None:
            with zipfile.ZipFile(data, "r") as z:
                z.extractall(root)
            st.experimental_rerun()


def Data():
    root = "data/images/images/"
    root2 = "data/images/images"
    if os.path.isdir(root):
        # st.write(os.listdir(root))
        if len(os.listdir(root)) == 1:
            st.error("You probably provided a zip with a parent folder inside")
            # st.title("About your Data:")
            # number_of_classes=len(os.listdir(os.path.join(root,os.listdir(root)[0])))
            # show_class_number=st.write(f"Number of classes : {number_of_classes}")
            # for i in range(number_of_classes):
            #     name = "Class " + str(i+1) + ": " + os.listdir(os.path.join(root,os.listdir(root)[0]))[i]
            #     with st.expander(name, expanded=True):
            #         data = os.listdir(os.path.join(root,os.listdir(root)[0]))[i]
            #         images = os.listdir(os.path.join(root, os.listdir(root)[0], data))
            #         # st.write(os.getcwd())
            #         c1,c2,c3 = st.columns(3)

            #         p = os.path.join(root, os.listdir(root)[0], data, images[0])
            #         image = Image.open(p)
            #         c1.image(image)
            #         p = os.path.join(root, os.listdir(root)[0], data, images[1])
            #         image = Image.open(p)
            #         c2.image(image)
        else:
            l = os.listdir("data/images/images")
            if len(l) == 0:
                st.warning("You have to Uploid data properly ", icon="⚠️")
            else:
                st.session_state.validate["upload_data"] = True
                st.title("About your Data:")
                number_of_classes = len(os.listdir(os.path.join(root)))
                st.write(f"Number of classes : {number_of_classes}")
                for i in range(number_of_classes):
                    name = "Class " + str(i + 1) + ": " + os.listdir(root)[i]
                    with st.expander(name, expanded=True):
                        data = os.listdir(root)[i]
                        # st.write(data)
                        images = os.listdir(os.path.join(root, data))
                        count_corpt_img, count_not_image_file = count_corapted_images(os.path.join(root, data))
                        st.write(f"There are {count_corpt_img} corupted images that get deleted")
                        st.write(
                            f"There are {count_not_image_file} not supported file that get deleted (file need to be ['jpeg', 'jpg', 'png', 'bmp', 'gif'])")
                        st.write(f"Number of Images: ", len(images))

                        c1, c2, c3, c4 = st.columns(4)
                        c = [c1, c2, c3, c4]
                        for j in range(len(c)):
                            x = random.randint(0, len(images) - 1)
                            p = os.path.join(root, data, images[x])
                            image = Image.open(p)
                            c[j].write(np.array(image).shape)
                            c[j].image(image)

        if st.button("Load new images"):
            pass

        if st.button("Delete Data", key="delete"):
            if os.path.isdir("data/data"):
                st.write("Deleting...")
                shutil.rmtree("data/data")
                shutil.rmtree(root)
                st.experimental_rerun()
            else:
                st.write("Deleting dataset...")
                shutil.rmtree(root)
                st.experimental_rerun()


# Create the GUI
create_gui()
Data()