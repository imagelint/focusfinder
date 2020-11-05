# Focusfinder
Focusfinder is a tool which finds a focus-point for any given image. this focus-point can then, for example, be used as a center point for cropping.

a neural network is trained with [fastai](https://www.fast.ai/) to find the focus-point.

## download
this subdirectory deals with creating, sorting and reorganizing data, that is then used to train the neural network.

it has the following structure:

- images/
    - raw_images/
    - norm_images/
- labels/
- node_modules/
    - node-fetch/
- nocaps.js
- unsplash.js
- transform_images.ipynb

*raw_images/* contains all raw training images, *norm_images/* contains the same images but normalized, so that they all have the same size. *labels/* is where the *.csv are saved, which basically are a table with a row for each image_name and its focus-point. 

*nocaps.js* and *unsplash.js* are scripts to download data. *tranform_images.ipynb* can then be used to resized the images and to calculate the focus-points in pixel for the new images.

## train

- models/
    - my_model.pkl
    - ...
- train_model.ipynb
- train_model.py

in *models/* the models that were trained with the *train_model* script are saved. the training script itself is available as a notebook, so with description, or a python script. both contain the same code. 