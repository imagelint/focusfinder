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

## Quickstart

### get data
#### download images
to download images use  
```node nocaps.js```  
and/or  
```node unsplash.js```

#### create train focus points (csv)
for the training some focus-points have to be set by hand. these information should in the end be saved in a csv with the following format:

| name | x_p | y_p |
|---|---|---|
| EKf428HQ130.jpg | 118.03252032520325 | 110.09756097560977 |
| ... | ... | ... |

where x_p and y_p are the positions of the focus-point in pixel.

#### normalize images and csv
to train a neural network all images have to have the same size. furthermore the focus-point is not necessarly given in pixel corrisponding to a resized img.

to resize all images for which a focus-point is given the *transform_images.py* script can be used. it not only resizes the images but also creates a new *.csv with all focus-points relativ to the new image-size and in pixel. the input csv shoud have the focus-points in relative values (from 0-1).
- download/
    - labels/    
        - unslash_labels.csv
        - nocaps_labels.csv
    - images/
        - raw_images/
            - nocaps/
                - ...
            - unsplash/
                - ...

with this csv structure:
| name | x_p | y_p |
|---|---|---|
| fee73b1e0ea41b91.jpg | 0.485 | 0.22875 |
| ... | ... | ... |


execute:   
``` python3 transform_images.py ```  
after using this script the following data should exist:

- download/
    - labels/    
        - train_labels.csv
    - images/
        - norm_images/
            - fee73b1e0ea41b91.jpg
            - ...

with this csv structure:
| name | x_p | y_p |
|---|---|---|
| fee73b1e0ea41b91.jpg | 496.64 | 234.24 |
| ... | ... | ... |

### train model
there are two options to train the model  
1. in a jupyter notebook  
this has the advantage that data can easly by visualized and each step can be seen on its own.  
in can be found at train/train_model.ipynb
2. python script  
the advantage of a simple python script that it only has to be executed once after setting the parameters accordingly. it can be found at train/train_model.py. to execute is use:  
```python3 train_model.py```  

both options save a model in /train/models/*.pkl. 

### use trained model
a simple webserver to use the trained model can be found under webserver/webserver.py