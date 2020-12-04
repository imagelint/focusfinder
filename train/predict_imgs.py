import cv2
import pandas as pd
import os
import random
from fastai.vision.all import *

dir_path = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[0:-1])

# set parameters
n_imgs = 50
random_seed = 80

# lr_min for 10 epochs:
# 20201129125255-10epochs-0.0005248074419796466trainrate_model.pkl

# set paths
model_path = dir_path + '/models/20201129125255-10epochs-0.0005248074419796466trainrate_model.pkl'
test_imgs_path = dir_path+'/download/images/norm_images/'
img_names_csv = dir_path+'/download/labels/train_labels.csv'
save_imgs_path = dir_path+'/train/test/'

def get_focus_point(path_name):
    dfb = next(iter(df[df['name']==path_name.name].index), ('no match for '+path_name.name))
    return tensor([df['x_p'][dfb], df['y_p'][dfb]])

df = pd.read_csv(img_names_csv)
model = load_learner(model_path)

# get x random indexes of imgs
idxs = random.sample(range(0, len(df['name'])), n_imgs)

count = 1
for i in idxs:
    img_path = test_imgs_path + df['name'][i]
    img_path = dir_path+'/test.jpg'
    res = model.predict(img_path)
    focuspoint = (int(res[0][0][0]),int(res[0][0][1]))
    print(focuspoint)
    image = cv2.imread(img_path)
    pimage = cv2.circle(image, focuspoint, 5, (255,0,0), 2)
    cv2.imwrite((save_imgs_path + df['name'][i]), pimage)
    #print('{}/{}'.format(count, n_imgs))
    count += 1
