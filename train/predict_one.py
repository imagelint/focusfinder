from fastai.vision.all import *
#from webfunctions import get_focus_point
#import re

def get_focus_point(path_name):
    dfb = next(iter(df[df['name']==path_name.name].index), ('no match for '+path_name.name))
    return tensor([df['x_p'][dfb], df['y_p'][dfb]])

model = load_learner('../train/models/focusfinder_4000_5e-3.pkl')

test_img_path = '../download/images/norm_images/00a5c9469db810ee.jpg'

res = model.predict(test_img_path)
print(res)