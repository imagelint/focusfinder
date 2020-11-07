from fastai.vision.all import *
import re
import time
import os

#########################################################################
#                         set parameters                                #
#########################################################################

# set paths for images and label.csv
file_path = os.path.dirname(os.path.realpath(__file__))
dir_path = "/".join(file_path.split("/")[0:-1])
path = Path(dir_path)
labels_path = path/'download/labels/train_labels.csv'
images_path = path/'download/images/norm_images'

batch_size = 10

epochs = 1

# good learn rate can be found with fastai functin learn.lr_find()
train_rate = 4e-5 # lr_min=0.012022644281387329, lr_steep=3.630780702224001e-05

model_metrics = mse

#########################################################################
#                           main code                                   #
#########################################################################

print("Starting Focusfinder train script")

def get_focus_point(path_name):
    dfb = next(iter(df[df['name']==path_name.name].index), ('no match for '+path_name.name))
    return tensor([float(df['x_p'][dfb]), float(df['y_p'][dfb])])

# load csv
df = pd.read_csv(labels_path)
# define fastai datablock
imgs = DataBlock(
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_focus_point,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    batch_tfms=[Normalize.from_stats(*imagenet_stats)],
    item_tfms=Resize(244),
)

# create dataloader
# Adjust the batch size to fit your hardware. A higher batch size needs more (gpu) ram.
dls = imgs.dataloaders(images_path, bs=batch_size)
print("Data loaded successfully")

# load pretrained model
learn = cnn_learner(dls, resnet18, y_range=(-1,1), metrics=model_metrics)

# train model with epochs and learn_rate 
print("Starting training with train_rate {} for {} epochs\n".format(train_rate, epochs))
learn.fine_tune(epochs, train_rate)

# save model with unique name
time_now = time.localtime()
model_name = str(time_now.tm_year) + str(time_now.tm_mon) + str(time_now.tm_mday) + str(time_now.tm_hour) + str(time_now.tm_min) + str(time_now.tm_sec) + "-" + str(epochs) + "epochs-" + str(train_rate) + "trainrate_model.pkl"
learn.export(('./models/'+model_name))

print('Training done and model {} saved'.format(model_name))