from fastai.vision.all import *
import re
import time
import os


#########################################################################
#                         set parameters                                #
#########################################################################

batch_size = 10
model_arch = resnet18
model_metrics = mse
epochs = 1
learn_rate = 4e-5 
# good learn rate can be found with fastai functin learn.lr_find()

# set paths for images and label.csv
file_path = os.path.dirname(os.path.realpath(__file__))
dir_path = "/".join(file_path.split("/")[0:-1])
path = Path(dir_path)
labels_path = path/'download/labels/train_labels.csv'
images_path = path/'download/images/norm_images'
final_model_path = path/'models/'

#########################################################################
#                           main code                                   #
#########################################################################

print("Starting Focusfinder train script")

def get_focus_point(path_name):
    dfb = next(iter(df[df['name']==path_name.name].index), ('no match for '+path_name.name))
    return tensor([df['x_p'][dfb], df['y_p'][dfb]])

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
learn = cnn_learner(dls, model_arch, y_range=(-1,1), metrics=model_metrics)

# train model with epochs and learn_rate 
print("Starting training with learn_rate {} for {} epochs\n".format(learn_rate, epochs))
learn.fine_tune(epochs, learn_rate)

res = learn.validate()
print("result: {}".format(res))

# save model with unique name
time_now = time.localtime()
model_name = str(time_now.tm_year) + str(time_now.tm_mon) + str(time_now.tm_mday) + str(time_now.tm_hour) + str(time_now.tm_min) + str(time_now.tm_sec) + "-" + str(epochs) + "epochs-" + str(learn_rate) + "trainrate_model.pkl"
learn.export((final_model_path/model_name))

print('Training done and model {} saved'.format(model_name))
