from fastai.vision.all import *
import re
import time
import os

print("Starting Focusfinder train script")

# define functions that return a focus point for a given file_name
def label_func(path_name):
    x = re.search(r'\d\w+.jpg', str(path_name))
    return x.group()

def get_focus_point(x):
    file_name = label_func(x)
    dfb = next(iter(df[df['name']==file_name].index), ('no match '+file_name))
    return tensor([float(df['x_p'][dfb]), float(df['y_p'][dfb])])

# set paths for images and label.csv
file_path = os.path.dirname(os.path.realpath(__file__))
dir_path = "/".join(file_path.split("/")[0:-1])
path = Path(dir_path)
labels_path = path/'download/labels/norm_labels_4000.csv'
images_path = path/'download/images/norm_images'

# load csv
df = pd.read_csv(labels_path, names=['name','x_p','y_p'], header=0)

# define fastai datablock
imgs = DataBlock(
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_focus_point,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    batch_tfms=[*aug_transforms(size=(256, 256)), 
                Normalize.from_stats(*imagenet_stats)],
    item_tfms=Resize(256),
)

# create dataloader
# Adjust the batch size to fit your hardware. A higher batch size needs more (gpu) ram.
batch_size = 8
dls = imgs.dataloaders(images_path, bs=batch_size)
print("Data loaded successfully")

# load pretrained model
learn = cnn_learner(dls, resnet18, y_range=(-1,1))

# train model with epochs and learn_rate (good learn rate can be found with fastai functin learn.lr_find())
epochs = 30
train_rate = 2e-4

print("Starting training with train_rate {} for {} epochs\n".format(train_rate, epochs))
learn.fine_tune(epochs, train_rate)

# save model with unique name
time_now = time.localtime()
model_name = str(time_now.tm_year) + str(time_now.tm_mon) + str(time_now.tm_mday) + str(time_now.tm_hour) + str(time_now.tm_min) + str(time_now.tm_sec) + "-" + str(epochs) + "epochs-" + str(train_rate) + "trainrate_model.pkl"
learn.export(('./models/'+model_name))

print('training done')