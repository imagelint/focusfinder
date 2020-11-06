from fastai.vision.all import *
import re
import os

print("Starting Focusfinder test script")

def get_focus_point(path_name):
    dfb = next(iter(df[df['name']==path_name.name].index), ('no match for '+path_name.name))
    return tensor([float(df['x_p'][dfb]), float(df['y_p'][dfb])])

# set paths for images and label.csv
file_path = os.path.dirname(os.path.realpath(__file__))
dir_path = "/".join(file_path.split("/")[0:-1])
path = Path(dir_path)
labels_path = path/'download/labels/train_labels.csv'
images_path = path/'download/images/norm_images'
model_path = path/'train/models/2020116165217-100epochs-4e-05trainrate_model.pkl'
# load csv
df = pd.read_csv(labels_path)
# define fastai datablock
imgs = DataBlock(
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_focus_point,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    batch_tfms=[*aug_transforms(size=(244, 244)),
                Normalize.from_stats(*imagenet_stats)],
    item_tfms=Resize(244),
)

# create dataloader
# Adjust the batch size to fit your hardware. A higher batch size needs more (gpu) ram.
batch_size = 10
dls = imgs.dataloaders(images_path, bs=batch_size)
print("Data loaded successfully")

model = load_learner(model_path)
preds, y = model.get_preds(dl=dls)

distances = []
for _p, _y in zip(preds, y):
    d = np.linalg.norm(_p-_y[0])
    distances.append(d)

mean = np.mean(distances)
print('The mean is {:.2f} or {:.2f} px'.format(mean, (mean*244)))