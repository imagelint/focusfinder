from fastai.vision.all import *
import re

def label_func(path_name):
    x = re.search(r'0\w+.jpg', str(path_name))
    return x.group()

def get_focus_point(x):
    file_name = label_func(x)
    dfb = next(iter(df[df['name']==file_name].index), ('no match '+file_name))
    return [float(df['x_p'][dfb]), float(df['y_p'][dfb])]

path = Path('../')
labels_path = path/'download/labels/100_norm_imgs.csv'
# images_path = path/'download/images'

images_path = path/'download/norm_images'

df = pd.read_csv(labels_path, names=['name','x_p','y_p'], header=None)

imgs = DataBlock(
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_focus_point,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    batch_tfms=aug_transforms(),
    item_tfms=Resize(256),
)

dls = imgs.dataloaders(images_path)

learn = cnn_learner(dls, resnet18, y_range=(-1,1))
learn.fine_tune(3, 8e-3)

learn.export('./models/80_imgs_test_.pkl')

print('training done')