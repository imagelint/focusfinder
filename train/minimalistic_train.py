from fastai.vision.all import *
import re

def get_focus_point(path_name):
    dfb = next(iter(df[df['name']==path_name.name].index), ('no match for '+path_name.name))
    return tensor([float(df['x_p'][dfb]), float(df['y_p'][dfb])])

df = pd.read_csv(Path('download/labels/train_labels.csv'), names=['name','x_p','y_p'], header=0)
imgs = DataBlock(blocks=(ImageBlock, PointBlock), get_items=get_image_files, get_y=get_focus_point, splitter=RandomSplitter(valid_pct=0.2, seed=42), batch_tfms=[*aug_transforms(size=(244, 244)), Normalize.from_stats(*imagenet_stats)], item_tfms=Resize(244),)
dls = imgs.dataloaders(Path('download/images/norm_images'), bs=16)

cnn_learner(dls, resnet18, y_range=(-1,1)).fine_tune(3, 4e-5).export(('./models/m.pkl'))