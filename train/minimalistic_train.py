from fastai.vision.all import *
import re

def get_focus_point(path_name):
    file_name = re.search(r'\d\w+.jpg', str(path_name)).group()
    dfb = next(iter(df[df['name']==file_name].index), ('no match for '+file_name))
    return tensor([float(df['x_p'][dfb]), float(df['y_p'][dfb])])

df = pd.read_csv(Path('download/labels/norm_labels_4000.csv'), names=['name','x_p','y_p'], header=0)
imgs = DataBlock(blocks=(ImageBlock, PointBlock), get_items=get_image_files, get_y=get_focus_point, splitter=RandomSplitter(valid_pct=0.2, seed=42), batch_tfms=[*aug_transforms(size=(256, 256)), Normalize.from_stats(*imagenet_stats)], item_tfms=Resize(256),)
dls = imgs.dataloaders(Path('download/norm_images'), bs=2)

learn = cnn_learner(dls, resnet18, y_range=(-1,1))
learn.fine_tune(3, 2e-5)
learn.export(('./models/m.pkl'))