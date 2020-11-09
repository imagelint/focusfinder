from fastai.vision.all import *
import re
import time
import os

def train_model(model_arch=resnet18, learn_rate=4e-5, epochs=10, bs=8):
    #########################################################################
    #                         set parameters                                #
    #########################################################################

    # set paths for images and label.csv
    file_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = "/".join(file_path.split("/")[0:-1])
    path = Path(dir_path)
    labels_path = path/'download/labels/train_labels.csv'
    images_path = path/'download/images/norm_images'
    final_model_path = path/'models/'
    # good learn rate can be found with fastai functin learn.lr_find()
    #
    # for resnet 18 with train_labels.csv:
    # lr_min=0.012022644281387329, lr_steep=3.630780702224001e-05

    model_metrics = mse

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
    dls = imgs.dataloaders(images_path, bs=bs)
    print("Data loaded successfully")

    # load pretrained model
    learn = cnn_learner(dls, model_arch, y_range=(-1,1), metrics=model_metrics)

    # train model with epochs and learn_rate 
    print("Starting training with learn_rate {} for {} epochs\n".format(learn_rate, epochs))
    learn.fine_tune(epochs, learn_rate)

    res = learn.validate()
    print(res)

    # save model with unique name
    time_now = time.localtime()
    model_name = str(time_now.tm_year) + str(time_now.tm_mon) + str(time_now.tm_mday) + str(time_now.tm_hour) + str(time_now.tm_min) + str(time_now.tm_sec) + "-" + str(epochs) + "epochs-" + str(learn_rate) + "trainrate_model.pkl"
    learn.export((final_model_path+model_name))

    print('Training done and model {} saved'.format(model_name))

    # should return final mse
    return res
if __name__=='__main__':
    res = train_model()