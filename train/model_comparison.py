from fastai.vision.all import *
import re
import time
import os

def compare_models():
    # set those two parameters
    epochs = 30
    batch_size = 32

    # resnet 18
    # lr_min=0.33113112449646, lr_steep=0.0691830962896347
    # resnet 34
    # lr_min=0.33113112449646, lr_steep=1.0964781722577754e-06
    # resnet 50
    # lr_min=0.33113112449646, lr_steep=6.309573450380412e-07

    model_names = {
        'resnet18':resnet18, 
        'resnet34':resnet34,
        'resnet50':resnet50,
        }

    learn_rates = {
        'resnet18':[0.3,7e-2],
        'resnet34':[0.3,1e-6],
        'resnet50':[0.3,6e-7],
    }
    
    results_df = pd.DataFrame(columns=['name','learn_rate','res_1','res_2'])

    for _name, _model in model_names.items():
        for _learn_rate in learn_rates[_name]:
            print('Model {} with learn_rate {}'.format(_name,_learn_rate))
            res = train_model(model_arch=_model, learn_rate=_learn_rate, epochs=epochs,bs=batch_size)

            results_df = results_df.append({
                'name': _name,
                'learn_rate': _learn_rate, 
                'res_1': res[0],
                'res_2': res[1],
            },ignore_index=True)

    print(results_df)
    file_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = "/".join(file_path.split("/")[0:-1])
    path = Path(dir_path)
    result_csv_path = path/'models/model_comparison_result.csv'
    results_df.to_csv(result_csv_path)
    
def get_dataframe():
    file_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = "/".join(file_path.split("/")[0:-1])
    path = Path(dir_path)
    labels_path = path/'download/labels/train_labels.csv'
    df = pd.read_csv(labels_path)
    return df

df = get_dataframe()

def get_focus_point(path_name):
    dfb = next(iter(df[df['name']==path_name.name].index), ('no match for '+path_name.name))
    return tensor([df['x_p'][dfb], df['y_p'][dfb]])

def train_model(model_arch=resnet18, learn_rate=4e-5, epochs=10, bs=8):

    # set paths for images and label.csv
    file_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = "/".join(file_path.split("/")[0:-1])
    path = Path(dir_path)
    images_path = path/'download/images/norm_images'
    final_model_path = path/'models/'

    model_metrics = mse

    #print("Starting Focusfinder train script")

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
    #print("Data loaded successfully")

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
    learn.export((final_model_path/model_name))

    print('Training done and model {} saved'.format(model_name))

    # should return final mse
    return res

if __name__=='__main__':
    compare_models()