from flask import Flask
from fastai.vision.all import *

def get_focus_point(path_name):
        dfb = next(iter(df[df['name']==path_name.name].index), ('no match for '+path_name.name))
        return tensor([df['x_p'][dfb], df['y_p'][dfb]])

app = Flask(__name__)

@app.route('/')
def index():
    global get_focus_point    
    model = load_learner('./models/20201129125255-10epochs-0.0005248074419796466trainrate_model.pkl')

    test_img_path = './download/images/norm_images/00a5c9469db810ee.jpg'

    res = model.predict(test_img_path)
    print(res)
    return 'Hello world!'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8081)