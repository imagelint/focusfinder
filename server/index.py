from flask import Flask, request, jsonify
from urllib import parse
from fastai.vision.all import *
import os

model_name = '20201129125255-10epochs-0.0005248074419796466trainrate_model.pkl'

def get_model_path(model_name):
    file_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = "/".join(file_path.split("/")[0:-1])
    models_path = dir_path+'/models/'
    return models_path + model_name

def get_focus_point(path_name):
        dfb = next(iter(df[df['name']==path_name.name].index), ('no match for '+path_name.name))
        return tensor([df['x_p'][dfb], df['y_p'][dfb]])

def predict_focus_point(image_path):
    # check if file is an img
    # TODO: Check if the path is a reasonable folder
    if not Path(image_path).is_file():
        print("ERROR")
        return None
    try:
        prediction = model.predict(image_path)
        return prediction[0][0]
    except:
        print("ERROR")
        return None

model = load_learner(get_model_path(model_name))
app = Flask(__name__)

@app.route('/')
def index():
    image_path = request.args.get('file', '')
    image_path = parse.unquote(image_path)
    focus_point = predict_focus_point(image_path)
    if focus_point is None:
        return jsonify(error=str('Image not found')), 404
    return jsonify([float(focus_point[0]), float(focus_point[1])])

# TODO: Only expose webserver on localhost
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8081)