# # import numpy as np
# # from flask import Flask, render_template, request, jsonify
# # import pickle

# # flask_app = Flask(__name__)

# # # Load the pickle file
# # model = pickle.load(open(r'D:\Hero_ML\model.pkl', "rb"))

# # @flask_app.route("/")
# # def home():
# #     return render_template("index.html")

# # @flask_app.route("/predict")
# # def predict():
# #     float_features = [float(x) for x in request.form.values()]
# #     features=np.array(float_features)
# #     prediction=model.predict(features)
# #     return render_template("index.html", "the flower species is{}".format(prediction))

# # if __name__ == "__main__":
# #     flask_app.run(debug=True)

# import numpy as np
# from flask import Flask, render_template, request
# import pickle

# flask_app = Flask(__name__)

# # Load the pickle file
# model = pickle.load(open(r'D://Hero_ML//model.pkl', "rb"))

# @flask_app.route("/")
# def home():
#     return render_template("D://Hero_ML//template//index.html")

# @flask_app.route("/predict")
# def predict():
#     float_features = [float(x) for x in request.form.values()]
#     features = np.array(float_features)
#     prediction = model.predict(features)
#     return render_template("D://Hero_ML//template//index.html", prediction_text="The flower species is {}".format(prediction))

# if __name__ == "__main__":
#     flask_app.run(debug=True)
import numpy as np
from flask import Flask, render_template, request
import pickle

flask_app = Flask(__name__, template_folder='D://Hero_ML//template')

# Load the pickle file
model = pickle.load(open(r'D://Hero_ML//model.pkl', "rb"))

@flask_app.route("/")
def home():
    return render_template("index.html")

@flask_app.route("/predict", methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = np.array(float_features).reshape(1, -1)
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="The flower species is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)
