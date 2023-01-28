import tensorflow as tf
from tensorflow.io import read_file
from tensorflow.image import decode_image, resize
from tensorflow import expand_dims

from flask import Flask, render_template, request

loaded_model = tf.keras.models.load_model("./model/tea_leaves_diseases")

# create function for load and preprocess
def load_and_preprocess(filename, img_shape=224):
    img = read_file(filename)
    img = decode_image(img, channels=3)
    img = resize(img, size=[img_shape, img_shape])
    img = img / 255.0 
    img = expand_dims(img, axis=0)
    inference = loaded_model.predict(img)
    classx = inference.argmax()
    class_names = [
        "algal leaf",
        "anthracnose",
        "bird eye spot",
        "brown blight",
        "gray blight",
        "healthy leaf",
        "red leaf spot",
        "white spot",
    ]

    return class_names[classx]


app = Flask(__name__, template_folder="templates",static_folder='static')


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        file.save("file.png")
        prediction = load_and_preprocess("file.png")
        return render_template("result.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
