from flask import Flask, request, render_template, jsonify

from flask import Flask, request, render_template

import os
import numpy as np
import tensorflow as tf
import cv2
import tqdm
import pandas as pd

from keras_cv.models import EfficientNetV2Backbone
from tensorflow.keras.models import load_model


import keras_cv
import keras

import h5py



# Define custom objects
custom_objects = {
    "EfficientNetV2Backbone": EfficientNetV2Backbone
}

# Load the model
model = load_model('./model/my_model.h5', custom_objects=custom_objects)
print("Model loaded successfully!")


app = Flask(__name__)

# Define constants
UPLOAD_FOLDER = './static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
TARGET_IMAGE_SIZE = (128, 128)  # Adjust based on your model's input size

#config
class CFG:
    verbose = 1  # Verbosity
    seed = 42  # Random seed
    neg_sample = 0.01 # Downsample negative calss
    pos_sample = 5.0  # Upsample positive class
    preset = "efficientnetv2_b2_imagenet"  # Name of pretrained classifier
    image_size = [128, 128]  # Input image size
    epochs = 8 # Training epochs
    batch_size = 256  # Batch size
    lr_mode = "cos" # LR scheduler mode from one of "cos", "step", "exp"
    class_names = ['target']
    num_classes = 1

keras.utils.set_random_seed(CFG.seed)


#importing necessary files
training_df = pd.read_csv("./static/output.csv")
training_validation_hdf5 = h5py.File("./static/filtered_data.hdf5", 'r')

print("working1")

# Categorical features which will be one hot encoded
CATEGORICAL_COLUMNS = ["sex", "anatom_site_general",
            "tbp_tile_type","tbp_lv_location", ]

# Numeraical features which will be normalized
NUMERIC_COLUMNS = ["age_approx", "tbp_lv_nevi_confidence", "clin_size_long_diam_mm",
           "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio", "tbp_lv_color_std_mean",
           "tbp_lv_deltaLBnorm", "tbp_lv_minorAxisMM", ]

# Tabular feature columns
FEAT_COLS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

#building necessary foncs
def build_augmenter():
    # Define augmentations
    aug_layers = [
        keras_cv.layers.RandomCutout(height_factor=(0.02, 0.06), width_factor=(0.02, 0.06)),
        keras_cv.layers.RandomFlip(mode="horizontal"),
    ]
    
    # Apply augmentations to random samples
    aug_layers = [keras_cv.layers.RandomApply(x, rate=0.5) for x in aug_layers]
    
    # Build augmentation layer
    augmenter = keras_cv.layers.Augmenter(aug_layers)

    # Apply augmentations
    def augment(inp, label):
        images = inp["images"]
        aug_data = {"images": images}
        aug_data = augmenter(aug_data)
        inp["images"] = aug_data["images"]
        return inp, label
    return augment


def build_decoder(with_labels=True, target_size=CFG.image_size):
    def decode_image(inp):
        # Read jpeg image
        file_bytes = inp["images"]
        image = tf.io.decode_jpeg(file_bytes)
        
        # Resize
        image = tf.image.resize(image, size=target_size, method="area")
        
        # Rescale image
        image = tf.cast(image, tf.float32)
        image /= 255.0
        
        # Reshape
        image = tf.reshape(image, [*target_size, 3])
        
        inp["images"] = image
        return inp

    def decode_label(label, num_classes):
        label = tf.cast(label, tf.float32)
        label = tf.reshape(label, [num_classes])
        return label

    def decode_with_labels(inp, label=None):
        inp = decode_image(inp)
        label = decode_label(label, CFG.num_classes)
        return (inp, label)

    return decode_with_labels if with_labels else decode_image


def build_dataset(
    isic_ids,
    hdf5,
    features,
    labels=None,
    batch_size=32,
    decode_fn=None,
    augment_fn=None,
    augment=False,
    shuffle=1024,
    cache=True,
    drop_remainder=False,
):
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter()

    AUTO = tf.data.experimental.AUTOTUNE

    images = [None]*len(isic_ids)
    for i, isic_id in enumerate(tqdm.tqdm(isic_ids, desc="Loading Images ")):
        images[i] = hdf5[isic_id][()]
        
    inp = {"images": images, "features": features}
    slices = (inp, labels) if labels is not None else inp

    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.cache() if cache else ds
    ds = ds.map(decode_fn, num_parallel_calls=AUTO)
    if shuffle:
        ds = ds.shuffle(shuffle, seed=CFG.seed)
        opt = tf.data.Options()
        opt.deterministic = False
        ds = ds.with_options(opt)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(augment_fn, num_parallel_calls=AUTO) if augment else ds
    ds = ds.prefetch(AUTO)
    return ds

print("working2")
## Train
print("# Training:")
training_features = dict(training_df[FEAT_COLS])
training_ids = training_df.isic_id.values
training_labels = training_df.target.values
training_ds = build_dataset(training_ids, training_validation_hdf5, training_features, 
                         training_labels, batch_size=CFG.batch_size,
                         shuffle=True, augment=True)

print("working3")
#The almighty space_feature
feature_space = keras.utils.FeatureSpace(
    features={
        # Categorical features encoded as integers
        "sex": "string_categorical",
        "anatom_site_general": "string_categorical",
        "tbp_tile_type": "string_categorical",
        "tbp_lv_location": "string_categorical",
        # Numerical features to discretize
        "age_approx": "float_discretized",
        # Numerical features to normalize
        "tbp_lv_nevi_confidence": "float_normalized",
        "clin_size_long_diam_mm": "float_normalized",
        "tbp_lv_areaMM2": "float_normalized",
        "tbp_lv_area_perim_ratio": "float_normalized",
        "tbp_lv_color_std_mean": "float_normalized",
        "tbp_lv_deltaLBnorm": "float_normalized",
        "tbp_lv_minorAxisMM": "float_normalized",
    },
    output_mode="concat",
)


#adapting
training_ds_with_no_labels = training_ds.map(lambda x, _: x["features"])
feature_space.adapt(training_ds_with_no_labels)
print("working4")

# Image preprocessing logic
def load_and_preprocess_image(image_path, target_size):
    """
    Preprocess the image for the model.
    :param image_path: Path to the image
    :param target_size: Target size for resizing
    :return: Preprocessed image as NumPy array
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Handle image upload
            image_file = request.files["image"]
            if not image_file:
                raise ValueError("No image uploaded.")
            image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_path)
            preprocessed_image = load_and_preprocess_image(image_path, TARGET_IMAGE_SIZE)

            # Handle feature inputs
            input_features = {
                'age_approx': float(request.form["age_approx"]),
                'clin_size_long_diam_mm': float(request.form["clin_size_long_diam_mm"]),
                'tbp_lv_areaMM2': float(request.form["tbp_lv_areaMM2"]),
                'tbp_lv_area_perim_ratio': float(request.form["tbp_lv_area_perim_ratio"]),
                'tbp_lv_color_std_mean': float(request.form["tbp_lv_color_std_mean"]),
                'tbp_lv_deltaLBnorm': float(request.form["tbp_lv_deltaLBnorm"]),
                'tbp_lv_minorAxisMM': float(request.form["tbp_lv_minorAxisMM"]),
                'tbp_lv_nevi_confidence': float(request.form["tbp_lv_nevi_confidence"]),
                'sex': request.form["sex"],
                'anatom_site_general': request.form["anatom_site_general"],
                'tbp_tile_type': request.form["tbp_tile_type"],
                'tbp_lv_location': request.form["tbp_lv_location"]
            }
            
            # Convert the dictionary values to tf.Tensors
            features_tensor = {key: tf.convert_to_tensor(value) for key, value in input_features.items()}

            # Pass the features as a dictionary to the FeatureSpace
            preprocessed_x = feature_space(features_tensor)

            # Extract the values (it should contain the 71 transformed feature values)
            preprocessed_values = preprocessed_x.numpy()

            
            preprocessed_features = np.array(preprocessed_values)
            preprocessed_features = preprocessed_features.reshape(1, -1)
            print("working0000")

            # Make prediction
            prediction = model.predict({"images": preprocessed_image, "features": preprocessed_features})
            result = "Signs of Cancer" if prediction > 0.5 else "No Signs of Cancer"
            probability = prediction[0][0]

            return render_template("index.html", result=result, probability=f"{probability:.2f}", image_path=image_file.filename)

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")
if __name__ == '__main__' :
    app.run(host="0.0.0.0", port=5000 , debug=True)