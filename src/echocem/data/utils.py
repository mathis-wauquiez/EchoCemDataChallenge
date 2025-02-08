import numpy as np
import pandas as pd

def get_raw_annotations_and_image(y_table, images_path, filename):
    ind = y_table.index.get_loc(filename.split(".")[0])
    data = y_table.iloc[ind].values.reshape(160, -1)
    img = np.load(images_path / filename)
    return data, img

def iterate_on_raw_annotations_and_images(y_table, images_path):
    for filename in y_table.index:
        yield get_raw_annotations_and_image(filename, y_table, images_path)