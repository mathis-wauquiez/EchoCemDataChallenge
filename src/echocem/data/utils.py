import numpy as np
import pandas as pd

def get_raw_annotations_and_image(y_table, images_path, filename):
    """
    corrigée
    """

    base = filename.split(".")[0] 
    ind = y_table.index.get_loc(base)
    
    ann_1d = y_table.iloc[ind].values
    
    
    if len(ann_1d) == 43520 :
        if -1 in ann_1d:
            ann_1d=ann_1d[ann_1d!=-1]
            ann_2d_pre = ann_1d.reshape(160, -1)
            ann_2d=np.zeros((160,272))
            ann_2d = np.zeros((160, 272), dtype=ann_2d_pre.dtype)
            ann_2d[:, :160]=ann_2d_pre
        else :
	        ann_2d=ann_1d.reshape(160,-1)
    else:
        raise ValueError(f"Unexpected annotation size {len(ann_1d)} for {filename}")


    img = np.load(images_path / filename)


    if img.shape == (160, 272):
	    pass  
    elif img.shape == (160, 160):
        padded = np.zeros((160, 272), dtype=img.dtype)
        padded[:, :160] = img
        img = padded
    else:
        raise ValueError(f"Unexpected image shape {img.shape} for {filename}")

    return ann_2d, img

def iterate_on_raw_annotations_and_images(y_table, images_path):
    for filename in y_table.index:
        yield get_raw_annotations_and_image(filename, y_table, images_path)
