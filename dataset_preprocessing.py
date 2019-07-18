"""
Modules contains methods for dataset preprocessing.
"""
import os
import shutil
import zipfile
import re
import numpy as np
from PIL import Image
from pathlib import Path


def unzip_data(path_from, path_to=None, pwd=b"CITRE.2019"):
    """
    Iterates over all archive files in the `path_from` directory and unzip them to `path_to` dir.
    :path_from: str or PosixPath to dir where .zip files are stored
    :path_to: str or PosixPath to dir where to extract files
    """
    # by default extract to the same folder
    if path_to is None:
        path_to = path_from
    for file_name in os.listdir(path_from):
        if re.match('([a-zA-Z0-9]+)\.zip', file_name):
            with open(Path(path_from) / file_name, 'rb') as f:
                zf = zipfile.ZipFile(f)
                zf.extractall(path_to, pwd=pwd)


def create_dataset_of_combined_images(path_from, path_to=None, classes=("AML", "HD")):
    """
    Combines images by pixelwise max operation. `path_from` contains folders which names start from class name.
    :path_from: str or PosixPath to dir with dirs with images which we are going to combine
    :path_to: str or PosixPath to dir where we would save
    """
    init_folders_lst = os.listdir(path_from)
    # option for inplace dataset modification
    if path_to is None:
        path_to = path_from
    # create folders with corresponding classes
    for cls_name in classes:
        if not os.path.exists(Path(path_to) / cls_name):
            os.mkdir(Path(path_to) / cls_name)
    
    def detect_class(folder_name, classes):
        """
        Detects on of the classes among `classes` from `folder_name`.
        """
        for cls in classes:
            if cls == folder_name[:len(cls)]:
                return cls
        return None
    
    for folder_name in init_folders_lst:
        img_prefix = folder_name.split('_')[-1]
        cls_name   = detect_class(folder_name, classes)
        im_1 = Image.open(Path(path_from) / folder_name / (img_prefix + '.jpg'))
        im_2 = Image.open(Path(path_from) / folder_name / (img_prefix + "(df).jpg"))
        im_1_np = np.array(im_1)
        im_2_np = np.array(im_2)
        im_res = Image.fromarray(np.maximum(im_1_np, im_2_np))
        im_res.save(Path(path_to) / cls_name / (img_prefix + "_comb.jpg"), format="JPEG")
    
    # remove original folders in the case of inplace modifications
    if path_from == path_to:
        for folder_name in os.listdir(path_to):
            if not folder_name in classes:
                shutil.rmtree(Path(path_to) / folder_name)


def remove_corrupted_images(path_images, path_file_corrupted_lst):
    """
    Removes from subfolders of `path_images` folder files from the file `path_file_corrupted_lst`.
    """
    path_images = Path(path_images)
    path_file_corrupted_lst = Path(path_file_corrupted_lst)
    # read bad filenames from the .txt file
    bad_lst = []
    with open(path_file_corrupted_lst, "r") as f:
        for el in f:
            bad_lst.append(el[:-1])
    # remove bad files from the folder
    for cls_folder in os.listdir(path_images):
        for file_name in os.listdir(path_images / cls_folder):
            if file_name in bad_lst:
                os.remove(path_images / cls_folder / file_name)


def split_image_to_measurement_patches(img_path, path_to, x_n=7, y_n=5, name_prefix=None, exclude_perimeter_patches=True):
    """
    Splits big `img_path` image to subimages and saves those pathes to the `path_to` folder
    with name starting from `name_prefix`.
    :img_path: str or PosixPath to image we would split
    :path_to: str or PosixPath to folder where we would save our patches
    :x_n: number of parts in which we split image in x axis
    :y_n: number of parts in which we split image in y axis
    :name_prefix: str, prefix from which name of each new path starts
    :exclude_perimeter_patches: bool flag shows if we have to exclude perimeter patches
    """
    im = Image.open(img_path)
    im_np = np.array(im)
    dx, dy = im_np.shape[0] // x_n, im_np.shape[1] // y_n
    # iterate over interior patches
    name_counter = 0
    if exclude_perimeter_patches:
        k = 1
    else:
        k = 0
    for i in range(k, x_n - k):
        for j in range(k, y_n - k):
            im_np_patch = im_np[i*dx : (i+1)*dx, j*dy : (j+1)*dy, :]
            im_patch_to_save = Image.fromarray(im_np_patch)
            if name_prefix is None:
                # extract `name_prefix` from the path; we expect file has extension from 3 symbols: `.xxx`
                name_prefix = os.path.split(img_path)[-1][:-4]
            im_patch_to_save.save(path_to / "{}_{:0>2}.jpg".format(name_prefix, name_counter))
            name_counter += 1
    

def convert_to_imagenet_form(path, percentage_train=0.7, percentage_valid=0.2, percentage_test=0.1):
    """
    Inplace convert dataset from form:
    path/
        class_1/
        class_2/
        ...
        class_n/
    to ImageNet dataset form:
    path/
        train/
            class_1/
            class_2/
            ...
            class_n/
        valid/
            class_1/
            class_2/
            ...
            class_n/
        test/
            class_1/
            class_2/
            ...
            class_n/
    Puts `percentage_train` images of `class_1` to `train/class_1` and so on and so forth.
    """
    path = Path(path)
    classes = os.listdir(path)
    group_names  = ["train", "valid", "test"]
    # create necessary new folders
    for el in group_names:
        os.mkdir(path / el)
    for group in group_names:
        for cls_name in classes:
            os.mkdir(path  / group / cls_name)
    # move files from original folders
    for cls_name in classes:
        # number of elements in the class
        num_els_in_cls = len(os.listdir(path / cls_name))
        # iteratively move files to the new subfolders
        for idx, el in enumerate(os.listdir(path / cls_name)):
            if idx < int(percentage_train * num_els_in_cls):
                shutil.move(path / cls_name / el, path / "train" / cls_name / el)
            elif idx < int((percentage_train + percentage_valid) * num_els_in_cls):
                shutil.move(path / cls_name / el, path / "valid" / cls_name / el)
            else:
                shutil.move(path / cls_name / el, path / "test" / cls_name / el)
        # remove now empty original folders
        os.rmdir(path / cls_name)


if __name__ == "__main__":
    path_raw_data = Path("/storage_2/CITRE/CITRE_raw_data/")
    path = Path("/storage_2/CITRE/CITRE_classification")
    unzip_data(path_raw_data, path)
    print("Data was unziped.")
    
    create_dataset_of_combined_images(path)
    print("Original images were combined.")

    path_bad_images = Path("/storage_2/CITRE/bad_images.txt")
    remove_corrupted_images(path, path_bad_images)
    print("Corrupted images were removed.")

    # split all images to measurable patches
    for folder_name in os.listdir(path):
        for file_name in os.listdir(path / folder_name):
            # split image inplace
            split_image_to_measurement_patches(path / folder_name / file_name, path / folder_name)
            # remove original image
            os.remove(path / folder_name / file_name)
    print("First iteration of splitting was done.")

    # split all measurable patches to the images of sizes `224 x 224`
    # magic numbers `x_n`, `y_n` calculated from the sizes of original images
    x_n = 6
    y_n = 9
    for folder_name in os.listdir(path):
        for file_name in os.listdir(path / folder_name):
            # split image inplace
            split_image_to_measurement_patches(path / folder_name / file_name, path / folder_name, x_n=x_n, y_n=y_n,
            name_prefix=None, exclude_perimeter_patches=False)
            # remove original image
            os.remove(path / folder_name / file_name)
    print("Second iteration of splitting was done.")

    # create ImageNet-like dataset
    convert_to_imagenet_form(path)
    print("Dataset was converted to ImageNet form.")