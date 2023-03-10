
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import skimage.util
# TODO: You may import any library that you deem necessary.
# - If it is not one of 'scipy', 'skimage', 'cv2', 'sklearn', 'matplotlib', 'PIL', 'numpy' then
#   it has to be reasoned and explained.
# - Obviously, libraries implementing the solution of this homework are forbidden.

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--n_r", default=4, type=int, help="Number of red values for a palette.")
parser.add_argument("--n_g", default=4, type=int, help="Number of green values for a palette.")
parser.add_argument("--n_b", default=3, type=int, help="Number of blue values for a palette.")
parser.add_argument("--grid_size", default=4, type=int, help="Number of blocks on one side of a grid.")

class ImageDatastore:
    """
    Simple class for storing a list of images.
    It contains two lists 'fileList' with Path objects for each image and 'images' with np.ndarray objects,
    which are the images themselves.
    - Images are loaded only on demand through either 'load' function or index based access on an object of this class.
    """

    def __init__(self, pathToImages : str, reDesc : str = "*.jpg", includeSubfolders : bool = True):
        self.fileList = list(Path(pathToImages).rglob(reDesc) if includeSubfolders else Path(pathToImages).glob(reDesc))
        self.images = [None] * len(self.fileList)

    def __getitem__(self, imgIdx : int) -> np.ndarray:
        return self.load(imgIdx)

    def __len__(self) -> int:
        return len(self.images)

    def load(self, imgIdx : int) -> np.ndarray:
        if self.images[imgIdx] is None:
            with Image.open(self.fileList[imgIdx]) as imgHandle:
                self.images[imgIdx] = np.asarray(imgHandle, np.uint8)
        return self.images[imgIdx]

def rgb2idx(img : np.ndarray, palette : np.ndarray) -> np.ndarray:
    """Computes indices to a palette for an image according to the least variance."""
    diffImg = img[:, :, :, None] - palette.T[None, None, :, :]
    idxImg = np.argmin(np.sum(diffImg ** 2, axis=2), axis=2)
    return idxImg

def main(args : argparse.Namespace):
    # TODO: Preparation of an image database.
    # - Download an image database (111MB)
    #   http://imagedatabase.cs.washington.edu/groundtruth/icpr2004.imgset.rar
    # - We will work with the directory /icpr2004.imgset/groundtruth

    # NOTE: Palette creation.
    # - Do not make the palette much larger otherwise the algorithm will be very slow.
    deltaR = 1 / args.n_r
    vecR = np.linspace(0, 1 - deltaR, args.n_r) + deltaR / 2
    deltaG = 1 / args.n_g
    vecG = np.linspace(0, 1 - deltaG, args.n_g) + deltaG / 2
    deltaB = 1 / args.n_b
    vecB = np.linspace(0, 1 - deltaB, args.n_b) + deltaB / 2
    rr, gg, bb = np.meshgrid(vecR, vecG, vecB)
    palette = np.vstack([rr.ravel(), gg.ravel(), bb.ravel()]).T
    paletteSize = palette.shape[0]

    # NOTE: Load the images from the groundtruth directory into memory.
    # - The following class behaves as a read-only list but you can access its inner variables
    #   if you want/need to - beware that images are not loaded on initialisation.
    imgs = ImageDatastore("icpr2004.imgset/groundtruth", "*.jpg", True)
    
    # NOTE: Preprocessing
    # Constant size of the grid.
    gridSize = args.grid_size

    # TODO: We need to preprocess the images and compute their features.
    # - We will be storing the histogram of an image and the mean colour in KxK regions into
    #   the variable 'features'.
    #
    # NOTE: Conversion to indices 'rgb2idx' is quite slow and I am not sure if it is possible
    # to do a least variance conversion faster in numpy.
    # Therefore, it might be useful to compute the features once, save them in a file using
    # 'np.save', then skip the feature generation and load your stored features with 'np.load'
    features = np.zeros((len(imgs), paletteSize + gridSize * gridSize * 3))

    for imgIdx in range(len(imgs)):
        imgFloat = skimage.util.img_as_float(imgs[imgIdx])
        
        # Convert 'imgFloat' to an image of indices to the palette 'palette'. (This is quite slow)
        idxImg = rgb2idx(imgFloat, palette)

        # TODO: Create a normalised histogram of colours (its sum should be 1)
        colorHist = None

        # Store the histogram
        features[imgIdx, : paletteSize] = colorHist

        # TODO: Split the image into 'gridSize' * 'gridSize' regions and compute the mean
        # colour of each region (from the original image 'imgFloat', not from the palette).
        # - Store the computed mean colour into the feature vector.
        #   features[imgIdx, paletteSize + ??] = ??
        pass

    # TODO: Task
    # Choose an index of a query image Q
    # Identify the 3 most common colours in the query image (from the palette).
    # - In parts (1) and (2), we will work only with these 3 colours - imagine that we take only the 3 respective
    #   histogram columns and ignore the rest.
    #
    # Find 9 of the most similar images to the query image based on:
    # 1. The sum of euclidean distances of the histograms for the three selected colours.
    # 2. The intersection of the histograms of the three selected colours.
    #    - Sum of intersections for each colour.
    # 3. gridded_color distance
    #    - Comparison between colours computed on a grid (from features) - it was explained in the lecture.

    # To get only K smallest/largest values in an array, you may use 'np.argpartition' or simply
    # 'np.argsort'.
    
    # TODO: Draw the query image together with the 9 most similar images in a single figure.
    # - Compare the results of the three implemented metrics.

    q = 67
    queryImg = skimage.util.img_as_float(imgs[q])
    queryFeatures = features[q, :]
    

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
