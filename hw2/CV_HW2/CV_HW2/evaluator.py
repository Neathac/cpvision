
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.util
import skimage.morphology
import skimage.feature
import skimage.transform
import hw2

parser = argparse.ArgumentParser()
parser.add_argument("--test", default="harris", type=str, help="Evaluated task: either 'harris' for Harris corner detection or 'susan' for SUSAN corner detection.")
parser.add_argument("--image", default="data/corner2.gif", type=str, help="Input image for either task or one of 'disk', 'rectangle', 'stars', 'checkerboard' for debug images.")
parser.add_argument("--reference", default=None, type=str, help="Reference result showed alongside the solution.")
parser.add_argument("--angle", default=0, type=float, help="Angle of rotation for the image.")
parser.add_argument("--min_distance", default=10, type=float, help="Minimla distance for non-maximal suppression.")
parser.add_argument("--threshold_rel", default=0.2, type=float, help="Relative threshold for non-maximal suppression.")
parser.add_argument("--harris_sigma", default=3, type=float, help="Gaussian standard deviation for harris corner detection.")
parser.add_argument("--harris_alpha", default=0.05, type=float, help="Harris trace empirical constant.")
parser.add_argument("--susan_radius", default=3, type=int, help="Radius of the disc used in SUSAN algorithm.")
parser.add_argument("--susan_t", default=0.1, type=float, help="Threshold for intensity similarity in SUSAN algorithm.")
parser.add_argument("--susan_g", default=0.5, type=float, help="Geometric threshold multiplier in SUSAN algorithm.")

def checkerboard(size : int = 32, rows : int = 8, columns : int = 8, grayHalf : bool = True) -> np.ndarray:
    """
    Creates a checkerboard with 'rows' rows, 'columns' columns, where each square has the size 'size'
    Optionaly makes the right half of the checkerboard half as dark as the left one.

    Arguments:
    - 'size' - Size of a single tile in the checkerboard. (Default: 32)
    - 'rows' - Number of rows in the checkerboard. (Default: 8)
    - 'columns' - Number of columns in the checkerboard. (Default: 8)
    - 'grayHalf' - Whether the right half should be half as dark as the left one. (Default: True)

    Returns:
    - Single channel image (np.ndarray) with checkerboard pattern.
    """
    board = np.ones((rows * size, columns * size))
    rr, cc = np.meshgrid(np.arange(0, board.shape[0]), np.arange(0, board.shape[1]), indexing='ij')
    indices = (rr // size) + (cc // size)
    board[indices % 2 == 0] = 0
    if grayHalf:
        board[np.logical_and(indices % 2 == 0, cc // size >= columns / 2)] = 0.5
    return board

def createDisk(radius : int, size : int) -> np.ndarray:
    """
    Creates a white disk, which doesn't have any corners.

    Arguments:
    - 'radius' - Radius of the disk.
    - 'size' - Size of the image - should be at least 2*radius+1.

    Returns:
    - np.ndarray with an image of a white disk.
    """
    img = np.zeros((size, size))
    mid = size // 2
    img[mid - radius : mid + radius + 1, mid - radius : mid + radius + 1] = skimage.morphology.disk(radius, dtype=float)
    return img

def createRectangle(a : int, b : int, size : int) -> np.ndarray:
    """
    Creates a white rectangle, which has four obvious corners.

    Arguments:
    - 'a' - The vertical size of the rectangle.
    - 'b' - The horizontal size of the rectangle.
    - 'size' - The size of the resulting image.

    Returns:
    - np.ndarray with an image of a white rectangle.
    """
    img = np.zeros((size, size))
    mid = size // 2
    img[mid - a // 2 : mid + a // 2, mid - b //2 : mid + b // 2] = 1.0
    return img

def createStars(radius : int, padding : int, num : int) -> np.ndarray:
    """
    Creates an image with a series of stars with different intensities.

    Arguments:
    - 'radius' - Radius of a single star.
    - 'padding' - Padding of a single star.
    - 'num' - Number of stars in one row/column.

    Returns:
    - np.ndarray with num*num stars of increasing intensity.
    """
    star = skimage.morphology.star(radius, dtype=float)
    star = np.pad(star, padding)
    side = star.shape[0]
    img = np.zeros((star.shape[0] * num, star.shape[1] * num))
    for i in range(num * num):
        r, c = i // num, i % num
        img[r * side : r * side + side, c * side : c * side + side] = star * ((i + 1) / (num * num))
    return img

def loadImage(name : str) -> np.ndarray:
    """
    Loads an image or creates a debug image based on the provided name.

    Arguments:
    - 'name' - Name of a debug image or a path to some real image.

    Returns:
    - np.ndarray with the loaded or created image.
    """
    names = {
        "disk" : lambda : createDisk(70, 200),
        "rectangle" : lambda : createRectangle(50, 100, 150),
        "stars" : lambda : createStars(40, 10, 4),
        "checkerboard" : lambda : checkerboard(32, 8, 8, True),
    }
    if name not in names.keys():
        img = skimage.util.img_as_float(skimage.io.imread(name, as_gray=True))
    else:
        img = names[name]()
    return img

def rotateImage(image : np.ndarray, angle : float) -> np.ndarray:
    """
    Rotates the given image by an angle given in degrees.

    Arguments:
    - 'image' - Rotated image.
    - 'angle' - Angle of rotation in degrees.

    Returns:
    - np.ndarray with the rotated image.
    """
    return skimage.transform.rotate(image, angle, resize=True) if angle != 0 else image

def evaluateHarrisCorners(args : argparse.Namespace) -> None:
    """
    Evaluation function for Harris corner detection.
    It executes the implementation in 'hw2.py' and displays the result including non-maximal suppression
    for the final corner detection.
    If 'args.reference' is not None, then it loads the reference file and displays it alongside
    the solution.
    """
    print("Evaluation of Harris corner detection for the image: {}.".format(args.image))
    image = rotateImage(loadImage(args.image), args.angle)
    startHarris = time.time()
    harrisResult = hw2.harrisCorners(image, sigma=args.harris_sigma, alpha=args.harris_alpha)
    endHarris = time.time()
    print(">>> The algorithm finished in {:>7.4f} seconds.".format(endHarris - startHarris))

    if args.reference is not None:
        isReference = True
        reference = np.load(args.reference)
    else:
        isReference = False

    fig, ax = plt.subplots(3 if isReference else 2, 4, figsize=(14, 8 if isReference else 6))
    for i, method in enumerate([hw2.HarrisMethods.HARRIS, hw2.HarrisMethods.SHI_TOMASI, hw2.HarrisMethods.TRIGGS, hw2.HarrisMethods.BROWN]):
        peaks = skimage.feature.corner_peaks(harrisResult[method], min_distance=args.min_distance, threshold_rel=args.threshold_rel)
        ax[0, i].set_title("GREEN - solution, RED - reference")
        ax[0, i].imshow(image, cmap="gray")
        ax[0, i].scatter(peaks[:, 1], peaks[:, 0], marker="x", c="green", s=80, linewidth=2)
        ax[0, i].set_axis_off()
        ax[1, i].set_title("{} response - solution".format(method))
        ax[1, i].imshow(harrisResult[method], cmap="gray")
        ax[1, i].set_axis_off()
        if isReference:
            refPeaks = skimage.feature.corner_peaks(reference[method], min_distance=args.min_distance, threshold_rel=args.threshold_rel)
            ax[0, i].scatter(refPeaks[:, 1], refPeaks[:, 0], marker="o", facecolors="none", edgecolors="red", s=120, linewidth=2)
            ax[2, i].set_title("{} response - reference".format(method))
            ax[2, i].imshow(reference[method], cmap="gray")
            ax[2, i].set_axis_off()
    fig.tight_layout()
    plt.show()

def evaluateSusanCorners(args : argparse.Namespace) -> None:
    """
    Evaluation function for SUSAN corner detection.
    It executes the implementation in 'hw2.py' and displays the result including non-maximal suppression
    for the final corner detection.
    If 'args.reference' is not None, then it loads the reference file and displays it alongside
    the solution.
    """
    print("Evaluation of SUSAN corner detection for the image {}.".format(args.image))
    image = rotateImage(loadImage(args.image), args.angle)
    startSusan = time.time()
    susanResult = hw2.susanCorners(image, radius=args.susan_radius, threshold=args.susan_t, gMultiplier=args.susan_g)
    endSusan = time.time()
    print(">>> The algorithm finished in {:>7.4f} seconds.".format(endSusan - startSusan))

    if args.reference is not None:
        isReference = True
        reference = np.load(args.reference)
    else:
        isReference = False

    fig, ax = plt.subplots(2, 3 if isReference else 2, figsize=(12 if isReference else 10, 8))
    for i, method in enumerate([hw2.SusanMethods.HARD, hw2.SusanMethods.SOFT]):
        peaks = skimage.feature.corner_peaks(susanResult[method], min_distance=args.min_distance, threshold_rel=args.threshold_rel)
        ax[i, 0].set_title("GREEN - solution, RED - reference")
        ax[i, 0].imshow(image, cmap="gray")
        ax[i, 0].scatter(peaks[:, 1], peaks[:, 0], marker="x", c="green", s=80, linewidth=2)
        ax[i, 0].set_axis_off()
        ax[i, 1].set_title("{} susan - solution".format(method))
        ax[i, 1].imshow(susanResult[method], cmap="gray")
        ax[i, 1].set_axis_off()
        if isReference:
            refPeaks = skimage.feature.corner_peaks(reference[method], min_distance=args.min_distance, threshold_rel=args.threshold_rel)
            ax[i, 0].scatter(refPeaks[:, 1], refPeaks[:, 0], marker="o", facecolors="none", edgecolors="red", s=120, linewidth=2)
            ax[i, 2].set_title("{} susan - reference".format(method))
            ax[i, 2].imshow(reference[method], cmap="gray")
            ax[i, 2].set_axis_off()
    fig.tight_layout()
    plt.show()

def main(args : argparse.Namespace):
    # Select the evaluated test.
    tests = {
        "harris" : evaluateHarrisCorners,
        "susan" : evaluateSusanCorners,
    }
    if args.test not in tests.keys():
        raise ValueError("Unknown type of test '{}', please, choose one of 'harris' or 'susan'!".format(args.test))
    tests[args.test](args)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
