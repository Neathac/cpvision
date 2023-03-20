
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import cv2
import hw1

parser = argparse.ArgumentParser()
parser.add_argument("--test", default="bayer", type=str, help="Evaluated task: either 'bayer' for bayer demosaicing or 'mediancut' for median cut algorithm.")
parser.add_argument("--image", default="Lighthouse_bggr.png", type=str, help="Input image for either task.")
parser.add_argument("--tol", default=2.5, type=float, help="RMSE tolerance.")
parser.add_argument("--num_colors", default=16, type=int, help="Number of colours requested in the mediac cut algorithm.")

def evaluateBayerDemosaicing(args : argparse.Namespace) -> None:
    # NOTE: This method runs the solution of the bayer demosaicing assignment, computes
    # the opencv implementation of the same algorithm and compares them for bayer masks.
    # Due to opencv performance optimisations, the RMSE of a correct solution might reach
    # up to 2.5 on the supplied images so there is an error tolerance.
    print("Evaluation of bayer demosaicing for the image: {}.".format(args.image))
    image = skimage.io.imread(args.image)
    startDemosaic = time.time()
    demosaicResult = hw1.bayerDemosaic(image)
    endDemosaic = time.time()
    print(">>> The algorithm finished in {:>7.4f} seconds.".format(endDemosaic - startDemosaic))
    names = ["bggr", "gbrg", "rggb", "grbg"]
    trueDemosaics = []
    for name, code in zip(names, [cv2.COLOR_BAYER_BG2BGR, cv2.COLOR_BAYER_GB2BGR, cv2.COLOR_BAYER_RG2BGR, cv2.COLOR_BAYER_GR2BGR]):
        trueDemosaic = cv2.demosaicing(image, code)
        trueDemosaics.append(trueDemosaic)
        diff = np.asarray(trueDemosaic, float) - np.asarray(demosaicResult[name], float)
        print(np.mean(diff[1 : -1, 1 : -1, 1] ** 2))
        err, nonEdgeErr = np.sqrt(np.mean(diff ** 2)), np.sqrt(np.mean(diff[1 : -1, 1 : -1] ** 2))
        if err < args.tol:
            print(">>> SUCCESS in {} test! (your total RMSE: {:>6.2f} and non-edge RMSE: {:>6.2f})".format(name.upper(), err, nonEdgeErr))
        else:
            print(">>> FAILED {} test with RMSE: {:>6.2f} (non-edge: {:>6.2f}), it should be less than {:>6.2f}".format(name.upper(), err, nonEdgeErr, args.tol))
    
    fig, ax = plt.subplots(3, 4, figsize=(10, 8))
    for i, (name, trueDemosaic) in enumerate(zip(names, trueDemosaics)):
        solution = demosaicResult[name]
        ax[0, i].set_title("Implementation: {}".format(name.upper()))
        ax[0, i].imshow(solution.astype(np.uint8))
        ax[1, i].set_title("OPENCV: {}".format(name.upper()))
        ax[1, i].imshow(trueDemosaic.astype(np.uint8))
        ax[2, i].set_title("Per-pixel RMSE: {}".format(name.upper()))
        ax[2, i].imshow(np.sqrt(np.mean((solution.astype(float) - trueDemosaic.astype(float)) ** 2, axis=2)).astype(np.uint8))
    fig.tight_layout()
    plt.show()

def evaluateMedianCut(args : argparse.Namespace) -> None:
    # NOTE: This method runs the solution of the median cut assignment, converts the resulting
    # palette into unsigned integers and shows the results for comparison.
    print("Evaluation of median cut for the image {}.".format(args.image))
    image = skimage.io.imread(args.image)
    startMedianCut = time.time()
    paletteResult, idxImg = hw1.medianCut(image, args.num_colors)
    paletteResult = np.asarray(paletteResult, np.uint8)
    endMedianCut = time.time()
    print(">>> The algorithm finished in {:>7.4f} seconds.".format(endMedianCut - startMedianCut))
    print("Palette colours:")
    print(np.asarray(paletteResult))

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].set_title("Original image")
    ax[0].imshow(image)
    ax[1].set_title("Index image for the palette")
    ax[1].imshow(idxImg)
    ax[2].set_title("Palette reconstruction from {} colours".format(paletteResult.shape[0]))
    ax[2].imshow(paletteResult[idxImg])
    fig.tight_layout()
    plt.show()

def main(args : argparse.Namespace):
    # Select the evaluated task.
    tests = {
        "bayer" : evaluateBayerDemosaicing,
        "mediancut" : evaluateMedianCut,
    }
    if args.test not in tests.keys():
        raise ValueError("Unknown type of test '{}', please choose one of 'bayer' or 'mediancut'!".format(args.test))
    tests[args.test](args)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
