
import argparse
from cv2 import eigen, imshow, waitKey
import numpy as np
import hw2help
import cv2 # Added import
import skimage
# TODO: Import any necessary libraries.

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--image", default="corner2.gif", type=str, help="Image to load.")
parser.add_argument("--task", default="susan", type=str, help="Selected task: corners/susan.")

def responseCorners(toOpen : str, window, response, threshold, sigma):
    # Preprocess the image

    if toOpen.endswith(".gif"):
        # Gifs themselves are unreadable straight up, so we capture their first frame
        gifWorkaround = cv2.VideoCapture(toOpen)
        ret, inputImage = gifWorkaround.read()
        gifWorkaround.release()
        if not ret:
            raise ValueError("Couldn't open the gif")
    else:
        inputImage = cv2.imread(toOpen)
    outputImage = inputImage.copy()    
    # By default, an image isn't Grayscale but BGR after being read
    inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)  
    
    
    # Could potentially be user-defined
    WINDOW_SIZE = window
    RESPONSE_CONSTANT = response
    THRESHOLD = threshold
    GAUSSIAN_FILTER_SIGMA = sigma

    # Getting both directions
    windowOffset = int(WINDOW_SIZE/2)
    # Ensure our window doesn't go off-bounds
    xSize = inputImage.shape[0] - windowOffset
    ySize = inputImage.shape[1] - windowOffset
    
    # Numpy helps out our gradient search
    yDerivative, xDerivative = np.gradient(inputImage)
    # Gradient is just an array of derivatives, so we can turn it into our matrix values
    
    # Smoothing here is significantly faster, but smoothing later on performs better
    Ixx = skimage.filters.gaussian(np.float_power(xDerivative,2), GAUSSIAN_FILTER_SIGMA, truncate=2)
    Iyy = skimage.filters.gaussian(np.float_power(yDerivative,2), GAUSSIAN_FILTER_SIGMA, truncate=2)
    Ixy = skimage.filters.gaussian(xDerivative * yDerivative, GAUSSIAN_FILTER_SIGMA, truncate=2)
    #Ixx = np.float_power(xDerivative,2)
    #Iyy = np.float_power(yDerivative,2)
    #Ixy = xDerivative * yDerivative
    # Keep track of found corners for different response functions
    HarrisCorners = []
    ShiCorners = []
    TriggCorners = []
    BSWinderCorners = []

    for x in range(windowOffset, xSize):
        for y in range(windowOffset, ySize):
            xLowBorder = x - windowOffset
            yLowBorder = y - windowOffset
            # Odd numbers need compensation after int conversion in offset
            if windowOffset % 2 == 1:
                xUpperBorder = x + windowOffset + 1
                yUpperBorder = y + windowOffset + 1
            else:
                # Rectangular windows don't have this issue
                xUpperBorder = x + windowOffset
                yUpperBorder = y + windowOffset

            # Store the corresponding window matrix values
            #currentIxx = skimage.filters.gaussian(
            #    Ixx[yLowBorder : yUpperBorder, xLowBorder : xUpperBorder],GAUSSIAN_FILTER_SIGMA,truncate=2)
            #currentIyy = skimage.filters.gaussian(
            #    Iyy[yLowBorder : yUpperBorder, xLowBorder : xUpperBorder],GAUSSIAN_FILTER_SIGMA,truncate=2)
            #currentIxy = skimage.filters.gaussian(
            #    Ixy[yLowBorder : yUpperBorder, xLowBorder : xUpperBorder],GAUSSIAN_FILTER_SIGMA,truncate=2)

            currentIxx = Ixx[yLowBorder : yUpperBorder, xLowBorder : xUpperBorder]
            currentIyy = Iyy[yLowBorder : yUpperBorder, xLowBorder : xUpperBorder]
            currentIxy = Ixy[yLowBorder : yUpperBorder, xLowBorder : xUpperBorder]
            # Sums to be plugged into our matrix
            # Gradient function gets us partial derivatives
            # In the next step, we precomputed their squares / multiples for according positions
            # All that is left is to sum the values corresponding to our current window and plug them into the matrix
            # These values make up our Gaussian filter
            xxSum = currentIxx.sum()
            yySum = currentIyy.sum()
            xySum = currentIxy.sum()

            gaussianFilter = np.array([[xxSum, xySum], [xySum, yySum]])
            eigenValues = eigen(gaussianFilter)[1]

            # Determinant and trace
            determinant = eigenValues[0]*eigenValues[1]
            trace = xxSum + yySum

            # Harris Corner response
            HarrisR = determinant - RESPONSE_CONSTANT*np.float_power(trace, 2)
            # Shi and Tomasi response
            ShiR = min(eigenValues[0], eigenValues[1])
            # Triggs response
            TriggR = min(eigenValues[0], eigenValues[1]) - RESPONSE_CONSTANT*max(eigenValues[0], eigenValues[1])
            # Brown, Szeliski, and Winder response
            BSWinderR = determinant/(eigenValues[0]+eigenValues[1])

            if HarrisR > THRESHOLD:
                HarrisCorners.append([x,y,HarrisR])
            if ShiR > THRESHOLD:
                ShiCorners.append([x,y,ShiR])
            if TriggR > THRESHOLD:
                TriggCorners.append([x,y,TriggR])
            if BSWinderR > THRESHOLD:
                BSWinderCorners.append([x,y,BSWinderR])
    dotColor = (0,0,255) # Red in BGR
    harrisImage = outputImage.copy()
    for i in HarrisCorners:
        harrisImage = cv2.circle(harrisImage, [i[0], i[1]], 1, dotColor, 1)
    cv2.imshow("Harris Corners",harrisImage)

    shiImage = outputImage.copy()
    for i in ShiCorners:
        shiImage = cv2.circle(shiImage, [i[0], i[1]], 1, dotColor, 1)
    cv2.imshow("Shi Corners",shiImage)

    triggImage = outputImage.copy()
    for i in TriggCorners:
        triggImage = cv2.circle(triggImage, [i[0], i[1]], 1, dotColor, 1)
    cv2.imshow("Trigg Corners",triggImage)

    bswImage = outputImage.copy()
    for i in BSWinderCorners:
        bswImage = cv2.circle(bswImage, [i[0], i[1]], 1, dotColor, 1)
    cv2.imshow("BSW Corners",bswImage)
    cv2.waitKey(0)

def susan(toOpen : str):
    if toOpen.endswith(".gif"):
        # Gifs themselves are unreadable straight up, so we capture their first frame
        gifWorkaround = cv2.VideoCapture(toOpen)
        ret, inputImage = gifWorkaround.read()
        gifWorkaround.release()
        if not ret:
            raise ValueError("Couldn't open the gif")
    else:
        inputImage = cv2.imread(toOpen)
    outputImage = inputImage.copy()  
    inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY) 
    corners = np.zeros([len(outputImage), len(outputImage[0])])
    MASK_RADIUS = 2
    SUSAN_MASK = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ]
    maximum = 0
    uniformAreas = []
    for i in range(MASK_RADIUS, len(inputImage) - MASK_RADIUS):
        for j in range(MASK_RADIUS, len(inputImage[i]) - MASK_RADIUS):
            # Use the mask filter on relevant elements
            susanMask = np.multiply(inputImage[i-MASK_RADIUS:i+MASK_RADIUS+1, j-MASK_RADIUS:j+MASK_RADIUS+1],SUSAN_MASK)
            usanValue = 0
            if np.array_equal(susanMask, np.zeros([5,5])):
                uniformAreas.append([i,j])
            # Compute USAN
            for x in susanMask:
                for y in x:
                    if y > 0:
                        usanValue += np.exp(-np.power((y-inputImage[i][j])/27,6))
            corners[i][j] = usanValue
            if usanValue > maximum:
                maximum = usanValue
    geometricThreshold = (maximum/2)+2

    for i in range(MASK_RADIUS, len(inputImage) - MASK_RADIUS):
        for j in range(MASK_RADIUS, len(inputImage[i]) - MASK_RADIUS):
            if(corners[i][j] >=geometricThreshold):
                corners[i][j] = 0
            else:
                corners[i][j] = geometricThreshold - corners[i][j]
            
    for i in uniformAreas:
        corners[i[0],i[1]] = 0
    dotColor = (0,0,255) # Red in BGR
    coordinates = skimage.feature.peak_local_max(corners, min_distance=5)
    for i in coordinates:
        outputImage = cv2.circle(outputImage, [i[1], i[0]], 1, dotColor, 1)
    #for i in range(len(corners)):
    #    for j in range(len(corners[i])):
    #        if(corners[i][j]>0):
    #            outputImage = cv2.circle(outputImage, [j, i], 1, dotColor, 1)
    cv2.imshow("Detected Corners",outputImage)
    cv2.waitKey(0)

def main(args : argparse.Namespace):
    # TODO (Task 1 'corners'): Implement corner detector with response functions from the lecture.
    # It is at the end of slides for the 7th lecture under 'Harris corner detector'.
    #
    # NOTE: The matrix A is 2x2, therefore, both its determinant and trace are easily
    # computable. With those, you are able to compute eigenvalues of the matrix as well.
    #
    # Find a good sigma for gradient smoothing - start with sigma=3 and use
    # 'skimage.filters.gaussian(img, sigma, truncate=2)' to filter the image with gaussian filter.
    if args.task == "corners":
        # Small window sizes tend to detect corners very well
        # Larger sigmas blur the image a lot, making the detector detect corners between edges a lot
        # Sigma 0 for window of size 3 performs the best
        # These parameters seem to perform better then most
        # A strange phoenomenon appears with increasing window sizes - The Harris corner detector seems inverted,
        # while the other response functions perform significantly better then usual
        responseCorners(args.image,3,0.06,1000.00,1)

    # TODO (Task 2 'susan'): Implement the SUSAN detector from the lecture.
    # Compare the effectivness of corner detectors on rotated images. What is the percentage
    # of detected corners in the image after rotation ('skimage.transform.rotate()').
    # Is it dependent on the angle of the rotation?
    # 
    # Test your algorithm on images 'corner2.gif', 'test_corner.gif' and custom 'hw2help.checkerboard()'
    if args.task == "susan":
        susan(args.image)

    # TODO (Task 3): Prove the equality with gaussian derivations attached to the homework.
    # - Write the proof on a piece of paper (and submit its picture) or in LaTeX.
    # - Please, don't write the proof in this source file.
    #
    # NOTE: This task does not require any programming - it is a pen&paper exercise.
    pass


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
