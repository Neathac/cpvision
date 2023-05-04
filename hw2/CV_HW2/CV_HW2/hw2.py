
import argparse
from typing import Dict
import numpy as np
import skimage.filters
import scipy.ndimage

parser = argparse.ArgumentParser()
# These arguments will not be used during evaluation but you can use them for experimentation
# if it's easier for you to debug your algorithms in this file.
parser.add_argument("--image", default="data/corner2.gif", type=str, help="Image to load.")
parser.add_argument("--task", default="harris", type=str, help="Selected task.")

class HarrisMethods:
    """
    This class contains constant values of the names for Harris corner detection methods.
    Use these constants in your solution to avoid typing errors. Evaluator
    expects these constants in the returned dictionary.
    """
    HARRIS = "harris"
    SHI_TOMASI = "shi_tomasi"
    TRIGGS = "triggs"
    BROWN = "brown"

class SusanMethods:
    """
    This class contains constant values of the names for Susan corner detection methods.
    Use these constants in your solution to avoid typing errors. Evaluator expects these constants
    in the returned dicitionary.
    """
    HARD = "hard"
    SOFT = "soft"

def harrisCorners(image : np.ndarray, sigma : float, alpha : float) -> Dict[str, np.ndarray]:
    # TODO (Task 1 'Harris corner detection'): Implement corner detection algorithm based on pixel
    # response computed from a matrix of derivatives - Harris corner detector. You are supposed to
    # implement all four response functions described in the lecture and return them in a dictionary
    # as showcased at the end of this function.
    
    WINDOW_SIZE = 2
    RESPONSE_CONSTANT = alpha
    # THRESHOLD = threshold
    GAUSSIAN_FILTER_SIGMA = sigma

    # Getting both directions
    windowOffset = int(WINDOW_SIZE/2)
    # Ensure our window doesn't go off-bounds
    xSize = image.shape[0] - windowOffset
    ySize = image.shape[1] - windowOffset

    # The requirements for your implementation are the following:
    # - All 4 response functions described in the lecture have to be implemented.
    # - Use 'scipy.ndimage.sobel' to compute image derivatives.
    # - Use 'skimage.filters.gaussian' to smooth derivatives as described in the lecture with the given 'sigma'.
    # - Use 'alpha' as the harris trace empirical constant.
    # - Compute eigenvalues for every pixel manually - it is simple because matrix A is 2x2.
    
    xDerivative = scipy.ndimage.sobel(image, 0)  # horizontal derivative
    yDerivative = scipy.ndimage.sobel(image, 1)  # vertical derivative

    Ixx = skimage.filters.gaussian(np.float_power(xDerivative,2), GAUSSIAN_FILTER_SIGMA, truncate=2)
    Iyy = skimage.filters.gaussian(np.float_power(yDerivative,2), GAUSSIAN_FILTER_SIGMA, truncate=2)
    Ixy = skimage.filters.gaussian(xDerivative * yDerivative, GAUSSIAN_FILTER_SIGMA, truncate=2)

    HarrisCorners = []
    ShiCorners = []
    TriggCorners = []
    BSWinderCorners = []
    
    responseHarris = np.zeros(image.shape)
    responseShiTomasi = np.zeros(image.shape)
    responseTriggs = np.zeros(image.shape)
    responseBrown = np.zeros(image.shape)

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
            # In the next step, we precomputed their squares / multiples for according positions
            # All that is left is to sum the values corresponding to our current window and plug them into the matrix
            # These values make up our Gaussian filter
            xxSum = currentIxx.sum()
            yySum = currentIyy.sum()
            xySum = currentIxy.sum()

            # gaussianFilter = np.array([[xxSum, xySum], [xySum, yySum]])
            mean = ((xxSum+yySum)/2)
            determinant = (xxSum*yySum) - (xySum*xySum)
            eigenValues = [mean + np.sqrt(np.power(mean, 2)-determinant), mean - np.sqrt(np.power(mean, 2)-determinant)]

            # Determinant and trace
            # determinant = eigenValues[0]*eigenValues[1]
            trace = xxSum + yySum

            # Harris Corner response
            HarrisR = determinant - RESPONSE_CONSTANT*np.float_power(trace, 2)
            responseHarris[y, x] = HarrisR
            # Shi and Tomasi response
            ShiR = min(eigenValues[0], eigenValues[1])
            responseShiTomasi[y, x] = ShiR
            # Triggs response
            TriggR = min(eigenValues[0], eigenValues[1]) - RESPONSE_CONSTANT*max(eigenValues[0], eigenValues[1])
            responseTriggs[y, x] = TriggR
            # Brown, Szeliski, and Winder response
            if eigenValues[0]+eigenValues[1] != 0:
                BSWinderR = ((eigenValues[0]*eigenValues[1])/(eigenValues[0]+eigenValues[1]))
            else:
                BSWinderR = ((eigenValues[0]*eigenValues[1])/1)
            responseBrown[y, x] = BSWinderR
    # It is forbidden:
    # - to use any functions for eigenvalue decomposition - both skimage and opencv contain functions, which
    #   return eigenvalues for each pixel, similarly, numpy has methods for general eigen decomposition.
    #
    # Additionally, investigate the impact of image rotation on the performance of Harris corner detection:
    # - Is the algorithm invariant to rotation?
    # - Are there any problems/exceptions?
    # - Explain your findings.
    # - You can write the explanation here in a commented section.
    
    '''
    Unsurprisingly, rotations that maintain relative spatial relations and size of the image - meaning 0, 90, 180... - degree rotations are invariant. Seeing as we
    base our corner detection off of derivatives in x and y directions, the intensity of these derivatives remains unchanged, although it may change polarity.
    Surprisingly - discounting corners created by padding for rotated images - Harris detection remains remarkably consistent for skewed images - angles of 45, 135... -
    degree rotations. There are slight improvements in places even, as it handles very sligthly curved surfaces - such as sides of a triangle - marginally better. 
    This is no doubt precisely because of the way derivatives work. Rotation may turn these "curved" surfaces into flat ones in the eyes of x and y, but may in turn create
    other "curved" surfaces out of previously flat ones, making this a double-edged sword. It doesn't appear to be significant improvement or worsening in either case, however.
    '''

    # You can evaluate Harris corner detection with:
    # >>> python evaluator.py --test=harris --image=path/to/image
    #
    # The evaluator has the following harris related arguments:
    # --image ... path to an image or 'disk', 'rectangle', 'stars', 'checkerboard' for debug images.
    # --reference ... path to a reference solution such as "ref/ref_harris_rectangle.npz"
    #   - References were generated with the default argument values.
    # --harris_sigma ... 'sigma'
    # --harris_alpha ... 'alpha'
    # --min_distance ... minimal distance between corners in non-maximal suppression.
    # --threshold_rel ... relative threshold for corner non-maximal suppression.
    # --angle ... Angle of rotation.

    responses = {
        HarrisMethods.HARRIS : responseHarris,
        HarrisMethods.SHI_TOMASI : responseShiTomasi,
        HarrisMethods.TRIGGS : responseTriggs,
        HarrisMethods.BROWN : responseBrown,
    }

    return responses

def create_circular_mask(radius):
    center = (int(radius/2), int(radius/2))
    Y, X = np.ogrid[:radius, :radius]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= int(radius/2)
    return mask

def susanCorners(image : np.ndarray, radius : int, threshold : float, gMultiplier : float) -> Dict[str, np.ndarray]:
    # TODO (Task 2 'SUSAN corner detection'): Implement the SUSAN edge and corner detector, which was mentioned
    # in the lecture. Since it wasn't thoroughly describedo n the lecture, you can use additional sources:
    # - Rundown of the paper: https://users.fmrib.ox.ac.uk/~steve/susan/susan/node6.html
    # - SUSAN paper: https://link.springer.com/article/10.1023/A:1007963824710
    # - Other sources.
    susanHard = np.zeros(image.shape)
    susanSoft = np.zeros(image.shape)
    maximum = 0
    mask = create_circular_mask(radius=(radius*2)+1)
    # tempMask = np.ones((radius, radius))
    # mask = tempMask > 0
    
    mask[radius, radius] = False
    maximum = int(np.sum(mask))
    g = gMultiplier*maximum
    
    for x in range(radius+1, image.shape[0] - (radius+1)):
        for y in range(radius, image.shape[1] - radius):
            imagePiece = image[x-radius:x+radius+1, y-radius:y+radius+1]
            croppedPixels = imagePiece[mask]
            centerPoint = image[x, y]
            hardBrightnessSum = 0
            softBrightnessSum = 0

            for i in croppedPixels:
                softBrightnessSum += np.exp(-np.power((i - centerPoint)/threshold, 6))
                hardDifference = np.abs(i - centerPoint)
                if hardDifference <= threshold:
                    hardBrightnessSum += 1
            if g > hardBrightnessSum:
                susanHard[x, y] = g - hardBrightnessSum
            if g > softBrightnessSum:
                susanSoft[x, y] = g - softBrightnessSum
    
    # The requirements for your implementation are the following:
    # - Variable radius of the USAN neighbourhood given by 'radius'.
    # - Variable threshold 't' given by 'threshold'.
    # - Variable geometric constant 'g' computed as 'gMultiplier'*'maxUsanValue' - 'gMultiplier' is usually 0.75 for edges adn 0.5 for corners.
    # - Both hard and soft USAN value computation - two different 'c' functions in the referenced paper:
    #   - hard - Number of pixels that fall within threshold.
    #   - soft - Sum of pixels weighted by an exponential function.
    # - You are supposed to return the SUSAN response/map for both hard and soft USAN value computation.
    #
    # Additionally, investigate the impact of image rotation on the performance of SUSAN corner detection:
    # - Is the algorithm invariant to rotation?
    # - Are there any problems/exceptions?
    # - Explain your findings.
    # - You can write the explanation here in a commented section.

    '''
    That isn't a straightforward question. The SUSAN algorithm appears to be invariant to rotation IF and only if the image is rotated in such a way where 
    its relative spatial relationships are perserved and the image dimensions are as well. This means that no change in detected corners occurs for images rotated by
    0, 90, or 180 degrees. There are, however, significantly more detected corners in images - not all of which are correct - if the image is rotated by 
    45, 135, or other such angles. This probably has most to do with what pixels fall within the mask, seeing as composing a circular mask out of square pixels may
    cause extra pixels to fall within the mask, or otherwise not, thus changing the proportion of bright and non-bright pixels that get considered for each mask, even
    if the geometric area for a perfect circle remains unchanged.
    '''
    
    # You can evaluate SUSAN corner detection with:
    # >>> python evaluator.py --test=susan --image=path/to/image
    #
    # The evaluator has the following susan related arguments:
    # --image ... path to an image or 'disk', 'rectangle', 'stars', 'checkerboard' for debug images.
    # --reference ... path to a reference solution such as "ref/ref_susan_rectangle.npz"
    #   - References were generated with the default argument values.
    # --susan_radius ... 'radius'
    # --susan_t ... 'threshold'
    # --susan_g ... 'gMultiplier'
    # --min_distance ... minimal distance between corners in non-maximal suppression.
    # --threshold_rel ... relative threshold for corner non-maximal suppression.
    # --angle ... Angle of rotation.

    

    susan = {
        SusanMethods.HARD : susanHard,
        SusanMethods.SOFT : susanSoft,
    }

    return susan

def main(args : argparse.Namespace):
    # TODO (Task 3 'Gaussian derivative equality'): Prove that sigma*nabla^2G == dGdsigma.
    # The task is properly described in the assignment PDF.
    # - Do not write the solution here - write it on a paper or in a formula-friendly writing software
    #   and submit the photo/PDF with your solution.
    raise NotImplementedError()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
