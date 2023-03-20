
import argparse
from typing import Sequence, Tuple, Dict
import numpy as np
import cv2 # OpenCV
import scipy.signal
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
# These arguments will not be used during evaluation but you can use them for experimentation
# if it's easier for you to debug your algorithms in this file.
parser.add_argument("--image", default="Lighthouse_bggr.png", type=str, help="Image to load.")
parser.add_argument("--task", default="bayer", type=str, help="Selected task.")
parser.add_argument("--num_colors", default=4, type=int, help="Square root of the number of desired colors.")

def getAvg(bggr, position, directions):
    avg = 0
    for i in directions:
        avg += bggr[position[0]+i[0]][position[1]+i[1]]
    return(int(np.ceil(avg/len(directions))))

def averagePattern(image : np.ndarray, evenAvg : list[list[list[list[int]]]], oddAvg : list[list[list[list[int]]]]):
    directions = [evenAvg, oddAvg]
    newImage = np.zeros((len(image),len(image[0]),3), np.uint8)
    for i in range(1,len(image)-1):
        rowParity = directions[i%2]
        # iterate over inner columns
        for j in range(1,len(image[i])-1):
            columnParity = rowParity[j%2]
            newImage[i][j][0] = getAvg(image, [i,j], columnParity[0])
            newImage[i][j][1] = getAvg(image, [i,j], columnParity[1])
            newImage[i][j][2] = getAvg(image, [i,j], columnParity[2])

    # Fill in the edges
    for i in range(1,len(newImage[0])-1):
        newImage[0][i] = newImage[1][i]
        newImage[len(newImage)-1][i] = newImage[len(newImage)-2][i]
    for i in range(1,len(newImage)-1):
        newImage[i][0] = newImage[i][1]
        newImage[i][len(newImage[i])-1] = newImage[i][len(newImage[i])-2]
    return newImage

def bayerDemosaic(bayerImage : np.ndarray) -> Dict[str, np.ndarray]:
    # TODO (Task 1 'Bayer demosaicing') Given a Bayer encoded image (one of 'Lighthouse_bggr.png',
    # 'o1.jpg_gbrg.pgm', 'o2.jpg_grbg.pgm', 'o3.jpg_bggr.pgm', 'o4.jpg_rggb.pgm'), compute
    # the demosaiced image by simple averaging in a 3x3. Bayer pattern can have four forms:
    # BGGR, RGGB, GBRG, GRBG and your solution should compute and return all four in a dictionary.
    # Your solution is compared against OpenCV implementation and all images should
    # be within error tolerance.
    # The returned images should have values in the range [0, 255].
    #
    # Evaluation of this method can look like this:
    # >>> python .\evaluator.py --test=bayer --image=data/Lighthouse_bggr.png --tol=2.5
    #
    # Hints:
    # - To avoid for loops, you can use something like the chessboard pattern from the first practicals
    #   to select only red/green/blue pixels and process the whole iamge at once.
    #   - Also, if you define an averaging mask in 3x3 neighbourhood, you can compute the result
    #     of the averaging on the whole image with convolution 'scipy.signal.convolve'
    # - To get within tolerance, you have to solve edge pixels as well - you cannot use pixels with
    #   value 0 outside of the boundary in averaging.
    # - To extend an image by additional columns/rows, you can use 'np.pad'.
    # - Try to avoid unnecessary duplication of code.

    # Possible pixel-pattern configurations
    diagonalAvg = [[-1, -1], [1,1],[-1,1],[1,-1]]
    straightAvg = [[0, -1], [0,1],[-1,0],[1,0]]
    centeredDiagonalAvg = [[-1, -1], [1,1],[-1,1],[1,-1],[0,0]]
    singleAvg = [[0, 0]]
    horizontalAvg = [[0, -1],[0,1]]
    verticalAvg = [[-1,0],[1,0]]

    # Compose pixel configurations of 3x3 windows by row by pattern
    rggbOddAvg = [[verticalAvg, centeredDiagonalAvg, horizontalAvg],[diagonalAvg, straightAvg, singleAvg]]
    rggbEvenAvg = [[singleAvg, straightAvg, diagonalAvg],[horizontalAvg, centeredDiagonalAvg,verticalAvg ]]

    bggrOddAvg = [[horizontalAvg, centeredDiagonalAvg, verticalAvg],[singleAvg, straightAvg, diagonalAvg]]
    bggrEvenAvg = [[diagonalAvg, straightAvg, singleAvg],[verticalAvg, centeredDiagonalAvg, horizontalAvg]]

    grbgEvenAvg = [[horizontalAvg, straightAvg, verticalAvg],[singleAvg, centeredDiagonalAvg, diagonalAvg]]
    grbgOddAvg = [[diagonalAvg, centeredDiagonalAvg, singleAvg],[verticalAvg, straightAvg, horizontalAvg]]

    gbrgEvenAvg = [[verticalAvg, straightAvg, horizontalAvg],[diagonalAvg, centeredDiagonalAvg, singleAvg]]
    gbrgOddAvg = [[singleAvg, centeredDiagonalAvg, diagonalAvg],[horizontalAvg, straightAvg,verticalAvg ]]

    bggr, rggb, gbrg, grbg = averagePattern(bayerImage, bggrEvenAvg, bggrOddAvg), averagePattern(bayerImage, rggbEvenAvg, rggbOddAvg), averagePattern(bayerImage, gbrgEvenAvg, gbrgOddAvg), averagePattern(bayerImage, grbgEvenAvg, grbgOddAvg)

    result = {
        "bggr" : np.asarray(bggr, float),
        "rggb" : np.asarray(rggb, float),
        "gbrg" : np.asarray(gbrg, float),
        "grbg" : np.asarray(grbg, float),
    }

    return result

def medianCut(image : np.ndarray, numColors : int) -> Tuple[np.ndarray, np.ndarray]:
    flatArray = []
        # For easier array manipulation
        # Row
    for i in range(len(image)):
        # Columns
        for j in image[i]:
            flatArray.append(j)
    result = medianCut(np.array(flatArray), args.colReduce-1)
    mapped = verifyClusters(np.array(flatArray), np.array(result))

    
    
    palette = None
    idxImage = None
    return palette, idxImage

def preMedianCut(image : np.ndarray, numColors : int) -> list:
    # TODO (Task 2 'Median cut'): Implement a function that finds a colour palette of an image
    # according to division by median (generally known as 'median cut'). We assume that the image
    # is in RGB colour space. The number of requested colours is provided as one of the arguments.
    #
    # If the image contains fewer colours than the requested number then return the original set
    # of colours from the image.
    #
    # In addition to the palette, return a 2D image with the same resolution as 'image' filled
    # with indices into your palette. For example, if 'numColors' is 3, 'image' has the shape (N,M,3)
    # then the index image will have shape (N, M) and it will be filled with values [0, 1, 2].
    # The indices should correspond to the groups of pixels created during the algorithm.
    #
    # Evaluation of this method can look like this:
    # >>> python .\evaluator.py --test=mediancut --image=data/im36.jpg --num_colors=32
    #
    # Hints:
    # - Flatten the image by reshaping it into (-1, 3) - it is easier to manage the groups like that.
    # - Difference between the smallest and largest number in a 1D array can be computed with 'np.ptp'
    # - Median of an array can be computed with 'np.median'.
    # - You can sort with 'np.sort' and you can compute sorted indices into an array with 'np.argsort'.
    # - You can find unique values/rows in a 2D matrix with 'np.unique'.
    # - To get the index of the largest/smallest value in an array, use 'np.argsort'/'np.argmin'

    if len(image) == 0:
        return 
        
    if numColors == 0:
        rAverage = np.mean(image[:,0])
        gAverage = np.mean(image[:,1])
        bAverage = np.mean(image[:,2])
        return [[int(rAverage), int(gAverage), int(bAverage)]]
    # Get the greatest difference in color values along individual channels
    rRange = np.max(image[:,0]) - np.min(image[:,0])
    gRange = np.max(image[:,1]) - np.min(image[:,1])
    bRange = np.max(image[:,2]) - np.min(image[:,2])
    # Array indexing purposes
    greatestRangeIndex = 0
    if max(rRange, gRange, bRange) is rRange:
        greatestRangeIndex = 0
    elif max(rRange, gRange, bRange) is gRange:
        greatestRangeIndex = 1
    else:
        greatestRangeIndex = 2

    # Sort by appropriate channel and get median
    image = image[image[:, greatestRangeIndex].argsort()]
    medianIndex = int(np.ceil(len(image)/2))
    
    firstSplit = preMedianCut(image[0:medianIndex], numColors-1)
    secondSplit = image[medianIndex:]
    rAverage = np.mean(secondSplit[:,0])
    gAverage = np.mean(secondSplit[:,1])
    bAverage = np.mean(secondSplit[:,2])
    toReturn = []
    toReturn.append([int(rAverage), int(gAverage), int(bAverage)])
    # Unpack the lists for better output format
    
    for list in firstSplit:
        toReturn.append(list)
    
    return toReturn

def verifyClusters(image : np.array, newPallete : np.array):
    # Getting Mean Squared Error
    mappedImage = []
    distanceSum = 0
    for i in image:
        closestCentroid = 0
        minDistance = np.linalg.norm(i-newPallete[0])
        for j in range(1,len(newPallete)):
            distance = np.linalg.norm(i-newPallete[j])
            if distance < minDistance:
                closestCentroid = j
                minDistance = distance
        mappedImage.append([i[0],i[1],i[2],closestCentroid])
        distanceSum += minDistance
    mse = distanceSum/len(image)

    # Get average centroid distance
    distances = 0
    for i in newPallete:
        distance = 0
        for j in newPallete:
            if i is not j:
                distance += np.linalg.norm(i-j)
        distances +=(distance/(len(newPallete)-1))

    if mse < distances/((len(newPallete)/2)):
        print("The mean squared error is " + str(mse) + ", which is less then half of the average distance between individual centroids")
        print("The pallete fullfills optimal criteria")
    else:
        print("Something went wrong, the mean square error is greater then average quantile size")
    return mappedImage

def main(args : argparse.Namespace):
    if args.task == "bayer":
        bggrImage = cv2.imread(args.image)
        bayerDemosaic(bggrImage)
    elif args.task == "palette":
        readImage = cv2.imread(args.image)
        medianCut(readImage, args.num_colors)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
