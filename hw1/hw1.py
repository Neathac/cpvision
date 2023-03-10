
import argparse
import numpy as np
import cv2 # OpenCV

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--image", default="Lighthouse_bggr.png", type=str, help="Image to load.")
parser.add_argument("--task", default="bayer", type=str, help="Selected task.")
parser.add_argument("--colReduce", default=4, type=int, help="Square root of the number of desired colors.")



def getAvg(bggr, position, directions):
    avg = 0
    for i in directions:
        avg += bggr[position[0]+i[0]][position[1]+i[1]][0]
    return(int(np.ceil(avg/len(directions))))

def task01Perform(toOpen : str):
    bggrImage = cv2.imread(toOpen)

    # Create a new array with bggr image dimensions
    newImage = np.zeros((len(bggrImage),len(bggrImage[0]),3), np.uint8)

    diagonalAvg = [[-1, -1], [1,1],[-1,1],[1,-1]]
    horizontalAvg = [[0, -1], [0,1],[-1,0],[1,0]]
    greenAvg = [[-1, -1], [1,1],[-1,1],[1,-1],[0,0]]
    oddRow = [[[[-1,0],[1,0]], greenAvg, [[0, -1],[0,1]]],[diagonalAvg, horizontalAvg, [[0,0]]]]
    evenRow = [[[[0,0]], horizontalAvg, diagonalAvg],[[[0, -1],[0,1]], greenAvg,[[-1,0],[1,0]] ]]
    directions = [evenRow,oddRow ]
    # iterate over the inner rows
    for i in range(1,len(bggrImage)-1):
        rowParity = directions[i%2]
        # iterate over inner columns
        for j in range(1,len(bggrImage[i])-1):
            columnParity = rowParity[j%2]
            newImage[i][j][0] = getAvg(bggrImage, [i,j], columnParity[0])
            newImage[i][j][1] = getAvg(bggrImage, [i,j], columnParity[1])
            newImage[i][j][2] = getAvg(bggrImage, [i,j], columnParity[2])

    # Fill in the edges
    for i in range(1,len(newImage[0])-1):
        newImage[0][i] = newImage[1][i]
        newImage[len(newImage)-1][i] = newImage[len(newImage)-2][i]
    for i in range(1,len(newImage)-1):
        newImage[i][0] = newImage[i][1]
        newImage[i][len(newImage[i])-1] = newImage[i][len(newImage[i])-2]
    cv2.imshow("image", newImage) 
    cv2.waitKey()
    cv2.destroyAllWindows()

def medianCut(image : np.array, numColors : int) -> list:

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
    
    firstSplit = medianCut(image[0:medianIndex], numColors-1)
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
    return

def main(args : argparse.Namespace):
    if args.task == "bayer":
        task01Perform(args.image)
    elif args.task == "palette":
        readImage = cv2.imread(args.image)
        flatArray = []
        # For easier array manipulation
        # Row
        for i in range(len(readImage)):
            # Columns
            for j in readImage[i]:
                flatArray.append(j)
        result = medianCut(np.array(flatArray), args.colReduce-1)
        verifyClusters(np.array(flatArray), np.array(result))
        return result

    # Task 3 is taken care of in the verifyClusters function

    pass


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
