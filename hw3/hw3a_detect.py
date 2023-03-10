
import argparse
import numpy as np
import cv2
# TODO: You may import any library that you deem necessary.
# - If it is not one of 'scipy', 'skimage', 'cv2', 'sklearn', 'matplotlib', 'PIL', 'numpy' then
#   it has to be reasoned and explained.
# - Obviously, libraries implementing the solution of this homework are forbidden.

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--video", default="du3_data/car1.mp4", type=str, help="Video for car detection.")

def main(args : argparse.Namespace):
    # TODO: 2D Kalman filter
    # 
    # Implement an algorithm for tracking of a car in a video. You may start with the video
    # 'car1.mp4' with green background, and thus, easier car detection.

    # NOTE: Video input
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise ValueError("Error while opening video stream or file '{}'!".format(args.video))

    # First frame
    ret, firstFrame = cap.read()
    if not ret:
        raise ValueError("Can't read the first frame of the video '{}'!".format(args.video))
    frameShape = firstFrame.shape

    # TODO: Initialise the variables of Kalman filter.
    # - State of the car (x position, y position, velocity in x, velocity in y)
    # - Acceleration in the direction x and y is 2D random variable with covariance matrix S.
    #   - Then Q = G * S * G^T (^T marks transposition)
    #   - You may use S = [[1, 0], [0, 1]]
    # - Create matrices A and G based on the 1D example from the lecture/last practical.
    # - You may use R = [[1, 0], [0, 1]]

    # TODO: We do not know the real state of the car.
    # - We need to find the car in every frame.

    # NOTE: If you want to learn more about estimation of covariance matrices of noise S and R,
    # then look at https://onlinelibrary.wiley.com/doi/full/10.1002/acs.2783
    # - This is not required for the homework.

    # Go through all frames of the video.
    while cap.isOpened():
        # Read the next frame.
        ret, currentFrame = cap.read()
        if not ret:
            break

        # TODO: Detect the car.
        # TODO: Compute the estimated and corrected position.
        # TODO: Store the detected, estimated and corrected positions for visualisation.
        pass

    # TODO: Plot the three types of position of the car in a single graph.
    pass

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
