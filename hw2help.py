
import numpy as np

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
    