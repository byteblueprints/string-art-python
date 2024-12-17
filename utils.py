import math
from typing import Any

import cv2
import numpy as np
from numpy import floating


def get_image_in_grayscale(path: str) -> np.ndarray:
    """
    Reads an image from the specified file path, converts it to grayscale,
    and resizes it to 500x500 pixels.

    Args:
        path (str): The file path of the image to process.

    Returns:
        numpy.ndarray: The resized grayscale image as a NumPy array.
    """
    grayscale_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(grayscale_img, (500, 500))
    return img_resized


def get_nails_coordinates(input_num_of_nails: int, center_x: int, radius: int) -> list:
    """
       Calculate the nails coordinates using the trigonometry

       Args:
           input_num_of_nails :  Number of nails needed.
           center_x :  Center offset x.
           radius :  Radius of the circle.

       Returns:
           circle_coordinates: the result image.
           ex: [(x,y), (x1,y1), (x2,y2)]
       """
    nails_coordinates = []
    for i in range(input_num_of_nails):
        theta = 2 * math.pi * i / input_num_of_nails
        x = center_x + (math.floor(radius * math.cos(theta)))
        y = center_x + (math.floor(radius * math.sin(theta)))
        nails_coordinates.append((x, y))
    return nails_coordinates


def get_all_possible_line_combinations(nail_coordinates: list) -> dict:
    """
    Generates all possible line combinations between nails, considering each pair of nails
    and skipping nearby ones based on a defined `skip` value. The lines are stored in both
    directions (i.e., for each pair of nails, lines are stored for both start-to-end and
    end-to-start connections).

    Args:
        nail_coordinates (list of tuple): A list of (x, y) coordinates representing
                                          the positions of the nails.

    Returns:
        dict: A dictionary where the keys are string representations of the nail pair
              combinations (e.g., "0_5" and "5_0"), and the values are the corresponding
              line coordinates between the nails.
    """
    processed_combinations = []
    line_combinations = {}
    line_count = 0
    skip = 20
    total_nails = len(nail_coordinates)

    for start_index in range(total_nails):
        for end_index in range(start_index + skip, total_nails):
            # Generate unique connection identifiers
            connection_key = f"{start_index}_{end_index}"
            reverse_connection_key = f"{end_index}_{start_index}"

            # Skip if the connection has already been processed or if it's the same nail
            if connection_key in processed_combinations or start_index == end_index:
                continue

            # Get the line coordinates between two nails
            line_vector = get_bresenham_line_coordinates(
                nail_coordinates[start_index][0], nail_coordinates[start_index][1],
                nail_coordinates[end_index][0], nail_coordinates[end_index][1]
            )

            # Mark both directions of the line as processed and store the line
            processed_combinations.append(connection_key)
            processed_combinations.append(reverse_connection_key)
            line_combinations[connection_key] = line_vector
            line_combinations[reverse_connection_key] = line_vector

            line_count += 1
            print(f'\rGenerating Line: {line_count}', end='', flush=True)

    return line_combinations


def get_bresenham_line_coordinates(x0: int, y0: int, x1: int, y1: int) -> list:
    """
    Calculates the coordinates of all points on a line between two points
    using Bresenham's line algorithm.

    Args:
        x0 (int): The x-coordinate of the starting point.
        y0 (int): The y-coordinate of the starting point.
        x1 (int): The x-coordinate of the ending point.
        y1 (int): The y-coordinate of the ending point.

    Returns:
        list of tuple: A list of (x, y) coordinates that form the line between
        the two points.
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    line_coordinates = []
    while True:
        line_coordinates.append((x0, y0))

        if x0 == x1 and y0 == y1:
            break

        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return line_coordinates


def calculate_line_score(error_image: np.ndarray, line_coordinates: list[tuple[int, int]]) -> floating[Any]:
    """
    Calculate the score of a line based on pixel values in the given error image.

    The score is computed as the average pixel value along the line.

    Args:
        error_image (np.ndarray): A 2D numpy array representing the error image (grayscale).
        line_coordinates (list[tuple[int, int]]): A list of (x, y) coordinates representing the line.

    Returns:
        float: The average pixel value along the line.
    """
    pixel_values = [error_image[y][x] for x, y in line_coordinates]
    return np.mean(pixel_values)


def subtract_matrix(error: np.ndarray, line_mask: np.ndarray) -> np.ndarray:
    """
    Subtracts two matrices element-wise, ensuring values remain within the valid range [0, 255].

    The function converts both input matrices to `int32` to avoid underflow during subtraction.
    It then clips the result to the range [0, 255] and returns the final matrix as `uint8`.

    Args:
        error (np.ndarray): The first input matrix.
        line_mask (np.ndarray): The second input matrix to subtract from the first.

    Returns:
        np.ndarray: A matrix resulting from the subtraction, clipped to the range [0, 255]
                    and converted to `uint8` type.
    """
    result = np.clip(error.astype(np.int32) - line_mask.astype(np.int32), 0, 255)
    return result.astype(np.uint8)
