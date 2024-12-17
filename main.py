import numpy as np

from greedy_algorithm import generate_string_art
from utils import get_image_in_grayscale, get_nails_coordinates, get_all_possible_line_combinations


def generate_portrait_string_art(image_path: str, num_of_nails: int, radius: int, center: int,
                                 max_iterations: int, output_scaling_factor: int, string_weight: int) -> np.ndarray:
    """
    Generates a string art portrait from a grayscale image using the greedy algorithm.

    This function loads the specified image, calculates the nail coordinates,
    generates all possible line combinations between nails, and then applies
    the greedy algorithm to iteratively draw the best lines that approximate
    the image in a string art style.

    Args:
        image_path (str): The file path of the input image.
        num_of_nails (int): The number of nails to use in the string art.
        radius (int): The radius of the circular arrangement of nails.
        center (int): The center point of the nail arrangement.
        max_iterations (int): The maximum number of lines to draw.
        output_scaling_factor (int): The scaling factor for the output image to enhance visual quality.
        string_weight (int): The pixel intensity reduction per thread, simulating thread darkness.

    Returns:
        np.ndarray: The generated string art image (scaled) as a NumPy array.
    """

    # Step 1: Load the grayscale image
    grayscale_img = get_image_in_grayscale(image_path)

    # Step 2: Generate nail coordinates based on the number of nails and radius
    nail_coordinates = get_nails_coordinates(num_of_nails, center, radius)

    # Step 3: Get all possible line combinations between nails
    line_combinations = get_all_possible_line_combinations(nail_coordinates)

    # Step 4: Generate the string art using the greedy algorithm
    scaled_output = generate_string_art(
        line_combinations,
        grayscale_img,
        max_iterations,
        nail_coordinates,
        output_scaling_factor,
        string_weight
    )

    return scaled_output


# Example usage:
if __name__ == "__main__":
    image_path = "./resources/taton-moise-zWQ7zsBr5WU-unsplash.jpg"
    num_of_nails = 250
    radius = 249
    center = 250
    max_iterations = 4000
    output_scaling_factor = 20
    string_weight = 20

    # Generate the string art
    string_art_image = generate_portrait_string_art(
        image_path, num_of_nails, radius, center, max_iterations, output_scaling_factor, string_weight
    )

    # Display the result (optional)
    import cv2

    cv2.imshow("String Art Portrait", cv2.resize(string_art_image, (800, 800), interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
