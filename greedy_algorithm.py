import numpy as np
import cv2
from utils import calculate_line_score, subtract_matrix


def generate_string_art(
        line_combinations: dict[str, list[tuple[int, int]]],
        base_image: np.ndarray,
        max_lines: int,
        nail_positions: list[tuple[int, int]],
        scale_factor: int,
        thread_weight: float
) -> np.ndarray:
    """
    Generates a string art representation of an image using a greedy algorithm.

    The algorithm iteratively selects the best line (from precomputed options)
    based on pixel intensity scores from a negative image. It minimizes the
    error matrix while visually constructing the string art.

    Args:
        line_combinations (dict[str, list[tuple[int, int]]]): Precomputed line pixel coordinates.
            Keys are strings formatted as "start_nail_end_nail", and values are lists of (x, y) points.
        base_image (np.ndarray): Grayscale input image (negative format is used for processing).
        max_lines (int): The maximum number of lines (threads) to draw.
        nail_positions (list[tuple[int, int]]): List of (x, y) positions for all nails on the canvas.
        scale_factor (int): Scale factor for upscaling the output image.
        thread_weight (float): Intensity reduction per thread, simulating the "darkness" of a thread.

    Returns:
        np.ndarray: Scaled final image of the generated string art.
    """
    # Constants
    SKIP_NEARBY_NAILS = 20  # Avoid short lines
    RECENT_NAIL_LIMIT = 30  # Limit for recently visited nails
    LINE_THICKNESS = 2  # Thickness of the thread lines

    # Image dimensions and initialization
    img_height, img_width = base_image.shape[:2]
    total_nails = len(nail_positions)
    empty_canvas = np.full((img_height, img_width), 255, dtype=np.uint8)  # Blank white canvas
    negative_image = empty_canvas - base_image  # Initial error matrix
    scaled_canvas = np.full((img_height * scale_factor, img_width * scale_factor), 255, dtype=np.uint8)

    # Variables for tracking progress
    recent_nails = []  # Recently visited nails
    nail_sequence = []  # Sequence of drawn nails
    drawn_lines = set()  # To track already drawn lines
    current_nail = 0  # Start nail
    iteration_count = 0  # Line counter

    nail_sequence.append(current_nail)

    while iteration_count < max_lines:
        best_score = -1
        best_target_nail = -1

        # Search for the best possible line to draw
        for offset in range(SKIP_NEARBY_NAILS, total_nails - SKIP_NEARBY_NAILS):
            target_nail = (current_nail + offset) % total_nails
            line_key = f"{current_nail}_{target_nail}"

            # Skip redundant lines or recently used nails
            if target_nail in recent_nails or line_key in drawn_lines:
                continue

            # Score the line based on its contribution to reducing the error matrix
            line_coordinates = line_combinations[line_key]
            score = calculate_line_score(negative_image, line_coordinates)

            if score > best_score:
                best_score = score
                best_target_nail = target_nail

        # Break if no valid target nail is found
        if best_target_nail == -1:
            break

        # Update the error matrix
        best_line_key = f"{current_nail}_{best_target_nail}"
        line_coordinates = line_combinations[best_line_key]
        line_mask = np.zeros((img_height, img_width), dtype=np.float64)

        for x, y in line_coordinates:
            line_mask[y, x] = thread_weight

        negative_image = subtract_matrix(negative_image, line_mask)

        # Draw the line on the upscaled canvas
        start_point = (
            nail_positions[current_nail][0] * scale_factor,
            nail_positions[current_nail][1] * scale_factor
        )
        end_point = (
            nail_positions[best_target_nail][0] * scale_factor,
            nail_positions[best_target_nail][1] * scale_factor
        )
        cv2.line(scaled_canvas, start_point, end_point, (0, 0, 0), LINE_THICKNESS, cv2.LINE_AA)

        # Update tracking structures
        drawn_lines.update({best_line_key, f"{best_target_nail}_{current_nail}"})
        recent_nails.append(best_target_nail)
        if len(recent_nails) > RECENT_NAIL_LIMIT:
            recent_nails.pop(0)

        current_nail = best_target_nail
        nail_sequence.append(current_nail)
        iteration_count += 1

        # Display progress
        cv2.imshow('Negative Image', negative_image)
        cv2.imshow('String Art Progress',
                   cv2.resize(scaled_canvas, (img_width, img_height), interpolation=cv2.INTER_AREA))
        cv2.waitKey(1)

        print(f'\rDrawing Line {iteration_count}/{max_lines}', end='', flush=True)

    print("\nString Art generation complete!")
    return scaled_canvas
