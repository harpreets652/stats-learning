import numpy as np


def generate_data(step_size, circle_center=(0.5, 0.5), circle_radius=0.25):
    """
    Generates labeled data in a unit square. Positive data inside of specified circle, negative outside

    :param step_size: granularity of the grid
    :param circle_center: center of the circle containing positive data
    :param circle_radius: radius of the circle
    :return:  2-dim x, 1-dim y training data
    """
    # each element in both arrays corresponds to a point in the grid
    x1, x2 = np.meshgrid(np.linspace(0, 1, step_size),
                         np.linspace(0, 1, step_size))

    x, y = [], []
    for i in range(0, x1.shape[0]):
        for j in range(0, x1.shape[1]):
            x.append((x1[i][j], x2[i][j]))

            # set label: 1 if in circle, -1 if outside
            pt_label = 1 if is_point_in_circle(circle_center, circle_radius, (x1[i][j], x2[i][j])) else 0
            y.append(pt_label)

    return np.array(x), np.array(y)


def is_point_in_circle(circle_center, radius, test_point):
    """
    Check if test_point falls inside the circle specified with center and radius

    :param circle_center: (x1, x2)
    :param radius: radius of the circle
    :param test_point: point to test
    :return: True if test point in circle, False otherwise
    """
    dist_to_center_sqrd = (test_point[0] - circle_center[0]) ** 2 + (test_point[1] - circle_center[1]) ** 2

    radius_sqrd = radius ** 2

    return dist_to_center_sqrd <= radius_sqrd
