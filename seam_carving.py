import cv2
import argparse
import numpy as np
from progress.bar import ChargingBar


class MinPathCell:
    def __init__(self):
        self.energy = 0
        self.cumulative_energy = 0
        self.pos = (0, 0)
        self.prev = None


def compute_energy(image):
    energy = np.zeros((image.shape[0] - 2, image.shape[1] - 2))
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            partial_x = (int(image[y, x - 1]) - int(image[y, x + 1])) / 2
            partial_y = (int(image[y - 1, x]) - int(image[y + 1, x])) / 2

            energy[y - 1, x - 1] = abs(partial_x) + abs(partial_y)
    return energy


def create_min_path_cells(energy):
    cells = []
    for x in range(0, energy.shape[1]):
        cell_column = []
        for y in range(0, energy.shape[0]):
            cell = MinPathCell()
            cell.pos = (x + 1, y + 1)
            cell.energy = energy[y, x]
            cell_column.append(cell)
        cells.append(cell_column)

    return cells


def pick_next_cell_horizontal(cells, cell, x, y):
    cell_column_size = len(cells[0])
    left_up_cost = 1e9 if y == 0 else cells[x - 1][y - 1].cumulative_energy
    left_cost = cells[x - 1][y].cumulative_energy
    left_down_cost = 1e9 if y == cell_column_size - 1 else cells[x - 1][y + 1].cumulative_energy

    if left_up_cost <= left_cost and left_up_cost <= left_down_cost:
        cell.prev = cells[x - 1][y - 1]
        cell.cumulative_energy = cell.energy + left_up_cost
    elif left_cost <= left_up_cost and left_cost <= left_down_cost:
        cell.prev = cells[x - 1][y]
        cell.cumulative_energy = cell.energy + left_cost
    else:
        cell.prev = cells[x - 1][y + 1]
        cell.cumulative_energy = cell.energy + left_down_cost


def pick_next_cell_vertical(cells, cell, x, y):
    cell_row_size = len(cells)

    up_left_cost = 1e9 if x == 0 else cells[x - 1][y - 1].cumulative_energy
    up_cost = cells[x][y - 1].cumulative_energy
    up_right_cost = 1e9 if x == cell_row_size - 1 else cells[x + 1][y - 1].cumulative_energy

    if up_left_cost <= up_cost and up_left_cost <= up_right_cost:
        cell.prev = cells[x - 1][y - 1]
        cell.cumulative_energy = cell.energy + up_left_cost
    elif up_cost <= up_left_cost and up_cost <= up_right_cost:
        cell.prev = cells[x][y - 1]
        cell.cumulative_energy = cell.energy + up_cost
    else:
        cell.prev = cells[x + 1][y - 1]
        cell.cumulative_energy = cell.energy + up_right_cost


def pick_min_cell_horizontal(cells):
    # now find the cell on the right that is part of the cheapest path
    min_cell = cells[-1][1]
    min_path_cost = cells[-1][1].cumulative_energy
    for rightmost_cell in cells[-1][1:-1]:
        if rightmost_cell.cumulative_energy < min_path_cost:
            min_path_cost = rightmost_cell.cumulative_energy
            min_cell = rightmost_cell

    return min_cell


def pick_min_cell_vertical(cells):
    # now find the cell on the bottom that is part of the cheapest path
    min_cell = cells[1][-1]
    min_path_cost = cells[1][-1].cumulative_energy
    for column in cells[1:-1]:
        bottom_cell = column[-1]
        if bottom_cell.cumulative_energy < min_path_cost:
            min_path_cost = bottom_cell.cumulative_energy
            min_cell = bottom_cell

    return min_cell


def compute_cells_horizontal(cells):
    for x, cell_column in enumerate(cells):
        for y, cell in enumerate(cell_column):
            if x == 0:
                cell.cumulative_energy = cell.energy
                continue

            pick_next_cell_horizontal(cells, cell, x, y)


def compute_cells_vertical(cells):
    for y in range(len(cells[0])):
        for x in range(len(cells)):
            cell = cells[x][y]
            if y == 0:
                cell.cumulative_energy = cell.energy
                continue

            pick_next_cell_vertical(cells, cell, x, y)


def compute_seam_in_direction(cells, min_cell):
    # backtrace the min cell to find the path
    pixels = []
    cur_cell = min_cell
    while cur_cell.prev is not None:
        pixels.append(cur_cell.pos)
        cur_cell = cur_cell.prev

    return pixels


def compute_seam_horizontal(cells):
    compute_cells_horizontal(cells)
    min_cell = pick_min_cell_horizontal(cells)
    return compute_seam_in_direction(cells, min_cell)


def compute_seam_vertical(cells):
    compute_cells_vertical(cells)
    min_cell = pick_min_cell_vertical(cells)
    return compute_seam_in_direction(cells, min_cell)


def remove_seam_horizontal(image, seam):
    # sort the seam by x value
    seam.sort(key=lambda p: p[0])

    new_image_shape = list(image.shape)
    new_image_shape[0] -= 1
    new_image_shape = tuple(new_image_shape)

    new_image = np.zeros(new_image_shape)
    for x in range(0, image.shape[1]):
        if x < 2:
            remove_point = (x, seam[0][1])
        elif x == image.shape[1] - 1:
            remove_point = (image.shape[1] - 1, seam[-1][1])
        else:
            remove_point = seam[x - 2]

        offset = 0
        for y in range(0, image.shape[0] - 1):
            if (x, y) == remove_point:
                offset = 1
            new_image[y, x] = image[y + offset, x]

    return new_image


def remove_seam_vertical(image, seam):
    # sort the seam by y value
    seam.sort(key=lambda p: p[1])

    new_image_shape = list(image.shape)
    new_image_shape[1] -= 1
    new_image_shape = tuple(new_image_shape)

    new_image = np.zeros(new_image_shape)
    for y in range(0, image.shape[0]):
        if y < 2:
            remove_point = (seam[0][0], y)
        elif y == image.shape[0] - 1:
            remove_point = (seam[-1][0], image.shape[0] - 1)
        else:
            remove_point = seam[y - 2]

        offset = 0
        for x in range(0, image.shape[1] - 1):
            if (x, y) == remove_point:
                offset = 1
            new_image[y, x] = image[y, x + offset]

    return new_image


def remove_seam(args, image, seam):
    if args.vertical_slices:
        return remove_seam_vertical(image, seam)
    else:
        return remove_seam_horizontal(image, seam)


def draw_seam(seam, image):
    # make all the pixels in the min path red
    for pixel in seam:
        image[pixel[1], pixel[0], 0] = 0
        image[pixel[1], pixel[0], 1] = 0
        image[pixel[1], pixel[0], 2] = 255


def compute_seam(args, image):
    energy = compute_energy(image)
    cells = create_min_path_cells(energy)
    if args.vertical_slices:
        seam = compute_seam_vertical(cells)
    else:
        seam = compute_seam_horizontal(cells)

    return seam


parser = argparse.ArgumentParser(description='Compute and remove seams from images to crop them')
parser.add_argument('command', type=str,
                    help='draw|crop')
parser.add_argument('-p', '--pixel-count', default=30,
                    help='Number of pixels to remove in one direction', type=int)
parser.add_argument('-o', '--out-file', default='seam_carved.png',
                    help='The file to write the final picture to', type=str)
parser.add_argument('-y', '--vertical-slices', action='store_true', default=False,
                    help='Remove vertical slices from the image to crop its width')
parser.add_argument('-x', '--horizontal-slices', action='store_true', default=True,
                    help='Remove horizontal slices from the image to crop its height. This is the default')
args = parser.parse_args()

image = cv2.imread('landscape2.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if args.command == 'draw':
    seam = compute_seam(args, gray_image)
    draw_seam(seam, image)

    cv2.imwrite(args.out_file, image)
elif args.command == 'crop':
    final_image = image

    bar = ChargingBar('Removing pixels', max=args.pixel_count)
    for i in range(args.pixel_count):
        seam = compute_seam(args, gray_image)
        gray_image = remove_seam(args, gray_image, seam)
        final_image = remove_seam(args, final_image, seam)

        bar.next()
    bar.finish()

    cv2.imwrite(args.out_file, final_image)
    print('Converted Size: %s -> %s' % (str(image.shape), str(final_image.shape)))
else:
    print('Invalid command. It must be either draw or crop.')
