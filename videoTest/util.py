import math
import cv2

# Get euclidean distance of two points
def eud_dist(a_x, a_y, b_x, b_y):
    dist = ((a_x - b_x) ** 2) + ((a_y - b_y) ** 2)
    return math.sqrt(dist)


# Get midpoint of two points
def find_midpoint(a_x, a_y, b_x, b_y):
    return ((a_x + b_x) / 2, (a_y + b_y) / 2)


# Convert normalized hand coordiates to image coordinates
def hlist_to_coords(hlist, dsize):
    ret = []
    for hl in hlist:
        temp1 = []
        for val in hl.landmark._values:
            temp = [val.x * dsize[0], val.y * dsize[1]]
            temp1.append(temp)
        ret.append(temp1)
    return ret


# Get the distance between two fingertips
def finger_to_finger_dist(hl, f1, f2):
    i1 = f1 * 4
    i2 = f2 * 4
    point1 = hl[i1]
    point2 = hl[i2]
    distance = eud_dist(point1[0], point1[1], point2[0], point2[1])
    return distance


# Check if thumb is near finger based on a upper and lower bound
def is_thumb_near_finger(hl, finger, lb, ub):
    return is_finger_near_finger(hl, 1, finger, lb, ub)


# Check if finger is near finger based on a upper and lower bound
def is_finger_near_finger(hl, f1, f2, lb, ub):
    return lb < finger_to_finger_dist(hl, f1, f2) < ub


# Get the half dimensions of a image
def get_half_dimensions(img):
    # percent by which the image is resized
    scale_percent = 50

    # calculate the 50 percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)
    return dsize


# Draw the grid lines
def drawlines(dim, img, b):
    offset = b.side_length

    # Green color in BGR
    color = (0, 255, 0)

    # Line thickness of 9 px
    thickness = 2

    x_start = 0
    y_start = 0
    while x_start <= dim[0]:
        img = cv2.line(img, (x_start, y_start), (x_start, dim[1]), color, thickness)
        x_start += offset

    x_start = 0
    while y_start <= dim[1]:
        img = cv2.line(img, (x_start, y_start), (dim[0], y_start), color, thickness)
        y_start += offset

    return img