import sys
import time
import math
import random
import argparse
import subprocess
import threading
import itertools
from copy import deepcopy

import cv2
import yaml
import numpy as np
from skimage.util import random_noise
import imutils

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from image_utilities import random_perspective_transform

FONT = 'din1451mittelschrift.ttf'
FONTNAME = ''

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

DATA_SPLIT = 0.8

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ'
NUMBERS = '0123456789'
HIGHER_CHARS = ['Å', 'Ä', 'Ö']

DEFAULT_CHARACTERS = None

args = None

character_to_output = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'A': 10,
    'B': 11,
    'C': 12,
    'D': 13,
    'E': 14,
    'F': 15,
    'G': 16,
    'H': 17,
    'I': 18,
    'J': 19,
    'K': 20,
    'L': 21,
    'M': 22,
    'N': 23,
    'O': 24,
    'P': 25,
    'Q': 26,
    'R': 27,
    'S': 28,
    'T': 29,
    'U': 30,
    'V': 31,
    'W': 32,
    'X': 33,
    'Y': 34,
    'Z': 35,
    'Å': 36,
    'Ä': 37,
    'Ö': 38,
    '-': 39,
    'Garbage': 40
}

existing_plates = []


class Generator:
    def __init__(self, template):
        self.template = template

    def next(self):
        next_char_template = self.remaining[0]
        self.remaining = self.remaining[1:]

        if next_char_template == 'a':
            if random.random() > 0.5:
                return random.choice(LETTERS)

        if next_char_template == 'A':
            return random.choice(LETTERS)

        if next_char_template == '0':
            if random.random() > 0.5:
                return random.choice(NUMBERS)

        if next_char_template == '1':
            return random.choice(NUMBERS)

        if next_char_template == '-':
            return '-'
        return ''

    def generate(self):
        self.remaining = self.template
        plate = ''
        while len(self.remaining) > 0:
            plate += self.next()
        return plate


def generate_dataset_folders(dataset_folder, out_folder):
    process = subprocess.Popen(['python', '/home/roboto/Documents/Dippen/mkyolodir.py', dataset_folder, out_folder],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    _, _ = process.communicate()


def traverse_or_default(item, keys, default=0):
    # Returns the value of the dictionary contains the whole chain of keys
    # otherwise returns the default value given
    # Useful to not need to use so many if statements
    for i, key in enumerate(keys):
        if key in item:
            item = item[key]
            if i == len(keys)-1:
                return item
        else:
            return default
    return default


def make_plate(plate, settings):
    # Generates a plate image based on the given settings and the given "plate" string
    img = Image.open(settings['base_img'])

    char_height = settings['char_height']

    draw = ImageDraw.Draw(img)

    # Check if we need to shrink the font size to be able to fit all characters
    y_shrinkage = 0
    first_width = 0
    total_width = 0
    while total_width == 0 or total_width > settings['window_width']:
        font = ImageFont.truetype(settings['font'], char_height)
        total_width = 0
        for char in plate:
            char_width = font.getsize(char)[0]
            total_width += char_width + settings['char_space']

        if first_width == 0:
            first_width = total_width

        if total_width > settings['window_width']:
            char_height -= 1
            y_shrinkage += 1

    y_start = settings['y_start']
    y_start += int(y_shrinkage/2)
    x_shrinkage_ratio = total_width/first_width
    y_shrinkage_ratio = char_height/settings['char_height']

    next_x = settings['x_start'] + int((settings['window_width'] - total_width)/2)
    characters = []  # {'char': 'a', 'p1': (55, 126), 'p2': (110, 226)}
    if DEFAULT_CHARACTERS is not None:
        characters += deepcopy(DEFAULT_CHARACTERS)

    # For each character in the given plate string, draw it, create the box
    # data object and calculate position for next character
    for char in plate:
        # Draw the character
        char_width = font.getsize(char)[0]
        draw.text((next_x, y_start), char, BLACK, font=font)

        # Calculate box position/dimensions
        box_padding_top = int(traverse_or_default(settings, ['character_specific_padding', char.lower(), 'top'], default=traverse_or_default(settings, ['box_padding', 'top'])) * y_shrinkage_ratio)
        box_padding_bottom = int(traverse_or_default(settings, ['character_specific_padding', char.lower(), 'bottom'], default=traverse_or_default(settings, ['box_padding', 'bottom'])) * y_shrinkage_ratio)
        box_padding_left = int(traverse_or_default(settings, ['character_specific_padding', char.lower(), 'left'], default=traverse_or_default(settings, ['box_padding', 'left'])) * x_shrinkage_ratio)
        box_padding_right = int(traverse_or_default(settings, ['character_specific_padding', char.lower(), 'right'], default=traverse_or_default(settings, ['box_padding', 'right'])) * x_shrinkage_ratio)
        
        # There is extra space reserved in the font for taller characters such as ÅÄÖ
        if char not in HIGHER_CHARS:
            # Correct the box size accordingly
            box_padding_top -= int(char_height * 0.15)

        y0 = y_start - box_padding_top
        y1 = y_start + char_height + box_padding_bottom
        x0 = next_x - box_padding_left
        x1 = int(next_x + char_width + box_padding_right)

        letter_box_data = {
            'char': char,
            'p1': (x0, y0),
            'p2': (x1, y1)
        }
        characters.append(letter_box_data)

        # Finally advance the next_x position
        next_x = int(next_x + char_width + settings['char_space'])

    return img, characters


def load_yaml(path):
    f = open(path, 'r')
    data = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    if 'font' not in data:
        print("Please provide a 'font' value in your settings file")
        sys.exit(1)

    data['window_width'] = data['x_end'] - data['x_start']
    data['window_height'] = data['y_end'] - data['y_start']

    return data


def square(img):
    h,w = img.shape[:2]
    d = max(w, h)
    return cv2.resize(img, (d,d)), d/w, d/h


class Counter(object):
    # Thread safe counter
    def __init__(self):
        self._number_of_read = 0
        self._counter = itertools.count()
        self._read_lock = threading.Lock()

    def increment(self):
        next(self._counter)

    def value(self):
        with self._read_lock:
            value = next(self._counter) - self._number_of_read
            self._number_of_read += 1
        return value


class GeneratorThread(threading.Thread):
   def __init__(self, threadID, counter, plate, args, settings, subfolder):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.plate = plate
        self.args = args
        self.settings = settings
        self.subfolder = subfolder
        self.counter = counter
   def run(self):
        generate_plate(self.plate, self.args, self.settings, self.subfolder)
        self.counter.increment()


def pillow_to_cv(img):
    img = img.convert('RGB')
    img = np.array(img)
    # Convert RGB to BGR
    img = img[:, :, ::-1].copy()
    return img


def multadd(l, v):
    return [i+v*(i+10) for i in l]


def add_noise(img):
    # Salt and pepper
    img = random_noise(img, mode='s&p', amount=(random.random()*0.1))

    # Random dimming
    black = np.zeros(img.shape, img.dtype)
    a = random.randint(20, 100)/100  # At least 20 % of the original image
    img = cv2.addWeighted(img, a, black, 1-a, 0)

    img = (img)*255/2.  # Rescale from [0-1] to [0-255]
    img = np.uint8(img)  # Truncate decimals

    return img


def draw_random_lines(img):
    lines_to_draw = random.randint(3,10)

    h,w = img.shape[:2]
    for i in range(lines_to_draw):
        thickness = random.randint(11-lines_to_draw, 20-lines_to_draw)
        p1, p2 = random_line(0+thickness,0+thickness,w-thickness,h-thickness)
        cv2.line(img, p1, p2, random_color(), thickness, cv2.LINE_AA)
    
    return img

def random_line(minx, miny, maxx, maxy):
    x0 = random.randint(minx, minx + maxx//2)
    y0 = random.randint(miny, maxy)

    x1 = random.randint(minx + maxx//2, maxx)
    y1 = random.randint(miny, maxy)
    return (x0,y0), (x1,y1)


def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def dot_2d(matrix, p):
    (x, y, _) = matrix.dot((p[0], p[1], 1))
    return (x, y)


def transform_character_rectangles(characters, matrix):
    transformed = []
    for character in characters:
        tl = character['p1']
        br = character['p2']
        tr = (br[0], tl[1])
        bl = (tl[0], br[1])

        tl = dot_2d(matrix, tl)
        br = dot_2d(matrix, br)
        tr = dot_2d(matrix, tr)
        bl = dot_2d(matrix, bl)

        new_points = [tl, br, tr, bl]

        # Top left
        smallest_x = 99999
        smallest_y = 99999

        # Bottom right
        largest_x = 0
        largest_y = 0

        for p in new_points:
            smallest_x = min(smallest_x, p[0])
            smallest_y = min(smallest_y, p[1])
            largest_x = max(largest_x, p[0])
            largest_y = max(largest_y, p[1])

        p1 = (int(smallest_x), int(smallest_y))
        p2 = (int(largest_x), int(largest_y))
        character['p1'] = p1
        character['p2'] = p2
        transformed.append(character)
    return transformed


def perspective_transform(img, points):
    (h, w) = img.shape[:2]

    anchor_points = np.float32(points)
    target_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    matrix = cv2.getPerspectiveTransform(anchor_points, target_points)
    warped = cv2.warpPerspective(img, matrix, (w, h))

    return warped, matrix


def low_res(img):
    h,w = img.shape[:2]
    scaling_factor = max(0.1, random.random())
    new_dimensions = (int(w*scaling_factor), int(h*scaling_factor))
    low_res = cv2.resize(img, new_dimensions)
    return low_res, scaling_factor


def box_contains_point(box, point):
    x1,y1,x2,y2 = box
    px, py = point

    if x1 > px or x2 < px:
        return False
    if y1 > py or y2 < py:
        return False
    return True


def hypotenuse(a, b):
    hyp = math.sqrt(math.pow(a, 2) + math.pow(b, 2))
    return hyp

def box_center_x(box):
    center_x = box[0] + box[2]/2
    return center_x

def box_center_y(box):
    center_y = box[1] + box[3]/2
    return center_y


def find_character_edges(thresh, char):
    w, h = thresh.shape[:2]
    center_x, center_y = w/2, h/2

    # 1. Find contours in the character image crop
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) > 0:
        # Turn the contours into tuples of (bounding_box, contour_area)
        boxes = [(cv2.boundingRect(c), cv2.contourArea(c)) for c in cnts]

        largest_area = max([area for _, area in boxes])
        # Filter out any boxes less than half the size of the largest contour and make new list of only boxes
        boxes = [box for (box, area) in boxes if area > 0.5*largest_area]

        if len(boxes) > 1:
            # Filter out any bounding boxes that are too small
            if char == '-':
                # The hyphen bounding box is not going to be as large as the others
                filtered_boxes = [box for box in boxes if box[2] > 0.25*w]
            else:
                filtered_boxes = [box for box in boxes if box[2] > 0.4*w and box[3] > 0.4*h]
            if len(filtered_boxes) != 0:
                # Just forget it if we end up with 0 boxes
                boxes = filtered_boxes

        # Order the boxes by their distance to the center of their crop.
        boxes = sorted(boxes, key=lambda box: hypotenuse(abs(box_center_x(box)-center_x), abs(box_center_y(box)-center_y)))
        # The best box is the one with the shortest distance to the center
        best_box = boxes[0]

        for box in boxes:
            if box_contains_point(box, (center_x, center_y)):
                # Finally, if one of the boxes actually contain the center point
                # we use that one instead (this could be different than the previously selected best box)
                best_box = box
    
    # 2. Find most central coherent body's coordinates AND the first and last non-bg pixel coordinates)
    y_histogram = np.sum(thresh, axis=1)
    x_histogram = np.sum(thresh, axis=0)
    x0, y0 = None, None
    x1, y1 = None, None
    x_min, y_min, x_max, y_max = None, None, None, None
    mid_x, mid_y = None, None

    y_total_gravity = sum(y_histogram)
    gravity = 0
    for i, v in enumerate(y_histogram):
        gravity += v
        if mid_y is None and gravity > y_total_gravity/2:
            mid_y = i
        # Also save the first and last non-bg coords
        if v > 0:
            if y_min is None and i < len(y_histogram)/2:
                y_min = i
            y_max = i
    x_total_gravity = sum(x_histogram)
    gravity = 0
    for i, v in enumerate(x_histogram):
        gravity += v
        if mid_x is None and gravity > x_total_gravity/2:
            mid_x = i
        # Also save the first and last non-bg coords
        if v > 0:
            if x_min is None and i < len(x_histogram)/2:
                x_min = i
            x_max = i

    count = 0
    # Find last 0 from top to middle
    for val in y_histogram[:mid_y]:
        if val == 0:
            y0 = None
        elif y0 is None:
            y0 = count
        count += 1
    # Find first 0 from middle to bottom
    for val in y_histogram[mid_y:]:
        y1 = count
        if val == 0:
            break
        count += 1

    # Find last 0 from left to middle
    count = 0
    for val in x_histogram[:mid_x]:
        if val == 0:
            x0 = None
        elif x0 is None:
            x0 = count
        count += 1

    # Find first 0 from middle to right
    for val in x_histogram[mid_x:]:
        x1 = count
        if val == 0:
            break
        count += 1

    x_min = 0 if x_min is None else x_min
    y_min = 0 if y_min is None else y_min
    x_max = w-1 if x_max is None else x_max
    y_max = h-1 if y_max is None else y_max

    # return the distance to the character from the edges 
    x1, y1 = len(x_histogram)-x1-1, len(y_histogram)-y1-1
    return x0, y0, x1, y1, x_min, y_min, x_max, y_max, best_box


def char_crop(img, char):
    # char = {'char': 'a', 'p1': (0,0), 'p2': (50, 150)}
    return img[int(char['p1'][1]): int(char['p2'][1]), int(char['p1'][0]): int(char['p2'][0])]


def magic_adjust_label(img, char, name):
    global args

    # Adjust the crop so there is no white space around the character
    char_img = char_crop(img, char)
    char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    char_img = cv2.threshold(char_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if args.debug:
        cv2.imshow(f'{name} before', cv2.bitwise_not(char_img.copy()))

    char_h,char_w = char_img.shape[:2]
    
    character, (x1,y1) , (x2,y2) = char['char'], char['p1'], char['p2']
    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)
    x1_diff, y1_diff, x2_diff, y2_diff, x_min, y_min, x_max, y_max, best_box = find_character_edges(char_img, char['char'])

    if x1_diff is None and best_box is None:
        return char

    # If we were able to detect a contour that's likely to be our character we trust that
    if best_box is not None:
        x1_diff, y1_diff = best_box[0], best_box[1]
        x2_diff, y2_diff = char_w-best_box[2], char_h-best_box[3]
        _,_,cw,ch = best_box
        x2_diff = char_w - (x1_diff+cw)
        y2_diff = char_h - (y1_diff+ch)
    else:
        x2_diff = x1_diff
        y2_diff = y1_diff

    if character in ['Å', 'Ä', 'Ö']:
        # Because these characters have disconnected parts we only adjust the bottom according to the best contour/body detected
        x1 += x_min
        y1 += y_min
        x2 -= (char_w-x_max-1)
        y2 -= y2_diff
    elif character == '-':
        # Keep height same if it's a hyphen
        x1 += x1_diff
        x2 -= x2_diff
    else:
        x1 += x1_diff
        x2 -= x2_diff
        y1 += y1_diff
        y2 -= y2_diff
    char['p1'] = (x1,y1)
    char['p2'] = (x2,y2)

    if args.debug:
        char_img = cv2.threshold(cv2.cvtColor(char_crop(img, char), cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.imshow(f'{name} after', char_img)

    return char


def resize_and_pad(img, target_size):
    targetW = targetH = target_size 
    h,w = img.shape[:2]

    if w == h == target_size:
        return img, 1, 0, 0

    if w == h and w != targetW:
        # Given img has correct proportions but wrong size
        img = cv2.resize(img, (targetW, targetH))
        return img, targetW/w, 0, 0

    wDiff, hDiff = targetW/w, targetH/h
    if w > h:
        newW = targetW
        newH = min(round(wDiff * h), target_size)
        scale_ratio = wDiff
    else:
        newW = min(round(hDiff * w), target_size)
        newH = targetH
        scale_ratio = hDiff

    img = cv2.resize(img, (newW, newH))
    # Create black image of correct size
    newImg = np.zeros((targetH, targetW, 3), np.uint8) * 254

    if newW == targetW:
        # Pad height
        x0 = 0
        x1 = targetW
        missing_height = targetH - newH
        y0 = round(missing_height / 2)
        y1 = y0 + newH
    else:
        # Pad width
        y0 = 0
        y1 = targetH
        missing_width = targetW - newW
        x0 = round(missing_width / 2)
        x1 = x0 + newW

    # Insert the resized image into the black image
    newImg[y0:y1, x0:x1] = img

    return newImg, scale_ratio, x0, y0


def resize_letters(letters, x_scale, y_scale):
    # letters = [{'char': 'a', 'p1': (0,0), 'p2': (50, 150)}, ...]
    for i, letter in enumerate(letters):
        x1, y1 = letter['p1']
        x2, y2 = letter['p2']
        letter['p1'] = (x1*x_scale, y1*y_scale)
        letter['p2'] = (x2*x_scale, y2*y_scale)
        letters[i] = letter
    return letters


def random_blur(img):
    kernel_size = random.randint(1,20)
    kernel = (kernel_size, kernel_size)
    img = cv2.blur(img, kernel)
    return img


def generate_plate(plate, args, settings, subfolder):
    (img, letters) = make_plate(plate, settings)

    plate_name = plate.replace('-', '')

    img = pillow_to_cv(img)
    if args.debug:
        tmp = img.copy()
        for letter in letters:
            cv2.rectangle(tmp, tuple(letter['p1']), tuple(letter['p2']), random_color(), 2)
        cv2.imshow("Original", tmp)

    if args.warp:
        img, matrix = random_perspective_transform(img)
        letters = transform_character_rectangles(letters, matrix)
        # letters = [{'char': 'a', 'p1': (0,0), 'p2': (50, 150)}, ...]

    if args.adjust:
        # Adjust boxes
        for i, char in enumerate(letters):
            if char['char'] != "Garbage":
                char = magic_adjust_label(img, char, f'char #{i}')
                letters[i] = char

    if args.lines:
        img = draw_random_lines(img)

    if args.noise:
        img = add_noise(img)

    if args.blur:
        img = random_blur(img)

    if  args.low_res:
        img, scaling_factor = low_res(img)
        letters = resize_letters(letters, x_scale=scaling_factor, y_scale=scaling_factor)

    if args.square:
        img, x_scale, y_scale = square(img)
        letters = resize_letters(letters, x_scale=x_scale, y_scale=y_scale)
    elif args.letterbox:
        img, scale_ratio, x_offset, y_offset = resize_and_pad(img, 640)
        x_scale = y_scale = scale_ratio
        # letters = [{'char': 'a', 'p1': (0,0), 'p2': (50, 150)}, ...]
        for i, letter in enumerate(letters):
            x1, y1 = letter['p1']
            x2, y2 = letter['p2']
            letter['p1'] = (x_offset + x1*x_scale, y_offset + y1*y_scale)
            letter['p2'] = (x_offset + x2*x_scale, y_offset + y2*y_scale)
            letters[i] = letter

    if args.debug:
        if args.show_bbox:
            with_rects = img.copy()
            for letter in letters:
                cv2.rectangle(with_rects, (int(letter['p1'][0]), int(letter['p1'][1])), (int(
                    letter['p2'][0]), int(letter['p2'][1])), random_color(), 2)

            cv2.imshow('plate', with_rects)
        else:
            cv2.imshow('plate', img)
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == ord('r'):
            cv2.imshow('plate', img)
            key = cv2.waitKey(0)
            if key == ord('q'):
                cv2.destroyAllWindows()
                sys.exit(0)

    if not args.debug:
        filename = f'{settings["file_prefix"]}{plate_name}_{int(time.time())}'
        img_file = f'{args.dataset_folder}/{args.output}/{subfolder}/images/{filename}.jpg'
        cv2.imwrite(img_file, img)

        labels = []
        for letter in letters:
            (h, w) = img.shape[:2]
            char = letter['char']
            width = (letter['p2'][0] - letter['p1'][0]) / w
            height = (letter['p2'][1] - letter['p1'][1]) / h
            x_center = (letter['p1'][0] / w) + width/2
            y_center = (letter['p1'][1] / h) + height/2
            label = f'{character_to_output[char]} {round(x_center, 2)} {round(y_center, 2)} {round(width, 2)} {round(height, 2)}\n'
            labels.append(label)

        label_file = f'{args.dataset_folder}/{args.output}/{subfolder}/labels/{filename}.txt'
        f = open(label_file, 'w+')
        f.writelines(labels)
        f.close()

        if args.debug:
            print("Wrote image to:", img_file)
            print("Wrote labels to:", label_file)


parser = argparse.ArgumentParser()
parser.add_argument('--plate', type=str, help='Generate THIS specific plate')
parser.add_argument('--amount', type=int, help='How many plates to generate')
parser.add_argument('--settings', type=str, required=True, help='The path to the settings file')
parser.add_argument('--warp', action='store_true', help='Apply random warp to the image.')
parser.add_argument('--debug', action='store_true', help="Images will be shown and not saved. Use this for getting the character placement and boxes right.")
parser.add_argument('--show_bbox', action='store_true', help='Show bounding boxes when debug is enabled')
parser.add_argument('--noise', action='store_true', help='How much noise to apply. Default: 0')
parser.add_argument('--dataset_folder', type=str, help='The folder in which the dataset should be created')
parser.add_argument('--output', type=str, required=True, help='Where to write the data')
parser.add_argument('--low-res', action='store_true', help="Randomly make images lower resolution")
parser.add_argument('--square', action='store_true', help="Stretch images to be square")
parser.add_argument('--letterbox', action='store_true', help="Letterbox images to be square")
parser.add_argument('--blur', action='store_true', help="Apply blur of random intensity")
parser.add_argument('--multi-thread', action='store_true', help="Apply blur of random intensity")
parser.add_argument('--lines', action='store_true', help='Draw random lines over the images')
parser.add_argument('--adjust', action='store_true', help='Adjust boxes to better match the bounds of the characters')
args = parser.parse_args()

if __name__ == '__main__':
    command = "".join(sys.argv)

    t0 = time.time()
    generate_dataset_folders(args.dataset_folder, args.output)

    if not args.debug:
        history_file = open(f'{args.dataset_folder}/history.txt', 'a+')
        history_file.write(command + '\n')
        history_file.close()

    settings = load_yaml(args.settings)
    settings['fontname'] = settings['font'].split('/')[-1].replace('.ttf', '')

    if 'characters' in settings:
        DEFAULT_CHARACTERS = deepcopy(settings['characters'])

    if args.plate and args.amount:
        print("Please select only a specific plate OR an amount of random plates")
        parser.print_help()
        sys.exit()

    generator = Generator(settings['template'])

    if args.amount:
        subfolder = 'training'
        count = 0
        plates = []
        while count < args.amount:
            count += 1
            plate = generator.generate()
            if plate not in existing_plates:
                existing_plates.append(plate)
                plates.append(plate)
            percentage = round(100*(count/args.amount), 2)
            print(f'{percentage} % pre-generated')
        
        if args.multi_thread:
            print("Starting generation jobs")
            counter = Counter()
            threads = []
            count = 0
            for plate in plates:
                if count/args.amount > DATA_SPLIT:
                    subfolder = 'testing'
                threadId = len(threads)
                t = GeneratorThread(threadId, counter, plate, args, settings, subfolder)
                threads.append(t)
                count += 1
            
            # Start first 12 threads
            for prev_thread_idx, t in enumerate(threads[:12]):
                t.start()
            
            done_count = 0
            while done_count < len(plates)-1:
                new_count = counter.value()
                if new_count != done_count:
                    # Start threads as other threads finish
                    new_threads = new_count-done_count
                    while new_threads > 0:
                        if prev_thread_idx + 1 < len(threads):
                            threads[prev_thread_idx+1].start()
                            prev_thread_idx += 1
                        new_threads -= 1
                    done_count = new_count
                    print(f"{int(done_count/len(plates) *100)} %")
                time.sleep(0.5)
            for t in threads:
                t.join()
        else:
            count = 0
            for plate in plates:
                if count/args.amount > DATA_SPLIT:
                    subfolder = 'testing'
                generate_plate(plate, args, settings, subfolder)
                count += 1
                print(f"{int(count/len(plates) *100)} %")
        print("Done!")


    if args.plate:
        plate = args.plate.upper()
        generate_plate(plate, args, settings, '/training')

    t1 = time.time()
    print(f"Took {t1 - t0} seconds")

