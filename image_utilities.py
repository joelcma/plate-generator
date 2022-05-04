import sys
import random

import cv2
import numpy as np

DEBUG = False

def random_perspective_transform(img):
    # letters = [{'char': 'a', 'p1': (0,0), 'p2': (50, 150)}, ...]
    (h, w) = img.shape[:2]
    # Locate points of the documents or object which you want to transform
    start_points = np.float32([
        [0, 0],
        [w-1, 0],
        [w-1, h-1],
        [0, h-1]
    ])

    x_sign = 1
    if random.randint(1, 2) % 2 == 0:
        x_sign = -1  
    y_sign = 1
    if random.randint(1, 2) % 2 == 0:
        y_sign = -1
    
    # x_warp_up = x_sign * random.randint(0, 100)
    # x_warp_down = x_sign * random.randint(50, 200)
    # y_warp_left = y_sign * random.randint(50, 200)
    # y_warp_right = y_sign * random.randint(50, 200)
    x_warp_up = x_sign * random.randint(0, int(h*0.8))
    x_warp_down = x_sign * random.randint(0, int(h*0.8))
    y_warp_left = y_sign * random.randint(0, int(w*0.8))
    y_warp_right = y_sign * random.randint(0, int(w*0.8))

    tl = [0+x_warp_up, 0+y_warp_left]
    tr = [w+x_warp_up, 0+y_warp_right]
    br = [w+x_warp_down, h+y_warp_right]
    bl = [0+x_warp_down, h+y_warp_left]

    min_x = min(tl[0], tr[0], br[0], bl[0])
    min_y = min(tl[1], tr[1], br[1], bl[1])
    max_x = max(tl[0], tr[0], br[0], bl[0])
    max_y = max(tl[1], tr[1], br[1], bl[1])
    new_width = max_x - min_x
    new_height = max_y - min_y

    if min_x < 0:
        diff = abs(min_x)
        tl[0] += diff
        tr[0] += diff
        bl[0] += diff
        br[0] += diff

    if min_y < 0:
        diff = abs(min_y)
        tl[1] += diff
        tr[1] += diff
        bl[1] += diff
        br[1] += diff

    if max_x > w:
        diff = abs(min_x)
        tl[0] -= diff
        tr[0] -= diff
        bl[0] -= diff
        br[0] -= diff

    if max_y > h:
        diff = abs(min_y)
        tl[1] -= diff
        tr[1] -= diff
        bl[1] -= diff
        br[1] -= diff

    target_points = np.float32([
        tl,
        tr,
        br,
        bl
    ])

    matrix = cv2.getPerspectiveTransform(start_points, target_points)
    warped = cv2.warpPerspective(img, matrix, (new_width, new_height))
    
    return warped, matrix

def random_glare_slow(img):
    glare = (random.randint(-100, 100), random.randint(-100, 100), random.randint(-100, 100))
    (h,w) = img.shape[:2]
    for j in range(h):
        for i in range(w):
            glare_amount = i/w
            pixel = img[j][i]
            pixel[0] = clamp(pixel[0] + int(glare[0]*glare_amount), 0, 255)
            pixel[1] = clamp(pixel[1] + int(glare[1]*glare_amount), 0, 255)
            pixel[2] = clamp(pixel[2] + int(glare[2]*glare_amount), 0, 255)
    return img

def random_glare_fast(img):
    color_maps = [
        cv2.COLORMAP_AUTUMN,
        cv2.COLORMAP_BONE,
        cv2.COLORMAP_JET,
        cv2.COLORMAP_WINTER,
        cv2.COLORMAP_RAINBOW,
        cv2.COLORMAP_OCEAN,
        cv2.COLORMAP_SUMMER,
        cv2.COLORMAP_SPRING,
        cv2.COLORMAP_COOL,
        #cv2.COLORMAP_HSV,
        cv2.COLORMAP_PINK,
        cv2.COLORMAP_HOT
    ]
    color_map = random.choice(color_maps)
    img = cv2.applyColorMap(img, color_map)
    cv2.putText(img, f'color_map: {color_map}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return img

def clamp(val, mini, maxi):
    return max(min(val, maxi), mini)

if __name__ == '__main__':
    while True:
        img = cv2.imread('plates/AZT882.png')

        img = random_glare_slow(img)
        
        img = random_perspective_transform(img, [])
        cv2.imshow('warped', img)

        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)
        cv2.destroyAllWindows()
