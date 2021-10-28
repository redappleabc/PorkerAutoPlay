
import cv2
import numpy as np
from PIL import Image, ImageGrab
import scraper.table_screen_based as table


def take_screenshot(virtual_box=False):
    screenshot = ImageGrab.grab()
    return screenshot


def get_tableImage_and_topLeftCorner(original_screenshot):
    cv2_screenshot = cv2.cvtColor(np.array(original_screenshot), cv2.COLOR_BGR2RGB)
    cv2_screenshot = cv2.blur(cv2_screenshot,(3,3))
    mask = cv2.inRange(cv2_screenshot,(60,100,10),(110,160,80))
    x=0
    y=0
    w=0
    h=0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 2)
    max_area = 0
    for i , contour in enumerate(contours):
        # discard small contours
        area = cv2.contourArea(contour)
        if area > max_area and area >30000:
            [x,y,w,h] = cv2.boundingRect(contour)
    h = int(table.table_height*w/table.table_width)
    if h < 350:
        return None , None
    cv2_screenshot = cv2.rectangle(cv2_screenshot,(x,y),(x+w,y+h),(0,0,255),3)
    return original_screenshot.crop((x,y,x+w,y+h)), [x,y,w,h]



def find_template_on_screen(template, screenshot, threshold, extended=False):
    """Find template on screen"""
    (row , col, channel) = screenshot.shape
    if channel != 1:
        screenshot = cv2.cvtColor(screenshot,cv2.COLOR_RGB2GRAY)
    res = cv2.matchTemplate(screenshot, template, cv2.TM_SQDIFF_NORMED)
    loc = np.where(res <= threshold)
    min_val, _, min_loc, _ = cv2.minMaxLoc(res)
    bestFit = min_loc
    count = 0
    points = []
    for pt in zip(*loc[::-1]):
        # cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        count += 1
        points.append(pt)

    return count, points, bestFit, min_val


def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

