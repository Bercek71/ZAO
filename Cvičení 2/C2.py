import cv2 as cv
import numpy as np
from PIL import ImageGrab
from pynput import keyboard
from pynput.mouse import Button, Controller
import time
import threading

def getTemplateCenter(template, grey_image):
    tmp_out = cv.matchTemplate(grey_image, template, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(tmp_out)
    # print(min_val)
    center = (min_loc[0] + template.shape[0] // 2, min_loc[1] + template.shape[1] // 2)
    return center, tmp_out, min_val

def getTemplatesCenters(templates, grey_image):
    centers = []
    tmp_outs = []
    min_vals = []
    for template in templates:
        center, tmp_out, min_val = getTemplateCenter(template, grey_image)
        centers.append(center)
        tmp_outs.append(tmp_out)
        min_vals.append(min_val)
    return centers, tmp_outs, min_vals

image_grab_cnvt = None
center = None
tmp_out = None

# Define global variables
def darts():
    mouse = Controller()
    n_frame = 0
    window_name = 'frame'
    win_out_name = 'frame_out'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.namedWindow(win_out_name, cv.WINDOW_NORMAL)
    template = cv.imread('darts.png', 0)

    screen = ImageGrab.grab(bbox=None)
    screen_np = np.array(screen)
    screen_width = screen.size[0]
    screen_height = screen.size[1]

    def on_press(key):
        if key == keyboard.Key.q:
            return False 


    with keyboard.Listener(on_press=on_press) as listener:
        nextCenter = None
        while True:
            image_grab = ImageGrab.grab(bbox=None)
            image_grab_np = np.array(image_grab)
            image_grab_cnvt = cv.cvtColor(image_grab_np, cv.COLOR_RGB2BGR)        
            image_grab_gray = cv.cvtColor(image_grab_cnvt, cv.COLOR_BGR2GRAY)
            center, tmp_out, min_val = getTemplateCenter(template, image_grab_gray)
            cv.imshow(win_out_name, tmp_out)
            cv.circle(image_grab_cnvt, center, 30, (0, 255, 0), -1)
            cv.imshow(window_name, image_grab_cnvt)
            if min_val > 0.1:
                continue
            # print(min_val)
            mouse.position = (center[0], center[1])
            mouse.click(Button.left, 1)
            time.sleep(0.7)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break  
            if not listener.running:
                break

    cv.destroyAllWindows()

def duckHunt():
    mouse = Controller()
    n_frame = 0
    window_name = 'frame'
    win_out_name = 'frame_out'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.namedWindow(win_out_name, cv.WINDOW_NORMAL)
    template1 = cv.imread('duck1.png', 0)
    template2 = cv.imread('duck2.png', 0)
    # template3 = cv.imread('duck3.png', 0)
    # template4 = cv.imread('duck4.png', 0)
    template5 = cv.imread('duck5.png', 0)
    template6 = cv.imread('duck6.png', 0)
    templates = [template1, template2, template5, template6]

    screen = ImageGrab.grab(bbox=None)
    screen_np = np.array(screen)
    screen_width = screen.size[0]
    screen_height = screen.size[1]
    def on_press(key):
        if key == keyboard.Key.q:
            return False 
    with keyboard.Listener(on_press=on_press) as listener:
        nextCenter = None
        while True:
            image_grab = ImageGrab.grab(bbox=(0, 0, screen_width / 2, screen_height))
            image_grab_np = np.array(image_grab)
            image_grab_cnvt = cv.cvtColor(image_grab_np, cv.COLOR_RGB2BGR)
            image_grab_gray = cv.cvtColor(image_grab_np, cv.COLOR_BGR2GRAY)

            centers, tmp_outs, min_vals = getTemplatesCenters(templates, image_grab_gray)
            # find lowest min_val
            min_val = min(min_vals)
            min_val_index = min_vals.index(min_val)
            center = centers[min_val_index]
            tmp_out = tmp_outs[min_val_index]
            cv.circle(image_grab_cnvt, center, 30, (0, 255, 0), -1)
            cv.imshow(window_name, image_grab_cnvt)
            cv.imshow(win_out_name, tmp_out)
            if min_val > 0.1:
                continue
            print(min_val)
            print("SHOOTING")
            mouse.position = (center[0], center[1])
            mouse.click(Button.left, 1)
            time.sleep(0.3)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break  
            if not listener.running:
                break
    cv.destroyAllWindows()


darts()