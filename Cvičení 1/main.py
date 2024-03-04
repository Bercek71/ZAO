import cv2
import numpy as np
import pyautogui


# Function to find template in screenshot and perform mouse event
def find_template_and_click(template_path, screenshot_path):
    # Load template and screenshot images
    template = cv2.imread(template_path, 0)
    screenshot = cv2.imread(screenshot_path, 0)

    # Perform template matching
    res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc

    # Calculate center of the matched template
    w, h = template.shape[::-1]
    center_x = top_left[0] + w // 2
    center_y = top_left[1] + h // 2

    # Perform mouse event (click) at the center of the matched template
    pyautogui.click(center_x, center_y)


# Example usage
template_path = 'template.png'
screenshot_path = 'screenshot.png'

# Continuously capture screenshots and perform template matching
while True:
    # Take screenshot of the game window
    screenshot = pyautogui.screenshot()
    screenshot.save(screenshot_path)

    # Find template in the screenshot and perform mouse event
    find_template_and_click(template_path, screenshot_path)
