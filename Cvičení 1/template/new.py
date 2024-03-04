import cv2
import numpy as np
import pyautogui
import time

# Function to continuously take screenshots of the game
def take_screenshot():
    screenshot = pyautogui.screenshot()
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# Function to load the template image
def load_template(template_path):
    return cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

# Function to find template in screenshot
def find_template(screen, template):
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    template_width, template_height = template.shape[::-1]
    center_x = max_loc[0] + template_width // 2
    center_y = max_loc[1] + template_height // 2
    return center_x, center_y

# Function to perform mouse event to the center of the localized template
def mouse_event(x, y):
    pyautogui.moveTo(x, y)

# Main function
def main():
    # URL of the game
    game_url = "https://www.addictinggames.com/shooting/dart-master"

    # Load template image
    template_path = "darts.png"
    template = load_template(template_path)

    # Main loop
    while True:
        # Take screenshot
        screenshot = take_screenshot()

        # Find template in screenshot
        try:
            center_x, center_y = find_template(screenshot, template)
        except Exception as e:
            print("Template not found:", e)
            continue

        # Perform mouse event to the center of the localized template
        mouse_event(center_x, center_y)

        # Add a delay to prevent excessive processing
        time.sleep(1)

if __name__ == "__main__":
    main()
