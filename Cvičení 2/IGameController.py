import webbrowser
import cv2 as cv
import numpy as np
from PIL import ImageGrab
from pynput.mouse import Button, Controller
from pynput import keyboard
from pynput.keyboard import Listener
import time


class IGameController():
    def __init__(self, gameUrl, timeout)  -> None:

        self.gameUrl = gameUrl
        self.driver = webbrowser
        self.timeout = timeout

        self.mouseController = Controller()


        screenShot = ImageGrab.grab(bbox=None)
        screenShot_np = np.array(screenShot)
        self.screenWidth = screenShot.size[0]
        self.screenHeight = screenShot.size[1]
        self.bbox = (0, 0, self.screenWidth // 2, self.screenHeight)

        Listener(on_press=self.on_press).start()

    def on_press(self, key):
        if key == keyboard.Key.esc:
            quit()

    def getScreenshotBlackAndWhite(self, bbox=None):
        if bbox is None:
            screenShot = ImageGrab.grab(bbox=self.bbox)
        else:
            screenShot = ImageGrab.grab(bbox=bbox)
        screenShot_np = np.array(screenShot)
        screenShot_cnvt = cv.cvtColor(screenShot_np, cv.COLOR_RGB2BGR)
        screenShot_gray = cv.cvtColor(screenShot_cnvt, cv.COLOR_BGR2GRAY)
        return screenShot_gray
    
    def openGame(self):
        self.driver.open(self.gameUrl)


    def start(self):
        pass