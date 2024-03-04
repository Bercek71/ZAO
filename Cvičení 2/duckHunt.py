import cv2 as cv
import time
from pynput.mouse import Button
from IGameController import IGameController


class DuckHuntController(IGameController):
    def __init__(self, timeout, template, treshold) -> None:
        super().__init__("https://duckhuntjs.com/", timeout)
        self.treshold = treshold
        self.template = cv.imread(template, 0)

    def start(self):
        # self.openGame()
        # time.sleep(5)
        while True:
            screenShot = self.getScreenshotBlackAndWhite()
            tmp_out = cv.matchTemplate(screenShot, self.template, cv.TM_SQDIFF_NORMED)
            min_val, _, min_loc, _ = cv.minMaxLoc(tmp_out)
            # print(min_val)
            if min_val > self.treshold:
                continue
            center = (min_loc[0] + self.template.shape[0] // 2, min_loc[1])
            self.mouseController.position = center
            self.mouseController.click(Button.left, 1)
            time.sleep(self.timeout)