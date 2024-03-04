import cv2 as cv
from pynput.mouse import Button
import time
from IGameController import IGameController

class DartsController(IGameController):
    def __init__(self, templatePath, timeout, treshold) -> None:
        super().__init__("https://www.addictinggames.com/shooting/dart-master", timeout=timeout)
        self.template = cv.imread(templatePath, 0)
        # self.windowName = "Darts"
        self.treshold = treshold
        # cv.namedWindow(self.windowName, cv.WINDOW_NORMAL)
    
    def getLastCenter(self):
            screenShot = self.getScreenshotBlackAndWhite()
            tmp_out_next = cv.matchTemplate(screenShot, self.template, cv.TM_SQDIFF_NORMED)
            min_val_next, max_val_next, min_loc_next, max_loc_next = cv.minMaxLoc(tmp_out_next)
            if min_val_next > self.treshold:
                return None
            return (min_loc_next[0] + self.template.shape[0] // 2, min_loc_next[1] + self.template.shape[1] // 2)
            

    def predict(self, lastCenter, center):
        if lastCenter is None:
            return center
        dx = center[0] - lastCenter[0]
        dy = center[1] - lastCenter[1]
        distance = (dx ** 2 + dy ** 2)
        if distance == 0:
            return center
        distance = distance ** 0.5
        if distance > 10:
            return None
        if distance < 2:
            return None
        normalizedDx = dx // distance
        normalizedDy = dy // distance
        print("Returning prediction")
        predictedCenter = (center[0] + (normalizedDx * 20 )), center[1] + (normalizedDy * 20 )
        print(f"""Center {center} """)
        print(f"""Predicted Center {predictedCenter} """)
        return predictedCenter
    def start(self):
        self.openGame()
        time.sleep(5)
        lastCenter = None
        while True:
            screenShot = self.getScreenshotBlackAndWhite()
            tmp_out = cv.matchTemplate(screenShot, self.template, cv.TM_SQDIFF_NORMED)
            min_val, _, min_loc, _ = cv.minMaxLoc(tmp_out)
            if min_val > self.treshold:
                continue
            center = (min_loc[0] + self.template.shape[0] // 2, min_loc[1] + self.template.shape[1] // 2)
            center = self.predict(lastCenter, center)
            if center is None:
                lastCenter = self.getLastCenter()
                continue
            self.mouseController.position = center
            self.mouseController.click(Button.left, 1)
            lastCenter = None
            time.sleep(self.timeout)
            screenShot = self.getScreenshotBlackAndWhite()
            tmp_out_next = cv.matchTemplate(screenShot, self.template, cv.TM_SQDIFF_NORMED)
            min_val_next, max_val_next, min_loc_next, max_loc_next = cv.minMaxLoc(tmp_out_next)
            if min_val_next > self.treshold:
                continue
            lastCenter = (min_loc_next[0] + self.template.shape[0] // 2, min_loc_next[1] + self.template.shape[1] // 2)
            