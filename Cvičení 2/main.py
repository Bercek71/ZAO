from IGameController import IGameController
import time
from darts import DartsController
import cv2 as cv
from duckHunt import DuckHuntController

# dartsController = DartsController("imgs/darts/darts.png", 0.6, 0.09)
# dartsController.start()
template = "imgs/duckHunt/template4.png"

duckController = DuckHuntController(0.3, template, 0.0015)
duckController.start()