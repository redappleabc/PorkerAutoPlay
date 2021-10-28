import logging
import random
import time

import numpy as np

import pymouse

class MouseControl():
    def __init__(self):
        self.mouse = pymouse.PyMouse()
        self.old_x = int(np.round(np.random.uniform(0, 500, 1)))
        self.old_y = int(np.round(np.random.uniform(0, 500, 1)))

    def click(self, x, y,repeat):
        self.mouse.move(x, y)
        for i in range(repeat):
            self.mouse.click(x, y)
        time.sleep(np.random.uniform(0.01, 0.1, 1)[0])

    def mouse_mover(self, x1, y1, x2, y2):
        speed = 1
        stepMin = 7
        stepMax = 20
        rd1 = int(np.round(np.random.uniform(stepMin, stepMax, 1)[0]))
        rd2 = int(np.round(np.random.uniform(stepMin, stepMax, 1)[0]))

        max_diff = max(abs(x2-x1),abs(y2-y1))
        step = 30.0
        if max_diff < 200:
            step = 30.0
        elif max_diff < 300:
            step = 35.0
        elif max_diff < 400:
            step = 40.0
        elif max_diff < 700:
            step = 45.0
        elif max_diff < 1000:
            step = 50.0
        
        if abs(x2 - x1) > 0:
            xa = list(np.arange(x1, x2, (x2-x1)/(5*max_diff/step)))
            xa = [int(i) for i in xa]
        else: xa = []
        if abs(y2 - y1) > 0:
            ya = list(np.arange(y1, y2, (y2-y1)/(5*max_diff/step)))
            ya = [int(i) for i in ya]
        else: ya = []
        if x2 - x1 == 0 and y2 - y1 == 0:
            return
        for k in range(0, max(0, len(xa) - len(ya))):
            ya.append(y2)
        for k in range(0, max(0, len(ya) - len(xa))):
            xa.append(x2)
        ya.append(y2)
        xa.append(x2)
        xTremble = 3
        yTremble = 3
        for i in range(len(max(xa, ya))):
            x = xa[i] + int(+random.random() * xTremble)
            y = ya[i] + int(+random.random() * yTremble)
            self.mouse.move(x, y)
            time.sleep(np.random.uniform(0.01 * speed, 0.03 * speed, 1)[0])
        self.old_x = x2
        self.old_y = y2

    def mouse_clicker(self, x2, y2, buttonToleranceX, buttonToleranceY,repeat = 1):
        old_x1 = self.mouse.position()[0]
        old_y1 = self.mouse.position()[1]
        time.sleep(np.random.uniform(0.1, 0.2, 1)[0])
        self.mouse_mover(old_x1, old_y1, x2, y2)
        self.click(x2, y2 ,repeat)
        #self.mouse.press(x2 + xrand, y2 + yrand)
        time.sleep(np.random.uniform(0.1, 0.5, 1)[0])
        #self.mouse.release(x2,y2)

    def mouse_pressor(self,x1,y1,toleranceX,toleranceY):
        xrand = int(np.random.uniform(0, toleranceX, 1)[0])
        yrand = int(np.random.uniform(0, toleranceY, 1)[0])
        time.sleep(np.random.uniform(0.1, 0.2, 1)[0])
        self.mouse.press(x1 + xrand, y1 + yrand)
        time.sleep(np.random.uniform(0.1, 0.5, 1)[0])
    

    def drag_and_drop(self,x1,y1,x2,y2):
        old_x1 = self.mouse.position()[0]
        old_y1 = self.mouse.position()[1]
        self.mouse_mover(old_x1, old_y1, x1, y1)
        self.mouse_pressor(x1,y1,1,1)
        self.mouse_mover(x1,y1,x2,y2)
        self.mouse.release(x2,y2)


