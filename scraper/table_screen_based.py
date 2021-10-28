from tools.screen_operations import take_screenshot
from tools.screen_operations import get_tableImage_and_topLeftCorner
from tools.screen_operations import pil_to_cv2,cv2_to_pil
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMessageBox
import json
import time
import cv2 as cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from scraper.table_scraper_nn import predict
import copy
table_width = 540.0#536#
table_height = 960.0
class TableScreenBased():
    def __init__(self, gui_signals):
        self.screenshot = None
        self.tableImage = None
        self.cv_tableImage = None
        self.showImage = None
        self.tlc = None
        self.gui_signals = gui_signals
        self.me_cardSize = None
        self.enemy_cardSize = None
        self.tableType = None
        self.playerNumber = 0

        self.me_fantasy  = False
        self.enemy_left_fantasy = False
        self.enemy_right_fantasy = False
        
        self.gui_signals.ui.button_getTable.clicked.connect(self.recogTableImage)
        self.gui_signals.ui.radio_tableType2.toggled.connect(self.setTableType2)
        self.gui_signals.ui.radio_tableType3.toggled.connect(self.setTableType3)

        self.fantasyMaskImage = cv2.imread("assets\\fantasy_mask.jpg",0)

        with open("tablelayout.json") as f:
            self.all_layout = json.load(f)


        with open("assets\pics\model.json", 'r') as json_file:
                    loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("assets\pics\model.h5")
        self.class_mapping = None
        with open("assets\pics\model_classes.json") as json_file:
            self.class_mapping = json.load(json_file)



    def take_screenshot(self):
        self.screenshot = take_screenshot()
     
    def get_table_image(self):
        self.take_screenshot()
        if self.tlc is not None:
            self.tableImage = self.screenshot.crop((self.tlc[0],self.tlc[1],self.tlc[0]+self.tlc[2],self.tlc[1]+self.tlc[3]))
            self.cv_tableImage = pil_to_cv2(self.tableImage)
            self.showImage = np.copy(self.cv_tableImage)
            #self.cv_tableImage = cv2.blur(self.cv_tableImage,(3,3))
            return True
        else:
            return False  
    def check_fantasy(self):
        xScale = self.tlc[2]/table_width
        yScale = self.tlc[3]/table_height
        x = int(self.layout["fantasy_region"]["x"] * xScale)
        y = int(self.layout["fantasy_region"]["y"] * yScale)
        width = int(self.layout["fantasy_region"]["width"] * xScale)-10
        height = int(self.layout["fantasy_region"]["height"] * yScale)-1
        fantasy_image =  self.cv_tableImage[y:y+height,x:x+width]
        mask = cv2.inRange(fantasy_image,(0,50,50),(100,220,220))
        count = np.count_nonzero(mask)
        if(count > width*height/3):
            self.me_fantasy = True
        else:
            self.me_fantasy = False
        
        if self.tableType == "2Table":
            x = int(self.layout["enemy_left_fantasy_region"]["x"] * xScale)
            y = int(self.layout["enemy_left_fantasy_region"]["y"] * yScale)
            width = int(self.layout["enemy_left_fantasy_region"]["width"] * xScale)-1
            height = int(self.layout["enemy_left_fantasy_region"]["height"] * yScale)-1
            fantasy_image =  self.cv_tableImage[y:y+height,x:x+width]
            mask = cv2.inRange(fantasy_image,(30,40,80),(100,100,130))
            count = np.count_nonzero(mask)
            if(count > width*height/4):
                self.enemy_left_fantasy = True
            else:
                self.enemy_left_fantasy = False
        elif self.tableType == "3Table":
            x = int(self.layout["enemy_left_fantasy_region"]["x"] * xScale)
            y = int(self.layout["enemy_left_fantasy_region"]["y"] * yScale)
            width = int(self.layout["enemy_left_fantasy_region"]["width"] * xScale)-1
            height = int(self.layout["enemy_left_fantasy_region"]["height"] * yScale)-1
            fantasy_image =  self.cv_tableImage[y:y+height,x:x+width]
            mask = cv2.inRange(fantasy_image,(30,40,80),(100,100,130))
            count = np.count_nonzero(mask)
            if(count > width*height/4):
                self.enemy_left_fantasy = True
            else:
                self.enemy_left_fantasy = False
            
            x = int(self.layout["enemy_right_fantasy_region"]["x"] * xScale)
            y = int(self.layout["enemy_right_fantasy_region"]["y"] * yScale)
            width = int(self.layout["enemy_right_fantasy_region"]["width"] * xScale)-1
            height = int(self.layout["enemy_right_fantasy_region"]["height"] * yScale)-1
            fantasy_image =  self.cv_tableImage[y:y+height,x:x+width]
            mask = cv2.inRange(fantasy_image,(30,40,80),(100,100,130))
            count = np.count_nonzero(mask)
            if(count > width*height/4):
                self.enemy_right_fantasy = True
            else:
                self.enemy_right_fantasy = False

    def confirm_button_is_showed(self):
        xScale = self.tlc[2]/table_width
        yScale = self.tlc[3]/table_height
        x = int(self.layout["confirm_button"]["x"] * xScale)
        y = int(self.layout["confirm_button"]["y"] * yScale)
        width = int(self.layout["confirm_button"]["width"] * xScale)
        height = int(self.layout["confirm_button"]["height"] * yScale)
        button_image =  self.cv_tableImage[y:y+height,x:x+width]
        mask = cv2.inRange(button_image,(20,50,100),(200,250,250))
        count = np.count_nonzero(mask)
        if(count > width*height/2):
            return True
        else:
            return False

    def get_fantasy_cards(self,card = None):
        if card is not None:
            self.get_table_image()
        fantasy_cards = []
        initialMaskImage = self.fantasyMaskImage
        yScale = self.tlc[3]/table_height
        y1 = int(745*yScale)
        y2 = int(855*yScale)
        fantasyImage = self.cv_tableImage[y1:y2,:]
        fantasyImage = cv2.cvtColor(fantasyImage,cv2.COLOR_BGR2GRAY)
        initialMaskImage = cv2.resize(initialMaskImage,(fantasyImage.shape[::-1]))
        _,initialMaskImage = cv2.threshold(initialMaskImage,1,1,cv2.THRESH_BINARY)
        fantasyCardImage = fantasyImage#cv2.blur(fantasyImage,(3,3))
        _,fantasyCardImage = cv2.threshold(fantasyCardImage,120,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        fantasyCardImage = np.array(fantasyCardImage*initialMaskImage,np.uint8)
        cardMask = np.zeros_like(fantasyCardImage)
        contours, _ = cv2.findContours(fantasyCardImage, 1, 2)
        if not contours:
            return []
        for i , contour in enumerate(contours):
            area = cv2.contourArea(contour)
            [x,y,w,h] = cv2.boundingRect(contour)
            if  area < 500 and area >50 and h > 8:
                cv2.drawContours(cardMask,contours,i,255,-1)
        contours, _ = cv2.findContours(cardMask, 1, 2)
        if not contours:
            return []
        new_contours = []
        visited_flag = []
        for i  in range(len(contours)):
            if i  in visited_flag:
                continue
            [x_i,y_i,w_i,h_i] = cv2.boundingRect(contours[i])
            new_contour = np.copy(contours[i])
            for j in range(i+1, len(contours)):
                [x_j,y_j,w_j,h_j] = cv2.boundingRect(contours[j])
                intersect_x = min(x_j + w_j,x_i + w_i) - max(x_j,x_i)
                if intersect_x == min(w_j,w_i) or intersect_x >= max(w_j,w_i)/2:
                    if y_i > y_j:
                        new_contour = np.copy(contours[j])#np.append(new_contour,contours[j],axis = 0)
                    visited_flag += [j]
            new_contours += [new_contour]
        if not new_contours:
            return []

        new_contours = sorted(new_contours,key = lambda x:np.amin(x,axis = 0)[0][0])
        for i , contour in enumerate(new_contours):
            [x,y,w,h] = cv2.boundingRect(contour)
            img = self.cv_tableImage[y1+y-3:y1+y+h+3,x-3:x+w+3]
            card_img = np.ones((h+10,w+10,3),np.uint8)*246
            if card_img[2:h+8,2:w+8].shape != img.shape:
                continue
            card_img[2:h+8,2:w+8] = img
            
            card_class = predict(img,self.model,self.class_mapping)
            if card is not None and card == card_class:
                return [int(x+w/2),int(y1+y+h/2)]
            if card is not None and i == len(new_contours) - 1:
                return [-1,-1]
            fantasy_cards += [card_class]
            cv2.putText(self.showImage, card_class,(x,y1-(80-y*2)), cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),int(2),cv2.LINE_AA)
        return np.array(fantasy_cards)
    def tableRecognition(self):

        tableCards = None
        while True :
            tableCards = np.array([])
            if self.tlc is not None and not self.get_table_image():
                break
            self.check_fantasy()
            condition = self.confirm_button_is_showed()
            self.correction_table_layout()

            for team, value in self.layout.items():
                if team in  ["confirm_button" , "fantasy_region", "enemy_left_fantasy_region" ,"enemy_right_fantasy_region","fantasy_confirm_button"]:
                    continue
                if self.me_fantasy and  (team == "new_card" or team == "thrown_card"):
                        continue
                if self.tableType == "2Table" and self.enemy_left_fantasy and team == "enemy_left":
                    tableCards = np.append(tableCards,np.array(["NC",]*13))
                    continue

                if self.tableType == "3Table" and self.enemy_left_fantasy and team == "enemy_left":
                    tableCards = np.append(tableCards,np.array(["NC",]*13))
                    continue
                if self.tableType == "3Table" and self.enemy_right_fantasy and team == "enemy_right":
                    tableCards = np.append(tableCards,np.array(["NC",]*13))
                    continue

                xScale = self.tlc[2]/table_width
                yScale = self.tlc[3]/table_height
                scaled_height = int(value["card_size"]["height"]*yScale/2)
                scaled_width = int(value["card_size"]["width"]*xScale/2)
                for name,coordinate in value.items():
                    if name == "card_size":
                        continue
                    else:
                        x = int(coordinate["x"] * xScale)
                        y = int(coordinate["y"] * yScale)
                        img =  np.copy(self.cv_tableImage[y:y+scaled_height,x:x+scaled_width])
                        '''mask = cv2.inRange(img , (220,220,220),(255,255,255))
                        card_img = np.ones_like(img)*220
                        contours, _ = cv2.findContours(mask, 1, 2)
                        if  len(contours) != 0:
                            maxArea = 0
                            yy = None
                            for i , contour in enumerate(contours):
                                area = cv2.contourArea(contour)
                                if  area > maxArea:
                                    [xx,yy,ww,hh] = cv2.boundingRect(contour)
                                    maxArea = area
                            if yy is not None:
                                card_img[yy+1:yy+hh,xx+1:xx+ww] = img[yy+1:yy+hh,xx+1:xx+ww]
                        '''
                        card_class = predict(img,self.model,self.class_mapping)
                        tableCards = np.append(tableCards,card_class)
                        cv2.putText(self.showImage,card_class,(x,y), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),int(2),cv2.LINE_AA)
                        cv2.circle(self.showImage,(x,y),5,(0,0,255),5)
            if self.me_fantasy:
                time.sleep(5)
                tableCards = np.append(tableCards,self.get_fantasy_cards())
            if self.showImage is not None:
                self.gui_signals.signal_update_screenshot_pic.emit(cv2_to_pil(self.showImage))
            else:
                self.gui_signals.signal_update_screenshot_pic.emit(self.screenshot)
            if not condition:
                print("not recog confirm button")
            if condition or self.me_fantasy or self.gui_signals.pause_thread:
                break
            time.sleep(.1)
        return tableCards

    def getTableType(self):
        if self.tableImage is None:
            print("error:tableImage is None")
            return
        cv_tableImage = pil_to_cv2(self.tableImage)
        height = cv_tableImage.shape[0]
        width = cv_tableImage.shape[1]
        subImage = cv_tableImage[70:int(height/3),:]
        gray = cv2.cvtColor(subImage,cv2.COLOR_BGR2GRAY)
        blurred = cv2.blur(gray,(3,3))
        thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 3)
        contours, _ = cv2.findContours(np.copy(thresh), cv2.RETR_EXTERNAL, 2)
        cnt = 0
        for i , contour in enumerate(contours):
            # discard small contours
            [x,y,w,h] = cv2.boundingRect(contour)
            if h > 30:
                cnt += 1
        if cnt < 18:
            self.tableType = self.gui_signals.ui.radio_tableType2.text()
            self.gui_signals.ui.radio_tableType2.setChecked(True)
        else:
            self.tableType = self.gui_signals.ui.radio_tableType3.text()
            self.gui_signals.ui.radio_tableType3.setChecked(True)
        self.set_table_layout(self.tableType)
            
    def set_table_layout(self,table_name):
        self.layout = self.all_layout[table_name.lower()]
       
    def correction_table_layout(self):
        self.layout = copy.deepcopy(self.all_layout[self.tableType.lower()])
        if self.me_fantasy:
            for team, value in self.layout.items():
                    if team != "me":
                        continue
                    for name,coordinate in value.items():
                        if name == "card_size":
                             self.layout[team][name]["width"] = coordinate["fantasy_width"]
                             self.layout[team][name]["height"] = coordinate["fantasy_height"]
                             continue
                        self.layout[team][name]["x"] = coordinate["fantasy_x"]
                        self.layout[team][name]["y"] = coordinate["fantasy_y"]
        self.me_cardSize = (self.layout["me"]["card_size"]["width"] ,self.layout["me"]["card_size"]["height"])
        self.enemy_cardSize = (self.layout["enemy_left"]["card_size"]["height"],self.layout["me"]["card_size"]["width"])
    def getPlayerNumber(self):
        pass
    def getStartPlayer(self):
        pass

    
    def recogTableImage(self):
        self.gui_signals.ui.button_start.setEnabled(False)
        self.take_screenshot()
        self.tableImage, self.tlc = get_tableImage_and_topLeftCorner(self.screenshot)
        if self.tableImage is not None:
            self.gui_signals.signal_update_screenshot_pic.emit((self.tableImage))
            self.getTableType()
            self.gui_signals.get_table = True
            if self.gui_signals.jit_compiled:
                self.gui_signals.ui.button_start.setEnabled(True)
            else:
                self.gui_signals.signal_update_can_start.emit("Just a minute. Now Loading Data")
        else:
            self.gui_signals.signal_update_screenshot_pic.emit(self.screenshot)
   
    def setTableType2(self):
        radioButton = self.gui_signals.ui.sender()
        if not radioButton.isChecked():
            return
        if  radioButton.text() == self.tableType:
            return
        if self.tableType is not None:
            reply = QMessageBox.question(self.gui_signals.ui, '確認', 'ゲームの設定を変更しますか？',QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                self.gui_signals.ui.radio_tableType3.setChecked(True)
                return
        self.tableType = radioButton.text()
        self.set_table_layout(self.tableType)
    
    def setTableType3(self):
        radioButton = self.gui_signals.ui.sender()
        if not radioButton.isChecked():
            return
        if  radioButton.text() == self.tableType:
            return
        if self.tableType is not None:
            reply = QMessageBox.question(self.gui_signals.ui, '確認', 'ゲームの設定を変更しますか？',QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                self.gui_signals.ui.radio_tableType2.setChecked(True)
                return
        self.tableType = radioButton.text()
        self.set_table_layout(self.tableType)
        