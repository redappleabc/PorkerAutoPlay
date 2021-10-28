# pylint: disable=ungrouped-imports

from sys import platform
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMessageBox
from PyQt5 import QtGui,QtCore
from PIL.ImageQt import ImageQt
from scraper.table_scraper_nn import CardNeuralNetwork
# pylint: disable=unnecessary-lambda
class UIActionAndSignals(QObject):  # pylint: disable=undefined-variable
    signal_update_screenshot_pic = pyqtSignal(object)
    signal_update_can_start = pyqtSignal(object)

    def __init__(self, ui_main_window):
        QObject.__init__(self)  # pylint: disable=undefined-variable
        self.signal_update_screenshot_pic.connect(self.update_screenshot_pic)
        self.signal_update_can_start.connect(self.receiveMessage)
        self.ui = ui_main_window
        self.ui.button_start.clicked.connect(self.startbuttonclicked)
        self.ui.button_train.clicked.connect(self.trainForCardRecognition)
        self.ui.button_stop.clicked.connect(self.stopbuttonclicked)
        self.ui.button_start.setEnabled(False)
        self.ui.button_stop.setEnabled(False)


        self.pause_thread = True
        self.exit_thread = False

        self.jit_compiled = False
        self.get_table = False
        self.ui.button_train.hide()

    def startbuttonclicked(self):
        self.pause_thread = False
        self.ui.button_stop.setEnabled(True)
        self.ui.button_start.setEnabled(False)
    


    def stopbuttonclicked(self):
        self.pause_thread = True
        self.ui.button_stop.setEnabled(False)
        self.ui.button_start.setEnabled(True)


    
    def close(self):
        self.pause_thread = True
        self.exit_thread = False
    @pyqtSlot(object)  
    def receiveMessage(self,message):
        if message == "You can start":
            self.jit_compiled = True
        msg = QMessageBox(self.ui)
        msg.setText(message)
        msg.exec()
        if self.get_table and self.jit_compiled:
            self.ui.button_start.setEnabled(True)

    @pyqtSlot(object)
    def update_screenshot_pic(self, screenshot):
        """Update label with screenshot picture"""
        #log.info("Convert to to pixmap")
        qim = ImageQt(screenshot).copy()
        self.screenshot_image = QtGui.QPixmap.fromImage(qim)
        #log.info("Update screenshot picture")
        self.ui.label_tableImage.setPixmap(self.screenshot_image.scaled(self.ui.label_tableImage.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
        #self.ui.label_tableImage.adjustSize()

    def trainForCardRecognition(self):
        CardNeuralNetwork.create_augmented_images()
        Card_nn = CardNeuralNetwork()
        Card_nn.train_neural_network()
        Card_nn.save_model_to_disk()
        print("traing end")
        

       





      

        
        

    
