from PyQt5 import uic,QtCore
from PyQt5.QtWidgets import QMainWindow
class UiPokerbot(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self, None, QtCore.Qt.WindowStaysOnTopHint)
        uic.loadUi('gui/ui/mainwindow.ui', self)
        self.show()
    def  closeEvent(self, event):
        # do stuff
        self.button_stop.setEnabled(True)
        self.button_stop.click()
        event.accept() # let the window close
