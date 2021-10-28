import sys
import threading
import time
import logging
from PyQt5.QtCore import *
from PyQt5 import QtGui, QtWidgets,QtCore
from PyQt5.QtWidgets import QMessageBox
from gui.gui_launcher import UiPokerbot
from gui.action_and_signals import  UIActionAndSignals
from scraper.table_screen_based import TableScreenBased
from poker_control.poker_control import Poker_Control

class ThreadManager(threading.Thread):
    def __init__(self, threadID, name, counter, gui_signals):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self._stopevent = threading.Event(  )
        self.gui_signals = gui_signals
        self.name = name
        self.counter = counter
        self.table = TableScreenBased(self.gui_signals)
        self.poker_control = Poker_Control(self.gui_signals,self.table)


    def run(self):
        loger = logging.getLogger(__name__)
        #compile_jit()
        self.gui_signals.signal_update_can_start.emit("You can start")
        print("jit compiled")
        while True:
            # reload table if changed
            loger.info("thread")
            if  self.gui_signals.pause_thread:
                while self.table.tlc is None or self.gui_signals.pause_thread:
                    time.sleep(0.1)
                    if self.gui_signals.exit_thread:
                        sys.exit()

            tableCards = self.table.tableRecognition()
            #print(tableCards)
            if False and not self.gui_signals.pause_thread and tableCards is not None:
                self.poker_control.run(tableCards)
            time.sleep(.1)
            if self.gui_signals.pause_thread:
                print("is stoped")
            if self.gui_signals.exit_thread:
                 sys.exit(-1)
                 break



def compile_jit():
    from decision_maker.monte_carlo_jit import calc_best_next_state_using_jit,calc_best_next_state_using_jit_3
    from numba.typed import List
    _ = List(['NC', 'NC', 'NC', 'NC', 'NC', 'NC',  'NC','NC', 'NC', 'NC','NC', 'NC','NC'])
    calc_best_next_state_using_jit( _,_,List(["3S","3D","QS","7S","3C"]),List(["NC","NC","NC","NC"]),0,2)
    _ = List(['AC', 'NC', 'NC', 'JS', 'KS', 'NC', 'NC', 'NC', '3S', 'QS', 'NC', 'NC', 'NC'])
    __ = List(['AS', 'NC', 'NC', 'JC', 'KC', 'NC', 'NC', 'NC', '3C', 'QC', 'NC', 'NC', 'NC'])
    calc_best_next_state_using_jit( _,__,List(["3D","JD","QD"]),List(["NC","NC","NC","NC"]),1)
    #calc_best_next_state_using_jit_3( _,__,__,List(["3D","JD","QD"]),List(["NC","NC","NC","NC"]),1)
    #calc_best_next_state_using_jit_3( _,__,__,List(["3S","3D","QS","7S","3C"]),List(["NC","NC","NC","NC"]),0,2)
   


# ==== MAIN PROGRAM =====
if __name__ == '__main__':

    logging.basicConfig(filename="log.txt",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
    loger = logging.getLogger(__name__)
    
    loger.info("start")
    app = QtWidgets.QApplication(sys.argv)
    ui = UiPokerbot()
    gui_signals = UIActionAndSignals(ui)
    thread = ThreadManager(1, "Thread-1", 1, gui_signals)
    thread.start()
    
    try:
        sys.exit(app.exec_())
    except:
        print("Preparing to exit...")
        gui_signals.exit_thread = True