import numpy as np
from decision_maker.fantasy import calc_best_fantasy_state
from decision_maker.monte_carlo_jit import calc_best_next_state_using_jit,calc_best_next_state_using_jit_3
from tools.mouse_action import MouseControl
import time
from numba.typed import List
import scraper.table_screen_based as table
from timeit import default_timer as timer 
import logging
logging.basicConfig(filename="log.txt",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
loger = logging.getLogger(__name__)
class Poker_Control():
    def __init__(self,gui_signals,table):
        self.mouse = MouseControl()
        self.gui_signals = gui_signals
        self.table = table
        self.me_state = None
        self.me_new_card = None
        self.me_thrown_card = ["NC","NC","NC","NC"]
        self.pre_me_new_card = None
        self.pre_action = None
        self.enemy_left_state = None
        self.enemy_right_state = None
        self.step = 0
        self.start_time = 0

    def run(self,poker_state):
        #print(poker_state)
        self.start_time = timer()
        print("")
        print("--------------------new-------------------------")
        if self.table.me_fantasy:
            loger.info("---------------fantasy-------------")
            loger.info("recog cards is")
            loger.info(poker_state)
            self.checkState(poker_state)
            if self.me_new_card.size < 13 or len(set(self.me_new_card)) != len(self.me_new_card):
                
                print("fantasy cards recog error")
                return
            
            loger.info("fantasy cards is")
            loger.info(self.me_new_card)
            print(self.me_new_card)
            loger.info("caculating fantasy")
            fantasy_state = calc_best_fantasy_state(self.me_new_card)
            fantasy_state_set = set(fantasy_state)
            if not fantasy_state_set.issubset(set(self.me_new_card)):
                loger.info("-----------------error---------------")
                return
            loger.info("best_fantasy_state is")
            loger.info(fantasy_state)

            print("best fantasy state is")
            print(fantasy_state[0:3])
            print(fantasy_state[3:8])
            print(fantasy_state[8:13])
            self.fantasy_act(fantasy_state)
            return
        print("calc next state")
        self.checkState(poker_state)
        self.get_Step()
        if self.step < 0:
            print("recog error")
            return


        me_front_state = self.me_state[0:3]
        me_middle_state = self.me_state[3:8]
        me_back_state = self.me_state[8:13]
        print("enemy_left_state is")
        print(self.enemy_left_state)
        if self.table.tableType == "3Table":
            print("enemy_right_state is")
            print(self.enemy_right_state)

        print(f'the step is {self.step}')
        print("current_state")
        print(me_front_state)
        print(me_middle_state)
        print(me_back_state)
        print("calculating")
        self.me_state = list(self.me_state)
        self.enemy_left_state = list(self.enemy_left_state)
        self.me_new_card = list(self.me_new_card)
        print("new cards is")
        print(self.me_new_card)
        print("thrown cards is ")
        print(self.me_thrown_card)
        if self.table.tableType == "3Table" and self.table.enemy_left_fantasy:
            me_next_state, action ,max_confidence= calc_best_next_state_using_jit(List(self.me_state),List(self.enemy_right_state),List(self.me_new_card),List(self.me_thrown_card),self.step)
        elif self.table.tableType == "3Table" and self.table.enemy_right_fantasy:
            me_next_state, action ,max_confidence= calc_best_next_state_using_jit(List(self.me_state),List(self.enemy_left_state),List(self.me_new_card),List(self.me_thrown_card),self.step)
        elif self.table.tableType == "3Table" and np.all(self.enemy_left_state == "NC"):
            me_next_state, action ,max_confidence= calc_best_next_state_using_jit(List(self.me_state),List(self.enemy_right_state),List(self.me_new_card),List(self.me_thrown_card),self.step)
        elif self.table.tableType == "3Table" and np.all(self.enemy_right_state == "NC"):
            me_next_state, action ,max_confidence= calc_best_next_state_using_jit(List(self.me_state),List(self.enemy_left_state),List(self.me_new_card),List(self.me_thrown_card),self.step)
        elif self.table.tableType =="2Table":     
            me_next_state, action ,max_confidence= calc_best_next_state_using_jit(List(self.me_state),List(self.enemy_left_state),List(self.me_new_card),List(self.me_thrown_card),self.step)
        else:
            print("3table")
            me_next_state, action ,max_confidence= calc_best_next_state_using_jit_3(List(self.me_state),List(self.enemy_left_state),List(self.enemy_right_state),List(self.me_new_card),List(self.me_thrown_card),self.step)
        me_next_state = me_next_state.split("_")
        print("next_state is")
        print(me_next_state[0:3])
        print(me_next_state[3:8])
        print(me_next_state[8:13])
        if self.gui_signals.pause_thread:
            return
        self.act(action)
        self.pre_action = action
        self.pre_me_new_card = self.me_new_card
        print("action is done")
        print("--------------end------------------")
        print("")
        print("")
        
    def get_Step(self):
        me_front_state = self.me_state[0:3]
        me_middle_state = self.me_state[3:8]
        me_back_state = self.me_state[8:13]
        if np.all(me_back_state != "NC") and np.all(self.me_new_card == "NC") and np.all(me_front_state == "NC") and np.all(me_middle_state == "NC"):
            self.step = 0
            self.me_new_card = np.copy(self.me_state[8:13])
            self.me_state[:] = "NC" 
            me_back_state[:] = "NC"
        elif np.all(self.me_new_card != "NC"):
            self.step = int(5 - np.count_nonzero(self.me_state == "NC")/2)
        elif self.pre_action is not None:
            self.act(self.pre_action)
            print("previous action is done")
            self.step = -1
        else:
            print("recog error or start point is wrong")
            self.step = -1
        if self.step < 0:
            print("confirmbtn is not clicked")
            #return

    def checkState(self,poker_state):
        if self.table.tableType =="2Table":
            self.enemy_left_state = poker_state[0:13]
            self.me_state = poker_state[13:26]
            if self.table.me_fantasy:
                self.me_new_card = poker_state[26:]
                self.me_new_card = np.delete(self.me_new_card,np.where(self.me_new_card == "NC"))
            else:
                self.me_new_card = poker_state[26:29]
                self.me_thrown_card = poker_state[29:33]
        elif self.table.tableType == "3Table":
            self.enemy_left_state = poker_state[0:13]
            self.enemy_right_state = poker_state[13:26]
            self.me_state = poker_state[26:39]
            self.me_thrown_card = poker_state[39:42]
            if self.table.me_fantasy:
                self.me_new_card = poker_state[39:]
                self.me_new_card = np.delete(self.me_new_card,np.where(self.me_new_card == "NC"))
            else:
                self.me_new_card = poker_state[39:42]
                self.me_thrown_card = poker_state[42:46]
        else:
            print("tableType error")
        return
        if self.step == 0:
            me_front_state = self.me_state[0:3]
            me_middle_state = self.me_state[3:8]
            self.me_new_card = np.copy(self.me_state[8:13])
            if np.all(me_front_state == "NC") and np.all(me_middle_state == "NC") and not np.all(self.me_new_card == "NC"):
                self.me_state[:] = "NC" 
                return True
            else:
                return False
        if self.step > 0:
            if self.me_new_card.size == 3 and  not np.all(self.me_new_card == "NC"):
                return True
            else:
                return False
    def act(self,str_action):
        me_front_state = np.copy(self.me_state[0:3])
        me_middle_state = np.copy(self.me_state[3:8])
        me_back_state = np.copy(self.me_state[8:13])
        card_pos = [0,0,0]
        card_pos[0] = np.count_nonzero(me_front_state != "NC")
        card_pos[1] = np.count_nonzero(me_middle_state != "NC")
        card_pos[2] = np.count_nonzero(me_back_state != "NC")

        xScale = float(self.table.tlc[2])/table.table_width
        yScale = float(self.table.tlc[3])/table.table_height
        target_positionX = [0,0,0]
        target_positionY = [0,0,0]
        target_positionX[0] = int((self.table.layout["me"]["top1"]["x"] + self.table.me_cardSize[0]/2.0) * xScale + self.table.tlc[0])
        target_positionY[0] = int((self.table.layout["me"]["top1"]["y"] + self.table.me_cardSize[1]/2.0) * yScale + self.table.tlc[1])
        target_positionX[1] = int((self.table.layout["me"]["mid1"]["x"] + self.table.me_cardSize[0]/2.0) * xScale + self.table.tlc[0])
        target_positionY[1] = int((self.table.layout["me"]["mid1"]["y"] + self.table.me_cardSize[1]/2.0) * yScale + self.table.tlc[1])
        target_positionX[2] = int((self.table.layout["me"]["bottom1"]["x"] + self.table.me_cardSize[0]/2.0) * xScale + self.table.tlc[0])
        target_positionY[2] = int((self.table.layout["me"]["bottom1"]["y"] + self.table.me_cardSize[1]/2.0) * yScale + self.table.tlc[1])
        diff_X = int((self.table.me_cardSize[0]+5)*xScale)
        action = np.array(str_action.split("_"))
        print("action is")
        print(action)
        action = action.astype(np.int)
        if self.step == 0:
            position_X = int((self.table.layout["me"]["bottom1"]["x"] + self.table.me_cardSize[0]/2.0) * xScale + self.table.tlc[0])
            position_Y = int((self.table.layout["me"]["bottom1"]["y"] + self.table.me_cardSize[1]/2.0) * yScale + self.table.tlc[1])
            for i, target in enumerate(action):
                if target == 2:
                    continue
                else:
                    self.mouse.drag_and_drop(i*diff_X + position_X,position_Y,target_positionX[target] + card_pos[target]*diff_X,target_positionY[target])
                    card_pos[target] += 1
                    time.sleep(.1)
        else:
            position_new0_X = int((self.table.layout["new_card"]["new1"]["x"] + self.table.layout["new_card"]["card_size"]["width"]/2.2) * xScale + self.table.tlc[0])
            position_new0_Y = int((self.table.layout["new_card"]["new1"]["y"] + self.table.layout["new_card"]["card_size"]["height"]/2.2) * yScale + self.table.tlc[1])
            position_new1_X = int((self.table.layout["new_card"]["new2"]["x"] + self.table.layout["new_card"]["card_size"]["width"]/2.2) * xScale + self.table.tlc[0])
            position_new1_Y = int((self.table.layout["new_card"]["new2"]["y"] + self.table.layout["new_card"]["card_size"]["height"]/2.2) * yScale + self.table.tlc[1])
            position_new2_X = int((self.table.layout["new_card"]["new3"]["x"] + self.table.layout["new_card"]["card_size"]["width"]/2.2) * xScale + self.table.tlc[0])
            position_new2_Y = int((self.table.layout["new_card"]["new3"]["y"] + self.table.layout["new_card"]["card_size"]["height"]/2.2) * yScale + self.table.tlc[1])
            if action[0] == 0:
                self.mouse.drag_and_drop(position_new1_X,position_new1_Y,target_positionX[action[1]] + card_pos[action[1]]*diff_X,target_positionY[action[1]])
                card_pos[action[1]] += 1

                #print(f"the action is from new_card 1 to {action[1]}")
                time.sleep(.1)
                self.mouse.drag_and_drop(position_new2_X,position_new2_Y,target_positionX[action[2]] + card_pos[action[2]]*diff_X,target_positionY[action[2]])
                card_pos[action[2]] +=1
                #print(f"the action is from new_card 2 to {action[2]}")
                time.sleep(.1)
            elif action[0] == 1:
                self.mouse.drag_and_drop(position_new0_X,position_new0_Y,target_positionX[action[1]] + card_pos[action[1]]*diff_X,target_positionY[action[1]])
                card_pos[action[1]] += 1
               
                #print(f"the action is from new_card 0 to {action[1]}")
                time.sleep(.1)
                self.mouse.drag_and_drop(position_new2_X,position_new2_Y,target_positionX[action[2]] +  card_pos[action[2]]*diff_X,target_positionY[action[2]])
                card_pos[action[2]] += 1
                #print(f"the action is from new_card 2 to {action[2]}")
                time.sleep(.1)
            elif action[0] == 2:
                self.mouse.drag_and_drop(position_new0_X,position_new0_Y,target_positionX[action[1]] + card_pos[action[1]]*diff_X,target_positionY[action[1]])
                card_pos[action[1]] += 1
                
                #print(f"the action is from new_card 0 to {action[1]}")
                time.sleep(.1)
                self.mouse.drag_and_drop(position_new1_X,position_new1_Y,target_positionX[action[2]] + card_pos[action[2]]*diff_X,target_positionY[action[2]])
                card_pos[action[2]] += 1
                #print(f"the action is from new_card 1 to {action[2]}")
                time.sleep(.1)

        confirmBtn_positionX = int( (self.table.layout["confirm_button"]["x"]+self.table.layout["confirm_button"]["width"]/2.0) * xScale + self.table.tlc[0])
        confirmBtn_positionY = int( (self.table.layout["confirm_button"]["y"]+self.table.layout["confirm_button"]["height"]/2.0) * yScale + self.table.tlc[1])
        self.mouse.mouse_clicker(confirmBtn_positionX,confirmBtn_positionY ,1,1,1)
        time.sleep(.5)
    def fantasy_act(self,fantasy_state):
        xScale = float(self.table.tlc[2])/table.table_width
        yScale = float(self.table.tlc[3])/table.table_height
        diff_X = int((self.table.me_cardSize[0]+5)*xScale)
        me_front_state = fantasy_state[0:3]
        me_middle_state = fantasy_state[3:8]
        me_back_state = fantasy_state[8:13]
        card_pos = [0,0,0]
        xScale = float(self.table.tlc[2])/table.table_width
        yScale = float(self.table.tlc[3])/table.table_height
        target_positionX = [0,0,0]
        target_positionY = [0,0,0]
        target_positionX[0] = int((self.table.layout["me"]["top1"]["x"] + self.table.me_cardSize[0]/2.0) * xScale + self.table.tlc[0])
        target_positionY[0] = int((self.table.layout["me"]["top1"]["y"] + self.table.me_cardSize[1]/2.0) * yScale + self.table.tlc[1])
        target_positionX[1] = int((self.table.layout["me"]["mid1"]["x"] + self.table.me_cardSize[0]/2.0) * xScale + self.table.tlc[0])
        target_positionY[1] = int((self.table.layout["me"]["mid1"]["y"] + self.table.me_cardSize[1]/2.0) * yScale + self.table.tlc[1])
        target_positionX[2] = int((self.table.layout["me"]["bottom1"]["x"] + self.table.me_cardSize[0]/2.0) * xScale + self.table.tlc[0])
        target_positionY[2] = int((self.table.layout["me"]["bottom1"]["y"] + self.table.me_cardSize[1]/2.0) * yScale + self.table.tlc[1])
        #top
        for i in range(3):
            top_card = me_front_state[i]
            position = self.table.get_fantasy_cards(top_card)
            while True:
                position = self.table.get_fantasy_cards(top_card)
                if len(position) == 2 and  position != [-1,-1]:
                    break
            position[0] += self.table.tlc[0]
            position[1] += self.table.tlc[1]
            self.mouse.drag_and_drop(position[0],position[1],target_positionX[0] + card_pos[0]*diff_X,target_positionY[0])
            card_pos[0]+=1
            time.sleep(1)
        for i in range(5):
            middle_card = me_middle_state[i]
            while True:
                position = self.table.get_fantasy_cards(middle_card)
                if len(position) == 2 and position != [-1,-1]:
                    break
            
            position[0] += self.table.tlc[0]
            position[1] += self.table.tlc[1]
            self.mouse.drag_and_drop(position[0],position[1],target_positionX[1] + card_pos[1]*diff_X,target_positionY[1])
            card_pos[1]+=1
            time.sleep(1)
        back_card = me_back_state[2]
        while True:
            position = self.table.get_fantasy_cards(back_card)
            if len(position) == 2 and position != [-1,-1]:
                break
        position[0] += self.table.tlc[0]
        position[1] += self.table.tlc[1]
        self.mouse.drag_and_drop(position[0],position[1],target_positionX[2] + 2*diff_X,target_positionY[2])
        time.sleep(max(65- (timer()-self.start_time),0)) 
        confirmBtn_positionX = int( (self.table.layout["fantasy_confirm_button"]["x"]+self.table.layout["fantasy_confirm_button"]["width"]/2.0) * xScale + self.table.tlc[0])
        confirmBtn_positionY = int( (self.table.layout["fantasy_confirm_button"]["y"]+self.table.layout["fantasy_confirm_button"]["height"]/2.0) * yScale + self.table.tlc[1])
        self.mouse.mouse_clicker(confirmBtn_positionX,confirmBtn_positionY ,1,1,1)
        time.sleep(1)

        
