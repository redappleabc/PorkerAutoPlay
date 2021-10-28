import numpy as np
import operator
import sys
from tools.evaluator import evaluate
from numba import njit, cuda, void, int32, int64, float32, float64,vectorize
card_dic = [
    "2S","3S","4S","5S","6S","7S","8S","9S","TS","JS","QS","KS","AS",
    "2H","3H","4H","5H","6H","7H","8H","9H","TH","JH","QH","KH","AH",
    "2D","3D","4D","5D","6D","7D","8D","9D","TD","JD","QD","KD","AD",
    "2C","3C","4C","5C","6C","7C","8C","9C","TC","JC","QC","KC","AC"
]
cards = list(card_dic)

def get_remain_card(me_current_state,enemy_current_state,new_card):
    #cards = np.arange(1,53)
    remain_card = np.setdiff1d(cards,me_current_state)
    remain_card = np.setdiff1d(remain_card,enemy_current_state)
    remain_card = np.setdiff1d(remain_card,new_card)
    return remain_card
'''
# for cuda
@njit
def get_remain_card(me_current_state,enemy_current_state,new_card):
    remain_card = [
        "2S","3S","4S","5S","6S","7S","8S","9S","TS","JS","QS","KS","AS",
        "2H","3H","4H","5H","6H","7H","8H","9H","TH","JH","QH","KH","AH",
        "2D","3D","4D","5D","6D","7D","8D","9D","TD","JD","QD","KD","AD",
        "2C","3C","4C","5C","6C","7C","8C","9C","TC","JC","QC","KC","AC"
    ]
    for item in list(me_current_state + enemy_current_state + new_card):
        remain_card.remove(item)
    return np.array(remain_card)
'''

# make randomly last state
#@jit
def make_random_final_state(current_state,remain_card):
    np.random.shuffle(remain_card)
    result_state = np.copy(current_state)
    np.place(result_state,current_state=="NC",remain_card)
    return result_state

def make_random_fantasy_state(number):
    shuffle_cards = np.array(cards)
    np.random.shuffle(shuffle_cards)
    return shuffle_cards[0:number]

#make random next state
#@jit
def make_next_state(current_state,new_card,disposition):
    front = current_state[0:3]
    middle = current_state[3:8]
    back = current_state[8:13]
    for key,value in enumerate(disposition):
        if value == 0:
            if np.where(front=="NC")[0].size == 0:
                return np.array([])
            front[np.where(front == "NC")[0][0]] = new_card[key]
            
        elif value == 1:
            if np.where(middle=="NC")[0].size == 0:
                return np.array([])
            middle[np.where(middle=="NC")[0][0]] = new_card[key]
        elif value == 2:
            if  np.where(back == "NC")[0].size ==0:
                return np.array([])
            back[np.where(back == "NC")[0][0]] = new_card[key]
    return np.concatenate((front,middle,back))

def calc_monte_carlo_reward(me_final_state,enemy_final_state):
    pass
#@jit
def calc_best_next_state(me_current_state,enemy_current_state,new_card,step,loop_cnt=100):
    remain_card = get_remain_card(me_current_state,enemy_current_state,new_card)
    expectation_set = {}
    state_set = {}
    if step == 0:#new card size =5
        for idx1 in np.arange(3):
            for idx2 in np.arange(3):
                for idx3 in np.arange(3):
                    for idx4 in np.arange(3):
                        for idx5 in np.arange(3):
                            idx = np.array([idx1,idx2,idx3,idx4,idx5])
                            if np.count_nonzero(idx)<3:#i.e. 0,0,0,0,2
                                continue
                            else:
                                str_index = "_".join(str(e) for e in idx)
                                expectation_set[str_index] = 0
                                tmp_me_current_state = np.copy(me_current_state)
                                me_next_state =make_next_state(tmp_me_current_state,new_card,idx)
                                if me_next_state.size == 0:
                                    continue
                                tmp_me_next_state = np.copy(me_next_state)
                                tmp_enemy_current_state = np.copy(enemy_current_state)
                                state_set [str_index] = me_next_state  
                                for loop in np.arange(loop_cnt):
                                    enemy_random_final_state = make_random_final_state(tmp_enemy_current_state,remain_card)
                                    me_random_final_state = make_random_final_state(tmp_me_next_state,get_remain_card(enemy_random_final_state,tmp_me_next_state,new_card))
                                    evaluate_value = evaluate(me_random_final_state,enemy_random_final_state)
                                    #if(evaluate_value > expectation_set[str_index]):
                                    expectation_set[str_index] += evaluate_value 
    if step > 0:
        for i in np.arange(3):
            tmp_new_card = np.copy(new_card)
            tmp_new_card = np.delete(tmp_new_card,i)
            for idx1 in np.arange(3):
                for idx2 in np.arange(3):
                    str_idx = np.array([i,idx1,idx2])
                    idx = np.array([idx1,idx2])
                    str_index = "_".join(str(e) for e in str_idx)
                    expectation_set[str_index] = 0
                    tmp_me_current_state = np.copy(me_current_state)
                    me_next_state =make_next_state(tmp_me_current_state,tmp_new_card,idx)
                    if me_next_state.size ==0:
                        expectation_set[str_index] = -1000000000000
                        continue
                    tmp_me_next_state = np.copy(me_next_state)
                    tmp_enemy_current_state = np.copy(enemy_current_state)
                    state_set [str_index] = me_next_state
                       
                    for loop in np.arange(step*loop_cnt):  
                        enemy_random_final_state = make_random_final_state(tmp_enemy_current_state,remain_card)   
                        me_random_final_state = make_random_final_state(tmp_me_next_state,get_remain_card(enemy_random_final_state,tmp_me_next_state,new_card))
                        evaluate_value = evaluate(me_random_final_state,enemy_random_final_state)
                        
                        expectation_set[str_index] += evaluate_value
    arg_max = max(expectation_set.items(),key = operator.itemgetter(1))[0]
    return state_set[arg_max] , arg_max,expectation_set[arg_max]








        



