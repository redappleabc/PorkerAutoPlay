import sys
sys.path.append('E:\MyWork\PineappleOFCP_progressive') 
import numpy as np  
from decision_maker.monte_carlo import make_random_fantasy_state
from decision_maker.fantasy import calc_best_fantasy_state
from tools.evaluator import evaluate
if __name__ == '__main__':
    '''total_reward = 0
    cnt = 0
    for i in range(10000):
        fantasy_state = make_random_fantasy_state(16)
        #print(fantasy_state)
        max_reward = calc_fantasy_max_reward(fantasy_state)
        total_reward += max_reward
        cnt += 1
        #print(max_reward)
    print("when card size is 16 total is", total_reward)
    print("averge 16 is ", total_reward/cnt)
    print("cnt is", cnt)
    '''
    hand = ['AS', 'KH' ,'JH','TD', '8D', '6D', '5H', '5C', '4H' ,'4C', '4D', '3H', '2S' ,'2D']
    for i in range(1000):
     print(calc_best_fantasy_state(hand))
   
   