import numpy as np
from numba import jit, cuda
FRONT = {
    "66":1,
    "77":2,
    "88":3,
    "99":4,
    "1010":5,#TT
    "1111":6,#JJ
    "1212":7,#QQ
    "1313":8,#KK
    "1414":9,#AA
    "222":10,
    "333":11,
    "444":12,
    "555":13,
    "666":14,
    "777":15,
    "888":16,
    "999":17,
    "101010":18,#TTT
    "111111":19,#JJJ
    "121212":20,#QQQ
    "131313":21,#KKK
    "141414":22#AAA
}
MIDDLE = {
    "Three of a kind":2,
    "Straight":4,
    "Flush":8,
    "Full House":12,
    "Four of a kind":20,
    "Straight Flush":30,
    "Royal Flush":50
}
BACK = {
    "Straight":2,
    "Flush":4,
    "Full House":6,
    "Four of a kind":10,
    "Straight Flush":15,
    "Royal Flush":25
}

#compare two final state and return good state
def compare_state(state_1,state_2):
    front_hand_1 = np.copy(state_1[0:3])
    middle_hand_1 = np.copy(state_1[3:8])
    back_hand_1 = np.copy(state_1[8:13])

    front_hand_2 = np.copy(state_2[0:3])
    middle_hand_2 = np.copy(state_2[3:8])
    back_hand_2 = np.copy(state_2[8:13])

    front_winner = compare_hands([front_hand_1,front_hand_2])
    middle_winner = compare_hands([middle_hand_1,middle_hand_2])
    back_winner = compare_hands([back_hand_1,back_hand_2])
    first = 0
    if front_winner == "second":
        first -= 1
    elif front_winner == "first":
        first += 1
    if middle_winner == "second":
        first -= 1
    elif middle_winner == "first":
        first += 1
    if back_winner == "second":
        first -= 1
    elif back_winner == "first":
        first += 1
    if first >= 0:
        return state_1
    elif first < 0:
        return state_2


def compare_hands(hands,position = None):
    scores = [(i, get_score(hand)) for i, hand in enumerate(hands)]
    winner = sorted(scores , key=lambda x:x[1])[-1][0]
    if scores[0][1] == scores[1][1]:
        return "same"
    elif winner == 0:
        return "first"
    else:
        return "second"

#@cuda.jit
def compare_hands_and_get_royalty(hands,position = None):
    scores = [(i, get_score(hand)) for i, hand in enumerate(hands)]
    winner = sorted(scores , key=lambda x:x[1])[-1][0]

    first = scores[0][1]
    second = scores[1][1]
    first_score = 0
    second_score = 0
    if position is None:
        if scores[0][1] == scores[1][1]:
            return False
        return winner == 1
    elif position == "front":
        if first[0][0] == 3:
            first_score = FRONT[str(first[1][0]+2)*3]
        elif first[0][0] == 2 and first[1][0] + 2 >= 6 :
            first_score = FRONT[str(first[1][0]+2)*2]
        if second[0][0] == 3:
            second_score = FRONT[str(second[1][0]+2)*3]
        elif second[0][0] == 2 and second[1][0] + 2 >= 6 :
            second_score = FRONT[str(second[1][0]+2)*2]
        
    elif position == "middle":
        if first[0][0] == 3:
            first_score = MIDDLE["Three of a kind"]
        elif np.sum(first[0]) == 6:
            first_score =  MIDDLE["Straight"]
        elif np.sum(first[0]) == 7:
            first_score =  MIDDLE["Flush"]
        elif first[0][0] == 3 and first[0][1] == 2:
            first_score =  MIDDLE["Full House"]
        elif first[0][0] == 4:
            first_score = MIDDLE["Four of a kind"]
        elif first[0][0] == 5 and first[1][0] == 12:
            first_score = MIDDLE["Royal Flush"]
        elif first[0][0] == 5:
            first_score = MIDDLE["Straight Flush"]

        if second[0][0] == 3:
            second_score = MIDDLE["Three of a kind"]
        elif np.sum(second[0]) == 6:
            second_score =  MIDDLE["Straight"]
        elif np.sum(second[0]) == 7:
            second_score =  MIDDLE["Flush"]
        elif second[0][0] == 3 and second[0][1] == 2:
            second_score =  MIDDLE["Full House"]
        elif second[0][0] == 4:
            second_score = MIDDLE["Four of a kind"]
        elif second[0][0] == 5 and second[1][0] == 12:#9 is T
            second_score = MIDDLE["Royal Flush"]
        elif second[0][0] == 5:
            second_score = MIDDLE["Straight Flush"]

    elif position == "back":
        if np.sum(first[0]) == 6:
            first_score =  BACK["Straight"]
        elif np.sum(first[0]) == 7:
            first_score =  BACK["Flush"]
        elif first[0][0] == 3 and first[0][1] == 2:
            first_score =  BACK["Full House"]
        elif first[0][0] == 4:
            first_score = BACK["Four of a kind"]
        elif first[0][0] == 5 and first[1][0] == 12:#9 is T
            first_score = BACK["Royal Flush"]
        elif first[0][0] == 5:
            first_score = BACK["Straight Flush"]

        if np.sum(second[0]) == 6:
            second_score =  BACK["Straight"]
        elif np.sum(second[0]) == 7:
            second_score =  BACK["Flush"]
        elif second[0][0] == 3 and second[0][1] == 2:
            second_score =  BACK["Full House"]
        elif second[0][0] == 4:
            second_score = BACK["Four of a kind"]
        elif second[0][0] == 5 and second[1][0] == 12:
            second_score = BACK["Royal Flush"]
        elif second[0][0] == 5:
            second_score = BACK["Straight Flush"]
    return first_score,second_score
#@cuda.jit
def get_score(hand):
    ranks = '23456789TJQKA'
    rcounts = {ranks.find(r): ''.join(hand).count(r) for r, _ in hand}.items()
    score, ranks = zip(*sorted((cnt, rank) for rank, cnt in rcounts)[::-1])
    if len(score) == 5:
        if ranks[0:2] == (12, 3): #adjust if 5 high straight
            ranks = (3, 2, 1, 0, -1)
        straight = ranks[0] - ranks[4] == 4
        flush = len({suit for _, suit in hand}) == 1
        '''no pair, straight, flush, or straight flush'''
        score = ([(1,), (3,1,1,1)], [(3,1,1,2), (5,)])[flush][straight]
    return score, ranks
#@cuda.jit
def evaluate(me_state,enemy_state):
    me_front_hand = np.copy(me_state[0:3])
    me_middle_hand = np.copy(me_state[3:8])
    me_back_hand = np.copy(me_state[8:13])

    enemy_front_hand = np.copy(enemy_state[0:3])
    enemy_middle_hand = np.copy(enemy_state[3:8])
    enemy_back_hand = np.copy(enemy_state[8:13])

    me_penalty = compare_hands_and_get_royalty([me_middle_hand,me_front_hand]) or compare_hands_and_get_royalty([me_back_hand,me_middle_hand])
    enemy_penalty = compare_hands_and_get_royalty([enemy_middle_hand,enemy_front_hand]) or compare_hands_and_get_royalty([enemy_back_hand,enemy_middle_hand])

    
    me_front_score,enemy_front_score = compare_hands_and_get_royalty([me_front_hand,enemy_front_hand],"front")
    me_middle_score,enemy_middle_score = compare_hands_and_get_royalty([me_middle_hand,enemy_middle_hand],"middle")
    me_back_score,enemy_back_score = compare_hands_and_get_royalty([me_back_hand,enemy_back_hand],"back")

    me_score = me_front_score+me_middle_score+me_back_score
    enemy_score = enemy_front_score+enemy_middle_score+enemy_back_score
    if me_penalty and enemy_penalty:
        return 0#-enemy_score

    if me_penalty and not enemy_penalty:
        return 0#(6+enemy_score)
    elif not me_penalty and enemy_penalty:
        return 6+me_score
    
    diff = 0
    if me_front_score > enemy_front_score:
        diff += 1
    elif  me_front_score < enemy_front_score:
        diff -=1
    if me_middle_score > enemy_middle_score:
        diff += 1
    if me_middle_score < enemy_middle_score:
        diff -= 1
    if me_back_score > enemy_back_score:
        diff += 1
    if me_back_score < enemy_back_score:
        diff -= 1
    if diff == 3 or diff == -3:
        diff = diff*2
    return me_score + diff - enemy_score





#############   for fantasy ###################################
def calc_reward_of_position(hand,position = None):
    scores =get_score(hand)
    position_score = 0
    if position == "front":
        if scores[0][0] == 3:
            position_score = FRONT[str(scores[1][0]+2)*3]
        elif scores[0][0] == 2 and scores[1][0] + 2 >= 6 :
            position_score = FRONT[str(scores[1][0]+2)*2]
    
    elif position == "middle":
        if scores[0][0] == 3:
            position_score = MIDDLE["Three of a kind"]
        elif np.sum(scores[0]) == 6:
            position_score =  MIDDLE["Straight"]
        elif np.sum(scores[0]) == 7:
            position_score =  MIDDLE["Flush"]
        elif scores[0][0] == 3 and scores[0][1] == 2:
            position_score =  MIDDLE["Full House"]
        elif scores[0][0] == 4:
            position_score = MIDDLE["Four of a kind"]
        elif scores[0][0] == 5 and scores[1][0] == 12:
            position_score = MIDDLE["Royal Flush"]
        elif scores[0][0] == 5:
            position_score = MIDDLE["Straight Flush"]


    elif position == "back":
        if np.sum(scores[0]) == 6:
            position_score =  BACK["Straight"]
        elif np.sum(scores[0]) == 7:
            position_score =  BACK["Flush"]
        elif scores[0][0] == 3 and scores[0][1] == 2:
            position_score =  BACK["Full House"]
        elif scores[0][0] == 4:
            position_score = BACK["Four of a kind"]
        elif scores[0][0] == 5 and scores[1][0] == 12:#9 is T
            position_score = BACK["Royal Flush"]
        elif scores[0][0] == 5:
            position_score = BACK["Straight Flush"]
       
    return position_score


def get_penalty(hands):
    scores = [(i, get_score(hand)) for i, hand in enumerate(hands)]
    winner = sorted(scores , key=lambda x:x[1])[-1][0]
    if len(hands[0]) > len(hands[1]):
        if scores[0][1][0][0] == 3 and scores[0][1][0][1] == 2:
            return False
        elif scores[0][1][0][0] == 3 and scores[0][1][0][1] == 1 and scores[0][1][0][2] == 1 and scores[1][1][0][0] == 3:
            return scores[1][1][1][0] > scores[0][1][1][0]
        if scores[1][1][0][0] == 3 and  scores[0][1][0][0] == 2 and scores[0][1][0][1] == 2:
            return True
        elif scores[1][1][0][0] != 3 and  scores[0][1][0][0] == 2 and scores[0][1][0][1] == 2:
            return False
        elif scores[0][1][0][0] == 2 and scores[0][1][0][1] == 1 and scores[0][1][0][2] == 1:
            if scores[1][1][1][0] > scores[0][1][1][0]:
                return  True
            elif scores[1][1][1][0] == scores[0][1][1][0] and scores[1][1][1][1] > scores[0][1][1][1]:
                return True
            elif scores[1][1][1][0] == scores[0][1][1][0] and scores[1][1][1][1] == scores[0][1][1][1]:
                return False
        elif scores[0][1][0][0] == 1:
            if  scores[1][1][1][0] > scores[0][1][1][0]:
                return  True
            elif scores[1][1][1][0] == scores[0][1][1][0] and scores[1][1][1][1] > scores[0][1][1][1]:
                return True
            elif scores[1][1][1][0] == scores[0][1][1][0] and scores[1][1][1][1] == scores[0][1][1][1] and scores[1][1][1][2] > scores[0][1][1][2]:
                return True
            elif scores[1][1][1][0] == scores[0][1][1][0] and scores[1][1][1][1] == scores[0][1][1][1] and scores[1][1][1][1] == scores[0][1][1][1]:
                return False
        


    if scores[0][1] == scores[1][1]:
        return False
    return winner == 1


def evaluate_me_state(me_state,enemy_state = None):
    me_front_hand = np.copy(me_state[0:3])
    me_middle_hand = np.copy(me_state[3:8])
    me_back_hand = np.copy(me_state[8:13])
    me_penalty = get_penalty([me_middle_hand,me_front_hand]) or get_penalty([me_back_hand,me_middle_hand])
    if me_penalty:
        return -100
    me_front_score = calc_reward_of_position(me_front_hand,"front")
    me_middle_score = calc_reward_of_position(me_middle_hand,"middle")
    me_back_score = calc_reward_of_position(me_back_hand,"back")
    if me_front_score >= 10:
        me_front_score += 12
    if me_middle_score >= 20:
        me_middle_score += 12
    if me_back_score >= 10:
        me_back_score += 12

    me_score = me_front_score+me_middle_score+me_back_score
    return me_score
   

#print(compare_hands_and_get_royalty([["KS","KD"],["6S","6D"]]))


