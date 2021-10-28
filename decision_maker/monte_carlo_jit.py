
from numba import jit,njit,config
import numba
import numpy as np
from numba.typed import List,Dict
from numba.core import types
import math
import random

# set the threading layer before any parallel target compilation
config.THREADING_LAYER = 'threadsafe'

@njit
def make_idx(x,unit):
    if x >= 10 and unit == 3:
        result = x*math.pow(10,4)+x*math.pow(10,2)+x
    elif x >= 10 and unit == 2:
        result = x*math.pow(10,2)+x
    elif unit ==3:
        result = x*math.pow(10,2)+x*math.pow(10,1)+x
    elif unit ==2:
        result = x*10+x
    return int(result)
@njit
def compare_hands_and_get_royalty(hands,position = None):
    FRONT = Dict.empty(key_type=types.int64,value_type=types.int64)
    FRONT[66] = 1
    FRONT[77] = 2
    FRONT[88] = 3
    FRONT[99] = 4
    FRONT[1010] = 5
    FRONT[1111] = 6
    FRONT[1212] = 7
    FRONT[1313] = 8
    FRONT[1414] = 9
    FRONT[222] = 10
    FRONT[333] = 11
    FRONT[444] = 12
    FRONT[555] = 13
    FRONT[666] = 14
    FRONT[777] = 15
    FRONT[888] = 16
    FRONT[999] = 17
    FRONT[101010] = 18
    FRONT[111111] = 19
    FRONT[121212] = 20
    FRONT[131313] = 21
    FRONT[141414] = 22

    MIDDLE = Dict.empty(key_type=types.int64,value_type=types.int64)
    MIDDLE[0] = 2 
    MIDDLE[1] = 4
    MIDDLE[2] = 8
    MIDDLE[3] = 12
    MIDDLE[4] = 20
    MIDDLE[5] = 30
    MIDDLE[6] = 50

    BACK = Dict.empty(key_type=types.int64,value_type=types.int64)
    BACK[0] = 2
    BACK[1] = 4
    BACK[2] = 6
    BACK[3] = 10
    BACK[4] = 15
    BACK[5] = 25
    scores = [(i, get_score(hand)) for i, hand in enumerate(hands)]
    #winner = sorted(scores , key=lambda x:x[1])[-1][0]

    first = scores[0][1]
    second = scores[1][1]
    first_score = 0
    second_score = 0
    if position == "front":
        if first[0][0] == 3:
            first_score = FRONT[make_idx(first[1][0]+2,3)]
        elif first[0][0] == 2 and first[1][0] + 2 >= 6 :
            first_score = FRONT[make_idx(first[1][0]+2,2)]
        if second[0][0] == 3:
            second_score = FRONT[make_idx(second[1][0]+2,3)]
        elif second[0][0] == 2 and second[1][0] + 2 >= 6 :
            second_score = FRONT[make_idx(second[1][0]+2,2)]
        
    elif position == "middle":
        if first[0][0] == 3:
            first_score = MIDDLE[0]
        elif sum(first[0]) == 6:
            first_score =  MIDDLE[1]
        elif sum(first[0]) == 7:
            first_score =  MIDDLE[2]
        elif first[0][0] == 3 and first[0][1] == 2:
            first_score =  MIDDLE[3]
        elif first[0][0] == 4:
            first_score = MIDDLE[4]
        elif first[0][0] == 5 and first[1][0] == 12:
            first_score = MIDDLE[5]
        elif first[0][0] == 5:
            first_score = MIDDLE[6]

        if second[0][0] == 3:
            second_score = MIDDLE[0]
        elif sum(second[0]) == 6:
            second_score =  MIDDLE[1]
        elif sum(second[0]) == 7:
            second_score =  MIDDLE[2]
        elif second[0][0] == 3 and second[0][1] == 2:
            second_score =  MIDDLE[3]
        elif second[0][0] == 4:
            second_score = MIDDLE[4]
        elif second[0][0] == 5 and second[1][0] == 12:#9 is T
            second_score = MIDDLE[5]
        elif second[0][0] == 5:
            second_score = MIDDLE[6]

    elif position == "back":
        if sum(first[0]) == 6:
            first_score =  BACK[0]
        elif sum(first[0]) == 7:
            first_score =  BACK[1]
        elif first[0][0] == 3 and first[0][1] == 2:
            first_score =  BACK[2]
        elif first[0][0] == 4:
            first_score = BACK[3]
        elif first[0][0] == 5 and first[1][0] == 12:#9 is T
            first_score = BACK[4]
        elif first[0][0] == 5:
            first_score = BACK[5]

        if sum(second[0]) == 6:
            second_score =  BACK[0]
        elif sum(second[0]) == 7:
            second_score =  BACK[1]
        elif second[0][0] == 3 and second[0][1] == 2:
            second_score =  BACK[2]
        elif second[0][0] == 4:
            second_score = BACK[3]
        elif second[0][0] == 5 and second[1][0] == 12:
            second_score = BACK[4]
        elif second[0][0] == 5:
            second_score = BACK[5]
    return first_score,second_score

@njit
def additional_score(hand):
    pair = 1
    two_pair = 13
    three_kind = 40
    scores =  get_score(hand)
    position_score = 0
    if scores[0][0] == 2:
        position_score = (scores[1][0] + 1) * pair
    if scores[0][0] == 2 and scores[0][1] == 2:
        position_score = two_pair + (scores[1][0] + 1) * pair + (scores[1][1] + 1) * pair
    if scores[0][0] == 3:
        position_score =  three_kind  + (scores[1][0] + 1) * pair
    return position_score



@njit
def get_score(hand):
    rcounts = Dict.empty(key_type=types.int64,value_type=types.int64)
    ranks = '23456789TJQKA'
    for r, _ in hand:
        rcounts[ranks.find(r)] = ''.join(hand).count(r)
    score_rank = sorted([(rcounts[key], key) for key in rcounts.keys()])[::-1]
    score_rank = [[score for score,rank in score_rank],[rank for score,rank in score_rank]]
    score = score_rank[0]
    ranks = score_rank[1]
    if len(score) == 5:
        if ranks[0:2] == [12, 3]: #adjust if 5 high straight
            ranks = [3, 2, 1, 0, -1]
        straight = ranks[0] - ranks[4] == 4
        suits = Dict.empty(key_type=types.int64,value_type=types.int64)
        for _,suit in hand:
            suits[ord(suit)] = ord(suit)
        flush = len(suits) == 1
        if flush and straight:
            score = [5,0,0,0,0]
        elif flush:
            score = [3,1,1,2,0]
        elif straight:
            score = [3,1,1,1,0]
        else:
            score = [1,1,1,1,1]
    if len(score) < 5:
        score += [0,]*(5-len(score))
    if len(ranks) < 5:
        ranks += [-1,]*(5-len(ranks))
    return score,ranks



@njit
def make_random_final_state(current_state,remain_card):
    size = len(remain_card)
    int_cards = np.arange(size)
    np.random.shuffle(int_cards)
    result_state = (current_state)
    cnt = 0
    for i,item in enumerate(result_state):
        if item == "NC":
            result_state[i] = remain_card[int_cards[cnt]]
            cnt+=1
    
    return list(result_state)
    
@njit
def make_next_state(current_state,new_card,disposition):
    current_state = list(current_state)
    front = current_state[0:3]
    middle = current_state[3:8]
    back = current_state[8:13]
    
    for key,value in enumerate(disposition):
        if value == 0:
            if "NC" in front:
                front[front.index("NC")] = new_card[key]
            else:
                return ["NC"]
        elif value == 1:
            if "NC" in middle:
                middle[middle.index("NC")] = new_card[key]
            else:
                return ["NC"]
        elif value == 2:
            if "NC" in back:
                back[back.index("NC")] = new_card[key]
            else:
                return ["NC"]
    return list(front+middle+back)


@njit
def is_penalty(hands):
    scores = [(i, get_score(hand)) for i, hand in enumerate(hands)]
    winner = sorted(scores , key=lambda x:x[1])[-1][0]
    if len(hands[0]) > len(hands[1]):
        if scores[0][1][0][0] == 3 and scores[0][1][0][1] == 2:
            return False
        elif scores[0][1][0][0] == 3 and scores[0][1][0][1] == 1 and scores[0][1][0][2] == 1 and scores[1][1][0][0] == 3:
            return scores[1][1][1][0] > scores[0][1][1][0]
        if scores[0][1][0][0] == 2 and scores[0][1][0][1] == 2:
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
@njit
def evaluate(me_state,enemy_state):
    me_state = list(me_state)
    enemy_state = list(enemy_state)
    me_front_hand =me_state[0:3]
    me_middle_hand =me_state[3:8]
    me_back_hand =me_state[8:13]
    enemy_front_hand =enemy_state[0:3]
    enemy_middle_hand =enemy_state[3:8]
    enemy_back_hand =enemy_state[8:13]
    me_penalty = is_penalty([me_middle_hand,me_front_hand]) or is_penalty([me_back_hand,me_middle_hand])
    enemy_penalty = is_penalty([enemy_middle_hand,enemy_front_hand]) or is_penalty([enemy_back_hand,enemy_middle_hand])

    me_front_score,enemy_front_score = compare_hands_and_get_royalty([me_front_hand,enemy_front_hand],"front")
    me_middle_score,enemy_middle_score = compare_hands_and_get_royalty([me_middle_hand,enemy_middle_hand],"middle")
    me_back_score,enemy_back_score = compare_hands_and_get_royalty([me_back_hand,enemy_back_hand],"back")

    if me_penalty: 
        return -9999999#0,0,0,0,0,0
    table_score = 0
    if me_front_score > enemy_front_score:
        table_score += 1
    elif  me_front_score < enemy_front_score:
        table_score -= 1
    if me_middle_score > enemy_middle_score:
        table_score += 1
    if me_middle_score < enemy_middle_score:
        table_score -= 1
    if me_back_score > enemy_back_score:
       table_score += 1
    if me_back_score < enemy_back_score:
        table_score -= 1
    if table_score ==  3 or  table_score == -3:
        table_score *= 2
    if me_front_score == 7:
        me_front_score += 12
    if me_front_score == 8:
        me_front_score += 15
    if me_front_score >= 9:
        me_front_score += 19
    return me_front_score  + me_middle_score  + me_back_score + table_score
@jit
def get_remain_card(new_card):
    remain_card = [
        "2S","3S","4S","5S","6S","7S","8S","9S","TS","JS","QS","KS","AS",
        "2H","3H","4H","5H","6H","7H","8H","9H","TH","JH","QH","KH","AH",
        "2D","3D","4D","5D","6D","7D","8D","9D","TD","JD","QD","KD","AD",
        "2C","3C","4C","5C","6C","7C","8C","9C","TC","JC","QC","KC","AC"
    ]
   
    for item in new_card:
        if item != "NC" and item in remain_card:
            remain_card.remove(item)
    return remain_card
@njit
def get_hand(hand):
    if len(hand) == 0:
        return "high_card"
    if len(hand) == 1:
        return "high_card"
    hand_len = len(hand)
    rcounts = Dict.empty(key_type=types.int64,value_type=types.int64)
    ranks = '23456789TJQKA'
    for r, _ in hand:
        rcounts[ranks.find(r)] = ''.join(hand).count(r)
    score_rank = sorted([(rcounts[key], key) for key in rcounts.keys()])[::-1]
    score_rank = [[score for score,rank in score_rank],[rank for score,rank in score_rank]]
    score = score_rank[0]
    hand_ranks = score_rank[1]
    if hand_len >= 4 and  score[0] == 4:
        return "four_kind"
    if hand_len >= 4 and  score[0] == 2 and score[1] == 2:
        return "two_pair"
    if hand_len == 5 and score[0] == 3 and score[1] == 2:
        return "full_house"
    if hand_len >= 2:
        if score[0] == 2:
            return "one_pair"
        if score[0] == 3:
            return "three_kind"
    flush = True
    if hand_len > 1:
        _suit = hand[0][1]
        for _, suit in hand:
            if _suit != suit:
                flush = False
                break
    if len(hand_ranks) > 1 and hand_ranks[0] - hand_ranks[hand_len-1] < 5:  #straight
        if flush:
            return "flush_straight"
        return "straight"
    if len(hand_ranks) > 1 and hand_ranks[0] == 12 and hand_ranks[1] <= 3:
        if flush:
            return "flush_straight"
        return "straight"
    if flush:
        return "flush"
    return "high_card"


@njit
def can_advance_with_new_card(position_state,new_card,position = "no_front"):
    if len(position_state) < 5:
        position_hand = get_hand(position_state)
        prediction_hand = get_hand(position_state+[new_card])
        if position == "front":
            if position_hand != "one_pair" and prediction_hand == "one_pair":
                return True
            if position_hand != "three_kind" and prediction_hand == "three_kind":
                return True
        else:
            if prediction_hand  in ["flush_straight","flush","straight","full_house"]:
                return True
            if position_hand != "four_kind" and prediction_hand == "four_kind":
                return True
            if position_hand != "three_kind" and prediction_hand == "three_kind":
                return True
            '''
            if position_hand == "high_card" and prediction_hand == "two_pair":
                return True
            '''
            '''
            if len(position_state) == 4 or len(position_state) == 3:
                if position_hand != "one_pair"  and prediction_hand == "one_pair":
                    return True
            '''
    return False
@njit
def is_advanced(current_hand,next_hand):
    royalty_order = ["high_card","one_pair","two_pair","three_kind","straight","flush","full_house","four_kind","flush_straight"]
    if next_hand in ["straight","flush","full_house","flush_straight"]:
        return True
    if current_hand in ["straight","flush","flush_straight"] and next_hand in ["one_pair","two_pair","three_kind","four_kind"]:
        return True
    return royalty_order.index(next_hand) > royalty_order.index(current_hand)
@njit
def is_regressed(current_hand,next_hand):
    royalty_order = ["high_card","one_pair","two_pair","three_kind","straight","flush","full_house","four_kind","flush_straight"]
    if next_hand in ["straight","flush","full_house","flush_straight"]:
        return False
    if current_hand in ["straight","flush","flush_straight"] and next_hand in ["one_pair","two_pair","three_kind","four_kind"]:
        return False
    return royalty_order.index(next_hand) < royalty_order.index(current_hand)
@njit
def pre_check_3(current_state,next_state):
    current_state = list(current_state)
    next_state = list(next_state)
    current_front = current_state[0:3]
    current_middle = current_state[3:8]
    current_back = current_state[8:13]
    if "NC" in current_front:
        current_front = current_front[0:current_front.index("NC")]
    if "NC" in current_middle:
        current_middle = current_middle[0:current_middle.index("NC")]
    if "NC" in current_back:
        current_back = current_back[0:current_back.index("NC")]

    next_front = next_state[0:3]
    next_middle = next_state[3:8]
    next_back = next_state[8:13]
    if "NC" in next_front:
        next_front = next_front[0:next_front.index("NC")]
    if "NC" in next_middle:
        next_middle = next_middle[0:next_middle.index("NC")]
    if "NC" in next_back:
        next_back = next_back[0:next_back.index("NC")]


    current_front_hand = get_hand(current_front)
    current_middle_hand = get_hand(current_middle)
    current_back_hand = get_hand(current_back)

    next_front_hand = get_hand(next_front)
    next_middle_hand = get_hand(next_middle)
    next_back_hand = get_hand(next_back)
    
    #condition if front new card disturb middle or back
    if len(next_front) - len(current_front) == 2:
        if current_front_hand == "high_card" and len(current_front) == 0 and next_front_hand == "one_pair":
            return True
        elif current_front_hand in ["high_card","flush","flush_straight","straight"] and next_front_hand in ["high_card","flush","flush_straight","straight"]:
            new_card = next_front[len(next_front)-2]
            if can_advance_with_new_card(current_middle,new_card):
                return False
            if can_advance_with_new_card(current_back,new_card):
                return False
            new_card = next_front[len(next_front)-1]
            if can_advance_with_new_card(current_middle,new_card):
                return False
            if can_advance_with_new_card(current_back,new_card):
                return False
        elif current_front_hand == "high_card" and len(current_front_hand) == 1 and next_front_hand == "one_pair":
            if next_front_hand[0] == next_front_hand[1]:
                new_card = next_front_hand[2]
            if  next_front_hand[0] == next_front_hand[2]:
                new_card = next_front_hand[1]
            if can_advance_with_new_card(current_middle,new_card):
                return False
            if can_advance_with_new_card(current_back,new_card):
                return False   
        elif current_front_hand == "high_card" and next_front_hand  in ["three_kind"]:
            return True
        
        #return True

    elif len(next_front) - len(current_front) == 1:
        if current_front_hand == "high_card" and next_front_hand  == "one_pair":
            return True
        elif current_front_hand in ["high_card","flush","flush_straight","straight"] and next_front_hand in ["high_card","flush","flush_straight","straight"]:
            new_card = next_front[len(next_front)-1]
            if can_advance_with_new_card(current_middle,new_card):
                return False
            if can_advance_with_new_card(current_back,new_card):
                return False
        #return True
    if len(next_middle) - len(current_middle) == 2:
        if len(current_back) == 5 and len(current_front) == 3:
            return True
        if is_regressed(current_middle_hand,next_middle_hand):
            new_card1 = next_middle[len(next_middle)-1]
            if can_advance_with_new_card(current_back,new_card1):
                return False
            if current_front_hand != "one_pair" and can_advance_with_new_card(current_front,new_card1,"front"):
                return False
            new_card2 = next_middle[len(next_middle)-2]
            if can_advance_with_new_card(current_back,new_card2):
                return False
            if current_front_hand != "one_pair" and can_advance_with_new_card(current_front,new_card2,"front"):
                return False
            #if with onley one card  can advance current middle return false
            if can_advance_with_new_card(current_middle,new_card1) or can_advance_with_new_card(current_middle,new_card2):
                return False

        if next_middle_hand in  ["straight","flush","full_house","flush_straight"]:
            return True
        new_card_1 = next_middle[len(next_middle)-1]
        new_card_2 = next_middle[len(next_middle)-2]
        if current_middle_hand in ["high_card","straight","flush","flush_straight"] and next_middle_hand in ["three_kind","two_pair"]:
            return True
        if current_middle_hand == "one_pair" and next_middle_hand in ["two_pair","four_kind"]:
            return True
        if current_middle_hand == "high_card" and len(current_middle) == 0 and next_middle_hand == "one_pair":
            return True
        else:
            if can_advance_with_new_card(current_back,new_card_1):
                return False
            if current_front_hand != "one_pair" and can_advance_with_new_card(current_front,new_card_1,"front"):
                return False
            if can_advance_with_new_card(current_back,new_card_2):
                return False
            if current_front_hand != "one_pair" and can_advance_with_new_card(current_front,new_card_2,"front"):
                return False
        #return True
    elif len(next_middle) - len(current_middle)  == 1:
        if is_advanced(current_middle_hand,next_middle_hand):
                return True
        else:
            new_card = next_middle[len(next_middle)-1]
            if can_advance_with_new_card(current_back,new_card):
                return False
            if current_front_hand != "one_pair" and can_advance_with_new_card(current_front,new_card,"front"):
                return False
        if is_regressed(current_middle_hand,next_middle_hand):
            new_card = next_middle[len(next_middle)-1]
            if can_advance_with_new_card(current_back,new_card):
                return False
            if current_front_hand != "one_pair" and can_advance_with_new_card(current_front,new_card,"front"):
                return False
       
        #return True



    if len(next_back) - len(current_back) == 2:
        if len(current_middle) == 5 and len(current_front) == 3:
            return True
        if is_regressed(current_back_hand,next_back_hand):
            new_card1 = next_back[len(next_back)-1]
            if can_advance_with_new_card(current_middle,new_card1):
                return False
            if current_front_hand not in  ["one_pair","three_kind"] and can_advance_with_new_card(current_front,new_card1,"front"):
                return False
            new_card2 = next_back[len(next_back)-2]
            if can_advance_with_new_card(current_middle,new_card2):
                return False
            if current_front_hand not in ["one_pair","three_kind"] and can_advance_with_new_card(current_front,new_card2,"front"):
                return False
            #if with onley one card  can advance current back return false
            if can_advance_with_new_card(current_back,new_card1) or can_advance_with_new_card(current_back,new_card2):
                return False
        if next_back_hand in  ["straight","flush","full_house","flush_straight"]:
            return True
        new_card_1 = next_back[len(next_back)-1]
        new_card_2 = next_back[len(next_back)-2]
        if current_back_hand == ["high_card","straight","flush","flush_straight"] and next_back_hand in ["three_kind","two_pair"]:
            return True
        if current_back_hand == "one_pair" and next_back_hand in ["two_pair","four_kind"]:
            return True
        if current_back_hand == "high_card" and len(current_back) == 0 and next_back_hand == "one_pair":
            return True
        else:
            if can_advance_with_new_card(current_middle,new_card_1):
                return False
            if current_front_hand != "one_pair" and can_advance_with_new_card(current_front,new_card_1,"front"):
                return False
            if can_advance_with_new_card(current_middle,new_card_2):
                return False
            if current_front_hand != "one_pair" and can_advance_with_new_card(current_front,new_card_2,"front"):
                return False
        #return True
    elif  len(next_back) - len(current_back) == 1:
        if is_regressed(current_back_hand,next_back_hand):
            new_card = next_back[len(next_back)-1]
            if can_advance_with_new_card(current_middle,new_card):
                return False
            if current_front_hand != "one_pair" and can_advance_with_new_card(current_front,new_card,"front"):
                return False
        if is_advanced(current_back_hand,next_back_hand):
            return True
        else:
            new_card = next_back[len(next_back)-1]
            if can_advance_with_new_card(current_middle,new_card):
                return False
            if current_front_hand != "one_pair" and can_advance_with_new_card(current_front,new_card,"front"):
                return False
        #return True


    #penalty
    
    ranks = '23456789TJQKA'
    
    max_back_card = 0
    max_middle_card = 0
    max_front_card = 0
    for r, _ in next_back:
        if ranks.find(r) > max_back_card:
            max_back_card = ranks.find(r)
    for r, _ in next_middle:
        if ranks.find(r) > max_middle_card:
            max_middle_card = ranks.find(r)
    for r, _ in next_front:
        if ranks.find(r) > max_front_card:
            max_front_card = ranks.find(r)
    
   
    if len(current_front) < 2:
        if len(next_back) - len(current_back) == 1 and len(next_front) - len(current_front) == 1:
            if ranks.find(next_back[-1][0]) > ranks.find(next_front[-1][0]):
               if random.randint(1,10) < 10:
                   return False
        if len(next_middle) - len(current_middle) == 1 and len(next_front) - len(current_front) == 1:
            if ranks.find(next_middle[-1][0]) > ranks.find(next_front[-1][0]):
               if random.randint(1,10) < 10:
                   return False
    ###########must be consider
    '''
    if len(next_middle) > 0 and len(next_back) > 0:
        for hand in ["straight","flush","flush_straight","four_kind","two_pair"]:
            if next_back_hand == hand and next_middle_hand == hand:
                if max_middle_card > max_back_card and len(next_middle) >= len(next_back):
                    return False
    '''
    '''
    if len(next_middle) > 0 and len(next_back) > 0:
        for hand in ["four_kind","two_pair"]:
            if next_back_hand == hand and next_middle_hand == hand:
                 if is_penalty([next_back,next_middle]): # if middle is high than back
                    return False
    '''
    '''
    if len(next_middle) == 5 and len(next_back) == 5:
        for hand in ["one_pair","three_kind","high_card"]:
            if next_back_hand == hand and next_middle_hand == hand:
                if max_middle_card > max_back_card:
                    return False 
    '''
    ##################################################
    if len(next_front) == 3 and len(next_middle) == 5:
        if is_penalty([next_middle,next_front]): # if middle is high than back
            return False 
    # when back and middle is full if middle is high than back it is false
    if len(next_back) == 5 and len(next_middle) == 5:
        if is_penalty([next_back,next_middle]): # if middle is high than back
            return False
    ##########################

                
    return True
    
            
       
   
    






@njit
def pre_check_5(state):
    state = list(state)
    front = state[0:3]
    middle = state[3:8]
    back = state[8:13]
    if "NC" in front:
        front = front[0:front.index("NC")]
    if "NC" in middle:
        middle = middle[0:middle.index("NC")]
    if "NC" in back:
        back = back[0:back.index("NC")]
    
    middle_hand =  get_hand(middle)
    back_hand = get_hand(back)
    front_hand = get_hand(front)
    if len(middle) == 0:
        if back_hand not in ["flush_straight","flush","straight","full_house","two_pair","four_kind"]:
            return False
    if len(back) == 0:
        if middle_hand not in ["flush_straight","flush","straight","full_house","two_pair","four_kind"]:
            return False
    if len(middle) == 4:
        if middle_hand not in ["flush_straight","flush","straight","two_pair","four_kind","three_kind"]:
            return False
    if len(back) == 4:
        if back_hand not in ["flush_straight","flush","straight","two_pair","four_kind","three_kind"]:
            return False
    if len(front) > 1:
        if front not in ["one_pair"]:
            return False
    if len(middle) == 3:
        if middle_hand not in ["flush_straight","flush","straight","three_kind","one_pair"]:
            return False
    if len(back) == 3:
        if back_hand not in ["flush_straight","flush","straight","three_kind"]:
            return False
    
    if len(middle) == 2:
        if middle_hand not in ["flush_straight","flush","straight","one_pair"]:
            return False
    if len(back) == 2:
        if back_hand not in ["flush_straight","flush","straight"]:
            return False


    ranks = '23456789TJQKA'
    max_back_card = 0
    max_middle_card = 0
    max_front_card = 0
    for r, _ in back:
        if ranks.find(r) > max_back_card:
            max_back_card = ranks.find(r)
    for r, _ in middle:
        if ranks.find(r) > max_middle_card:
            max_middle_card = ranks.find(r)
    for r, _ in front:
        if ranks.find(r) > max_front_card:
            max_front_card = ranks.find(r)        
    if len(middle) > 0 and len(back) > 0:
        for hand in ["straight","flush","flush_straight","one_pair"]:
            if back_hand == hand and middle_hand == hand:
                if max_middle_card > max_back_card and len(middle) >= len(back):
                    return False
    if middle_hand in ["flush","flush_straight"] and back_hand == "straight":
        return False
    #back is start over from straight
    if back_hand == "straight" and max_back_card < 5:
        return False 
    if back_hand not in ["straight","flush","two_pair","three_kind","flush_straight","four_kind"]:
        return False
    
    if len(middle) == 1 and len(front) == 1:
        if max_middle_card > max_front_card:
            return False
    if len(middle) == 0 and len(front) == 1:
        if max_front_card < 10:
            return False
    if len(front) == 0 and len(middle) == 1:
        if max_middle_card > 10:
            return False
    
    


    H_cnt = 0
    S_cnt = 0
    D_cnt = 0
    C_cnt = 0
    for rank,suit in (front+middle+back):
        if suit == "H":
            H_cnt += 1
        elif suit == "S":
            S_cnt += 1
        elif suit == "D":
            D_cnt += 1
        elif suit == "C":
            C_cnt += 1
    disturb_flush = False
    if front_hand == "high_card" and len(front) > 0:
        for _,suit in front:
            if suit == "S":
                suits_cnt = S_cnt
            elif suit == "H":
                suits_cnt = H_cnt
            elif suit == "D":
                suits_cnt = D_cnt
            elif suit == "C":
                suits_cnt = C_cnt
            if len(middle) > 0:
                for _,c_suit in middle:
                    if suit == c_suit:
                        if middle_hand not in ["one_pair","two_pair","four_kind" ,"three_kind"] and  not(middle_hand == "straight" and len(middle) >= suits_cnt) and middle_hand != "flush_straight":   
                            disturb_flush = True
            if len(back) > 0:
                for _,c_suit in back:
                    if suit == c_suit:
                        if back_hand not in ["one_pair","two_pair","four_kind" ,"three_kind"] and not(back_hand == "straight" and len(back) >= suits_cnt) and back_hand != "flush_straight":
                            disturb_flush = True
    if middle_hand == "high_card" and len(middle) > 0:
        for _,suit in middle:
            if suit == "S":
                suits_cnt = S_cnt
            elif suit == "H":
                suits_cnt = H_cnt
            elif suit == "D":
                suits_cnt = D_cnt
            elif suit == "C":
                suits_cnt = C_cnt
            if len(back) > 0:
                for _,c_suit in (back):
                    if suit == c_suit:
                        if back_hand not in ["one_pair","two_pair","four_kind" ,"three_kind"] and not(back_hand == "straight" and len(back) > suits_cnt) and back_hand != "flush_straight":
                            disturb_flush = True
            if len(front) > 0:
                for _,c_suit in (front):
                    if suit == c_suit:
                        if front_hand not in ["one_pair","three_kind"]:
                            disturb_flush = True
    if back_hand == "high_card" and  len(back) > 0:
        for _,suit in back:
            if suit == "S":
                suits_cnt = S_cnt
            elif suit == "H":
                suits_cnt = H_cnt
            elif suit == "D":
                suits_cnt = D_cnt
            elif suit == "C":
                suits_cnt = C_cnt
            if len(front) > 0:
                for _,c_suit in (front):
                    if suit == c_suit:
                     if front_hand not in ["one_pair","three_kind"]:
                            disturb_flush = True
            if len(middle) > 0:
                for _,c_suit in middle:
                    if suit == c_suit:
                        if middle_hand not in ["one_pair","two_pair","four_kind" ,"three_kind"] and  not(middle_hand == "straight" and len(middle) > suits_cnt) and middle_hand != "flush_straight":   
                            disturb_flush = True       
    if disturb_flush:
        return False


    #penalty
    if back_hand== "one_pair" and middle_hand == "three_kind":
        return False
    
    if back_hand == "high_card":
        return False

    
#######################################
    '''
    if len(front) == 1 and len(middle) == 1:
        #if one pair is separated 
        if front[0][0] == middle[0][0]:
            return False
        virtual_hand = get_hand(front+middle)
        if virtual_hand in ["straight","flush","flush_straight"] and  back_hand not in ["straight","flush_straight"]:
            return False 
        if virtual_hand == "flush" and back_hand == "flush" and max(max_front_card,max_middle_card) < max_back_card:
            return False 
    '''
##############################################
    return True
    



                      
@njit(parallel = True)
def calc_best_next_state_using_jit(me_current_state,enemy_current_state,new_card,thrown_card,step,loop_cnt=0):
    me_current_state = list(me_current_state)
    enemy_current_state = list(enemy_current_state)
    new_card = list(new_card)
    thrown_card = list(thrown_card)
    remain_card = get_remain_card(List(me_current_state+enemy_current_state+new_card + thrown_card))
    expectation_list = [0]
    state_list = [""]
    action_list = [""]

    front = me_current_state[0:3]
    middle = me_current_state[3:8]
    back = me_current_state[8:13]
    if "NC" in front:
        front = front[0:front.index("NC")]
    if "NC" in middle:
        middle = middle[0:middle.index("NC")]
    if "NC" in back:
        back = back[0:back.index("NC")]

    if step == 0:#new card size =5
        for idx1 in range(3):
            for idx2 in range(3):
                for idx3 in range(3):
                    for idx4 in range(3):
                        for idx5 in range(3):
                            idx = np.array([idx1,idx2,idx3,idx4,idx5])
                            if np.count_nonzero(idx)<3:#i.e. 0,0,0,0,2
                                continue
                            else:
                                tmp_me_current_state =me_current_state
                                me_next_state =make_next_state(List(tmp_me_current_state),List(new_card),List(idx))
                                if len(me_next_state) == 1:
                                    continue
                                if not pre_check_5(List(me_next_state)):
                                    continue
                                str_index = [""]
                                for e in idx:
                                    str_index += [str(e)]
                                str_index = "_".join(str_index)
                                me_next_state_str = "_".join(me_next_state)
                                state_list += [me_next_state_str]
                                action_list += [str_index]
                                expectation_list += [0]                     
                                    
    if step > 0:
        for i in range(3):
            tmp_new_card = list(new_card)
            del tmp_new_card[i]
            for idx1 in range(3):
                for idx2 in range(3):
                    str_idx = np.array([i,idx1,idx2])
                    idx = np.array([idx1,idx2])
                    str_index = [""]
                    for e in str_idx:
                        str_index += [str(e)]
                    str_index = "_".join(str_index)
                   
                    tmp_me_current_state = me_current_state
                    me_next_state =make_next_state(List(tmp_me_current_state),List(tmp_new_card),List(idx))
                    if len(me_next_state) == 1:
                        continue
                    if step < 2 and not pre_check_3(List(tmp_me_current_state),List(me_next_state)):
                        continue
                    me_next_state_str = "_".join(me_next_state)
                    state_list += [me_next_state_str]
                    action_list += [str_index]
                    expectation_list += [0]
    expectation_list = expectation_list[1:len(expectation_list)]
    state_list = state_list[1:len(state_list)]
    action_list = action_list[1:len(action_list)]
    if loop_cnt == 0 and len(action_list) != 0:
        loop_cnt = int(13000 * 20 / len(action_list))
    if len(action_list) == 0:
        print("**********action list is empty*************")
    print(len(action_list),"thread is calculating reward")
    if len(action_list) > 1:
        for i in numba.prange(len(action_list)):
            tmp_me_next_state = str(state_list[i]).split("_")
            tmp_enemy_current_state = enemy_current_state
            penalty_cnt = 0
            for loop in range(loop_cnt):
                enemy_random_final_state = make_random_final_state(List(tmp_enemy_current_state),List(remain_card))
                me_random_final_state = make_random_final_state(List(tmp_me_next_state),List(get_remain_card(List(enemy_random_final_state+tmp_me_next_state+new_card+thrown_card))))
                score = evaluate(List(me_random_final_state),List(enemy_random_final_state))
                if score == -9999999:
                        penalty_cnt += 1
                else:
                    if score > 0:
                        expectation_list[i]  += int(math.pow(score,1 + (5-step) / 10.0))
                    else:
                        expectation_list[i]  += score
                if penalty_cnt == loop_cnt:
                    expectation_list[i] = -9999999999999999

   
    print("the count is ",len(action_list))  
    print("loop cnt is  ", loop_cnt)                        
    max_value = -99999999999999999
    max_key = 0
    for i in range(len(expectation_list)):
        value = expectation_list[i]
        if value > max_value:
            max_key = i
            max_value = value
    print("expection value is", expectation_list[max_key])
    return state_list[max_key] , action_list[max_key][1:],expectation_list[max_key]




@njit(parallel = True)
def calc_best_next_state_using_jit_3(me_current_state,enemy_left_current_state,enemy_right_current_state,new_card,thrown_card,step,loop_cnt=0):
    me_current_state = list(me_current_state)
    enemy_left_current_state = list(enemy_left_current_state)
    enemy_right_current_state = list(enemy_right_current_state)
    new_card = list(new_card)
    thrown_card = list(thrown_card)
    remain_card = get_remain_card(List(me_current_state+enemy_left_current_state+enemy_right_current_state+new_card + thrown_card))
    expectation_list = [0]
    state_list = [""]
    action_list = [""]

    front = me_current_state[0:3]
    middle = me_current_state[3:8]
    back = me_current_state[8:13]
    if "NC" in front:
        front = front[0:front.index("NC")]
    if "NC" in middle:
        middle = middle[0:middle.index("NC")]
    if "NC" in back:
        back = back[0:back.index("NC")]

    if step == 0:#new card size =5
        for idx1 in range(3):
            for idx2 in range(3):
                for idx3 in range(3):
                    for idx4 in range(3):
                        for idx5 in range(3):
                            idx = np.array([idx1,idx2,idx3,idx4,idx5])
                            if np.count_nonzero(idx)<3:#i.e. 0,0,0,0,2
                                continue
                            else:
                                tmp_me_current_state =me_current_state
                                me_next_state =make_next_state(List(tmp_me_current_state),List(new_card),List(idx))
                                if len(me_next_state) == 1:
                                    continue
                                if not pre_check_5(List(me_next_state)):
                                    continue
                                str_index = [""]
                                for e in idx:
                                    str_index += [str(e)]
                                str_index = "_".join(str_index)
                                me_next_state_str = "_".join(me_next_state)
                                state_list += [me_next_state_str]
                                action_list += [str_index]
                                expectation_list += [0]                     
                                    
    if step > 0:
        for i in range(3):
            tmp_new_card = list(new_card)
            del tmp_new_card[i]
            for idx1 in range(3):
                for idx2 in range(3):
                    str_idx = np.array([i,idx1,idx2])
                    idx = np.array([idx1,idx2])
                    str_index = [""]
                    for e in str_idx:
                        str_index += [str(e)]
                    str_index = "_".join(str_index)
                   
                    tmp_me_current_state = me_current_state
                    me_next_state =make_next_state(List(tmp_me_current_state),List(tmp_new_card),List(idx))
                    if len(me_next_state) == 1:
                        continue
                    if step < 4 and not pre_check_3(List(tmp_me_current_state),List(me_next_state)):
                        continue     
                    
                    me_next_state_str = "_".join(me_next_state)
                    state_list += [me_next_state_str]
                    action_list += [str_index]
                    expectation_list += [0]
    expectation_list = expectation_list[1:len(expectation_list)]
    state_list = state_list[1:len(state_list)]
    action_list = action_list[1:len(action_list)]
    if  loop_cnt == 0 and len(action_list) != 0:
        loop_cnt = int(8000 * 20 / len(action_list))
    if len(action_list) == 0:
        print("**********action list is empty*************")
    print(len(action_list),"thread is calculating reward")
    if len(action_list) > 1:
        for i in numba.prange(len(action_list)):
            tmp_me_next_state = str(state_list[i]).split("_")
            tmp_enemy_left_current_state = enemy_left_current_state
            tmp_enemy_right_current_state = enemy_right_current_state
            penalty_cnt = 0
            for loop in range(loop_cnt):
                enemy_left_random_final_state = make_random_final_state(List(tmp_enemy_left_current_state),List(remain_card))
                enemy_right_random_final_state = make_random_final_state(List(tmp_enemy_right_current_state),List(get_remain_card(List(enemy_left_random_final_state+enemy_right_current_state+tmp_me_next_state+new_card+thrown_card ))))
                me_random_final_state = make_random_final_state(List(tmp_me_next_state),List(get_remain_card(List(enemy_left_random_final_state+tmp_me_next_state+new_card+thrown_card+enemy_right_random_final_state))))
                score1 = evaluate(List(me_random_final_state),List(enemy_left_random_final_state))
                score2 = evaluate(List(me_random_final_state),List(enemy_right_random_final_state))
                if score1 == -9999999:
                        penalty_cnt += 1
                else:
                    expectation_list[i]  += score1
                if score2 == -9999999:
                    penalty_cnt += 1
                else:
                    expectation_list[i]  += score2
                if penalty_cnt == 2*loop_cnt:
                    expectation_list[i] = -9999999999999999

   
    print("the count is ",len(action_list))  
    print("loop cnt is  ", loop_cnt)                        
    max_value = -99999999999999999
    max_key = 0
    for i in range(len(expectation_list)):
        value = expectation_list[i]
        if value > max_value:
            max_key = i
            max_value = value
    print("expection value is", expectation_list[max_key])
    return state_list[max_key] , action_list[max_key][1:],expectation_list[max_key]





'''
_ = List(['7S', '7C', '7D', 'TS', 'TC', '5S', '5H', 'NC', 'QD', '8D', 'NC', 'NC', 'NC'])
__ = List(['NC', 'NC', 'NC', 'NC', 'NC', 'NC', 'NC', 'NC', 'NC', 'NC', 'NC', 'NC', 'NC'])
calc_best_next_state_using_jit( _,__,List(["9C","JH","4D"]),List(["AH","8C","NC","NC"]),3)

current_state = ['NC', 'NC', 'NC' ,'6H' ,'TD' ,'NC' ,'NC' ,'NC', 'KS', 'KD', 'TD' ,'NC' ,'NC']
print(pre_check_5(current_state))
next_state =  ['NC', 'NC', 'NC' ,'6D' ,'5S' ,'5D' ,'JS' ,'JC', 'JH', 'TS', '9D' ,'8H' ,'NC']
print(pre_check_3(current_state,next_state))
#print(is_advanced(get_hand(['JH', 'TS', '9D', '8H']),get_hand(['JH', 'TS', '9D', '8H', 'JC'])))
#print(can_advance_with_new_card(["9S","8S"],"9C"))'''