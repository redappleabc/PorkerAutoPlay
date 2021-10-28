from itertools import product
from itertools import chain
from itertools import combinations
from tools.evaluator import evaluate_me_state,compare_state
from numba import jit,cuda
max_reward = 0
best_states = []
#@cuda.jit
def get_full_house(_four_kinds,_three_kinds,_one_pairs):
    four_kinds = _four_kinds
    three_kinds = _three_kinds
    one_pairs = _one_pairs
    three_kinds += [list(item) for item in chain.from_iterable(list(combinations(four_kind,3)) for four_kind in four_kinds)]
    one_pairs +=   [list(item) for item in chain.from_iterable(list(combinations(four_kind,2)) for four_kind in four_kinds)]
    one_pairs +=  [list(item) for item in chain.from_iterable(list(combinations(three_kind,2)) for three_kind in three_kinds)]
    full_house = list((three_kinds+one_pairs) for three_kinds,one_pairs in list(product(three_kinds, one_pairs)))
    full_house = [item for item in full_house if len(set(item)) == len(item)]
    return full_house

#@cuda.jit
def get_high_card(hand):
    if hand is None or len(hand) < 1:
        print("get high card error")
        return []
    if len(hand) == 1:
        return hand
    ranks = '23456789TJQKA'
    rcounts = {ranks.find(r): ''.join(hand).count(r) for r, _ in hand}.items()
    hand_ranks ,score= zip(*sorted((rank, cnt) for rank, cnt in rcounts)[::-1])
    return [[item for item in hand if item[0] == ranks[hand_ranks[0]]][0]]
#@cuda.jit
def get_possible_2cards(hand):
    if hand is None or len(hand) < 2:
        print("get possible_2card error")
        return []
    ranks = '23456789TJQKA'
    rcounts = {ranks.find(r): ''.join(hand).count(r) for r, _ in hand}.items()
    hand_ranks ,score= zip(*sorted((rank, cnt) for rank, cnt in rcounts)[::-1])
    one_pairs = [[rank+suit for rank,suit in hand if rank == ranks[hand_ranks[i]]] for i,item in enumerate(score) if item == 2]
    return one_pairs
#@cuda.jit
def get_possible_3cards(hand):
    if hand is None or len(hand) < 3:
        print("get possible_3card error")
        return []
    ranks = '23456789TJQKA'
    rcounts = {ranks.find(r): ''.join(hand).count(r) for r, _ in hand}.items()
    hand_ranks ,score= zip(*sorted((rank, cnt) for rank, cnt in rcounts)[::-1])
    four_kinds = [[rank+suit for rank,suit in hand if rank == ranks[hand_ranks[i]]] for i,item in enumerate(score) if item == 4]
    three_kinds = [[rank+suit for rank,suit in hand if rank == ranks[hand_ranks[i]]] for i,item in enumerate(score) if item == 3]
    three_kinds += [list(item) for item in chain.from_iterable(list(combinations(four_kind,3)) for four_kind in four_kinds)]
    return three_kinds
#@cuda.jit
def get_possible_4cards(hand):
    if hand is None or len(hand) < 4:
        print("get possible_4card error")
        return []
    ranks = '23456789TJQKA'
    rcounts = {ranks.find(r): ''.join(hand).count(r) for r, _ in hand}.items()
    hand_ranks ,score= zip(*sorted((rank, cnt) for rank, cnt in rcounts)[::-1])
    four_kinds = [[rank+suit for rank,suit in hand if rank == ranks[hand_ranks[i]]] for i,item in enumerate(score) if item == 4]
    one_pairs = [[rank+suit for rank,suit in hand if rank == ranks[hand_ranks[i]]] for i,item in enumerate(score) if item == 2]
    _two_pairs = list(combinations(one_pairs, 2))
    two_pairs = [item1+item2 for item1,item2,in _two_pairs]
    return four_kinds + two_pairs
#@cuda.jit
def get_possible_5cards(hand):
    if hand is None or len(hand) < 5:
        print("get possible_5card error")
        return []
    ranks = '23456789TJQKA'
    suits = "SDHC"
    rcounts = {ranks.find(r): ''.join(hand).count(r) for r, _ in hand}.items()
    hand_ranks ,score= zip(*sorted((rank, cnt) for rank, cnt in rcounts)[::-1])
    four_kinds = [[rank+suit for rank,suit in hand if rank == ranks[hand_ranks[i]]] for i,item in enumerate(score) if item == 4]
    three_kinds = [[rank+suit for rank,suit in hand if rank == ranks[hand_ranks[i]]] for i,item in enumerate(score) if item == 3]
    one_pairs = [[rank+suit for rank,suit in hand if rank == ranks[hand_ranks[i]]] for i,item in enumerate(score) if item == 2]
    full_house = get_full_house(four_kinds,three_kinds,one_pairs)
    straits = []
    for i in range(len(hand_ranks)-4):
        if hand_ranks[i] - hand_ranks[i+4] == 4:
            list_1 = [rank+suit for rank,suit in hand if rank == ranks[hand_ranks[i]]]
            list_2 = [rank+suit for rank,suit in hand if rank == ranks[hand_ranks[i+1]]]
            list_3 = [rank+suit for rank,suit in hand if rank == ranks[hand_ranks[i+2]]]
            list_4 = [rank+suit for rank,suit in hand if rank == ranks[hand_ranks[i+3]]]
            list_5 = [rank+suit for rank,suit in hand if rank == ranks[hand_ranks[i+4]]]
            straits += list(product(list_5, list_4,list_3,list_2,list_1))
    if len(hand_ranks) > 4 and hand_ranks[0] - hand_ranks[-4] == 9:
        list_1 = [rank+suit for rank,suit in hand if rank == ranks[hand_ranks[-1]]]
        list_2 = [rank+suit for rank,suit in hand if rank == ranks[hand_ranks[-2]]]
        list_3 = [rank+suit for rank,suit in hand if rank == ranks[hand_ranks[-3]]]
        list_4 = [rank+suit for rank,suit in hand if rank == ranks[hand_ranks[-4]]]
        list_5 = [rank+suit for rank,suit in hand if rank == ranks[hand_ranks[0]]]
        straits += list(product(list_5,list_1, list_2,list_3,list_4))

    flushes = []
    suits_pair = [[item for item in hand if item[1] == suit ] for suit in suits]
    for i, flush in enumerate(suits_pair):
        if len(flush) >=5 :
            flushes += combinations(flush, 5)
    
    
    all_card = list(set(tuple(sorted(item)) for item in flushes + straits + full_house))
    return [list(item) for item in all_card]



#@cuda.jit
def state_fit_and_calc_reward(back_hand,mid_hand,front_hand,remain_hand):
    back_lack = 5-len(back_hand)
    mid_lack = 5 - len(mid_hand)
    front_lack = 3 - len(front_hand)

    loops = max(back_lack,mid_lack,front_lack)
    for i in range(loops):
        if len(front_hand) < 3:
            high_card = get_high_card(remain_hand)
            front_hand += high_card
            remain_hand = list(set(remain_hand) - set(high_card))
        if len(mid_hand) < 5 :
            high_card = get_high_card(remain_hand)
            mid_hand += get_high_card(remain_hand)
            remain_hand = list(set(remain_hand) - set(high_card))
        if len(back_hand) < 5:
            high_card = get_high_card(remain_hand)
            back_hand += get_high_card(remain_hand)
            remain_hand = list(set(remain_hand) - set(high_card))
    final_state = front_hand+mid_hand+back_hand
    score = evaluate_me_state(final_state)
    return score , final_state

#@cuda.jit
def update_best_states(back_five_card,mid_five_card,front_three_card,front_hand):
    remain_hand = list(set(front_hand) - set(front_three_card))
    reward,final_state = state_fit_and_calc_reward(back_five_card,mid_five_card,front_three_card,remain_hand)
    global max_reward,best_states
    if reward > max_reward:
        max_reward = reward
        best_states = [final_state]
    elif reward == max_reward:
        best_states = best_states + [final_state]
#@cuda.jit
def configure_mid_cards(back_five_card,mid_five_cards,mid_hand):
    for mid_five_card in mid_five_cards:
        front_hand = list(set(mid_hand) - set(mid_five_card))
        front_three_cards = get_possible_3cards(front_hand)
        front_two_cards = get_possible_2cards(front_hand)
        if front_three_cards:
            for front_three_card in front_three_cards:
                update_best_states(back_five_card,mid_five_card,front_three_card,front_hand)
        elif front_two_cards:
            for front_two_card in front_two_cards:
                    update_best_states(back_five_card,mid_five_card,front_two_card,front_hand)
        else:
            update_best_states(back_five_card,mid_five_card,[],front_hand)
#@cuda.jit
def configure_back_cards(back_five_cards,back_hand):
    for back_five_card in back_five_cards:
        mid_hand = list(set(back_hand) - set(back_five_card))
        mid_five_cards = get_possible_5cards(mid_hand)
        mid_four_cards = get_possible_4cards(mid_hand)
        mid_three_cards = get_possible_3cards(mid_hand)
        mid_two_cards = get_possible_2cards(mid_hand)
        if mid_five_cards:
            configure_mid_cards(back_five_card,mid_five_cards,mid_hand)
        if mid_four_cards:
            configure_mid_cards(back_five_card,mid_four_cards,mid_hand)
        if mid_three_cards:
            configure_mid_cards(back_five_card,mid_three_cards,mid_hand)
        if mid_two_cards:
            configure_mid_cards(back_five_card,mid_two_cards,mid_hand)

def select_most_fit_state():
    if len(best_states) == 0:
        return best_states[0]
    best_state = best_states[0]
    for i ,state in enumerate(best_states):
        best_state = compare_state(best_state,state)
    return best_state


def calc_best_fantasy_state(hand):
    back_hand = hand
    back_five_cards = get_possible_5cards(back_hand)
    back_four_cards = get_possible_4cards(back_hand)
    back_three_cards = get_possible_3cards(back_hand)
    back_two_cards = get_possible_2cards(back_hand)
    if back_five_cards:
        configure_back_cards(back_five_cards,back_hand)
    if back_four_cards:
        configure_back_cards(back_four_cards,back_hand)
    if back_three_cards:
        configure_back_cards(back_three_cards,back_hand)
    if back_two_cards:
        configure_back_cards(back_two_cards,back_hand)
    return select_most_fit_state()





'''
#get_possible_5cards(hand)

def calc_fantasy_max_reward(hand):
    global max_reward
    max_reward = 0
    
    return (max_reward)

for best_state in best_states:
    print(best_state[0:3])
    print(best_state[3:8])
    print(best_state[8:13])
    print()
'''