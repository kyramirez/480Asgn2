import random
import time
from itertools import combinations
import math

# texas holdem bot using monte carlo tree search

SUITS = ['♠', '♥', '♦', '♣']  # spades, hearts, diamonds, clubs
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
# mapping hand rankings to number scores
HAND_RANKINGS = {
    'High card': 0,
    'One pair': 1,
    'Two pair': 2,
    'Three of a Kind': 3,
    'Straight': 4,
    'Flush': 5,
    'Full House': 6,
    'Four of a Kind': 7,
    'Straight Flush': 8,
    'Royal Flush': 9
}
# converting card number to string format
def card_to_str(card):
    return RANKS[card % 13] + SUITS[card // 13]

# generate a full deck 52 cards
def create_the_deck():
    return list(range(52))

# shuffle the deck
def shuffle(deck):
    random.shuffle(deck)

# deal number of n cards from the deck
def deal_cards(deck, n):
    return [deck.pop() for _ in range(n)]

# get the rank of cards 1-12
def card_ranks(card):
    return card % 13

# get the suit of the card 1-3
def card_suit(card):
    return card // 13

# evaluate a 7 card hand and return rank and kicker vals
def eval_cards(cards):
    ranks = sorted([card % 13 for card in cards], reverse=True)
    suits = [card // 13 for card in cards]
    rank_counts = {r: ranks.count(r) for r in set(ranks)}
    suit_counts = {s: suits.count(s) for s in set(suits)}
    is_flush = max(suit_counts.values()) >= 5
    is_straight = False
    straight_high = 0
    sorted_ranks = sorted(set(ranks))
    for i in range(len(sorted_ranks) - 4):
        if sorted_ranks[i + 4] - sorted_ranks[i] == 4:
            is_straight = True
            straight_high = sorted_ranks[i + 4]
    if set([12, 0, 1, 2, 3]).issubset(ranks): # wheel
        is_straight = True
        straight_high = 3

    # determine the hand type based on count and combinations
    if is_straight and is_flush:
        return (HAND_RANKINGS['Straight Flush'], [straight_high])
    if 4 in rank_counts.values():
        four = max(k for k, v in rank_counts.items() if v == 4)
        kicker = max(k for k in ranks if k != four)
        return (HAND_RANKINGS['Four of a Kind'], [four, kicker])
    if 3 in rank_counts.values() and 2 in rank_counts.values():
        three = max(k for k, v in rank_counts.items() if v == 3)
        pair = max(k for k, v in rank_counts.items() if v == 2)
        return (HAND_RANKINGS['Full House'], [three, pair])
    if is_flush:
        flushed_cards = [r for r, s in sorted(zip(ranks, suits), reverse=True) if suits.count(s) >= 5]
        return (HAND_RANKINGS['Flush'], flushed_cards[:5])
    if is_straight:
        return (HAND_RANKINGS['Straight'], [straight_high])
    if 3 in rank_counts.values():
        three = max(k for k, v in rank_counts.items() if v == 3)
        kickers = [k for k in ranks if k != three][:2]
        return (HAND_RANKINGS['Three of a Kind'], [three] + kickers)
    pairs = [k for k, v in rank_counts.items() if v == 2]
    if len(pairs) >= 2:
        top_two = sorted(pairs, reverse=True)[:2]
        kicker = max(k for k in ranks if k not in top_two)
        return (HAND_RANKINGS['Two pair'], top_two + [kicker])
    if 2 in rank_counts.values():
        pair = max(k for k, v in rank_counts.items() if v == 2)
        kickers = [k for k in ranks if k != pair][:3]
        return (HAND_RANKINGS['One pair'], [pair] + kickers)
    return (HAND_RANKINGS['High card'], ranks[:5])

# simulate a complete hand and return true if the bot wins or ties
def simulate(my_hand, community, opp_hand, remaining):
    need = 5 - len(community)
    sim_comm = community + random.sample(remaining, need)
    my_score = eval_cards(my_hand + sim_comm)
    opp_score = eval_cards(opp_hand + sim_comm)
    return my_score >= opp_score

# estimate win probability using UCB1
def estimate_win_prob(my_hand, community, time_lim=10): # UCB1
    start = time.time()
    wins = {}
    sims = {}
    total_sims = 0

    # create a deck and remove known cards
    deck = create_the_deck()
    known = set(my_hand + community)
    deck = [card for card in deck if card not in known]
    possible_opp_hands = list(combinations(deck, 2))
    if len(possible_opp_hands) > 300:
        possible_opp_hands = random.sample(possible_opp_hands, 300) # larger sample

    # initialize for UCB1
    for hand in possible_opp_hands:
        wins[hand] = 0
        sims[hand] = 0

    # run the rollouts using UCB1 until the time runs out
    while time.time() - start < time_lim:
        # use UCB1 to select the opp hand
        total = total_sims + 1
        ucb_vals = {
            h: (wins[h] / sims[h]) + math.sqrt(2 * math.log(total) / sims[h])
            if sims[h] > 0 else float('inf')
            for h in possible_opp_hands
        }
        opp_hand = max(ucb_vals, key=ucb_vals.get)

        remaining_deck = [card for card in deck if card not in opp_hand]
        result = simulate(my_hand, community, list(opp_hand), remaining_deck)

        wins[opp_hand] += int(result)
        sims[opp_hand] += 1
        total_sims += 1

    total_wins = sum(wins[h] for h in possible_opp_hands)
    return total_wins / total_sims if total_sims > 0 else 0.0

# decide whether to fold or stay based on estimation
def make_decisions(my_hand, community, phase):
    print(f"\n[{phase}] Your hand: {[card_to_str(c) for c in my_hand]}")
    print(f"Community cards: {[card_to_str(c) for c in community]}")
    win_prob = estimate_win_prob(my_hand, community, time_lim=10)
    print(f"Estimated win probability: {win_prob:.2%}")
    return "stay" if win_prob >= 0.5 else "fold"

# return the hand name string from the score tuple
def hand_rank_name(score_tuple):
    for name, rank in HAND_RANKINGS.items():
        if rank == score_tuple[0]:
            return name.title()
    return "Unknown"

# play a single hand of texas holdem against an opponent
def play_hand():
    deck = create_the_deck()
    shuffle(deck)

    my_hand = deal_cards(deck, 2)
    opp_hand = deal_cards(deck, 2) # these will be hidden

    # deal
    community = []
    decision = make_decisions(my_hand, community, phase="PreFLop")
    if decision == "fold":
        print("Bot folds before the flop")
        return

    # flop 3 cards
    community += deal_cards(deck, 3)
    decision = make_decisions(my_hand, community, phase="Pre-turn")
    if decision == "fold":
        print("Bot folds before the turn")
        return

    # turn 1 card
    community += deal_cards(deck, 1)
    decision = make_decisions(my_hand, community, phase="Pre-River")
    if decision == "fold":
        print("Bot folds before the river")
        return

    # river
    community += deal_cards(deck, 1)

    # the final showdown
    my_score = eval_cards(my_hand + community)
    opp_score = eval_cards(opp_hand + community)
    print(f"Final community: {[card_to_str(c) for c in community]}")
    print(f"Opponent hand: {[card_to_str(c) for c in opp_hand]}")
    print(f"Your score: {my_score} ({hand_rank_name(my_score)}), Opponent score: {opp_score} ({hand_rank_name(opp_score)})")
    print(f"You win!" if my_score >= opp_score else "You lose.")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    play_hand()

