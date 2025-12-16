import functools

# --- CONFIGURATION ---

MIN_PROB = 0.25
MAX_PROB = 0.75

class GoalConfig:
    def __init__(self, target_main, target_off, max_malus, strict_order=False):
        """
        :param target_main: The higher target (e.g., 9)
        :param target_off: The lower target (e.g., 7)
        :param max_malus: Max allowed red nodes (e.g., 4)
        :param strict_order:
        If True: L1 must be >= target_main and L2 >= target_off.
        If False: (L1>=Main & L2>=Off) OR (L1>=Off & L2>=Main) is accepted.
        """
        self.target_main = target_main
        self.target_off = target_off
        self.max_malus = max_malus
        self.strict_order = strict_order

# --- MEMOIZATION & RECURSION ---

@functools.lru_cache(maxsize=None)
def get_win_prob(prob, l1_s, l1_r, l2_s, l2_r, mal_s, mal_r,
                 t_main, t_off, max_mal, strict):
    """
    Recursive Expectimax function to find probability of success.
    Arguments are flattened to allow lru_cache hashing.
    """

    # 1. Failure Conditions
    if mal_s > max_mal:
        return 0.0

    # Check mathematical impossibility (Not enough nodes left to reach target)
    # Note: For non-strict, we check if it's impossible to reach EITHER config
    total_l1 = l1_s + l1_r
    total_l2 = l2_s + l2_r

    if strict:
        if total_l1 < t_main or total_l2 < t_off:
            return 0.0
    else:
        # If we can't make (9/7) AND we can't make (7/9), it's over.
        can_make_normal = (total_l1 >= t_main and total_l2 >= t_off)
        can_make_flipped = (total_l1 >= t_off and total_l2 >= t_main)
        if not can_make_normal and not can_make_flipped:
            return 0.0

    # 2. Terminal State (No moves left)
    if l1_r == 0 and l2_r == 0 and mal_r == 0:
        # Check Win Condition
        if strict:
            return 1.0 if (l1_s >= t_main and l2_s >= t_off and mal_s <= max_mal) else 0.0
        else:
            c1 = (l1_s >= t_main and l2_s >= t_off)
            c2 = (l1_s >= t_off and l2_s >= t_main)
            return 1.0 if ((c1 or c2) and mal_s <= max_mal) else 0.0

    # 3. Recursion
    best_outcome = -1.0

    # Helper to calculate next state probability
    def next_step(is_success, curr_prob):
        if is_success: return max(MIN_PROB, curr_prob - 0.1)
        else: return min(MAX_PROB, curr_prob + 0.1)

    # -- Try Line 1 --
    if l1_r > 0:
        p_s = prob
        p_f = 1.0 - prob
        # Win Prob if Success + Win Prob if Fail
        ev = (p_s * get_win_prob(round(next_step(True, prob), 2), l1_s+1, l1_r-1, l2_s, l2_r, mal_s, mal_r, t_main, t_off, max_mal, strict) +
              p_f * get_win_prob(round(next_step(False, prob), 2), l1_s, l1_r-1, l2_s, l2_r, mal_s, mal_r, t_main, t_off, max_mal, strict))
        best_outcome = max(best_outcome, ev)

    # -- Try Line 2 --
    if l2_r > 0:
        p_s = prob
        p_f = 1.0 - prob
        ev = (p_s * get_win_prob(round(next_step(True, prob), 2), l1_s, l1_r, l2_s+1, l2_r-1, mal_s, mal_r, t_main, t_off, max_mal, strict) +
              p_f * get_win_prob(round(next_step(False, prob), 2), l1_s, l1_r, l2_s, l2_r-1, mal_s, mal_r, t_main, t_off, max_mal, strict))
        best_outcome = max(best_outcome, ev)

    # -- Try Malus --
    if mal_r > 0:
        p_s = prob
        p_f = 1.0 - prob
        # Note: Success on Malus increases mal_s (bad), Fail keeps mal_s same (good)
        ev = (p_s * get_win_prob(round(next_step(True, prob), 2), l1_s, l1_r, l2_s, l2_r, mal_s+1, mal_r-1, t_main, t_off, max_mal, strict) +
              p_f * get_win_prob(round(next_step(False, prob), 2), l1_s, l1_r, l2_s, l2_r, mal_s, mal_r-1, t_main, t_off, max_mal, strict))
        best_outcome = max(best_outcome, ev)

    return best_outcome

# --- PUBLIC API ---

def calculate_best_move(state, prioritized_goals):
    """
    state: dict with keys: prob, l1_s, l1_r, l2_s, l2_r, mal_s, mal_r
    prioritized_goals: list of GoalConfig objects, ordered by priority (e.g., [Goal(9,7), Goal(7,7)])
    """

    # 1. Iterate through goals to find the highest tier that is still possible
    active_goal = None

    for goal in prioritized_goals:
        # Check current probability of achieving this goal
        win_chance = get_win_prob(
            round(state['prob'], 2),
            state['l1_s'], state['l1_r'],
            state['l2_s'], state['l2_r'],
            state['mal_s'], state['mal_r'],
            goal.target_main, goal.target_off, goal.max_malus, goal.strict_order
        )

        # If this goal is mathematically possible (prob > 0), lock it in.
        # Usually, if 9/7 chance is 0.0001%, we still want to chase it.
        # Only switch if chance is exactly 0.0
        if win_chance > 0.0:
            active_goal = goal
            break

    if active_goal is None:
        return "Impossible to reach any goals", {}, None

    # 2. Calculate EV for each move based on the Active Goal
    moves = {}
    curr_prob = round(state['prob'], 2)

    # Helper for next prob
    def get_next_p(p, success):
        return round(max(MIN_PROB, p - 0.1) if success else min(MAX_PROB, p + 0.1), 2)

    # -- EV for Line 1 --
    if state['l1_r'] > 0:
        p_s = curr_prob
        p_f = 1.0 - curr_prob
        ev = (p_s * get_win_prob(get_next_p(curr_prob, True), state['l1_s']+1, state['l1_r']-1, state['l2_s'], state['l2_r'], state['mal_s'], state['mal_r'], active_goal.target_main, active_goal.target_off, active_goal.max_malus, active_goal.strict_order) +
              p_f * get_win_prob(get_next_p(curr_prob, False), state['l1_s'], state['l1_r']-1, state['l2_s'], state['l2_r'], state['mal_s'], state['mal_r'], active_goal.target_main, active_goal.target_off, active_goal.max_malus, active_goal.strict_order))
        moves["Line 1"] = ev

    # -- EV for Line 2 --
    if state['l2_r'] > 0:
        p_s = curr_prob
        p_f = 1.0 - curr_prob
        ev = (p_s * get_win_prob(get_next_p(curr_prob, True), state['l1_s'], state['l1_r'], state['l2_s']+1, state['l2_r']-1, state['mal_s'], state['mal_r'], active_goal.target_main, active_goal.target_off, active_goal.max_malus, active_goal.strict_order) +
              p_f * get_win_prob(get_next_p(curr_prob, False), state['l1_s'], state['l1_r'], state['l2_s'], state['l2_r']-1, state['mal_s'], state['mal_r'], active_goal.target_main, active_goal.target_off, active_goal.max_malus, active_goal.strict_order))
        moves["Line 2"] = ev

    # -- EV for Malus --
    if state['mal_r'] > 0:
        p_s = curr_prob
        p_f = 1.0 - curr_prob
        ev = (p_s * get_win_prob(get_next_p(curr_prob, True), state['l1_s'], state['l1_r'], state['l2_s'], state['l2_r'], state['mal_s']+1, state['mal_r']-1, active_goal.target_main, active_goal.target_off, active_goal.max_malus, active_goal.strict_order) +
              p_f * get_win_prob(get_next_p(curr_prob, False), state['l1_s'], state['l1_r'], state['l2_s'], state['l2_r'], state['mal_s'], state['mal_r']-1, active_goal.target_main, active_goal.target_off, active_goal.max_malus, active_goal.strict_order))
        moves["Malus"] = ev

    if not moves:
        return "No moves", {}, active_goal

    best_move_name = max(moves, key=moves.get)
    return best_move_name, moves, active_goal
