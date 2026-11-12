V_FAVOR = 0.25
V_AGAINST_NB = 0.625
V_AGAINST_B = 0.75
ALPHA_UP = 0.45
ALPHA_DOWN = 0.17
TRUST_THRESHOLD = 0.55
PROMISE_THRESHOLD = 0.70


class SotaMood:
    def __init__(self, valence=0.30, kept=2, broken=0):
        self.mood = valence
        self.kept = kept
        self.broken = broken
        self.history = []

    def broken_promise(self):
        return self.broken / (self.kept + self.broken)

    def trust(self):
        return 1 - self.broken_promise()

    def target(self, p):
        b = self.broken_promise()
        v_against = (1 - b) * V_AGAINST_NB + b * V_AGAINST_B
        return (
            (1 - b) * p["favor"] * V_FAVOR
            + p["against"] * v_against
            + (p["neither"] + b * p["favor"]) * self.mood
        )

    def update(self, p):
        T = self.target(p)
        alpha = ALPHA_UP if T > self.mood else ALPHA_DOWN
        self.mood = self.mood + alpha * (T - self.mood)
        return T, alpha

    def band(self):
        return "positive" if self.mood <= 0.50 else "negative"

    def decide(self, promised):
        return "silent" if self.trust() >= TRUST_THRESHOLD and promised else "betray"

    def reveal(self, round_num, promised, sota_move, player_move):
        if promised:
            if player_move == "silent":
                self.kept += 1
                self.mood += ALPHA_DOWN * (V_FAVOR - self.mood)
            else:
                self.broken += 1
                self.mood += ALPHA_UP * (V_AGAINST_B - self.mood)
        self.history.append(
            f"Round {round_num}: "
            f"{'The player promised to stay silent' if promised else 'The player made no promise'}, "
            f"then chose {player_move}. Sota chose {sota_move}."
        )
