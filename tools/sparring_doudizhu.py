"""陪练桥 v2: 细胞冠军 vs DouZero-Resnet-best (开源 SOTA 斗地主 AI)
架构: GameEnv(players) + RoleDispatcher (叫牌/打牌两阶段统一调度)
冠军经 C++ serve_cand 子进程应答 (44 obs + K 候选 → 选中索引)
"""
import sys, os, subprocess, random, argparse

DZ = "/tmp/opencode/douzero-resnet-2.0/Douzero_Resnet"
sys.path.insert(0, DZ)

CHAMPION = "/home/caixuf/code/kun-cellular/checkpoints/doudizhu_cand_scorer.bin"
SERVE = "/home/caixuf/code/kun-cellular/build/train_doudizhu_bc_distill"
SEATS = ["first", "second", "third"]

CARD2RANK = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
             11: 8, 12: 9, 13: 10, 14: 11, 17: 12, 20: 13, 30: 14}
RANK_TOTAL = {r: (1 if r >= 13 else 4) for r in range(15)}
ORDER = ['landlord', 'landlord_down', 'landlord_up']


def trick_from_move(move):
    if not move:
        return (0, -1, 0)
    if set(move) == {20, 30}:
        return (4, 14, 2)
    ranks = {CARD2RANK[c] for c in move}
    if len(ranks) == 1:
        r = ranks.pop()
        return ({1: 1, 2: 2, 3: -1, 4: 3}.get(len(move), -1), r, len(move))
    return (-1, CARD2RANK[min(move)], len(move))


def build_obs(infoset):
    """infoset → 44 维观测 (与 C++ current_observation 逐维对齐)"""
    obs = [0.0] * 44
    myh = {}
    for c in infoset.player_hand_cards:
        r = CARD2RANK[c]
        myh[r] = myh.get(r, 0) + 1
        mx = 1.0 if r >= 13 else 4.0
        obs[r] = min(1.0, obs[r] + 1.0 / mx)
    played = {}
    for entry in infoset.card_play_action_seq:
        mv = entry[1] if isinstance(entry, tuple) else entry   # 此版 env 存 (座位, 出牌) 元组
        if not mv:
            continue
        for c in mv:
            r = CARD2RANK[c]
            played[r] = played.get(r, 0) + 1
    for r in range(15):
        u = max(0, RANK_TOTAL[r] - played.get(r, 0) - myh.get(r, 0))
        mx = 1.0 if r >= 13 else 4.0
        if 8 <= r <= 14:
            obs[15 + (r - 8)] = u / mx
        if r <= 7:
            obs[36 + r] = u / mx
    lm = infoset.last_move
    ttype, trank, tcount = trick_from_move(lm if lm else [])
    pos = infoset.player_position
    if lm:
        obs[22] = ttype / 4.0
        obs[23] = trank / 14.0
        obs[24] = tcount / 4.0
        if infoset.last_pid == pos:
            obs[25] = 1.0
        elif pos != 'landlord':
            mate = 'landlord_up' if pos == 'landlord_down' else 'landlord_down'
            if infoset.last_pid == mate:
                obs[26] = 1.0
        if infoset.last_pid == 'landlord' and pos != 'landlord':
            obs[27] = 1.0
    obs[28] = 1.0 if pos == 'landlord' else 0.0
    ncl = infoset.num_cards_left_dict
    obs[29] = min(1.0, ncl[pos] / 20.0)
    if pos != 'landlord':
        mate = 'landlord_up' if pos == 'landlord_down' else 'landlord_down'
        obs[30] = min(1.0, ncl[mate] / 20.0)
    obs[31] = min(1.0, ncl['landlord'] / 20.0)
    idx = ORDER.index(pos)
    nxt, prv = ORDER[(idx + 1) % 3], ORDER[(idx - 1) % 3]
    obs[32] = 1.0 if nxt == 'landlord' else 0.0
    obs[33] = 1.0 if prv == 'landlord' else 0.0
    obs[34] = min(1.0, ncl[nxt] / 20.0)
    obs[35] = min(1.0, ncl[prv] / 20.0)
    return obs


def enum_candidates(infoset):
    """冠军候选语义 (单张/对子/炸弹/火箭/过牌) ∩ DouZero 合法动作"""
    lm = infoset.last_move
    ttype, trank, _ = trick_from_move(lm if lm else [])
    cands, acts, seen = [], [], set()

    def push(action, ctype, crank, ccount):
        key = (ctype, crank, ccount)
        if key in seen:
            return
        seen.add(key)
        f = [0.0] * 12
        if ctype == 0:
            f[4] = 1.0
        else:
            f[{1: 0, 2: 1, 3: 2, 4: 3}[ctype]] = 1.0
            f[5] = crank / 14.0
            f[6] = ccount / 4.0
            f[7] = (ccount / 4.0) if ctype != 4 else 0.5
            if lm:
                f[8] = (crank - trank) / 14.0
                if ctype == ttype and crank > trank:
                    f[9] = 1.0
                if ctype == 3 and ttype not in (3, 4):
                    f[10] = 1.0
                if ctype == 4 and ttype != 4:
                    f[11] = 1.0
            else:
                f[8] = -1.0
        cands.append(f)
        acts.append(action)

    for a in infoset.legal_actions:
        if len(a) == 0:
            if lm:
                push([], 0, -1, 0)
        elif len(a) == 1:
            push(a, 1, CARD2RANK[a[0]], 1)
        elif len(a) == 2:
            if a == [20, 30]:
                push(a, 4, 14, 2)
            elif a[0] == a[1]:
                push(a, 2, CARD2RANK[a[0]], 2)
        elif len(a) == 4 and a[0] == a[1] == a[2] == a[3]:
            push(a, 3, CARD2RANK[a[0]], 4)
    return cands, acts


class ChampionAgent:
    """细胞冠军: serve_cand 子进程应答"""

    def __init__(self, model=CHAMPION, seat=None, env_ref=None):
        self.p = subprocess.Popen([SERVE, "serve_cand", model],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        self.seat = seat

    def act(self, infoset):
        if infoset.player_position in SEATS:   # 叫牌阶段: 随机
            return random.choice(infoset.legal_actions)
        obs = build_obs(infoset)
        cands, acts = enum_candidates(infoset)
        if not cands:
            return []
        self.p.stdin.write(" ".join(repr(x) for x in obs) + "\n")
        self.p.stdin.write(f"{len(cands)}\n")
        for f in cands:
            self.p.stdin.write(" ".join(repr(x) for x in f) + "\n")
        self.p.stdin.flush()
        idx = int(self.p.stdout.readline().strip())
        return acts[min(max(idx, 0), len(acts) - 1)]


class RandomBidAgent:
    """随机叫牌 (官方 auto_test 同款先例)"""

    def act(self, infoset):
        return random.choice(infoset.legal_actions)


class DouZeroAgent:
    """DouZero-Resnet 打牌代理 (按角色懒加载 best 权重)"""

    _cache = {}

    def act(self, infoset):
        if infoset.player_position in SEATS:   # 叫牌阶段: 随机 (官方 auto_test 同款)
            return random.choice(infoset.legal_actions)
        from douzero.evaluation.deep_agent import DeepAgent
        role = infoset.player_position
        if role not in DouZeroAgent._cache:
            ckpt = f"{DZ}/baseline/best/{role}.ckpt"
            DouZeroAgent._cache[role] = DeepAgent(role, ckpt)
        return DouZeroAgent._cache[role].act(infoset)


class RoleDispatcher:
    """叫牌(first/second/third)与打牌(landlord/...)两阶段统一调度"""

    def __init__(self, seat_agents, env_ref):
        self.seat_agents = seat_agents   # {seat: agent}
        self.env_ref = env_ref

    def act(self, infoset):
        pos = getattr(infoset, "player_position", None)
        if pos in SEATS:                    # 叫牌阶段
            ag = self.seat_agents.get(pos)
        else:                               # 打牌阶段: role → seat
            role = pos
            seat = SEATS[list(self.env_ref.position).index(role)]
            ag = self.seat_agents.get(seat)
        return ag.act(infoset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--champion-seat", type=str, default="random",
                    choices=["random", "first", "second", "third"])
    ap.add_argument("--model", type=str, default=CHAMPION)
    args = ap.parse_args()

    from game_eval import GameEnv
    rng = random.Random(args.seed)
    wins = 0
    role_stats = {"landlord": [0, 0], "farmer": [0, 0]}
    champion_proc = ChampionAgent(args.model)
    for g in range(args.games):
        deck = [c for c in range(3, 15) for _ in range(4)] + [17] * 4 + [20, 30]
        rng.shuffle(deck)
        data = {'first': sorted(deck[:17]), 'second': sorted(deck[20:37]),
                'third': sorted(deck[37:]), 'three_landlord_cards': sorted(deck[17:20])}
        seat = args.champion_seat if args.champion_seat != "random" else rng.choice(SEATS)
        champion_proc.seat = seat
        seat_agents = {s: (champion_proc if s == seat else DouZeroAgent())
                       for s in SEATS}
        env = GameEnv(None)
        players = {k: {k: RoleDispatcher(seat_agents, env)} for k in SEATS + ORDER}
        env.players = players
        env.bid_init(dict(data))
        while not env.bid_over:
            env.step()
        if env.draw:
            continue
        while not env.game_over:
            env.step()
        role = env.position[SEATS.index(seat)]
        # 胜负: 冠军阵营 = 地主 或 农民 (任一农民出完)
        champ_hand = env.info_sets[role].player_hand_cards
        if role == 'landlord':
            win = len(champ_hand) == 0
        else:
            mate = [r for r in ORDER if r != 'landlord' and r != role]
            win = any(len(env.info_sets[r].player_hand_cards) == 0 for r in mate)
        key = 'landlord' if role == 'landlord' else 'farmer'
        wins += win
        role_stats[key][0] += win
        role_stats[key][1] += 1
        champion_proc.p.stdin.flush()   # 跨局复用 (serve_cand 每决策自复位, 无状态)
        if (g + 1) % 25 == 0:
            print(f"[{g+1}/{args.games}] 累计胜率 {100.0*wins/(g+1):.1f}%")
    print(f"[陪练战报] 冠军 vs DouZero-Resnet(best): {wins}/{args.games} = {100.0*wins/args.games:.1f}%")
    champion_proc.p.stdin.close()
    champion_proc.p.wait(timeout=10)
    for r, s in role_stats.items():
        if s[1]:
            print(f"  {r}: {s[0]}/{s[1]} = {100.0*s[0]/s[1]:.1f}%")


if __name__ == "__main__":
    main()
