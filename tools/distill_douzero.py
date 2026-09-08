"""DouZero 蒸馏数据生成器 (路线 1: 教师升级)
三座次全是 DouZero-Resnet 自博弈; 每个决策点捕获:
  我们的 44 维 obs + 候选枚举 + DouZero 实际动作映射为标签
输出: v4 格式 (DDZC magic, version=4) — 与 train_cand 管线直接兼容
语义外动作 (顺子/三带/飞机) 跳过 (已知选择偏差, 诚实记录跳过率)
"""
import sys, os, random, argparse, pickle
import numpy as np

DZ = "/tmp/opencode/douzero-resnet-2.0/Douzero_Resnet"
sys.path.insert(0, DZ)

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
        return ({1: 1, 2: 2}.get(len(move), (3 if len(move) == 4 else -1)), r, len(move))
    return (-1, CARD2RANK[min(move)], len(move))


def build_obs(infoset):
    obs = [0.0] * 44
    myh = {}
    for c in infoset.player_hand_cards:
        r = CARD2RANK[c]
        myh[r] = myh.get(r, 0) + 1
        mx = 1.0 if r >= 13 else 4.0
        obs[r] = min(1.0, obs[r] + 1.0 / mx)
    played = {}
    for entry in infoset.card_play_action_seq:
        mv = entry[1] if isinstance(entry, tuple) else entry
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


class CaptureDouZeroAgent:
    """DouZero 打牌代理 + 蒸馏捕获 (标签 = DouZero 动作 ∩ 我们的语义)"""

    _cache = {}
    n_bid = 0
    n_play = 0
    n_pos_none = 0

    def __init__(self, rec):
        self.rec = rec   # (obs44, cands列表, acts列表, label) 缓冲

    def act(self, infoset):
        pos = getattr(infoset, 'player_position', 'MISSING')
        if pos in SEATS:
            CaptureDouZeroAgent.n_bid += 1
            return random.choice(infoset.legal_actions)
        if pos in ORDER:
            CaptureDouZeroAgent.n_play += 1
        else:
            CaptureDouZeroAgent.n_pos_none += 1
            print(f"[diag] 异常 position: {pos!r}")
            return random.choice(infoset.legal_actions)
        from douzero.evaluation.deep_agent import DeepAgent
        role = infoset.player_position
        if role not in CaptureDouZeroAgent._cache:
            CaptureDouZeroAgent._cache[role] = DeepAgent(role, f"{DZ}/baseline/best/{role}.ckpt")
        # 先构建我们的候选 (动作前)
        obs = build_obs(infoset)
        cands, acts = enum_candidates(infoset)
        action = CaptureDouZeroAgent._cache[role].act(infoset)
        # 标签映射: DouZero 动作 ∈ 我们的语义?
        if action:
            if set(action) == {20, 30}:
                key = (4, 14, 2)
            elif len(action) == 1:
                key = (1, CARD2RANK[action[0]], 1)
            elif len(action) == 2 and action[0] == action[1]:
                key = (2, CARD2RANK[action[0]], 2)
            elif len(action) == 4 and action[0] == action[1] == action[2] == action[3]:
                key = (3, CARD2RANK[action[0]], 4)
            else:
                key = None
        else:
            key = (0, -1, 0)
        label = -1
        if key is not None:
            for i, f in enumerate(cands):
                ck = (int(f[4]) or ({0: 0, 1: 0, 2: 0, 3: 0}.get(0) or 0),)  # placeholder
            # 直接用 acts 匹配 (同键已去重, 首个匹配即标签)
            for i, a in enumerate(acts):
                if not action and not a:
                    label = i
                    break
                if action and a and len(a) == len(action) and a == action:
                    label = i
                    break
        self.rec.append((obs, cands, label))
        return action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", type=str, default="/tmp/opencode/doudizhu_douzero_distill.bin")
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    from game_eval import GameEnv

    random.seed(args.seed)
    np.random.seed(args.seed)

    rec = []
    rec_agent = CaptureDouZeroAgent(rec)
    wins = 0
    skip_total = 0
    for g in range(args.games):
        deck = [c for c in range(3, 15) for _ in range(4)] + [17] * 4 + [20, 30]
        random.shuffle(deck)
        data = {'first': sorted(deck[:17]), 'second': sorted(deck[20:37]),
                'third': sorted(deck[37:]), 'three_landlord_cards': sorted(deck[17:20])}
        # [修复] 不清空 rec — 跨局累积 (此前只写了最后一局的样本!)
        env = GameEnv(None)
        players = {k: {k: rec_agent} for k in SEATS + ORDER}
        env.players = players
        env.bid_init(dict(data))
        while not env.bid_over:
            env.step()
        if env.draw:
            continue
        while not env.game_over:
            env.step()
        # 胜负: 地主是谁 + 阵营判定 (rec 全部是 DouZero — 不分你我, 每样本按当时视角标 won)
        # 简化: 不区分角色 won — v1 蒸馏只用 CE, won 字段写 0 占位 (AWR 后续启用)

        if (g + 1) % 100 == 0:
            print(f"[{g+1}/{args.games}] 累计捕获 {len(rec)} 决策 "
                  f"(bid={CaptureDouZeroAgent.n_bid} play={CaptureDouZeroAgent.n_play})", flush=True)

    # 写盘 v4
    with open(args.out, 'wb') as f:
        f.write(b'DDZC')
        f.write((4).to_bytes(4, 'little'))
        f.write((args.games).to_bytes(4, 'little'))
        n = 0
        for obs, cands, label in rec:
            if label < 0:
                continue   # 语义外动作 (跳过率见后)
            f.write(np.array(obs, dtype=np.float32).tobytes())
            f.write(len(cands).to_bytes(4, 'little'))
            for cf in cands:
                f.write(np.array(cf, dtype=np.float32).tobytes())
            f.write(label.to_bytes(4, 'little'))
            f.write((0).to_bytes(4, 'little'))   # gid 占位
            f.write((0).to_bytes(4, 'little'))   # won 占位 (AWR 后续)
            n += 1
    kept = n
    skipped = len(rec) - kept
    print(f"[蒸馏数据] {args.games} 局 → 保留 {kept} 样本 (语义外跳过 {skipped}, 跳过率 {100.0*skipped/max(1,len(rec)):.1f}%)")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
