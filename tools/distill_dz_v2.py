"""M9 异火适配器: DouZero 蒸馏数据生成器 v2 (特征空间相容修复)
v1 三处特征投毒 (2026-09-09 破案, docs/superpowers/plans/2026-09-09-online-rl-protocol.md):
  1. obs 块错位: 座次[32..35]/低牌记牌[36..43] (旧布局) ≠ v4 [32..39]/[40..43]
  2. 记牌语义: 干净公式 (总-全出-己) ≠ 我们的座0怪癖 (总-他座出-己, 己出计入未见)
  3. 候选 f[7]: 出牌张数/4 ≠ 我们的持有量/4 (火箭 = 王持有/2)
v2 修复: 逐特征对齐我们的引擎语义; gid/won 真实填写 (供价值头/AWR)
输出: v4 内联格式 (176B obs + K×48B cands + label + gid + won), 与 p9_fuse 兼容
"""
import sys, os, random, argparse
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


def move_key(action):
    """DouZero 动作 → 我们的候选键 (type, rank, count); None = 语义外"""
    if not action:
        return (0, -1, 0)
    if set(action) == {20, 30}:
        return (4, 14, 2)
    if len(action) == 1:
        return (1, CARD2RANK[action[0]], 1)
    if len(action) == 2 and action[0] == action[1]:
        return (2, CARD2RANK[action[0]], 2)
    if len(action) == 4 and action[0] == action[1] == action[2] == action[3]:
        return (3, CARD2RANK[action[0]], 4)
    return None


def build_obs_v4(infoset):
    """44 维 obs — 与我们引擎 seat-0 视角逐语义一致 (含座0记牌怪癖)"""
    obs = [0.0] * 44
    pos = infoset.player_position
    myh = {}
    for c in infoset.player_hand_cards:
        r = CARD2RANK[c]
        myh[r] = myh.get(r, 0) + 1
        mx = 1.0 if r >= 13 else 4.0
        obs[r] = min(1.0, obs[r] + 1.0 / mx)
    # 逐座已出 (怪癖语义需要)
    played = {}
    played_by = {p: {} for p in ORDER}
    for entry in infoset.card_play_action_seq:
        mv = entry[1] if isinstance(entry, tuple) else entry
        pid = entry[0] if isinstance(entry, tuple) else None
        if not mv:
            continue
        for c in mv:
            r = CARD2RANK[c]
            played[r] = played.get(r, 0) + 1
            if pid is not None:
                played_by[pid][r] = played_by[pid].get(r, 0) + 1
    # 座0怪癖: unseen_ours = 总量 − 他座已出 − 己方手牌 = 总量 − 全出 − 己方手牌 + 己座已出
    played_self = played_by.get(pos, {})
    for r in range(15):
        u = max(0, RANK_TOTAL[r] - played.get(r, 0) - myh.get(r, 0) + played_self.get(r, 0))
        mx = 1.0 if r >= 13 else 4.0
        if 8 <= r <= 14:
            obs[15 + (r - 8)] = u / mx          # v4: 高牌未见 [15..21]
        if r <= 7:
            obs[32 + r] = u / mx                # v4: 低牌未见 [32..39]
    lm = infoset.last_move
    ttype, trank, tcount = trick_from_move(lm if lm else [])
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
    obs[40] = 1.0 if nxt == 'landlord' else 0.0   # v4: 座次 [40..43]
    obs[41] = 1.0 if prv == 'landlord' else 0.0
    obs[42] = min(1.0, ncl[nxt] / 20.0)
    obs[43] = min(1.0, ncl[prv] / 20.0)
    return obs, myh


def enum_candidates_v4(infoset, myh):
    """候选枚举 + 特征 — f[7] = 持有量/4 (火箭 = 王持有/2), 与我们引擎逐位一致"""
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
            f[7] = ((myh.get(13, 0) + myh.get(14, 0)) / 2.0) if ctype == 4 \
                else (myh.get(crank, 0) / 4.0)
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
                push(a, 0, -1, 0)
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


class CaptureAgent:
    """DouZero 代理 + 相容蒸馏捕获"""
    _cache = {}
    n_decisions = 0
    n_kept = 0
    n_oom = 0   # 语义外动作

    def __init__(self, rec):
        self.rec = rec

    def act(self, infoset):
        pos = infoset.player_position
        if pos in SEATS:
            return random.choice(infoset.legal_actions)   # 叫牌座: 随机 (与 v1 协议一致)
        from douzero.evaluation.deep_agent import DeepAgent
        if pos not in CaptureAgent._cache:
            CaptureAgent._cache[pos] = DeepAgent(pos, f"{DZ}/baseline/best/{pos}.ckpt")
        obs, myh = build_obs_v4(infoset)
        cands, acts = enum_candidates_v4(infoset, myh)
        action = CaptureAgent._cache[pos].act(infoset)
        key = move_key(action)
        label = -1
        if key is None:
            CaptureAgent.n_oom += 1
        if key is not None:
            for i, a in enumerate(acts):
                if (not key[0] and not a) or (a and action and a == action):
                    label = i
                    break
        CaptureAgent.n_decisions += 1
        if label >= 0:
            CaptureAgent.n_kept += 1
        self.rec.append((pos, obs, cands, label))
        return action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", type=str, default="/tmp/opencode/dz_v2.bin")
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    from game_eval import GameEnv

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    f_out = open(args.out, "wb")
    f_out.write(b'DDZC')
    f_out.write((4).to_bytes(4, 'little'))
    f_out.write((args.games).to_bytes(4, 'little'))

    rec = []
    agent = CaptureAgent(rec)
    for g in range(args.games):
        deck = [c for c in range(3, 15) for _ in range(4)] + [17] * 4 + [20, 30]
        random.shuffle(deck)
        data = {'first': sorted(deck[:17]), 'second': sorted(deck[20:37]),
                'third': sorted(deck[37:]), 'three_landlord_cards': sorted(deck[17:20])}
        env = GameEnv(None)
        env.players = {k: {k: agent} for k in SEATS + ORDER}
        env.bid_init(dict(data))
        while not env.bid_over:
            env.step()
        if env.draw:
            continue
        while not env.game_over:
            env.step()
        winner = getattr(env, 'winner', None)   # GameEnv 结算字段
        if winner is None:
            continue
        if (g + 1) % 100 == 0:
            print(f"[{g+1}/{args.games}] 累计 {len(rec)} 决策 (winner={winner})", flush=True)

        # won: 记录时刻该决策座位的团队胜负
        for pos, obs, cands, label in rec:
            if label < 0:
                continue
            won = 1 if ((pos == 'landlord') == (winner == 'landlord')) else 0
            f_out.write(np.array(obs, dtype=np.float32).tobytes())
            f_out.write(len(cands).to_bytes(4, 'little'))
            for cf in cands:
                f_out.write(np.array(cf, dtype=np.float32).tobytes())
            f_out.write(label.to_bytes(4, 'little'))
            f_out.write((g + 1).to_bytes(4, 'little'))   # gid 真实局号
            f_out.write(won.to_bytes(4, 'little'))
        rec.clear()   # 就地清空 (agent 持有同一列表引用; 重绑会静默丢失后续局)
    f_out.close()
    print(f"[蒸馏 v2] {args.games} 局 → {args.out}")
    print(f"[诊断] 决策 {CaptureAgent.n_decisions} | 保留 {CaptureAgent.n_kept} | "
          f"语义外 {CaptureAgent.n_oom} (跳过率 {100.0*(1-CaptureAgent.n_kept/max(1,CaptureAgent.n_decisions)):.1f}%)")


if __name__ == "__main__":
    main()
