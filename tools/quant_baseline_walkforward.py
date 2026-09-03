#!/usr/bin/env python3
"""
量化高手基线策略 Walk-Forward 多窗口稳健性验证
------------------------------------------------
纯规则策略（无生命体演化），用于回答一个前置问题：
「在当前特征（20/60日均线剪刀差 + ATR 波动率倒数加权 + 截面强弱多空）下，
  这批 43 品种日线数据是否存在可被跨多个不重叠时间窗口复现的正向 alpha？」

如果这一步都无法在多数窗口中获得正夏普，那么让生命体在同样的特征空间里
演化黑箱策略也大概率只是过拟合训练窗口的噪声，而非学到真实规律。

方法：
1. 周频调仓（每周最后一个交易日重新计算目标仓位，减少高频换手磨损）
2. 截面强弱：按 MA5-MA20 剪刀差排序，做多前 20%，做空后 20%
3. 仓位按 ATR（日内振幅）倒数加权，总杠杆封顶 0.8
4. 换手死区：仓位变化 < 0.10 不换仓，进一步抑制磨损
5. Walk-forward：把 2007-01 至 2026-09 切成多个不重叠的 2 年窗口，
   分别统计每个窗口的夏普/收益/回撤，要求多数窗口为正夏普才算「有效」

不训练/不写入任何生命体检查点，不改动 include/kun/cellular/ 底座。
"""
import os
import glob
import csv
import math
import statistics
import json

BASE_DIR = "/home/caixuf/code/kunquant/data/history"
OUT_REPORT = "/home/caixuf/code/kun-cellular/checkpoints/quant_baseline_walkforward_report.json"


def load_assets():
    assets = {}
    for path in sorted(glob.glob(os.path.join(BASE_DIR, "*.csv"))):
        sym = os.path.basename(path)[:-4]
        rows = []
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 7:
                    continue
                try:
                    d, o, h, l, c, v = row[1], float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])
                except ValueError:
                    continue
                if o <= 0 or c <= 0:
                    continue
                rows.append((d, o, h, l, c, v))
        rows.sort(key=lambda x: x[0])
        if rows:
            assets[sym] = rows
    return assets


def precompute(assets):
    precomp = {}
    for sym, rows in assets.items():
        by_date = {}
        closes = []
        for d, o, h, l, c, v in rows:
            closes.append(c)
            by_date[d] = (o, h, l, c, v, len(closes) - 1)
        precomp[sym] = (by_date, closes)
    return precomp


def is_week_end(dates, i):
    """判断 dates[i] 是否是该周最后一个交易日（下一个交易日跨入新的一周或已是最后一天）"""
    if i == len(dates) - 1:
        return True
    import datetime
    d0 = datetime.date.fromisoformat(dates[i])
    d1 = datetime.date.fromisoformat(dates[i + 1])
    return d1.isocalendar()[1] != d0.isocalendar()[1] or d1.year != d0.year


def feature_ma_diff(precomp, sym, dt):
    by_date, closes = precomp[sym]
    if dt not in by_date:
        return None
    o, h, l, c, v, idx = by_date[dt]
    if idx < 20:
        return None
    ma5 = sum(closes[idx - 4:idx + 1]) / 5.0
    ma20 = sum(closes[idx - 19:idx + 1]) / 20.0
    return (ma5 - ma20) / (c + 1e-8)


def run_backtest(assets, precomp, all_dates, start_date, end_date, leverage=0.8,
                  sleeve_frac=0.2, rebalance_dead_zone=0.10, fee_bp=0.00015):
    dates = [d for d in all_dates if start_date <= d <= end_date]
    if len(dates) < 60:
        return None

    positions = {s: 0.0 for s in assets}
    cap = 1_000_000.0
    peak = cap
    max_dd = 0.0
    daily_rets = []
    trades = 0
    target = {s: 0.0 for s in assets}

    for i in range(len(dates) - 1):
        dt = dates[i]
        nxt = dates[i + 1]

        # 每周最后一个交易日重新计算目标仓位（周频调仓，减少磨损）
        if is_week_end(dates, i):
            ranked = []
            for s in assets:
                sig = feature_ma_diff(precomp, s, dt)
                if sig is None:
                    continue
                ranked.append((sig, s))
            if len(ranked) >= 10:
                ranked.sort()
                n = len(ranked)
                sleeve = max(3, int(n * sleeve_frac))
                picks = []
                wsum = 0.0
                for k in range(sleeve):
                    sig, s = ranked[n - 1 - k]
                    by_date, closes = precomp[s]
                    o, h, l, c, v, idx = by_date[dt]
                    vol = max(0.01, (h - l) / c)
                    w = 1.0 / vol
                    picks.append((s, w))
                    wsum += w
                for k in range(sleeve):
                    sig, s = ranked[k]
                    by_date, closes = precomp[s]
                    o, h, l, c, v, idx = by_date[dt]
                    vol = max(0.01, (h - l) / c)
                    w = 1.0 / vol
                    picks.append((s, -w))
                    wsum += w
                scale = leverage / wsum if wsum > 1e-6 else 0.0
                new_target = {s: 0.0 for s in assets}
                for s, w in picks:
                    new_target[s] = w * scale
                target = new_target

        # 换手死区 + 持仓收益结算（每日结算，避免只统计周末造成的夏普偏差）
        day_pnl = 0.0
        for s in assets:
            by_date, closes = precomp[s]
            desired = target[s]
            delta = abs(desired - positions[s])
            if delta > rebalance_dead_zone:
                trades += 1
                cap -= cap * delta * fee_bp
                positions[s] = desired
            ret = 0.0
            if dt in by_date and nxt in by_date:
                o1, h1, l1, c1, v1, i1 = by_date[dt]
                o2, h2, l2, c2, v2, i2 = by_date[nxt]
                ret = (c2 - c1) / (c1 + 1e-8)
            day_pnl += positions[s] * ret * cap

        before = cap
        cap += day_pnl
        daily_rets.append(day_pnl / before if before > 0 else -0.1)
        peak = max(peak, cap)
        if peak > 0:
            max_dd = max(max_dd, (peak - cap) / peak)

    if len(daily_rets) < 20:
        return None
    mean = sum(daily_rets) / len(daily_rets)
    std = statistics.pstdev(daily_rets) if len(daily_rets) > 1 else 0.0
    sharpe = (mean / std) * math.sqrt(252) if std > 1e-12 else 0.0
    pnl = (cap - 1_000_000.0) / 1_000_000.0
    return {
        "start": dates[0], "end": dates[-1], "n_days": len(daily_rets),
        "sharpe": sharpe, "pnl": pnl, "max_dd": max_dd, "trades": trades,
        "final_capital": cap,
    }


def main():
    print("=" * 70)
    print("  量化高手基线策略 Walk-Forward 多窗口稳健性验证 (纯规则，无生命体)")
    print("=" * 70)
    assets = load_assets()
    precomp = precompute(assets)
    all_dates = sorted(set().union(*[set(d for d, *_ in rows) for rows in assets.values()]))
    print(f"  ↳ 加载 {len(assets)} 品种，{all_dates[0]} 至 {all_dates[-1]}，共 {len(all_dates)} 交易日\n")

    # 划分不重叠的 2 年 walk-forward 窗口，从 2007 起（前两年用于预热 MA20）
    windows = []
    year = 2007
    while year < 2026:
        windows.append((f"{year}-01-01", f"{year+1}-12-31"))
        year += 2

    results = []
    pos_count = 0
    for start, end in windows:
        r = run_backtest(assets, precomp, all_dates, start, end)
        if r is None:
            continue
        results.append(r)
        tag = "✓ 正收益" if r["pnl"] > 0 else "✗ 负收益"
        if r["sharpe"] > 0:
            pos_count += 1
        print(f"  窗口 {r['start']} ~ {r['end']} | 夏普 {r['sharpe']:+.2f} | "
              f"收益 {r['pnl']*100:+6.1f}% | 回撤 {r['max_dd']*100:5.1f}% | "
              f"换仓 {r['trades']:5d} 次 | {tag}")

    total_windows = len(results)
    print("\n" + "-" * 70)
    print(f"  统计: {total_windows} 个不重叠 2 年窗口中，{pos_count} 个窗口夏普为正 "
          f"({pos_count/total_windows*100:.0f}%)" if total_windows else "  无有效窗口")

    # 全周期 2016-2026 样本外整体表现（与之前生命体训练报告的样本外区间对齐，便于横向对比）
    full_oos = run_backtest(assets, precomp, all_dates, "2016-01-01", "2026-09-01")
    if full_oos:
        print(f"\n  [全周期样本外 2016-2026] 夏普 {full_oos['sharpe']:+.3f} | "
              f"收益 {full_oos['pnl']*100:+.1f}% | 回撤 {full_oos['max_dd']*100:.1f}% | "
              f"换仓 {full_oos['trades']} 次 | 期末资金 {full_oos['final_capital']:.0f}")

    verdict = "PASS" if total_windows > 0 and pos_count / total_windows >= 0.6 and full_oos and full_oos["sharpe"] > 0.3 else "FAIL"
    print(f"\n  基线门禁判定 (>=60% 窗口正夏普 且 全周期样本外夏普>0.3): {verdict}")
    if verdict == "FAIL":
        print("  → 结论：当前特征空间（MA剪刀差 + ATR权重 + 截面多空）在此数据集上")
        print("    不具备跨窗口稳健的 alpha。继续用生命体演化同样的特征只会拟合噪声，")
        print("    不建议在不改变特征/频率设计的前提下再投入生命体训练资源。")
    else:
        print("  → 结论：基线具备跨窗口稳健性，可以在此特征基础上让生命体演化非线性组合方式。")

    report = {
        "trainer": "tools/quant_baseline_walkforward.py",
        "strategy": "rule_based_no_organism",
        "n_assets": len(assets),
        "rebalance": "weekly",
        "windows": results,
        "positive_window_ratio": (pos_count / total_windows) if total_windows else 0.0,
        "full_oos_2016_2026": full_oos,
        "verdict": verdict,
        "base_untouched": True,
        "adas_untouched": True,
    }
    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已写入: {OUT_REPORT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
