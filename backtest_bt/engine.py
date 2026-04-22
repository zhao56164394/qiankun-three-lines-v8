# -*- coding: utf-8 -*-
"""
engine.py — Backtrader 引擎组装

组装 Cerebro: 数据源 + 策略 + 佣金 + 分析器
"""
import backtrader as bt
import backtrader.analyzers as btanalyzers
import time

from .config import INIT_CAPITAL, BACKTEST_START, BACKTEST_END
from .commission import CNStockCommission, ZeroCommission
from .feeds import create_zz1000_feed, load_zz1000_df
from .feeds import StockData
from .strategy import QKSXStrategy
from .scanner import scan_candidates, load_seg_events, build_daily_512_snapshot

import pandas as pd


def build_cerebro(candidates=None, start_date=None, end_date=None,
                  commission=True, verbose=False, max_positions=None,
                  daily_buy_limit=None, position_mode=None):
    """组装 Backtrader Cerebro 引擎

    Args:
        candidates: dict {code: DataFrame} 候选股, 为 None 时自动扫描
        start_date: 回测起始日期
        end_date: 回测结束日期
        commission: True=A股佣金, False=零佣金
        verbose: 是否打印详细日志
        max_positions: 最大持仓数 (覆盖默认)
        daily_buy_limit: 每日买入上限 (覆盖默认)
        position_mode: 仓位模式 (覆盖默认)

    Returns:
        cerebro: 配置好的 Cerebro 实例
    """
    start_date = start_date or BACKTEST_START
    end_date = end_date or BACKTEST_END

    cerebro = bt.Cerebro()

    # === 资金 ===
    cerebro.broker.setcash(INIT_CAPITAL)

    # === 佣金 ===
    if commission:
        cerebro.broker.addcommissioninfo(CNStockCommission())
    else:
        cerebro.broker.addcommissioninfo(ZeroCommission())

    # === 数据源: datas[0] = 中证1000 (哨兵) ===
    print("加载中证1000数据...")
    zz_df = load_zz1000_df(start_date, end_date)  # 只读一次, 复用
    zz_feed = create_zz1000_feed(start_date, end_date, zz_df=zz_df)
    cerebro.adddata(zz_feed)
    print(f"  中证1000: 数据已加载")

    # === 数据源: datas[1:] = 个股 ===
    if candidates is None:
        candidates = scan_candidates(start_date, end_date, verbose=True)

    print(f"\n加载 {len(candidates)} 只候选股数据...")
    t0 = time.time()
    loaded = 0
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    for code, df in candidates.items():
        # df 已经是处理好的 DataFrame (index=DatetimeIndex)
        # 日期过滤 (scanner 可能返回超出范围的数据)
        df_filtered = df[(df.index >= start_ts) & (df.index <= end_ts)]

        if len(df_filtered) < 20:  # 数据太少跳过
            continue

        feed = StockData(dataname=df_filtered, name=code)
        # 默认不绘图 (几百只不可能全画)
        feed.plotinfo.plot = False
        cerebro.adddata(feed)
        loaded += 1

    elapsed = time.time() - t0
    print(f"  加载完成: {loaded} 只, 耗时 {elapsed:.1f}s")

    # === 构建512快照 (带缓存) ===
    print("\n构建512分级快照...")
    t0 = time.time()
    try:
        trade_dates = sorted(zz_df.index.strftime('%Y-%m-%d').tolist())
        snapshots_512 = _load_512_snapshots(start_date, end_date, trade_dates)
        elapsed_512 = time.time() - t0
        print(f"  512快照: {len(snapshots_512)} 个交易日, 耗时 {elapsed_512:.1f}s")
    except FileNotFoundError as e:
        print(f"  [警告] 512数据不可用: {e}, 将跳过分级过滤")
        snapshots_512 = {}

    # === 策略 ===
    strat_kwargs = {'verbose': verbose, 'snapshots_512': snapshots_512}
    if max_positions is not None:
        strat_kwargs['max_positions'] = max_positions
    if daily_buy_limit is not None:
        strat_kwargs['daily_buy_limit'] = daily_buy_limit
    if position_mode is not None:
        strat_kwargs['position_mode'] = position_mode

    cerebro.addstrategy(QKSXStrategy, **strat_kwargs)

    # === 分析器 ===
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe',
                        riskfreerate=0.02)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
    cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')

    return cerebro


def run_backtest(cerebro):
    """运行回测, 返回策略实例"""
    print(f"\n{'='*60}")
    print(f"开始回测...")
    print(f"  初始资金: {cerebro.broker.getvalue():,.0f}")
    print(f"  数据源: {len(cerebro.datas)} 个 (1个哨兵 + {len(cerebro.datas)-1}只个股)")
    print(f"{'='*60}")

    t0 = time.time()
    results = cerebro.run()
    elapsed = time.time() - t0

    strat = results[0]
    print(f"\n回测耗时: {elapsed:.1f}s")

    # 打印分析器结果
    print_analysis(strat)

    return strat


def print_analysis(strat):
    """打印分析器结果"""
    print(f"\n{'='*60}")
    print("分析器结果:")
    print(f"{'='*60}")

    # 夏普比率
    try:
        sharpe = strat.analyzers.sharpe.get_analysis()
        sr = sharpe.get('sharperatio', None)
        if sr is not None:
            print(f"  夏普比率: {sr:.3f}")
    except Exception:
        pass

    # 最大回撤
    try:
        dd = strat.analyzers.drawdown.get_analysis()
        print(f"  最大回撤: {dd.max.drawdown:.1f}%")
        print(f"  最长回撤: {dd.max.len} 天")
    except Exception:
        pass

    # SQN
    try:
        sqn = strat.analyzers.sqn.get_analysis()
        sqn_val = sqn.get('sqn', None)
        if sqn_val is not None:
            print(f"  SQN: {sqn_val:.2f}")
    except Exception:
        pass

    # 交易统计
    try:
        ta = strat.analyzers.trades.get_analysis()
        total = ta.total.total if hasattr(ta, 'total') else 0
        if total > 0:
            won = ta.won.total if hasattr(ta.won, 'total') else 0
            lost = ta.lost.total if hasattr(ta.lost, 'total') else 0
            print(f"  交易笔数: {total} (盈利={won}, 亏损={lost})")
            if total > 0:
                print(f"  胜率: {won/total*100:.1f}%")
    except Exception:
        pass


# ============================================================
# 512快照缓存
# ============================================================
import os
import pickle
import hashlib

def _get_512_cache_path(start_date, end_date):
    """生成512快照缓存文件路径"""
    from .config import SEG_EVENTS_PATH
    # 缓存 key = 日期区间 + seg_events文件修改时间
    seg_mtime = os.path.getmtime(SEG_EVENTS_PATH) if os.path.exists(SEG_EVENTS_PATH) else 0
    key = f"512_{start_date}_{end_date}_{seg_mtime}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'data_layer', 'data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'snap512_{h}.pkl')


def _load_512_snapshots(start_date, end_date, trade_dates):
    """加载512快照, 优先使用缓存

    Args:
        start_date: 起始日期
        end_date: 结束日期
        trade_dates: 交易日列表

    Returns:
        dict: {date_str: {combo: mean_excess_ret}}
    """
    cache_path = _get_512_cache_path(start_date, end_date)

    # 尝试加载缓存
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                snapshots = pickle.load(f)
            print(f"  [512] 使用缓存: {len(snapshots)} 个交易日")
            return snapshots
        except Exception as e:
            print(f"  [512] 缓存加载失败: {e}, 重新构建")

    # 缓存未命中, 重新计算
    print(f"  [512] 缓存未命中, 加载段首事件...")
    seg_events = load_seg_events()
    print(f"  [512] 构建快照 ({len(trade_dates)} 个交易日)...")
    snapshots = build_daily_512_snapshot(seg_events, trade_dates)

    # 保存缓存
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(snapshots, f)
        cache_mb = os.path.getsize(cache_path) / 1024 / 1024
        print(f"  [512] 缓存已保存: {cache_path} ({cache_mb:.1f}MB)")
    except Exception as e:
        print(f"  [512] 缓存保存失败: {e}")

    return snapshots
