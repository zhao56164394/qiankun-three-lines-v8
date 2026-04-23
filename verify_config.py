# -*- coding: utf-8 -*-
"""
verify_config.py — 配置一致性与数据新鲜度检查

启动 dashboard 前运行，或作为 CI 检查。
用法:
  python verify_config.py          # 检查并报告
  python verify_config.py --strict  # 检查失败则 exit(1)
"""
import os
import sys
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_config_consistency():
    """检查策略参数是否从单一来源导入，没有重复定义。"""
    errors = []

    from backtest_bt import config as cfg

    # 检查 backtest_capital.py 的常量是否来自 config
    import backtest_capital as bc

    # 日期对齐
    if cfg.BACKTEST_START != bc.YEAR_START:
        errors.append(f'BACKTEST_START({cfg.BACKTEST_START}) != YEAR_START({bc.YEAR_START})')
    if cfg.BACKTEST_END != bc.YEAR_END:
        errors.append(f'BACKTEST_END({cfg.BACKTEST_END}) != YEAR_END({bc.YEAR_END})')

    # 检查 backtest_8gua.py GUA_STRATEGY 结构 (v8.0 单一真相源)
    try:
        import backtest_8gua as b8
        required_keys = {'sell', 'active', 'pool_threshold', 'pool_depth',
                         'pool_days_min', 'pool_days_max'}
        forbidden_keys = {'trend_max', 'retail_max'}
        for gua, strat in b8.GUA_STRATEGY.items():
            missing = required_keys - set(strat.keys())
            if missing:
                errors.append(f'GUA_STRATEGY[{gua}] 缺少必填字段: {sorted(missing)}')
            leftover = forbidden_keys & set(strat.keys())
            if leftover:
                errors.append(f'GUA_STRATEGY[{gua}] 含已废弃字段: {sorted(leftover)}')
            if strat.get('pool_threshold') != -250:
                errors.append(f'GUA_STRATEGY[{gua}] pool_threshold 应统一为 -250 (当前 {strat.get("pool_threshold")})')
    except ImportError:
        pass

    # 检查 live/config.py 参数对齐 (live/qmt_strategy 兼容保留, 只检查共享参数)
    try:
        from live import config as lc
        for name in ['POOL_THRESHOLD', 'MAX_POSITIONS', 'DAILY_BUY_LIMIT',
                     'SKIP_HEXAGRAMS', 'INNER_SELL_METHOD']:
            cfg_val = getattr(cfg, name)
            lc_val = getattr(lc, name, None)
            if lc_val is not None and cfg_val != lc_val:
                errors.append(f'live/config.py {name} 不一致: config={cfg_val}, live={lc_val}')
    except ImportError:
        pass

    return errors


def check_data_freshness():
    """检查关键数据文件的日期覆盖是否一致。"""
    import pandas as pd
    errors = []
    warnings = []

    data_dir = os.path.join(os.path.dirname(__file__), 'data_layer', 'data')
    foundation_dir = os.path.join(data_dir, 'foundation')

    files_to_check = {
        'zz1000_daily.csv': os.path.join(data_dir, 'zz1000_daily.csv'),
        'market_bagua_daily.csv': os.path.join(foundation_dir, 'market_bagua_daily.csv'),
        'daily_bagua_sequence.csv': os.path.join(foundation_dir, 'daily_bagua_sequence.csv'),
        'daily_5d_scores.csv': os.path.join(foundation_dir, 'daily_5d_scores.csv'),
    }

    dates = {}
    for name, path in files_to_check.items():
        if not os.path.exists(path):
            errors.append(f'文件不存在: {name}')
            continue
        df = pd.read_csv(path, usecols=['date'], encoding='utf-8-sig')
        max_date = str(df['date'].max())
        dates[name] = max_date

    if dates:
        max_all = max(dates.values())
        for name, d in dates.items():
            if d < max_all:
                gap = (pd.Timestamp(max_all) - pd.Timestamp(d)).days
                if gap > 1:
                    warnings.append(f'{name} 落后 {gap} 天 (最新: {d}, 应为: {max_all})')

    return errors, warnings


def check_snapshot_naming():
    """检查快照文件是否使用了标准字段命名 (ren_gua/di_gua)。"""
    import json
    errors = []

    snap_path = os.path.join(os.path.dirname(__file__),
                             'data_layer', 'data', 'bagua_debug_baseline_snapshot.json')
    if not os.path.exists(snap_path):
        return ['baseline snapshot 不存在']

    with open(snap_path, 'r', encoding='utf-8') as f:
        snap = json.load(f)

    old_names = {'market_gua', 'stock_gua', 'zz_gua', 'stk_gua', 'market_name', 'stock_name'}
    for gua, payload in snap.get('payloads', {}).items():
        for key in ['matrix_df', 'detail_signals', 'detail_trades']:
            records = payload.get(key, [])
            if records and isinstance(records, list) and len(records) > 0:
                found_old = old_names & set(records[0].keys())
                if found_old:
                    errors.append(f'快照 {gua}/{key} 含旧字段名: {found_old}')

    return errors


def main():
    strict = '--strict' in sys.argv
    all_ok = True

    print('=' * 60)
    print('配置一致性 & 数据新鲜度检查')
    print('=' * 60)

    # 1. 配置一致性
    print('\n[1] 配置一致性...')
    config_errors = check_config_consistency()
    if config_errors:
        all_ok = False
        for e in config_errors:
            print(f'  ✗ {e}')
    else:
        print('  ✓ 所有配置参数一致')

    # 2. 数据新鲜度
    print('\n[2] 数据新鲜度...')
    data_errors, data_warnings = check_data_freshness()
    if data_errors:
        all_ok = False
        for e in data_errors:
            print(f'  ✗ {e}')
    if data_warnings:
        for w in data_warnings:
            print(f'  ⚠ {w}')
    if not data_errors and not data_warnings:
        print('  ✓ 所有数据文件日期一致')

    # 3. 快照字段命名
    print('\n[3] 快照字段命名...')
    snap_errors = check_snapshot_naming()
    if snap_errors:
        all_ok = False
        for e in snap_errors:
            print(f'  ✗ {e}')
    else:
        print('  ✓ 快照使用标准命名 (ren_gua/di_gua)')

    print('\n' + '=' * 60)
    if all_ok:
        print('✓ 全部检查通过')
    else:
        print('✗ 存在问题，请修复后重试')
        if strict:
            sys.exit(1)


if __name__ == '__main__':
    main()
