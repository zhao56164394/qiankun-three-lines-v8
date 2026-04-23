# -*- coding: utf-8 -*-
"""
数据层 - 增量更新脚本

每日收盘后运行,只追加当天新数据,不全量重算。
全量重算仍用 prepare_zz1000.py / prepare_stocks.py。

用法:
  python data_layer/update_daily.py                  # 更新所有
  python data_layer/update_daily.py --zz1000-only    # 只更新中证1000
  python data_layer/update_daily.py --stock 000001   # 只更新单只股票

数据源:
  中证1000: e:/A/A股数据_zip/指数/指数_日_kline.zip
  个股:     e:/A/A股数据_zip/daily_qfq.zip
"""
import sys
import os
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import zipfile
from datetime import datetime
from strategy.indicator import calc_trend_line, calc_retail_line, calc_main_force_line
from bagua_engine import calc_xiang_gua


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
ZZ1000_PATH = os.path.join(DATA_DIR, 'zz1000_daily.csv')
STOCKS_DIR = os.path.join(DATA_DIR, 'stocks')

INDEX_ZIP = 'E:/BaiduSyncdisk/A股数据_zip/指数/指数_日_kline.zip'
STOCK_ZIP = 'E:/BaiduSyncdisk/A股数据_zip/daily_qfq.zip'
STOCK_LIST = 'E:/BaiduSyncdisk/A股数据_zip/股票列表.csv'

# 每日增量CSV数据源（ZIP不够新时用这些补充）
STOCK_DAILY_CSV_ROOT = 'E:/BaiduSyncdisk/A股数据_每日指标/增量数据/每日指标'
INDEX_DAILY_CSV_ROOT = 'E:/BaiduSyncdisk/指数数据/增量数据/指数日线行情'


def recalc_tail_indicators(df, col_close='close', col_high='high', col_low='low',
                            n_tail=5):
    """
    对DataFrame的最后n_tail行重新计算指标

    由于MA/EMA有记忆效应，无法只计算最后1天。
    策略: 保留全部历史用于计算，只更新尾部结果。
    """
    closes = df[col_close].values.astype(float)
    highs = df[col_high].values.astype(float)
    lows = df[col_low].values.astype(float)

    # 全量重算指标 (因为EMA有记忆)
    trend = calc_trend_line(closes, highs, lows)
    main_force = calc_main_force_line(closes)

    # 更新最后 n_tail 行
    start = max(0, len(df) - n_tail)
    df.loc[start:, 'trend'] = trend[start:]
    df.loc[start:, 'main_force'] = main_force[start:]

    # 如果有 retail 列
    if 'retail' in df.columns:
        retail = calc_retail_line(closes)
        df.loc[start:, 'retail'] = retail[start:]

    # 更新象卦
    _update_gua_column(df, closes, highs, lows, n_tail)

    return df


def _update_gua_column(df, closes, highs, lows, n_tail=5):
    """
    更新象卦列 (统一逻辑, 大盘和个股通用)

    象卦: 趋势线(250日) >= 50 / 趋势线20日变化 > 0 / 主力线20日变化MA10 > 0
    """
    # 全量计算象卦 (因为MA有记忆效应)
    gua_list, _, _, _ = calc_xiang_gua(closes, highs, lows)

    # 只更新最后 n_tail 行
    n = len(df)
    for i in range(max(0, n - n_tail), n):
        df.at[i, 'gua'] = gua_list[i]


# ============================================================
# 每日CSV补充数据源
# ============================================================
def _find_daily_csv_dates(root, after_date):
    """扫描每日增量CSV目录,返回 > after_date 的所有日期文件路径"""
    results = []
    after_int = int(after_date.replace('-', ''))
    for month_dir in sorted(os.listdir(root)):
        month_path = os.path.join(root, month_dir)
        if not os.path.isdir(month_path):
            continue
        for fname in sorted(os.listdir(month_path)):
            if not fname.endswith('.csv'):
                continue
            date_str = fname[:8]
            if date_str.isdigit() and int(date_str) > after_int:
                results.append(os.path.join(month_path, fname))
    return results


def _supplement_zz1000_from_csv(combined, last_date):
    """从每日指数CSV补充中证1000数据"""
    if not os.path.exists(INDEX_DAILY_CSV_ROOT):
        return combined, 0

    csv_files = _find_daily_csv_dates(INDEX_DAILY_CSV_ROOT, last_date)
    if not csv_files:
        return combined, 0

    new_rows = []
    for fpath in csv_files:
        df = pd.read_csv(fpath, encoding='utf-8-sig')
        zz = df[df.iloc[:, 0].astype(str) == '000852.SH']
        if len(zz) == 0:
            continue
        row = zz.iloc[0]
        date_val = str(int(row.iloc[1]))
        date_fmt = f'{date_val[:4]}-{date_val[4:6]}-{date_val[6:8]}'
        new_rows.append({
            'date': date_fmt,
            'close': float(row.iloc[3]),   # 收盘点位
            'high': float(row.iloc[4]),    # 最高点位
            'low': float(row.iloc[5]),     # 最低点位
            'trend': np.nan,
            'main_force': np.nan,
            'gua': '',
        })

    if not new_rows:
        return combined, 0

    new_df = pd.DataFrame(new_rows)
    for col in combined.columns:
        if col not in new_df.columns:
            new_df[col] = ''
    combined = pd.concat([combined, new_df[combined.columns]], ignore_index=True)
    return combined, len(new_rows)


def _supplement_stock_from_csv(combined, code, last_date):
    """从每日个股CSV补充单只股票数据 (使用预加载缓存)"""
    if not hasattr(_supplement_stock_from_csv, '_cache'):
        return combined, 0

    cache = _supplement_stock_from_csv._cache
    code_with_suffix = code + '.SZ' if code.startswith(('00', '30')) else code + '.SH'

    new_rows = []
    for date_fmt, df in cache.items():
        match = df[df['code'] == code_with_suffix]
        if len(match) == 0:
            continue
        row = match.iloc[0]
        new_rows.append({
            'date': date_fmt,
            'open': float(row['open']),
            'close': float(row['close']),
            'high': float(row['high']),
            'low': float(row['low']),
            'trend': np.nan,
            'retail': np.nan,
            'main_force': np.nan,
            'gua': '',
        })

    if not new_rows:
        return combined, 0

    new_df = pd.DataFrame(new_rows)
    for col in combined.columns:
        if col not in new_df.columns:
            new_df[col] = ''
    combined = pd.concat([combined, new_df[combined.columns]], ignore_index=True)
    return combined, len(new_rows)


def _preload_daily_csvs(last_date):
    """预加载所有需要的每日CSV到内存 (避免每只股票都读一次文件)"""
    if not os.path.exists(STOCK_DAILY_CSV_ROOT):
        return

    csv_files = _find_daily_csv_dates(STOCK_DAILY_CSV_ROOT, last_date)
    if not csv_files:
        return

    cache = {}
    for fpath in csv_files:
        fname = os.path.basename(fpath)
        date_str = fname[:8]
        date_fmt = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}'
        df = pd.read_csv(fpath, encoding='utf-8-sig',
                         usecols=['股票代码', '交易日期', '开盘价', '最高价', '最低价', '收盘价'],
                         dtype={'股票代码': str})
        df = df.rename(columns={'股票代码': 'code', '开盘价': 'open', '最高价': 'high',
                                '最低价': 'low', '收盘价': 'close'})
        cache[date_fmt] = df
        print(f'  预加载CSV: {date_fmt} ({len(df)} 只)')

    _supplement_stock_from_csv._cache = cache


# ============================================================
# 中证1000 增量更新
# ============================================================
def update_zz1000():
    """增量更新中证1000数据"""
    print('=== 增量更新中证1000 ===')

    if not os.path.exists(ZZ1000_PATH):
        print(f'  !! 缓存不存在, 请先运行: python data_layer/prepare_zz1000.py')
        return False

    # 读取现有缓存
    existing = pd.read_csv(ZZ1000_PATH, encoding='utf-8-sig')
    last_date = str(existing['date'].iloc[-1])
    print(f'  现有数据最后日期: {last_date}')

    # 从zip加载源数据
    with zipfile.ZipFile(INDEX_ZIP) as zf:
        names = [n for n in zf.namelist() if '000852' in n]
        if not names:
            print('  !! 找不到000852数据文件')
            return False
        with zf.open(names[0]) as f:
            raw = pd.read_csv(f, encoding='utf-8-sig')

    raw_dates = raw.iloc[:, 0].astype(str).values
    raw_closes = raw.iloc[:, 4].values.astype(float)
    raw_highs = raw.iloc[:, 5].values.astype(float)
    raw_lows = raw.iloc[:, 6].values.astype(float)

    # 找到新增日期
    new_mask = raw_dates > last_date
    n_new = new_mask.sum()

    if n_new == 0:
        print(f'  ZIP无新增 (最新: {raw_dates[-1]})')
        # ZIP没新数据,但每日CSV可能有
        combined = existing.copy()
        last = str(existing['date'].iloc[-1])
        combined, n_csv = _supplement_zz1000_from_csv(combined, last)
        if n_csv > 0:
            print(f'  CSV补充: +{n_csv} 天')
            combined = recalc_tail_indicators(
                combined, col_close='close', col_high='high', col_low='low',
                n_tail=n_csv + 5)
            combined.to_csv(ZZ1000_PATH, index=False, encoding='utf-8-sig')
            print(f'  已保存, 范围: {combined["date"].iloc[0]} ~ {combined["date"].iloc[-1]}')
        else:
            print(f'  无新增数据')
        return True

    print(f'  新增: {n_new} 天 ({raw_dates[new_mask][0]} ~ {raw_dates[new_mask][-1]})')

    # 追加新行 (先用NaN占位)
    new_rows = []
    for i in np.where(new_mask)[0]:
        new_rows.append({
            'date': raw_dates[i],
            'close': raw_closes[i],
            'high': raw_highs[i],
            'low': raw_lows[i],
            'trend': np.nan,
            'main_force': np.nan,
            'gua': '',
        })

    new_df = pd.DataFrame(new_rows)

    # 合并并添加可能缺失的列
    for col in existing.columns:
        if col not in new_df.columns:
            new_df[col] = ''

    combined = pd.concat([existing, new_df[existing.columns]], ignore_index=True)

    # 重新计算尾部指标 (需要全部历史做warmup)
    print('  重新计算指标...')
    combined = recalc_tail_indicators(
        combined, col_close='close', col_high='high', col_low='low',
        n_tail=n_new + 5)

    # 保存前检查是否还有更新的每日CSV数据
    current_last = str(combined['date'].iloc[-1])
    combined, n_csv = _supplement_zz1000_from_csv(combined, current_last)
    if n_csv > 0:
        print(f'  CSV补充: +{n_csv} 天')
        combined = recalc_tail_indicators(
            combined, col_close='close', col_high='high', col_low='low',
            n_tail=n_csv + 5)

    # 保存
    combined.to_csv(ZZ1000_PATH, index=False, encoding='utf-8-sig')
    print(f'  已保存: {ZZ1000_PATH}')
    print(f'  数据范围: {combined["date"].iloc[0]} ~ {combined["date"].iloc[-1]} ({len(combined)}行)')
    return True


# ============================================================
# 个股增量更新
# ============================================================
def update_single_stock(code, zf=None):
    """增量更新单只个股数据"""
    cache_path = os.path.join(STOCKS_DIR, f'{code}.csv')

    # 打开zip
    close_zf = False
    if zf is None:
        zf = zipfile.ZipFile(STOCK_ZIP)
        close_zf = True

    try:
        fname = f'{code}_daily_qfq.csv'
        if fname not in zf.namelist():
            return False

        with zf.open(fname) as f:
            raw = pd.read_csv(f, encoding='utf-8-sig', dtype={'日期': str})

        # 只保留2014-06-01之后
        raw = raw[raw['日期'] >= '2014-06-01'].reset_index(drop=True)
        if len(raw) < 60:
            return False

        # 检查现有缓存
        if os.path.exists(cache_path):
            existing = pd.read_csv(cache_path, encoding='utf-8-sig')
            last_date = str(existing['date'].iloc[-1])

            # 找新增数据
            new_mask = raw['日期'].values > last_date
            n_new = new_mask.sum()

            if n_new == 0:
                # ZIP没新数据, 但每日CSV可能有
                combined = existing.copy()
                combined, n_csv = _supplement_stock_from_csv(combined, code, last_date)
                if n_csv > 0:
                    combined = recalc_tail_indicators(combined, n_tail=n_csv + 5)
                    combined.to_csv(cache_path, index=False, encoding='utf-8-sig')
                return True

            # 追加新行
            new_raw = raw[new_mask].reset_index(drop=True)
            new_rows = pd.DataFrame({
                'date': new_raw['日期'].values,
                'open': new_raw['开盘'].values.astype(float),
                'close': new_raw['收盘'].values.astype(float),
                'high': new_raw['最高'].values.astype(float),
                'low': new_raw['最低'].values.astype(float),
                'trend': np.nan,
                'retail': np.nan,
                'main_force': np.nan,
                'gua': '',
            })

            combined = pd.concat([existing, new_rows[existing.columns]],
                                 ignore_index=True)

        else:
            # 首次创建
            opens = raw['开盘'].values.astype(float)
            closes = raw['收盘'].values.astype(float)
            highs = raw['最高'].values.astype(float)
            lows = raw['最低'].values.astype(float)

            if (closes <= 0).any():
                return False

            combined = pd.DataFrame({
                'date': raw['日期'].values,
                'open': opens, 'close': closes,
                'high': highs, 'low': lows,
                'trend': np.nan, 'retail': np.nan, 'main_force': np.nan,
                'gua': '',
            })
            n_new = len(combined)

        # 重新计算指标
        combined = recalc_tail_indicators(
            combined, n_tail=n_new + 5)

        # ZIP更新后再检查每日CSV是否有更新的数据
        current_last = str(combined['date'].iloc[-1])
        combined, n_csv = _supplement_stock_from_csv(combined, code, current_last)
        if n_csv > 0:
            combined = recalc_tail_indicators(combined, n_tail=n_csv + 5)

        # 保存
        combined.to_csv(cache_path, index=False, encoding='utf-8-sig')
        return True

    except Exception as e:
        print(f'  错误 {code}: {e}')
        return False

    finally:
        if close_zf:
            zf.close()


def update_all_stocks():
    """增量更新全市场个股"""
    print('\n=== 增量更新全市场个股 ===')

    # 加载股票列表
    df_list = pd.read_csv(STOCK_LIST, encoding='utf-8-sig', dtype=str)
    active_codes = set(df_list['股票代码'].values)

    os.makedirs(STOCKS_DIR, exist_ok=True)

    # 预加载每日CSV补充数据 (比ZIP更新的日期)
    # 先找一个股票的最后日期作为基准
    sample = os.path.join(STOCKS_DIR, '000001.csv')
    if os.path.exists(sample):
        sample_df = pd.read_csv(sample, usecols=['date'], dtype={'date': str})
        base_date = str(sample_df['date'].iloc[-1])
        print(f'  当前数据最后日期: {base_date}')
        _preload_daily_csvs(base_date)
    else:
        print('  首次运行, 跳过CSV补充')

    success = 0
    skip = 0
    fail = 0

    try:
        with zipfile.ZipFile(STOCK_ZIP) as zf:
            all_files = zf.namelist()
            all_codes = [f.replace('_daily_qfq.csv', '') for f in all_files
                         if f.endswith('_daily_qfq.csv')
                         and f.startswith(('00', '60', '30', '68'))]
            all_codes = [c for c in all_codes if c in active_codes]

            print(f'  待更新: {len(all_codes)} 只')

            for i, code in enumerate(all_codes):
                if (i + 1) % 200 == 0:
                    print(f'  进度: {i+1}/{len(all_codes)} ({success}更新 {skip}跳过)')

                result = update_single_stock(code, zf)
                if result:
                    success += 1
                else:
                    fail += 1
    finally:
        if hasattr(_supplement_stock_from_csv, '_cache'):
            del _supplement_stock_from_csv._cache

    print(f'\n  完成! 更新{success}只, 失败{fail}只')

    return success, fail


def _clear_pkl_cache(pattern='*'):
    """清除 backtest_capital.py 的 pkl 缓存"""
    for f in glob.glob(os.path.join(DATA_DIR, f'_cache_{pattern}.pkl')):
        os.remove(f)
        print(f'  已清除缓存: {os.path.basename(f)}')


# ============================================================
# 入口
# ============================================================
def _rebuild_baseline_snapshot_safe():
    """重建 dashboard 裸跑快照；失败不影响数据更新本身。"""
    try:
        import rebuild_baseline_snapshot as rbs
        print('\n  重建 dashboard 裸跑快照...')
        rbs.main(verbose=True)
    except Exception as e:
        print(f'  !! 裸跑快照重建失败: {e}')


def update_all():
    """更新所有数据"""
    print('=' * 60)
    print(f'  数据增量更新 — {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('=' * 60)

    ok1 = update_zz1000()
    if ok1:
        success, fail = update_all_stocks()
        # 数据更新后清除pkl缓存，下次回测自动重建
        print('\n  清除回测缓存...')
        _clear_pkl_cache()
        _rebuild_baseline_snapshot_safe()
    else:
        print('\n  !! 中证1000更新失败, 跳过个股更新')

    print('\n' + '=' * 60)
    print('  更新完成')
    print('=' * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='数据增量更新')
    parser.add_argument('--zz1000-only', action='store_true',
                        help='只更新中证1000')
    parser.add_argument('--stock', type=str, default='',
                        help='只更新指定股票代码')
    args = parser.parse_args()

    if args.stock:
        print(f'更新单只股票: {args.stock}')
        result = update_single_stock(args.stock)
        print(f'结果: {"成功" if result else "失败"}')
        if result:
            _clear_pkl_cache('stocks')
    elif args.zz1000_only:
        update_zz1000()
        _clear_pkl_cache('zz1000*')
    else:
        update_all()
