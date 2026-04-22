# -*- coding: utf-8 -*-
"""
数据层 - 通过 miniQMT (xtdata) 增量更新

替代 update_daily.py 的 zip 数据源，直接从 QMT 获取日线数据。
需要 QMT 客户端运行中。

用法:
  python data_layer/update_xtdata.py                  # 增量更新所有
  python data_layer/update_xtdata.py --zz1000-only    # 只更新中证1000
  python data_layer/update_xtdata.py --stock 000001   # 只更新单只股票
  python data_layer/update_xtdata.py --full-download   # 全量下载(首次建库)
"""
import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.update_daily import (
    recalc_tail_indicators, _clear_pkl_cache,
    DATA_DIR, ZZ1000_PATH, STOCKS_DIR,
)

# xtdata 延迟导入
xtdata = None

FULL_DOWNLOAD_START = '20140601'
BATCH_SIZE = 500
ZZ1000_XT_CODE = '000852.SH'

# stage 映射 (复用 update_daily.py 逻辑)
CIRCLE_STAGE = {
    '111': '顶部', '110': '分歧', '101': '护盘', '100': '崩塌',
    '000': '底部', '001': '蓄力', '010': '试探', '011': '启动',
}


# ============================================================
# xtdata 初始化
# ============================================================
def init_xtdata():
    """导入并初始化 xtdata"""
    global xtdata
    if xtdata is not None:
        return True
    # xtquant 在 QMT 安装目录下，需手动加路径
    xt_site = r'D:\国金证券QMT交易端\bin.x64\Lib\site-packages'
    if xt_site not in sys.path:
        sys.path.insert(0, xt_site)
    try:
        from xtquant import xtdata as _xd
        xtdata = _xd
        return True
    except ImportError as e:
        print(f'  !! xtquant 导入失败: {e}')
        print(f'  !! 检查路径: {xt_site}')
        print(f'  !! 该目录是否存在: {os.path.isdir(xt_site)}')
        xt_pkg = os.path.join(xt_site, 'xtquant')
        print(f'  !! xtquant包是否存在: {os.path.isdir(xt_pkg)}')
        return False


# ============================================================
# 代码格式转换
# ============================================================
def to_xt_code(code):
    """'000001' → '000001.SZ', '600000' → '600000.SH'"""
    if code.startswith(('6', '5')):
        return f'{code}.SH'
    return f'{code}.SZ'


def from_xt_code(xt_code):
    """'000001.SZ' → '000001'"""
    return xt_code.split('.')[0]


# ============================================================
# 数据获取
# ============================================================
def fetch_ohlc(xt_code, start_date, end_date):
    """
    从 xtdata 获取日线 OHLC 数据

    Args:
        xt_code: xtdata格式代码, 如 '000001.SZ'
        start_date: 开始日期, 如 '20140601'
        end_date: 结束日期, 如 '20260406'

    Returns:
        DataFrame(date, open, close, high, low) 或 None
    """
    try:
        data = xtdata.get_market_data(
            field_list=['open', 'high', 'low', 'close'],
            stock_list=[xt_code],
            period='1d',
            start_time=start_date,
            end_time=end_date,
        )
    except Exception as e:
        print(f'  !! 获取数据失败 {xt_code}: {e}')
        return None

    if data is None or 'close' not in data:
        return None

    # get_market_data 返回 {field: DataFrame(index=stock, columns=timestamp)}
    close_df = data['close']
    if close_df is None or close_df.empty:
        return None

    timestamps = close_df.columns
    closes = close_df.loc[xt_code].values.astype(float)
    opens = data['open'].loc[xt_code].values.astype(float)
    highs = data['high'].loc[xt_code].values.astype(float)
    lows = data['low'].loc[xt_code].values.astype(float)

    # 时间戳转日期字符串 — 统一输出 YYYY-MM-DD 格式
    dates = []
    for ts in timestamps:
        s = str(ts).strip()
        # 纯数字: YYYYMMDD 或毫秒时间戳
        digits = s.split('.')[0]  # 去掉可能的小数部分
        if digits.isdigit() and len(digits) == 8:
            # YYYYMMDD → YYYY-MM-DD
            dates.append(f'{digits[:4]}-{digits[4:6]}-{digits[6:8]}')
        elif digits.isdigit() and len(digits) >= 13:
            # 毫秒时间戳
            dt = datetime.fromtimestamp(int(digits) / 1000)
            dates.append(dt.strftime('%Y-%m-%d'))
        elif len(s) >= 10 and s[4] == '-':
            # 已经是 YYYY-MM-DD
            dates.append(s[:10])
        else:
            # fallback: 尝试当数字处理
            try:
                ts_int = int(float(s))
                ts_str = str(ts_int)
                if len(ts_str) == 8:
                    dates.append(f'{ts_str[:4]}-{ts_str[4:6]}-{ts_str[6:8]}')
                else:
                    dt = datetime.fromtimestamp(ts_int / 1000)
                    dates.append(dt.strftime('%Y-%m-%d'))
            except (ValueError, OSError):
                dates.append(s[:10])

    df = pd.DataFrame({
        'date': dates, 'open': opens, 'close': closes,
        'high': highs, 'low': lows,
    })

    # 过滤无效数据 (收盘价<=0)
    df = df[df['close'] > 0].reset_index(drop=True)
    return df if len(df) > 0 else None


def download_batch(xt_codes, start_date, end_date):
    """
    分批下载历史数据到 xtdata 本地缓存

    Args:
        xt_codes: xtdata格式代码列表
        start_date, end_date: 日期范围
    """
    total = len(xt_codes)
    for i in range(0, total, BATCH_SIZE):
        batch = xt_codes[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f'  下载批次 {batch_num}/{total_batches} ({len(batch)}只)...')

        for code in batch:
            try:
                xtdata.download_history_data(code, '1d', start_date, end_date)
            except Exception as e:
                print(f'    !! 下载失败 {code}: {e}')

        if i + BATCH_SIZE < total:
            time.sleep(1)  # 批间暂停


# ============================================================
# 个股增量更新
# ============================================================
def update_single_stock_xt(code):
    """通过 xtdata 增量更新单只个股"""
    cache_path = os.path.join(STOCKS_DIR, f'{code}.csv')
    xt_code = to_xt_code(code)
    today = datetime.now().strftime('%Y%m%d')

    try:
        if os.path.exists(cache_path):
            # 增量: 读现有CSV最后日期
            existing = pd.read_csv(cache_path, encoding='utf-8-sig')
            last_date = str(existing['date'].iloc[-1]).replace('-', '')

            ohlc = fetch_ohlc(xt_code, last_date, today)
            if ohlc is None or len(ohlc) == 0:
                return True  # 无新数据

            # 过滤已有日期
            last_date_fmt = str(existing['date'].iloc[-1])
            ohlc = ohlc[ohlc['date'] > last_date_fmt].reset_index(drop=True)
            if len(ohlc) == 0:
                return True

            n_new = len(ohlc)
            new_rows = ohlc.copy()
            new_rows['trend'] = np.nan
            new_rows['retail'] = np.nan
            new_rows['main_force'] = np.nan
            new_rows['year_gua'] = ''
            new_rows['month_gua'] = ''
            new_rows['day_gua'] = ''

            # 对齐列
            for col in existing.columns:
                if col not in new_rows.columns:
                    new_rows[col] = ''
            combined = pd.concat([existing, new_rows[existing.columns]],
                                 ignore_index=True)
        else:
            # 首次: 全量下载
            ohlc = fetch_ohlc(xt_code, FULL_DOWNLOAD_START, today)
            if ohlc is None or len(ohlc) < 60:
                return False

            combined = ohlc.copy()
            combined['trend'] = np.nan
            combined['retail'] = np.nan
            combined['main_force'] = np.nan
            combined['year_gua'] = ''
            combined['month_gua'] = ''
            combined['day_gua'] = ''
            n_new = len(combined)

        # 重算指标
        combined = recalc_tail_indicators(combined, n_tail=n_new + 5)

        # 保存
        os.makedirs(STOCKS_DIR, exist_ok=True)
        combined.to_csv(cache_path, index=False, encoding='utf-8-sig')
        return True

    except Exception as e:
        print(f'  错误 {code}: {e}')
        return False


# ============================================================
# 中证1000 增量更新
# ============================================================
def update_zz1000_xt():
    """通过 xtdata 增量更新中证1000"""
    print('=== 增量更新中证1000 (xtdata) ===')
    today = datetime.now().strftime('%Y%m%d')

    if not os.path.exists(ZZ1000_PATH):
        print(f'  !! 缓存不存在, 请先运行: python data_layer/prepare_zz1000.py')
        return False

    existing = pd.read_csv(ZZ1000_PATH, encoding='utf-8-sig')
    last_date = str(existing['date'].iloc[-1]).replace('-', '')
    print(f'  现有数据最后日期: {existing["date"].iloc[-1]}')

    # 下载并获取数据
    try:
        xtdata.download_history_data(ZZ1000_XT_CODE, '1d', last_date, today)
    except Exception as e:
        print(f'  !! 下载中证1000失败: {e}')
        return False

    ohlc = fetch_ohlc(ZZ1000_XT_CODE, last_date, today)
    if ohlc is None:
        print('  !! 获取中证1000数据失败')
        return False

    # 过滤已有日期
    last_date_fmt = str(existing['date'].iloc[-1])
    ohlc = ohlc[ohlc['date'] > last_date_fmt].reset_index(drop=True)
    n_new = len(ohlc)

    if n_new == 0:
        print(f'  无新增数据')
        return True

    print(f'  新增: {n_new} 天 ({ohlc["date"].iloc[0]} ~ {ohlc["date"].iloc[-1]})')

    # 追加新行
    new_rows = []
    for _, row in ohlc.iterrows():
        new_rows.append({
            'date': row['date'], 'close': row['close'],
            'high': row['high'], 'low': row['low'],
            'trend': np.nan, 'main_force': np.nan,
            'ma30': np.nan, 'ma120': np.nan, 'ma250': np.nan,
            'year_gua': '', 'month_gua': '', 'day_gua': '',
        })
    new_df = pd.DataFrame(new_rows)

    for col in existing.columns:
        if col not in new_df.columns:
            new_df[col] = ''
    combined = pd.concat([existing, new_df[existing.columns]], ignore_index=True)

    # 重算指标
    print('  重新计算指标...')
    combined = recalc_tail_indicators(
        combined, col_close='close', col_high='high', col_low='low',
        n_tail=n_new + 5)

    # 补充 stage 列
    for col, stage_col in [('year_gua', 'year_stage'),
                           ('month_gua', 'month_stage'),
                           ('day_gua', 'day_stage')]:
        if stage_col in combined.columns:
            for i in range(max(0, len(combined) - n_new - 5), len(combined)):
                g = str(combined.at[i, col]).zfill(3)
                combined.at[i, stage_col] = CIRCLE_STAGE.get(g, '')

    combined.to_csv(ZZ1000_PATH, index=False, encoding='utf-8-sig')
    print(f'  已保存: {ZZ1000_PATH}')
    print(f'  数据范围: {combined["date"].iloc[0]} ~ {combined["date"].iloc[-1]} ({len(combined)}行)')
    return True


# ============================================================
# 全量更新入口
# ============================================================
def get_all_stock_codes():
    """从 xtdata 获取沪深A股代码列表"""
    try:
        xt_codes = xtdata.get_stock_list_in_sector('沪深A股')
        # 只保留主板+创业板+科创板
        codes = [from_xt_code(c) for c in xt_codes
                 if c.startswith(('00', '30', '60', '68'))]
        return sorted(codes)
    except Exception as e:
        print(f'  !! 获取股票列表失败: {e}')
        return []


def update_all_stocks_xt(full_download=False):
    """增量更新全市场个股"""
    print('\n=== 增量更新全市场个股 (xtdata) ===')

    codes = get_all_stock_codes()
    if not codes:
        print('  !! 无法获取股票列表')
        return 0, 0

    print(f'  股票总数: {len(codes)} 只')

    # 分批下载到 xtdata 本地缓存
    today = datetime.now().strftime('%Y%m%d')
    if full_download:
        start = FULL_DOWNLOAD_START
        print(f'  全量下载模式: {start} ~ {today}')
    else:
        # 增量: 只下载最近30天 (足够覆盖节假日)
        start = (datetime.now() - pd.Timedelta(days=30)).strftime('%Y%m%d')

    xt_codes = [to_xt_code(c) for c in codes]
    print(f'  下载日线数据 ({start} ~ {today})...')
    download_batch(xt_codes, start, today)

    # 逐只更新
    print(f'\n  开始更新CSV...')
    os.makedirs(STOCKS_DIR, exist_ok=True)
    success = 0
    fail = 0
    failed_codes = []

    for i, code in enumerate(codes):
        if (i + 1) % 200 == 0:
            print(f'  进度: {i+1}/{len(codes)} ({success}更新 {fail}失败)')

        if update_single_stock_xt(code):
            success += 1
        else:
            fail += 1
            failed_codes.append(code)

    print(f'\n  完成! 更新{success}只, 失败{fail}只')
    if failed_codes and len(failed_codes) <= 20:
        print(f'  失败列表: {", ".join(failed_codes)}')
    return success, fail


def update_all_xt(full_download=False):
    """更新所有数据 (中证1000 + 全市场个股)"""
    print('=' * 60)
    print(f'  数据增量更新 (xtdata) — {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('=' * 60)

    if not init_xtdata():
        return

    ok1 = update_zz1000_xt()
    if ok1:
        success, fail = update_all_stocks_xt(full_download=full_download)
        print('\n  清除回测缓存...')
        _clear_pkl_cache()
    else:
        print('\n  !! 中证1000更新失败, 跳过个股更新')

    print('\n' + '=' * 60)
    print('  更新完成')
    print('=' * 60)


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='数据增量更新 (xtdata)')
    parser.add_argument('--zz1000-only', action='store_true',
                        help='只更新中证1000')
    parser.add_argument('--stock', type=str, default='',
                        help='只更新指定股票代码')
    parser.add_argument('--full-download', action='store_true',
                        help='全量下载(从2014-06-01)')
    args = parser.parse_args()

    if not init_xtdata():
        sys.exit(1)

    if args.stock:
        code = args.stock
        xt_code = to_xt_code(code)
        today = datetime.now().strftime('%Y%m%d')
        start = FULL_DOWNLOAD_START if args.full_download else \
            (datetime.now() - pd.Timedelta(days=30)).strftime('%Y%m%d')
        print(f'下载 {code} ({start} ~ {today})...')
        xtdata.download_history_data(xt_code, '1d', start, today)
        result = update_single_stock_xt(code)
        print(f'结果: {"成功" if result else "失败"}')
        if result:
            _clear_pkl_cache('stocks')
    elif args.zz1000_only:
        update_zz1000_xt()
        _clear_pkl_cache('zz1000*')
    else:
        update_all_xt(full_download=args.full_download)
