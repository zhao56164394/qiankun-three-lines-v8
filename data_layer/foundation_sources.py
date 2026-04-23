# -*- coding: utf-8 -*-
"""
foundation_sources.py

新底座原始数据读取层：
- 统一 code/date 标准化
- 只做加载与轻度标准化
- 不做策略语义加工
"""
import os
import io
import zipfile
from typing import Iterable, List, Optional

import pandas as pd

from data_layer.foundation_config import (
    PATHS,
    CORE_INDEX_CODES,
    CORE_INDEX_ALIASES,
    STOCK_BASIC_RENAME,
    DAILY_METRICS_RENAME,
    MONEYFLOW_RENAME,
    CHIP_RENAME,
    INDEX_DAILY_RENAME,
    NUMERIC_COLUMNS,
)


def normalize_code(value) -> str:
    if pd.isna(value):
        return ''
    s = str(value).strip()
    if not s:
        return ''
    if '.' in s:
        left, right = s.split('.', 1)
        if left.isdigit() and len(left) <= 6:
            return left.zfill(6)
        return left[-6:].zfill(6) if left[-6:].isdigit() else left
    digits = ''.join(ch for ch in s if ch.isdigit())
    if len(digits) >= 6:
        return digits[-6:]
    return s.zfill(6) if s.isdigit() else s


def normalize_ts_code(value) -> str:
    if pd.isna(value):
        return ''
    s = str(value).strip()
    if not s:
        return ''
    if '.' in s:
        left, right = s.split('.', 1)
        if left.isdigit() and len(left) <= 6:
            return f'{left.zfill(6)}.{right.upper()}'
    code = normalize_code(s)
    if code.startswith(('5', '6', '9')):
        return f'{code}.SH'
    return f'{code}.SZ'


def normalize_date(value) -> str:
    if pd.isna(value):
        return ''
    s = str(value).strip()
    if not s:
        return ''
    if len(s) == 8 and s.isdigit():
        return f'{s[:4]}-{s[4:6]}-{s[6:8]}'
    try:
        dt = pd.to_datetime(s)
        return dt.strftime('%Y-%m-%d')
    except Exception:
        return s[:10]


def to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def latest_csv_under(root: str) -> Optional[str]:
    if not os.path.exists(root):
        return None
    candidates: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith('.csv'):
                candidates.append(os.path.join(dirpath, filename))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1]


def list_csv_files_under(root: str) -> List[str]:
    if not os.path.exists(root):
        return []
    candidates: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith('.csv'):
                candidates.append(os.path.join(dirpath, filename))
    candidates.sort()
    return candidates


def csv_for_date(root: str, date: str, suffix: str = '.csv') -> Optional[str]:
    date_digits = normalize_date(date).replace('-', '')
    if not os.path.exists(root):
        return None
    matched: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if date_digits in filename and filename.lower().endswith(suffix):
                matched.append(os.path.join(dirpath, filename))
    if not matched:
        return None
    matched.sort()
    return matched[-1]


def load_stock_basic() -> pd.DataFrame:
    df = pd.read_csv(PATHS['stock_basic'], encoding='utf-8-sig', dtype=str)
    df = df.rename(columns=STOCK_BASIC_RENAME)
    keep = [c for c in STOCK_BASIC_RENAME.values() if c in df.columns]
    df = df[keep].copy()
    df['code'] = df['code'].map(normalize_code)
    df['ts_code'] = df['ts_code'].map(normalize_ts_code)
    df['list_date'] = df['list_date'].map(normalize_date)
    df['exchange'] = df['exchange'].replace({'SH': 'SSE', 'SZ': 'SZSE'})
    df['is_st'] = df['name'].fillna('').str.contains('ST', case=False, na=False)
    return df.sort_values('code').reset_index(drop=True)


def load_daily_metrics(date: Optional[str] = None, file_path: Optional[str] = None) -> pd.DataFrame:
    path = file_path or (csv_for_date(PATHS['stock_daily_metrics_root'], date) if date else latest_csv_under(PATHS['stock_daily_metrics_root']))
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path, encoding='utf-8-sig')
    df = df.rename(columns=DAILY_METRICS_RENAME)
    cols = [c for c in DAILY_METRICS_RENAME.values() if c in df.columns]
    df = df[cols].copy()
    df['code'] = df['code'].map(normalize_code)
    df['date'] = df['date'].map(normalize_date)
    df = to_numeric(df, NUMERIC_COLUMNS['daily_metrics'])
    if 'amount_k' in df.columns:
        df['amount'] = df['amount_k'] * 1000.0
        df = df.drop(columns=['amount_k'])
    if 'total_mv_wan' in df.columns:
        df['total_mv'] = df['total_mv_wan'] * 10000.0
        df = df.drop(columns=['total_mv_wan'])
    if 'circ_mv_wan' in df.columns:
        df['circ_mv'] = df['circ_mv_wan'] * 10000.0
        df = df.drop(columns=['circ_mv_wan'])
    return df.sort_values(['date', 'code']).reset_index(drop=True)


def load_daily_metrics_history(start_date: Optional[str] = None) -> pd.DataFrame:
    if start_date is not None:
        start_date = normalize_date(start_date)

    parts = []
    for path in list_csv_files_under(PATHS['stock_daily_metrics_root']):
        try:
            df = load_daily_metrics(file_path=path)
        except Exception:
            continue
        if df.empty:
            continue
        if start_date is not None:
            df = df[df['date'] >= start_date].copy()
        if not df.empty:
            parts.append(df)

    if not parts:
        return pd.DataFrame(columns=['date', 'code', 'open', 'high', 'low', 'close'])

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(['date', 'code']).drop_duplicates(['date', 'code'], keep='last').reset_index(drop=True)
    return out



def load_moneyflow(date: Optional[str] = None, file_path: Optional[str] = None) -> pd.DataFrame:
    path = file_path
    if path is None and date is not None:
        path = csv_for_date(PATHS['stock_moneyflow_root'], date)
    if path is None:
        sample_root = os.path.join(os.path.dirname(PATHS['stock_moneyflow_root']), '资金流向_示例', '个股资金流向')
        path = latest_csv_under(PATHS['stock_moneyflow_root']) or latest_csv_under(sample_root)
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path, encoding='utf-8-sig')
    df = df.rename(columns=MONEYFLOW_RENAME)
    cols = [c for c in MONEYFLOW_RENAME.values() if c in df.columns]
    df = df[cols].copy()
    df['code'] = df['code'].map(normalize_code)
    df['date'] = df['date'].map(normalize_date)
    df = to_numeric(df, NUMERIC_COLUMNS['moneyflow'])
    if {'small_buy', 'small_sell'}.issubset(df.columns):
        df['small_net'] = df['small_buy'] - df['small_sell']
    if {'large_buy', 'large_sell'}.issubset(df.columns):
        df['large_net'] = df['large_buy'] - df['large_sell']
    if {'super_large_buy', 'super_large_sell'}.issubset(df.columns):
        df['super_large_net'] = df['super_large_buy'] - df['super_large_sell']
    return df.sort_values(['date', 'code']).reset_index(drop=True)


def load_moneyflow_history(start_date: Optional[str] = None) -> pd.DataFrame:
    if start_date is not None:
        start_date = normalize_date(start_date)

    parts = []
    for path in list_csv_files_under(PATHS['stock_moneyflow_year_root']):
        try:
            df = load_moneyflow(file_path=path)
        except Exception:
            continue
        if df.empty:
            continue
        if start_date is not None:
            df = df[df['date'] >= start_date].copy()
        if not df.empty:
            parts.append(df)

    for path in list_csv_files_under(PATHS['stock_moneyflow_root']):
        try:
            df = load_moneyflow(file_path=path)
        except Exception:
            continue
        if df.empty:
            continue
        if start_date is not None:
            df = df[df['date'] >= start_date].copy()
        if not df.empty:
            parts.append(df)

    if not parts:
        return pd.DataFrame(columns=['date', 'code', 'small_net', 'large_net', 'super_large_net'])

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(['date', 'code']).drop_duplicates(['date', 'code'], keep='last').reset_index(drop=True)
    return out


def load_chip_distribution(date: Optional[str] = None, file_path: Optional[str] = None) -> pd.DataFrame:
    path = file_path or (csv_for_date(PATHS['chip_root'], date) if date else latest_csv_under(PATHS['chip_root']))
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path, encoding='utf-8-sig')
    df = df.rename(columns=CHIP_RENAME)
    cols = [c for c in CHIP_RENAME.values() if c in df.columns]
    df = df[cols].copy()
    df['code'] = df['code'].map(normalize_code)
    df['date'] = df['date'].map(normalize_date)
    df = to_numeric(df, NUMERIC_COLUMNS['chip'])
    return df.sort_values(['date', 'code']).reset_index(drop=True)


def load_limit_ladder(date: Optional[str] = None, file_path: Optional[str] = None) -> pd.DataFrame:
    path = file_path or (csv_for_date(PATHS['limit_ladder_root'], date) if date else latest_csv_under(PATHS['limit_ladder_root']))
    if not path:
        return pd.DataFrame(columns=['date', 'code', 'lb_count'])
    df = pd.read_csv(path, encoding='utf-8-sig')
    rename_map = {'股票代码': 'code', '交易日期': 'date', '连板次数': 'lb_count'}
    df = df.rename(columns=rename_map)
    keep = [c for c in ['code', 'date', 'lb_count'] if c in df.columns]
    df = df[keep].copy()
    df['code'] = df['code'].map(normalize_code)
    df['date'] = df['date'].map(normalize_date)
    if 'lb_count' in df.columns:
        df['lb_count'] = pd.to_numeric(df['lb_count'], errors='coerce').fillna(0)
    return df.sort_values(['date', 'code']).reset_index(drop=True)


def load_limit_summary() -> pd.DataFrame:
    up = pd.read_csv(PATHS['limit_up_summary'], encoding='utf-8-sig') if os.path.exists(PATHS['limit_up_summary']) else pd.DataFrame()
    down = pd.read_csv(PATHS['limit_down_summary'], encoding='utf-8-sig') if os.path.exists(PATHS['limit_down_summary']) else pd.DataFrame()
    broken = pd.read_csv(PATHS['limit_broken_board_summary'], encoding='utf-8-sig') if os.path.exists(PATHS['limit_broken_board_summary']) else pd.DataFrame()

    def _pick(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=['date', value_name])
        date_col = next((c for c in df.columns if '日期' in c), df.columns[0])
        value_col = next((c for c in df.columns if c != date_col), df.columns[-1])
        out = df[[date_col, value_col]].copy()
        out.columns = ['date', value_name]
        out['date'] = out['date'].map(normalize_date)
        out[value_name] = pd.to_numeric(out[value_name], errors='coerce')
        return out

    out = _pick(up, 'zt_count').merge(_pick(down, 'dt_count'), on='date', how='outer')
    out = out.merge(_pick(broken, 'zb_count'), on='date', how='outer')
    return out.sort_values('date').reset_index(drop=True)


def load_index_daily(date: Optional[str] = None, file_path: Optional[str] = None) -> pd.DataFrame:
    path = file_path or (csv_for_date(PATHS['index_daily_root'], date) if date else latest_csv_under(PATHS['index_daily_root']))
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path, encoding='utf-8-sig')
    df = df.rename(columns=INDEX_DAILY_RENAME)
    keep = [c for c in INDEX_DAILY_RENAME.values() if c in df.columns]
    df = df[keep].copy()
    df['index_code'] = df['index_code'].astype(str).str.strip().str.upper()
    df['date'] = df['date'].map(normalize_date)
    df = to_numeric(df, NUMERIC_COLUMNS['index_daily'])
    return df.sort_values(['date', 'index_code']).reset_index(drop=True)



def load_index_closes_from_stock_metrics(date: Optional[str] = None) -> pd.DataFrame:
    path = csv_for_date(PATHS['index_daily_root'], date) if date else latest_csv_under(PATHS['index_daily_root'])
    if not path:
        return pd.DataFrame(columns=['date'])
    raw = pd.read_csv(path, encoding='utf-8-sig')
    code_col = next((c for c in raw.columns if '指数代码' in c), None)
    date_col = next((c for c in raw.columns if '交易日期' in c), None)
    close_col = next((c for c in raw.columns if '收盘' in c), None)
    name_col = next((c for c in raw.columns if '指数名称' in c or '指数简称' in c), None)
    if not code_col or not date_col or not close_col:
        return pd.DataFrame(columns=['date'])
    raw = raw[[c for c in [code_col, name_col, date_col, close_col] if c is not None]].copy()
    raw.columns = ['index_code', 'index_name', 'date', 'close'] if name_col else ['index_code', 'date', 'close']
    if 'index_name' not in raw.columns:
        raw['index_name'] = ''
    raw['index_code'] = raw['index_code'].astype(str).str.strip().str.upper()
    raw['index_name'] = raw['index_name'].astype(str).str.strip()
    raw['date'] = raw['date'].map(normalize_date)
    raw['close'] = pd.to_numeric(raw['close'], errors='coerce')

    rows = []
    for _, row in raw.iterrows():
        key = CORE_INDEX_CODES.get(row['index_code']) or CORE_INDEX_ALIASES.get(row['index_name'])
        if key:
            rows.append({'date': row['date'], 'key': key, 'close': row['close']})
    if not rows:
        return pd.DataFrame(columns=['date'])
    out = pd.DataFrame(rows)
    out = out.pivot_table(index='date', columns='key', values='close', aggfunc='last').reset_index()
    return out


def load_index_closes_history(start_date: Optional[str] = None) -> pd.DataFrame:
    if start_date is not None:
        start_date = normalize_date(start_date)

    rows = []
    if os.path.exists(PATHS['index_daily_zip']):
        with zipfile.ZipFile(PATHS['index_daily_zip']) as zf:
            for name in zf.namelist():
                code = os.path.basename(name).replace('.csv', '').upper()
                key = CORE_INDEX_CODES.get(code)
                if not key:
                    continue
                try:
                    raw = pd.read_csv(io.BytesIO(zf.read(name)), encoding='utf-8-sig')
                except Exception:
                    continue
                date_col = next((c for c in raw.columns if '交易日期' in c), None)
                close_col = next((c for c in raw.columns if '收盘' in c), None)
                if not date_col or not close_col:
                    continue
                out = raw[[date_col, close_col]].copy()
                out.columns = ['date', 'close']
                out['date'] = out['date'].map(normalize_date)
                out['close'] = pd.to_numeric(out['close'], errors='coerce')
                if start_date is not None:
                    out = out[out['date'] >= start_date].copy()
                if not out.empty:
                    out['key'] = key
                    rows.extend(out[['date', 'key', 'close']].to_dict('records'))

    if os.path.exists(PATHS['index_daily_zip_legacy']):
        alias_map = {
            '000001': 'sh_close',
            '399001': 'sz_close',
            '000300': 'hs300_close',
            '000905': 'csi500_close',
            '000852': 'csi1000_close',
            '000985': 'allA_close',
        }
        with zipfile.ZipFile(PATHS['index_daily_zip_legacy']) as zf:
            for name in zf.namelist():
                base = os.path.basename(name)
                code = base.split('_')[0].strip()
                key = alias_map.get(code)
                if not key:
                    continue
                try:
                    raw = pd.read_csv(io.BytesIO(zf.read(name)), encoding='utf-8-sig')
                except Exception:
                    continue
                date_col = next((c for c in raw.columns if '日期' in c), None)
                close_col = next((c for c in raw.columns if '收盘' in c or c == '收盘'), None)
                if not date_col:
                    date_col = raw.columns[0]
                if not close_col:
                    close_col = raw.columns[2] if len(raw.columns) >= 3 else None
                if not close_col:
                    continue
                out = raw[[date_col, close_col]].copy()
                out.columns = ['date', 'close']
                out['date'] = out['date'].map(normalize_date)
                out['close'] = pd.to_numeric(out['close'], errors='coerce')
                if start_date is not None:
                    out = out[out['date'] >= start_date].copy()
                if not out.empty:
                    out['key'] = key
                    rows.extend(out[['date', 'key', 'close']].to_dict('records'))

    if not rows:
        return pd.DataFrame(columns=['date'])

    out = pd.DataFrame(rows)
    out = out.pivot_table(index='date', columns='key', values='close', aggfunc='last').reset_index()
    out = out.sort_values('date').drop_duplicates(['date'], keep='last').reset_index(drop=True)

    inc = None
    try:
        inc = load_index_closes_from_stock_metrics()
    except Exception:
        inc = None
    if inc is not None and not inc.empty:
        out = out.merge(inc, on='date', how='outer', suffixes=('_hist', '_inc'))
        for col in ['allA_close', 'hs300_close', 'csi500_close', 'csi1000_close', 'sh_close', 'sz_close']:
            hist_col = f'{col}_hist'
            inc_col = f'{col}_inc'
            if hist_col in out.columns and inc_col in out.columns:
                out[col] = out[inc_col].combine_first(out[hist_col])
                out = out.drop(columns=[hist_col, inc_col])
            elif hist_col in out.columns:
                out = out.rename(columns={hist_col: col})
            elif inc_col in out.columns:
                out = out.rename(columns={inc_col: col})

    return out.sort_values('date').reset_index(drop=True)


def load_industry_components(date: Optional[str] = None, file_path: Optional[str] = None) -> pd.DataFrame:
    path = file_path or (csv_for_date(PATHS['industry_component_root'], date) if date else latest_csv_under(PATHS['industry_component_root']) or PATHS['industry_component_full'])
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=['code', 'industry_name'])
    df = pd.read_csv(path, encoding='utf-8-sig')
    rename_map = {'股票代码': 'code', '股票名称': 'name', '指数名称': 'industry_name'}
    df = df.rename(columns=rename_map)
    keep = [c for c in ['code', 'name', 'industry_name'] if c in df.columns]
    df = df[keep].copy()
    df['code'] = df['code'].map(normalize_code)
    df = df.dropna(subset=['code'])
    df = df[df['code'] != '']
    df = df.groupby('code', as_index=False).first()
    return df


def load_concept_components(date: Optional[str] = None, file_path: Optional[str] = None) -> pd.DataFrame:
    path = file_path or (csv_for_date(PATHS['concept_component_root'], date) if date else latest_csv_under(PATHS['concept_component_root']) or PATHS['concept_component_full'])
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=['code', 'concept_count'])
    df = pd.read_csv(path, encoding='utf-8-sig')
    rename_map = {'股票代码': 'code', '指数名称': 'concept_name'}
    df = df.rename(columns=rename_map)
    if 'code' not in df.columns:
        return pd.DataFrame(columns=['code', 'concept_count'])
    df['code'] = df['code'].map(normalize_code)
    df = df[df['code'] != '']
    if 'concept_name' not in df.columns:
        df['concept_name'] = 'concept'
    out = df.groupby('code').agg(concept_count=('concept_name', 'nunique')).reset_index()
    return out
