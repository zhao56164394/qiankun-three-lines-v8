# -*- coding: utf-8 -*-
"""
一次性修复脚本: 将CSV中混入的 YYYYMMDD 日期格式统一为 YYYY-MM-DD，并去除重复行。

用法:
  python data_layer/fix_date_format.py           # 修复所有CSV
  python data_layer/fix_date_format.py --dry-run  # 只检测不修改
"""
import os
import re
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_layer.update_daily import DATA_DIR, ZZ1000_PATH, STOCKS_DIR


def fix_date_col(df):
    """修复 date 列: YYYYMMDD → YYYY-MM-DD, 去重"""
    changed = False

    # 修复格式
    def norm(d):
        nonlocal changed
        s = str(d).strip()
        if re.match(r'^\d{8}$', s):
            changed = True
            return f'{s[:4]}-{s[4:6]}-{s[6:8]}'
        return s

    df['date'] = df['date'].apply(norm)

    # 去重 (保留第一条)
    before = len(df)
    df = df.drop_duplicates(subset='date', keep='first').reset_index(drop=True)
    if len(df) < before:
        changed = True

    return df, changed


def fix_file(path, label='', dry_run=False):
    """修复单个CSV文件"""
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except Exception as e:
        print(f'  !! 读取失败 {path}: {e}')
        return False

    if 'date' not in df.columns:
        return False

    df_fixed, changed = fix_date_col(df)
    if not changed:
        return False

    name = label or os.path.basename(path)
    if dry_run:
        print(f'  需修复: {name} (修复后 {len(df_fixed)} 行)')
    else:
        df_fixed.to_csv(path, index=False, encoding='utf-8-sig')
        print(f'  已修复: {name} (修复后 {len(df_fixed)} 行)')
    return True


def main():
    dry_run = '--dry-run' in sys.argv
    mode = '检测模式 (不修改文件)' if dry_run else '修复模式'
    print(f'=== CSV日期格式修复 — {mode} ===\n')

    fixed_count = 0

    # 1. 修复中证1000
    if os.path.exists(ZZ1000_PATH):
        if fix_file(ZZ1000_PATH, 'zz1000_daily.csv', dry_run):
            fixed_count += 1

    # 2. 修复所有个股
    if os.path.isdir(STOCKS_DIR):
        files = sorted([f for f in os.listdir(STOCKS_DIR) if f.endswith('.csv')])
        print(f'\n  扫描个股CSV: {len(files)} 个文件...')
        for f in files:
            path = os.path.join(STOCKS_DIR, f)
            if fix_file(path, f, dry_run):
                fixed_count += 1

    print(f'\n=== 完成! 共 {fixed_count} 个文件{"需修复" if dry_run else "已修复"} ===')


if __name__ == '__main__':
    main()
