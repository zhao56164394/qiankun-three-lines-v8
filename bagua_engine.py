# -*- coding: utf-8 -*-
"""
八卦规则引擎 v3.0: 统一象卦体系 (位置-速度-主力动向)

三爻定义 (大盘+个股统一):
  初爻(底) = 位置: 趋势线(250日)>=50 → 阳, <50 → 阴
  二爻(中) = 速度: 趋势线20日变化>0 → 阳, <=0 → 阴
  三爻(上) = 主力动向: 主力线20日变化MA10>0 → 阳, <=0 → 阴

三爻组合 → 八卦映射(先天八卦序):
  二进制  爻(初二三)  卦名    卦象    含义
  000     阴阴阴     坤 ☷   地      低位+降+主力撤 = 深熊探底
  001     阴阴阳     艮 ☶   山      低位+降+主力进 = 底部吸筹
  010     阴阳阴     坎 ☵   水      低位+升+主力撤 = 反弹乏力
  011     阴阳阳     巽 ☴   风      低位+升+主力进 = 底部爆发
  100     阳阴阴     震 ☳   雷      高位+降+主力撤 = 高位出货
  101     阳阴阳     离 ☲   火      高位+降+主力进 = 高位护盘
  110     阳阳阴     兑 ☱   泽      高位+升+主力撤 = 牛末减仓
  111     阳阳阳     乾 ☰   天      高位+升+主力进 = 疯牛主升

注意: 卦名映射参考先天八卦的二进制对应关系(坤000→乾111)
"""
import numpy as np

# 八卦定义表: binary → (卦名, 卦象, 含义)
BAGUA_TABLE = {
    '000': ('坤', '☷', '深熊探底'),  # 低位+降+主力撤
    '001': ('艮', '☶', '底部吸筹'),  # 低位+降+主力进
    '010': ('坎', '☵', '反弹乏力'),  # 低位+升+主力撤
    '011': ('巽', '☴', '底部爆发'),  # 低位+升+主力进
    '100': ('震', '☳', '高位出货'),  # 高位+降+主力撤
    '101': ('离', '☲', '高位护盘'),  # 高位+降+主力进
    '110': ('兑', '☱', '牛末减仓'),  # 高位+升+主力撤
    '111': ('乾', '☰', '疯牛主升'),  # 高位+升+主力进
}


def encode_market_state_dynamic(market_trend, market_speed, breadth_momo, trend_anchor):
    """将市场级三指标编码为三爻与卦码 (位置相对锚-速度-广度动能)"""
    if market_trend is None or market_speed is None or breadth_momo is None or trend_anchor is None:
        return None, None, None, None
    if np.isnan(market_trend) or np.isnan(market_speed) or np.isnan(breadth_momo) or np.isnan(trend_anchor):
        return None, None, None, None
    yao1 = 1 if market_trend >= trend_anchor else 0
    yao2 = 1 if market_speed > 0 else 0
    yao3 = 1 if breadth_momo > 0 else 0
    return yao1, yao2, yao3, f"{yao1}{yao2}{yao3}"


def encode_yao(trend_val, speed_val, main_force_dir):
    """将三个指标编码为三爻二进制 (位置-速度-主力动向)"""
    if trend_val is None or speed_val is None or main_force_dir is None:
        return None
    if np.isnan(trend_val) or np.isnan(speed_val) or np.isnan(main_force_dir):
        return None
    yao1 = 1 if trend_val >= 50 else 0      # 初爻: 位置
    yao2 = 1 if speed_val > 0 else 0        # 二爻: 速度
    yao3 = 1 if main_force_dir > 0 else 0   # 三爻: 主力动向
    return f"{yao1}{yao2}{yao3}"


def calc_xiang_gua(closes, highs, lows, trend_period=250, speed_lookback=20, mf_ma_period=10):
    """
    计算象卦序列 (趋势线250 + 速度20 + 主力20日变化MA10)

    参数:
        closes, highs, lows: numpy数组 (日线)
        trend_period: 趋势线LLV/HHV窗口, 默认250
        speed_lookback: 速度回看天数, 默认20
        mf_ma_period: 主力变化的MA平滑窗口, 默认10

    返回:
        gua_list: list[str], 每天的象卦编码('000'~'111'), 无效为''
        trend: numpy数组, 趋势线值
        speed: numpy数组, 速度值
        main_force_dir: numpy数组, 主力动向值
    """
    from strategy.indicator import calc_trend_line, calc_main_force_line

    closes = np.array(closes, dtype=float)
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    n = len(closes)

    # 趋势线(250日窗口)
    trend = calc_trend_line(closes, highs, lows, period=trend_period)

    # 速度: 趋势线20日变化
    speed = np.full(n, np.nan)
    for i in range(speed_lookback, n):
        if not np.isnan(trend[i]) and not np.isnan(trend[i - speed_lookback]):
            speed[i] = trend[i] - trend[i - speed_lookback]

    # 主力线
    main_force = calc_main_force_line(closes)

    # 主力变化20日
    mf_chg20 = np.full(n, np.nan)
    for i in range(speed_lookback, n):
        if not np.isnan(main_force[i]) and not np.isnan(main_force[i - speed_lookback]):
            mf_chg20[i] = main_force[i] - main_force[i - speed_lookback]

    # 主力变化20日的MA10
    import pandas as pd
    mf_chg20_series = pd.Series(mf_chg20)
    main_force_dir = mf_chg20_series.rolling(mf_ma_period, min_periods=mf_ma_period).mean().values

    # 编码象卦
    gua_list = []
    for i in range(n):
        g = encode_yao(trend[i], speed[i], main_force_dir[i])
        gua_list.append(g if g else '')

    return gua_list, trend, speed, main_force_dir


if __name__ == '__main__':
    print("八卦编码表 (象卦体系 v3.0):")
    print(f"{'二进制':>6} | {'初爻':>4} | {'二爻':>4} | {'三爻':>6} | {'卦名':>4} | {'含义':>8}")
    print("-" * 55)
    for b in ['000', '001', '010', '011', '100', '101', '110', '111']:
        info = BAGUA_TABLE[b]
        y1 = '阳' if b[0] == '1' else '阴'
        y2 = '阳' if b[1] == '1' else '阴'
        y3 = '阳' if b[2] == '1' else '阴'
        print(f"{b:>6} | {y1:>4} | {y2:>4} | {y3:>6} | {info[0]}{info[1]:>3} | {info[2]:>8}")
