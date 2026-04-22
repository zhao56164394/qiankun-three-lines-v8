# -*- coding: utf-8 -*-
"""
八卦卦象渲染组件
"""

# 卦象表
BAGUA_INFO = {
    '000': {'name': '坤', 'symbol': '\u2637', 'meaning': '至暗时刻', 'yy': '阴', 'color': '#22c55e'},
    '001': {'name': '艮', 'symbol': '\u2636', 'meaning': '底部蓄力', 'yy': '阴', 'color': '#86efac'},
    '010': {'name': '坎', 'symbol': '\u2635', 'meaning': '反弹无力', 'yy': '阴', 'color': '#4ade80'},
    '011': {'name': '巽', 'symbol': '\u2634', 'meaning': '风起云涌', 'yy': '阳', 'color': '#f59e0b'},
    '100': {'name': '震', 'symbol': '\u2633', 'meaning': '雷霆坠落', 'yy': '阴', 'color': '#ef4444'},
    '101': {'name': '离', 'symbol': '\u2632', 'meaning': '主力护盘', 'yy': '阳', 'color': '#fb923c'},
    '110': {'name': '兑', 'symbol': '\u2631', 'meaning': '散户狂欢', 'yy': '阴', 'color': '#a78bfa'},
    '111': {'name': '乾', 'symbol': '\u2630', 'meaning': '如日中天', 'yy': '阳', 'color': '#ef4444'},
}


def render_gua_badge(gua_code, size='large'):
    """
    生成单个卦象的HTML徽章

    Args:
        gua_code: 三位二进制如 '111'
        size: 'large' / 'small'
    """
    gua_code = str(gua_code).zfill(3)
    info = BAGUA_INFO.get(gua_code, {'name': '?', 'symbol': '?',
                                       'meaning': '未知', 'yy': '?', 'color': '#666'})

    if size == 'large':
        return f"""
        <div style="text-align:center; padding:10px;">
            <div style="font-size:48px; line-height:1;">{info['symbol']}</div>
            <div style="font-size:20px; font-weight:bold; color:{info['color']};">
                {info['name']} ({info['yy']})
            </div>
            <div style="font-size:13px; color:#888;">{info['meaning']}</div>
            <div style="font-size:11px; color:#555;">{gua_code}</div>
        </div>
        """
    else:
        return f"""
        <span style="display:inline-block; padding:2px 8px; border-radius:4px;
                     background:{info['color']}22; border:1px solid {info['color']}44;
                     color:{info['color']}; font-size:13px;">
            {info['symbol']}{info['name']}
        </span>
        """


def render_yinyang_light(yy_str):
    """
    渲染阴阳信号灯

    Args:
        yy_str: 如 '阴阴阳'
    """
    lights = []
    labels = ['年', '月', '日']
    for i, ch in enumerate(yy_str):
        if ch == '阳':
            color = '#ef4444'
            bg = '#ef444422'
        else:
            color = '#22c55e'
            bg = '#22c55e22'
        lights.append(
            f'<div style="display:inline-block; width:50px; text-align:center; '
            f'padding:8px; margin:0 4px; border-radius:8px; background:{bg}; '
            f'border:2px solid {color};">'
            f'<div style="font-size:12px; color:#888;">{labels[i]}</div>'
            f'<div style="font-size:18px; font-weight:bold; color:{color};">{ch}</div>'
            f'</div>'
        )
    return '<div style="display:flex; justify-content:center;">' + ''.join(lights) + '</div>'


def render_three_layer(year_code, month_code, day_code):
    """渲染年月日三层卦象"""
    html = '<div style="display:flex; justify-content:space-around;">'
    for label, code in [('年 (大周期)', year_code),
                         ('月 (中趋势)', month_code),
                         ('日 (短波动)', day_code)]:
        html += f'<div style="flex:1; text-align:center; padding:5px;">'
        html += f'<div style="font-size:12px; color:#888; margin-bottom:5px;">{label}</div>'
        html += render_gua_badge(code, 'large')
        html += '</div>'
    html += '</div>'
    return html
