# -*- coding: utf-8 -*-
"""
LWC 组件 — 将 TradingView Lightweight Charts 嵌入 Streamlit

直接通过 CDN 加载 JS, 不依赖 lightweight-charts python 包,
避免与 QMT 环境冲突。
"""
import json
import streamlit.components.v1 as components


# TradingView Lightweight Charts CDN (v4)
LWC_CDN = "https://unpkg.com/lightweight-charts@4.2.2/dist/lightweight-charts.standalone.production.js"


def render_candlestick(
    df,
    markers=None,
    lines=None,
    height=500,
    title="",
    volume=True,
    theme="dark",
    segments=None,
    header_info=None,
    header_points=None,
):
    """
    渲染 K 线图 + 可选成交量 + 买卖标记 + 叠加线 + 背景分段

    Args:
        df: DataFrame, 必须含 date/open/high/low/close 列, 可选 volume
        markers: list of dict, 格式:
            [{'time': '2024-01-05', 'position': 'belowBar',
              'color': '#ef4444', 'shape': 'arrowUp', 'text': '买入'}, ...]
        lines: list of dict, 格式:
            [{'name': 'trend', 'data': [{'time':'...','value':50}, ...],
              'color': '#f59e0b', 'width': 2}, ...]
        height: 图表高度
        title: 标题
        volume: 是否显示成交量
        theme: 'dark' | 'light'
        segments: list of dict, 分段背景信息
        header_info: dict, 图内顶部信息条
        header_points: list of dict, 鼠标所指日期对应的信息条数据
    """
    # 转换 OHLC 数据
    ohlc_data = []
    vol_data = []
    for _, row in df.iterrows():
        dt = str(row['date'])[:10]
        ohlc_data.append({
            'time': dt,
            'open': round(float(row['open']), 3),
            'high': round(float(row['high']), 3),
            'low': round(float(row['low']), 3),
            'close': round(float(row['close']), 3),
        })
        if volume and 'volume' in df.columns:
            vol_data.append({
                'time': dt,
                'value': int(row['volume']),
                'color': 'rgba(239,68,68,0.4)' if row['close'] >= row['open']
                         else 'rgba(34,197,94,0.4)',
            })

    ohlc_json = json.dumps(ohlc_data)
    vol_json = json.dumps(vol_data) if vol_data else '[]'
    markers_json = json.dumps(markers or [], ensure_ascii=False)
    segments_json = json.dumps(segments or [], ensure_ascii=False)
    header_json = json.dumps(header_info or {}, ensure_ascii=False)
    header_points_json = json.dumps(header_points or [], ensure_ascii=False)

    # 构建叠加线 JS
    lines_js = ""
    if lines:
        for i, line_cfg in enumerate(lines):
            line_data = json.dumps(line_cfg['data'])
            color = line_cfg.get('color', '#f59e0b')
            width = line_cfg.get('width', 2)
            name = line_cfg.get('name', f'line_{i}')
            lines_js += f"""
            var lineSeries_{i} = chart.addLineSeries({{
                color: '{color}',
                lineWidth: {width},
                title: '{name}',
                priceScaleId: 'overlay_trend',
                lastValueVisible: false,
                priceLineVisible: false,
            }});
            lineSeries_{i}.setData({line_data});
            """

    # 主题配色 (A股: 红涨绿跌)
    if theme == 'dark':
        bg_color = '#1a1a2e'
        text_color = '#d1d5db'
        grid_color = 'rgba(42, 46, 57, 0.5)'
        border_color = 'rgba(42, 46, 57, 0.8)'
        up_color = '#ef4444'
        down_color = '#22c55e'
        crosshair_color = 'rgba(255,255,255,0.3)'
        header_bg = 'rgba(13,13,26,0.82)'
        header_border = 'rgba(255,255,255,0.08)'
    else:
        bg_color = '#ffffff'
        text_color = '#333333'
        grid_color = 'rgba(200, 200, 200, 0.3)'
        border_color = 'rgba(200, 200, 200, 0.5)'
        up_color = '#ef4444'
        down_color = '#22c55e'
        crosshair_color = 'rgba(0,0,0,0.3)'
        header_bg = 'rgba(255,255,255,0.88)'
        header_border = 'rgba(0,0,0,0.08)'

    # 成交量子图 JS
    volume_js = ""
    if vol_data:
        volume_js = f"""
        var volumeSeries = chart.addHistogramSeries({{
            priceFormat: {{ type: 'volume' }},
            priceScaleId: 'vol',
        }});
        chart.priceScale('vol').applyOptions({{
            scaleMargins: {{ top: 0.85, bottom: 0 }},
        }});
        volumeSeries.setData({vol_json});
        """

    title_html = f'<div style="color:{text_color};font-size:14px;font-weight:bold;padding:8px 0 4px 12px;font-family:Microsoft YaHei,sans-serif;">{title}</div>' if title else ''

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="{LWC_CDN}"></script>
        <style>
            body {{ margin: 0; padding: 0; background: {bg_color}; }}
            #chart-wrapper {{ position: relative; width: 100%; height: {height}px; overflow: hidden; border-radius: 8px; }}
            #chart-container {{ width: 100%; height: {height}px; position: relative; z-index: 1; }}
            #segment-overlay {{ position: absolute; inset: 0; z-index: 2; pointer-events: none; }}
            .segment-block {{ position: absolute; top: 42px; bottom: 0; border-right: 1px solid rgba(255,255,255,0.06); box-sizing: border-box; }}
            #header-band {{ position: absolute; left: 0; right: 0; top: 0; min-height: 42px; z-index: 3; pointer-events: none; padding: 6px 12px 7px 12px; background: {header_bg}; border-bottom: 1px solid {header_border}; box-sizing: border-box; }}
            #header-line1 {{ color: {text_color}; font-size: 14px; font-weight: 700; line-height: 1.2; }}
            #header-line2 {{ color: {text_color}; font-size: 12px; opacity: 0.92; margin-top: 2px; line-height: 1.2; }}
        </style>
    </head>
    <body>
        {title_html}
        <div id="chart-wrapper">
            <div id="chart-container"></div>
            <div id="segment-overlay"></div>
            <div id="header-band">
                <div id="header-line1"></div>
                <div id="header-line2"></div>
            </div>
        </div>
        <script>
            var chart = LightweightCharts.createChart(
                document.getElementById('chart-container'), {{
                layout: {{
                    background: {{ type: 'solid', color: '{bg_color}' }},
                    textColor: '{text_color}',
                    fontFamily: 'Microsoft YaHei, sans-serif',
                }},
                grid: {{
                    vertLines: {{ color: '{grid_color}' }},
                    horzLines: {{ color: '{grid_color}' }},
                }},
                crosshair: {{
                    mode: LightweightCharts.CrosshairMode.Normal,
                    vertLine: {{ color: '{crosshair_color}', width: 1, style: 2 }},
                    horzLine: {{ color: '{crosshair_color}', width: 1, style: 2 }},
                }},
                rightPriceScale: {{
                    borderColor: '{border_color}',
                    autoScale: true,
                    scaleMargins: {{ top: 0.12, bottom: 0.08 }},
                }},
                timeScale: {{
                    borderColor: '{border_color}',
                    timeVisible: false,
                }},
                localization: {{
                    locale: 'zh-CN',
                }},
            }});
            chart.priceScale('overlay_trend').applyOptions({{
                visible: false,
                scaleMargins: {{ top: 0.12, bottom: 0.08 }},
            }});

            var candleSeries = chart.addCandlestickSeries({{
                upColor: '{up_color}',
                downColor: '{down_color}',
                borderUpColor: '{up_color}',
                borderDownColor: '{down_color}',
                wickUpColor: '{up_color}',
                wickDownColor: '{down_color}',
            }});
            var ohlcData = {ohlc_json};
            candleSeries.setData(ohlcData);

            var markers = {markers_json};
            if (markers.length > 0) {{
                candleSeries.setMarkers(markers);
            }}

            {lines_js}
            {volume_js}

            var segmentOverlay = document.getElementById('segment-overlay');
            var segments = {segments_json};
            var headerInfo = {header_json};
            var headerPoints = {header_points_json};
            var headerMap = {{}};
            for (var hpIdx = 0; hpIdx < headerPoints.length; hpIdx++) {{
                var hp = headerPoints[hpIdx];
                if (hp && hp.time) {{
                    headerMap[String(hp.time)] = hp;
                }}
            }}

            function setHeader(info) {{
                var line1 = document.getElementById('header-line1');
                var line2 = document.getElementById('header-line2');
                if (!info || Object.keys(info).length === 0) {{
                    line1.textContent = '';
                    line2.textContent = '';
                    return;
                }}
                line1.textContent = info.line1 || '';
                line2.textContent = info.line2 || '';
            }}

            function renderSegments() {{
                segmentOverlay.innerHTML = '';
                if (!segments || segments.length === 0) return;
                for (var i = 0; i < segments.length; i++) {{
                    var seg = segments[i];
                    var x1 = chart.timeScale().timeToCoordinate(seg.start_time);
                    var x2 = chart.timeScale().timeToCoordinate(seg.end_time);
                    if (x1 === null || x2 === null) continue;
                    var left = Math.min(x1, x2);
                    var right = Math.max(x1, x2);
                    var width = Math.max(2, right - left + 6);
                    var el = document.createElement('div');
                    el.className = 'segment-block';
                    el.style.left = left + 'px';
                    el.style.width = width + 'px';
                    el.style.background = seg.color || 'rgba(255,255,255,0.08)';
                    segmentOverlay.appendChild(el);
                }}
            }}

            function updateVisiblePriceRange() {{
                if (!ohlcData || ohlcData.length === 0) return;
                var timeScale = chart.timeScale();
                var visibleRange = timeScale.getVisibleRange();
                if (!visibleRange || !visibleRange.from || !visibleRange.to) return;
                var from = String(visibleRange.from).slice(0, 10);
                var to = String(visibleRange.to).slice(0, 10);
                var visibleBars = ohlcData.filter(function(bar) {{
                    return String(bar.time) >= from && String(bar.time) <= to;
                }});
                if (visibleBars.length === 0) return;
                var minLow = visibleBars[0].low;
                var maxHigh = visibleBars[0].high;
                for (var i = 1; i < visibleBars.length; i++) {{
                    if (visibleBars[i].low < minLow) minLow = visibleBars[i].low;
                    if (visibleBars[i].high > maxHigh) maxHigh = visibleBars[i].high;
                }}
                var actualMin = minLow;
                var actualMax = maxHigh;
                if (actualMax <= actualMin) {{
                    actualMax = actualMin + Math.max(Math.abs(actualMin) * 0.001, 1);
                }}
                candleSeries.applyOptions({{
                    autoscaleInfoProvider: function() {{
                        return {{
                            priceRange: {{
                                minValue: actualMin,
                                maxValue: actualMax,
                            }}
                        }};
                    }}
                }});
            }}

            function updateHeaderFromTime(time) {{
                if (!time) {{
                    setHeader(headerInfo);
                    return;
                }}
                var key = String(time).slice(0, 10);
                if (headerMap[key]) {{
                    setHeader(headerMap[key]);
                }} else {{
                    setHeader(headerInfo);
                }}
            }}

            chart.timeScale().fitContent();
            chart.priceScale('right').applyOptions({{
                autoScale: true,
                scaleMargins: {{ top: 0.12, bottom: 0.08 }},
            }});
            setHeader(headerInfo);
            renderSegments();

            chart.timeScale().subscribeVisibleTimeRangeChange(function() {{
                renderSegments();
            }});

            chart.subscribeCrosshairMove(function(param) {{
                if (!param || !param.time) {{
                    setHeader(headerInfo);
                    return;
                }}
                updateHeaderFromTime(param.time);
            }});

            window.addEventListener('mouseleave', function() {{
                setHeader(headerInfo);
            }});

            window.addEventListener('resize', function() {{
                chart.applyOptions({{ width: document.getElementById('chart-container').clientWidth }});
                renderSegments();
            }});
        </script>
    </body>
    </html>
    """

    total_height = height + (30 if title else 0) + 10
    components.html(html, height=total_height, scrolling=False)




def render_market_bagua_chart(df, height=460, title="市场爻主图"):
    """渲染市场卦主图：K线 + 趋势锚线 + 变卦标记"""
    if df is None or len(df) == 0:
        return

    plot_df = df.copy()
    markers = []
    if 'changed' in plot_df.columns:
        changed_rows = plot_df[plot_df['changed'] == 1]
        for _, row in changed_rows.iterrows():
            gua_name = row.get('gua_name', '')
            gua_code = row.get('gua_code', '')
            markers.append({
                'time': str(row['date'])[:10],
                'position': 'aboveBar',
                'color': '#f59e0b',
                'shape': 'circle',
                'text': f"{gua_name}".strip(),
            })

    lines = []
    if 'market_trend_55' in plot_df.columns:
        trend_data = []
        for _, row in plot_df.dropna(subset=['market_trend_55']).iterrows():
            trend_data.append({
                'time': str(row['date'])[:10],
                'value': round(float(row['market_trend_55']), 3),
            })
        if trend_data:
            lines.append({'name': 'trend55', 'data': trend_data, 'color': '#f59e0b', 'width': 2})

    if 'market_trend_anchor_120' in plot_df.columns:
        anchor_data = []
        for _, row in plot_df.dropna(subset=['market_trend_anchor_120']).iterrows():
            anchor_data.append({
                'time': str(row['date'])[:10],
                'value': round(float(row['market_trend_anchor_120']), 3),
            })
        if anchor_data:
            lines.append({'name': 'anchor120', 'data': anchor_data, 'color': '#60a5fa', 'width': 2})

    render_candlestick(
        plot_df.rename(columns={
            'market_open_proxy': 'open',
            'market_high_proxy': 'high',
            'market_low_proxy': 'low',
            'market_close_proxy': 'close',
        }),
        markers=markers,
        lines=lines,
        height=height,
        title=title,
        volume=False,
        theme='dark',
    )


def render_market_regime_index_chart(index_df, market_df, height=520, title="指数日线 + 市场卦背景"):
    """渲染真实指数K线主图，并叠加市场卦背景段与顶部卦信息"""
    if index_df is None or len(index_df) == 0 or market_df is None or len(market_df) == 0:
        return

    plot_df = index_df.copy().sort_values('date').reset_index(drop=True)
    market = market_df.copy().sort_values('date').reset_index(drop=True)

    market_cols = ['date', 'gua_code', 'gua_name', 'changed']
    for extra_col in ['prev_gua_display', 'seg_id', 'seg_day', 'yao_1', 'yao_2', 'yao_3', 'gua_meaning', 'market_trend_55', 'market_trend_anchor_120']:
        if extra_col in market.columns:
            market_cols.append(extra_col)

    merged = plot_df.merge(
        market[market_cols],
        on='date',
        how='left'
    )

    markers = []
    changed_rows = merged[merged['changed'] == 1].copy()
    for _, row in changed_rows.iterrows():
        gua_name = row.get('gua_name', '') or ''
        gua_code = row.get('gua_code', '') or ''
        markers.append({
            'time': str(row['date'])[:10],
            'position': 'aboveBar',
            'color': '#f59e0b',
            'shape': 'circle',
            'text': f"{gua_name}".strip() if gua_name else '',
        })

    lines = []
    if 'market_trend_55' in merged.columns:
        trend_data = []
        for _, row in merged.dropna(subset=['market_trend_55']).iterrows():
            trend_data.append({
                'time': str(row['date'])[:10],
                'value': round(float(row['market_trend_55']), 3),
            })
        if trend_data:
            lines.append({'name': 'trend55', 'data': trend_data, 'color': '#f59e0b', 'width': 2})

    if 'market_trend_anchor_120' in merged.columns:
        anchor_data = []
        for _, row in merged.dropna(subset=['market_trend_anchor_120']).iterrows():
            anchor_data.append({
                'time': str(row['date'])[:10],
                'value': round(float(row['market_trend_anchor_120']), 3),
            })
        if anchor_data:
            lines.append({'name': 'anchor120', 'data': anchor_data, 'color': '#60a5fa', 'width': 2})

    segments = []
    if 'seg_id' in merged.columns:
        seg_df = merged.dropna(subset=['seg_id', 'gua_code']).copy()
        if len(seg_df) > 0:
            for _, grp in seg_df.groupby('seg_id', sort=True):
                grp = grp.sort_values('date')
                last = grp.iloc[-1]
                gua_code = str(last.get('gua_code', '')).zfill(3)
                color = 'rgba(255,255,255,0.06)'
                if gua_code == '000':
                    color = 'rgba(34,197,94,0.14)'
                elif gua_code == '001':
                    color = 'rgba(134,239,172,0.14)'
                elif gua_code == '010':
                    color = 'rgba(74,222,128,0.14)'
                elif gua_code == '011':
                    color = 'rgba(245,158,11,0.14)'
                elif gua_code == '100':
                    color = 'rgba(239,68,68,0.14)'
                elif gua_code == '101':
                    color = 'rgba(251,146,60,0.14)'
                elif gua_code == '110':
                    color = 'rgba(167,139,250,0.14)'
                elif gua_code == '111':
                    color = 'rgba(239,68,68,0.18)'
                segments.append({
                    'start_time': str(grp.iloc[0]['date'])[:10],
                    'end_time': str(grp.iloc[-1]['date'])[:10],
                    'seg_id': int(float(last['seg_id'])) if str(last['seg_id']) != 'nan' else None,
                    'gua_code': gua_code,
                    'gua_name': last.get('gua_name', ''),
                    'color': color,
                })

    header_info = {}
    header_points = []
    valid_rows = merged.dropna(subset=['gua_code'])
    if len(valid_rows) > 0:
        for _, row in valid_rows.iterrows():
            gua_name = row.get('gua_name', '') or ''
            gua_code = str(row.get('gua_code', '')).zfill(3)
            meaning = row.get('gua_meaning', '') or ''
            prev_gua = row.get('prev_gua_display', '') or ''
            prev_part = f"前卦 {prev_gua} → {gua_code}" if prev_gua else f"卦码 {gua_code}"
            yao_values = []
            for col in ['yao_1', 'yao_2', 'yao_3']:
                val = row.get(col)
                yao_values.append(str(int(float(val))) if str(val) != 'nan' else '-')
            seg_day = row.get('seg_day')
            seg_day_text = ''
            if str(seg_day) != 'nan':
                seg_day_text = f"段内第{int(float(seg_day))}天"
            point_info = {
                'time': str(row['date'])[:10],
                'line1': f"{gua_name} {gua_code} · {meaning}".strip(' ·'),
                'line2': f"{prev_part} | 爻 {'/'.join(yao_values)}" + (f" | {seg_day_text}" if seg_day_text else ''),
            }
            header_points.append(point_info)
        header_info = header_points[-1]

    render_candlestick(
        plot_df,
        markers=markers,
        lines=lines,
        height=height,
        title=title,
        volume=False,
        theme='dark',
        segments=segments,
        header_info=header_info,
        header_points=header_points,
    )



def render_equity_chart(df_equity, height=400, title="策略净值"):
    """
    渲染净值曲线 (用 LWC Area 图, 比 Plotly 更流畅)

    Args:
        df_equity: DataFrame, 必须含 date, nav 列
        height: 图表高度
        title: 标题
    """
    area_data = []
    for _, row in df_equity.iterrows():
        area_data.append({
            'time': str(row['date'])[:10],
            'value': round(float(row['nav']), 4),
        })

    area_json = json.dumps(area_data)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="{LWC_CDN}"></script>
        <style>
            body {{ margin: 0; padding: 0; background: #1a1a2e; }}
            #eq-container {{ width: 100%; height: {height}px; }}
            .eq-title {{
                color: #d1d5db; font-size: 14px; font-weight: bold;
                padding: 8px 0 4px 12px; font-family: Microsoft YaHei, sans-serif;
            }}
        </style>
    </head>
    <body>
        <div class="eq-title">{title}</div>
        <div id="eq-container"></div>
        <script>
            var chart = LightweightCharts.createChart(
                document.getElementById('eq-container'), {{
                layout: {{
                    background: {{ type: 'solid', color: '#1a1a2e' }},
                    textColor: '#d1d5db',
                    fontFamily: 'Microsoft YaHei, sans-serif',
                }},
                grid: {{
                    vertLines: {{ color: 'rgba(42,46,57,0.5)' }},
                    horzLines: {{ color: 'rgba(42,46,57,0.5)' }},
                }},
                rightPriceScale: {{
                    borderColor: 'rgba(42,46,57,0.8)',
                }},
                timeScale: {{
                    borderColor: 'rgba(42,46,57,0.8)',
                    timeVisible: false,
                }},
            }});

            var areaSeries = chart.addAreaSeries({{
                topColor: 'rgba(245,158,11,0.4)',
                bottomColor: 'rgba(245,158,11,0.05)',
                lineColor: '#f59e0b',
                lineWidth: 2,
                crosshairMarkerVisible: true,
                crosshairMarkerRadius: 4,
            }});

            areaSeries.setData({area_json});

            // 基准线 1.0
            areaSeries.createPriceLine({{
                price: 1.0,
                color: '#666666',
                lineWidth: 1,
                lineStyle: 2,
                axisLabelVisible: true,
                title: '基准',
            }});

            chart.timeScale().fitContent();
        </script>
    </body>
    </html>
    """
    components.html(html, height=height + 40, scrolling=False)
