# -*- coding: utf-8 -*-
"""
八卦罗盘组件 — 6+2 市场周期圆盘

布局 (上牛下熊、左右震荡):

              乾(111) 如日中天        ← 12点 = 牛顶
             /   12:00   \\
    巽(011) /              \\ 兑(110)
    启动  10:00            2:00  分歧
           /                \\
  [坎(010)]    上涨 ← → 下跌   [离(101)]  ← 9点/3点 = 过渡震荡
   过渡  9:00    ← ● →    3:00  过渡
           \\                /
    艮(001) \\              / 震(100)
    蓄力   8:00          4:00  崩塌
             \\   6:00   /
              坤(000) 至暗时刻        ← 6点 = 熊底

6+2 模型:
  主路径(6步): 乾→兑→震→坤→艮→巽→乾
  过渡态(2个): 离(101)=下跌过渡震荡, 坎(010)=上涨过渡震荡

三层卦 (同一公式, 不同频率):
  年卦(时针): 月线趋势线MA12位置 / MA12的6月变化 / 月线主力线MA12
  月卦(分针): 月线趋势线位置 / 趋势线3月变化 / 月线主力线
  日卦(秒针): 日线趋势线MA30位置 / MA30的10日变化 / 日线主力线MA10

牛熊周期 (每次只变1爻):
  牛市: 坤→艮→巽→乾 (主力先行→方向转升→位置到高)
  熊市: 乾→兑→震→坤 (主力撤退→方向转降→位置到低)
"""

import math
import json

# === 八卦定义 ===
BAGUA_COMPASS = {
    '111': {'name': '乾', 'meaning': '如日中天', 'nature': '天',
            'color': '#ef4444', 'detail': '高位+上升+主力强', 'role': '主路·牛顶'},
    '110': {'name': '兑', 'meaning': '分歧初现', 'nature': '泽',
            'color': '#a78bfa', 'detail': '高位+上升+主力弱', 'role': '主路·下跌起点'},
    '101': {'name': '离', 'meaning': '主力护盘', 'nature': '火',
            'color': '#fb923c', 'detail': '高位+下降+主力强', 'role': '过渡·下跌震荡'},
    '100': {'name': '震', 'meaning': '雷霆坠落', 'nature': '雷',
            'color': '#ef4444', 'detail': '高位+下降+主力弱', 'role': '主路·崩塌'},
    '011': {'name': '巽', 'meaning': '风起云涌', 'nature': '风',
            'color': '#f59e0b', 'detail': '低位+上升+主力强', 'role': '主路·启动'},
    '010': {'name': '坎', 'meaning': '方向试探', 'nature': '水',
            'color': '#4ade80', 'detail': '低位+上升+主力弱', 'role': '过渡·上涨震荡'},
    '001': {'name': '艮', 'meaning': '底部蓄力', 'nature': '山',
            'color': '#86efac', 'detail': '低位+下降+主力强', 'role': '主路·上涨起点'},
    '000': {'name': '坤', 'meaning': '至暗时刻', 'nature': '地',
            'color': '#22c55e', 'detail': '低位+下降+主力弱', 'role': '主路·熊底'},
}

# 6+2 圆盘排列: 按时钟位置 (上牛下熊, 左右震荡)
# 12点→2点→3点→4点→6点→8点→9点→10点
COMPASS_ORDER = ['111', '110', '101', '100', '000', '001', '010', '011']
#                 乾12   兑2    离3    震4    坤6    艮8    坎9    巽10

# 每个卦在时钟上的角度 (0=12点, 顺时针)
COMPASS_ANGLES = {
    '111':   0,   # 乾 12点
    '110':  60,   # 兑  2点
    '101':  90,   # 离  3点 (过渡态)
    '100': 120,   # 震  4点
    '000': 180,   # 坤  6点
    '001': 240,   # 艮  8点
    '010': 270,   # 坎  9点 (过渡态)
    '011': 300,   # 巽 10点
}


from data_layer.gua_data import clean_gua as _clean_gua_code


def _gua_angle(gua_code):
    """获取卦在圆盘上的角度"""
    gua_code = _clean_gua_code(gua_code)
    return COMPASS_ANGLES.get(gua_code, 0)


def render_unified_compass(dates_gua_list, initial_index=None):
    """
    渲染统一的八卦罗盘组件 (HTML+Canvas+JS)

    - 不选日期范围时: 显示今天, 三个圆盘静态
    - 选日期范围后: 可播放回测, 指针动起来

    Args:
        dates_gua_list: list of dict, 每个dict包含:
            {date, year_gua, month_gua, day_gua}
        initial_index: 初始显示第几天 (None=最后一天即最新)
    """
    # 预处理数据
    js_data = []
    for item in dates_gua_list:
        yg = _clean_gua_code(item['year_gua'])
        mg = _clean_gua_code(item['month_gua'])
        dg = _clean_gua_code(item['day_gua'])
        js_data.append({
            'd': item['date'],
            'ya': COMPASS_ANGLES.get(yg, 0), 'yg': yg,
            'ma': COMPASS_ANGLES.get(mg, 0), 'mg': mg,
            'da': COMPASS_ANGLES.get(dg, 0), 'dg': dg,
        })

    data_json = json.dumps(js_data, ensure_ascii=False)

    # 八卦信息 → JS
    bagua_js = {}
    for code, info in BAGUA_COMPASS.items():
        bagua_js[code] = {
            'name': info['name'], 'color': info['color'],
            'meaning': info['meaning'],
            'role': info['role'], 'nature': info['nature'],
        }
    bagua_json = json.dumps(bagua_js, ensure_ascii=False)

    # 圆盘上各卦位置 → JS
    positions = []
    for code in COMPASS_ORDER:
        info = BAGUA_COMPASS[code]
        a_deg = COMPASS_ANGLES[code]
        is_transition = code in ('101', '010')  # 过渡态
        positions.append({
            'code': code,
            'name': info['name'],
            'color': info['color'],
            'angle': a_deg,
            'role': info['role'],
            'transition': is_transition,
        })
    positions_json = json.dumps(positions, ensure_ascii=False)

    if initial_index is None:
        initial_index = len(js_data) - 1

    # 三爻行为描述 → JS (按年/月/日分层)
    yao_desc_js = json.dumps({
        'year': {
            '1__0': '高位(月MA12≥50)', '0__0': '低位(月MA12<50)',
            '1__1': '上升(6月变化>0)',  '0__1': '下降(6月变化≤0)',
            '1__2': '主强(主力MA12≥0)', '0__2': '主弱(主力MA12<0)',
        },
        'month': {
            '1__0': '高位(月趋势≥50)', '0__0': '低位(月趋势<50)',
            '1__1': '上升(3月变化>0)',  '0__1': '下降(3月变化≤0)',
            '1__2': '主强(月主力≥0)',   '0__2': '主弱(月主力<0)',
        },
        'day': {
            '1__0': '高位(MA30≥50)',   '0__0': '低位(MA30<50)',
            '1__1': '上升(10日变化>0)', '0__1': '下降(10日变化≤0)',
            '1__2': '主强(主力MA10≥0)', '0__2': '主弱(主力MA10<0)',
        },
        'short': {
            '1__0': '高位', '0__0': '低位',
            '1__1': '上升', '0__1': '下降',
            '1__2': '主强', '0__2': '主弱',
        },
    }, ensure_ascii=False)

    S = 240  # 单个圆盘尺寸
    is_single = len(js_data) == 1

    html = f'''
    <div id="bagua-compass" style="background:#0d0d1a; border-radius:12px;
         padding:20px; margin:10px 0; font-family: Microsoft YaHei, SimHei, sans-serif;">

        <!-- 日期与综合状态 -->
        <div style="text-align:center; margin-bottom:6px;">
            <span id="bc-date" style="color:#ddd; font-size:20px; font-weight:bold;
                  font-family:monospace;"></span>
        </div>
        <div style="text-align:center; margin-bottom:12px;">
            <span id="bc-overall" style="color:#888; font-size:15px;"></span>
        </div>

        <!-- 三个Canvas -->
        <div style="display:flex; justify-content:center; align-items:flex-start;
                    gap:10px; flex-wrap:wrap;">
            <div style="text-align:center;">
                <div style="color:#ef4444; font-size:13px; font-weight:bold;
                     margin-bottom:4px;">年 · 大周期 (时针)</div>
                <canvas id="cv-y" width="{S}" height="{S}"></canvas>
                <div id="lb-y" style="font-size:14px; font-weight:bold; margin-top:4px;">-</div>
            </div>
            <div style="text-align:center;">
                <div style="color:#f59e0b; font-size:13px; font-weight:bold;
                     margin-bottom:4px;">月 · 中趋势 (分针)</div>
                <canvas id="cv-m" width="{S}" height="{S}"></canvas>
                <div id="lb-m" style="font-size:14px; font-weight:bold; margin-top:4px;">-</div>
            </div>
            <div style="text-align:center;">
                <div style="color:#3b82f6; font-size:13px; font-weight:bold;
                     margin-bottom:4px;">日 · 短波动 (秒针)</div>
                <canvas id="cv-d" width="{S}" height="{S}"></canvas>
                <div id="lb-d" style="font-size:14px; font-weight:bold; margin-top:4px;">-</div>
            </div>
        </div>

        <!-- 卦变 -->
        <div id="bc-change" style="text-align:center; margin-top:8px;
             color:#444; font-size:13px; min-height:20px;"></div>

        <!-- 控制条 (回测模式才显示) -->
        <div id="bc-ctrl" style="display:{'none' if is_single else 'flex'};
             align-items:center; gap:10px; margin-top:14px;
             justify-content:center; flex-wrap:wrap;">
            <button onclick="window._bcPrev()" style="{_btn()}" title="上一天">⏮</button>
            <button id="bc-play" onclick="window._bcToggle()" style="{_btn()}" title="播放/暂停">▶</button>
            <button onclick="window._bcNext()" style="{_btn()}" title="下一天">⏭</button>
            <input id="bc-slider" type="range" min="0" max="0" value="0"
                   oninput="window._bcSeek(this.value)"
                   style="flex:1; min-width:180px; max-width:500px; accent-color:#f59e0b;"/>
            <select id="bc-speed" onchange="window._bcSpeed(this.value)" style="{_sel()}">
                <option value="500">慢速</option>
                <option value="200" selected>正常</option>
                <option value="80">快速</option>
                <option value="30">极速</option>
            </select>
            <span id="bc-prog" style="color:#666; font-size:12px; min-width:80px;">-</span>
        </div>

        <!-- 三爻注释 + 八卦说明 -->
        <details style="margin-top:16px; cursor:pointer;">
            <summary style="color:#888; font-size:13px; padding:6px 0;
                     user-select:none;">三爻编码规则 · 八卦含义 · 6+2模型</summary>
            <div id="bc-legend" style="margin-top:8px;"></div>
        </details>
    </div>

    <script>
    (function() {{
        const DATA = {data_json};
        const B = {bagua_json};
        const POS = {positions_json};
        const YAO_DESC = {yao_desc_js};
        const S = {S};
        const CX = S/2, CY = S/2;
        const RO = S*0.44;   // 外圆
        const RI = S*0.30;   // 内圆
        const RT = S*0.37;   // 卦名半径
        const RP = S*0.32;   // 指针长度

        let idx = {initial_index};
        let playing = false, timer = null, speed = 200;
        const slider = document.getElementById('bc-slider');
        slider.max = DATA.length - 1;
        slider.value = idx;

        const cvY = document.getElementById('cv-y').getContext('2d');
        const cvM = document.getElementById('cv-m').getContext('2d');
        const cvD = document.getElementById('cv-d').getContext('2d');

        // --- 绘制单个圆盘 ---
        function draw(ctx, angle, pColor, curCode, layer) {{
            ctx.clearRect(0, 0, S, S);

            // 牛熊分区背景 (上半=牛区淡红, 下半=熊区淡绿)
            // 上半圆 (牛)
            ctx.beginPath();
            ctx.arc(CX, CY, RO, Math.PI, 0);
            ctx.lineTo(CX+RO, CY);
            ctx.arc(CX, CY, RI, 0, Math.PI, true);
            ctx.closePath();
            ctx.fillStyle = 'rgba(239,68,68,0.04)';
            ctx.fill();
            // 下半圆 (熊)
            ctx.beginPath();
            ctx.arc(CX, CY, RO, 0, Math.PI);
            ctx.arc(CX, CY, RI, Math.PI, 0, true);
            ctx.closePath();
            ctx.fillStyle = 'rgba(34,197,94,0.04)';
            ctx.fill();

            // 外圆 + 内圆
            ctx.beginPath(); ctx.arc(CX, CY, RO, 0, Math.PI*2);
            ctx.strokeStyle = '#444'; ctx.lineWidth = 2; ctx.stroke();
            ctx.beginPath(); ctx.arc(CX, CY, RI, 0, Math.PI*2);
            ctx.strokeStyle = '#333'; ctx.lineWidth = 1; ctx.stroke();

            // 牛/熊标注
            ctx.font = (S*0.04) + 'px Microsoft YaHei, SimHei, sans-serif';
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillStyle = 'rgba(239,68,68,0.25)';
            ctx.fillText('牛', CX, CY - RI*0.55);
            ctx.fillStyle = 'rgba(34,197,94,0.25)';
            ctx.fillText('熊', CX, CY + RI*0.55);
            ctx.fillStyle = 'rgba(150,150,150,0.18)';
            ctx.fillText('震荡', CX - RI*0.65, CY);
            ctx.fillText('震荡', CX + RI*0.65, CY);

            // 八个扇区
            for (let i = 0; i < POS.length; i++) {{
                const p = POS[i];
                const isCur = (p.code === curCode);
                const aRad = (p.angle - 90) * Math.PI / 180;

                // 找相邻扇区的边界角度
                const prevIdx = (i - 1 + POS.length) % POS.length;
                const nextIdx = (i + 1) % POS.length;
                const prevA = POS[prevIdx].angle;
                const nextA = POS[nextIdx].angle;

                // 扇区起止角 (处理环绕)
                let startDeg = p.angle - angleDiff(prevA, p.angle) / 2;
                let endDeg = p.angle + angleDiff(p.angle, nextA) / 2;
                const startRad = (startDeg - 90) * Math.PI / 180;
                const endRad = (endDeg - 90) * Math.PI / 180;

                // 分割线
                ctx.beginPath();
                ctx.moveTo(CX + RI * Math.cos(startRad), CY + RI * Math.sin(startRad));
                ctx.lineTo(CX + RO * Math.cos(startRad), CY + RO * Math.sin(startRad));
                ctx.strokeStyle = p.transition ? '#444' : '#333';
                ctx.lineWidth = p.transition ? 1.5 : 1;
                ctx.stroke();

                // 高亮当前扇区
                if (isCur) {{
                    ctx.beginPath();
                    ctx.arc(CX, CY, RO, startRad, endRad);
                    ctx.arc(CX, CY, RI, endRad, startRad, true);
                    ctx.closePath();
                    ctx.fillStyle = p.color + '30';
                    ctx.fill();
                    // 外弧高亮线
                    ctx.beginPath();
                    ctx.arc(CX, CY, RO, startRad, endRad);
                    ctx.strokeStyle = p.color;
                    ctx.lineWidth = 3;
                    ctx.stroke();
                }}

                // 过渡态标记 (虚线弧)
                if (p.transition && !isCur) {{
                    ctx.beginPath();
                    ctx.arc(CX, CY, RO-1, startRad, endRad);
                    ctx.strokeStyle = '#555';
                    ctx.lineWidth = 1;
                    ctx.setLineDash([3, 3]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }}

                // 卦名
                const tx = CX + RT * Math.cos(aRad);
                const ty = CY + RT * Math.sin(aRad);
                const fs = isCur ? S*0.08 : (p.transition ? S*0.055 : S*0.065);
                ctx.font = (isCur ? 'bold ' : '') + fs + 'px Microsoft YaHei, SimHei, sans-serif';
                ctx.fillStyle = isCur ? p.color : (p.transition ? '#666' : '#999');
                ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText(p.name, tx, ty);
            }}

            // 指针
            const pRad = (angle - 90) * Math.PI / 180;
            const px = CX + RP * Math.cos(pRad);
            const py = CY + RP * Math.sin(pRad);
            const tailL = S * 0.06;
            const tx2 = CX - tailL * Math.cos(pRad);
            const ty2 = CY - tailL * Math.sin(pRad);
            const perpR = pRad + Math.PI / 2;
            const w = S * 0.022;

            ctx.beginPath();
            ctx.moveTo(px, py);
            ctx.lineTo(CX + w*Math.cos(perpR), CY + w*Math.sin(perpR));
            ctx.lineTo(tx2, ty2);
            ctx.lineTo(CX - w*Math.cos(perpR), CY - w*Math.sin(perpR));
            ctx.closePath();
            ctx.fillStyle = pColor; ctx.globalAlpha = 0.9; ctx.fill();
            ctx.globalAlpha = 1.0;

            // 指针尖端点
            ctx.beginPath(); ctx.arc(px, py, S*0.014, 0, Math.PI*2);
            ctx.fillStyle = pColor; ctx.fill();

            // 中心轴
            ctx.beginPath(); ctx.arc(CX, CY, S*0.055, 0, Math.PI*2);
            ctx.fillStyle = '#1a1a2e'; ctx.strokeStyle = pColor; ctx.lineWidth = 2;
            ctx.fill(); ctx.stroke();
            ctx.beginPath(); ctx.arc(CX, CY, S*0.02, 0, Math.PI*2);
            ctx.fillStyle = pColor; ctx.fill();

            // 三爻行为标注 (圆盘外围四角, 按层级显示)
            if (curCode) {{
                const yd = YAO_DESC[layer] || YAO_DESC['short'];
                const y1 = curCode[2], y2 = curCode[1], y3 = curCode[0];
                const d1 = yd[y1+'__0'] || '?';
                const d2 = yd[y2+'__1'] || '?';
                const d3 = yd[y3+'__2'] || '?';
                const fs = S * 0.04;
                ctx.font = fs + 'px Microsoft YaHei, SimHei, sans-serif';
                ctx.textBaseline = 'middle';
                // 左上: 三爻(势)
                ctx.textAlign = 'left';
                ctx.fillStyle = y3==='1' ? 'rgba(239,68,68,0.55)' : 'rgba(34,197,94,0.55)';
                ctx.fillText('势·'+d3, 4, fs*1.2);
                // 右上: 二爻(用)
                ctx.textAlign = 'right';
                ctx.fillStyle = y2==='1' ? 'rgba(239,68,68,0.55)' : 'rgba(34,197,94,0.55)';
                ctx.fillText('用·'+d2, S-4, fs*1.2);
                // 左下: 初爻(体)
                ctx.textAlign = 'left';
                ctx.fillStyle = y1==='1' ? 'rgba(239,68,68,0.55)' : 'rgba(34,197,94,0.55)';
                ctx.fillText('体·'+d1, 4, S-fs*0.6);
            }}
        }}

        function angleDiff(a, b) {{
            let d = b - a;
            while (d < 0) d += 360;
            while (d > 360) d -= 360;
            return d || 360;
        }}

        // --- 渲染帧 ---
        function render(i) {{
            if (i < 0 || i >= DATA.length) return;
            const d = DATA[i];

            draw(cvY, d.ya, '#ef4444', d.yg, 'year');
            draw(cvM, d.ma, '#f59e0b', d.mg, 'month');
            draw(cvD, d.da, '#3b82f6', d.dg, 'day');

            document.getElementById('bc-date').textContent = d.d;

            // 标签
            const yi = B[d.yg], mi = B[d.mg], di = B[d.dg];
            setLabel('lb-y', yi, d.yg, 'year'); setLabel('lb-m', mi, d.mg, 'month'); setLabel('lb-d', di, d.dg, 'day');

            // 综合判断
            const positions = [d.yg, d.mg, d.dg].map(g => g ? g[0] : '0');
            const upCount = positions.filter(x => x === '1').length;
            let ov = '';
            if (upCount === 3) ov = '三层高位 — 全面看多';
            else if (upCount === 0) ov = '三层低位 — 全面看空';
            else if (upCount === 2) ov = '偏强格局 — 整体向好';
            else ov = '偏弱格局 — 整体承压';
            document.getElementById('bc-overall').textContent = ov;

            // 卦变
            if (i > 0) {{
                const p = DATA[i-1];
                let ch = [];
                if (p.yg !== d.yg) ch.push('年: ' + B[p.yg]?.name + '→' + B[d.yg]?.name);
                if (p.mg !== d.mg) ch.push('月: ' + B[p.mg]?.name + '→' + B[d.mg]?.name);
                if (p.dg !== d.dg) ch.push('日: ' + B[p.dg]?.name + '→' + B[d.dg]?.name);
                const el = document.getElementById('bc-change');
                el.textContent = ch.length > 0 ? '卦变: ' + ch.join(' | ') : '';
                el.style.color = ch.length > 0 ? '#f59e0b' : '#444';
            }} else {{
                document.getElementById('bc-change').textContent = '';
            }}

            slider.value = i;
            document.getElementById('bc-prog').textContent = (i+1) + ' / ' + DATA.length;
        }}

        function setLabel(id, info, guaCode, layer) {{
            const el = document.getElementById(id);
            if (!info || !guaCode) {{ el.textContent = '-'; return; }}
            // 三爻行为解读 (按层级)
            const yd = YAO_DESC[layer] || YAO_DESC['short'];
            const y1 = guaCode[2], y2 = guaCode[1], y3 = guaCode[0]; // 初爻=末位, 三爻=首位
            const d1 = yd[y1 + '__0'] || '?'; // 初爻: 位置
            const d2 = yd[y2 + '__1'] || '?'; // 二爻: 方向
            const d3 = yd[y3 + '__2'] || '?'; // 三爻: 主力
            const y1c = y1==='1' ? '#ef4444' : '#22c55e';
            const y2c = y2==='1' ? '#ef4444' : '#22c55e';
            const y3c = y3==='1' ? '#ef4444' : '#22c55e';
            el.innerHTML = '<span style="color:' + info.color + ';font-size:15px;">' + info.name +
                           '</span> <span style="color:#888;font-size:12px;">' +
                           info.meaning + '</span>' +
                           '<div style="margin-top:3px;font-size:11px;line-height:1.5;color:#777;">' +
                           '<span style="color:'+y1c+'">●</span> ' + d1 +
                           ' <span style="color:#444">|</span> ' +
                           '<span style="color:'+y2c+'">●</span> ' + d2 +
                           ' <span style="color:#444">|</span> ' +
                           '<span style="color:'+y3c+'">●</span> ' + d3 +
                           '</div>';
        }}

        // --- 播放控制 ---
        function step() {{
            if (idx < DATA.length - 1) {{ idx++; render(idx); }}
            else {{ window._bcStop(); }}
        }}
        window._bcToggle = function() {{
            if (playing) window._bcStop();
            else {{
                playing = true;
                document.getElementById('bc-play').textContent = '⏸';
                timer = setInterval(step, speed);
            }}
        }};
        window._bcStop = function() {{
            playing = false;
            document.getElementById('bc-play').textContent = '▶';
            if (timer) {{ clearInterval(timer); timer = null; }}
        }};
        window._bcNext = function() {{ window._bcStop(); if (idx < DATA.length-1) {{ idx++; render(idx); }} }};
        window._bcPrev = function() {{ window._bcStop(); if (idx > 0) {{ idx--; render(idx); }} }};
        window._bcSeek = function(v) {{ window._bcStop(); idx = parseInt(v); render(idx); }};
        window._bcSpeed = function(v) {{
            speed = parseInt(v);
            if (playing) {{ clearInterval(timer); timer = setInterval(step, speed); }}
        }};

        // --- 注释说明 ---
        function buildLegend() {{
            let h = '';

            // 三层指标对比表
            h += '<div style="color:#ddd;font-size:14px;font-weight:bold;margin-bottom:8px;border-bottom:1px solid #333;padding-bottom:6px;">三层卦 · 同一公式 不同频率</div>';
            h += '<table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:14px;">';
            h += '<tr style="color:#888;border-bottom:1px solid #333;">';
            h += '<th style="padding:5px;text-align:left;"></th>';
            h += '<th style="padding:5px;text-align:center;color:#ef4444;">年卦 (时针)</th>';
            h += '<th style="padding:5px;text-align:center;color:#f59e0b;">月卦 (分针)</th>';
            h += '<th style="padding:5px;text-align:center;color:#3b82f6;">日卦 (秒针)</th>';
            h += '</tr>';
            const rows = [
                ['数据频率', '月线OHLC', '月线OHLC', '日线OHLC'],
                ['初爻·体(位置)', '趋势线MA12 ≥50', '趋势线 ≥50', 'MA30 ≥50'],
                ['二爻·用(方向)', 'MA12的6月变化 >0', '趋势线3月变化 >0', 'MA30的10日变化 >0'],
                ['三爻·势(主力)', '主力线MA12 ≥0', '主力线 ≥0', '主力线MA10 ≥0'],
                ['更新频率', '月末更新', '月末更新', '每日更新'],
                ['周期含义', '牛熊大周期', '中期震荡趋势', '短期波动节奏'],
            ];
            rows.forEach((r, ri) => {{
                const bg = ri % 2 === 0 ? '#0f0f1f' : '#111';
                const isHeader = ri === 0 || ri === 4 || ri === 5;
                h += '<tr style="background:'+bg+';">';
                h += '<td style="padding:4px 5px;color:#aaa;font-weight:bold;">'+r[0]+'</td>';
                for (let c = 1; c <= 3; c++) {{
                    const clr = isHeader ? '#ccc' : '#999';
                    h += '<td style="padding:4px 5px;text-align:center;color:'+clr+';">'+r[c]+'</td>';
                }}
                h += '</tr>';
            }});
            h += '</table>';



            // 6+2 模型
            h += '<div style="color:#ddd;font-size:14px;font-weight:bold;margin:10px 0 8px;border-bottom:1px solid #333;padding-bottom:6px;">6+2 市场周期模型</div>';
            h += '<div style="color:#aaa;font-size:12px;margin-bottom:10px;">';
            h += '主路径(6步): 乾→兑→震→坤→艮→巽→乾<br/>';
            h += '过渡态(2个): <span style="color:#fb923c;">离</span>(下跌震荡) + <span style="color:#4ade80;">坎</span>(上涨震荡)<br/>';
            h += '牛市启动: 先主力进场(三爻=1) → 再方向转升(二爻=1) → 最后位置到高位(初爻=1)<br/>';
            h += '熊市启动: 先主力撤退(三爻=0) → 再方向转降(二爻=0) → 最后位置到低位(初爻=0)</div>';

            // 八卦卡片
            h += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;">';
            POS.forEach(p => {{
                const info = B[p.code];
                
                
                
                
                const border = p.transition ? '1px dashed '+info.color : '1px solid #222';
                h += '<div style="background:#111;border-radius:6px;padding:8px;border-left:3px solid '+info.color+';border:'+border+';">';
                h += '<div style="display:flex;justify-content:space-between;align-items:center;">';
                h += '<span style="color:'+info.color+';font-weight:bold;font-size:15px;">'+info.name+'</span>';
                
                h += '</div>';
                h += '<div style="color:#aaa;font-size:11px;margin-top:1px;">'+info.nature+' | '+p.code+'</div>';
                h += '<div style="color:#ddd;font-size:12px;margin-top:3px;font-weight:bold;">'+info.meaning+'</div>';
                h += '<div style="color:#666;font-size:11px;">'+info.role+'</div>';
                
                h += '</div>';
            }});
            h += '</div>';
            document.getElementById('bc-legend').innerHTML = h;
        }}

        // 初始化
        buildLegend();
        render(idx);
    }})();
    </script>
    '''
    return html


def _btn():
    return ("background:#222; color:#ddd; border:1px solid #444; border-radius:6px; "
            "padding:6px 14px; cursor:pointer; font-size:16px;")

def _sel():
    return ("background:#222; color:#ddd; border:1px solid #444; border-radius:6px; "
            "padding:4px 8px; font-size:13px; cursor:pointer;")
