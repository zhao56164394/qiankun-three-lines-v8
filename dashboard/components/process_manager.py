# -*- coding: utf-8 -*-
"""
进程管理器 — 管理实盘/模拟盘策略进程

运行时文件存放在 live/runtime/:
  {mode}.pid          — 进程 PID
  {mode}.status.json  — 状态信息 (启动时间、心跳等)
  {mode}.log          — stdout/stderr 日志
"""
import os
import sys
import json
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, 'live', 'runtime')
VENV_PYTHON = os.path.join(PROJECT_ROOT, '.venv311', 'Scripts', 'python.exe')
QMT_SCRIPT = os.path.join(PROJECT_ROOT, 'live', 'qmt_strategy.py')


def _ensure_runtime_dir():
    os.makedirs(RUNTIME_DIR, exist_ok=True)


def _pid_file(mode):
    return os.path.join(RUNTIME_DIR, f'{mode}.pid')


def _status_file(mode):
    return os.path.join(RUNTIME_DIR, f'{mode}.status.json')


def _log_file(mode):
    return os.path.join(RUNTIME_DIR, f'{mode}.log')


def _is_process_alive(pid):
    """Windows: 用 tasklist 检测进程是否存活"""
    try:
        result = subprocess.run(
            ['tasklist', '/FI', f'PID eq {pid}', '/NH'],
            capture_output=True, text=True, timeout=5,
        )
        return str(pid) in result.stdout
    except Exception:
        return False


def _cleanup(mode):
    """清理运行时文件"""
    for f in [_pid_file(mode), _status_file(mode)]:
        if os.path.exists(f):
            os.remove(f)


def get_status(mode):
    """
    获取策略运行状态

    Returns:
        dict: {running, pid, start_time, last_heartbeat, mode, ...}
    """
    _ensure_runtime_dir()
    pid_path = _pid_file(mode)

    if not os.path.exists(pid_path):
        return {'running': False, 'mode': mode}

    try:
        with open(pid_path, 'r') as f:
            pid = int(f.read().strip())
    except (ValueError, OSError):
        _cleanup(mode)
        return {'running': False, 'mode': mode}

    # 检测进程是否存活
    if not _is_process_alive(pid):
        # 读取最后的状态信息（可能包含失败原因）
        last_status = {'running': False, 'mode': mode, 'note': '进程已退出'}
        status_path = _status_file(mode)
        if os.path.exists(status_path):
            try:
                with open(status_path, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    last_phase = saved.get('phase', '')
                    if last_phase:
                        last_status['last_phase'] = last_phase
                    last_status['last_heartbeat'] = saved.get('last_heartbeat', '')
            except (json.JSONDecodeError, OSError):
                pass
        _cleanup(mode)
        return last_status

    # 读取状态文件
    status = {'running': True, 'mode': mode, 'pid': pid}
    status_path = _status_file(mode)
    if os.path.exists(status_path):
        try:
            with open(status_path, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                status.update(saved)
                # 心跳中的 PID 是子进程实际 PID，可能与 Popen PID 不同
                child_pid = saved.get('pid')
                if child_pid and child_pid != pid:
                    status['child_pid'] = child_pid
        except (json.JSONDecodeError, OSError):
            pass

    return status


def is_any_running():
    """检查是否有任何策略在运行"""
    for mode in ('live', 'simulate'):
        s = get_status(mode)
        if s['running']:
            return True, mode
    return False, None


def start_strategy(mode):
    """
    启动策略进程

    Args:
        mode: 'live' | 'simulate'

    Returns:
        (bool, str): (成功?, 消息)
    """
    _ensure_runtime_dir()

    # 互斥检查
    other = 'simulate' if mode == 'live' else 'live'
    other_status = get_status(other)
    if other_status['running']:
        other_label = '实盘' if other == 'live' else '模拟盘'
        return False, f'{other_label}正在运行中，请先停止'

    # 已在运行?
    current = get_status(mode)
    if current['running']:
        return False, '已在运行中'

    # 构建命令
    cmd = [VENV_PYTHON, '-u', QMT_SCRIPT]
    if mode == 'simulate':
        cmd.append('--simulate')

    # 启动子进程
    log_path = _log_file(mode)
    try:
        log_fh = open(log_path, 'w', encoding='utf-8')
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

        # 写入 PID
        with open(_pid_file(mode), 'w') as f:
            f.write(str(proc.pid))

        # 写入状态
        status = {
            'mode': mode,
            'pid': proc.pid,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(_status_file(mode), 'w', encoding='utf-8') as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

        label = '实盘' if mode == 'live' else '模拟盘'
        return True, f'{label}已启动 (PID: {proc.pid})'

    except Exception as e:
        return False, f'启动失败: {e}'


def stop_strategy(mode):
    """
    停止策略进程

    Returns:
        (bool, str): (成功?, 消息)
    """
    status = get_status(mode)
    if not status['running']:
        return False, '未在运行'

    pid = status['pid']
    child_pid = status.get('child_pid')

    # 先杀子进程（实际的 python 进程），再杀父进程
    killed = False
    for p in [child_pid, pid]:
        if p and _kill_process(p):
            killed = True

    _cleanup(mode)
    label = '实盘' if mode == 'live' else '模拟盘'
    if killed:
        return True, f'{label}已停止 (PID: {pid})'
    else:
        return False, f'终止进程失败 (PID: {pid})'


def get_log_tail(mode, n_lines=50):
    """读取日志最后 n 行"""
    log_path = _log_file(mode)
    if not os.path.exists(log_path):
        return ''
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        return ''.join(lines[-n_lines:])
    except OSError:
        return ''
def _kill_process(pid):
    """Windows: 终止进程树"""
    try:
        subprocess.run(
            ['taskkill', '/F', '/PID', str(pid), '/T'],
            capture_output=True, text=True, timeout=10,
        )
        return True
    except Exception:
        return False
