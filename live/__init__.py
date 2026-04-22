# -*- coding: utf-8 -*-
"""
乾坤三线 v1.0 — miniQMT 实盘交易模块

模块结构:
  config.py         策略配置中心 (所有可调参数)
  signal_engine.py   信号引擎 (买入/卖出信号生成)
  risk_manager.py    风控模块 (仓位控制+八卦过滤)
  trade_logger.py    交易日志持久化
  qmt_strategy.py    miniQMT 主策略入口
"""
