# -*- coding: utf-8 -*-
"""
本地模拟券商（paper trading）—— miniQMT 的 drop-in 替身。

目的：在没有 QMT / 真实账户时，把实盘脚本的完整下单闭环（查资产→下单→成交→
      更新持仓→跨月保存）先在本地跑通。成交规则与回测一致（手续费+滑点）。
状态持久化到 JSON，所以每月运行一次、持仓会累积，行为贴近真实账户。

接入真实 miniQMT 时无需改这里：把 etf_momentum_live.py 的 BROKER 改成 "qmt" 即可。
"""
import json     # 把账户状态存成 JSON 文本文件
import os        # 判断状态文件是否已存在（os.path.exists）


class PaperTrader:
    def __init__(self, state_path, init_cash, commission, slippage, lot=100):
        # 这些参数存成实例属性（self.xxx），后续方法里都能用
        self.path = state_path          # 账户状态 JSON 的保存路径
        self.commission = commission    # 手续费率（单边）
        self.slippage = slippage        # 滑点率（单边）
        self.lot = lot                  # 最小交易单位（A股ETF为100股），此处暂未强制
        self._px = {}                   # {纯代码: 现价}，由 set_prices 注入，用于估值/成交
        # 状态持久化：文件已存在就读回上次的现金和持仓（跨次运行接着用）；
        # 不存在就是首次运行，用初始资金建一个空仓账户并落盘。
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                s = json.load(f)        # 把 JSON 文本解析回 Python 字典
            self.cash = s["cash"]
            # 只保留数量>0 的持仓（过滤掉历史上清过仓、值为0的残留键）
            self.positions_ = {k: v for k, v in s["positions"].items() if v > 0}
        else:
            self.cash = float(init_cash)
            self.positions_ = {}        # {纯代码: 持仓股数}
            self._save()

    # ---- 行情注入（真实 QMT 这步由券商提供，这里手动喂收盘价） ----
    def set_prices(self, price: dict):
        self._px = dict(price)          # 拷贝一份，避免外部字典被改时影响内部状态

    # ---- 查询（对应 xttrader.query_stock_asset / query_stock_positions） ----
    def total_asset(self):
        # 总资产 = 现金 + 所有持仓的市值（数量×现价）
        mv = sum(vol * self._px.get(code, 0.0) for code, vol in self.positions_.items())
        return self.cash + mv

    def positions(self):
        # 返回当前持仓（再过滤一次0，保证对外只暴露真实持仓）
        return {c: v for c, v in self.positions_.items() if v > 0}

    # ---- 下单（对应 xttrader.order_stock，这里同步撮合，立即成交） ----
    def order(self, code, side, shares):
        code = code.split(".")[0]       # "510300.SH" → "510300"，内部统一按纯代码存
        px = self._px.get(code)
        if not px or shares <= 0:       # 没有现价或股数非正 → 直接忽略这笔
            return
        if side == "BUY":
            fill = px * (1 + self.slippage)     # 买入成交价：现价上浮一个滑点（买得更贵）
            cost = fill * shares                 # 买入花的钱
            fee = cost * self.commission         # 手续费
            self.cash -= (cost + fee)            # 现金减少
            # 持仓增加：dict.get(code, 0) 取已有持仓，没有则按0算，再加新买的
            self.positions_[code] = self.positions_.get(code, 0) + shares
        else:  # SELL
            held = self.positions_.get(code, 0)
            shares = min(shares, held)           # 最多只能卖出已持有的数量
            if shares <= 0:
                return
            fill = px * (1 - self.slippage)      # 卖出成交价：现价下浮一个滑点（卖得更便宜）
            proceeds = fill * shares             # 卖出收到的钱
            fee = proceeds * self.commission
            self.cash += (proceeds - fee)        # 现金增加
            self.positions_[code] = held - shares
            if self.positions_[code] == 0:       # 清仓后把这个键删掉，保持字典干净
                del self.positions_[code]
        self._save()                             # 每笔成交后立即落盘，防程序中断丢状态

    def _save(self):
        # 把现金+持仓写回 JSON 文件
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"cash": self.cash, "positions": self.positions_},
                      f, ensure_ascii=False, indent=2)   # 中文不转义、缩进2格便于人读
