# -*- coding: utf-8 -*-
"""
J1 跨市场样本外验证 —— 把 A 股动量轮动的"同一套逻辑 + 同一套参数"原封不动搬到美股 ETF 池上跑。

为什么做这个(对应 IMPROVEMENTS J1):A 股回测里池子含黄金/纳指这些"事后赢家",有池前视嫌疑。
若策略逻辑是普适的(动量 + 波动目标 + 趋势过滤),它在"完全没参与设计的"美股样本上也该有效;
若只在 A 股有效 → A 股的优势很可能是池子/样本特定,不是真 alpha。这是给"池前视"唯一的根治性回答。

设计要点(保证是真正的样本外):
  - 参数零调整:lookbacks=21/63/126、top_n=3、vol_target=0.15、trend_ma=200 全部沿用 A 股部署值。
  - 池子按"资产类别"选,不按收益选:美股大盘/科技/小盘/新兴/黄金/REITs + 长债防守——
    是"一个美国投资者会持有的标准资产类别 ETF",不是挑出来的近期赢家(QQQ/SPY 是规模最大的两只,
    排除它们才不自然)。这条先验对应 B1 的"池成员只能基于先验"教训。
  - 无涨跌停:美股 ETF 无日涨跌停限制,回测不传 cant 掩码(其它口径 T+1/成本照旧)。
  - 公平对照:同池内比"策略 vs 等权持有 vs 等权+风控 vs 买入持有 SPY",无论池怎么选都是同池比较。

数据源(本机 IP 多半被免源封,在能联网的机器跑):
  python crossmarket.py          # 自动按 stooq(PoW)→ yfinance → 本地 usdata/*.csv 顺序尝试

运行:
  python crossmarket.py          # 拉真实美股数据 + 跨市场对照 + bootstrap
"""
import os
import contextlib

import momentum_core as mc
import etf_momentum as e
from etf_momentum import backtest, perf, bench_nav, _print_table, bootstrap_selection, _boot_report

# ---------------- 美股 universe(先验/资产类别驱动,非收益驱动)----------------
POOL_US = {
    "SPY":  "标普500",        # 美大盘(对应沪深300)
    "QQQ":  "纳指100",        # 美科技成长(对应创业板/纳指)
    "IWM":  "罗素2000",       # 美小盘(对应中证1000)
    "EEM":  "新兴市场",       # 国际分散
    "GLD":  "黄金",           # 商品避险(对应518880)
    "VNQ":  "REITs",          # 房地产(额外分散)
}
DEFENSE_US = ("TLT", "美长期国债")   # 防守资产(对应国债511010)
BENCH_US = "SPY"                    # 基准 / 大盘趋势风向标

# stooq 符号映射(canonical → stooq 后缀形式)
STOOQ_SYMS = {"SPY": "spy.us", "QQQ": "qqq.us", "IWM": "iwm.us",
              "EEM": "eem.us", "GLD": "gld.us", "VNQ": "vnq.us", "TLT": "tlt.us"}
_US_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


@contextlib.contextmanager
def _us_globals():
    """临时把 A 股 POOL/DEFENSE/BENCH 换成美股,跑完恢复。

    backtest 用 etf_momentum 模块全局(POOL/BENCH/DEFENSE);decide_targets 用 momentum_core 全局
    (POOL/DEFENSE)。两处都要换。trend_code 不能靠改全局——decide_targets 的默认参数在 def 时已绑定,
    必须通过 backtest 的 trend_code 参数显式传(见 run_crossmarket)。"""
    g = e.__dict__
    saved = (mc.POOL, mc.DEFENSE, e.POOL, e.DEFENSE, e.BENCH)
    mc.POOL = POOL_US
    mc.DEFENSE = DEFENSE_US
    g["POOL"] = POOL_US
    g["DEFENSE"] = DEFENSE_US
    g["BENCH"] = BENCH_US
    try:
        yield
    finally:
        mc.POOL, mc.DEFENSE = saved[0], saved[1]
        g["POOL"], g["DEFENSE"], g["BENCH"] = saved[2], saved[3], saved[4]


# ---------------- 数据源:逐个尝试,stooq → yfinance → 本地 CSV ----------------
def _parse_stooq_csv(text, canon):
    import io
    import pandas as pd
    if not text.lstrip().startswith("Date,"):
        return None
    df = pd.read_csv(io.StringIO(text))
    df["Date"] = pd.to_datetime(df["Date"])
    s = df.set_index("Date")["Close"].astype(float).sort_index()
    s.name = canon
    return s


def _stooq_fetch(stq, session):
    """单标的:首取若返回 PoW 挑战页,解 SHA-256 前缀 0×d、POST /__verify 拿 cookie 再取。
    成功返回 Series,失败(被 Access denied/仍挑战)返回 None。"""
    import re
    import hashlib
    url = f"https://stooq.com/q/d/l/?s={stq}&i=d"
    r = session.get(url, timeout=25)
    s = _parse_stooq_csv(r.text, stq.split(".")[0])
    if s is not None:
        return s
    m = re.search(r'const c="([^"]+)"', r.text)
    if not m:
        return None
    d = int(re.search(r',d=(\d+),', r.text).group(1))
    c = m.group(1)
    prefix = "0" * d
    n = 0
    while not hashlib.sha256((c + str(n)).encode()).hexdigest().startswith(prefix):
        n += 1
    session.post("https://stooq.com/__verify", data={"c": c, "n": n},
                 headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=25)
    r2 = session.get(url, timeout=25)
    return _parse_stooq_csv(r2.text, stq.split(".")[0])


def _load_us_stooq():
    """stooq 免费日线 CSV(带 PoW 反爬)。本机出口 IP 可能被 Access denied;正常联网机器通常可通。"""
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": _US_UA, "Referer": "https://stooq.com/"})
    out = {}
    for canon, stq in STOOQ_SYMS.items():
        try:
            ser = _stooq_fetch(stq, s)
            if ser is not None and len(ser):
                ser.name = canon
                out[canon] = ser
        except Exception as ex:
            print(f"  [stooq] {canon} 失败: {ex}")
    return out


def _load_us_yfinance():
    """yfinance(需 pip install yfinance;本机无 pip 装不了,但用户机器可)。auto_adjust=True 取复权收。"""
    try:
        import yfinance as yf
    except ImportError:
        return None
    import datetime as dt
    import pandas as pd
    syms = list(STOOQ_SYMS)
    end = dt.date.today().strftime("%Y-%m-%d")
    df = yf.download(syms, start="2003-01-01", end=end, progress=False, auto_adjust=True)
    close = df["Close"] if "Close" in df.columns.get_level_values(0) else df.xs("Close", axis=1, level=0)
    out = {}
    for c in syms:
        if c in close.columns:
            s = pd.to_numeric(close[c], errors="coerce").dropna()
            s.name = c
            out[c] = s
    return out


def _load_us_csv(usdir):
    """本地 usdata/<SYM>.csv(Date,Close 两列即可)。用户可用任意源导出后放入,绕过联网限制。"""
    import pandas as pd
    out = {}
    for c in STOOQ_SYMS:
        p = os.path.join(usdir, c + ".csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        dcol = next((x for x in df.columns if x.lower() == "date"), None)
        ccol = next((x for x in df.columns if x.lower() in ("close", "adj close", "adjusted")), None)
        if dcol and ccol:
            df["_d"] = pd.to_datetime(df[dcol])
            s = df.set_index("_d")[ccol].astype(float).sort_index()
            s.name = c
            out[c] = s
    return out


def load_us_etfs(usdir=None):
    """按 stooq → yfinance → 本地 CSV 顺序尝试,拼成 {canonical: 收盘 Series}→ DataFrame。
    全部标的都拿到才返回;否则抛错并给出补数据的方法。"""
    import pandas as pd
    usdir = usdir or os.path.join(os.path.dirname(__file__), "usdata")
    print(f"[J1] 取美股 ETF(池 {list(POOL_US)} + 防守 {DEFENSE_US[0]})")
    # yfinance 最可靠(走 cookie+crumb,抗 429);stooq 免依赖但广泛被出口 IP 封;CSV 是离线兜底
    sources = [("yfinance", _load_us_yfinance),
               ("stooq(PoW)", _load_us_stooq),
               (f"本地CSV({usdir})", lambda: _load_us_csv(usdir))]
    series, used = {}, None
    for name, fn in sources:
        try:
            got = fn()
        except Exception as ex:
            print(f"  [{name}] 异常: {ex}")
            got = None
        if got:
            series.update(got)
            used = used or name
            if len(series) >= len(STOOQ_SYMS):
                break
    missing = [c for c in STOOQ_SYMS if c not in series]
    if missing:
        raise RuntimeError(
            f"美股数据不全(缺 {missing});试过 {[n for n, _ in sources]}。\n"
            f"  解决:① 能联网的机器上 `pip install yfinance` 后重跑(本函数自动用);或\n"
            f"        ② 每个标的导出日线 CSV(Date,Close 两列)放到 {usdir}/<SYM>.csv 再跑。")
    print(f"  数据源: {used}  共 {len(series)} 只")
    px = pd.concat(series, axis=1).sort_index().ffill().dropna(how="all")
    # 起点锚定:BENCH+DEFENSE(SPY+TLT)首个共同非NaN日,与 A 股 _anchor_start 同口径(数据驱动、无前视)
    ready = px.index[px[[BENCH_US, DEFENSE_US[0]]].notna().all(axis=1)]
    start = ready[0]
    px = px.loc[start:]
    yrs = (px.index[-1] - start).days / 365.25
    print(f"  [J1 起点] {start.date()}(SPY+TLT 首个共同日)~ {px.index[-1].date()} ({yrs:.1f} 年)")
    return px


# ---------------- 跨市场对照 ----------------
def run_crossmarket(px=None):
    """美股上跑四条对比 + 选股边际 bootstrap:
      等权全池(不风控) / 等权全池+vol+trend(不选股) / 动量轮动top3+vol+trend / 买入持有SPY。
    回答两个问题:① 策略逻辑能否泛化(跑赢 SPY B&H);② 选股边际是否跨市场存在(跑赢等权持有)。"""
    px = load_us_etfs() if px is None else px
    with _us_globals():
        nav_sel, ret_sel, n_sel, _ = backtest(px, hold_all=False, trend_ma=200, trend_code=BENCH_US)
        nav_hold, ret_hold, _, _ = backtest(px, hold_all=True, trend_ma=200, trend_code=BENCH_US)
        nav_eq, ret_eq, _, _ = backtest(px, hold_all=True, trend_ma=None, vol_target=None)   # 等权持有不风控
        bn = bench_nav(px)
        boot = bootstrap_selection(px, extra_cfg=dict(trend_code=BENCH_US))
    p_sel, p_hold, p_eq = perf(nav_sel, ret_sel), perf(nav_hold, ret_hold), perf(nav_eq, ret_eq)
    pb = perf(bn, bn.pct_change().fillna(0))

    print("=" * 72)
    print("J1 跨市场样本外验证(美股 ETF 池,参数零调整:全沿用 A 股部署值)")
    print("=" * 72)
    print(f"  池:{list(POOL_US)}  防守:{DEFENSE_US[0]}  基准/趋势:{BENCH_US}")
    print(f"  区间:{nav_sel.index[0].date()} ~ {nav_sel.index[-1].date()} ({p_sel['年数']:.1f} 年)  调仓 {n_sel} 次\n")
    _print_table([
        ("等权全池(不风控)", p_eq),
        ("等权全池+vol+trend(不选股)", p_hold),
        ("动量轮动top3+vol+trend", p_sel),
        (f"买入持有{BENCH_US}", pb),
    ])
    d_sel_eq = p_sel["夏普"] - p_eq["夏普"]
    d_sel_hold = p_sel["夏普"] - p_hold["夏普"]
    d_vs_bh = p_sel["夏普"] - pb["夏普"]
    print(f"\n  [选股 vs 等权持有(不风控)] 夏普 {p_eq['夏普']:.2f}→{p_sel['夏普']:.2f} ({d_sel_eq:+.2f})")
    print(f"  [选股 vs 等权+风控]        夏普 {p_hold['夏普']:.2f}→{p_sel['夏普']:.2f} ({d_sel_hold:+.2f})")
    print(f"  [策略 vs {BENCH_US} 买入持有]   夏普 {pb['夏普']:.2f}→{p_sel['夏普']:.2f} ({d_vs_bh:+.2f})")
    print("\n  判读:")
    if d_vs_bh <= 0:
        print(f"   → 策略在美股未跑赢 {BENCH_US} 买入持有:逻辑未泛化,A 股优势可能样本/池特定(J1 假设成立)")
    else:
        print(f"   → 策略在美股跑赢 {BENCH_US} 买入持有:动量+风控逻辑跨市场成立,A 股结果非纯池前视")
    if d_sel_hold > 0.05:
        print(f"   → 选股在美股也跑赢'等权+风控':选股 alpha 跨市场存在(需下方 bootstrap 确认)")
    else:
        print(f"   → 选股在美股无可靠优势(≤'等权+风控'):与 A 股 J2 一致——价值在风控+分散,非选股")
    print(f"   （注:d_sel_eq={d_sel_eq:+.2f} 是'选股+风控 vs 等权不风控',含风控贡献,不能当选股证据;")
    print(f"    干净的选股检验是 d_sel_hold={d_sel_hold:+.2f},即同风控下'选 vs 不选',见下方 bootstrap）")
    print("\n" + "=" * 72)
    _boot_report(boot, title=f"J1 美股:动量选股 vs 等权全池(均 vol+trend,trend={BENCH_US})",
                 on="动量选股", off="等权全池")
    print("  注:美股 ETF 无涨跌停(回测不传 cant 掩码);参数全沿用 A 股部署值、未重调(真样本外)。")


def main():
    run_crossmarket()


if __name__ == "__main__":
    main()
