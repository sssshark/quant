# A 股 ETF 动量轮动 + 多策略组合（量化研究 / 实盘骨架）

一套**单一决策源、回测与实盘共用同一份逻辑**的 ETF 动量策略框架：核心是一个动量轮动 CTA
策略，外加一个组合层，把 CTA 与国债时序动量等独立策略叠加成多策略组合。配套完整的过拟合
检验（walk-forward / bootstrap / PBO / Deflated Sharpe）与跨市场样本外验证。

> 回测区间约 13 年（2013 起至 2026），月末调仓、T+1 成交、含手续费/滑点与涨跌停过滤。
> 本仓库是个人量化研究项目，**不含投资建议**；实盘请自行核实并自负盈亏。

---

## 策略

**CTA（主线）** —— `cta.decide_targets`，单一信号 + 多层风控：

- **信号**：多周期混合动量（21/63/126 日涨幅均值），在 7 只 ETF 池里选动量最高的前 3 只等权持有。
- **风控（只减不加、可叠加）**：年化波动目标 → 大盘趋势过滤（沪深300 跌破 MA200 砍仓挪国债）→
  动量崩溃保护 → 回撤控制。绝对动量 ≤0 的标的切防守资产（国债 ETF）。
- **标的池（先验驱动，无前视）**：沪深300 / 中证500 / 创业板 / 中证1000 / 红利 / 黄金 / 纳指；
  防守资产 国债ETF。成员仅按"资产类别 + 上市年限"挑选，**绝不**用回测收益挑（前视教训）。

**国债时序动量（第二策略，已验证）** —— `BondMomentumStrategy`：国债混合动量 >0 且价 >MA200 则满仓，否则空仓。
与 CTA 相关性 −0.115（股债跷跷板，熊市更负约 −0.10~−0.15），是唯一通过"三道关"的独立 alpha 来源。

**多策略组合（B 型）** —— `multi_strategy.py`：多个独立 `Strategy` 在组合层叠加，**不在单策略内部做多因子**。
提供等权 `MultiStrategy` 与滚动 `RiskParityMulti`（1/σ 反波动分配）两种组合器。

## 结果一览

| 配置 | 夏普 | 年化 | 回撤 | 说明 |
|---|---|---|---|---|
| **CTA 等权全池+风控+C2（`real` 默认口径）** | **1.08** | **11.5%** | **−12.4%** | **默认回测+实盘口径**（hold_all=true、C2 开）：选股无可靠 alpha（PBO 0.67）等权更稳；C2 把回撤从 −16.9% 压到 −12.4%（P=0.122 未达 5% 但本职强） |
| CTA 动量选股+风控（`hold_all=false`） | 1.19 | 15.0% | −20.2% | 选股版，诊断/对照（`nomomentum` 子命令），非默认 |
| CTA 等权全池+风控（C2 关） | 1.01 | 11.4% | −16.9% | hold_all 但不开 C2；对照用 |
| 国债时序动量（独立） | 1.51 | 3.1% | −4.4% | 第二策略（与 CTA 收益相关性 −0.115） |
| CTA + 国债 50/50（等权组合） | 1.29 | 7.4% | −7.5% | > 单 CTA（1.01），分散有效；部署默认 |
| CTA + 国债（risk-parity，月频可部署） | 1.45 | 5.1% | −5.3% | 对等权 1.29 **不显著**（P=0.182）；日频诊断上界 1.91（P=0.001）依赖全样本σ前视。默认仍等权 |

> 注：2026-07-12 起 CTA 默认开 C2（见表第 1 行）。国债/组合行（第 4–6 行）CTA 部分随默认开 C2 略变，其数字为变更前口径；分散有效结论不变，精确值重跑 `python engine.py multi`。

被检验并**证伪/否决**的候选（负结果，见 `diag/` 与 `IMPROVEMENTS`）：A 股均值回归、配对交易、
行业板块轮动/择时、商品期限结构（供需信号）——均过不了"三道关"，保留脚本作为方法论记录。

## 快速开始

```bash
# ⚠️ 必须用 miniconda 的 python（3.13，依赖齐全）；不要装 akshare
~/miniconda3/bin/python engine.py            # 默认 real：等权全池+风控（vol+trend+C2） vs 基准，出图
~/miniconda3/bin/python engine.py multi      # 多策略组合（CTA + 国债）回测
~/miniconda3/bin/python engine.py nomomentum # hold_all=True 等权全池 vs 动量选股对照
~/miniconda3/bin/python test_momentum.py     # 全部回归测试
~/miniconda3/bin/python live.py              # 实盘骨架（默认 paper/DRY_RUN，只打印计划）
```

`engine.py` 其它模式：`sweep`（参数扫描）/ `robust`（稳健性）/ `wf`（walk-forward 样本外）/
`pbo`（过拟合概率）/ `boot` `bootmom`（bootstrap 显著性）/ `dsr`（Deflated Sharpe）/
`universe`（池消融）/ `regime`（牛熊震荡拆解）/ `attrib` `mfat`（CAPM / 多因子归因）等。
完整列表见 `engine.py` 的 `main()`。

首次运行需在仓库根放 `.tushare_token`（一行你的 tushare token；已 gitignore）。

## 目录结构

| 文件 | 作用 |
|---|---|
| `cta.py` | **决策大脑（唯一来源）**：标的池、参数、`decide_targets`、涨跌停表。回测/实盘都调它 |
| `engine.py` | 向量化回测 + 全套研究/稳健性检验（1700+ 行，主力） |
| `multi_strategy.py` | 多策略组合层：`Strategy` 协议 / `CTAStrategy` / `BondMomentumStrategy` / `MultiStrategy` / `RiskParityMulti` |
| `bt.py` | Backtrader 事件驱动回测（撮合更贴实盘；需 backtrader+akshare，本机未装） |
| `live.py` | miniQMT 实盘骨架（`BROKER=paper` 走本地模拟，`qmt` 连真实账户） |
| `paper_broker.py` | 本地模拟券商（miniQMT 的 drop-in 替身，状态持久化 JSON） |
| `crossmarket.py` | 跨市场样本外验证：同一套策略+参数搬到美股 ETF 池 |
| `test_momentum.py` | 回归测试（pytest 兼容） |
| `diag/` | 独立诊断脚本（每个候选策略必须过"三道关"才考虑落地） |
| `量化笔记/` | 零基础量化术语讲解（按资产类别拆分） |
| `IMPROVEMENTS.md` | 改进清单与结案记录（A/C/D/E/F/G/H/J/M 编号体系） |

## 方法论与可信度

- **单一决策源**：策略逻辑只存在于 `cta.decide_targets`；回测怎么选，实盘就怎么选，避免两处漂移。
- **三道关**：任何新策略进组合前必须同时满足 ① 与基本盘相关性 |corr|<0.3 ② 自身夏普>0.5 ③ 50/50 组合夏普 > 各自最大。
- **无前视**：池子成员、调仓信号、成交口径（T 日信号、T+1 成交、T+2 起吃收益）均严格无未来函数。
- **防过拟合**：参数扫描、单参扰动、walk-forward、purged CV、PBO、配对 block bootstrap、Deflated Sharpe 全套检验。
- **K=1 守恒**：多策略重构的安全网——单策略经组合路径的净值必须逐点等于 `decide_targets` 直连路径（atol 1e-9），由 `test_multi_k1_equivariance` 保护。

## 数据与依赖

- 行情走 **tushare HTTP**（默认中转 `http://47.116.63.181:8000/dataapi`，path 式路由、http 明文——token 与查询参数经此服务器；可 `TUSHARE_API` 环境变量覆盖，如切回旧 https 代理需同时改请求格式）。
- ETF **前复权**：用 `fund_daily` 的 `close + pre_close` 经 `engine._qfq_from_pre_close` 对齐（tushare `pro_bar` 的 `adj` 对 ETF 不生效）。
- 涨跌停：A 股 ETF 按板块分档（创业板/科创板 ±20%，主板/跨境/商品 ±10%），集中在 `cta._limit`，回测/实盘共用。
- **gitignore 敏感文件**（勿提交）：`.tushare_token`、`live_config.json`、`paper_account.json`、`last_rebalance.txt`、`usdata/`。实盘凭证从环境变量 > `live_config.json` 读取（样例见 `live_config.example.json`）。

## 文档

- [`IMPROVEMENTS.md`](IMPROVEMENTS.md) —— 改进清单（84 条，已完成 61；高/中优先级全清），是变更的权威日志。
- [`量化笔记/`](量化笔记/) —— 零基础量化术语系统讲解。
- [`CLAUDE.md`](CLAUDE.md) —— 给 AI 编程助手的仓库操作指引（环境/命令/架构/不变量）。

## 风险声明

本仓库仅供量化方法学习与研究。回测结果不代表实盘表现；程序化交易需按交易所要求报备；
真金白银交易请本人确认后执行，建议先在模拟账户跑通。
