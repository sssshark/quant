export const meta = {
  name: 'quant-fix-verified',
  description: 'Apply fixes for the 25 confirmed audit findings across 3 non-conflicting file-domains, each fix verified by re-reading + re-running',
  phases: [
    { title: 'Fix', detail: '3 parallel fix agents: docs(IMPROVEMENTS), engine(cta+engine+DSR), diag(diag/*.py)' },
    { title: 'Verify', detail: 're-read every changed file + run test_momentum.py + targeted re-run; report per-finding outcome' },
  ],
}

// Each fix agent gets a precise spec (id, file, line, exact fix instructions distilled from notes_for_fix).
// BUCKETED BY FILE-DOMAIN to avoid parallel-edit conflicts on the same file.

const FIX_SCHEMA = {
  type: 'object',
  properties: {
    domain: { type: 'string' },
    applied: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          file: { type: 'string' },
          outcome: { type: 'string', enum: ['fixed', 'skipped', 'no_change_needed', 'deferred'] },
          what_changed: { type: 'string', description: 'exact edit made (old→new) or why skipped' },
          test_status: { type: 'string', description: 'py_compile / test_momentum result for this file' },
        },
        required: ['id', 'file', 'outcome', 'what_changed', 'test_status'],
      },
    },
    remaining_issues: { type: 'string', description: 'any finding you could NOT fully fix + why' },
    test_suite_result: { type: 'string', description: 'final test_momentum.py pass count or error' },
  },
  required: ['domain', 'applied', 'remaining_issues', 'test_suite_result'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    domain: { type: 'string' },
    verified: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          fix_holds: { type: 'boolean' },
          regression_risk: { type: 'string', enum: ['none', 'low', 'medium', 'high'] },
          check_done: { type: 'string', description: 'what you read/ran to confirm (file:line re-read, test name, command)' },
          residual_concern: { type: 'string' },
        },
        required: ['id', 'fix_holds', 'regression_risk', 'check_done', 'residual_concern'],
      },
    },
    new_findings: { type: 'array', items: { type: 'string' } },
    suite_pass: { type: 'string' },
  },
  required: ['domain', 'verified', 'new_findings', 'suite_pass'],
}

const DOMAINS = [
  {
    key: 'docs',
    label: 'fix:docs(IMPROVEMENTS.md)',
    prompt: `你负责修复 /home/poisson/program/quant_program/quant/IMPROVEMENTS.md 中的 7 条"结论过时"文档缺陷。这是纯文档编辑，无代码风险。逐条用 Edit 工具修复（精确匹配 old_string）：

【1. stale-header-count-79-vs-84 (L8, medium, CONFIRMED)】
L8 header 仍写"共 79 条:[高] 16/[中] 37/[低] 22"，但 L25 进度块写"总 84 ┃ 已完成 61 ┃ 待办 23"。正确总数是 84（A9+B6+C10+D7+E7+F8+G6+H11+I4+J7+M9=84）。
修复：把 L8 的"共 79 条"改成与 L25 一致的总数 84，并使优先级拆分自洽。最简方案：L8 写"共 84 条（A–J 节 75 条 + M 节 9 条）"，并把 16/37/22 这组陈旧拆分标注为"（历史优先级拆分，当前完成状态见下方进度块 L25：[高]0/[中]0/[低]23）"。务必保证总数=84。同时 grep 核验 \`grep -cE '^- \\\\[x\\\\]|^- \\\\[' IMPROVEMENTS.md\` 与声称总数吻合。

【2. stale-changelog-c1-direction-reversal (L327, medium, CONFIRMED)】
L327 changelog C1 行仍写 inv_vol"0/3 优于 equal（全输）"（污染期），但正文 L93 与 L338 显示最终版是"3/3 网格全胜"。方向矛盾未调和。
修复（按 notes 选项 a，保留审计轨迹）：在 L327 该单元格末尾追加标注"【已过时:本行 0/3 全输系污染期数据;E3/E4 后 3/3 全胜(L338),pre_close 最终版 3/3 全胜 Δ+0.039/P=0.166(L93)】"。不要改写历史数字。

【3. stale-changelog-a2-sign-reversal (L335, low, PLAUSIBLE)】
L335 A2 行 +0.13/P=0.39（污染期），最终版 L70 翻成 -0.030/P=0.675（符号反转）。notes 说"可选次要措辞清理"。
修复（最小，保留 L335 历史值不替换）：把 L335 现有免责"注:此结论基于去污前数据,绝对值见重定基线说明"改成"注:此结论基于去污前数据,符号与绝对值均见重定基线说明(L340, 翻负 -0.030)"。明确符号也反转。

【4. stale-changelog-f3-polluted-numbers (L331, medium, CONFIRMED)】
L331 F3 changelog 仍写污染期数字（熊市+1.3%/基准-6.6%/回撤-13.1%），正文 L129 最终版是熊市+4.4%/基准-4.5%/回撤-14.9%。magnitude 偏 3-4x，且 Sortino 从-0.91(震荡弱项)翻到+0.22(震荡微正)。
修复（按 notes 选项1，替换为最终值，最干净）：把 L331 结果单元格的数字换成 L129 的 pre_close 最终值："策略熊市年化 +4.4%（基准 -4.5%）、回撤 -14.9%（基准 -52.2%，仅约 1/3.5）；牛市年化 12.1% 略输基准 13.6%（夏普 2.26>2.03）；震荡微正（Sortino 0.22）；反向波动熊市年化 +4.7% 优于等权 +4.4%"。

【5. stale-changelog-b1b2-f1-polluted-absolute (L332/L333/L328, medium, CONFIRMED)】
L332(B1)/L333(B2)/L328(F1) 三行 changelog 仍是污染期绝对值，与正文矛盾。
修复（按 notes，逐行替换为 pre_close 最终值）：
- L332: "年化 17.9→11.2、夏普 0.89→0.59、回撤 -24.8→-41.8" → "年化 15.1%→11.8%、夏普 1.20→0.94、回撤 -20.2%→-20.3%"（回撤修正最重要）
- L333: "原 7 只年化 17.9→16.6(-1.3pp)" → "年化 14.0% vs 15.1%(-1.2pp)"
- L328: C1 夏普"0.81<0.89"→ 改为 pre_close 口径"~1.234 vs ~1.198(P=0.166 不显著)"；至少加 staleness tag。
notes 提醒：别盲改"1.20→0.94"(B1 扩池对比)这类"相对基线的 Δ"——它们内部自洽、不是"最终 CTA 夏普=1.20"的声称，只改那些把"最终 CTA 夏普=1.20"当点估计的单元格。L328 的 0.81<0.89 是污染期 C1 单元格绝对值，需改。

【6. stale-sharpe-1.20-vs-1.19-datasource-drift (L371 + 正文多处, medium, CONFIRMED)】
Q1 changelog(L371)记数据源切换使 CTA 夏普 1.20→1.19 末位漂移、README 已更新到 1.19/15.0%，但 IMPROVEMENTS 正文仍普遍用 1.20/1.198 当最终值（L70/82/84/93/96/99/129/131/179/184/188）。
修复（按 notes 推荐的"单条说明"法，避免 sed 误伤相对 Δ）：在顶部"重定基线"块（约 L30-48 区域）末尾加一条说明："⚠ **数据源漂移(2026-07-11)**:数据源切换使 CTA 夏普 1.20→1.19 末位漂移、年化 15.1%→15.0%；正文仍标 1.20/1.198 系 pre_close 版口径，实盘当前值见 README 1.19/15.0%。相对比较(如 1.20→0.94 扩池)内部自洽不受影响。" 不要全局 sed 替换 1.20→1.19（会破坏相对 Δ 数字）。

【7. stale-q2-claim-no-flip-contradicted (L372, low, CONFIRMED)】
L372 Q2 changelog 声称"deep-concl 全量核验:所有引用数字重跑,无结论翻转"——但本审计发现 L327(C1 0/3→3/3)、L335(A2 +0.13→-0.030 符号翻转)、L331(F3 污染期数字)的历史行仍矛盾，Q2 的"无翻转"只覆盖了正文 keep/reject 决策、没覆盖 changelog 表绝对值。
修复（改 Q2 单元格措辞，scope 精确化）：把 L372 该声称改成"核验范围限于各条目正文默认判断(keep/reject)未翻转;解决记录表的历史行(L327 C1 0/3、L331 F3 +1.3%、L335 A2 +0.13 等)仍保留历次基线的绝对值、未回填最终版(L93=3/3、L129=+4.4%、L70=-0.030)"。

【通用约束】
- 只改 IMPROVEMENTS.md，不动任何 .py。
- 用 Read 先精确读出每行原文再 Edit（old_string 必须逐字匹配，含中文标点/全角）。
- 改完对每条报 outcome=fixed + what_changed(精确到 old→new)。
- 全部 7 条都做（不要 defer）。
- 不需要跑 test_momentum（文档无测试），但可 grep 自检总数=84。`,
  },
  {
    key: 'engine',
    label: 'fix:engine(cta.py+engine.py+DSR)',
    prompt: `你负责修复 /home/poisson/program/quant_program/quant 的核心代码缺陷（cta.py + engine.py），含代码正确性 + 数据 + 过拟合(DSR)三类。逐条用 Read 精确读再 Edit。注意 engine.py 是 1702 行大文件，编辑后必须 py_compile + 跑 test_momentum。

【代码正确性 3 条】

【1. defense-cash-treated-as-equity-by-risk-filters (cta.py:332, medium, CONFIRMED) —— 最重要】
问题：当 defense_cash != DEFENSE[0]（如 511880 货基）时，step2b(绝对动量切仓)/step3 把权重累加到 dcode_cash(511880)，但 step3-6 的 eq_codes 过滤用 \`c != dcode\`(511010)——漏排 511880——导致货基被 vol/trend/crash/drawdown 过滤器缩放、且 step4-6 的 proceeds 路由到 dcode(国债) 而非 dcode_cash，破坏 B2 分档设计。
修复（按 notes 最小且与 B2 一致）：
1. 在 decide_targets 里 dcode/dcode_cash 定义后，加 \`defense_codes = {dcode, dcode_cash}\`（集合自动去重，两者相等时即 {dcode}）。
2. 所有 eq_codes 过滤统一改为 \`eq_codes = [c for c in target if c not in defense_codes]\`，覆盖：max_weight 集中度(L326附近)、step3 vol(L332附近)、step4 trend(L346附近)、step5 crash(L361附近)、step6 drawdown(L375附近)。逐个 grep \`eq_codes =\` 找全。
3. recipient 路由不改（step3 用 dcode_cash、step4-6 用 dcode 是 B2 有意分档），只确保 eq_codes 已排除两防御码。
4. 读 cta.py 实际行号（332 等可能已漂移），以代码现状为准。
回归：构造 defense_cash='511880' + trend 触发场景，断言 dcode_cash 最终权重不受 step4-6 缩放。

【2. limit-price-bankers-rounding-vs-ashare-half-up (engine.py:195-196, low, CONFIRMED)】
问题：limit_masks 用 \`(pc*(1+lim)).round(2)\`（numpy 银行家舍入 half-to-even），A 股官方是四舍五入到分。pc=0.95×1.1=1.045，A 股涨停=1.05，numpy round 给 1.04 → close=1.04(未触板) 被误判 cant_buy=True。已知分歧 pre_close: 0.15,0.25,0.35,0.75,0.95,1.15,1.95 等 .x5 边界。
修复（按 notes，L195+L196 都改）：用 decimal ROUND_HALF_UP。在 limit_masks 内加 helper：
\`\`\`
from decimal import Decimal, ROUND_HALF_UP
def _rup(x):
    return float(Decimal(str(float(x))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
\`\`\`
然后 cant_buy[code] = cl >= pc.multiply(_rup*(1+lim)...)——注意要向量作用在 Series 上：用 \`(pc*(1+lim)).map(_rup)\` 或 \`(pc*(1+lim)).apply(_rup)\`。cant_sell 同理用 \`pc*(1-lim)\`。
注意 import 语句放文件顶部或函数内（看现有风格）。

【3. realized-vol-uses-equalweight-even-for-inv-vol-holdings (cta.py:197, low, CONFIRMED)】
问题：_realized_vol 总用等权组合方差(sum(day)/len(day))，即使持仓是 inv_vol 加权——vol_target 标定在等权 vol 却作用在 inv_vol 权重上。默认 WEIGHTING='equal' 无影响，是 inv_vol(A/B 工具默认关)下的近似不精确。
修复（按 notes 选项2，最低风险，因 inv_vol 默认关）：在 _realized_vol 调用处(L334附近)加一行注释文档化这是等权 vol 近似；或给 _realized_vol 加可选 weights=None 参数(默认 None=等权向后兼容)、传实际 inv_vol 权重算加权组合收益。notes 说选 2 更稳。本条可只加注释（标注"等权 vol 近似,inv_vol 下为粗略风控标定"）+ 不改默认行为。

【数据 2 条】

【4. data-ffill-fabricates-suspension-prices (engine.py:132, low, PLAUSIBLE) —— IMPROVEMENTS 已记为 E7】
问题：\`.ffill()\` 无条件把停牌/未上市 gap 填成平稳价，0% 收益喂给 _realized_vol(压低 vol→vol_target 不缩仓) 和 blended_momentum(稀释趋势)，_data_quality_guard 抓不到(0% 低于阈值)。
修复（按 notes 最低成本）：把 L132 的无条件 ffill 改成"有界 ffill"——用 \`px.ffill(limit=3)\`（只向前填 ≤3 交易日，更长 gap 留 NaN）；或更彻底：价格面板留 NaN、只在算收益处(engine.py:290 rets=px.pct_change().fillna(0.0))显式填 0，使停牌日贡献显式 0 收益而非平稳价。**推荐 limit=3 有界 ffill**（最小改动、保 warm-up/blend_momentum 的 NaN-skip 行为——cta.py L169-180/L129-131 已 skip NaN）。eastmoney 路径 L171 同改。
注意：改完跑 test_momentum + \`python engine.py real\` 抽检净值起点不变(2013-03-25)。

【5. data-tushare-single-fetch-silent-nan (engine.py:119, low, PLAUSIBLE)】
问题：_load_via_tushare 的 per-symbol _ts_post 无 try/except 无 retry，单次 transient 失败(RuntimeError)杀掉整次加载；且 load_real 的 eastmoney fallback 只在 missing-token 时触发，transient 失败会直接抛、不降级。
修复（按 notes）：在 L118-131 的 per-symbol 循环里，给 _ts_post 调用包 try/except + 有界 retry+backoff（镜像 eastmoney L155-166 的 4 次重试+退避）。最终失败则 loud warning 跳过该 symbol（或 per-symbol 回退 akshare）。**不要**改成"静默降级到 eastmoney"（会换口径）——要 loud warning。

【过拟合 DSR 1 条】

【6. dsr-n-trials-capped-100-undercounts-search (engine.py:1134, medium, PLAUSIBLE)】
问题：run_dsr 的 \`for N in (10,50,100)\` 只扫到 100，但 backlog(49 行 A/B + 27 族 + robust 7 轴×5 + WF_GRID 8 + sweep)等效试验数远超 100，N=100 是把 DSR 顶上 0.95+ 的乐观假设。按项目自报参数(g3=-0.98/g4=15.6/n=3225/sr=1.198)重算：N=200→0.908、N=500→0.852，跌破 0.95。
修复（按 notes，日频+月频两口径都扩）：
1. engine.py:1134 \`for N in (10, 50, 100):\` → \`for N in (10, 50, 100, 200, 500, 1000):\`（日频/月频两口径同步扩——看 run_dsr 是否两处都有循环，两处都改）。
2. 修订 deflated_sharpe docstring(L1095-96)与 run_dsr 解释文字(L1146附近)：当前"N 偏大→DSR 偏保守"与 L189"backlog 就是 A/B 选参记录"自相矛盾——backlog 证明等效试验数远超 100。改成："N 是等效独立试验数的敏感性扫描,[10,50,100]仅覆盖最乐观;按 backlog 量级真实 N 约 200-500,届时日频 DSR 落 0.85-0.91、月频 0.90-0.94,'经得起多重比较'不再在两口径同时成立。"
3. 不改 IMPROVEMENTS（那是 docs agent 的域）——只改 engine.py 代码+docstring+打印。

【通用约束】
- 改完每个文件 \`~/miniconda3/bin/python -m py_compile <file>\` 确认无语法错。
- 全部改完 \`~/miniconda3/bin/python test_momentum.py\` 确认仍全过(基线 24/24)。
- defense-cash 修复后，建议加/跑一个 defense_cash != DEFENSE[0] 的回归断言（可放 test_momentum.py，但若时间紧可只 print 验证）。
- 每条报 outcome + what_changed + test_status。remaining_issues 报任何无法完成的 + 原因。
- 行号可能漂移，以 Read 实际内容为准。`,
  },
  {
    key: 'diag',
    label: 'fix:diag(diag/*.py)',
    prompt: `你负责修复 /home/poisson/program/quant_program/quant/diag/ 下的诊断脚本缺陷。这些脚本是"负结果"诊断——负结果可能是 bug 造成的伪证伪，必须修对才能信。逐条 Read 精确读再 Edit。

【1. epo-constant-corr-target-degenerates-at-N2 (diag_portfolio_opt.py:65, high, CONFIRMED) —— 最关键】
问题：N=2 时常数相关收缩目标 F 数学上恒等于样本协方差 S（只有一个 off-diag corr，F[0,1]=rbar*sd0*sd1=corr[0,1]*sd0*sd1=S[0,1]；对角 d_i=S_ii），故 gamma=||F-S||²=0，delta=0（3041/3041 rebalance 全 delta=0）。即"EPO(LW-shrunk min-var)"实为**未收缩的样本协方差 min-var**，负结论部分是"用了无法收缩的目标"的产物。sklearn identity-target LedoitWolf 在同数据给 delta~0.02，证明正确目标能收缩。
修复（按 notes 选项1，最彻底）：把 ledoit_wolf_cov 的收缩目标从常数相关 F 改成 Ledoit-Wolf 2004 缩放单位阵 F=mu*I（mu=trace(S)/N，即 sklearn/OAS 默认的"well-conditioned estimator"）。这样 N=2 时 F≠S、gamma>0、delta>0，"LW-shrunk min-var"标签才名副其实。具体：
- 删掉常数相关目标构造（rbar/denom/np.fill_diagonal(F,d) 那段）。
- 改成：\`mu = np.trace(S) / N; F = mu * np.eye(N)\`。
- gamma=\`np.sum((F-S)**2)\` 现在非零。
- 注意：改成 identity target 后，rho 的对角近似(trace(pi_mat)=sum_i Var(s_ii))**正好变正确**（见 finding 3 notes 选项2：identity target 下 f_ij 是常数、与 s_ij 不相关，rho=trace(pi_mat) 精确）——所以改 target 同时修好 finding 3。
- 改完重跑 Layer1(static L145)/Layer2(rolling L177)/Layer3(monthly L243) 看 delta~0.02-0.05 后 EPO-vs-1/sigma 的 Δ 是否变。

【2. epo-minvar-negative-weight-equal-fallback (diag_portfolio_opt.py:84, medium, CONFIRMED)】
问题：min_var_weights 的 \`if np.any(w<=0): return np.ones(N)/N\` 在 15.7% rolling rebalance 触发（478/3041），把 EPO 退化成等权、人为压低它与 1/σ 的差异。但 notes 指出：478 里只有 16%(77个)是负权重（corr>0.19），84%(401个)是 LinAlgError——后者是 BondMomentumStrategy 空仓时 rolling bond 全 0 → Sigma 奇异(det=0)。
修复（按 notes，两个触发分别治）：
1. 负权重分支：换成 long-only 约束 min-var——clip 负权重到 0 再 renormalize（N=2 退化干净），或解 QP \`min w'Sw s.t. sum(w)=1, w>=0\`。notes 说这会把 Layer3 Δ 从 +0.015 翻到 -0.060——"verify the Delta change is intended"（是预期的，因为修复后 EPO 不再被退化成等权）。
2. LinAlgError 分支：在 np.linalg.solve 前给 Sigma 加对角抖动 \`Sigma + 1e-8*I\`，或检测零方差资产用其 1/σ fallback，而非 50/50。
改完重跑 Layer3 bootstrap 看 Δ。**deploy 结论(EPO 月频不显著,保持 RiskParity/等权)在 bug 版和修复版都成立**——所以只改诊断报告的 Δ，不动部署。

【3. epo-rho-definition-wrong-but-dormant (diag_portfolio_opt.py:73, low, CONFIRMED)】
问题：rho=trace(pi_mat)=sum_i Var(s_ii) 是错的量（应为 sum_ij Cov(s_ij,f_ij)），但 N=2 时 gamma=0 使其 moot。
修复：若你已按 finding 1 改成 identity target，**本条自动修好**（identity target 下 rho=trace(pi_mat) 精确）。若保留常数相关 target，则需按 LW2004 实现 cross-moment rho——但既然 finding 1 建议改 identity，本条随 finding 1 一并解决。在注释里注明"identity target 下 rho=trace(pi_mat) 精确"。
可选加固：加 \`assert N==2\` guard 让限制结构化（如果脚本永远只用 N=2）。

【4. bonds2-token-path-filenotfound / data-diag-token-path-mismatch (diag_bonds2.py:32, medium, CONFIRMED, 两 finding 同一 bug)】
问题：diag_bonds2.py:32 \`open(os.path.join(os.path.dirname(__file__), '.tushare_token'))\`——单 dirname=diag/，token 在 repo 根 → FileNotFoundError，脚本根本跑不起来。兄弟脚本 diag_sector.py:54 用双 dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 正确。
修复：改 L32 成 \`here = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); token = open(os.path.join(here, ".tushare_token"), encoding="utf-8").read().strip()\`，镜像 diag_sector.py:54-55。可选加 TUSHARE_TOKEN env 检查(如 engine.py)。
修复后跑 \`~/miniconda3/bin/python diag/diag_bonds2.py\` 确认能 fetch 511260/511220、多债对比表打印（这是全新结果，原结果因脚本跑不起来不可信）。

【5. rsrs-bonds-mr-T+1-vs-engine-T+2-caliber-mismatch (diag_rsrs.py:132 + diag_bonds.py:63 + diag_mr.py:74, low, CONFIRMED)】
问题：timing_backtest(及 diag_bonds.py:63 bond_momentum、diag_mr.py:74 mr_backtest)只用单 shift(1) → 收益从 T+1 起；engine.backtest 是两 lag(signal T→execute T+1→w_lag=shift(1)→收益 T+2)。诊断比 engine CTA 基准多 1 天收益，绝对夏普略高估（不影响 RSRS-vs-SMA 相对排序、不影响 corr 互补性结论，但绝对值不可与 engine 1.19 直接比）。
修复（按 notes，加第二 shift 对齐 engine T+2 口径，diag_sector.py:100-101 已是此标准）：
- diag_rsrs.py timing_backtest(L132-135)：现 \`pos=signal.reindex().where().ffill().fillna(0)\` 后 \`pos=pos.shift(1)\` → 改成 \`pos_exec=pos.shift(1).fillna(0)\`（T+1 成交）+ \`w_lag=pos_exec.shift(1).fillna(0)\`（T+2 收益），\`gross=w_lag*ret\`。turnover 用 pos_exec。
- diag_bonds.py bond_momentum(L63-66)：\`pos=sig.where().ffill()\` → \`weights=pos.shift(1)\` → \`w_lag=weights.shift(1)\` → \`gross=w_lag*ret\`。
- diag_mr.py mr_backtest(L74)：\`w_lag=weights.shift(1).fillna(0)\` → \`w_lag=weights.shift(1).shift(1).fillna(0)\`（两 shift）或两步。
- 更新注释"T+1 成交防前视"→"T+2 吃收益口径对齐 backtest"。
改完跑 \`~/miniconda3/bin/python diag/diag_rsrs.py\`（若无 token/sandbox 限制则跑；否则 py_compile）确认。

【通用约束】
- 改完每个文件 \`~/miniconda3/bin/python -m py_compile <file>\`。
- diag_portfolio_opt 改完尽量实跑（EPO 改 target 是数学变更，必须重跑看 delta）——用 ~/miniconda3/bin/python，若需 token 则 .tushare_token 在 repo 根（diag 脚本改完路径后能读到）。
- 每条报 outcome + what_changed + test_status。remaining_issues 报无法跑实数据验证的（如 sandbox 限网）+ 原因。
- 行号漂移以 Read 为准。`,
  },
]

phase('Fix')
phase('Verify')

// Stage 1: 3 fix agents in parallel (non-overlapping file domains)
const fixes = await parallel(DOMAINS.map(d => () =>
  agent(d.prompt, { label: d.label, phase: 'Fix', schema: FIX_SCHEMA, effort: 'high', agentType: 'general-purpose' })
    .then(r => ({ domain: d.key, fixReport: r }))
))

// Stage 2: verify each domain's fixes by re-reading + re-running
phase('Verify')
const verifications = await parallel(DOMAINS.map((d, i) => () =>
  agent(
    `你是验证员。验证 ${d.key} 域的修复是否真正落地、有无回归。打开 /home/poisson/program/quant_program/quant 实读每个被改文件的相关行，并跑验证命令。

【${d.key} 域修复报告（来自 fix agent）】
${JSON.stringify(fixes[i].fixReport, null, 2)}

【你的任务】
1. 对该域每条 finding，实读改后代码/文档，确认修复落地（old→new 确实改了、改对了）。
2. 跑验证：
   - docs 域：grep 核验 IMPROVEMENTS 总数=84、L327/L331/L335/L372 措辞已改、无残留 79。
   - engine 域：\`~/miniconda3/bin/python -m py_compile cta.py engine.py\` + \`~/miniconda3/bin/python test_momentum.py\`（必须仍 ≥24/24， ideally 24 或 24+新增）；defense-cash 改后构造 defense_cash!=DEFENSE[0] 场景实跑 decide_targets 断言货基不被 step4-6 缩放；limit-rounding 改后可加/跑 round-half-up 断言（pc=0.95 → 涨停 1.05 非 1.04）。
   - diag 域：\`~/miniconda3/bin/python -m py_compile diag/diag_portfolio_opt.py diag/diag_bonds2.py diag/diag_rsrs.py diag/diag_bonds.py diag/diag_mr.py\`；diag_bonds2 改路径后实跑 \`~/miniconda3/bin/python diag/diag_bonds2.py\` 确认能 fetch 多债（.tushare_token 在 repo 根）；diag_portfolio_opt 改 identity target 后实跑看 delta 是否非零；diag_rsrs T+2 改后 py_compile 或实跑。
3. 对每条给 fix_holds(bool) + regression_risk(none/low/medium/high) + check_done(你实读的行号/跑的命令) + residual_concern。
4. 报 new_findings（修复引入的新问题，若无空数组）+ suite_pass（test_momentum 最终通过数）。

**严格**：修复未真正落地或引入回归，fix_holds=false。`,
    { label: `verify:${d.key}`, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'xhigh' }
  )
))

// collect
const allApplied = fixes.filter(Boolean).flatMap(f => (f.fixReport?.applied || []).map(a => ({ ...a, domain: f.domain })))
const allVerified = verifications.filter(Boolean).flatMap(v => (v.verified || []).map(x => ({ ...x, domain: v.domain })))
const newFindings = verifications.filter(Boolean).flatMap(v => v.new_findings || []).filter(Boolean)
const suiteResults = verifications.filter(Boolean).map(v => ({ domain: v.domain, suite_pass: v.suite_pass }))

const fixedCount = allApplied.filter(a => a.outcome === 'fixed').length
const verifiedCount = allVerified.filter(v => v.fix_holds).length

log(`Fix phase: ${fixedCount}/${allApplied.length} applied as fixed`)
log(`Verify phase: ${verifiedCount}/${allVerified.length} verified holding; ${newFindings.length} new findings introduced`)

return {
  round: 'fix-1',
  applied: allApplied,
  verified: allVerified,
  new_findings: newFindings,
  suite_results: suiteResults,
  summary: `${fixedCount} fixed, ${verifiedCount} verified-holding, ${newFindings.length} new findings`,
}
