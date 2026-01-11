# Cache-hit-test

用 Codex 本地 `rollout-*.jsonl` 的 `token_count` 统计，计算缓存命中率与“真实倍率（M）”。

## 文件
- `codex倍率计算.py`：读取 `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`，输出 `I/C/U/O`、命中率、倍率与成本拆分。
- `codex-倍率基准与日志计算指南.md`：口径定义与说明。
- `测试文本-ABCD.md`：A/B/C/D 四段可复现测试提示词。

## 快速使用
```bash
python3 codex倍率计算.py \
  --date 2026-01-11 \
  --pin 0.875 --pout 7 \
  --alpha 0.1 \
  --sessions-root ~/.codex/sessions \
  --denom U+O \
  --breakdown --classify
```

> 注意：缓存折扣系数 `alpha` 不在日志里，需要按你的计费规则填写。
