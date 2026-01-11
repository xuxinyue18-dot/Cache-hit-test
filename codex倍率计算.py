#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import difflib
import glob
import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenTotals:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "TokenTotals") -> None:
        self.input_tokens += other.input_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_output_tokens += other.reasoning_output_tokens
        self.total_tokens += other.total_tokens


@dataclass
class RolloutSummary:
    path: str
    session_id: Optional[str]
    session_timestamp: Optional[str]
    user_text: str
    user_task_text: str
    tool_calls: int
    tool_names: dict
    totals: TokenTotals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从 ~/.codex/sessions 的 rollout JSONL 里读取 token_count，结合价格与缓存折扣，计算真实倍率。"
        )
    )
    parser.add_argument("--date", required=True, help="日期：YYYY-MM-DD（用于定位 sessions 目录）")
    parser.add_argument("--pin", type=float, required=True, help="输入单价（$/1M tokens）")
    parser.add_argument("--pout", type=float, required=True, help="输出单价（$/1M tokens）")
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="缓存折扣系数 α（缓存输入单价/普通输入单价），例如 1 或 0.1",
    )
    parser.add_argument(
        "--sessions-root",
        default=os.path.expanduser("~/.codex/sessions"),
        help="sessions 根目录（默认：~/.codex/sessions）",
    )
    parser.add_argument(
        "--denom",
        choices=["U+O", "U"],
        default="U+O",
        help="倍率分母（有效工作量）口径：U+O（默认）或 U",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="输出每个 rollout（会话）的分解结果与“模拟用户行为”分类（不打印原始提示词全文）。",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="对 rollout 做行为分类（冷启动/重复提问/小改动/工具密集）。需与 --breakdown 配合更有用。",
    )
    return parser.parse_args()


def load_last_token_totals(rollout_path: str) -> Optional[TokenTotals]:
    last: Optional[TokenTotals] = None
    with open(rollout_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") != "event_msg":
                continue
            payload = obj.get("payload") or {}
            if payload.get("type") != "token_count":
                continue
            info = payload.get("info") or {}
            total = info.get("total_token_usage") or {}
            last = TokenTotals(
                input_tokens=int(total.get("input_tokens") or 0),
                cached_input_tokens=int(total.get("cached_input_tokens") or 0),
                output_tokens=int(total.get("output_tokens") or 0),
                reasoning_output_tokens=int(total.get("reasoning_output_tokens") or 0),
                total_tokens=int(total.get("total_tokens") or 0),
            )
    return last


def load_rollout_summary(rollout_path: str) -> Optional[RolloutSummary]:
    session_id: Optional[str] = None
    session_timestamp: Optional[str] = None
    user_text_parts: list[str] = []
    user_task_parts: list[str] = []
    tool_calls = 0
    tool_names: dict[str, int] = {}
    totals = load_last_token_totals(rollout_path)
    if totals is None:
        return None

    def is_boilerplate_user_message(text: str) -> bool:
        t = text.strip()
        if not t:
            return True
        # Common harness/system scaffolding for this environment
        if "AGENTS.md instructions" in t:
            return True
        if "<environment_context>" in t:
            return True
        if "<INSTRUCTIONS>" in t and "Skills" in t:
            return True
        return False

    with open(rollout_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            typ = obj.get("type")
            payload = obj.get("payload") or {}

            if typ == "session_meta":
                session_id = payload.get("id")
                session_timestamp = payload.get("timestamp")
                continue

            if typ == "response_item":
                ptype = payload.get("type")
                if ptype == "message" and payload.get("role") == "user":
                    for item in payload.get("content") or []:
                        if item.get("type") == "input_text" and isinstance(item.get("text"), str):
                            text = item["text"]
                            user_text_parts.append(text)
                            if not is_boilerplate_user_message(text):
                                user_task_parts.append(text)
                elif ptype == "function_call":
                    tool_calls += 1
                    name = payload.get("name")
                    if isinstance(name, str) and name:
                        tool_names[name] = tool_names.get(name, 0) + 1
                elif ptype == "custom_tool_call":
                    tool_calls += 1
                    name = payload.get("name")
                    if isinstance(name, str) and name:
                        tool_names[name] = tool_names.get(name, 0) + 1

    user_text = "\n".join(user_text_parts).strip()
    user_task_text = "\n".join(user_task_parts).strip()
    if not user_task_text:
        user_task_text = user_text
    return RolloutSummary(
        path=rollout_path,
        session_id=session_id,
        session_timestamp=session_timestamp,
        user_text=user_text,
        user_task_text=user_task_text,
        tool_calls=tool_calls,
        tool_names=dict(sorted(tool_names.items(), key=lambda kv: (-kv[1], kv[0]))),
        totals=totals,
    )


def summarize_user_text(user_text: str) -> str:
    if not user_text:
        return "(empty)"
    s = " ".join(user_text.split())
    if len(s) <= 80:
        return s
    return s[:80] + "…"


def sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(a=a, b=b).ratio()


def fmt_money(x: float) -> str:
    return f"${x:.7f}"


def main() -> int:
    args = parse_args()
    yyyy, mm, dd = args.date.split("-")
    sessions_dir = os.path.join(args.sessions_root, yyyy, mm, dd)
    rollouts = sorted(glob.glob(os.path.join(sessions_dir, "rollout-*.jsonl")))

    if not rollouts:
        raise SystemExit(f"未找到 rollout：{sessions_dir}/rollout-*.jsonl")

    r = args.pout / args.pin

    overall = TokenTotals()
    per_file: list[RolloutSummary] = []

    for path in rollouts:
        summary = load_rollout_summary(path)
        if summary is None:
            continue
        overall.add(summary.totals)
        per_file.append(summary)

    if overall.input_tokens <= 0:
        raise SystemExit("未读取到 token_count.total_token_usage（input_tokens=0）")

    I = overall.input_tokens
    C = overall.cached_input_tokens
    U = I - C
    O = overall.output_tokens

    hit_rate = (C / I) if I else 0.0

    E = U + args.alpha * C + r * O
    W = (U + O) if args.denom == "U+O" else U

    if W <= 0:
        raise SystemExit("分母（有效工作量）为 0，无法计算倍率。")

    M = E / W
    P_eff = args.pin * M

    Cin = (U + args.alpha * C) / 1_000_000 * args.pin
    Cout = O / 1_000_000 * args.pout
    Ctotal = Cin + Cout

    baseline_cost = W / 1_000_000 * args.pin

    print(f"date: {args.date}")
    print(f"rollouts: {len(per_file)} (in {sessions_dir})")
    print("---")
    print(f"Pin: ${args.pin:.6f}/M, Pout: ${args.pout:.6f}/M, alpha: {args.alpha:g}, r=Pout/Pin: {r:g}")
    print(f"denom: {args.denom}")
    print("---")
    print(f"I(input_tokens): {I}")
    print(f"C(cached_input_tokens): {C}")
    print(f"U(uncached=I-C): {U}")
    print(f"O(output_tokens): {O}")
    print(f"cache_hit_rate(C/I): {hit_rate:.6%}")
    print("---")
    print(f"E(equivalent_baseline_tokens): {E:.1f}")
    print(f"W(work_tokens): {W}")
    print(f"M(real_multiplier=E/W): {M:.6f}x")
    print(f"P_eff(effective_price_per_M_work): ${P_eff:.6f}/M")
    print("---")
    print(f"Cin(input_cost): {fmt_money(Cin)}")
    print(f"Cout(output_cost): {fmt_money(Cout)}")
    print(f"Ctotal(total_cost): {fmt_money(Ctotal)}")
    print(f"baseline_cost(work_as_input): {fmt_money(baseline_cost)}")
    print(f"M(cost_ratio): {(Ctotal / baseline_cost) if baseline_cost else 0.0:.6f}x")

    if args.breakdown:
        print("---")
        print("per_rollout_breakdown:")

        # Build similarity against earlier sessions to emulate:
        # A: cold start (low similarity), B: repeat (very high similarity), C: small edit (medium-high),
        # D: tool-dense (many tool calls)
        summaries_sorted = sorted(
            per_file,
            key=lambda s: (s.session_timestamp or "", os.path.basename(s.path)),
        )

        # Precompute max similarity to any earlier session
        max_sim_to_prior: dict[str, float] = {}
        best_prior: dict[str, Optional[str]] = {}
        for i, s1 in enumerate(summaries_sorted):
            best = 0.0
            best_path: Optional[str] = None
            for j in range(i):
                s0 = summaries_sorted[j]
                score = sim(s0.user_task_text, s1.user_task_text)
                if score > best:
                    best = score
                    best_path = s0.path
            max_sim_to_prior[s1.path] = best
            best_prior[s1.path] = best_path

        def classify_one(s: RolloutSummary) -> str:
            if not args.classify:
                return "-"
            # Tool-dense first: lots of tool calls usually indicates “工具密集型”
            if s.tool_calls >= 30:
                return "tools_dense"
            score = max_sim_to_prior.get(s.path, 0.0)
            if score >= 0.995:
                return "repeat_prompt"
            if score >= 0.85:
                return "small_edit"
            return "cold_start"

        for s in summaries_sorted:
            t = s.totals
            I1 = t.input_tokens
            C1 = t.cached_input_tokens
            U1 = I1 - C1
            O1 = t.output_tokens
            E1 = U1 + args.alpha * C1 + r * O1
            W1 = (U1 + O1) if args.denom == "U+O" else U1
            M1 = (E1 / W1) if W1 else 0.0

            label = classify_one(s)
            score = max_sim_to_prior.get(s.path, 0.0)
            prior = best_prior.get(s.path)

            print(f"- file: {os.path.basename(s.path)}")
            print(f"  session_id: {s.session_id or '-'}")
            print(f"  session_ts: {s.session_timestamp or '-'}")
            print(f"  label: {label}")
            if args.classify:
                print(f"  max_similarity_to_prior: {score:.4f}")
                print(f"  best_prior_file: {os.path.basename(prior) if prior else '-'}")
            print(f"  user_task_preview: {summarize_user_text(s.user_task_text)}")
            print(f"  tool_calls: {s.tool_calls}")
            if s.tool_names:
                top = ", ".join([f"{k}={v}" for k, v in list(s.tool_names.items())[:6]])
                print(f"  tool_top: {top}")
            print(f"  I/C/U/O: {I1}/{C1}/{U1}/{O1}")
            print(f"  cache_hit_rate: {(C1 / I1) if I1 else 0.0:.6%}")
            print(f"  M(E/W): {M1:.6f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
