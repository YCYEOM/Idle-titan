#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TT3 성장 경제 몬테카를로 시뮬레이터
====================================
목적: "1시간 체크인 최적 / 저빈도 유저 손실 30% 이내" 설계가
      랜덤성(축복 3택1)과 계단 함수(영웅 해금)를 넣어도 유지되는지 검증.

핵심 검증 지표
  - B/A 비율 (타깃 유저 / 저빈도 유저의 일일 성장 단위) → 목표 1.60 ~ 1.80
  - 연쇄(Chain) 기여도 → 목표 10% 내외, 20% 초과 시 위험
  - 클립 트리거 발생량 (전설 축복 기준)

실행:
    python3 tt3_sim.py                  # 기본 검증
    python3 tt3_sim.py --sweep          # 민감도 분석 포함
    python3 tt3_sim.py --trials 1000 --days 60 --csv out.csv

의존성: 표준 라이브러리만 사용 (numpy/pandas 불필요)
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

# ============================================================
# 1. 설계 파라미터 (v4 문서의 수치를 그대로 코드화)
# ============================================================


@dataclass
class Config:
    # --- 2층: 균열석 ---
    stack_base: float = 60.0            # 스택 1개당 균열석 기본량
    stack_cap: int = 3                  # 동시 보유 스택 상한
    # ★ 55분. 60분 정각으로 두면 체크인 시각 흔들림과 경주가 되어
    #   약 50% 확률로 '스택 0개' 헛걸음이 발생한다 (시뮬레이션으로 발견된 함정).
    stack_interval_h: float = 55.0 / 60.0

    # 신선도 계수: (경과시간 상한, 계수) — 마지막 항목이 하한
    freshness_table: Tuple[Tuple[float, float], ...] = (
        (1.0, 1.00),
        (2.0, 0.90),
        (3.0, 0.80),
    )
    freshness_floor: float = 0.70       # ★ 저빈도 유저의 생명줄. 절대 건드리지 말 것

    # --- 연쇄(Chain) ---
    chain_window_h: float = 1.5         # 90분 이내 재수령 시 연쇄 인정
    chain_mults: Tuple[float, ...] = (1.05, 1.10, 1.18, 1.26, 1.35)
    chain_daily_cap: int = 5            # ★ 일 5회 상한 (근무일 내 완주 가능)
    chain_complete_bonus: float = 120.0  # 5연쇄 완주 보너스 균열석

    # --- 1층: 금화 (체크인 빈도와 무관해야 함) ---
    gold_per_hour: float = 100.0        # 정규화 단위
    gold_offline_eff: float = 0.90

    # --- 환산 ---
    rift_to_unit: float = 3.0           # 균열석 1개 = 성장 단위 몇 개인가

    # --- 각인 (균열석 소비처) ---
    engrave_cost_base: float = 40.0
    engrave_cost_growth: float = 1.08
    engrave_smax_gain: float = 2.5      # 각인 1레벨당 회차 한계 스테이지 +2.5

    # --- 프레스티지 ---
    smax_initial: float = 100.0
    tau_h: float = 3.0                  # ★ 진행 시상수. 최적 회차 길이를 결정하는 심장
    relic_base: float = 1.02            # ★ 유물 획득 밑수. 같은 축의 심장
    relic_coef: float = 2.0
    relic_offset: float = 50.0
    prestige_target_h: float = 10.0     # 이 시간 이상 경과한 체크인에서 승급
    relic_feedback_coef: float = 30.0   # 유물 누적 → s_max 영구 환류 (log 스케일)

    # --- 온보딩 예외 (D_쇼츠유입 전용) ---
    # 쇼츠로 유입된 신규 유저는 첫 승급 '한 번'만 짧은 시상수를 적용해
    # 초반 진도를 압축한다. 두 번째 승급부터는 기본 tau_h로 복귀하므로
    # 장기 곡선(B/A 밸런스)에는 영향을 주지 않는다.
    onboarding: bool = False            # True일 때만 온보딩 예외 활성화
    onboarding_tau_h: float = 0.8       # ★ 첫 회차 한정 진행 시상수

    # --- 축복 3택1 (랜덤성 주입원) ---
    # (확률, 다음 세션까지 적용되는 배율, 클립트리거 여부)
    blessing_table: Tuple[Tuple[float, float, bool], ...] = (
        (0.60, 1.00, False),   # 일반
        (0.27, 1.10, False),   # 희귀
        (0.10, 1.25, False),   # 영웅
        (0.03, 1.60, True),    # 전설 → 자동 클립 트리거
    )
    upgrade_ticket_shift: float = 0.5   # 등급상향권 사용 시 상위 티어로 이동할 확률

    # --- 영웅 해금 계단 (곡선 왜곡원) ---
    hero_step_smax: float = 50.0        # s_max가 이만큼 오를 때마다
    hero_step_mult: float = 1.15        # 금화 수급 계단 상승

    # --- 체크인 시각 흔들림 ---
    jitter_min: float = 12.0            # 표준편차(분). 회의/통화로 밀리는 현실 반영
    skip_prob: float = 0.08             # 체크인 자체를 건너뛸 확률


# ============================================================
# 2. 페르소나 (체크인 스케줄, 단위: 시)
# ============================================================

PERSONAS: Dict[str, List[float]] = {
    # 저빈도: 기상 + 취침
    "A_저빈도": [8.0, 23.0],
    # 타깃: 기상 + 근무 중 5회(연쇄 완주) + 퇴근 + 취침
    "B_타깃": [8.0, 9.0, 10.0, 11.33, 12.5, 13.67, 18.0, 23.0],
    # 헤비: 깨어있는 내내 1시간 간격
    "C_헤비": [8.0 + i for i in range(0, 16)],
    # 쇼츠유입: 초반 관심은 높지만 습관화 전. 하루 4회의 중간 빈도.
    # 첫 승급에만 온보딩 예외(짧은 tau_h)로 진도를 압축해 B형을 빠르게 추격.
    "D_쇼츠유입": [8.0, 12.5, 18.0, 22.5],
}

# 온보딩 예외를 적용받는 페르소나 집합 (첫 승급 1회 한정 tau_h 단축)
ONBOARDING_PERSONAS = frozenset({"D_쇼츠유입"})


# ============================================================
# 3. 시뮬레이션 엔진
# ============================================================


class Player:
    def __init__(self, cfg: Config, schedule: List[float], rng: random.Random):
        self.cfg = cfg
        self.schedule = schedule
        self.rng = rng

        self.t = 0.0                    # 시뮬레이션 절대 시각(시)
        self.smax_base = cfg.smax_initial
        self.engrave_lv = 0
        self.total_relics = 0.0

        self.rift_balance = 0.0
        self.rift_collected = 0.0       # 누적 획득 균열석(원화)
        self.rift_units = 0.0           # 진도 스케일 적용된 성장 단위
        self.rift_from_chain = 0.0      # 그중 연쇄 보너스 기여분(단위 환산)
        self.gold_units = 0.0

        self.stacks: List[float] = []   # 각 스택의 '생성 시각' 기억
        self.next_stack_t = cfg.stack_interval_h

        self.last_collect_t = -999.0
        self.chain_today = 0
        self.has_ticket = False

        self.last_prestige_t = 0.0
        self.prestige_count = 0          # 누적 승급 횟수 (온보딩 첫 회차 판정용)
        self.blessing_mult = 1.0
        self.clips = 0
        self.checkins = 0

        self.daily_log: List[Dict[str, float]] = []

    # ---- s_max: 각인 + 유물 환류 ----
    @property
    def smax(self) -> float:
        relic_bonus = self.cfg.relic_feedback_coef * math.log10(1.0 + self.total_relics)
        return self.smax_base + self.engrave_lv * self.cfg.engrave_smax_gain + relic_bonus

    # ---- 영웅 해금 계단 배율 ----
    def hero_mult(self) -> float:
        steps = int((self.smax - self.cfg.smax_initial) / self.cfg.hero_step_smax)
        steps = max(0, steps)
        return self.cfg.hero_step_mult ** steps

    # ---- 1층 금화 누적 ----
    def accrue_gold(self, until: float) -> None:
        dt = until - self.t
        if dt <= 0:
            return
        rate = (
            self.cfg.gold_per_hour
            * self.cfg.gold_offline_eff
            * self.hero_mult()
            * self.blessing_mult
        )
        self.gold_units += rate * dt
        self.t = until

    # ---- 2층 스택 생성 (상한 도달 시 생성 정지) ----
    def generate_stacks(self, now: float) -> None:
        while len(self.stacks) < self.cfg.stack_cap and self.next_stack_t <= now:
            self.stacks.append(self.next_stack_t)
            self.next_stack_t += self.cfg.stack_interval_h

    def freshness(self, age_h: float) -> float:
        for limit, mult in self.cfg.freshness_table:
            if age_h <= limit:
                return mult
        return self.cfg.freshness_floor

    # ---- 체크인 1회 ----
    def check_in(self, now: float) -> None:
        self.accrue_gold(now)
        self.generate_stacks(now)
        self.checkins += 1

        if not self.stacks:
            return

        # 신선도 적용
        raw = sum(self.cfg.stack_base * self.freshness(now - born) for born in self.stacks)

        # 연쇄 판정
        within = (now - self.last_collect_t) <= self.cfg.chain_window_h
        if within and self.chain_today < self.cfg.chain_daily_cap:
            self.chain_today += 1
            idx = min(self.chain_today, len(self.cfg.chain_mults)) - 1
            mult = self.cfg.chain_mults[idx]
        elif not within:
            self.chain_today = 0        # 끊기면 초기화만. 감소 페널티 없음
            mult = 1.0
        else:
            mult = 1.0                  # 일일 상한 도달 → 더 켜도 이득 없음

        scale = self.hero_mult()        # ★ 2층 가치도 진도에 비례 (v4 '스케일 프리' 정의)
        gained = raw * mult
        self.rift_from_chain += raw * (mult - 1.0) * self.cfg.rift_to_unit * scale

        # 5연쇄 완주 보너스
        if self.chain_today == self.cfg.chain_daily_cap and within:
            gained += self.cfg.chain_complete_bonus
            self.rift_from_chain += self.cfg.chain_complete_bonus * self.cfg.rift_to_unit * scale
            self.has_ticket = True

        self.rift_balance += gained
        self.rift_collected += gained
        self.rift_units += gained * self.cfg.rift_to_unit * scale

        # 스택 비우고 생성 타이머 재시작
        self.stacks.clear()
        self.next_stack_t = now + self.cfg.stack_interval_h
        self.last_collect_t = now

        self.spend_on_engrave()
        self.maybe_prestige(now)
        self.draw_blessing()

    # ---- 각인 강화 (탐욕적 소비) ----
    def spend_on_engrave(self) -> None:
        while True:
            cost = self.cfg.engrave_cost_base * (
                self.cfg.engrave_cost_growth ** self.engrave_lv
            )
            if self.rift_balance < cost:
                break
            self.rift_balance -= cost
            self.engrave_lv += 1

    # ---- 프레스티지 ----
    def maybe_prestige(self, now: float) -> None:
        elapsed = now - self.last_prestige_t
        if elapsed < self.cfg.prestige_target_h:
            return
        # ★ 온보딩 예외: 첫 승급(prestige_count == 0)에만 짧은 tau_h를 적용.
        #   짧은 시상수는 exp 항을 빠르게 0으로 보내 stage를 s_max에 근접시키고,
        #   결과적으로 초반 유물을 크게 부여해 신규 유저의 진도를 끌어올린다.
        if self.cfg.onboarding and self.prestige_count == 0:
            tau = self.cfg.onboarding_tau_h
        else:
            tau = self.cfg.tau_h
        stage = self.smax * (1.0 - math.exp(-elapsed / tau))
        exponent = stage - self.cfg.relic_offset
        try:
            relics = self.cfg.relic_coef * (self.cfg.relic_base ** exponent)
        except OverflowError:
            relics = float("inf")
        self.total_relics += max(0.0, relics)
        self.last_prestige_t = now
        self.prestige_count += 1

    # ---- 축복 3택1 ----
    def draw_blessing(self) -> None:
        idx = self._sample_tier()
        if self.has_ticket and self.rng.random() < self.cfg.upgrade_ticket_shift:
            idx = min(idx + 1, len(self.cfg.blessing_table) - 1)
            self.has_ticket = False
        _, mult, is_clip = self.cfg.blessing_table[idx]
        self.blessing_mult = mult
        if is_clip:
            self.clips += 1

    def _sample_tier(self) -> int:
        # 3택1이므로 3회 추첨 후 최상위 선택 (플레이어는 합리적으로 행동)
        best = 0
        for _ in range(3):
            r = self.rng.random()
            acc = 0.0
            for i, (p, _m, _c) in enumerate(self.cfg.blessing_table):
                acc += p
                if r <= acc:
                    best = max(best, i)
                    break
        return best

    # ---- 하루 실행 ----
    def run_day(self, day: int) -> None:
        day_start = day * 24.0
        times = []
        for base in self.schedule:
            if self.rng.random() < self.cfg.skip_prob:
                continue
            jitter = self.rng.gauss(0.0, self.cfg.jitter_min / 60.0)
            times.append(day_start + min(23.9, max(0.0, base + jitter)))
        times.sort()

        # 최소 간격 5분 보정
        cleaned: List[float] = []
        for t in times:
            if cleaned and t - cleaned[-1] < 5 / 60:
                continue
            cleaned.append(t)

        snap_gold, snap_rift = self.gold_units, self.rift_units
        for t in cleaned:
            if t <= self.t:
                continue
            self.check_in(t)
        self.accrue_gold(day_start + 24.0)

        self.daily_log.append(
            {
                "day": day + 1,
                "gold": self.gold_units - snap_gold,
                "rift": self.rift_units - snap_rift,
                "smax": self.smax,
            }
        )
        self.chain_today = 0            # 일일 연쇄 초기화


def growth_units(p: Player, cfg: Config) -> float:
    return p.gold_units + p.rift_units


def layer2_share(p: Player, cfg: Config) -> float:
    return p.rift_units / max(1e-9, growth_units(p, cfg))


def simulate(cfg: Config, name: str, days: int, seed: int) -> Player:
    rng = random.Random(seed)
    # 온보딩 페르소나는 첫 승급 예외를 켠 config 사본으로 실행한다.
    # (다른 페르소나·전역 밸런스에는 영향을 주지 않도록 사본만 교체)
    if name in ONBOARDING_PERSONAS and not cfg.onboarding:
        cfg = replace(cfg, onboarding=True)
    p = Player(cfg, PERSONAS[name], rng)
    for d in range(days):
        p.run_day(d)
    return p


# ============================================================
# 4. 배치 실행 및 리포트
# ============================================================


@dataclass
class Result:
    name: str
    units_mean: float
    units_p10: float
    units_p90: float
    smax_mean: float
    log10_relics: float
    chain_share: float
    clips_per_day: float
    checkins_per_day: float
    layer2_share: float = 0.0
    steady_units: float = 0.0


def run_batch(cfg: Config, days: int, trials: int, seed0: int = 1) -> Dict[str, Result]:
    out: Dict[str, Result] = {}
    for name in PERSONAS:
        units, smaxes, relics, chain, clips, checks, steady, l2 = [], [], [], [], [], [], [], []
        for k in range(trials):
            p = simulate(cfg, name, days, seed0 + k * 7919)
            units.append(growth_units(p, cfg))
            smaxes.append(p.smax)
            relics.append(math.log10(max(1.0, p.total_relics)))
            chain.append(p.rift_from_chain / max(1e-9, growth_units(p, cfg)))
            clips.append(p.clips / days)
            checks.append(p.checkins / days)
            l2.append(layer2_share(p, cfg))
            tail = p.daily_log[-7:]
            steady.append(sum(d["gold"] + d["rift"] for d in tail) / len(tail))
        srt = sorted(units)
        out[name] = Result(
            name=name,
            units_mean=statistics.fmean(units),
            units_p10=srt[int(0.10 * (len(srt) - 1))],
            units_p90=srt[int(0.90 * (len(srt) - 1))],
            smax_mean=statistics.fmean(smaxes),
            log10_relics=statistics.fmean(relics),
            chain_share=statistics.fmean(chain),
            clips_per_day=statistics.fmean(clips),
            checkins_per_day=statistics.fmean(checks),
            layer2_share=statistics.fmean(l2),
            steady_units=statistics.fmean(steady),
        )
    return out


BAR = "=" * 76


def report(cfg: Config, res: Dict[str, Result], days: int, trials: int) -> float:
    a, b, c = res["A_저빈도"], res["B_타깃"], res["C_헤비"]
    d = res.get("D_쇼츠유입")
    ratio = b.units_mean / a.units_mean
    ratio_heavy = c.units_mean / a.units_mean

    print(BAR)
    print(f" TT3 경제 시뮬레이션 | {days}일 × {trials}회 시행")
    print(BAR)
    hdr = f"{'페르소나':<12}{'체크인/일':>9}{'성장단위':>14}{'p10':>12}{'p90':>12}{'연쇄기여':>9}{'클립/일':>9}"
    print(hdr)
    print("-" * 76)
    table = (a, b, c, d) if d is not None else (a, b, c)
    for r in table:
        print(
            f"{r.name:<12}{r.checkins_per_day:>9.1f}{r.units_mean:>14,.0f}"
            f"{r.units_p10:>12,.0f}{r.units_p90:>12,.0f}"
            f"{r.chain_share*100:>8.1f}%{r.clips_per_day:>9.2f}"
        )
    print("-" * 76)
    print(f"{'최종 s_max':<12}", end="")
    for r in (a, b, c):
        print(f"{r.smax_mean:>20,.0f}", end="")
    print()
    print(f"{'log10(유물)':<12}", end="")
    for r in (a, b, c):
        print(f"{r.log10_relics:>20,.2f}", end="")
    print()
    print(BAR)

    def verdict(v: float, lo: float, hi: float) -> str:
        if v < lo:
            return "✗ 미달 (체크인 동기 부족)"
        if v > hi:
            return "✗ 초과 (라이트층 이탈 위험)"
        return "✓ 안전 구간"

    print(f" B/A 비율      : {ratio:.3f}   목표 1.60~1.80   {verdict(ratio, 1.60, 1.80)}")
    print(f" C/A 비율      : {ratio_heavy:.3f}   경고선 2.20     "
          f"{'✓' if ratio_heavy <= 2.20 else '✗ 헤비 독주'}")
    print(f" 연쇄 기여(B)  : {b.chain_share*100:.1f}%  목표 ~10%      "
          f"{'✓' if b.chain_share <= 0.20 else '✗ 연쇄 압박 과다'}")
    print(f" 2층 비중(B)   : {b.layer2_share*100:.1f}%  권장 40~55%   "
          f"{'✓' if 0.40 <= b.layer2_share <= 0.55 else '✗ 한쪽 축이 장식으로 전락'}")
    print(f" 정상상태 B/A  : {b.steady_units / max(1e-9, a.steady_units):.3f} (최근 7일 기준)")
    print(BAR)
    return ratio


# ============================================================
# 5. 민감도 분석 — "어느 파라미터가 설계를 무너뜨리는가"
# ============================================================

SWEEPS: List[Tuple[str, List[float]]] = [
    ("freshness_floor", [0.50, 0.60, 0.70, 0.80]),
    ("rift_to_unit", [2.20, 3.00, 4.00]),
    ("tau_h", [1.50, 2.50, 3.00, 3.50, 4.50]),
    ("relic_base", [1.018, 1.020, 1.022, 1.030]),
    ("chain_daily_cap", [3, 5, 8]),
    ("stack_cap", [2, 3, 5]),
    ("prestige_target_h", [6.0, 10.0, 14.0]),
]


def run_sweep(cfg: Config, days: int, trials: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    print("\n" + BAR)
    print(" 민감도 분석 (B/A 비율, 안전 구간 1.60~1.80)")
    print(BAR)
    for key, values in SWEEPS:
        print(f"\n[{key}]")
        for v in values:
            vv = int(v) if key in ("chain_daily_cap", "stack_cap") else v
            trial_cfg = replace(cfg, **{key: vv})
            r = run_batch(trial_cfg, days, trials)
            ratio = r["B_타깃"].units_mean / r["A_저빈도"].units_mean
            chain = r["B_타깃"].chain_share
            flag = "✓" if 1.60 <= ratio <= 1.80 else "✗"
            base = " ←기본값" if getattr(cfg, key) == vv else ""
            print(f"   {key}={vv:<8} B/A={ratio:5.3f} {flag}  연쇄기여={chain*100:4.1f}%{base}")
            rows.append({"param": key, "value": vv, "ratio": ratio, "chain_share": chain})
    print(BAR)
    return rows


# ============================================================
# 5b. D_쇼츠유입 온보딩 검증
# ============================================================


def verify_onboarding_d(
    cfg: Config, days: int = 7, trials: int = 300, seed0: int = 1
) -> Dict[str, float]:
    """D형이 7일 내에 B형 '진도'의 80%에 도달하는지 검증.

    진도 지표는 두 축으로 본다.
      - units : 누적 성장 단위(금화+균열석) → 실질 성장량
      - smax  : 진행 한계 스테이지 → 콘텐츠 도달 깊이
    둘 다 80% 이상이면 온보딩 설계가 목표를 만족한 것으로 판정한다.
    """
    def batch(name: str) -> Tuple[float, float]:
        us, sm = [], []
        for k in range(trials):
            p = simulate(cfg, name, days, seed0 + k * 7919)
            us.append(growth_units(p, cfg))
            sm.append(p.smax)
        return statistics.fmean(us), statistics.fmean(sm)

    b_units, b_smax = batch("B_타깃")
    d_units, d_smax = batch("D_쇼츠유입")

    ratio_units = d_units / max(1e-9, b_units)
    ratio_smax = d_smax / max(1e-9, b_smax)
    passed = ratio_units >= 0.80 and ratio_smax >= 0.80

    print(BAR)
    print(f" D_쇼츠유입 온보딩 검증 | {days}일 × {trials}회 시행 (목표: B형 진도의 80%+)")
    print(BAR)
    print(f"{'지표':<14}{'B_타깃':>16}{'D_쇼츠유입':>16}{'D/B':>10}{'판정':>10}")
    print("-" * 76)
    print(f"{'성장단위':<14}{b_units:>16,.0f}{d_units:>16,.0f}"
          f"{ratio_units:>10.3f}{'✓' if ratio_units >= 0.80 else '✗':>9}")
    print(f"{'s_max(진도)':<14}{b_smax:>16,.0f}{d_smax:>16,.0f}"
          f"{ratio_smax:>10.3f}{'✓' if ratio_smax >= 0.80 else '✗':>9}")
    print("-" * 76)
    verdict = "✓ 온보딩 성공 (7일 내 80% 도달)" if passed else "✗ 온보딩 미달 (부스트 부족)"
    print(f" 종합 판정 : {verdict}")
    print(BAR)
    return {
        "ratio_units": ratio_units,
        "ratio_smax": ratio_smax,
        "passed": float(passed),
    }


# ============================================================
# 5c. chain_mults 그리드 탐색
# ============================================================

# 5개 연쇄 배율을 (first, last) 2-노브로 압축: 나머지는 선형 보간.
# 단조증가 구조를 유지하면서 탐색 공간을 실용적으로 줄인다.
GRID_FIRST: Tuple[float, ...] = (1.05, 1.15, 1.25, 1.35, 1.45)
GRID_LAST: Tuple[float, ...] = (2.00, 2.60, 3.20, 3.60, 4.00, 4.50, 5.00)

# 목표 구간
BA_LO, BA_HI = 1.60, 1.80
CHAIN_LO, CHAIN_HI = 0.08, 0.12
CHAIN_TARGET = 0.10


def _make_chain_mults(first: float, last: float, n: int = 5) -> Tuple[float, ...]:
    if n == 1:
        return (first,)
    return tuple(first + (last - first) * i / (n - 1) for i in range(n))


def _ba_and_chain(cfg: Config, days: int, trials: int, seed0: int = 1) -> Tuple[float, float]:
    """A/B만 시뮬레이션해 (B/A 비율, B형 연쇄기여율)을 빠르게 산출."""
    def mean_units(name: str) -> Tuple[float, float]:
        us, ch = [], []
        for k in range(trials):
            p = simulate(cfg, name, days, seed0 + k * 7919)
            gu = growth_units(p, cfg)
            us.append(gu)
            ch.append(p.rift_from_chain / max(1e-9, gu))
        return statistics.fmean(us), statistics.fmean(ch)

    a_u, _ = mean_units("A_저빈도")
    b_u, b_ch = mean_units("B_타깃")
    return b_u / max(1e-9, a_u), b_ch


def grid_search_chain_mults(
    cfg: Config, days: int = 30, trials: int = 80, seed0: int = 1
) -> List[Dict[str, object]]:
    """chain_mults 그리드 탐색.

    제약: B/A ∈ [1.60, 1.80]  AND  연쇄기여(B) ∈ [8%, 12%]
    두 조건을 모두 만족하는 조합을 목표 근접도 순으로 정렬해 상위 5개 출력.
    """
    print("\n" + BAR)
    print(f" chain_mults 그리드 탐색 | {days}일 × {trials}회 | "
          f"제약: B/A∈[{BA_LO},{BA_HI}], 연쇄기여∈[{int(CHAIN_LO*100)}%,{int(CHAIN_HI*100)}%]")
    print(BAR)

    rows: List[Dict[str, object]] = []
    for first in GRID_FIRST:
        for last in GRID_LAST:
            if last <= first:
                continue
            mults = _make_chain_mults(first, last)
            trial_cfg = replace(cfg, chain_mults=mults)
            ratio, chain = _ba_and_chain(trial_cfg, days, trials, seed0)
            ba_ok = BA_LO <= ratio <= BA_HI
            ch_ok = CHAIN_LO <= chain <= CHAIN_HI
            valid = ba_ok and ch_ok
            # 점수: 목표(연쇄 10%, B/A 1.70)로부터의 거리 (낮을수록 우수)
            score = abs(chain - CHAIN_TARGET) * 10.0 + abs(ratio - 1.70)
            rows.append({
                "first": first, "last": last, "mults": mults,
                "ratio": ratio, "chain": chain,
                "ba_ok": ba_ok, "ch_ok": ch_ok, "valid": valid,
                "score": score,
            })

    valid_rows = sorted([r for r in rows if r["valid"]], key=lambda r: r["score"])
    top5 = valid_rows[:5]

    print(f"{'순위':<4}{'first':>7}{'last':>7}{'B/A':>9}{'연쇄기여':>10}"
          f"{'판정':>8}   chain_mults")
    print("-" * 76)
    if top5:
        for i, r in enumerate(top5, 1):
            mstr = "(" + ", ".join(f"{m:.2f}" for m in r["mults"]) + ")"
            print(f"{i:<4}{r['first']:>7.2f}{r['last']:>7.2f}{r['ratio']:>9.3f}"
                  f"{r['chain']*100:>9.1f}%{'✓':>7}   {mstr}")
    else:
        print("  ✗ 두 조건을 동시에 만족하는 조합이 없음. 근접 후보 상위 5개:")
        near = sorted(rows, key=lambda r: r["score"])[:5]
        for i, r in enumerate(near, 1):
            mstr = "(" + ", ".join(f"{m:.2f}" for m in r["mults"]) + ")"
            f_ba = "○" if r["ba_ok"] else "×"
            f_ch = "○" if r["ch_ok"] else "×"
            print(f"{i:<4}{r['first']:>7.2f}{r['last']:>7.2f}{r['ratio']:>9.3f}"
                  f"{r['chain']*100:>9.1f}%  B/A{f_ba} 연쇄{f_ch}  {mstr}")
    print("-" * 76)
    print(f" 탐색 조합 {len(rows)}개 중 조건 만족 {len(valid_rows)}개")
    print(BAR)
    return rows


# ============================================================
# 5d. 그리드 결과 시각화 (matplotlib 선택적)
# ============================================================


def plot_grid(rows: List[Dict[str, object]], path: str = "chain_grid.png") -> None:
    """그리드 결과를 산점도(x=B/A, y=연쇄기여%)로 저장.
    matplotlib이 없으면 ASCII 요약 표로 대체 출력한다.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")            # 헤드리스 환경 대비
        import matplotlib.pyplot as plt
    except Exception:
        print("\n[matplotlib 미설치 → 텍스트 산점 요약]")
        print(f"{'B/A':>8}{'연쇄기여%':>10}  판정")
        print("-" * 40)
        for r in sorted(rows, key=lambda r: (r["ratio"], r["chain"])):
            tag = "✓ 유효" if r["valid"] else ""
            print(f"{r['ratio']:>8.3f}{r['chain']*100:>10.1f}  {tag}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    # 목표 박스 (B/A 1.60~1.80, 연쇄 8~12%)
    ax.axvspan(BA_LO, BA_HI, color="#cfe8cf", alpha=0.4, zorder=0)
    ax.axhspan(CHAIN_LO * 100, CHAIN_HI * 100, color="#cfd8e8", alpha=0.4, zorder=0)

    inval = [r for r in rows if not r["valid"]]
    val = [r for r in rows if r["valid"]]
    if inval:
        ax.scatter([r["ratio"] for r in inval], [r["chain"] * 100 for r in inval],
                   c="#999999", s=40, marker="o", label="out of range", zorder=2)
    if val:
        ax.scatter([r["ratio"] for r in val], [r["chain"] * 100 for r in val],
                   c="#d1495b", s=90, marker="*", label="valid (both targets)", zorder=3,
                   edgecolors="black", linewidths=0.5)

    ax.axvline(1.70, color="#666", ls="--", lw=0.8)
    ax.axhline(CHAIN_TARGET * 100, color="#666", ls="--", lw=0.8)
    ax.set_xlabel("B/A ratio")
    ax.set_ylabel("Chain contribution (%)")
    ax.set_title("chain_mults grid search\n(target box: B/A 1.60-1.80, chain 8-12%)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\n산점도 저장: {path}")


# ============================================================
# 6. 엔트리포인트
# ============================================================


def main() -> None:
    ap = argparse.ArgumentParser(description="TT3 성장 경제 시뮬레이터")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--sweep", action="store_true", help="민감도 분석 실행 (느림)")
    ap.add_argument("--sweep-trials", type=int, default=80)
    ap.add_argument("--csv", type=str, default="", help="일별 로그 CSV 출력 경로")
    ap.add_argument("--verify-d", action="store_true",
                    help="D_쇼츠유입 온보딩 7일 80% 도달 검증")
    ap.add_argument("--grid", action="store_true",
                    help="chain_mults 그리드 탐색 (연쇄기여 8~12% 조합)")
    ap.add_argument("--grid-trials", type=int, default=80)
    ap.add_argument("--grid-days", type=int, default=30)
    ap.add_argument("--plot-path", type=str, default="chain_grid.png",
                    help="그리드 산점도 저장 경로")
    args = ap.parse_args()

    cfg = Config()
    res = run_batch(cfg, args.days, args.trials)
    report(cfg, res, args.days, args.trials)

    if args.verify_d:
        verify_onboarding_d(cfg, days=7, trials=args.trials)

    if args.grid:
        rows = grid_search_chain_mults(cfg, days=args.grid_days, trials=args.grid_trials)
        plot_grid(rows, args.plot_path)

    if args.sweep:
        rows = run_sweep(cfg, args.days, args.sweep_trials)
        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=["param", "value", "ratio", "chain_share"])
                w.writeheader()
                w.writerows(rows)
            print(f"\n민감도 결과 저장: {args.csv}")
    elif args.csv:
        p = simulate(cfg, "B_타깃", args.days, 42)
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["day", "gold", "rift", "smax"])
            w.writeheader()
            w.writerows(p.daily_log)
        print(f"\n일별 로그 저장: {args.csv}")


if __name__ == "__main__":
    main()
