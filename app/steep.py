"""持续陡坡分析：复用平滑海拔剖面，以滑动 100 米窗口识别持续上/下坡。

- 基于上传时保存的原始 GPX 轨迹（各原始 trkseg）计算，与遮蔽裁切、配速、
  漂移修复、停留识别、轨迹精简互不影响；
- 直接复用 elevation_profile 的结果：每个连续有效海拔区间已按 20 米重采样并做
  三点中位数平滑，缺失海拔处与原始 trkseg 边界天然断开；
- 每个区间内以完整 100 米窗口的首末高差计算坡度（高差 ÷ 100 米），窗口每次前移
  20 米；区间末尾不足 100 米的部分不产生窗口；
- 窗口坡度 ≥ 8% 记为上坡、≤ -8% 记为下坡，标记窗口中心前后各 10 米（共 20 米）
  的里程区间；
- 相接的同向标记合并为一个陡坡段；合并在单个连续有效海拔区间内进行，
  不跨原始 trkseg 边界或缺失海拔缺口；
- 合并后标记总长 ≥ 100 米的段才列出：起止里程（全程水平里程）、方向与段内各窗口
  坡度的最大绝对值；
- 全程无任何有效海拔时 has_ele=False、sections 为空；有海拔但无符合段时 sections 为空。
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

from .elevation import elevation_profile

WINDOW_M = 100.0          # 坡度窗口长度（米）：取首末高差
WINDOW_STEP_M = 20.0      # 窗口前移步长（米）
MARK_M = 20.0             # 命中窗口标记的里程区间长度（米，以窗口中心为中心）
GRADE_THRESHOLD = 0.08    # 陡坡坡度阈值（上坡 ≥、下坡 ≤ -）
MIN_SECTION_M = 100.0     # 列出陡坡段的最小标记总长（米）
_EPS_M = 1e-6             # 里程判定的浮点噪声阈值（米）
_EPS_GRADE = 1e-9         # 坡度阈值判定的浮点噪声阈值（仅吸收浮点误差）


@dataclass
class SteepSection:
    """一段持续陡坡（合并后的同向标记）。"""

    seg_index: int        # 原始 trkseg 序号（从 1 开始）
    direction: str        # "up" 上坡 / "down" 下坡
    start_m: float        # 段起点在全程水平里程中的位置（米）
    end_m: float          # 段终点在全程水平里程中的位置（米）
    max_abs_grade: float  # 段内各窗口坡度的最大绝对值（小数，如 0.12 表示 12%）


@dataclass
class SteepResult:
    """持续陡坡分析结果；无任何有效海拔时 has_ele=False。"""

    has_ele: bool
    distance_m: float           # 全程水平里程（含无海拔部分）
    sections: list = field(default_factory=list)  # list[SteepSection]


def _ele_at(ds: list, es: list, d: float) -> float:
    """在区间采样点（里程 ds 单调不减）上线性插值距离 d 处的海拔。"""
    if d <= ds[0]:
        return es[0]
    if d >= ds[-1]:
        return es[-1]
    i = bisect_right(ds, d) - 1
    d0, d1 = ds[i], ds[i + 1]
    if d1 <= d0:  # 里程重复的采样点：取前者
        return es[i]
    t = (d - d0) / (d1 - d0)
    return es[i] + (es[i + 1] - es[i]) * t


def _run_marks(ds: list, es: list) -> list:
    """单个连续有效海拔区间内的陡坡窗口标记。

    ds / es: 区间内里程（单调不减）与对应的平滑海拔；
    返回 [(标记起点, 标记终点, 方向, 坡度), ...]，里程为区间内里程。
    只使用完整 100 米窗口（区间末尾不足 100 米的部分不产生窗口）。
    """
    total = ds[-1]
    marks = []
    s = 0.0
    while s + WINDOW_M <= total + _EPS_M:
        grade = (_ele_at(ds, es, s + WINDOW_M) - _ele_at(ds, es, s)) / WINDOW_M
        center = s + WINDOW_M / 2.0
        half = MARK_M / 2.0
        if grade >= GRADE_THRESHOLD - _EPS_GRADE:
            marks.append((center - half, center + half, "up", grade))
        elif grade <= -(GRADE_THRESHOLD - _EPS_GRADE):
            marks.append((center - half, center + half, "down", grade))
        s += WINDOW_STEP_M
    return marks


def _merge_marks(marks: list) -> list:
    """相接的同向标记合并；返回 [(起点, 终点, 方向, 最大绝对坡度), ...]。"""
    merged = []
    for start, end, direction, grade in marks:
        if (merged and merged[-1][2] == direction
                and start <= merged[-1][1] + _EPS_M):
            last = merged[-1]
            merged[-1] = (last[0], max(last[1], end), direction,
                          max(last[3], abs(grade)))
        else:
            merged.append((start, end, direction, abs(grade)))
    return merged


def steep_sections(segments: list) -> SteepResult:
    """对整条轨迹（若干原始 trkseg）识别持续陡坡段。

    复用 elevation_profile 的平滑剖面：每个连续有效海拔区间独立加窗标记与合并，
    因此原始 trkseg 边界与缺失海拔缺口处不会合并；段里程为全程水平里程。
    """
    profile = elevation_profile(segments)
    sections = []
    for run in profile.runs:
        base = run.points[0][0]  # 区间起点的全程里程
        ds = [d - base for d, _ in run.points]
        es = [e for _, e in run.points]
        for start, end, direction, max_abs in _merge_marks(_run_marks(ds, es)):
            if end - start < MIN_SECTION_M - _EPS_M:
                continue  # 标记总长不足 100 米的短段不列出
            sections.append(SteepSection(
                seg_index=run.seg_index,
                direction=direction,
                start_m=base + start,
                end_m=base + end,
                max_abs_grade=max_abs,
            ))
    return SteepResult(
        has_ele=profile.has_ele,
        distance_m=profile.distance_m,
        sections=sections,
    )
