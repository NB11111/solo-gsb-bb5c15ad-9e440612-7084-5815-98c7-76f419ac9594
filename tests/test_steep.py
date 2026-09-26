"""持续陡坡分析的单元测试。

可直接运行：python3 tests/test_steep.py
或使用 pytest：python3 -m pytest tests/
"""
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.geo import EARTH_RADIUS_M, TrackPoint
from app.steep import _merge_marks, _run_marks, steep_sections

T0 = datetime(2026, 5, 1, 8, 0, 0, tzinfo=timezone.utc)


def make_seg(points):
    """points: [(lat, lon, ele 或 None, 秒偏移), ...]"""
    return [
        TrackPoint(lat=la, lon=lo, ele=ele, time=T0 + timedelta(seconds=s))
        for la, lo, ele, s in points
    ]


def step_lat(meters):
    """纯纬度变化时球面距离恰为 meters 的纬度差。"""
    return math.degrees(meters / EARTH_RADIUS_M)


def north_seg(eles, step_m=20.0, lat0=31.0, lon=121.0):
    """沿正北等距行进、海拔按 eles 给定的单 trkseg 轨迹。"""
    dlat = step_lat(step_m)
    return make_seg([
        (lat0 + i * dlat, lon, e, i * 10.0) for i, e in enumerate(eles)
    ])


def climb_eles(n, start=100.0, per_step=2.0):
    """n 个采样点、每 20 米变化 per_step 米的海拔序列（默认 10% 上坡）。"""
    return [start + i * per_step for i in range(n)]


def test_uniform_uphill_single_section():
    """500 米持续 10% 上坡：全部窗口命中，合并为一个上坡段。"""
    seg = north_seg(climb_eles(26))  # 0..500 m，海拔 100→150
    res = steep_sections([seg])
    assert res.has_ele
    assert len(res.sections) == 1
    s = res.sections[0]
    assert s.seg_index == 1 and s.direction == "up"
    # 首窗口 [0,100] 标记 [40,60]，末窗口 [400,500] 标记 [440,460]
    assert abs(s.start_m - 40.0) < 1e-6
    assert abs(s.end_m - 460.0) < 1e-6
    assert abs(s.max_abs_grade - 0.10) < 1e-6
    assert abs(res.distance_m - 500.0) < 1e-6


def test_uniform_downhill_single_section():
    """500 米持续 10% 下坡：合并为一个下坡段。"""
    seg = north_seg(climb_eles(26, start=150.0, per_step=-2.0))
    res = steep_sections([seg])
    assert len(res.sections) == 1
    s = res.sections[0]
    assert s.direction == "down"
    assert abs(s.start_m - 40.0) < 1e-6 and abs(s.end_m - 460.0) < 1e-6
    assert abs(s.max_abs_grade - 0.10) < 1e-6


def test_gentle_slope_no_sections():
    """7.5% 坡度未达 8% 阈值：有海拔但无陡坡段。"""
    seg = north_seg(climb_eles(51, per_step=1.5))  # 1000 m，7.5%
    res = steep_sections([seg])
    assert res.has_ele
    assert res.sections == []


def test_threshold_boundary_marked():
    """恰为 ±8% 的窗口被标记；±7.9% 不标记。"""
    ds = [0.0, 100.0, 200.0, 300.0]
    up = _run_marks(ds, [100.0, 108.0, 116.0, 124.0])
    assert len(up) == 11 and all(m[2] == "up" for m in up)
    down = _run_marks(ds, [124.0, 116.0, 108.0, 100.0])
    assert len(down) == 11 and all(m[2] == "down" for m in down)
    assert _run_marks(ds, [100.0, 107.9, 115.8, 123.7]) == []
    assert _run_marks(ds, [123.7, 115.8, 107.9, 100.0]) == []


def test_window_step_and_mark_bounds():
    """窗口每次前移 20 米；命中窗口标记其中心前后各 10 米的 20 米区间。"""
    ds = [float(20 * i) for i in range(16)]          # 0..300 m
    es = [100.0 + 0.1 * d for d in ds]               # 10% 上坡
    marks = _run_marks(ds, es)
    # 完整 100 米窗口：起点 0,20,...,200，共 11 个
    assert len(marks) == 11
    assert marks[0][:3] == (40.0, 60.0, "up")
    assert marks[-1][:3] == (240.0, 260.0, "up")
    for m in marks:
        assert abs(m[3] - 0.1) < 1e-9


def test_merge_only_same_direction():
    """只有相接且同向的标记才合并；异向或有间隔不合并。"""
    marks = [
        (40.0, 60.0, "up", 0.10),
        (60.0, 80.0, "up", 0.12),      # 相接同向 → 合并，最大绝对坡度 0.12
        (80.0, 100.0, "down", -0.09),  # 相接但异向 → 不合并
        (120.0, 140.0, "down", -0.11), # 有 20 米间隔 → 不合并
    ]
    merged = _merge_marks(marks)
    assert merged == [
        (40.0, 80.0, "up", 0.12),
        (80.0, 100.0, "down", 0.09),
        (120.0, 140.0, "down", 0.11),
    ]


def test_short_steep_filtered_by_min_length():
    """合并后标记总长 < 100 米不列出；恰为 100 米列出。"""
    # 160 米 10% 上坡：窗口起点 0..60 共 4 个 → 标记 [40,120]，总长 80 米 → 不列出
    seg = north_seg(climb_eles(9))
    res = steep_sections([seg])
    assert res.has_ele and res.sections == []
    # 180 米 10% 上坡：窗口起点 0..80 共 5 个 → 标记 [40,140]，总长恰 100 米 → 列出
    seg = north_seg(climb_eles(10))
    res = steep_sections([seg])
    assert len(res.sections) == 1
    s = res.sections[0]
    assert abs(s.start_m - 40.0) < 1e-6 and abs(s.end_m - 140.0) < 1e-6


def test_run_shorter_than_window():
    """区间不足 100 米：没有完整窗口，不产生任何标记。"""
    seg = north_seg(climb_eles(5))  # 0..80 m，10%
    res = steep_sections([seg])
    assert res.has_ele and res.sections == []


def test_missing_ele_gap_not_merged():
    """缺失海拔缺口两侧的陡坡段不合并，段里程为全程里程。"""
    eles = climb_eles(16) + [None, None] + climb_eles(16)
    seg = north_seg(eles)  # 0..660 m，缺口在 300..360 m
    res = steep_sections([seg])
    assert len(res.sections) == 2
    a, b = res.sections
    assert a.direction == "up" and b.direction == "up"
    assert abs(a.start_m - 40.0) < 1e-6 and abs(a.end_m - 260.0) < 1e-6
    # 第二个区间起点在 360 m 处（含缺口里程），段为 [400,620]
    assert abs(b.start_m - 400.0) < 1e-6 and abs(b.end_m - 620.0) < 1e-6
    assert a.end_m < b.start_m  # 缺口处未合并


def test_trkseg_boundary_not_merged():
    """原始 trkseg 边界两侧的陡坡段不合并。"""
    seg1 = north_seg(climb_eles(16), lat0=31.0, lon=121.0)
    seg2 = north_seg(climb_eles(16), lat0=32.0, lon=122.0)
    res = steep_sections([seg1, seg2])
    assert len(res.sections) == 2
    a, b = res.sections
    assert a.seg_index == 1 and b.seg_index == 2
    assert abs(a.start_m - 40.0) < 1e-6 and abs(a.end_m - 260.0) < 1e-6
    # 第二个 trkseg 的全程里程接着第一段累计：段为 [340,560]
    assert abs(b.start_m - 340.0) < 1e-6 and abs(b.end_m - 560.0) < 1e-6


def test_direction_change_two_sections():
    """先 10% 上坡 300 米再 10% 下坡 300 米：两个方向相反的段。"""
    eles = [100.0 + 2.0 * i for i in range(16)]  # 0..300 m：100→130
    eles += [128.0 - 2.0 * i for i in range(15)]  # 320..600 m：128→100
    seg = north_seg(eles)
    res = steep_sections([seg])
    assert len(res.sections) == 2
    up, down = res.sections
    assert up.direction == "up" and down.direction == "down"
    assert abs(up.start_m - 40.0) < 1e-6 and abs(up.end_m - 260.0) < 1e-6
    assert abs(down.start_m - 340.0) < 1e-6 and abs(down.end_m - 560.0) < 1e-6
    assert abs(up.max_abs_grade - 0.10) < 1e-6
    assert abs(down.max_abs_grade - 0.10) < 1e-6


def test_max_abs_grade_reported():
    """段内各窗口坡度的最大绝对值：10% 与 15% 混合时为 15%。"""
    eles = [100.0 + 2.0 * i for i in range(16)]       # 0..300 m：10%
    eles += [130.0 + 3.0 * i for i in range(1, 16)]   # 320..600 m：15%
    seg = north_seg(eles)
    res = steep_sections([seg])
    assert len(res.sections) == 1
    s = res.sections[0]
    assert s.direction == "up"
    assert abs(s.start_m - 40.0) < 1e-6 and abs(s.end_m - 560.0) < 1e-6
    assert abs(s.max_abs_grade - 0.15) < 1e-6


def test_no_ele_at_all():
    """全无有效海拔：has_ele=False，无陡坡段（里程仍统计）。"""
    seg = north_seg([None] * 10)
    res = steep_sections([seg])
    assert not res.has_ele
    assert res.sections == []
    assert abs(res.distance_m - 180.0) < 1e-6


def test_single_point_run_no_sections():
    """只有一个有效海拔点的区间：无完整窗口，无陡坡段。"""
    seg = north_seg([None, 88.0, None], step_m=50.0)
    res = steep_sections([seg])
    assert res.has_ele and res.sections == []


def test_isolated_spike_suppressed():
    """孤立尖峰被三点中位数平滑抑制，不产生虚假陡坡。"""
    eles = [100.0] * 5 + [130.0] + [100.0] * 5  # 0..200 m，中间一个尖峰
    seg = north_seg(eles)
    res = steep_sections([seg])
    assert res.has_ele and res.sections == []


def test_section_mileage_includes_leading_gap():
    """段起止里程为全程里程：前置无海拔部分计入偏移。"""
    eles = [None] * 5 + climb_eles(16)  # 0..80 m 无海拔，100..400 m 为 10% 上坡
    seg = north_seg(eles)
    res = steep_sections([seg])
    assert len(res.sections) == 1
    s = res.sections[0]
    assert abs(s.start_m - 140.0) < 1e-6 and abs(s.end_m - 360.0) < 1e-6


def _run_all():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
