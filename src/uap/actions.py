# src/uap/actions.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class ActionSpec:
    """A canonical action in the tool/action space."""
    name: str               # canonical action name (Chinese)
    desc: str               # human-readable description
    examples: Sequence[str] # example argument strings (inside parentheses)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    intro: str
    actions: Sequence[ActionSpec]


def _poi() -> TaskSpec:
    actions = [
        ActionSpec(
            name="搜索特定类别或特色的地点",
            desc="在指定地点或当前位置周围寻找特定类别/特色的地点。",
            examples=[
                "临近武汉博物馆, 类别'4s店', 特色'', 数量'n'",
                "广东, 类别'旅游景点', 特色'冷门', 数量'n'",
                "临近当前位置, 类别'停车场', 特色'免费', 数量'n'",
            ],
        ),
        ActionSpec(
            name="查询地点相关信息",
            desc="获取特定地点的详细信息。",
            examples=[
                "诸城市博物馆, '周三人流量'",
                "好客酒文化博览园, '营业时间'",
            ],
        ),
    ]
    return TaskSpec(
        name="POI",
        intro="我是一个兴趣点(POI)搜索助手。",
        actions=actions,
    )


def _navi() -> TaskSpec:
    actions = [
        ActionSpec(
            name="查询导航路线",
            desc="查询从起点到终点的导航路线，可包含途经点。",
            examples=[
                "当前位置, 大理, 交通方式'驾车'",
                "公司, 大悦城, 交通方式'驾车', 途经点['家','五大道']",
            ],
        ),
        ActionSpec(
            name="查询导航路线相关信息",
            desc="查询导航路线细节（如经过哪些省、红绿灯个数等）。",
            examples=[
                "郑州, 杭州, '经过哪些省'",
                "家, 汽车总站, '红绿灯个数'",
            ],
        ),
        ActionSpec(
            name="查询整体时间",
            desc="查询总耗时。",
            examples=[
                "家, 公司, 交通方式'地铁'",
                "公司, 机场, 交通方式'出租车'",
            ],
        ),
        ActionSpec(
            name="返回费用估算",
            desc="查询费用估算。",
            examples=[
                "上海, 杭州, 交通方式'高铁'",
                "公司, 家, 交通方式'打车'",
            ],
        ),
    ]
    return TaskSpec(name="Navi", intro="我是一个导航助手。", actions=actions)


def _transit() -> TaskSpec:
    actions = [
        ActionSpec(
            name="查询公交信息",
            desc="查询公交/地铁线路的基本信息。",
            examples=["355路", "4号线"],
        ),
        ActionSpec(
            name="搜索特定线路",
            desc="查询起点到终点的线路（公交/地铁/高铁/火车等）。",
            examples=[
                "百度大厦, 五道口, 地铁",
                "当前位置, 北医三院, 公交",
                "当前位置, 杭州, 高铁",
            ],
        ),
        ActionSpec(
            name="询问到达时间",
            desc="查询线路到达某站点/当前位置的时间。",
            examples=[
                "303路, 当前位置",
                "4号线, 灵境胡同站",
            ],
        ),
        ActionSpec(
            name="返回下一站信息",
            desc="查询线路下一站。",
            examples=["13号线, 西二旗站"],
        ),
        ActionSpec(
            name="返回实时交通信息",
            desc="获取实时交通信息（若你的工具支持）。",
            examples=["303路"],
        ),
        ActionSpec(
            name="返回费用估算",
            desc="查询费用估算。",
            examples=["家, 公司, 交通方式'公交'"],
        ),
    ]
    return TaskSpec(name="Transit", intro="我是一个公共交通查询助手。", actions=actions)


def _taxi() -> TaskSpec:
    actions = [
        ActionSpec(
            name="选择车型或价格偏好",
            desc="指定车型/价格偏好（可作为独立动作或融入查询）。",
            examples=["车型偏好'出租车', 价格偏好'无'"],
        ),
        ActionSpec(
            name="查询打车方案",
            desc="查询打车方案。",
            examples=[
                "百度科技园, 回龙观, 车型偏好'出租车', 价格偏好'无'",
                "当前位置, 明十三陵, 车型偏好'顺风车', 价格偏好'10到20块'",
            ],
        ),
        ActionSpec(
            name="返回排队人数",
            desc="查询排队人数。",
            examples=[""],
        ),
        ActionSpec(
            name="返回到达时间",
            desc="查询到达时间（若你的工具支持独立动作）。",
            examples=[""],
        ),
        ActionSpec(
            name="返回费用估算",
            desc="查询费用估算。",
            examples=["机场, 家, 交通方式'打车'"],
        ),
    ]
    return TaskSpec(name="Taxi", intro="我是一个叫车助手。", actions=actions)


TASK_SPECS: Dict[str, TaskSpec] = {
    "POI": _poi(),
    "Navi": _navi(),
    "Transit": _transit(),
    "Taxi": _taxi(),
}


def allowed_action_names(task: str) -> List[str]:
    """Single source of truth for affordance filtering."""
    return [a.name for a in TASK_SPECS[task].actions]
