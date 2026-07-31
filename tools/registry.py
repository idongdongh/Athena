"""最小工具注册表 + 目录发现。

对齐 hermes tools/registry.py 的核心思想：工具列表不靠硬编码，而是由各
工具文件在模块顶层调用 ``registry.register(...)`` 自注册，``discover()``
扫描 tools/ 目录把它们收集起来。

关键安全点：``discover()`` 先用 AST 静态判断一个模块顶层是否有
``registry.register(...)`` 调用，只有含注册的模块才 import。这样：
- 没有自注册的辅助模块不会被误 import；
- 含裸奔测试段（未用 ``if __name__ == "__main__"`` 保护）的工具文件，
  在它自己加上 register 之前不会被导入执行，避免误发真实 API 请求。
"""

import ast
import importlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass

# slots=True，装饰器自动在类中声明__slots__
@dataclass(slots=True)
class ToolEntry:
    """工具（条目）元数据。"""
    name: str
    schema: dict
    handler: Callable
    toolset: str

class ToolRegistry:
    """单例注册表，收集工具 schema + handler。"""

    def __init__(self):
        self._tools: Dict[str, ToolEntry] = {}

    # 将指定工具名称和元数据关联起来
    def register(self, name: str, schema: dict, handler: Callable, toolset: str = "default") -> None:
        self._tools[name] = ToolEntry(name, schema, handler, toolset)

    # 获取指定工具的元数据
    def get_entry(self, name: str) -> Optional[ToolEntry]:
        return self._tools.get(name)
    
    def definitions(self) -> List[Any]:
        """返回传给模型 API 的 tools schema 列表（Anthropic 原生格式）。

        Anthropic SDK 的 ``messages.create(tools=...)`` 要求 ``Iterable[ToolUnionParam]``
        （一个窄的 TypedDict 联合）；工具 schema 本质是 dict 子类型，但 Pyright 协变检查
        下 ``List[dict]`` 无法静态证明满足 ``Iterable[ToolUnionParam]``。这里把元素声明为
        ``Any``，让类型检查通过，运行时行为不变（仍是 list of dict）。
        """
        return [e.schema for e in self._tools.values()]


registry = ToolRegistry()


def _module_registers_tools(path: Path) -> bool:
    """静态判断模块顶层(AST的顶层）是否有 ``registry.register(...)`` 调用。

    同时检查包裹在顶层 ``if`` 块内的注册（如
    ``if registry is not None: registry.register(...)``），以便工具文件能在
    registry 不可用时（例如直接 ``python tools/xxx_tool.py`` 跑测试段）安全跳过注册。
    """
    try:
        # 读文件内容，解析成 AST 树，在树上遍历检查有没有 registry.register(...) 调用
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False

    def _is_register_call(stmt) -> bool:
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and stmt.value.func.attr == "register"
            and isinstance(stmt.value.func.value, ast.Name)
            and stmt.value.func.value.id == "registry"
        )

    for stmt in tree.body:
        if _is_register_call(stmt):
            return True
        if isinstance(stmt, ast.If):  # 顶层 if 块内也可能有注册
            if any(_is_register_call(s) for s in stmt.body):
                return True
    return False


def discover(tools_dir: Optional[Path] = None) -> List[str]:
    """工具模块的动态发现与加载器：扫描 tools/ 下所有 .py，只 import AST 顶层含 register 调用的模块。

    返回被成功加载（导入）的模块名列表。
    """
    # 最外层 Path 防止传 str 类型的路径，虽然声明了 Path 对象，但是解释器不会报错还是能执行
    d = Path(tools_dir or Path(__file__).resolve().parent)
    # 存储有注册语句的工具名，比如 web_search_tool
    loaded: List[str] = []
    # p 是工具脚本路径
    for p in sorted(d.glob("*.py")):
        # 结构性身份冲突：这两个即便通过 AST 闸门也不该被 import
        #   __init__: 包标识，import 无意义
        #   registry : 本模块自身，import 会触发循环依赖
        # 其他辅助模块（如 _path / _binary）由 AST 闸门过滤，无需进黑名单
        if p.stem in {"__init__", "registry"}:
            continue
        if not _module_registers_tools(p):
            continue
        # mod = tools.xx_tool
        mod = f"{d.name}.{p.stem}"
        try:
            # 等价于 import tools.xx_tool
            importlib.import_module(mod)
            loaded.append(p.stem)
        except Exception as e:  # 单个工具模块损坏不应拖垮整个 agent
            print(f"[tools] 跳过 {p.stem}: {e}")
    return loaded

