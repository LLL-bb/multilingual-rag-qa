"""入口脚本共用的启动逻辑。

放在仓库根目录,是因为 ``scripts/`` 和 ``eval/`` 两处的脚本都要用 ——
任何一边放都不合适。

做两件事:

1. 把 ``src/`` 加进 ``sys.path``,这样不安装成包也能直接跑脚本;
2. 把标准输出切成 UTF-8。Windows 控制台默认是 cp1252 或 gbk,
   而本仓库面向用户的输出全是中文,不切就会直接抛 UnicodeEncodeError。
   这不是锦上添花,是 Windows 上能不能跑的问题。

脚本里的用法::

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _bootstrap import bootstrap
    PROJECT_ROOT = bootstrap()
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def bootstrap() -> Path:
    src = str(PROJECT_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                # 输出被重定向到不支持重配置的对象时忽略即可
                pass

    return PROJECT_ROOT
