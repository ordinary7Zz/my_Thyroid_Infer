"""抑制 AutoGluon / joblib 加载旧版 sklearn 模型时的版本不一致刷屏。"""

from __future__ import annotations

import os
import warnings


def configure_sklearn_pickle_warnings() -> None:
    """
    默认忽略 sklearn InconsistentVersionWarning（1.6.1 模型 + 1.7.x 环境）。

    若需恢复：export THYROID_WARN_SKLEARN_VERSION=1
    """
    if os.environ.get("THYROID_WARN_SKLEARN_VERSION", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    try:
        from sklearn.exceptions import InconsistentVersionWarning

        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    except ImportError:
        pass
    # 兜底：部分环境 PYTHONWARNINGS 无法写 sklearn.exceptions.*（会报 Invalid -W）
    warnings.filterwarnings(
        "ignore",
        message=r"Trying to unpickle estimator .* from version .* when using version",
    )
