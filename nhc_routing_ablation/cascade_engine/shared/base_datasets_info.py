"""
数据集元信息 + 设备匹配逻辑（单一来源）。

合并自 Segmentation_Agent 和 Classification_Agent 的 base_datasets_info 配置段。
两个 config 内容基本一致，此处取并集作为唯一真相源。
"""

from __future__ import annotations

from typing import Optional


BASE_DATASETS_INFO: dict = {
    "TN3K": {
        "main_devices": [
            "GE Logiq E9",
            "ARIETTA 850",
            "RESONA 70B",
            "Hitachi Aloka Arietta V70",
        ],
        "centers": ["Zhujiang Hospital, Southern Medical University"],
        "years": [2016, 2020],
    },
    "ThyroidXL": {
        "main_devices": ["Hitachi Aloka Arietta V70"],
        "centers": [
            "National Hospital of Endocrinology, Hanoi, Vietnam"
        ],
        "years": [2023, 2025],
    },
    "TN5K": {
        "main_devices": ["GE Logiq E9", "GE S7"],
        "probe_frequency_mhz": [5, 12],
        "imaging_mode": "B-mode Ultrasound (2D)",
        "centers": [
            "National Cancer Center/Cancer Hospital, "
            "Chinese Academy of Medical Sciences and Peking Union Medical College"
        ],
    },
    "DDTI": {
        "main_devices": ["TOSHIBA Nemio 30, TOSHIBA Nemio MX"],
        "probe_frequency": "uniformly set to 12 MHz",
        "imaging_mode": "B-mode Ultrasound",
        "centers": ["IDIME (Instituto de Diagnóstico Médico)"],
    },
    "PKTN": {
        "main_devices": [],
        "centers": ["Department of Ultrasound, Peking University First Hospital"],
        "years": [2019, 2022],
    },
    "CineClip": {
        "main_devices": [],
        "centers": ["Stanford University Medical Center"],
    },
}


def infer_device_match(
    input_devices: Optional[list[str]],
    model_base_datasets: list[str],
) -> dict:
    """
    推断输入数据的设备与模型训练数据集的设备是否匹配。

    逻辑：遍历模型训练用到的每个 base_dataset，从 BASE_DATASETS_INFO 取其
    main_devices，检查 input_devices 是否与之有交集。

    Args:
        input_devices: 输入数据的设备列表，如 ["GE Logiq E9", "GE S7"]。
                       为 None 或空列表时无法匹配。
        model_base_datasets: 模型训练用到的原始数据集名称列表，
                             如 ["TN3K", "ThyroidXL"]。

    Returns:
        {
            "matched": bool,
            "matched_datasets": list[str],  # 有设备交集的数据集
            "reason": str,
        }
    """
    if not input_devices:
        return {
            "matched": False,
            "matched_datasets": [],
            "reason": "输入设备信息未知，无法匹配",
        }

    input_set = {d.strip().lower() for d in input_devices if d}
    matched_datasets = []

    for ds_name in model_base_datasets:
        ds_info = BASE_DATASETS_INFO.get(ds_name)
        if not ds_info:
            continue
        ds_devices = ds_info.get("main_devices", [])
        ds_set = {d.strip().lower() for d in ds_devices if d}
        if input_set & ds_set:
            matched_datasets.append(ds_name)

    if matched_datasets:
        return {
            "matched": True,
            "matched_datasets": matched_datasets,
            "reason": f"输入设备与训练数据集 {matched_datasets} 的设备匹配",
        }

    return {
        "matched": False,
        "matched_datasets": [],
        "reason": "输入设备与所有训练数据集的设备均不匹配",
    }
