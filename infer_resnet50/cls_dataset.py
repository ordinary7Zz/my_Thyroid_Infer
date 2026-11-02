import os
import json
from typing import Any, Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ThyroidClsDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_json: str,
        img_size: int = 224,
        mode: str = "train",
        task_key: str = "malignancy",
    ) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.task_key = task_key
        samples, skipped_label_minus1, missing_image_files = self._load_samples(
            image_dir=image_dir,
            label_json_path=label_json,
            task_key=task_key,
        )
        self.image_paths = [sample[0] for sample in samples]
        self.labels = [sample[1] for sample in samples]
        self.filenames = [sample[2] for sample in samples]
        self.skipped_label_minus1 = skipped_label_minus1
        self.missing_image_files = missing_image_files

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No labeled images found in '{image_dir}' for task '{task_key}'."
            )

        if mode == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    @staticmethod
    def _load_samples(
        image_dir: str,
        label_json_path: str,
        task_key: str,
    ) -> Tuple[List[Tuple[str, int, str]], int, int]:
        with open(label_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("label_json must be a list of dicts.")

        samples: List[Tuple[str, int, str]] = []
        skipped_label_minus1 = 0
        missing_image_files = 0
        found_task_key = False

        for item in data:
            if not isinstance(item, dict):
                continue

            filename = item.get("filename")
            if not filename:
                continue

            if task_key in item:
                found_task_key = True

            label = int(item.get(task_key, -1))
            if label == -1:
                skipped_label_minus1 += 1
                continue

            relative_path = str(filename)
            image_path = os.path.join(image_dir, os.path.normpath(relative_path))
            if not os.path.isfile(image_path):
                missing_image_files += 1
                continue

            samples.append((image_path, label, relative_path))

        if not found_task_key:
            raise ValueError(f"Task key '{task_key}' not found in label JSON.")

        return samples, skipped_label_minus1, missing_image_files

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.image_paths[idx]
        label = self.labels[idx]
        filename = self.filenames[idx]

        with Image.open(path) as img:
            img = img.convert("RGB")

        img_t = self.transform(img)
        label_t = torch.tensor(float(label), dtype=torch.float32)

        return {
            "image": img_t,
            "label": label_t,
            "filename": filename,
        }
