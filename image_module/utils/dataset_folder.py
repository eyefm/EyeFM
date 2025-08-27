import os
import os.path
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class MultiTaskDatasetFolder(VisionDataset):
    def __init__(
            self,
            root: str,
            tasks: List[str],
            csv_data,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            prefixes: Optional[Dict[str, str]] = None,
            max_images: Optional[int] = None
    ) -> None:
        super(MultiTaskDatasetFolder, self).__init__(root, transform=transform,
                                                     target_transform=target_transform)
        self.tasks = tasks

        prefixes = {} if prefixes is None else prefixes
        prefixes.update({task: '' for task in tasks if task not in prefixes})

        samples = dict()
        for task in self.tasks:
            samples[task] = csv_data.apply(lambda row: os.path.join(self.root, row[task]), axis=1).tolist()

        for task, task_samples in samples.items():
            if len(task_samples) == 0:
                msg = "Found 0 logs in subfolders of: {}\n".format(os.path.join(self.root, task))
                if extensions is not None:
                    msg += "Supported extensions are: {}".format(",".join(extensions))
                raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions
        self.samples = samples

        # Select random subset of dataset if so specified
        if isinstance(max_images, int):
            total_samples = len(list(self.samples.values())[0])
            np.random.seed(0)
            permutation = np.random.permutation(total_samples)
            for task in samples:
                self.samples[task] = [self.samples[task][i] for i in permutation][:max_images]

        self.cache = {}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if index in self.cache:
            sample_dict = deepcopy(self.cache[index])
        else:
            sample_dict = {}
            for task in self.tasks:
                path = self.samples[task][index]
                sample = pil_loader(path, convert_rgb=True)
                sample_dict[task] = sample

        if self.transform is not None:
            sample_dict = self.transform(sample_dict)

        return sample_dict

    def __len__(self) -> int:
        return len(list(self.samples.values())[0])


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.jpx')


def pil_loader(path: str, convert_rgb=True) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # with open(path, 'rb') as f:
    #     img = Image.open(f)
    img = Image.open(path)
    return img.convert('RGB') if convert_rgb else img


class MultiTaskImageFolder(MultiTaskDatasetFolder):
    def __init__(
            self,
            root: str,
            tasks: List[str],
            csv_data,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            prefixes: Optional[Dict[str,str]] = None,
            max_images: Optional[int] = None
    ):
        super(MultiTaskImageFolder, self).__init__(root, tasks, csv_data, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          prefixes=prefixes,
                                          max_images=max_images)
        self.imgs = self.samples
