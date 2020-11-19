from typing import List
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class CholecDataset(Dataset):

    def __init__(
        self,
        file_paths: List[str],
        file_labels: np.ndarray,
        transform=None,
        loader=pil_loader,
    ) -> None:

        self.file_paths = list(file_paths)
        self.file_labels_phase = list(file_labels)
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index: int):
        img_names = self.file_paths[index]
        labels_phase = self.file_labels_phase[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_phase


class ParseData(object):

    def __init__(self, root_dir):
        self.video_imgs_num = []

        root_dir = Path(root_dir)
        img_base_path = root_dir / "data_resize"
        phase_annotations = root_dir / "phase_annotations"
        #print(img_base_path, annot_tool_path, annot_timephase_path, out_path)

        class_labels = [
            "Preparation",
            "CalotTriangleDissection",
            "ClippingCutting",
            "GallbladderDissection",
            "GallbladderPackaging",
            "CleaningCoagulation",
            "GallbladderRetraction",
        ]

        cholec_df = pd.DataFrame(columns=[
            "image_path", "phase", "video_idx"
        ])
        for i in range(1, 8):
            vid_df = pd.DataFrame()
            img_path_for_vid = img_base_path / f"video{i:02d}"
            img_list = sorted(img_path_for_vid.glob('*.jpg'))
            # img_list = [str(i.relative_to(img_base_path)) for i in img_list]
            vid_df["image_path"] = img_list
            vid_df["video_idx"] = [i] * len(img_list)
            # add image class
            phase_path = phase_annotations / f"video{i:02d}-phase.txt"
            phases = pd.read_csv(phase_path, sep='\t')
            for j, p in enumerate(class_labels):
                phases["Phase"] = phases.Phase.replace({p: j})
            phase_list = []
            i = 0
            for Phase in phases["Phase"]:
                if i % 25 == 0:
                    phase_list.append(Phase)
                i += 1
            vid_df["phase"] = phase_list
            # vid_df = pd.concat([vid_df, phases], axis=1)
            # print(
            #     f"len(img_list): {len(img_list)} - vid_df.shape[0]:{vid_df.shape[0]}  - len(phases):{len(phase_list)}"
            # )
            self.video_imgs_num.append(len(img_list))
            # vid_df = vid_df.rename(columns={
            #     "Phase": "phase",
            # })
            cholec_df = cholec_df.append(vid_df, ignore_index=True, sort=False)

        print("DONE")
        print(cholec_df.shape)
        print(cholec_df.columns)

        self.datas = cholec_df


if __name__ == '__main__':
    data = ParseData('/home/wzy/dataset/')
    print(type(data.datas))
    image_path = data.datas['phase']
    print(image_path[0])
    print(type(image_path))
    for i in image_path:
        print(i)
        break
