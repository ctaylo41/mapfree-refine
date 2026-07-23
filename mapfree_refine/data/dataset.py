import torch.utils.data as data
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RigidTransform as Tf


class MapFreeDataset(data.Dataset):
    def __init__(self, scene_root):
        super().__init__()
        self.scene_root = Path(scene_root)

        self.poses = self.read_poses(scene_root)
        self.K = self.read_K(scene_root)
        self.image_pairs = self.read_image_pairs(scene_root)

    @staticmethod
    def read_K(scene_root):
        Ks = {}
        with open(scene_root + '/intrinsics.txt','r') as f:
            for line in f.readlines():
                if line.startswith('#'):
                    continue

                line = line.strip().split()
                img_name = line[0]
                fx, fy, cx, cy, W, H = map(float, line[1:])

                K = np.array([[fx, 0, cx], [0, fy, cy], [0,0,1]], dtype=np.float32)
                Ks[img_name] = K
        return Ks
    
    @staticmethod
    def read_poses(scene_root: Path, filename: str = 'poses.txt'):
        poses = {}
        try:
            with open(f"{scene_root}/{filename}", 'r') as f:
                for line in f.readlines():
                    if line.startswith('#'):
                        continue
                    line = line.strip().split()
                    img_name = line[0]
                    qt = np.array(list(map(float, line[1:])))
                    rotation = R.from_quat(qt[:4], scalar_first=True)
                    translation = qt[4:]
                    transform = Tf.from_components(translation, rotation)
                    poses[img_name] = transform
        except FileNotFoundError:
            poses = None
            pass

        return poses
    
    @staticmethod
    def read_image_pairs(scene_root):
        pairs = []
        seq_0 = sorted(Path(scene_root + '/seq0').iterdir())[0]
        
        seq_1_folder = Path(scene_root  + '/seq1')

        for seq_1 in sorted(seq_1_folder.iterdir()):
            if seq_1.is_file():
                pair = (seq_0, seq_1)
                pairs.append(pair)

        return pairs

    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, index):
        seq0_path, seq1_path = self.image_pairs[index]
        seq0_k, seq1_k = self.K["seq0/" + seq0_path.name], self.K["seq1/" + seq1_path.name]

        seq1_rel = None
        # poses.txt stores world->camera (per official loader)
        if self.poses is not None:
            seq0_pose = self.poses["seq0/" + seq0_path.name]
            seq1_pose = self.poses["seq1/" + seq1_path.name]
            seq1_rel = seq1_pose * seq0_pose.inv()
        else:
            seq0_pose, seq1_pose = None, None

        data = {
            'image0' : seq0_path,
            'image1' : seq1_path,
            'image0_k' :seq0_k,
            'image1_k' : seq1_k,
            'image0_pose' : seq0_pose,
            'image1_pose' : seq1_pose,
            'T_query_ref' : seq1_rel,
            'scene_id': self.scene_root.stem,

            'scene_root': str(self.scene_root),
            'pair_names': (seq0_path.name, seq1_path.name),


        }
        return data