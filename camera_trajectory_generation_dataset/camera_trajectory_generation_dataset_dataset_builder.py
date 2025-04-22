from typing import Iterator, Tuple, Any

import glob
import numpy as np
# import tensorflow as tf
import tensorflow_datasets as tfds
# import tensorflow_hub as hub
# import cv2
import os
import json
import pickle
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# def get_language_embedding(text: str):
#     if not hasattr(get_language_embedding, "_embed_model"):
#         get_language_embedding._embed_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
#     return get_language_embedding._embed_model([text])[0].numpy().astype(np.float16)

def get_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

class CameraTrajectoryGenerationDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(512, 512, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float16,
                            doc='Robot state, consists of [4x quaternion(xyzw), '
                            '3x translation(xyz)].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float16,
                        doc='Robot action, consists of [4x quaternion(xyzw), '
                            '3x translation(xyz)].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float16,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float16,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    # 'language_embedding': tfds.features.Tensor(
                    #     shape=(512,),
                    #     dtype=np.float16,
                    #     doc='Kona language embedding. '
                    #         'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    # ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path="/home/k-sakamoto/0142_camera_trajectory_generation/Data/filtered_video_clips_nymeria_v2_trans_07_rot_05/metas"),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            # data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case
            episode_id = episode_path.split("/")[-1].split(".")[0]

            image_pkl_path = os.path.join("/home/k-sakamoto/0142_camera_trajectory_generation/Data/filtered_video_clips_nymeria_v2_trans_07_rot_05/images_pkl",f"{episode_id}.pkl")
            data = parse_json(episode_path, image_pkl_path)
            

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                # compute Kona language embedding
                # language_embedding = self._embed([step['language_instruction']])[0].numpy().astype(np.float16)
                # language_embedding = get_language_embedding(step['language_instruction'])

                episode.append({
                    'observation': {
                        'image': step['image'],
                        'state': step['state'],
                    },
                    'action': step['action'],
                    'discount': 1.0,
                    'reward': float(i == (len(data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': step['language_instruction'],
                    # 'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        # path = "/home/k-sakamoto/0142_camera_trajectory_generation/Data/filtered_video_clips_nymeria_v2_trans_07_rot_05/metas"
        episode_paths = get_files(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

def parse_json(json_data_path, image_pkl_path):
    """
    input:
        json_data_path: dict. The json data to parse.
        e.g. 
        {
            "poses": [
                [-0.0012,0.00205,-0.0116,0.99992,-0.0018,-0.0135,0.00458],
                [-0.0012,0.00205,-0.0116,0.99992,-0.0018,-0.0135,0.00458],...
            ],
            ...
        }
    output:
        data: list of dicts. The parsed data.
        e.g.
        [
            {
                "image": np.ndarray. The image data.
                "action": np.ndarray. The action data. next camera pose
                "state": np.ndarray. The action data. current camera pose
                "language_instruction": str. The language instruction.
            },...
        ]
    """
    with open(json_data_path, "r") as f:
        json_data = json.load(f)
    
    # image_dir = json_data["image_dir"] + "_pkl"
    language_instruction = json_data["caption"]
    # language_instruction = "dummy instruction"
    data = []

    # pklファイルの読み込み
    with open(image_pkl_path, 'rb') as f:
        image_list = pickle.load(f)  # List of NumPy arrays
    
    for frame_id, current_pose in enumerate(json_data["poses"]):
        # image_path = os.path.join(image_dir, f"{str(frame_id).zfill(8)}.png")
        # image_bgr = cv2.imread(image_path)
        # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = image_list[frame_id]

        if frame_id < len(json_data["poses"]) - 1:
            next_pose = json_data["poses"][frame_id + 1]
        else:
            next_pose = current_pose
        data.append({
            "image": image_rgb,
            # "image": np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
            "action": np.asarray(next_pose, dtype=np.float16),
            "state": np.asarray(current_pose, dtype=np.float16),
            "language_instruction": language_instruction,
        })
    return data

