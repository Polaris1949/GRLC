import math
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from urllib import request

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.data import extract_tar
from tqdm import tqdm

from utils import safe_list_index
from utils import side_to_directed_lineseg

try:
    from av2.geometry.interpolate import compute_midpoint_line
    from av2.map.map_api import ArgoverseStaticMap
    from av2.map.map_primitives import Polyline
    from av2.utils.io import read_json_file
except ImportError:
    compute_midpoint_line = object
    ArgoverseStaticMap = object
    Polyline = object
    read_json_file = object


class ArgoverseV2Dataset(Dataset):
    """Dataset class for Argoverse 2 Motion Forecasting Dataset.

    See https://www.argoverse.org/av2.html for more information about the dataset.

    Args:
        root (string): the root folder of the dataset. If you've downloaded the raw .tar file, placing it in the root
            folder will skip downloading automatically.
        split (string): specify the split of the dataset: `"train"` | `"val"` | `"test"`.
        raw_dir (string, optional): optionally specify the directory of the raw data. By default, the raw directory is
            path/to/root/split/raw/. If specified, the path of the raw log is path/to/raw_dir/log_id. If all logs
            exist in the raw directory, file downloading/extraction will be skipped. (default: None)
        processed_dir (string, optional): optionally specify the directory of the processed data. By default, the
            processed directory is path/to/root/split/processed/. If specified, the path of the processed .pkl files is
            path/to/processed_dir/*.pkl. If all .pkl files exist in the processed directory, file downloading/extraction
            and data preprocessing will be skipped. (default: None)
        transform (callable, optional): a function/transform that takes in an :obj:`torch_geometric.data.Data` object
            and returns a transformed version. The data object will be transformed before every access. (default: None)
        dim (int, Optional): 2D or 3D data. (default: 3)
        num_historical_steps (int, Optional): the number of historical time steps. (default: 50)
        num_future_steps (int, Optional): the number of future time steps. (default: 60)
        predict_unseen_agents (boolean, Optional): if False, filter out agents that are unseen during the historical
            time steps. (default: False)
        vector_repr (boolean, Optional): if True, a time step t is valid only when both t and t-1 are valid.
            (default: True)
    """

    def __init__(self,
                 root: str,
                 split: str,
                 raw_dir: Optional[str] = None,
                 processed_dir: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 dim: int = 3,
                 num_historical_steps: int = 50,
                 num_future_steps: int = 60,
                 predict_unseen_agents: bool = False,
                 vector_repr: bool = True) -> None:
        root = os.path.expanduser(os.path.normpath(root))
        if not os.path.isdir(root):
            os.makedirs(root)
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'{split} is not a valid split')
        self.split = split

        if raw_dir is None:
            raw_dir = os.path.join(root, split, 'raw')
            self._raw_dir = raw_dir
            if os.path.isdir(self._raw_dir):
                self._raw_file_names = [name for name in os.listdir(self._raw_dir) if
                                        os.path.isdir(os.path.join(self._raw_dir, name))]
            else:
                self._raw_file_names = []
        else:
            raw_dir = os.path.expanduser(os.path.normpath(raw_dir))
            self._raw_dir = raw_dir
            if os.path.isdir(self._raw_dir):
                self._raw_file_names = [name for name in os.listdir(self._raw_dir) if
                                        os.path.isdir(os.path.join(self._raw_dir, name))]
            else:
                self._raw_file_names = []

        if processed_dir is None:
            processed_dir = os.path.join(root, split, 'processed')
            self._processed_dir = processed_dir
            if os.path.isdir(self._processed_dir):
                self._processed_file_names = [name for name in os.listdir(self._processed_dir) if
                                              os.path.isfile(os.path.join(self._processed_dir, name)) and
                                              name.endswith(('pkl', 'pickle'))]
            else:
                self._processed_file_names = []
        else:
            processed_dir = os.path.expanduser(os.path.normpath(processed_dir))
            self._processed_dir = processed_dir
            if os.path.isdir(self._processed_dir):
                self._processed_file_names = [name for name in os.listdir(self._processed_dir) if
                                              os.path.isfile(os.path.join(self._processed_dir, name)) and
                                              name.endswith(('pkl', 'pickle'))]
            else:
                self._processed_file_names = []

        self.dim = dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps
        self.predict_unseen_agents = predict_unseen_agents
        self.vector_repr = vector_repr
        self._url = f'https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/{split}.tar'
        self._num_samples = {
            'train': 199908,
            'val': 24988,
            'test': 24984,
        }[split]
        self._agent_types = ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 'static', 'background',
                             'construction', 'riderless_bicycle', 'unknown']
        self._agent_categories = ['TRACK_FRAGMENT', 'UNSCORED_TRACK', 'SCORED_TRACK', 'FOCAL_TRACK']
        self._polygon_types = ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']
        self._polygon_is_intersections = [True, False, None]
        self._point_types = ['DASH_SOLID_YELLOW', 'DASH_SOLID_WHITE', 'DASHED_WHITE', 'DASHED_YELLOW',
                             'DOUBLE_SOLID_YELLOW', 'DOUBLE_SOLID_WHITE', 'DOUBLE_DASH_YELLOW', 'DOUBLE_DASH_WHITE',
                             'SOLID_YELLOW', 'SOLID_WHITE', 'SOLID_DASH_WHITE', 'SOLID_DASH_YELLOW', 'SOLID_BLUE',
                             'NONE', 'UNKNOWN', 'CROSSWALK', 'CENTERLINE']
        self._point_sides = ['LEFT', 'RIGHT', 'CENTER']
        self._polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']
        super(ArgoverseV2Dataset, self).__init__(root=root, transform=transform, pre_transform=None, pre_filter=None)

    @property
    def raw_dir(self) -> str:
        return self._raw_dir

    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    def download(self) -> None:
        if not os.path.isfile(os.path.join(self.root, f'{self.split}.tar')):
            print(f'Downloading {self._url}', file=sys.stderr)
            request.urlretrieve(self._url, os.path.join(self.root, f'{self.split}.tar'))
        if os.path.isdir(os.path.join(self.root, self.split)):
            shutil.rmtree(os.path.join(self.root, self.split))
        if os.path.isdir(self.raw_dir):
            shutil.rmtree(self.raw_dir)
        os.makedirs(self.raw_dir)
        extract_tar(path=os.path.join(self.root, f'{self.split}.tar'), folder=self.raw_dir, mode='r')
        self._raw_file_names = [name for name in os.listdir(os.path.join(self.raw_dir, self.split)) if
                                os.path.isdir(os.path.join(self.raw_dir, self.split, name))]
        for raw_file_name in self.raw_file_names:
            shutil.move(os.path.join(self.raw_dir, self.split, raw_file_name), self.raw_dir)
        os.rmdir(os.path.join(self.raw_dir, self.split))

    def process(self) -> None:
        for raw_file_name in tqdm(self.raw_file_names):
            df = pd.read_parquet(os.path.join(self.raw_dir, raw_file_name, f'scenario_{raw_file_name}.parquet'))
            map_dir = Path(self.raw_dir) / raw_file_name
            map_path = sorted(map_dir.glob('log_map_archive_*.json'))[0]
            map_data = read_json_file(map_path)
            centerlines = {lane_segment['id']: Polyline.from_json_data(lane_segment['centerline'])
                           for lane_segment in map_data['lane_segments'].values()}
            map_api = ArgoverseStaticMap.from_json(map_path)
            data = dict()
            data['scenario_id'] = self.get_scenario_id(df)
            data['city'] = self.get_city(df)
            data['agent'] = self.get_agent_features(df)
            data.update(self.get_map_features(map_api, centerlines))
            with open(os.path.join(self.processed_dir, f'{raw_file_name}.pkl'), 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def get_scenario_id(df: pd.DataFrame) -> str:
        return df['scenario_id'].values[0]

    @staticmethod
    def get_city(df: pd.DataFrame) -> str:
        return df['city'].values[0]

    def get_agent_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.predict_unseen_agents:  # filter out agents that are unseen during the historical time steps
            historical_df = df[df['timestep'] < self.num_historical_steps]
            agent_ids = list(historical_df['track_id'].unique())
            df = df[df['track_id'].isin(agent_ids)]
        else:
            agent_ids = list(df['track_id'].unique())

        num_agents = len(agent_ids)
        av_idx = agent_ids.index('AV')

        # initialization
        valid_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
        predict_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        agent_id: List[Optional[str]] = [None] * num_agents
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)
        agent_category = torch.zeros(num_agents, dtype=torch.uint8)
        position = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)
        heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        velocity = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)

        for track_id, track_df in df.groupby('track_id'):
            agent_idx = agent_ids.index(track_id)
            agent_steps = track_df['timestep'].values

            valid_mask[agent_idx, agent_steps] = True
            current_valid_mask[agent_idx] = valid_mask[agent_idx, self.num_historical_steps - 1]
            predict_mask[agent_idx, agent_steps] = True
            if self.vector_repr:  # a time step t is valid only when both t and t-1 are valid
                valid_mask[agent_idx, 1: self.num_historical_steps] = (
                        valid_mask[agent_idx, :self.num_historical_steps - 1] &
                        valid_mask[agent_idx, 1: self.num_historical_steps])
                valid_mask[agent_idx, 0] = False
            predict_mask[agent_idx, :self.num_historical_steps] = False
            if not current_valid_mask[agent_idx]:
                predict_mask[agent_idx, self.num_historical_steps:] = False

            agent_id[agent_idx] = track_id
            agent_type[agent_idx] = self._agent_types.index(track_df['object_type'].values[0])
            agent_category[agent_idx] = track_df['object_category'].values[0]
            position[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['position_x'].values,
                                                                              track_df['position_y'].values],
                                                                             axis=-1)).float()
            heading[agent_idx, agent_steps] = torch.from_numpy(track_df['heading'].values).float()
            velocity[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['velocity_x'].values,
                                                                              track_df['velocity_y'].values],
                                                                             axis=-1)).float()

        if self.split == 'test':
            predict_mask[current_valid_mask
                         | (agent_category == 2)
                         | (agent_category == 3), self.num_historical_steps:] = True

        return {
            'num_nodes': num_agents,
            'av_index': av_idx,
            'valid_mask': valid_mask,
            'predict_mask': predict_mask,
            'id': agent_id,
            'type': agent_type,
            'category': agent_category,
            'position': position,
            'heading': heading,
            'velocity': velocity,
        }

    def get_map_features(self,
                         map_api: ArgoverseStaticMap,
                         centerlines: Mapping[str, Polyline]) -> Dict[Union[str, Tuple[str, str, str]], Any]:
        lane_segment_ids = map_api.get_scenario_lane_segment_ids()
        cross_walk_ids = list(map_api.vector_pedestrian_crossings.keys())
        polygon_ids = lane_segment_ids + cross_walk_ids
        num_polygons = len(lane_segment_ids) + len(cross_walk_ids) * 2

        # initialization
        polygon_position = torch.zeros(num_polygons, self.dim, dtype=torch.float)
        polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
        polygon_height = torch.zeros(num_polygons, dtype=torch.float)
        polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
        polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
        point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_type: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_side: List[Optional[torch.Tensor]] = [None] * num_polygons

        for lane_segment in map_api.get_scenario_lane_segments():
            lane_segment_idx = polygon_ids.index(lane_segment.id)
            centerline = torch.from_numpy(centerlines[lane_segment.id].xyz).float()
            polygon_position[lane_segment_idx] = centerline[0, :self.dim]
            polygon_orientation[lane_segment_idx] = torch.atan2(centerline[1, 1] - centerline[0, 1],
                                                                centerline[1, 0] - centerline[0, 0])
            polygon_height[lane_segment_idx] = centerline[1, 2] - centerline[0, 2]
            polygon_type[lane_segment_idx] = self._polygon_types.index(lane_segment.lane_type.value)
            polygon_is_intersection[lane_segment_idx] = self._polygon_is_intersections.index(
                lane_segment.is_intersection)

            left_boundary = torch.from_numpy(lane_segment.left_lane_boundary.xyz).float()
            right_boundary = torch.from_numpy(lane_segment.right_lane_boundary.xyz).float()
            point_position[lane_segment_idx] = torch.cat([left_boundary[:-1, :self.dim],
                                                          right_boundary[:-1, :self.dim],
                                                          centerline[:-1, :self.dim]], dim=0)
            left_vectors = left_boundary[1:] - left_boundary[:-1]
            right_vectors = right_boundary[1:] - right_boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[lane_segment_idx] = torch.cat([torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                                                             torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                                                             torch.atan2(center_vectors[:, 1], center_vectors[:, 0])],
                                                            dim=0)
            point_magnitude[lane_segment_idx] = torch.norm(torch.cat([left_vectors[:, :2],
                                                                      right_vectors[:, :2],
                                                                      center_vectors[:, :2]], dim=0), p=2, dim=-1)
            point_height[lane_segment_idx] = torch.cat([left_vectors[:, 2], right_vectors[:, 2], center_vectors[:, 2]],
                                                       dim=0)
            left_type = self._point_types.index(lane_segment.left_mark_type.value)
            right_type = self._point_types.index(lane_segment.right_mark_type.value)
            center_type = self._point_types.index('CENTERLINE')
            point_type[lane_segment_idx] = torch.cat(
                [torch.full((len(left_vectors),), left_type, dtype=torch.uint8),
                 torch.full((len(right_vectors),), right_type, dtype=torch.uint8),
                 torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_side[lane_segment_idx] = torch.cat(
                [torch.full((len(left_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                 torch.full((len(right_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

        for crosswalk in map_api.get_scenario_ped_crossings():
            crosswalk_idx = polygon_ids.index(crosswalk.id)
            edge1 = torch.from_numpy(crosswalk.edge1.xyz).float()
            edge2 = torch.from_numpy(crosswalk.edge2.xyz).float()
            start_position = (edge1[0] + edge2[0]) / 2
            end_position = (edge1[-1] + edge2[-1]) / 2
            polygon_position[crosswalk_idx] = start_position[:self.dim]
            polygon_position[crosswalk_idx + len(cross_walk_ids)] = end_position[:self.dim]
            polygon_orientation[crosswalk_idx] = torch.atan2((end_position - start_position)[1],
                                                             (end_position - start_position)[0])
            polygon_orientation[crosswalk_idx + len(cross_walk_ids)] = torch.atan2((start_position - end_position)[1],
                                                                                   (start_position - end_position)[0])
            polygon_height[crosswalk_idx] = end_position[2] - start_position[2]
            polygon_height[crosswalk_idx + len(cross_walk_ids)] = start_position[2] - end_position[2]
            polygon_type[crosswalk_idx] = self._polygon_types.index('PEDESTRIAN')
            polygon_type[crosswalk_idx + len(cross_walk_ids)] = self._polygon_types.index('PEDESTRIAN')
            polygon_is_intersection[crosswalk_idx] = self._polygon_is_intersections.index(None)
            polygon_is_intersection[crosswalk_idx + len(cross_walk_ids)] = self._polygon_is_intersections.index(None)

            if side_to_directed_lineseg((edge1[0] + edge1[-1]) / 2, start_position, end_position) == 'LEFT':
                left_boundary = edge1
                right_boundary = edge2
            else:
                left_boundary = edge2
                right_boundary = edge1
            num_centerline_points = math.ceil(torch.norm(end_position - start_position, p=2, dim=-1).item() / 2.0) + 1
            centerline = torch.from_numpy(
                compute_midpoint_line(left_ln_boundary=left_boundary.numpy(),
                                      right_ln_boundary=right_boundary.numpy(),
                                      num_interp_pts=int(num_centerline_points))[0]).float()

            point_position[crosswalk_idx] = torch.cat([left_boundary[:-1, :self.dim],
                                                       right_boundary[:-1, :self.dim],
                                                       centerline[:-1, :self.dim]], dim=0)
            point_position[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [right_boundary.flip(dims=[0])[:-1, :self.dim],
                 left_boundary.flip(dims=[0])[:-1, :self.dim],
                 centerline.flip(dims=[0])[:-1, :self.dim]], dim=0)
            left_vectors = left_boundary[1:] - left_boundary[:-1]
            right_vectors = right_boundary[1:] - right_boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[crosswalk_idx] = torch.cat(
                [torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                 torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                 torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
            point_orientation[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [torch.atan2(-right_vectors.flip(dims=[0])[:, 1], -right_vectors.flip(dims=[0])[:, 0]),
                 torch.atan2(-left_vectors.flip(dims=[0])[:, 1], -left_vectors.flip(dims=[0])[:, 0]),
                 torch.atan2(-center_vectors.flip(dims=[0])[:, 1], -center_vectors.flip(dims=[0])[:, 0])], dim=0)
            point_magnitude[crosswalk_idx] = torch.norm(torch.cat([left_vectors[:, :2],
                                                                   right_vectors[:, :2],
                                                                   center_vectors[:, :2]], dim=0), p=2, dim=-1)
            point_magnitude[crosswalk_idx + len(cross_walk_ids)] = torch.norm(
                torch.cat([-right_vectors.flip(dims=[0])[:, :2],
                           -left_vectors.flip(dims=[0])[:, :2],
                           -center_vectors.flip(dims=[0])[:, :2]], dim=0), p=2, dim=-1)
            point_height[crosswalk_idx] = torch.cat([left_vectors[:, 2], right_vectors[:, 2], center_vectors[:, 2]],
                                                    dim=0)
            point_height[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [-right_vectors.flip(dims=[0])[:, 2],
                 -left_vectors.flip(dims=[0])[:, 2],
                 -center_vectors.flip(dims=[0])[:, 2]], dim=0)
            crosswalk_type = self._point_types.index('CROSSWALK')
            center_type = self._point_types.index('CENTERLINE')
            point_type[crosswalk_idx] = torch.cat([
                torch.full((len(left_vectors),), crosswalk_type, dtype=torch.uint8),
                torch.full((len(right_vectors),), crosswalk_type, dtype=torch.uint8),
                torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_type[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [torch.full((len(right_vectors),), crosswalk_type, dtype=torch.uint8),
                 torch.full((len(left_vectors),), crosswalk_type, dtype=torch.uint8),
                 torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_side[crosswalk_idx] = torch.cat(
                [torch.full((len(left_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                 torch.full((len(right_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)
            point_side[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [torch.full((len(right_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                 torch.full((len(left_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

        num_points = torch.tensor([point.size(0) for point in point_position], dtype=torch.long)
        point_to_polygon_edge_index = torch.stack(
            [torch.arange(num_points.sum(), dtype=torch.long),
             torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)], dim=0)
        polygon_to_polygon_edge_index = []
        polygon_to_polygon_type = []
        for lane_segment in map_api.get_scenario_lane_segments():
            lane_segment_idx = polygon_ids.index(lane_segment.id)
            pred_inds = []
            for pred in lane_segment.predecessors:
                pred_idx = safe_list_index(polygon_ids, pred)
                if pred_idx is not None:
                    pred_inds.append(pred_idx)
            if len(pred_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack([torch.tensor(pred_inds, dtype=torch.long),
                                 torch.full((len(pred_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
                polygon_to_polygon_type.append(
                    torch.full((len(pred_inds),), self._polygon_to_polygon_types.index('PRED'), dtype=torch.uint8))
            succ_inds = []
            for succ in lane_segment.successors:
                succ_idx = safe_list_index(polygon_ids, succ)
                if succ_idx is not None:
                    succ_inds.append(succ_idx)
            if len(succ_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack([torch.tensor(succ_inds, dtype=torch.long),
                                 torch.full((len(succ_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
                polygon_to_polygon_type.append(
                    torch.full((len(succ_inds),), self._polygon_to_polygon_types.index('SUCC'), dtype=torch.uint8))
            if lane_segment.left_neighbor_id is not None:
                left_idx = safe_list_index(polygon_ids, lane_segment.left_neighbor_id)
                if left_idx is not None:
                    polygon_to_polygon_edge_index.append(
                        torch.tensor([[left_idx], [lane_segment_idx]], dtype=torch.long))
                    polygon_to_polygon_type.append(
                        torch.tensor([self._polygon_to_polygon_types.index('LEFT')], dtype=torch.uint8))
            if lane_segment.right_neighbor_id is not None:
                right_idx = safe_list_index(polygon_ids, lane_segment.right_neighbor_id)
                if right_idx is not None:
                    polygon_to_polygon_edge_index.append(
                        torch.tensor([[right_idx], [lane_segment_idx]], dtype=torch.long))
                    polygon_to_polygon_type.append(
                        torch.tensor([self._polygon_to_polygon_types.index('RIGHT')], dtype=torch.uint8))
        if len(polygon_to_polygon_edge_index) != 0:
            polygon_to_polygon_edge_index = torch.cat(polygon_to_polygon_edge_index, dim=1)
            polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
        else:
            polygon_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long)
            polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)

        map_data = {
            'map_polygon': {},
            'map_point': {},
            ('map_point', 'to', 'map_polygon'): {},
            ('map_polygon', 'to', 'map_polygon'): {},
        }
        map_data['map_polygon']['num_nodes'] = num_polygons
        map_data['map_polygon']['position'] = polygon_position
        map_data['map_polygon']['orientation'] = polygon_orientation
        if self.dim == 3:
            map_data['map_polygon']['height'] = polygon_height
        map_data['map_polygon']['type'] = polygon_type
        map_data['map_polygon']['is_intersection'] = polygon_is_intersection
        if len(num_points) == 0:
            map_data['map_point']['num_nodes'] = 0
            map_data['map_point']['position'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['orientation'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['magnitude'] = torch.tensor([], dtype=torch.float)
            if self.dim == 3:
                map_data['map_point']['height'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['type'] = torch.tensor([], dtype=torch.uint8)
            map_data['map_point']['side'] = torch.tensor([], dtype=torch.uint8)
        else:
            map_data['map_point']['num_nodes'] = num_points.sum().item()
            map_data['map_point']['position'] = torch.cat(point_position, dim=0)
            map_data['map_point']['orientation'] = torch.cat(point_orientation, dim=0)
            map_data['map_point']['magnitude'] = torch.cat(point_magnitude, dim=0)
            if self.dim == 3:
                map_data['map_point']['height'] = torch.cat(point_height, dim=0)
            map_data['map_point']['type'] = torch.cat(point_type, dim=0)
            map_data['map_point']['side'] = torch.cat(point_side, dim=0)
        map_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = polygon_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['type'] = polygon_to_polygon_type

        return map_data

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int) -> HeteroData:
        with open(self.processed_paths[idx], 'rb') as handle:
            return HeteroData(pickle.load(handle))

    def _download(self) -> None:
        return  # NOTE skip download
        # if complete raw/processed files exist, skip downloading
        if ((os.path.isdir(self.raw_dir) and len(self.raw_file_names) == len(self)) or
                (os.path.isdir(self.processed_dir) and len(self.processed_file_names) == len(self))):
            return
        self._processed_file_names = []
        self.download()

    def _process(self) -> None:
        # if complete processed files exist, skip processing
        if os.path.isdir(self.processed_dir) and len(self.processed_file_names) == len(self):
            return
        print('Processing...', file=sys.stderr)
        if os.path.isdir(self.processed_dir):
            for name in os.listdir(self.processed_dir):
                if name.endswith(('pkl', 'pickle')):
                    os.remove(os.path.join(self.processed_dir, name))
        else:
            os.makedirs(self.processed_dir)
        self._processed_file_names = [f'{raw_file_name}.pkl' for raw_file_name in self.raw_file_names]
        self.process()
        print('Done!', file=sys.stderr)


import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from utils import wrap_angle


class TargetBuilder(BaseTransform):

    def __init__(self,
                 num_historical_steps: int,
                 num_future_steps: int) -> None:
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps

    def __call__(self, data: HeteroData) -> HeteroData:
        origin = data['agent']['position'][:, self.num_historical_steps - 1]
        theta = data['agent']['heading'][:, self.num_historical_steps - 1]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(data['agent']['num_nodes'], 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        data['agent']['target'] = origin.new_zeros(data['agent']['num_nodes'], self.num_future_steps, 4)
        data['agent']['target'][..., :2] = torch.bmm(data['agent']['position'][:, self.num_historical_steps:, :2] -
                                                     origin[:, :2].unsqueeze(1), rot_mat)
        if data['agent']['position'].size(2) == 3:
            data['agent']['target'][..., 2] = (data['agent']['position'][:, self.num_historical_steps:, 2] -
                                               origin[:, 2].unsqueeze(-1))
        data['agent']['target'][..., 3] = wrap_angle(data['agent']['heading'][:, self.num_historical_steps:] -
                                                     theta.unsqueeze(-1))
        return data



from typing import Callable, Optional

import lightning as pl
from torch_geometric.loader import DataLoader


class ArgoverseV2DataModule(pl.LightningDataModule):

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_raw_dir: Optional[str] = None,
                 val_raw_dir: Optional[str] = None,
                 test_raw_dir: Optional[str] = None,
                 train_processed_dir: Optional[str] = None,
                 val_processed_dir: Optional[str] = None,
                 test_processed_dir: Optional[str] = None,
                 train_transform: Optional[Callable] = TargetBuilder(50, 60),
                 val_transform: Optional[Callable] = TargetBuilder(50, 60),
                 test_transform: Optional[Callable] = None,
                 **kwargs) -> None:
        super(ArgoverseV2DataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.train_processed_dir = train_processed_dir
        self.val_processed_dir = val_processed_dir
        self.test_processed_dir = test_processed_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def prepare_data(self) -> None:
        ArgoverseV2Dataset(self.root, 'train', self.train_raw_dir, self.train_processed_dir, self.train_transform)
        ArgoverseV2Dataset(self.root, 'val', self.val_raw_dir, self.val_processed_dir, self.val_transform)
        ArgoverseV2Dataset(self.root, 'test', self.test_raw_dir, self.test_processed_dir, self.test_transform)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ArgoverseV2Dataset(self.root, 'train', self.train_raw_dir, self.train_processed_dir,
                                                self.train_transform)
        self.val_dataset = ArgoverseV2Dataset(self.root, 'val', self.val_raw_dir, self.val_processed_dir,
                                              self.val_transform)
        self.test_dataset = ArgoverseV2Dataset(self.root, 'test', self.test_raw_dir, self.test_processed_dir,
                                               self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)


# Below is dataset generation middleware, Miku.
import math
from dataclasses import dataclass, asdict
import torch.nn as nn
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from utils import angle_between_2d_vectors

NUM_MAX_AGENTS = 135
NUM_HISTORICAL_STEPS = 50
TIME_SPAN = 10
INPUT_DIM = 2
HIDDEN_DIM = 128
A2A_RADIUS = 50
NUM_NODES = NUM_MAX_AGENTS * TIME_SPAN
NODE_FULL_GRAPH = torch.tensor([(i, j) for i in range(NUM_NODES) for j in range(NUM_NODES)]).transpose(0, 1)

@dataclass
class YamaiGraph:
    x: torch.Tensor
    edge_index: torch.Tensor

    @classmethod
    def from_data(cls, data: Data) -> "YamaiGraph":
        return cls(x=data.x, edge_index=data.edge_index)

    @classmethod
    def from_batch(cls, data: dict[str, Any]) -> "YamaiGraph":
        return cls(x=data['x'][0], edge_index=data['edge_index'][0])

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def num_features(self) -> int:
        return self.x.shape[1]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def __repr__(self) -> str:
        return f"YamaiGraph(x={self.x.shape}, edge_index={self.edge_index.shape})"

@dataclass
class Yamai:
    graphs: List[YamaiGraph]

    @classmethod
    def from_batch(cls, data: dict[str, Any]) -> "Yamai":
        return Yamai(graphs=[YamaiGraph.from_batch(i) for i in data['graphs']])

    def dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class MikuAgent:
    valid_mask: torch.Tensor # [92, 110]
    predict_mask: torch.Tensor # [92, 110]
    id: List[Optional[str]] # [92]
    type: torch.Tensor # [92]
    category: torch.Tensor # [92]
    position: torch.Tensor # [92, 110, 3]
    heading: torch.Tensor # [92, 110]
    velocity: torch.Tensor # [92, 110, 3]

    @property
    def num_nodes(self) -> int:
        return self.valid_mask.shape[0]

    @property
    def av_index(self) -> int:
        return self.num_nodes - 1

    @property
    def num_steps(self) -> int:
        return self.valid_mask.shape[1]

    def exists(self, agent_idx: int, step_idx: int):
        return self.valid_mask[agent_idx][step_idx]

    def stat(self, agent_idx: int, step_idx: int):
        pos_t = self.position[agent_idx][step_idx][:2]
        theta_t = self.heading[agent_idx][step_idx].unsqueeze(0)
        vel_t = self.velocity[agent_idx][step_idx][:2]
        return pos_t, theta_t, vel_t

    def __repr__(self) -> str:
        return (
            f"AstreaAgent(num_nodes={self.num_nodes}, num_steps={self.num_steps}, "
            f"valid_mask={self.valid_mask.shape}, predict_mask={self.predict_mask.shape}, "
            f"id={len(self.id)}, type={self.type.shape}, category={self.category.shape}, "
            f"position={self.position.shape}, heading={self.heading.shape}, velocity={self.velocity.shape})"
        )

    def to_graphs(self) -> List[YamaiGraph]:
        mask = self.valid_mask[:, :NUM_HISTORICAL_STEPS].contiguous()  # [A, T]
        pos_a = self.position[:, :NUM_HISTORICAL_STEPS, :INPUT_DIM].contiguous()  # [A, T, 2]
        motion_vector_a = torch.cat([pos_a.new_zeros(self.num_nodes, 1, INPUT_DIM),
                                     pos_a[:, 1:] - pos_a[:, :-1]], dim=1)  # [A, T, 2]
        head_a = self.heading[:, :NUM_HISTORICAL_STEPS].contiguous()  # [A, T]
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)  # [A, T, 2]

        vel = self.velocity[:, :NUM_HISTORICAL_STEPS, :INPUT_DIM].contiguous()  # [A, T, 2]

        x_a = torch.stack(  # [A, T, 4]
            [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),     # 表示智能体在两个连续时间步之间的移动距离
                angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),   # 计算智能体的运动向量和代理朝向向量之间的夹角
                torch.norm(vel[:, :, :2], p=2, dim=-1),        # 表示智能体在两个连续时间步之间的平均速度
                angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2])], dim=-1) # 计算智能体速度向量和代理朝向向量之间的夹角

        num_nodes = self.num_nodes
        num_nodes_t = num_nodes * TIME_SPAN
        node_full_graph = torch.tensor([(i, j) for i in range(num_nodes_t) for j in range(num_nodes_t)]).transpose(0, 1)
        yamai_graphs: List[YamaiGraph] = []

        for time_beg in range(0, NUM_HISTORICAL_STEPS, TIME_SPAN):
            time_end = time_beg + TIME_SPAN
            x_a_t = x_a[:, time_beg:time_end, :].reshape(num_nodes_t, 4)
            pos_a_t = pos_a[:, time_beg:time_end, :].reshape(num_nodes_t, 2)
            head_a_t = head_a[:, time_beg:time_end].reshape(num_nodes_t)
            head_vector_a_t = head_vector_a[:, time_beg:time_end, :].reshape(num_nodes_t, 2)
            rel_pos_a2a = pos_a_t[node_full_graph[0]] - pos_a_t[node_full_graph[1]]
            rel_head_a2a = wrap_angle(head_a_t[node_full_graph[0]] - head_a_t[node_full_graph[1]])
            r_a2a = torch.stack(
                [torch.norm(rel_pos_a2a, p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_a_t[node_full_graph[1]], nbr_vector=rel_pos_a2a),
                rel_head_a2a], dim=-1).reshape(num_nodes_t, num_nodes_t * 3)
            # node_full_graph.shape=torch.Size([2, 656100]), x_a_t.shape=torch.Size([810, 4]), r_a2a.shape=torch.Size([810, 2430])
            # print(f'{node_full_graph.shape=}, {x_a_t.shape=}, {r_a2a.shape=}')
            x_yamai = torch.cat([x_a_t, r_a2a], dim=1)
            edge_index_a2a = radius_graph(x=pos_a_t, r=A2A_RADIUS, loop=False, max_num_neighbors=300)
            yamai_graphs.append(YamaiGraph(x=x_yamai, edge_index=edge_index_a2a))

        return yamai_graphs

@dataclass
class MikuScene:
    scenario_id: str # uuid
    city: str # name
    agent: MikuAgent

    @classmethod
    def from_av2(cls, data: Any) -> "MikuScene":
        _agent = data['agent']
        _num_nodes = _agent['num_nodes']
        _av_index = int(_agent['av_index'])
        assert _num_nodes == _av_index + 1

        return cls(
            scenario_id=data['scenario_id'],
            city=data['city'],
            agent=MikuAgent(
                valid_mask=_agent['valid_mask'],
                predict_mask=_agent['predict_mask'],
                id=_agent['id'],
                type=_agent['type'],
                category=_agent['category'],
                position=_agent['position'],
                heading=_agent['heading'],
                velocity=_agent['velocity'],
            ),
        )

    def to_yamai(self) -> Yamai:
        return Yamai(graphs=self.agent.to_graphs())

class MikuDataset(ArgoverseV2Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 raw_dir: Optional[str] = None,
                 processed_dir: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 dim: int = 3,           # 数据维度
                 num_historical_steps: int = 50,
                 num_future_steps: int = 60,
                 predict_unseen_agents: bool = False,                #  预测未见智能体
                 vector_repr: bool = True) -> None:                  #  向量表示
        if processed_dir is None:
            processed_dir = os.path.join(root, split, "yamai")
        super().__init__(root, split, raw_dir, processed_dir, transform, dim,
                         num_historical_steps, num_future_steps, predict_unseen_agents, vector_repr)
        self._num_samples = {
            'train': 9,
            'val': 9,
            'test': 9,
        }[split]

    def process(self) -> None:
        for raw_file_name in tqdm(self.raw_file_names):               # tqdm是一个进度条库，用于在控制台显示进度
            df = pd.read_parquet(os.path.join(self.raw_dir, raw_file_name, f'scenario_{raw_file_name}.parquet'))     # 用pandas库读取一个Parquet文件
            map_dir = Path(self.raw_dir) / raw_file_name              # 使用pathlib库创建一个路径对象，指向包含地图数据的目录
            map_path = sorted(map_dir.glob('log_map_archive_*.json'))[0]   # 在map_dir目录中查找所有以log_map_archive_开头的 JSON 文件，并选择第一个文件作为地图数据文件
            map_data = read_json_file(map_path)              # 读取地图数据json文件，并将内容存储在map_data变量中
            centerlines = {lane_segment['id']: Polyline.from_json_data(lane_segment['centerline'])
                           for lane_segment in map_data['lane_segments'].values()}              # 从地图数据中提取车道线信息，并创建一个车道线id到Polyline对象的映射
            map_api = ArgoverseStaticMap.from_json(map_path)               # 使用ArgoverseStaticMap类从地图数据文件创建一个地图 API 对象
            data = dict()                 # 创建一个空字典，用于存储处理后的数据
            data['scenario_id'] = self.get_scenario_id(df)               # 调用self.get_scenario_id，从数据帧df中获取场景id，并将其添加到data字典中
            data['city'] = self.get_city(df)
            data['agent'] = self.get_agent_features(df)
            data.update(self.get_map_features(map_api, centerlines))        # 调用self.get_map_features方法，获取地图特征，并将结果更新到data字典中
            yamai = MikuScene.from_av2(data).to_yamai()
            with open(os.path.join(self.processed_dir, f'{raw_file_name}.pkl'), 'wb') as handle:   # 打开一个raw_file_name文件，并使用pickle序列化数据
                pickle.dump(yamai.dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)             #使用HIGHEST_PROTOCOL可以确保代码使用的是当前Python版本支持的最新协议


class MikuDataModule(pl.LightningDataModule):
    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_raw_dir: Optional[str] = None,
                 val_raw_dir: Optional[str] = None,
                 test_raw_dir: Optional[str] = None,
                 train_processed_dir: Optional[str] = None,
                 val_processed_dir: Optional[str] = None,
                 test_processed_dir: Optional[str] = None,
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 test_transform: Optional[Callable] = None,
                 **kwargs) -> None:
        super(MikuDataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.train_processed_dir = train_processed_dir
        self.val_processed_dir = val_processed_dir
        self.test_processed_dir = test_processed_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def prepare_data(self) -> None:               # 一次性的数据准备操作：确保数据集已经被加载和处理
        MikuDataset(self.root, 'train', self.train_raw_dir, self.train_processed_dir, self.train_transform)
        MikuDataset(self.root, 'val', self.val_raw_dir, self.val_processed_dir, self.val_transform)
        MikuDataset(self.root, 'test', self.test_raw_dir, self.test_processed_dir, self.test_transform)

    def setup(self, stage: Optional[str] = None) -> None:      # 为训练、验证和测试阶段准备数据：确保最新的数据集
        self.train_dataset = MikuDataset(self.root, 'train', self.train_raw_dir, self.train_processed_dir,
                                                self.train_transform)
        self.val_dataset = MikuDataset(self.root, 'val', self.val_raw_dir, self.val_processed_dir,
                                              self.val_transform)
        self.test_dataset = MikuDataset(self.root, 'test', self.test_raw_dir, self.test_processed_dir,
                                               self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)


# Below is the REAL generated dataset, Yamai.
from torch.utils.data import Dataset as EasyDataset, DataLoader as EasyDataLoader

class YamaiDataset(EasyDataset[Yamai]):
    def __init__(self, root: str, split: str, yamai_dir: str = "yamai") -> None:
        self.paths = [ent.path for ent in os.scandir(os.path.join(root, split, yamai_dir))]

    def __getitem__(self, idx: int) -> Yamai:
        with open(self.paths[idx], 'rb') as handle:
            return pickle.load(handle)

    def __len__(self) -> int:
        return len(self.paths)

class YamaiDataModule(pl.LightningDataModule):
    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_raw_dir: Optional[str] = None,
                 val_raw_dir: Optional[str] = None,
                 test_raw_dir: Optional[str] = None,
                 train_processed_dir: Optional[str] = None,
                 val_processed_dir: Optional[str] = None,
                 test_processed_dir: Optional[str] = None,
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 test_transform: Optional[Callable] = None,
                 **kwargs) -> None:
        super(YamaiDataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.train_processed_dir = train_processed_dir
        self.val_processed_dir = val_processed_dir
        self.test_processed_dir = test_processed_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def prepare_data(self) -> None:               # 一次性的数据准备操作：确保数据集已经被加载和处理
        YamaiDataset(self.root, 'train')
        YamaiDataset(self.root, 'val')
        YamaiDataset(self.root, 'test')

    def setup(self, stage: Optional[str] = None) -> None:      # 为训练、验证和测试阶段准备数据：确保最新的数据集
        self.train_dataset = YamaiDataset(self.root, 'train')
        self.val_dataset = YamaiDataset(self.root, 'val')
        self.test_dataset = YamaiDataset(self.root, 'test')

    def train_dataloader(self):
        return EasyDataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return EasyDataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return EasyDataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)
