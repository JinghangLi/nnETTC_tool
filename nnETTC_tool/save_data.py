# 1. 新建文件夹结构
# 2. 存储 RGB 图像，并修改时间戳为事件相机的时间戳
import os
from pathlib import Path

import pandas as pd
import rosbag
from tqdm import tqdm
import cv2
from cv_bridge import CvBridge
bridge = CvBridge()


# 创建目录的函数
def create_structure(base_path, structure):
    for name, content in structure.items():
        path = (Path(base_path) / name).absolute()
        # if content is a dict, create a child directory
        if isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            create_structure(path, content)
        else:
            path.mkdir(parents=True, exist_ok=True)


def rostime_to_ts(rostime):
    return rostime.secs + rostime.nsecs / float(1e9)


def save_rgb_and_remap_time(bag_path: str, image_save_dir: str, rgb_topic: str, rgb_events_trigger_map_df: pd.DataFrame):

    def _zyt_image_name_role(timestamp_us: int):
        ts_s_part = int(timestamp_us // 1e6)
        ts_us_part = int(timestamp_us % 1e6)
        # 12 digits in total, use 0 to fill the left
        return f'{ts_s_part:09d}_{ts_us_part:09d}'

    hybrid_map_index = 0
    exp_tus_list = []
    
    rgb_distorted_dir = Path(image_save_dir) / 'distorted'
    assert rgb_distorted_dir.exists(), f'{rgb_distorted_dir} does not exist'
    
    with rosbag.Bag(bag_path, 'r') as bag:
        rgb_bar = tqdm(total=bag.get_message_count(topic_filters=rgb_topic), desc="Processing RGB Image Messages")
        for rgb_index, (_, msg, _) in enumerate(bag.read_messages(topics=[rgb_topic])):
            rgb_header_ts = rostime_to_ts(msg.header.stamp)

            hybrid_row = rgb_events_trigger_map_df.iloc[hybrid_map_index]
            if abs(rgb_header_ts - hybrid_row['rgb_exp_end_ts']) < 1e-6: # limit by float precision, if time diff less than 1e-6s (1us), then match
                hybrid_map_index += 1
                if hybrid_map_index >= len(rgb_events_trigger_map_df):
                    break
                
                assert hybrid_row['rgb_bag_index'] == rgb_index, f'Error: rgb_bag_index [{rgb_index}] does not match.'
                
                # modify rgb ts to event ts
                exposure_start_timestamp_us = int(hybrid_row['event_trigger_tnsec'] / 1e3)
                exposure_end_timestamp_us = int(exposure_start_timestamp_us + hybrid_row['exposure_time_tus'])
                exp_tus_list.append([exposure_start_timestamp_us, exposure_end_timestamp_us])
                
                # save rgb image from msg
                rgb_img = bridge.compressed_imgmsg_to_cv2(msg)
                
                # distored image using config
                # rgb_img_distorted = cv2.remap(rgb_img, map1, map2, interpolation=cv2.INTER_LINEAR)
                
                image_name = _zyt_image_name_role(exposure_start_timestamp_us)
                cv2.imwrite(os.path.join(rgb_distorted_dir.as_posix(), f'{image_name}.png'), rgb_img)
            else:
                error_msg = f"Error: rgb_header_ts [{rgb_header_ts}] does not match rgb_events_trigger_map_df[{hybrid_row['rgb_exp_end_ts']}]"
                print(f"\033[91m{error_msg}\033[0m")
            
            rgb_bar.update(1)
            
        bag.close()
            
    # save exposure timestamps
    with open(os.path.join(image_save_dir, 'exposure_timestamps.txt'), 'w') as f:
        for exp_tus in exp_tus_list:
            f.write(f'{exp_tus[0]} {exp_tus[1]}\n')
                

            


