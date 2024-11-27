import argparse
from pathlib import Path

import pandas as pd
import rosbag
from tqdm import tqdm

from event_camera_py import Decoder


def rostime_to_ts(rostime):
    return rostime.secs + rostime.nsecs / float(1e9)


def get_rgb_trigger_timemap(rosbag_path: str, 
                            rgb_topic: str, 
                            exp_topic: str, 
                            rgb_freq: int) -> pd.DataFrame:
    """
                     <--- exposure_ts ---> exp_header_ts
                     <--- exposure_ts ---> rgb_header_ts
    rgb_exp_start_ts <--- exposure_ts ---> rgb_exp_end_ts
    """
    rgb_trigger_list = []
    last_trigger_ts = 0
    expected_interval = 1.0 / int(rgb_freq)
    bool_first_frame = True

    with rosbag.Bag(rosbag_path, 'r') as bag:
        # Get trigger time from exp_topic and match with rgb_trigger_df
        trig_bar = tqdm(total=bag.get_message_count(topic_filters=exp_topic), desc="Processing RGB Trigger Messages")

        for exp_cnt, (_, exp_msg, _) in enumerate(bag.read_messages(topics=[exp_topic])):
            rgb_exp_end_ts = rostime_to_ts(exp_msg.header.stamp)
            exposure_ts = exp_msg.exposure_time / float(1e6) # exp_msg.exposure_time unit is us, convert to s
            rgb_exp_start_ts = rgb_exp_end_ts - exposure_ts
            dt = rgb_exp_start_ts - last_trigger_ts
            last_trigger_ts = rgb_exp_start_ts

            # Judge if dt is bigger than 50% of expected_interval
            if not bool_first_frame and abs(dt - expected_interval) > expected_interval * 0.2:
                error_message = f"Error: {exp_cnt:010d}, RGB frequency is not [{int(rgb_freq)}], the time difference is [{dt}]"
                print(f"\033[91m{error_message}\033[0m")
                continue
            if bool_first_frame:
                bool_first_frame = False

            rgb_trigger_list.append({
                'rgb_exp_index': f"{exp_cnt:010d}",
                'rgb_exp_start_ts': rgb_exp_start_ts,
                'rgb_exp_end_ts': rgb_exp_end_ts,
                'exposure_time_ts': exposure_ts
            })
            trig_bar.update(1)
        rgb_trigger_df = pd.DataFrame(rgb_trigger_list)

        
        rgb_bar = tqdm(total=bag.get_message_count(topic_filters=rgb_topic), desc="Processing RGB Image Messages")
        for rgb_cnt, (_, msg, _) in enumerate(bag.read_messages(topics=[rgb_topic])):
            rgb_header_ts = rostime_to_ts(msg.header.stamp)
            # Find the corresponding row in rgb_trigger_df
            matching_row = rgb_trigger_df[rgb_trigger_df['rgb_exp_end_ts'] == rgb_header_ts]
            if matching_row.empty:
                error_message = f"Error: {rgb_cnt:010d}, No matching row found for rgb timestamp [{rgb_header_ts}]"
                print(f"\033[91m{error_message}\033[0m")
                continue
            else:
                rgb_trigger_df.loc[matching_row.index, 'has_rgb'] = 1 

            rgb_bar.update(1)
    bag.close()
    
    return rgb_trigger_df


def get_event_trigger_timemap(rosbag_path, event_topic=None):
    """
    |=== e_msg_header: e_msg_ts (computer clock) ====
    |
    |--- event 1 ----
    |--- event 2 ----
    |--- trig_events: trig_events_ts (event camera clock) ----
    |--- event 4 ----
    """
    decoder = Decoder()
    event_trig_list = []

    with rosbag.Bag(rosbag_path, 'r') as bag:
        bar_messages = tqdm(total=bag.get_message_count(topic_filters=event_topic), desc="Processing Evnet Trigger Messages")
        for e_cnt, (_, e_msg, _) in enumerate(bag.read_messages(topics=[event_topic])):
            decoder.decode_bytes(e_msg.encoding, e_msg.width, e_msg.height, e_msg.time_base, e_msg.events)
            trig_events = decoder.get_ext_trig_events()  # get single trigger event in one event message

            if trig_events.size != 0 and trig_events[0][0] == 0:
                e_msg_ts = rostime_to_ts(e_msg.header.stamp)
                event_trig_list.append({
                    'event_index': f"{e_cnt:010d}",
                    'e_msg_ts': e_msg_ts,
                    'event_trigger_tnsec': trig_events.tolist()[0][1]*1000  # nanosecond
                })

            bar_messages.update(1)
    bag.close()
    
    return pd.DataFrame(event_trig_list)


def match_triggers(rgb_trigger_df: pd.DataFrame, 
                   event_trigger_df: pd.DataFrame, 
                   threshold_ts=0.02,
                   zyt_hybrid_diff_ts=None) -> pd.DataFrame:
    """
    |=== e_msg_header: e_msg_ts (computer clock) ====
    |
    |--- event 1 ----
    |--- event 2 ----
    |--- trig_events: trig_events_ts (event camera clock) ----
    |--- event 4 ----
    """
    for _, e_msg_row in tqdm(event_trigger_df.iterrows(), total=len(event_trigger_df), desc="Matching triggers"):
        e_msg_ts = e_msg_row['e_msg_ts']
        # Find the corresponding row in rgb_trigger_df
        matching_row = rgb_trigger_df[(rgb_trigger_df['rgb_exp_start_ts'] - e_msg_ts).abs() < threshold_ts]
        
        if not matching_row.empty and len(matching_row) == 1:
            rgb_trigger_df.loc[matching_row.index, 'e_msg_ts'] = e_msg_ts   # second
            rgb_trigger_df.loc[matching_row.index, 'event_trigger_tnsec'] = e_msg_row['event_trigger_tnsec']  # nanosecond
        else:
            matching_ts_value = rgb_trigger_df.loc[
                (rgb_trigger_df['rgb_exp_start_ts'] - e_msg_ts).abs().idxmin(),
                'rgb_exp_start_ts'
            ]
            min_ts_diff = abs(e_msg_ts - matching_ts_value)
            error_message = (
                f"Error: Matching EVENT trigger to RGB failed:\n "
                f"e_msg_ts:    [{e_msg_ts}]\n "
                f"matching_ts: [{matching_ts_value}]\n "
                f"min_ts_diff: [{min_ts_diff}]"
            )
            print(f"\033[91m{error_message}\033[0m")
            continue
    
    trigger_debug_df = rgb_trigger_df
    
    # Align the timestamps of the zyt sensor and the hybrid sensor
    if zyt_hybrid_diff_ts is not None:
        assert zyt_hybrid_diff_ts > 0, "Error: zyt_hybrid_diff_ts < 0"
        rgb_trigger_df['rgb_exp_start_ts'] -= zyt_hybrid_diff_ts
        rgb_trigger_df['rgb_exp_end_ts'] -= zyt_hybrid_diff_ts
    
    # calculate the time difference between the event trigger and the rgb trigger
    rgb_trigger_df['t_offset'] = rgb_trigger_df['rgb_exp_start_ts'] * 1e6 - rgb_trigger_df['event_trigger_tnsec'] / 1e3  # microsecond
    
    # Select row as as input of calibration
    match_triggers_df = rgb_trigger_df[
        ['rgb_exp_start_ts','event_trigger_tnsec']]
    
    return match_triggers_df, trigger_debug_df



if __name__ == "__main__":
    """
    Code for getting image and events trigger timestamps map.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--rosbag_path", required=True, help="ROS bag file to extract or directory containing bags")
    parser.add_argument("--event_topic", required=True, help="Event topic")
    parser.add_argument("--rgb_topic",   required=True, help="RGB camera topic")
    parser.add_argument('--exp_topic',   default="/camera/exposure_time", help='Exposure topic name')
    parser.add_argument("--rgb_freq",    required=True, help="RGB camera frequence")
    parser.add_argument("--timemap_path",required=True, help="Time map save path")
    args = parser.parse_args()
    
    print(f"Start getting trigger timestamps map of: {args.rosbag_path}")

    rgb_trigger_df = get_rgb_trigger_timemap(rosbag_path=args.rosbag_path,
                                            rgb_topic=args.rgb_topic,
                                            exp_topic=args.exp_topic,
                                            rgb_freq=args.rgb_freq)
    rgb_trigger_df.to_csv(Path(args.timemap_path).parent / 'rgb_trigger_df.csv', sep=' ', header=True, index=False)


    event_trigger_df = get_event_trigger_timemap(rosbag_path=args.rosbag_path,
                                                 event_topic=args.event_topic)
    event_trigger_df.to_csv(Path(args.timemap_path).parent / 'event_trigger_df.csv', sep=' ', header=True, index=False)

    rgb_trigger_df = pd.read_csv(Path(args.timemap_path).parent / 'rgb_trigger_df.csv', sep=' ')
    event_trigger_df = pd.read_csv(Path(args.timemap_path).parent / 'event_trigger_df.csv', sep=' ')

    rgb_events_trigger_map_df, debug_df = match_triggers(rgb_trigger_df, event_trigger_df)

    # # Create a new directory named calib_<bag_filename> in the same directory as the bag file
    # bag_filename = os.path.basename(args.rosbag_path).replace('.bag', '')
    # calib_dir = os.path.join(os.path.dirname(args.rosbag_path), f'calib_{bag_filename}')
    # os.makedirs(calib_dir, exist_ok=True)

    debug_df.to_csv(Path(args.timemap_path).parent / 'trigger_debug.csv', sep=' ', header=True, index=False)

    rgb_events_trigger_map_df.to_csv(f"{args.timemap_path}", sep=' ', header=False, index=False)

    print(f"Finish getting trigger timestamps map, save to: {args.timemap_path}")
