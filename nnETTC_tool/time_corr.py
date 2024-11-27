import argparse
from pathlib import Path

import pandas as pd
import rosbag
from tqdm import tqdm

from event_camera_py import Decoder

class TimeCorrection:
    
    def __init__(self, rosbag_path=None, rgb_topic=None, exp_topic=None, rgb_expect_freq=None, event_topic=None):
        
        self.rosbag_path = rosbag_path
        self.rgb_topic = rgb_topic
        self.exp_topic = exp_topic
        self.rgb_expect_freq = rgb_expect_freq
        self.event_topic = event_topic
        
        self.event_count = 0

    def rostime_to_ts(self, rostime):
        return rostime.secs + rostime.nsecs / float(1e9)


    def get_rgb_trigger_timemap(self,
                                rosbag_path: str, 
                                rgb_topic: str, 
                                exp_topic: str, 
                                rgb_expect_freq: int) -> pd.DataFrame:
        """
                        <--- exposure_ts ---> exp_header_ts
                        <--- exposure_ts ---> rgb_header_ts
        rgb_exp_start_ts <--- exposure_ts ---> rgb_exp_end_ts
        """
        rgb_trigger_list = []
        last_trigger_ts = 0
        expect_time_interval = 1.0 / rgb_expect_freq
        bool_first_frame = True

        with rosbag.Bag(rosbag_path, 'r') as bag:
            # Get trigger time from exp_topic and match with rgb_trigger_df
            trig_bar = tqdm(total=bag.get_message_count(topic_filters=exp_topic), desc="Processing RGB Trigger Messages")

            for exp_cnt, (_, exp_msg, _) in enumerate(bag.read_messages(topics=[exp_topic])):
                rgb_exp_end_ts = self.rostime_to_ts(exp_msg.header.stamp)
                exposure_ts = exp_msg.exposure_time / 1e6
                rgb_exp_start_ts = rgb_exp_end_ts - exposure_ts
                dt = rgb_exp_start_ts - last_trigger_ts
                last_trigger_ts = rgb_exp_start_ts

                # Judge if dt is bigger than 20% of expected_interval
                if not bool_first_frame and abs(dt - expect_time_interval) > expect_time_interval * 0.2:
                    print(f"\033[91mError: {exp_cnt:010d}, RGB frequency is not [{rgb_expect_freq}], the time difference is [{dt}]\033[0m")
                    continue
                if bool_first_frame:
                    bool_first_frame = False

                rgb_trigger_list.append({
                    'rgb_exp_start_ts': rgb_exp_start_ts,
                    'rgb_exp_end_ts': rgb_exp_end_ts,
                    'exposure_time_tus': exp_msg.exposure_time
                })
                trig_bar.update(1)

            rgb_trigger_df = pd.DataFrame(rgb_trigger_list)

            
            rgb_bar = tqdm(total=bag.get_message_count(topic_filters=rgb_topic), desc="Processing RGB Image Messages")
            for rgb_index, (_, msg, _) in enumerate(bag.read_messages(topics=[rgb_topic])):
                rgb_header_ts = self.rostime_to_ts(msg.header.stamp)
                # Find the corresponding row in rgb_trigger_df
                matching_row = rgb_trigger_df[abs(rgb_trigger_df['rgb_exp_end_ts'] - rgb_header_ts) < 1e-6]  # limit by float precision, if time diff less than 1e-6s (1us), then match
                if matching_row.empty: 
                    error_message = f"Error: {rgb_index:010d}, No matching row found for rgb timestamp [{rgb_header_ts}]"
                    print(f"\033[91m{error_message}\033[0m")
                    continue
                else:
                    rgb_trigger_df.loc[matching_row.index, 'has_rgb'] = 1 
                    rgb_trigger_df.loc[matching_row.index, 'rgb_bag_index'] = rgb_index  # for save image use

                rgb_bar.update(1)
        bag.close()
        
        return rgb_trigger_df


    def get_event_trigger_timemap(self, rosbag_path: str, event_topic=None):
        """
        |=== e_msg_header: e_msg_header_ts (computer clock) ====
        |
        |--- event 1 ----
        |--- event 2 ----
        |--- trig_events: trig_events_ts (event camera clock) ----
        |--- event 4 ----
        """
        d = Decoder()
        event_trig_list = []
        event_count = 0
        
        with rosbag.Bag(rosbag_path, 'r') as bag:
            bar_messages = tqdm(total=bag.get_message_count(topic_filters=event_topic), desc="Processing Evnet Trigger Messages")
            for _, (_, e_msg, _) in enumerate(bag.read_messages(topics=[event_topic])):
                
                d.decode_bytes(
                    e_msg.encoding, e_msg.width, e_msg.height, e_msg.time_base, e_msg.events
                )
                cd_events = d.get_cd_events()
                event_count += cd_events.shape[0]
                
                trig_events = d.get_ext_trig_events()  # get single trigger event in one event message

                if trig_events.size != 0 and trig_events[0][0] == 0:
                    e_msg_header_ts = self.rostime_to_ts(e_msg.header.stamp)
                    event_trig_list.append({
                        'e_msg_header_ts': e_msg_header_ts,
                        'event_trigger_tnsec': trig_events.tolist()[0][1]*1e3  # nanosecond
                    })
                bar_messages.update(1)
        
        self.event_count = event_count
        
        return pd.DataFrame(event_trig_list)


    def match_triggers(
        self,
        rgb_trigger_df: pd.DataFrame, 
        event_trigger_df: pd.DataFrame, 
        tolerance_tms=15) -> pd.DataFrame:
        """
        Match the event triggers to the rgb triggers
        rgb_exp_start_ts | rgb_exp_end_ts | exposure_time_tus || has_rgb | rgb_bag_index || e_msg_header_ts | event_trigger_tnsec | event_count || corr_exposure_start_timestamp_us | corr_exposure_end_timestamp_us
        """
        for _, e_trigger_row in tqdm(event_trigger_df.iterrows(), total=len(event_trigger_df), desc="Matching triggers"):
            e_msg_header_ts = e_trigger_row['e_msg_header_ts']
            # Find the corresponding row in rgb_trigger_df
            tolerance_ts = tolerance_tms / 1000
            matching_row = rgb_trigger_df[(rgb_trigger_df['rgb_exp_start_ts'] - e_msg_header_ts).abs() < tolerance_ts]
            
            if not matching_row.empty and len(matching_row) == 1:
                ind = matching_row.index
                rgb_trigger_df.loc[ind, 'e_msg_header_ts'] = e_msg_header_ts   # second
                rgb_trigger_df.loc[ind, 'event_trigger_tnsec'] = e_trigger_row['event_trigger_tnsec']  # nanosecond
            else:
                matching_ts_value = rgb_trigger_df.loc[
                    (rgb_trigger_df['rgb_exp_start_ts'] - e_msg_header_ts).abs().idxmin(),
                    'rgb_exp_start_ts'
                ]
                min_ts_diff = abs(e_msg_header_ts - matching_ts_value)
                error_message = (
                    f"Error: Matching EVENT trigger to RGB failed:\n "
                    f"e_msg_header_ts: [{e_msg_header_ts}]\n "
                    f"matching_ts:     [{matching_ts_value}]\n "
                    f"min_ts_diff:     [{min_ts_diff}]"
                )
                print(f"\033[91m{error_message}\033[0m")
                continue
        
        time_correction_df = rgb_trigger_df
        # Select row as as input of calibration
        calib_need_map_df = rgb_trigger_df[['rgb_exp_start_ts','event_trigger_tnsec']]
        
        return time_correction_df, calib_need_map_df


    def get_time_correction_df(self):
        
        rgb_trigger_df = self.get_rgb_trigger_timemap(rosbag_path=self.rosbag_path, 
                                                        rgb_topic=self.rgb_topic, 
                                                        exp_topic=self.exp_topic, 
                                                        rgb_expect_freq=self.rgb_expect_freq)

        event_trigger_df = self.get_event_trigger_timemap(rosbag_path=self.rosbag_path,
                                                            event_topic=self.event_topic)

        time_correction_df, calib_need_map_df = self.match_triggers(rgb_trigger_df, event_trigger_df)
        
        # correct rgb timestamp to event timestamp
        corr_exposure_start_timestamp_us = int(e_trigger_row['event_trigger_tnsec'] / 1e3)
        corr_exposure_end_timestamp_us = int(corr_exposure_start_timestamp_us + rgb_trigger_df.loc[ind, 'exposure_time_tus'])
        # save rgb new timestamp, which is corrected by event timestamp
        rgb_trigger_df.loc[ind, 'corr_exposure_start_timestamp_us'] = corr_exposure_start_timestamp_us   # microsecond
        rgb_trigger_df.loc[ind, 'corr_exposure_end_timestamp_us'] = corr_exposure_end_timestamp_us
            
        return time_correction_df, calib_need_map_df, self.event_count
    
