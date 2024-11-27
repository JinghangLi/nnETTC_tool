import rosbag
from event_camera_py import Decoder
import h5py
import numpy as np
from tqdm import tqdm
import os
from colorama import Fore, Style
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()
from collections import defaultdict
import time
import glob
import argparse
import warnings
import yaml



class Event:
    def __init__(self, h5file, topic, bag, t=None, image_index=None):
        self.h5file = h5file
        self.topic = topic
        self.bag = bag
        self.time_list = t
        self.image_index=image_index    # The tenth image of timestamp

        self.name_parts = topic.split('/')
        path = 'prophesee/'+self.name_parts[1]

        if path in h5file:
            self.group = h5file[path]
        else:
            self.group = h5file.create_group(path)

        event_count = self.get_event_count(bag, topic)

        self.chunk_size = 40000

        self.x_ds = self.group.create_dataset(
            "x",
            (event_count,),
            compression="lzf",
            dtype="u2",
            chunks=(self.chunk_size,),
            maxshape=(None,),
        )
        self.y_ds = self.group.create_dataset(
            "y",
            (event_count,),
            compression="lzf",
            dtype="u2",
            chunks=(self.chunk_size,),
            maxshape=(None,),
        )
        self.t_ds = self.group.create_dataset(
            "t",
            (event_count,),
            compression="lzf",
            dtype="i8",
            chunks=(self.chunk_size,),
            maxshape=(None,),
        )
        self.p_ds = self.group.create_dataset(
            "p",
            (event_count,),
            compression="lzf",
            dtype="i1",
            chunks=(self.chunk_size,),
            maxshape=(None,),
        )

        # Create four lists and a counter cache area
        self.x_buf = []  
        self.y_buf = []  
        self.t_buf = []  
        self.p_buf = []  
        self.count_buf = 0

        # Writing position
        self.start = 0
        self.prev_msg = None

        # Create index mapping
        self.has_index_map = False

        self.decoder = Decoder()

        self.has_trigger = False

        self.time_offset_base = 0
        self.time_offset = 0
        self.trigger_count = 0
        self.last_event_time = 0
    # Add data to the buffer to count the total number of event camera data
    # for the specified topic in the given ROS bag file
    def get_event_count(self, bag, topic):
        ## TODO - get count from verify or timing script
        count = 0
        d = Decoder()
        for topic, msg, t in bag.read_messages(topics=[topic]):
            d.decode_bytes(
                msg.encoding, msg.width, msg.height, msg.time_base, msg.events
            )
            cd_events = d.get_cd_events()
            count += cd_events.shape[0]
        return count
    
    # Add data to buffer
    def add_events(self, x, y, t, p):
        msg_shape = x.shape[0]

        if msg_shape == 0:
            return

        self.x_buf.append(x)
        self.y_buf.append(y)
        self.t_buf.append(t)
        self.p_buf.append(p)
        self.count_buf += msg_shape

    # Write buffer to hdf5
    def flush_buffer(self, ignore_chunk=False):
        x = np.concatenate(self.x_buf)
        y = np.concatenate(self.y_buf)
        t = np.concatenate(self.t_buf)
        p = np.concatenate(self.p_buf)

        if not ignore_chunk:
            added_idx = self.chunk_size * (x.shape[0] // self.chunk_size)
        else:
            added_idx = x.shape[0]

        self.end_idx = self.start + added_idx

        self.x_ds[self.start : self.end_idx] = x[:added_idx]
        self.y_ds[self.start : self.end_idx] = y[:added_idx]
        self.t_ds[self.start : self.end_idx] = t[:added_idx]
        self.p_ds[self.start : self.end_idx] = p[:added_idx]

        self.start = self.start + added_idx

        if not ignore_chunk:
            self.x_buf = [x[added_idx:]]
            self.y_buf = [y[added_idx:]]
            self.t_buf = [t[added_idx:]]
            self.p_buf = [p[added_idx:]]
            self.count_buf = self.count_buf - added_idx
        else:
            self.x_buf = []
            self.y_buf = []
            self.t_buf = []
            self.p_buf = []
            self.count_buf = 0

    # Process event camera data messages from ROS bag and convert them into formatted data stored in an HDF5 file
    def process(self, msg, topic=None, bag_time=None):
        self.decoder.decode_bytes(
            msg.encoding, msg.width, msg.height, msg.time_base, msg.events
        )
        cd_events = self.decoder.get_cd_events()
        trig_events = self.decoder.get_ext_trig_events()

        # Received the timestamp of the tenth image and started recording events
        if self.has_trigger == False:
            if trig_events.size != 0 and trig_events.tolist()[0][0] == 0:
                if abs(msg.header.stamp.to_nsec() / 1e3 - self.time_list[self.image_index][1]) < 15000:
                    self.time_offset = trig_events.tolist()[0][1]
                    self.has_trigger = True
                    print("self.has_trigger",self.has_trigger)

        if self.has_trigger==True:
            # Synchronize time every 20 triggers
            if trig_events.size != 0 and trig_events.tolist()[0][0] == 0:
                self.trigger_count += 1
                if self.trigger_count % 20 == 0:
                    if len(self.time_list) > self.image_index-1+self.trigger_count :
                        if abs(msg.header.stamp.to_nsec() / 1e3 - self.time_list[self.image_index-1+self.trigger_count][1]) < 15000: # <15ms
                            self.time_offset = -(self.time_list[self.image_index-1+self.trigger_count][1]-self.time_list[self.image_index][1])+trig_events.tolist()[0][1]
                        else:
                            print("msg.header.stamp.to_nsec() / 1e3:",msg.header.stamp.to_nsec() / 1e3)
                            print("self.time_list[self.image_index-1+self.trigger_count][1]::",self.time_list[self.image_index-1+self.trigger_count][1])
                            print(Fore.RED,"Events synchronization failed,Maybe Event camera lost triggers!",Style.RESET_ALL)

            # Calculate the time offset for the events
            t = cd_events['t'][0:] - self.time_offset
            # Filter event data that meets the criteria
            has_started = t >= self.last_event_time
            x = cd_events['x'][has_started]
            y = cd_events['y'][has_started]
            t = t[has_started]
            p = cd_events['p'][has_started]
            if len(t) > 0:
                self.last_event_time = t[-1]+1
            else:
                print(self.last_event_time)
                print(Fore.GREEN,"Exist an event empty message!",Style.RESET_ALL)


            self.add_events(x, y, t, p)

            if self.count_buf > self.chunk_size:
                self.flush_buffer()

    # Ensure that all remaining data is processed and saved
    def finish(self):
        # Even if the amount of data in the buffer does not reach 
        # chunk_size is also written to ensure that all data is saved
        self.flush_buffer(ignore_chunk=True)
        self.x_ds.resize( (self.end_idx,) )
        self.y_ds.resize( (self.end_idx,) )
        self.t_ds.resize( (self.end_idx,) )
        self.p_ds.resize( (self.end_idx,) )

        self.compute_ms_index_map()

    def primary_time_ds(self):
        raise NotImplementedError("This would lead to something very expensive being computed")

    # Used to generate timestamp index mapping
    def index_map(self, index_map_ds, time_ds):
        index_map_ds_cl = 0

        remaining_times = time_ds[...].copy()
        cur_loc = 0
        chunk_size = 10000000
        num_events = self.t_ds.shape[0]

        while remaining_times.shape[0] > 0 and cur_loc < num_events:
            end = min( num_events, cur_loc+chunk_size )
            idx = cur_loc + np.searchsorted(self.t_ds[cur_loc:end], remaining_times)

            idx_remaining = (idx == end)
            idx_found = (idx < end)

            index_map_ds[index_map_ds_cl:index_map_ds_cl+idx_found.sum()] = idx[idx_found]

            remaining_times = remaining_times[idx_remaining]
            cur_loc = cur_loc + chunk_size
            index_map_ds_cl += idx_found.sum()

    # Generate an index mapping to associate millisecond timestamps with indexes in the event dataset
    def compute_ms_index_map(self):
        time_ds = np.arange(0, int(np.floor(self.t_ds[-1] / 1e3)) ) * 1e3
        index_map_ds = self.group.create_dataset("ms_map_idx", time_ds.shape, dtype="u8")

        self.index_map( index_map_ds, time_ds )

    def compute_index_map(self, sensor_processor):
        time_ds = sensor_processor.primary_time_ds()
        name = 'prophesee_'+self.name_parts[1]+'_ms_map_idx'
        index_map_ds = sensor_processor.group.create_dataset(name, time_ds.shape, dtype="u8")

        self.index_map( index_map_ds, time_ds )


def compute_temporal_index_maps(event_processor, processor):
    print("Starting index map computation")
    event_processor.compute_index_map(processor)

def process_bag(bag_file, h5_file, bag_name):
    bag = rosbag.Bag(bag_file)
    h5fn = h5_file + '/'+ bag_name + '.hdf5'
    hdf5file = h5py.File(h5fn, "w")
    image_index_base = 9

    # BFS_RGB_processor = BFS_RGB(hdf5file, "/camera/image_color/compressed", bag, t=None, image_index = image_index_base)
    event_processor = Event(hdf5file, "/event_cam_left/events", bag, t=BFS_RGB_processor.image_index_time, image_index = image_index_base)
    # for topic, msg, t in tqdm(bag.read_messages(topics=[BFS_RGB_processor.topic])):
    #     BFS_RGB_processor.process(msg) 
    # BFS_RGB_processor.finish()


    for topic, msg, t in tqdm(bag.read_messages(topics=[event_processor.topic])):
        event_processor.process(msg)
    event_processor.finish()


    # Compute the temporal index maps for Image and Imu
    compute_temporal_index_maps(event_processor, BFS_RGB_processor)
    hdf5file.close()

    # Set permissions
    os.chmod(h5fn, 0o666)
    print(Fore.GREEN, "Output file: ", h5_file, Style.RESET_ALL)

if __name__ == "__main__":
    
    path = 'my_data/zyt/calibration_20241113.bag'
    bag_out_dir = 'my_data/zyt/calibration_20241113'
    bagname = 'aaa'
    
    process_bag(path, bag_out_dir, bagname)
    
    
    # start_time = time.time()

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--bag_path",
    #                     default=None,
    #                     help="ROS bag file path")
    # parser.add_argument("--output_dir",
    #                     default=None,
    #                     help="Folder where to extract the data")
    # args = parser.parse_args()

    # print(f"aaaaa {args.output_dir}")

    # if args.output_dir is None:
    #     args.output_dir = f"{os.path.dirname(args.bag_path)}/fcw_decoded"
    #     os.makedirs(args.output_dir, exist_ok=True)

    # print('Data will be extracted in folder: {}'.format(args.output_dir))
    # if os.path.isdir(args.bag_path):
    #     rosbag_paths = sorted(glob.glob(os.path.join(args.bag_path, "*.bag")))
    # else:
    #     rosbag_paths = [args.bag_path]
    # print('Found {} rosbags'.format(len(rosbag_paths)))
    # # Decode the rosbags
    # for path in rosbag_paths:
    #     bagname = os.path.splitext(os.path.basename(path))[0]
    #     bag_out_dir = os.path.join(args.output_dir, "{}".format(bagname))
    #     os.makedirs(bag_out_dir, exist_ok=True)
    #     os.chmod(bag_out_dir, 0o700)
    #     print("Extracting {} to {}".format(path, bag_out_dir))
    #     process_bag(path, bag_out_dir, bagname)

    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(Fore.BLUE,f"Cost time: {execution_time} S",Style.RESET_ALL)
    