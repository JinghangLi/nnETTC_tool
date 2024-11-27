import h5py
from event_camera_py import Decoder
import numpy as np


class EventProcessor:
    UNITS_TO_SECONDS = 1e6

    def __init__(self, h5file, topic, bag, event_count):
        # super().__init__(h5file, topic, bag, time_calib)
        self.count = 0
        event_count = event_count
        self.chunk_size = 40000
        self.root = h5file
        if '/events' not in h5file:
            h5file.create_group('/events')
        self.group = h5file['/events']
    
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

        self.x_buf = []  #
        self.y_buf = []  #
        self.t_buf = []  #
        self.p_buf = []  #
        self.count_buf = 0

        self.start = 0
        self.prev_msg = None

        self.has_index_map = False

        self.decoder = Decoder()

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

    def add_events(self, x, y, t, p):
        msg_shape = x.shape[0]

        if msg_shape == 0:
            return

        self.x_buf.append(x)
        self.y_buf.append(y)
        self.t_buf.append(t)
        self.p_buf.append(p)
        self.count_buf += msg_shape

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

    def process(self, msg, topic=None, bag_time=None):
        self.decoder.decode_bytes(
            msg.encoding, msg.width, msg.height, msg.time_base, msg.events
        )
        cd_events = self.decoder.get_cd_events()
        trig_events = self.decoder.get_ext_trig_events()

        # t = self.sensor_to_global_us( cd_events['t'] ).astype(int)
        t = cd_events['t']
        has_started = t >= 0
        x = cd_events['x'][has_started]
        y = cd_events['y'][has_started]
        t = t[has_started]
        p = cd_events['p'][has_started]

        self.add_events(x, y, t, p)

        if self.count_buf > self.chunk_size:
            self.flush_buffer()

    def finish(self):
        self.flush_buffer(ignore_chunk=True)
        self.x_ds.resize( (self.end_idx,) )
        self.y_ds.resize( (self.end_idx,) )
        self.t_ds.resize( (self.end_idx,) )
        self.p_ds.resize( (self.end_idx,) )

        self.compute_ms_index_map()

    def primary_time_ds(self):
        raise NotImplementedError("This would lead to something very expensive being computed")

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

    def compute_ms_index_map(self):
        time_ds = np.arange(0, int(np.floor(self.t_ds[-1] / 1e3)) ) * 1e3
        index_map_ds = self.root.create_dataset("ms_to_idx", time_ds.shape, dtype="u8")

        self.index_map( index_map_ds, time_ds )

    # def compute_index_map(self, sensor_processor):
    #     time_ds = sensor_processor.primary_time_ds()
    #     index_map_name = time_ds.name.split("/")[-1] + "_map" + self.t_ds.name.replace("/", "_")
    #     index_map_ds = sensor_processor.group.create_dataset(index_map_name, time_ds.shape, dtype="u8")

    #     self.index_map( index_map_ds, time_ds )