from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, OnDemandFrameGenerationAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, Window, UIAction, UIKeyEvent
import numpy as np
import cv2

def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision SDK Get Started sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-event-file', dest='event_file_path', default="",
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    import numpy as np
    from metavision_core.event_io.raw_reader import RawReader
    from matplotlib import pyplot as plt

    def create_image_from_events(events, height, width):
        img = np.full((height, width, 3), 128, dtype=np.uint8)
        img[events['y'], events['x']] = 255 * events['p'][:, None]
        return img

    raw_stream = RawReader(args.event_file_path)  # use empty string to open a camera
    height, width = raw_stream.get_size()
    print(f'{height=}{width=}')

    while not raw_stream.is_done():
        events = raw_stream.load_delta_t(500)
        im = create_image_from_events(events, height, width)
        cv2.imshow('frame',im)
        cv2.waitKey(33)
        # ev_idx+=1
        # if ev_idx>=41: break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()