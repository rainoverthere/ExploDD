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
        '-i', '--input-event-file', dest='event_file_path', default="",#=======================================================================
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()
    # Events iterator on Camera
    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=100, start_ts=0)#每次100us#==========================================================
    # mv_iterator = EventsIterator(input_path=args.event_file_path, start_ts=5000, delta_t=1e3,
    #                              mode='delta_t')
    height, width = mv_iterator.get_size()  # Camera Geometry
    print(f'{height=} and {width=}')
   
    ev_idx=0
    empty_frame=0
    keep_empty_log=True

    # # with Window("Periodic frame generator", width, height, Window.RenderMode.BGR) as window:
    # # Do something whenever a frame is ready
    # def periodic_cb(ts, frame):
    #     # cv2.putText(frame, "Timestamp: " + str(ts), (0, 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
    #     print(ts)
    #     # print(frame)
    #     cv2.imshow('frame',frame)
    #     cv2.waitKey(33)
    #     # window.show(frame)

    # # Instantiate the frame generator
    # periodic_gen = PeriodicFrameGenerationAlgorithm(width, height, accumulation_time_us=2000, fps=2000)
    # periodic_gen.set_output_callback(periodic_cb)

    for evs in mv_iterator:
        ev_idx+=1
        #ev_idx代表这是ev_idx*100us这一帧
       
        # EventLoop.poll_and_dispatch()  # Dispatch system events to the window
        # periodic_gen.process_events(evs)  # Feed events to the frame generator
        # if window.should_close():
        #     break

        # print(evs.dtype) #查看dtype
        # break

        # if evs.shape[0]==0:
        #     # print(evs.shape[0])
        #     ev_idx+=1
        #     continue

        print(evs.dtype) if ev_idx == 1 else None
        # print(evs.shape[0]) if evs.shape[0] else None #真实序列的100us内产生的事件量基本都在20000以下
        print(evs.shape[0])
        if evs.shape[0]==0: empty_frame+=1
        # # t=evs['t'][:100]
        # # print(t)
        # t_min=evs['t'][0]
        # t_max=evs['t'][-1]
        # print(t_min)
        # print(t_max)
        # print('----------')
        
        # if ev_idx>=1: break
        print(f'{ev_idx=}')
        
    print(f'{empty_frame=}')

    # for evs in mv_iterator:
    #     # print("----- New event buffer! -----")
    #     if evs.size == 0:
    #         print("The current event buffer is empty.")
    #     else:
    #         # print(evs[:5])
    #         # print(evs[0])
    #         # print(evs.dtype)
    #         if np.min(evs['y']) != 0:
    #             continue
            
    #         print(f"The minimum y in this slice is {np.min(evs['y'])}")

    #         ev_idx+=1
    #         if ev_idx>=5: break



if __name__ == "__main__":
    main()