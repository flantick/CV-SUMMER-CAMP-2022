"""
Object detector based on Inference Engine
"""
import os
import sys

sys.path.append(r'/home/flantick/Downloads/open_model_zoo-master/demos/common/python')

import cv2
import numpy as np
from openvino.runtime import Core

import logging as log
import argparse
import pathlib
from time import perf_counter

from openvino.model_zoo.model_api.models import DetectionModel, DetectionWithLandmarks, RESIZE_TYPES, OutputTransform
from openvino.model_zoo.model_api.performance_metrics import PerformanceMetrics
from openvino.model_zoo.model_api.pipelines import get_user_config, AsyncPipeline
from openvino.model_zoo.model_api.adapters import create_core, OpenvinoAdapter
from images_capture import open_images_capture

from visualizers import ColorPalette


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an .xml \
        file with a trained model.', required=True, type=str)
    parser.add_argument('-i', '--input', help='Path to \
        image file', required=True, type=str)
    parser.add_argument('-d', '--device', help='Specify the target \
        device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
        Sample will look for a suitable plugin for device specified \
        (CPU by default)', default='CPU', type=str)
    parser.add_argument('-c', '--classes', help='File containing classes \
        names', type=str, default=None)
    parser.add_argument('-t', '--prob_threshold', default=0.5, type=float,
        help='Optional. Probability threshold for detections filtering.')
    return parser
  
  
def draw_detections(frame, detections, palette, labels, threshold):
    for detection in detections:
        class_id = int(detection.id)
        color = palette[class_id]

        det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
        xmin, ymin, xmax, ymax = detection.get_coords()
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                    (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

    return frame

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Start IE detection sample")

    # Initialize data input
    cap = open_images_capture(args.input, True)

    
    
    # Initialize OpenVINO
    ie = Core()

    
    # Initialize Plugin configs
    num_streams = "1"
    num_threads = 4
    plugin_config = get_user_config(args.device, num_streams, num_threads)
    model_adapter = OpenvinoAdapter(create_core(),
                                    args.model,
                                    device=args.device,
                                    plugin_config=plugin_config,
                                    max_num_requests=1,
                                    model_parameters={'input_layouts':None}
                                    )




    
    #Load SSD model
    model = DetectionModel.create_model('ssd', model_adapter)

    
    # Initialize pipeline
    detector_pipeline = AsyncPipeline(model)


    palette = ColorPalette(len(model.labels) if model.labels else 100)

    while True:

        # Get one image
        img = cap.read()

        # Start processing frame asynchronously
        frame_id = 0
        detector_pipeline.submit_data(img, frame_id, {'frame':img, 'start_time':0})
        detector_pipeline.await_any()
        results, meta = detector_pipeline.get_result(frame_id)
    
    
        # Draw detections in the image
        img = draw_detections(img, results, palette, None, args.prob_threshold)

    
        # Show image and wait for key press
        cv2.imshow("image wth dtcns", img)

        
        # Wait 1 ms and check pressed button to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
    # Destroy all windows
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    sys.exit(main())
