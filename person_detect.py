
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore, ExecutableNetwork
import os
import cv2
import argparse
import sys
import math

class Queue:
    '''
    Class for dealing with queues
    '''

    def __init__(self):
        self.queues = []

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max = q
            frame = image[y_min:y_max, x_min:x_max]
            yield frame

    def check_coords(self, coords):
        d = {k + 1: 0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0] > q[0] and coord[2] < q[2]:
                    d[i + 1] += 1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.threshold = threshold
        self.ie = IECore()
        self.exec_net = None
        try:
            self.model = self.ie.read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: This method needs to be completed by you
        '''
        self.exec_net = self.ie.load_network(network=self.model, device_name=self.device, num_requests=1)


    def predict(self, image):
        '''
        TODO: This method needs to be completed by you
        '''
        processed_image = self.preprocess_input(image)
        input_dict = {self.input_name: processed_image}
        # # Start asynchronous inference for specified request.
        infer_request_handle = self.exec_net.start_async(request_id=0, inputs=input_dict)
        infer_status = infer_request_handle.wait()
        outputs = None
        # Request status code: OK or RESULT_NOT_READY
        if infer_status == 0:
            outputs = infer_request_handle.outputs[self.output_name]
        coords = self.preprocess_outputs(outputs)
        output_image = self.draw_outputs(coords, image)
        return coords, output_image



    def draw_outputs(self, coords, image):
        '''
        TODO: This method needs to be completed by you
        '''
        out_image = np.copy(image)
        height, width, _ = np.shape(image)
        for box in coords:
            xmin, ymin, xmax, ymax = box
            xmin = math.floor(xmin * width)
            ymin = math.floor(ymin * height)
            xmax = math.floor(xmax * width)
            ymax = math.floor(ymax * height)
            cv2.rectangle(out_image, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
        return out_image



    '''
    Outputs
    The net outputs blob with shape: [1, 1, N, 7], where N is the number of detected bounding boxes. For each detection, 
    the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
    image_id - ID of the image in the batch
    label - predicted class ID
    conf - confidence for the predicted class
    (x_min, y_min) - coordinates of the top left bounding box corner
    (x_max, y_max) - coordinates of the bottom right bounding box corner.
    '''
    def preprocess_outputs(self, outputs):
        '''
        TODO: This method needs to be completed by you
        '''
        coords = []
        outputs = np.squeeze(outputs)
        for box in outputs:
            _, _, conf, x_min, y_min, x_max, y_max = box
            if conf > self.threshold:
                coords.append([x_min, y_min, x_max, y_max])
        return coords


    '''
    Inputs
    Name: input, shape: [1x3x320x544] - An input image in the format [BxCxHxW], where:    
    B - batch size
    C - number of channels
    H - image height
    W - image width
    Expected color order is BGR.
    '''
    def preprocess_input(self, image):
        '''
        TODO: This method needs to be completed by you
        '''
        n, c, h, w = self.input_shape
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        return image



def main(args):
    model = args.model
    device = args.device
    video_file = args.video
    max_people = args.max_people
    threshold = args.threshold
    output_path = args.output_path

    start_model_load_time = time.time()
    pd = PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time
    queue = Queue()
    try:
        queue_param = np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap = cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: " + video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps,
                                (initial_w, initial_h), True)
    #
    counter = 0
    start_inference_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            counter += 1
            coords, image = pd.predict(frame)
            num_people = queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text = ""
            y_pixel = 25
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text = ""
                y_pixel += 40
            out_video.write(image)
        total_time = time.time() - start_inference_time
        total_inference_time = round(total_time, 1)
        fps = counter / total_inference_time
        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time) + '\n')
            f.write(str(fps) + '\n')
            f.write(str(total_model_load_time) + '\n')
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)