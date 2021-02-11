import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import torch
import time

from cv2 import imwrite

from packnet_sfm.utils.image import load_image
from packnet_sfm.datasets.augmentations import resize_image, to_tensor

ONNX_FILE_PATH = '/home/nvadmin/packnet_ws/src/packnet_sfm_ros/ros/trt_packnet_sfm/src/packnet_kitti.onnx'
MODEL_NAME = "packnet_kitti.trt"
NET_INPUT_W = 192
NET_INPUT_H = 640
MAX_BATCH_SIZE = 1
MAX_GPU_MEM = 12

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()


def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # allow TensorRT to use up to MAX_GPU_MEM GB of GPU memory for tactic selection
    builder.max_workspace_size = MAX_GPU_MEM << 30

    # we have only one image in batch
    builder.max_batch_size = MAX_BATCH_SIZE

    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True
    
        # generate TensorRT engine optimized for the target platform
    print('Building an engine, this will take a while ...')
    tic = time.time()
    engine = builder.build_cuda_engine(network)
    toc = time.time()
    if engine is not None:
        print('Completed creating engine in {} s.'.format(toc - tic))

    return engine

def preprocess_data(input_file):

    image_shape = (NET_INPUT_H, NET_INPUT_W)
    # Load image
    image = load_image(input_file)
    # Resize and to tensor
    image = resize_image(image, image_shape)
    image = to_tensor(image).unsqueeze(0)

    return image

# def postprocess(output_data):
    
def main():
    # initialize TensorRT engine and parse ONNX model
    engine = build_engine(ONNX_FILE_PATH)
    if engine is None:
        raise SystemExit('ERROR: failed to build the TensorRT engine!')

    engine_path = MODEL_NAME
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    print('Serialized the TensorRT engine to file: %s' % engine_path)





if __name__ == '__main__':
    main()
    # output = preprocess_data("/home/nvadmin/TensorRT-7.1.3.4/samples/python/onnx_packnet/000000.png")
    # print(output.shape)
    # print(output.numpy().shape)