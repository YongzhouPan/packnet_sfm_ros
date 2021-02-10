import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import torch
import time

from cv2 import imwrite

from packnet_sfm.utils.image import load_image
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth

ONNX_FILE_PATH = '/home/nvadmin/TensorRT-7.1.3.4/samples/python/onnx_packnet/model.onnx'

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()
logging_time = False

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

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    builder.max_workspace_size = 8 << 30

    # we have only one image in batch
    builder.max_batch_size = 1

    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True
    
        # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")

    return engine, context

def preprocess_data(input_file):

    image_shape = (288, 384)
    # Load image
    image = load_image(input_file)
    # Resize and to tensor
    image = resize_image(image, image_shape)
    image = to_tensor(image).unsqueeze(0)

    return image

# def postprocess(output_data):
    
def main():
    # initialize TensorRT engine and parse ONNX model
    engine, context = build_engine(ONNX_FILE_PATH)

    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)
    
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    for i in range(10):
        # preprocess input data
        rgb_image = preprocess_data("/home/nvadmin/TensorRT-7.1.3.4/samples/python/onnx_packnet/00000{}.png".format(i))
        host_input = rgb_image.numpy()
        cuda.memcpy_htod_async(device_input, host_input, stream)

        # run inference
        if logging_time:
            tic = time.time()

        context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)

        if logging_time:
            toc = time.time()
            curr_time = toc - tic
            print(i, curr_time)

        cuda.memcpy_dtoh_async(host_output, device_output, stream)

        stream.synchronize()

            

        # postprocess results
        output_data = torch.Tensor(host_output).reshape((1, 1, 288, 384))

        rgb = rgb_image[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        viz_pred_inv_depth = viz_inv_depth(output_data[0][0]) * 255

        save_image = np.concatenate([rgb, viz_pred_inv_depth], 0)
        imwrite("/home/nvadmin/TensorRT-7.1.3.4/samples/python/onnx_packnet/00000{}_out.png".format(i), save_image[:, :, ::-1])





if __name__ == '__main__':
    main()
    # output = preprocess_data("/home/nvadmin/TensorRT-7.1.3.4/samples/python/onnx_packnet/000000.png")
    # print(output.shape)
    # print(output.numpy().shape)