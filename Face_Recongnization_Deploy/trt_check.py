import pycuda.driver as cuda
import tensorrt as trt
import pycuda.autoinit

if __name__ == '__main__':
    model_path = "engine.trt"

    with open(model_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    # cuda_mem = cuda.mem_alloc(1)
    del context
    del engine
    del runtime