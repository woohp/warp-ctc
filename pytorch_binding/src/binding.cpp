#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <numeric>
namespace py = pybind11;

#include "ctc.h"

#ifdef WARPCTC_ENABLE_GPU
    #include "THC.h"
    #include "THCTensor.h"
    #include "detail/reduce.h"
    extern THCState* state;
#else
    #include "TH.h"
#endif

int cpu_ctc(size_t _probs,
            size_t _grads,
            size_t _labels,
            size_t _label_sizes,
            size_t _sizes,
            int minibatch_size,
            size_t _costs) {

    THFloatTensor* probs = reinterpret_cast<THFloatTensor*>(_probs);
    THFloatTensor* grads = reinterpret_cast<THFloatTensor*>(_grads);
    THIntTensor* labels = reinterpret_cast<THIntTensor*>(_labels);
    THIntTensor* label_sizes = reinterpret_cast<THIntTensor*>(_label_sizes);
    THIntTensor* sizes = reinterpret_cast<THIntTensor*>(_sizes);
    THFloatTensor* costs = reinterpret_cast<THFloatTensor*>(_costs);

    float *probs_ptr = probs->storage->data + probs->storageOffset;
    float *grads_ptr;
    if (grads->storage) {
        grads_ptr = grads->storage->data + grads->storageOffset;
    } else {
        grads_ptr = NULL; // this will trigger the score forward code path
    }

    int *sizes_ptr = sizes->storage->data + sizes->storageOffset;
    int *labels_ptr = labels->storage->data + labels->storageOffset;
    int *label_sizes_ptr = label_sizes->storage->data + label_sizes->storageOffset;
    float *costs_ptr = costs->storage->data + costs->storageOffset;

    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.loc = CTC_CPU;
    options.num_threads = 0; // will use default number of threads
    options.blank_label = probs->size[2] - 1;

#if defined(CTC_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
#endif

    size_t cpu_size_bytes;
    get_workspace_size(label_sizes_ptr, sizes_ptr,
                       (int) probs->size[2], minibatch_size,
                       options, &cpu_size_bytes);

    float* cpu_workspace = (float*) new unsigned char[cpu_size_bytes];

    compute_ctc_loss(probs_ptr, grads_ptr,
                     labels_ptr, label_sizes_ptr,
                     sizes_ptr, probs->size[2],
                     minibatch_size, costs_ptr,
                     cpu_workspace, options);

    delete cpu_workspace;
    return 1;
}

#ifdef WARPCTC_ENABLE_GPU
int gpu_ctc(size_t _probs,
            size_t _grads,
            size_t _labels,
            size_t _label_sizes,
            size_t _sizes,
            int minibatch_size,
            size_t _costs) {

    THCudaTensor* probs = reinterpret_cast<THCudaTensor*>(_probs);
    THCudaTensor* grads = reinterpret_cast<THCudaTensor*>(_grads);
    THIntTensor* labels = reinterpret_cast<THIntTensor*>(_labels);
    THIntTensor* label_sizes = reinterpret_cast<THIntTensor*>(_label_sizes);
    THIntTensor* sizes = reinterpret_cast<THIntTensor*>(_sizes);
    THFloatTensor* costs = reinterpret_cast<THFloatTensor*>(_costs);

    float *probs_ptr = probs->storage->data + probs->storageOffset;
    float *grads_ptr;
    if (grads->storage) {
        grads_ptr = grads->storage->data + grads->storageOffset;
    } else {
        grads_ptr = NULL; // this will trigger the score forward code path
    }

    int *sizes_ptr = sizes->storage->data + sizes->storageOffset;
    int *labels_ptr = labels->storage->data + labels->storageOffset;
    int *label_sizes_ptr = label_sizes->storage->data + label_sizes->storageOffset;
    float *costs_ptr = costs->storage->data + costs->storageOffset;

    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.loc = CTC_GPU;
    options.stream = THCState_getCurrentStream(state);
    options.blank_label = probs->size[2] - 1;

    size_t gpu_size_bytes;
    get_workspace_size(label_sizes_ptr, sizes_ptr,
                       (int) probs->size[2], minibatch_size,
                       options, &gpu_size_bytes);

    float* gpu_workspace;
    THCudaMalloc(state, (void **) &gpu_workspace, gpu_size_bytes);

    compute_ctc_loss(probs_ptr, grads_ptr,
                     labels_ptr, label_sizes_ptr,
                     sizes_ptr, probs->size[2],
                     minibatch_size, costs_ptr,
                     gpu_workspace, options);

    THCudaFree(state, (void *) gpu_workspace);
    return 1;
}
#endif


PYBIND11_MODULE(_warp_ctc, m)
{
    m.def("cpu_ctc", &cpu_ctc);
    m.def("gpu_ctc", &gpu_ctc);
}
