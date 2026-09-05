#pragma once

/**
 * ============================================================================
 * Software-Defined Silicon Cellular Computer (SDSCC)
 * 硬件级 CUDA 原生张量图运行时 (SDSCC CUDA Hardware Runtime Engine)
 * ============================================================================
 * 
 * 体系结构定位：
 * 1. 业务绝对正交 (Domain-Agnostic)：纯张量动力学，严禁业务专用名词
 * 2. 26 种原子物理原语全部用 __device__ __forceinline__ 固化在 GPU 设备端
 * 3. 彻底打破反向传播在物理芯片上的内存墙，直接在 RTX 5060 显存中执行
 * 4. 采用 NVRTC + CUDA Driver API 运行时零依赖 JIT 编译，便携且零外部编译依赖
 */

#include "kun/cellular/sdsc_compact_genome.hpp"
#include "kun/cellular/cuda_ops.cuh"
#include <cuda.h>
#include <nvrtc.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>

namespace kun {

#define SDSC_CUDA_CHECK(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char* err_str = nullptr; \
        cuGetErrorString(err, &err_str); \
        std::cerr << "[SDSC CUDA ERROR] " << (err_str ? err_str : "Unknown") \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    } \
} while(0)

class SdscCUDAGraph {
public:
    SdscCUDAGraph() {
        init_cuda_context();
        compile_cuda_kernel();
    }

    ~SdscCUDAGraph() {
        free_device_memory();
    }

    // 将紧凑 SoA 基因组上传至 GPU 显存
    bool upload(const CompactSoAGenome& g) {
        if (g.num_cells == 0) return false;
        num_cells_ = g.num_cells;
        num_synapses_ = g.num_synapses;
        in_dim_ = g.in_dim;
        out_dim_ = g.out_dim;

        free_device_memory();

        SDSC_CUDA_CHECK(cuMemAlloc(&d_op_types_, num_cells_ * sizeof(uint8_t)));
        SDSC_CUDA_CHECK(cuMemAlloc(&d_gains_, num_cells_ * sizeof(float)));
        SDSC_CUDA_CHECK(cuMemAlloc(&d_inc_off_, (num_cells_ + 1) * sizeof(uint32_t)));
        if (num_synapses_ > 0) {
            SDSC_CUDA_CHECK(cuMemAlloc(&d_inc_from_, num_synapses_ * sizeof(uint32_t)));
            SDSC_CUDA_CHECK(cuMemAlloc(&d_inc_weight_, num_synapses_ * sizeof(float)));
        }
        SDSC_CUDA_CHECK(cuMemAlloc(&d_out_cell_ids_, out_dim_ * sizeof(uint32_t)));

        // 状态寄存器
        SDSC_CUDA_CHECK(cuMemAlloc(&d_states_, num_cells_ * sizeof(float)));
        SDSC_CUDA_CHECK(cuMemAlloc(&d_aux_states_, num_cells_ * sizeof(float)));
        SDSC_CUDA_CHECK(cuMemAlloc(&d_cell_outputs_, num_cells_ * sizeof(float)));

        SDSC_CUDA_CHECK(cuMemAlloc(&d_in_tensor_, in_dim_ * sizeof(float)));
        SDSC_CUDA_CHECK(cuMemAlloc(&d_out_tensor_, out_dim_ * sizeof(float)));

        // 异步主机 -> 显存传输
        SDSC_CUDA_CHECK(cuMemcpyHtoD(d_op_types_, g.op_types.data(), num_cells_ * sizeof(uint8_t)));
        SDSC_CUDA_CHECK(cuMemcpyHtoD(d_gains_, g.gains.data(), num_cells_ * sizeof(float)));
        SDSC_CUDA_CHECK(cuMemcpyHtoD(d_inc_off_, g.inc_off.data(), (num_cells_ + 1) * sizeof(uint32_t)));
        if (num_synapses_ > 0) {
            SDSC_CUDA_CHECK(cuMemcpyHtoD(d_inc_from_, g.inc_from.data(), num_synapses_ * sizeof(uint32_t)));
            SDSC_CUDA_CHECK(cuMemcpyHtoD(d_inc_weight_, g.inc_weight.data(), num_synapses_ * sizeof(float)));
        }
        SDSC_CUDA_CHECK(cuMemcpyHtoD(d_out_cell_ids_, g.out_cell_ids.data(), out_dim_ * sizeof(uint32_t)));

        reset_device_states();
        return true;
    }

    void reset_device_states() {
        if (num_cells_ == 0) return;
        SDSC_CUDA_CHECK(cuMemsetD8(d_states_, 0, num_cells_ * sizeof(float)));
        SDSC_CUDA_CHECK(cuMemsetD8(d_aux_states_, 0, num_cells_ * sizeof(float)));
        SDSC_CUDA_CHECK(cuMemsetD8(d_cell_outputs_, 0, num_cells_ * sizeof(float)));
    }

    // 运行单步 CUDA 并发前向推演
    void forward(const float* h_in, float* h_out) {
        if (!h_in || !h_out || num_cells_ == 0) return;

        // 拷贝输入到显存
        SDSC_CUDA_CHECK(cuMemcpyHtoD(d_in_tensor_, h_in, in_dim_ * sizeof(float)));

        // 1. 输入感受层注入
        void* in_args[] = { &in_dim_, &d_in_tensor_, &d_cell_outputs_ };
        uint32_t block_in = 256;
        uint32_t grid_in = (in_dim_ + block_in - 1) / block_in;
        SDSC_CUDA_CHECK(cuLaunchKernel(kernel_input_inject_, grid_in, 1, 1, block_in, 1, 1, 0, 0, in_args, 0));

        // 2. Kahn 拓扑推演内核
        uint32_t block_eval = 256;
        uint32_t hidden_cells = num_cells_ - in_dim_;
        uint32_t grid_eval = (hidden_cells + block_eval - 1) / block_eval;
        void* eval_args[] = {
            &num_cells_, &in_dim_, &out_dim_,
            &d_op_types_, &d_gains_,
            &d_inc_off_, &d_inc_from_, &d_inc_weight_,
            &d_states_, &d_aux_states_, &d_cell_outputs_
        };
        SDSC_CUDA_CHECK(cuLaunchKernel(kernel_eval_, grid_eval, 1, 1, block_eval, 1, 1, 0, 0, eval_args, 0));

        // 3. 输出效应收集内核
        uint32_t block_out = 256;
        uint32_t grid_out = (out_dim_ + block_out - 1) / block_out;
        void* out_args[] = { &out_dim_, &num_cells_, &d_cell_outputs_, &d_out_tensor_, &d_out_cell_ids_ };
        SDSC_CUDA_CHECK(cuLaunchKernel(kernel_output_collect_, grid_out, 1, 1, block_out, 1, 1, 0, 0, out_args, 0));

        // 同步并取回结果
        SDSC_CUDA_CHECK(cuCtxSynchronize());
        SDSC_CUDA_CHECK(cuMemcpyDtoH(h_out, d_out_tensor_, out_dim_ * sizeof(float)));
    }

    uint32_t num_cells() const { return num_cells_; }
    uint32_t num_synapses() const { return num_synapses_; }

private:
    void init_cuda_context() {
        SDSC_CUDA_CHECK(cuInit(0));
        CUdevice dev;
        SDSC_CUDA_CHECK(cuDeviceGet(&dev, 0));
        SDSC_CUDA_CHECK(cuDevicePrimaryCtxRetain(&ctx_, dev));
        SDSC_CUDA_CHECK(cuCtxSetCurrent(ctx_));
    }

    void compile_cuda_kernel() {
        std::string cuda_source = std::string(R"(
        typedef unsigned char uint8_t;
        typedef unsigned int uint32_t;
)") + SDSC_CUDA_DEVICE_OPS_SRC + R"(
        extern "C" __global__ void kernel_input_inject(
            uint32_t in_dim,
            const float* __restrict__ in_tensor,
            float* __restrict__ cell_outputs
        ) {
            uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < in_dim) {
                cell_outputs[tid] = in_tensor[tid];
            }
        }

        extern "C" __global__ void kernel_eval(
            uint32_t num_cells,
            uint32_t in_dim,
            uint32_t out_dim,
            const uint8_t*  __restrict__ op_types,
            const float*    __restrict__ gains,
            const uint32_t* __restrict__ inc_off,
            const uint32_t* __restrict__ inc_from,
            const float*    __restrict__ inc_weight,
            float*          __restrict__ states,
            float*          __restrict__ aux_states,
            float*          __restrict__ cell_outputs
        ) {
            uint32_t idx = in_dim + blockDim.x * blockIdx.x + threadIdx.x;
            if (idx < num_cells) {
                uint32_t b = inc_off[idx];
                uint32_t e = inc_off[idx + 1];

                float sum_input = 0.0f;
                for (uint32_t k = b; k < e; ++k) {
                    uint32_t src = inc_from[k];
                    sum_input += cell_outputs[src] * inc_weight[k];
                }

                cell_outputs[idx] = sdsc_cuda_eval_primitive(
                    op_types[idx], gains[idx], sum_input,
                    &states[idx], &aux_states[idx]
                );
            }
        }

        extern "C" __global__ void kernel_output_collect(
            uint32_t out_dim,
            uint32_t num_cells,
            const float* __restrict__ cell_outputs,
            float*       __restrict__ out_tensor,
            const uint32_t* __restrict__ out_cell_ids
        ) {
            uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < out_dim) {
                uint32_t tgt = out_cell_ids[tid];
                out_tensor[tid] = (tgt < num_cells) ? cell_outputs[tgt] : 0.0f;
            }
        }
        )";

        nvrtcProgram prog;
        nvrtcCreateProgram(&prog, cuda_source.c_str(), "sdsc_cuda_kernels.cu", 0, NULL, NULL);
        const char* opts[] = {"--gpu-architecture=compute_89"};
        nvrtcResult res = nvrtcCompileProgram(prog, 1, opts);
        if (res != NVRTC_SUCCESS) {
            size_t log_size;
            nvrtcGetProgramLogSize(prog, &log_size);
            std::vector<char> log(log_size);
            nvrtcGetProgramLog(prog, log.data());
            std::cerr << "[NVRTC CUDA COMPILE ERROR]\n" << log.data() << std::endl;
        }

        size_t ptx_size;
        nvrtcGetPTXSize(prog, &ptx_size);
        std::vector<char> ptx(ptx_size);
        nvrtcGetPTX(prog, ptx.data());
        nvrtcDestroyProgram(&prog);

        SDSC_CUDA_CHECK(cuModuleLoadDataEx(&module_, ptx.data(), 0, 0, 0));
        SDSC_CUDA_CHECK(cuModuleGetFunction(&kernel_input_inject_, module_, "kernel_input_inject"));
        SDSC_CUDA_CHECK(cuModuleGetFunction(&kernel_eval_, module_, "kernel_eval"));
        SDSC_CUDA_CHECK(cuModuleGetFunction(&kernel_output_collect_, module_, "kernel_output_collect"));
    }

    void free_device_memory() {
        if (d_op_types_) { cuMemFree(d_op_types_); d_op_types_ = 0; }
        if (d_gains_) { cuMemFree(d_gains_); d_gains_ = 0; }
        if (d_inc_off_) { cuMemFree(d_inc_off_); d_inc_off_ = 0; }
        if (d_inc_from_) { cuMemFree(d_inc_from_); d_inc_from_ = 0; }
        if (d_inc_weight_) { cuMemFree(d_inc_weight_); d_inc_weight_ = 0; }
        if (d_out_cell_ids_) { cuMemFree(d_out_cell_ids_); d_out_cell_ids_ = 0; }
        if (d_states_) { cuMemFree(d_states_); d_states_ = 0; }
        if (d_aux_states_) { cuMemFree(d_aux_states_); d_aux_states_ = 0; }
        if (d_cell_outputs_) { cuMemFree(d_cell_outputs_); d_cell_outputs_ = 0; }
        if (d_in_tensor_) { cuMemFree(d_in_tensor_); d_in_tensor_ = 0; }
        if (d_out_tensor_) { cuMemFree(d_out_tensor_); d_out_tensor_ = 0; }
    }

    uint32_t num_cells_{0};
    uint32_t num_synapses_{0};
    uint32_t in_dim_{0};
    uint32_t out_dim_{0};

    CUcontext ctx_{0};
    CUmodule  module_{0};
    CUfunction kernel_input_inject_{0};
    CUfunction kernel_eval_{0};
    CUfunction kernel_output_collect_{0};

    CUdeviceptr d_op_types_{0};
    CUdeviceptr d_gains_{0};
    CUdeviceptr d_inc_off_{0};
    CUdeviceptr d_inc_from_{0};
    CUdeviceptr d_inc_weight_{0};
    CUdeviceptr d_out_cell_ids_{0};

    CUdeviceptr d_states_{0};
    CUdeviceptr d_aux_states_{0};
    CUdeviceptr d_cell_outputs_{0};
    CUdeviceptr d_in_tensor_{0};
    CUdeviceptr d_out_tensor_{0};
};

} // namespace kun
