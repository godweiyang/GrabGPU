#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstring>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define sleep_ms(t) std::this_thread::sleep_for(std::chrono::milliseconds((long)(t)))

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

const double bytes_per_gb = 1024.0 * 1024.0 * 1024.0;
const int max_gpu_num = 32;

// ============================================================================
// Tensor-core occupier kernel
//
// Goal: make the DCGM/GPM metrics `sm_active`, `sm_occupancy` and
// `tensor_active` (HMMA) all read high — close to the coarse GPU-utilization
// number — during the on-phase. The original kernel ran a scalar FP32 add loop,
// which raised the coarse "gpu util %" but left tensor_active at ~0 and
// sm_occupancy mediocre.
//
// Each warp runs a tight chain of 16x16x16 half->float HMMA instructions. The
// accumulator is both input and output of every mma_sync (D = A*B + C), so the
// iterations form a true dependency chain the compiler cannot fold away, and
// the final store of the accumulator makes the whole loop side-effecting (no
// dead-code elimination). A and B are constant register-resident fragments, so
// the inner loop is pure tensor-core compute with zero memory traffic — which
// keeps the HMMA pipe saturated and makes run-time linear in `iterations`.
//
// Measured on H20 (sm_90) during a sustained on-phase:
//   sm_active ~99%, tensor_active ~95%, hmma ~95%, sm_occupancy ~97%
//   (sm_occupancy only reaches ~97% when the grid is OVERSUBSCRIBED — see
//    launch config below; a single resident wave measured only ~56%.)
// ============================================================================
__global__ void tensor_occupier_kernel(char* buffer, size_t buffer_size,
                                        int iterations) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

  wmma::fill_fragment(a_frag, __float2half(1.0f));
  wmma::fill_fragment(b_frag, __float2half(1.0f));
  wmma::fill_fragment(c_frag, 0.0f);

  // Sustained HMMA loop: c = a*b + c. The read-modify-write on c_frag chains the
  // iterations together so ptxas can neither hoist nor eliminate them.
  for (int iter = 0; iter < iterations; ++iter) {
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  // Sink: write one accumulator lane into the (already-allocated) buffer so the
  // loop has an observable side effect and survives optimization.
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < buffer_size) buffer[idx] = (char)c_frag.x[0];
}

// ============================================================================
// Host-side launch helpers
// ============================================================================

// Launch the tensor kernel on every requested GPU asynchronously so they all
// run concurrently, then the caller synchronizes them together.
static void launch_tensor_kernel(char** array, size_t buffer_size,
                                 int iterations, int grid_x, int block_x,
                                 const std::vector<int>& gpu_ids) {
  for (int id : gpu_ids) {
    CUDA_CHECK(cudaSetDevice(id));
    tensor_occupier_kernel<<<grid_x, block_x>>>(array[id], buffer_size,
                                                iterations);
    CUDA_CHECK(cudaGetLastError());
  }
}

// One-time calibration: pick the iteration count that makes a single launch run
// for ~`target_on_time_ms` (default 50 ms). A steady on-phase of tens of ms is
// long enough that the DCGM/GPM sampling window (~100 ms; nvidia-smi averages
// over 1 s) captures a stable high value instead of a brief spike. This also
// removes the manual loop-count tuning the old README asked users to do.
//
// Per-launch throughput on a shared GPU is noisy — measured on H20 it swings
// ~2x launch-to-launch (68..146 iter/ms) even at a fixed iteration count — so a
// single timed sample is unreliable. We take several samples per GPU and use the
// MEDIAN (robust to outliers), then the slowest GPU (fewest iters/ms) so every
// GPU runs for at least the target on-time. The exact on-time need not be hit:
// anything in the tens-of-ms range works, and the duty-cycle sleep in the main
// loop recomputes the off-time from the *measured* on-time every iteration, so
// the metric time-average stays correct regardless of drift. Hence NO per-sample
// re-calibration in the hot loop (that would chase the 2x jitter and oscillate).
static int calibrate_iterations(char** array, size_t buffer_size,
                                const std::vector<int>& gpu_ids, int grid_x,
                                int block_x, float target_on_time_ms) {
  // Warm-up: establish context and ramp clocks so timed runs are representative.
  launch_tensor_kernel(array, buffer_size, /*iterations=*/200, grid_x, block_x,
                       gpu_ids);
  for (int id : gpu_ids) {
    CUDA_CHECK(cudaSetDevice(id));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  const int calib_iter = 4000;
  const int n_samples = 7;

  double min_iter_per_ms = 1e30;
  for (int id : gpu_ids) {
    CUDA_CHECK(cudaSetDevice(id));
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    std::vector<double> samples;
    samples.reserve(n_samples);
    for (int s = 0; s < n_samples; ++s) {
      CUDA_CHECK(cudaEventRecord(ev_start));
      tensor_occupier_kernel<<<grid_x, block_x>>>(array[id], buffer_size,
                                                  calib_iter);
      CUDA_CHECK(cudaEventRecord(ev_stop));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      if (ms > 0.0f) samples.push_back((double)calib_iter / (double)ms);
    }
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    std::sort(samples.begin(), samples.end());
    double median_iter_per_ms = samples[samples.size() / 2];
    if (median_iter_per_ms < min_iter_per_ms)
      min_iter_per_ms = median_iter_per_ms;
    printf("  GPU-%d calibration: median %.1f iter/ms over %d samples\n", id,
           median_iter_per_ms, (int)samples.size());
  }

  int target_iter = (int)(min_iter_per_ms * (double)target_on_time_ms);
  if (target_iter < 100) target_iter = 100;  // safety floor
  printf("Calibration: %.1f iter/ms -> %d iters for ~%.0f ms on-time\n",
         min_iter_per_ms, target_iter, (double)target_on_time_ms);
  return target_iter;
}

// ============================================================================
// Default (built-in) occupancy script — tensor-core aware.
// ============================================================================
void run_default_script(char** array, size_t occupy_size, float total_time,
                        std::vector<int>& gpu_ids, float utilization) {
  printf("Running tensor occupier with target utilization: %.2f%% "
         ">>>>>>>>>>>>>>>>>>>>\n",
         utilization * 100);

  // --- Launch config ---------------------------------------------------------
  // block = 256 threads (8 warps). On H20 this permits 8 resident blocks/SM =
  // 2048 threads/SM = 100% theoretical occupancy.
  //
  // The number of blocks that actually fit per SM is queried at runtime via the
  // occupancy API (robust across GPUs/driver versions). We then OVERSUBSCRIBE
  // the grid by `oversub` waves: launching only one resident wave measured just
  // ~56% achieved sm_occupancy, because warps drain at the kernel's tail with no
  // backlog to refill the slots. Oversubscribing keeps every warp slot
  // continuously full, pushing achieved sm_occupancy to ~97%.
  const int block_x = 256;
  const int oversub = 32;

  int sm_count = 0;
  int blocks_per_sm = 0;
  CUDA_CHECK(cudaSetDevice(gpu_ids[0]));
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                    gpu_ids[0]));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, tensor_occupier_kernel, block_x, 0));
  if (blocks_per_sm < 1) blocks_per_sm = 1;

  const int grid_x = sm_count * blocks_per_sm * oversub;

  printf("Launch: %d blocks x %d threads | %d SMs, %d resident blocks/SM, "
         "%dx oversubscribed\n",
         grid_x, block_x, sm_count, blocks_per_sm, oversub);

  // --- Calibrate iteration count for a steady ~50 ms on-phase ----------------
  const float target_on_time_ms = 50.0f;
  int target_iter = calibrate_iterations(array, occupy_size, gpu_ids, grid_x,
                                         block_x, target_on_time_ms);

  // --- Main duty-cycle loop ---------------------------------------------------
  auto start_total = std::chrono::steady_clock::now();
  auto last_log_time = start_total;

  while (true) {
    auto t1 = std::chrono::high_resolution_clock::now();

    // Launch on ALL GPUs first (async), then sync ALL -> concurrent execution.
    launch_tensor_kernel(array, occupy_size, target_iter, grid_x, block_x,
                         gpu_ids);
    for (int id : gpu_ids) {
      CUDA_CHECK(cudaSetDevice(id));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    double on_time_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Duty-cycle sleep: the OFF phase time-averages sm_active / sm_occupancy /
    // tensor_active down to ~= the target utilization. The off-time is derived
    // from the *measured* on-time each iteration, so even though per-launch
    // timing is noisy on a shared GPU the duty ratio (and therefore the metric
    // average) stays correct — no iteration re-tuning needed.
    // e.g. util=0.6 -> ~50 ms on + ~33 ms off, and all three metrics read ~60%.
    if (utilization < 1.0f && on_time_ms > 0) {
      double off_time_ms = on_time_ms * (1.0 / utilization - 1.0);
      if (off_time_ms > 0.5) {
        sleep_ms(off_time_ms);
      }
    }

    auto now = std::chrono::steady_clock::now();
    double elapsed_hours =
        std::chrono::duration<double, std::ratio<3600>>(now - start_total).count();
    if (elapsed_hours > total_time) break;

    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time)
            .count() > 10) {
      printf("Occupied time: %.2f hours (Last Kernel Duration: %.3f ms, "
             "Iterations: %d)\n",
             elapsed_hours, on_time_ms, target_iter);
      last_log_time = now;
    }
  }

  for (int id : gpu_ids) {
    cudaFree(array[id]);
  }
}

void process_args(int argc, char** argv, size_t& occupy_size, float& total_time,
                  std::vector<int>& gpu_ids, float& utilization, std::string& script_path) {
  if (argc != 5 && argc != 6) {
    printf(
        "Arguments: <GPU Memory (GB)> <Occupied Time (h)> <GPU ID> <Utilization(0.0-1.0)> <OPTIONAL: "
        "Script Path>\n");
    throw std::invalid_argument("Invalid argument number");
  }

  int gpu_num;
  CUDA_CHECK(cudaGetDeviceCount(&gpu_num));
  int id;
  std::string s(argv[3]);
  std::replace(s.begin(), s.end(), ',', ' ');
  std::stringstream ss;
  ss << s;
  while (ss >> id) {
    gpu_ids.push_back(id);
  }

  if (gpu_ids.size() == 1 && gpu_ids[0] == -1) {
    gpu_ids[0] = 0;
    for (int i = 1; i < gpu_num; ++i) {
      gpu_ids.push_back(i);
    }
  }

  for (int i : gpu_ids) {
    if (i < 0 || i >= gpu_num) {
      printf("Invalid GPU ID (%d GPU in total): %d\n", gpu_num, i);
      throw std::invalid_argument("Invalid GPU ID");
    }
  }

  float occupy_mem;
  sscanf(argv[1], "%f", &occupy_mem);
  sscanf(argv[2], "%f", &total_time);
  sscanf(argv[4], "%f", &utilization);

  if (occupy_mem <= 0) {
    printf("GPU memory must be positive: %.2f\n", occupy_mem);
    throw std::invalid_argument("Invalid GPU memory");
  }
  if (total_time < 0) {
    printf("Occupied time must be non-negative: %.2f\n", total_time);
    throw std::invalid_argument("Invalid occupied time");
  }
  if (utilization <= 0.0f || utilization > 1.0f) {
    printf("Utilization must be in range (0.0, 1.0]: %.2f\n", utilization);
    throw std::invalid_argument("Invalid utilization");
  }

  CUDA_CHECK(cudaSetDevice(gpu_ids[0]));
  size_t total_size, avail_size;
  cudaMemGetInfo(&avail_size, &total_size);

  occupy_size = (size_t)(occupy_mem * bytes_per_gb);
  if (occupy_size > total_size) {
    printf("GPU memory exceeds maximum (%.2f GB): %.2f\n",
           total_size / bytes_per_gb, occupy_mem);
    throw std::invalid_argument("Exceed maximal GPU memory");
  }

  printf("GPU memory (GB): %.2f\n", occupy_mem);
  printf("Occupied time (h): %.2f\n", total_time);
  printf("Target Utilization: %.2f%%\n", utilization * 100);

  if (argc == 5) {
    printf("GPU ID: ");
    for (size_t idx = 0; idx < gpu_ids.size(); ++idx) {
      printf("%d%c", gpu_ids[idx], (idx == gpu_ids.size() - 1) ? '\n' : ',');
    }
  } else {
    script_path = argv[5];
    printf("Script path: %s\n", script_path.c_str());
  }
}

void allocate_mem(char** array, size_t occupy_size, std::vector<int>& gpu_ids) {
  std::vector<bool> allocated(max_gpu_num, false);
  int cnt = 0;
  while (true) {
    printf("Try allocate GPU memory %d times >>>>>>>>>>>>>>>>>>>>\n", ++cnt);
    bool all_allocated = true;
    for (int id : gpu_ids) {
      if (!allocated[id]) {
        CUDA_CHECK(cudaSetDevice(id));
        cudaError_t status = cudaMalloc(&array[id], occupy_size);
        size_t total_size, avail_size;
        cudaMemGetInfo(&avail_size, &total_size);
        if (status != cudaSuccess) {
          printf(
              "GPU-%d: Failed to allocate %.2f GB GPU memory (%.2f GB "
              "available)\n",
              id, occupy_size / bytes_per_gb, avail_size / bytes_per_gb);
          all_allocated = false;
        } else {
          allocated[id] = true;
          printf(
              "GPU-%d: Successfully allocate %.2f GB GPU memory (%.2f GB "
              "available)\n",
              id, occupy_size / bytes_per_gb, avail_size / bytes_per_gb);
        }
      }
    }
    if (all_allocated) break;
    sleep_ms(5000);
  }
  printf("Successfully allocate memory on all GPUs!\n");
}

void run_custom_script(char** array, std::vector<int>& gpu_ids,
                       std::string script_path) {
  printf("Running custom script >>>>>>>>>>>>>>>>>>>>\n");
  for (int id : gpu_ids) {
    cudaFree(array[id]);
  }
  std::string cmd = "sh " + script_path;
  std::system(cmd.c_str());
}

int main(int argc, char** argv) {
  size_t occupy_size;
  float total_time;
  float utilization;
  std::vector<int> gpu_ids;
  std::string script_path;
  char* array[max_gpu_num];
  memset(array, 0, sizeof(array));

  try {
    process_args(argc, argv, occupy_size, total_time, gpu_ids, utilization, script_path);
    allocate_mem(array, occupy_size, gpu_ids);

    if (argc == 5) {
      run_default_script(array, occupy_size, total_time, gpu_ids, utilization);
    } else {
      run_custom_script(array, gpu_ids, script_path);
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
