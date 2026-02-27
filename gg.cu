#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstring>

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
const int max_grid_dim = (1 << 15);
const int max_block_dim = 1024;
const int max_gpu_num = 32;

__global__ void default_script_kernel(char* array, size_t occupy_size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= occupy_size) return;

  float val = 0.0f;
  for (int k = 0; k < 2000; ++k) {
    val += k * 0.001f;
    if (k % 500 == 0) {
      array[i] = (char)(val);
    }
  }
  array[i]++;
}

void launch_default_script(char** array, size_t occupy_size,
                           std::vector<int>& gpu_ids) {
  int gd = (int)((occupy_size + max_block_dim - 1) / max_block_dim);
  if (gd > max_grid_dim) gd = max_grid_dim;

  for (int id : gpu_ids) {
    CUDA_CHECK(cudaSetDevice(id));
    default_script_kernel<<<gd, max_block_dim>>>(array[id], occupy_size);
    CUDA_CHECK(cudaGetLastError());
  }
}

void run_default_script(char** array, size_t occupy_size, float total_time,
                        std::vector<int>& gpu_ids, float utilization) {
  printf("Running default script with target utilization: %.2f%% >>>>>>>>>>>>>>>>>>>>\n",
         utilization * 100);

  auto start_total = std::chrono::steady_clock::now();
  auto last_log_time = start_total;

  while (true) {
    auto t1 = std::chrono::high_resolution_clock::now();

    launch_default_script(array, occupy_size, gpu_ids);

    for (int id : gpu_ids) {
      CUDA_CHECK(cudaSetDevice(id));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    double on_time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    if (utilization < 1.0f && on_time_ms > 0) {
      double off_time_ms = on_time_ms * (1.0 / utilization - 1.0);
      if (off_time_ms > 1.0) {
        sleep_ms(off_time_ms);
      }
    }

    auto now = std::chrono::steady_clock::now();
    double elapsed_hours =
        std::chrono::duration<double, std::ratio<3600>>(now - start_total).count();

    if (elapsed_hours > total_time) break;

    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time).count() > 10) {
      printf("Occupied time: %.2f hours (Last Kernel Duration: %.3f ms)\n",
             elapsed_hours, on_time_ms);
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
