#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <thread>

#define sleep(t) std::this_thread::sleep_for(std::chrono::milliseconds(t))

const float bytes_per_gb = (1 << 30);
const float ms_per_hour = 1000 * 3600;
const int max_grid_dim = (1 << 15);
const int max_block_dim = 1024;
const int max_sleep_time = 1e3;
const float sleep_interval = 1e16;
const int max_gpu_num = 32;

__global__ void default_script_kernel(char* array, size_t occupy_size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= occupy_size) return;
  array[i]++;
}

void launch_default_script(char** array, size_t occupy_size,
                           std::vector<cudaStream_t>& stream,
                           std::vector<int>& grid_dim,
                           std::vector<int>& gpu_ids) {
  int gd = std::min(grid_dim[rand() % grid_dim.size()],
                    int(occupy_size / max_block_dim));
  for (int id : gpu_ids) {
    cudaSetDevice(id);
    default_script_kernel<<<gd, max_block_dim, 0, stream[id]>>>(array[id],
                                                                occupy_size);
  }
}

void run_default_script(char** array, size_t occupy_size, float total_time,
                        std::vector<int>& gpu_ids) {
  srand(time(NULL));
  std::vector<cudaStream_t> stream(gpu_ids.size());
  for (int i = 0; i < gpu_ids.size(); ++i) {
    cudaStreamCreate(&stream[i]);
  }
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float time;
  size_t cnt = 0, sleep_time;
  std::vector<int> grid_dim;
  for (int i = 1; i <= max_grid_dim; i <<= 1) {
    grid_dim.push_back(i);
  }
  cudaEventRecord(start, 0);
  while (true) {
    launch_default_script(array, occupy_size, stream, grid_dim, gpu_ids);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    if (time / ms_per_hour > total_time) break;
    if (!((++cnt) % size_t(sleep_interval / occupy_size))) {
      cnt = 0;
      printf("Occupied time: %.2f hours\n", time / ms_per_hour);
      sleep_time = rand() % max_sleep_time + 1;
      sleep(sleep_time);
    }
  }
  for (int i = 0; i < gpu_ids.size(); ++i) {
    cudaStreamDestroy(stream[i]);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(array);
}

void process_args(int argc, char** argv, size_t& occupy_size, float& total_time,
                  std::vector<int>& gpu_ids, std::string& script_path) {
  if (argc != 4 && argc != 5) {
    printf(
        "Arguments: <GPU Memory (GB)> <Occupied Time (h)> <GPU ID> <OPTIONAL: "
        "Script Path>\n");
    throw std::invalid_argument("Invalid argument number");
  }
  if (argc == 5) {
    printf("Run custom script >>>>>>>>>>>>>>>>>>>>\n");
    script_path = argv[4];
  } else {
    printf("Run default script >>>>>>>>>>>>>>>>>>>>\n");
  }
  int gpu_num;
  cudaGetDeviceCount(&gpu_num);
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
      printf("Invalid GPU ID (%d GPU in total): %d\n", i, gpu_num);
      throw std::invalid_argument("Invalid GPU ID");
    }
  }

  float occupy_mem;
  size_t total_size, avail_size;
  cudaMemGetInfo(&avail_size, &total_size);
  sscanf(argv[1], "%f", &occupy_mem);
  sscanf(argv[2], "%f", &total_time);
  if (occupy_mem < 0) {
    printf("GPU memory must be positive: %.2f\n", occupy_mem);
    throw std::invalid_argument("Invalid GPU memory");
  }
  if (total_time < 0) {
    printf("Occupied time must be positive: %.2f\n", total_time);
    throw std::invalid_argument("Invalid occupied time");
  }
  occupy_size = occupy_mem * bytes_per_gb;
  if (occupy_size > total_size) {
    printf("GPU memory exceeds maximum (%.2f GB): %.2f\n",
           total_size / bytes_per_gb, occupy_mem);
    throw std::invalid_argument("Exceed maximal GPU memory");
  }

  printf("GPU memory (GB): %.2f\n", occupy_mem);
  printf("Occupied time (h): %.2f\n", total_time);
  if (argc == 4) {
    printf("GPU ID: ");
    for (int id = 0; id < gpu_ids.size(); ++id) {
      printf("%d%c", gpu_ids[id], ",\n"[id == gpu_ids.size() - 1]);
    }
  } else {
    printf("Script path: %s\n", script_path.c_str());
  }
}

void allocate_mem(char** array, size_t occupy_size, std::vector<int>& gpu_ids) {
  while (true) {
    bool all_allocated = true;
    for (int id : gpu_ids) {
      cudaSetDevice(id);
      size_t total_size, avail_size;
      cudaMemGetInfo(&avail_size, &total_size);
      cudaError_t status = cudaMalloc(&array[id], occupy_size);
      if (status != cudaSuccess) {
        printf(
            "GPU-%d: Failed to allocate %.2f GB GPU memory (%.2f GB "
            "available)\n",
            id, occupy_size / bytes_per_gb, avail_size / bytes_per_gb);
        all_allocated = false;
      } else {
        printf(
            "GPU-%d: Allocate %.2f GB GPU memory successfully (%.2f GB "
            "available)\n",
            id, occupy_size / bytes_per_gb, avail_size / bytes_per_gb);
      }
    }
    if (all_allocated) break;
    sleep(5000);
  }
}

void run_custom_script(char** array, std::vector<int>& gpu_ids,
                       std::string script_path) {
  for (int id : gpu_ids) {
    cudaFree(array[id]);
  }
  std::string cmd = "sh " + script_path;
  std::system(cmd.c_str());
}

int main(int argc, char** argv) {
  size_t occupy_size;
  float total_time;
  std::vector<int> gpu_ids;
  std::string script_path;
  char* array[max_gpu_num];

  process_args(argc, argv, occupy_size, total_time, gpu_ids, script_path);
  allocate_mem(array, occupy_size, gpu_ids);

  if (argc == 4) {
    run_default_script(array, occupy_size, total_time, gpu_ids);
  } else {
    run_custom_script(array, gpu_ids, script_path);
  }

  return 0;
}
