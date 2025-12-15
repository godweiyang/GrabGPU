#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <thread>
#include <cmath>

// 辅助宏
#define sleep_ms(t) std::this_thread::sleep_for(std::chrono::milliseconds((long)(t)))

const float bytes_per_gb = (1 << 30);
const float ms_per_hour = 1000 * 3600;
const int max_grid_dim = (1 << 15);
const int max_block_dim = 1024;
const int max_gpu_num = 32;

// 【关键修改 1】增加计算密度
// 增加一个循环，让每个线程多做一些无意义的计算，延长 Kernel 执行时间
// 这样可以掩盖 CPU 的 Launch Overhead
__global__ void default_script_kernel(char* array, size_t occupy_size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= occupy_size) return;
  
  // 循环次数：根据显卡性能不同，可能需要调整这个值
  // 1000次浮点运算通常能让 GPU 稍微忙一会儿，但不会太久
  // 这里混合了内存读写和计算
  float val = 0.0f;
  for (int k = 0; k < 2000; ++k) {
      val += k * 0.001f;
      // 为了防止编译器优化掉这个循环，我们偶尔写回内存
      if (k % 500 == 0) {
          array[i] = (char)(val); 
      }
  }
  array[i]++;
}

// 【关键修改 2】移除随机 Grid Size
// 始终使用最大 Grid，确保 GPU 核心被占满
void launch_default_script(char** array, size_t occupy_size,
                           std::vector<int>& grid_dim,
                           std::vector<int>& gpu_ids) {
  // 直接计算最大需要的 Grid，不再随机
  int gd = int((occupy_size + max_block_dim - 1) / max_block_dim);
  // 限制最大 Grid 维度，防止报错
  if (gd > max_grid_dim) gd = max_grid_dim;

  for (int id : gpu_ids) {
    cudaSetDevice(id);
    default_script_kernel<<<gd, max_block_dim, 0, NULL>>>(array[id],
                                                          occupy_size);
  }
}

void run_default_script(char** array, size_t occupy_size, float total_time,
                        std::vector<int>& gpu_ids, float utilization) {
  printf("Running default script with target utilization: %.2f%% >>>>>>>>>>>>>>>>>>>>\n", utilization * 100);
  srand(time(NULL));
  
  cudaEvent_t start_total, stop_total;
  cudaEventCreate(&start_total);
  cudaEventCreate(&stop_total);
  
  float elapsed_time_ms = 0;
  
  // Grid dim vector 实际上在这里没用了，因为我们在 launch 里重新计算了
  std::vector<int> grid_dim; 

  cudaEventRecord(start_total, 0);
  auto last_log_time = std::chrono::steady_clock::now();

  while (true) {
    auto t1 = std::chrono::high_resolution_clock::now();

    // 发起计算
    launch_default_script(array, occupy_size, grid_dim, gpu_ids);

    // 同步
    for (int id : gpu_ids) {
        cudaSetDevice(id);
        cudaDeviceSynchronize();
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    
    // 计算 GPU 实际工作时间 (毫秒)
    double on_time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // 动态睡眠控制
    if (utilization < 1.0f && on_time_ms > 0) {
        double off_time_ms = on_time_ms * (1.0f / utilization - 1.0f);
        
        // 只有当计算时间足够长，或者睡眠时间有意义时才睡眠
        // 如果 on_time_ms 太短（比如 < 1ms），说明 Kernel 还是太轻了，
        // 此时不睡眠，直接进入下一次循环以增加负载，直到积累了足够的时间（这里简化处理）
        if (off_time_ms > 1.0) { 
            sleep_ms(off_time_ms);
        }
    }

    cudaEventRecord(stop_total, 0);
    cudaEventSynchronize(stop_total);
    cudaEventElapsedTime(&elapsed_time_ms, start_total, stop_total);

    if (elapsed_time_ms / ms_per_hour > total_time) break;

    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time).count() > 10) {
        printf("Occupied time: %.2f hours (Last Kernel Duration: %.3f ms)\n", 
               elapsed_time_ms / ms_per_hour, on_time_ms);
        last_log_time = now;
    }
  }

  cudaEventDestroy(start_total);
  cudaEventDestroy(stop_total);
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
  sscanf(argv[4], "%f", &utilization);

  if (occupy_mem <= 0) {
    printf("GPU memory must be positive: %.2f\n", occupy_mem);
    throw std::invalid_argument("Invalid GPU memory");
  }
  if (total_time < 0) {
    printf("Occupied time must be positive: %.2f\n", total_time);
    throw std::invalid_argument("Invalid occupied time");
  }
  if (utilization <= 0.0f || utilization > 1.0f) {
      printf("Utilization must be in range (0.0, 1.0]: %.2f\n", utilization);
      throw std::invalid_argument("Invalid utilization");
  }

  occupy_size = occupy_mem * bytes_per_gb;
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
    for (int id = 0; id < gpu_ids.size(); ++id) {
      printf("%d%c", gpu_ids[id], ",\n"[id == gpu_ids.size() - 1]);
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
        cudaSetDevice(id);
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
