# 抢占显卡脚本
[知乎介绍](https://zhuanlan.zhihu.com/p/449629487)

## 下载方法（向下兼容）

**CUDA 10.1:**  
```shell
wget https://github.com/godweiyang/GrabGPU/releases/download/v1.0.1/gg_cu101
```

**CUDA 11.0:**  
```shell
wget https://github.com/godweiyang/GrabGPU/releases/download/v1.0.1/gg_cu110
```

**CUDA 12.1:**  
```shell
wget https://github.com/godweiyang/GrabGPU/releases/download/v1.0.1/gg_cu121
```

## 如果你的 CUDA 版本不适配，请自行编译

```shell
nvcc gg.cu -o gg
```

## 抢占到显卡后自动执行默认脚本
**使用方法：**  
```shell
./gg <占用显存 (GB)> <占用时间 (h)> <显卡序号>
```

**举例：**  
抢占 16 GB 显存 24 小时，使用 GPU 0, 1, 2, 3 来运行默认脚本。
```shell
./gg 16 24 0,1,2,3
```

## 抢占到显卡后自动执行自定义程序（比如训练模型）
**使用方法：**  
```shell
./gg <占用显存 (GB)> <占用时间 (h)> <显卡序号> <自定义脚本路径（.sh文件）>
```

**举例：**  
抢占 16 GB 显存 24 小时，使用 GPU 0, 1, 2, 3 来运行自定义脚本 `run.sh`。注意这里的占用时间是无效的，会直到自定义脚本执行完毕才释放显卡。
```shell
./gg 16 24 0,1,2,3 run.sh
```
