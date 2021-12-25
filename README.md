# Grab the GPUs to run your own code!

## Download (downward compatibility)

**CUDA 11.0:**  
```
wget https://github.com/godweiyang/GrabGPU/blob/master/grab_gpu_cu110
```

**CUDA 11.2:**  
```
wget https://github.com/godweiyang/GrabGPU/blob/master/grab_gpu_cu112
```

## Compile the source code

**Dynamic compile:**  
```shell
nvcc grab_gpu.cu -o grab_gpu
```

**Static compile:**  
```shell
nvcc grab_gpu.cu -o grab_gpu -Xcompiler -static-libgcc -Xcompiler -static-libstdc++ -l:libcudadevrt.a -l:libcudart_static.a
```

## Run default script
**Usage:**  
```shell
./grab_gpu <GPU Memory (GB)> <Occupied Time (h)> <GPU ID>
```

**Example:**  
Occupy 16 GB GPU memory for 24 hours using GPU 0, 1, 2, 3 to run default script.
```shell
./grab_gpu 16 24 0,1,2,3
```

## Run your own script

**Usage:**  
```shell
./grab_gpu <GPU Memory (GB)> <Occupied Time (h)> <GPU ID> <OPTIONAL: Script Path>
```

**Example:**  
Occupy 16 GB GPU memory using GPU 0, 1, 2, 3 to run your own `run.sh`. Note that the occupied time here is useless.
```shell
./grab_gpu 16 24 0,1,2,3 run.sh
```
