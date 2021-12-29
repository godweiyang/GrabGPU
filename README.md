# Grab the GPUs to run your own code!

## Download (downward compatibility)

**CUDA 10.1:**  
```shell
wget https://github.com/godweiyang/GrabGPU/releases/download/v1.0.0/gg_cu101
```

**CUDA 11.0:**  
```shell
wget https://github.com/godweiyang/GrabGPU/releases/download/v1.0.0/gg_cu110
```

**CUDA 11.2:**  
```shell
wget https://github.com/godweiyang/GrabGPU/releases/download/v1.0.0/gg_cu112
```

## Compile the source code

```shell
nvcc gg.cu -o gg
```

## Run default script
**Usage:**  
```shell
./gg <GPU Memory (GB)> <Occupied Time (h)> <GPU ID>
```

**Example:**  
Occupy 16 GB GPU memory for 24 hours using GPU 0, 1, 2, 3 to run default script.
```shell
./gg 16 24 0,1,2,3
```

## Run your own script

**Usage:**  
```shell
./gg <GPU Memory (GB)> <Occupied Time (h)> <GPU ID> <OPTIONAL: Script Path>
```

**Example:**  
Occupy 16 GB GPU memory using GPU 0, 1, 2, 3 to run your own `run.sh`. Note that the occupied time here is useless.
```shell
./gg 16 24 0,1,2,3 run.sh
```
