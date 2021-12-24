# Grab the GPUs to run your own code!

## Run default script
**Usage:**. 
```shell
./grab_gpu <GPU Memory (GB)> <Occupied Time (h)> <GPU ID>
```

**Example:**  
Occupy 16 GB GPU memory for 24 hours using GPU 0, 1, 2, 3 to run default script.
```shell
./grab_gpu 16 24 0,1,2,3
```

## Run default script

**Usage:**. 
```shell
./grab_gpu <GPU Memory (GB)> <Occupied Time (h)> <GPU ID> <OPTIONAL: Script Path>
```

**Example:**  
Occupy 16 GB GPU memory using GPU 0, 1, 2, 3 to run your own `run.sh`. Note that the occupied time here is useless.
```shell
./grab_gpu 16 24 0,1,2,3 run.sh
```

