import os
import sys
import warnings

print("=== Environment Variables ===")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"CUDA_ROOT: {os.environ.get('CUDA_ROOT', 'Not set')}")
print(f"JAX_PLATFORM_NAME: {os.environ.get('JAX_PLATFORM_NAME', 'Not set')}")
print(f"XLA_PYTHON_CLIENT_PREALLOCATE: {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'Not set')}")

print("\n=== Testing JAX (as before) ===")
try:
    import jax
    import jax.numpy as jnp
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    
    # Simple test
    x = jnp.ones(1000)
    y = jnp.sum(x)
    print(f"JAX test result: {y}")
    print(f"JAX device: {x.device()}")
except Exception as e:
    print(f"JAX error: {e}")

print("\n=== Testing TensorFlow (likely source of warnings) ===")
try:
    # This will likely trigger those warnings you see
    print("Importing TensorFlow...")
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check GPU availability
    print("TensorFlow GPU devices:", tf.config.list_physical_devices('GPU'))
    
    # Test TensorFlow GPU computation
    with tf.device('/GPU:0'):
        tf_x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        tf_y = tf.matmul(tf_x, tf_x)
    print(f"TensorFlow GPU test result: {tf_y}")
    
    # Memory growth settings (might help with OOM)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("TensorFlow memory growth enabled")
        except RuntimeError as e:
            print(f"TensorFlow memory growth error: {e}")
    
except Exception as e:
    print(f"TensorFlow error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Testing Imports from V-GPS Training ===")
# Let's try importing what the training script likely imports
try:
    print("Testing V-GPS imports...")
    
    # These are likely imported by the training script
    import_tests = [
        "import jax",
        "import jax.numpy as jnp", 
        "import tensorflow as tf",
        "from flax import linen as nn",
        "import optax",
        "import numpy as np"
    ]
    
    for import_test in import_tests:
        try:
            exec(import_test)
            print(f"✓ {import_test}")
        except Exception as e:
            print(f"✗ {import_test} - {e}")
            
    # Try importing from V-GPS if possible
    try:
        sys.path.insert(0, '/V-GPS')
        import octo
        print("✓ Successfully imported octo")
    except Exception as e:
        print(f"✗ Could not import octo: {e}")
        
except Exception as e:
    print(f"Import testing error: {e}")

print("\n=== TensorRT Check ===")
try:
    # Check if TensorRT is available
    import tensorrt as trt
    print(f"TensorRT version: {trt.__version__}")
except ImportError:
    print("TensorRT not available (this might be causing the TF-TRT warning)")

print("\n=== CUDA Library Conflicts Check ===")
# The warnings suggest multiple registrations - let's check for conflicts
import subprocess

# Check what's trying to use CUDA
print("Checking loaded CUDA libraries...")
try:
    result = subprocess.run(['ldd', '/opt/conda/envs/vgps/lib/python3.10/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so'], 
                          capture_output=True, text=True, cwd='/V-GPS')
    if 'cuda' in result.stdout.lower():
        print("TensorFlow is linked against CUDA libraries")
    print(f"TensorFlow CUDA dependencies found: {result.returncode == 0}")
except Exception as e:
    print(f"Could not check TensorFlow CUDA linking: {e}")

print("\n=== Memory Information ===")

# Detailed CPU RAM Information
try:
    import psutil
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    print("CPU Memory:")
    print(f"  Total RAM: {memory.total / 1024**3:.1f} GB")
    print(f"  Available RAM: {memory.available / 1024**3:.1f} GB")
    print(f"  Used RAM: {memory.used / 1024**3:.1f} GB ({memory.percent:.1f}%)")
    print(f"  Free RAM: {memory.free / 1024**3:.1f} GB")
    print(f"  Buffer/Cache: {(memory.buffers + memory.cached) / 1024**3:.1f} GB")
    print(f"  Swap Total: {swap.total / 1024**3:.1f} GB")
    print(f"  Swap Used: {swap.used / 1024**3:.1f} GB ({swap.percent:.1f}%)")
    print(f"  Swap Free: {swap.free / 1024**3:.1f} GB")
    
    # Memory per CPU core (useful for parallel processing)
    cpu_count = psutil.cpu_count()
    print(f"  RAM per CPU core: {memory.total / cpu_count / 1024**3:.1f} GB ({cpu_count} cores)")
    
except ImportError:
    print("psutil not available, checking with /proc/meminfo")
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = {}
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(':')] = int(parts[1]) * 1024  # Convert from kB to bytes
        
        total_ram = meminfo.get('MemTotal', 0) / 1024**3
        available_ram = meminfo.get('MemAvailable', 0) / 1024**3
        free_ram = meminfo.get('MemFree', 0) / 1024**3
        
        print("CPU Memory (from /proc/meminfo):")
        print(f"  Total RAM: {total_ram:.1f} GB")
        print(f"  Available RAM: {available_ram:.1f} GB")
        print(f"  Free RAM: {free_ram:.1f} GB")
    except Exception as e:
        print(f"Could not get CPU memory info: {e}")

# GPU memory (enhanced)
print("\nGPU Memory:")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        memory_info = result.stdout.strip().split(',')
        total_mb, used_mb, free_mb = map(int, memory_info)
        print(f"  GPU Memory: {total_mb}MB total, {used_mb}MB used, {free_mb}MB free")
        print(f"  GPU Memory: {total_mb/1024:.1f}GB total, {used_mb/1024:.1f}GB used, {free_mb/1024:.1f}GB free")
        print(f"  GPU Utilization: {used_mb/total_mb*100:.1f}%")
        
        # Check if GPU memory is getting close to full
        if used_mb/total_mb > 0.8:
            print("  ⚠️  WARNING: GPU memory usage >80%")
        if free_mb < 1024:  # Less than 1GB free
            print("  ⚠️  WARNING: GPU memory low (<1GB free)")
            
except Exception as e:
    print(f"Could not get GPU memory info: {e}")

# Process memory information
print("\nCurrent Process Memory:")
try:
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"  RSS (Physical): {memory_info.rss / 1024**3:.2f} GB")
    print(f"  VMS (Virtual): {memory_info.vms / 1024**3:.2f} GB")
    print(f"  Memory %: {process.memory_percent():.1f}%")
except:
    print("  Could not get process memory info")

print("\n=== Training Environment Simulation Complete ===")
