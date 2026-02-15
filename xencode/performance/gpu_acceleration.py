"""
GPU Acceleration Framework
Implements GPUAccelerationManager for AI workloads, CUDA and OpenCL support,
automatic GPU resource allocation, and GPU memory management and optimization.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import json
import secrets
import hashlib
from datetime import datetime
import numpy as np
import threading
import time
from dataclasses import dataclass
import queue
import gc


logger = logging.getLogger(__name__)


class GPUFramework(Enum):
    """Supported GPU frameworks."""
    CUDA = "cuda"
    OPENCL = "opencl"
    VULKAN = "vulkan"
    DIRECTML = "directml"
    METAL = "metal"


class GPUDeviceType(Enum):
    """Types of GPU devices."""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    GENERIC = "generic"


class GPUMemoryType(Enum):
    """Types of GPU memory."""
    GLOBAL = "global"
    SHARED = "shared"
    CONSTANT = "constant"
    TEXTURE = "texture"


@dataclass
class GPUDeviceInfo:
    """Information about a GPU device."""
    device_id: str
    name: str
    vendor: GPUDeviceType
    framework: GPUFramework
    compute_capability: str
    memory_size_mb: int
    memory_type: GPUMemoryType
    cores: int
    max_threads_per_block: int
    clock_speed_mhz: int
    is_available: bool
    utilization: float  # Current utilization percentage
    temperature: float  # Current temperature in Celsius
    power_draw: float   # Current power draw in watts
    metadata: Dict[str, Any]


@dataclass
class GPUAllocation:
    """Represents a GPU resource allocation."""
    allocation_id: str
    device_id: str
    requested_memory_mb: int
    allocated_memory_mb: int
    allocated_at: datetime
    allocated_by: str
    priority: int  # 1-5 scale, 5 being highest priority
    status: str  # pending, allocated, running, completed, failed
    metadata: Dict[str, Any]


@dataclass
class GPUWorkload:
    """Represents a workload to be executed on GPU."""
    workload_id: str
    task_type: str  # e.g., "matrix_multiplication", "neural_network_inference", "data_processing"
    data_size_mb: int
    estimated_runtime_ms: int
    priority: int  # 1-5 scale
    allocated_device: Optional[str]
    submitted_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    status: str  # queued, running, completed, failed, cancelled
    metrics: Dict[str, Any]  # Performance metrics


class GPUResourceManager:
    """Manages GPU resources and allocations."""
    
    def __init__(self):
        self.devices: Dict[str, GPUDeviceInfo] = {}
        self.allocations: Dict[str, GPUAllocation] = {}
        self.workload_queue: List[GPUWorkload] = []
        self.running_workloads: Dict[str, GPUWorkload] = {}
        self.lock = threading.Lock()
        
    def register_device(self, device_info: GPUDeviceInfo):
        """Register a GPU device."""
        with self.lock:
            self.devices[device_info.device_id] = device_info
            logger.info(f"Registered GPU device: {device_info.name} ({device_info.device_id})")
            
    def allocate_memory(self, device_id: str, requested_mb: int, allocated_by: str, priority: int = 3) -> Optional[GPUAllocation]:
        """Allocate GPU memory."""
        with self.lock:
            if device_id not in self.devices:
                logger.error(f"Device {device_id} not found")
                return None
                
            device = self.devices[device_id]
            
            # Calculate currently allocated memory
            allocated_memory = sum(
                alloc.allocated_memory_mb 
                for alloc in self.allocations.values() 
                if alloc.device_id == device_id and alloc.status in ["allocated", "running"]
            )
            
            available_memory = device.memory_size_mb - allocated_memory
            
            if available_memory < requested_mb:
                logger.warning(f"Not enough memory on {device_id}. Requested: {requested_mb}MB, Available: {available_memory}MB")
                return None
                
            allocation_id = f"alloc_{secrets.token_hex(8)}"
            
            allocation = GPUAllocation(
                allocation_id=allocation_id,
                device_id=device_id,
                requested_memory_mb=requested_mb,
                allocated_memory_mb=requested_mb,
                allocated_at=datetime.now(),
                allocated_by=allocated_by,
                priority=priority,
                status="allocated",
                metadata={"allocation_method": "manual"}
            )
            
            self.allocations[allocation_id] = allocation
            
            logger.info(f"Allocated {requested_mb}MB on {device_id} for {allocated_by}")
            return allocation
            
    def deallocate_memory(self, allocation_id: str):
        """Deallocate GPU memory."""
        with self.lock:
            if allocation_id in self.allocations:
                allocation = self.allocations[allocation_id]
                allocation.status = "completed"
                logger.info(f"Deallocated memory: {allocation_id}")
                
    def get_available_devices(self) -> List[GPUDeviceInfo]:
        """Get list of available GPU devices."""
        with self.lock:
            return [device for device in self.devices.values() if device.is_available]
            
    def get_device_utilization(self, device_id: str) -> float:
        """Get current utilization of a device."""
        with self.lock:
            if device_id in self.devices:
                return self.devices[device_id].utilization
            return 0.0


class CUDAManager:
    """Manages CUDA-specific operations."""
    
    def __init__(self):
        self.is_available = self._check_cuda_availability()
        self.cuda_version = self._get_cuda_version() if self.is_available else None
        self.devices = []
        
        if self.is_available:
            self._initialize_cuda_devices()
            
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import cupy
                return True  # Cupy can work with CUDA
            except ImportError:
                logger.warning("Neither PyTorch nor CuPy is available for CUDA operations")
                return False
        except Exception:
            logger.warning("CUDA not available")
            return False
            
    def _get_cuda_version(self) -> Optional[str]:
        """Get CUDA version."""
        try:
            import torch
            return torch.version.cuda
        except ImportError:
            try:
                import cupy
                return cupy.cuda.runtime.driverGetVersion().__str__()
            except Exception:
                return None
                
    def _initialize_cuda_devices(self):
        """Initialize CUDA devices."""
        try:
            import torch
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                device_info = GPUDeviceInfo(
                    device_id=f"cuda_{i}",
                    name=device_name,
                    vendor=self._get_vendor_from_name(device_name),
                    framework=GPUFramework.CUDA,
                    compute_capability=str(torch.cuda.get_device_capability(i)),
                    memory_size_mb=torch.cuda.get_device_properties(i).total_memory // (1024 * 1024),
                    memory_type=GPUMemoryType.GLOBAL,
                    cores=0,  # Would need specific CUDA API call
                    max_threads_per_block=0,  # Would need specific CUDA API call
                    clock_speed_mhz=0,  # Would need specific CUDA API call
                    is_available=True,
                    utilization=0.0,
                    temperature=0.0,
                    power_draw=0.0,
                    metadata={"compute_architecture": torch.cuda.get_device_capability(i)}
                )
                self.devices.append(device_info)
        except Exception as e:
            logger.error(f"Error initializing CUDA devices: {str(e)}")
            
    def _get_vendor_from_name(self, device_name: str) -> GPUDeviceType:
        """Determine vendor from device name."""
        device_name_lower = device_name.lower()
        if "nvidia" in device_name_lower or "geforce" in device_name_lower or "tesla" in device_name_lower:
            return GPUDeviceType.NVIDIA
        elif "amd" in device_name_lower or "radeon" in device_name_lower:
            return GPUDeviceType.AMD
        elif "intel" in device_name_lower:
            return GPUDeviceType.INTEL
        else:
            return GPUDeviceType.GENERIC
            
    def transfer_to_gpu(self, data: np.ndarray, device_id: str) -> Any:
        """Transfer data to GPU."""
        if not self.is_available:
            raise RuntimeError("CUDA is not available")
            
        try:
            import torch
            cuda_device = int(device_id.split("_")[1])  # Extract device number from "cuda_X"
            tensor = torch.from_numpy(data).to(f"cuda:{cuda_device}")
            return tensor
        except Exception as e:
            logger.error(f"Error transferring data to GPU: {str(e)}")
            raise
            
    def transfer_from_gpu(self, gpu_data: Any) -> np.ndarray:
        """Transfer data from GPU to CPU."""
        try:
            import torch
            if isinstance(gpu_data, torch.Tensor):
                return gpu_data.cpu().numpy()
            else:
                # Assume it's a CuPy array if not PyTorch
                import cupy
                if isinstance(gpu_data, cupy.ndarray):
                    return cupy.asnumpy(gpu_data)
        except Exception as e:
            logger.error(f"Error transferring data from GPU: {str(e)}")
            raise


class OpenCLManager:
    """Manages OpenCL-specific operations."""
    
    def __init__(self):
        self.is_available = self._check_opencl_availability()
        self.platforms = []
        self.devices = []
        
        if self.is_available:
            self._initialize_opencl()
            
    def _check_opencl_availability(self) -> bool:
        """Check if OpenCL is available."""
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            return len(platforms) > 0
        except ImportError:
            logger.warning("PyOpenCL is not available")
            return False
        except Exception:
            logger.warning("OpenCL not available")
            return False
            
    def _initialize_opencl(self):
        """Initialize OpenCL platforms and devices."""
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            
            for platform_idx, platform in enumerate(platforms):
                for device_idx, device in enumerate(platform.get_devices()):
                    device_info = GPUDeviceInfo(
                        device_id=f"opencl_{platform_idx}_{device_idx}",
                        name=device.name.strip(),
                        vendor=self._get_vendor_from_device(device.vendor),
                        framework=GPUFramework.OPENCL,
                        compute_capability=f"{device.version}",
                        memory_size_mb=device.global_mem_size // (1024 * 1024),
                        memory_type=GPUMemoryType.GLOBAL,
                        cores=device.max_compute_units,
                        max_threads_per_block=device.max_work_group_size,
                        clock_speed_mhz=getattr(device, 'max_clock_frequency', 0),
                        is_available=True,
                        utilization=0.0,
                        temperature=0.0,
                        power_draw=0.0,
                        metadata={
                            "vendor_id": device.vendor_id,
                            "driver_version": device.driver_version,
                            "extensions": [ext for ext in device.extensions.split() if ext]
                        }
                    )
                    self.devices.append(device_info)
        except Exception as e:
            logger.error(f"Error initializing OpenCL devices: {str(e)}")
            
    def _get_vendor_from_device(self, vendor_str: str) -> GPUDeviceType:
        """Determine vendor from OpenCL device vendor string."""
        vendor_lower = vendor_str.lower()
        if "nvidia" in vendor_lower:
            return GPUDeviceType.NVIDIA
        elif "amd" in vendor_lower or "advanced micro devices" in vendor_lower:
            return GPUDeviceType.AMD
        elif "intel" in vendor_lower:
            return GPUDeviceType.INTEL
        else:
            return GPUDeviceType.GENERIC


class GPUMemoryManager:
    """Manages GPU memory allocation and optimization."""
    
    def __init__(self):
        self.memory_pools = {}  # device_id -> pool of pre-allocated memory
        self.memory_usage = {}  # device_id -> current usage
        self.max_memory_usage = {}  # device_id -> max allowed usage
        self.optimization_enabled = True
        
    def initialize_memory_pool(self, device_id: str, pool_size_mb: int):
        """Initialize a memory pool for a device."""
        if self.optimization_enabled:
            # In a real implementation, this would pre-allocate memory on the GPU
            # For this demo, we'll just track the allocation
            self.memory_pools[device_id] = {
                "size_mb": pool_size_mb,
                "used_mb": 0,
                "free_mb": pool_size_mb
            }
            self.memory_usage[device_id] = 0
            self.max_memory_usage[device_id] = pool_size_mb
            
            logger.info(f"Initialized memory pool for {device_id}: {pool_size_mb}MB")
            
    def allocate_memory(self, device_id: str, size_mb: int) -> Optional[str]:
        """Allocate memory from the pool."""
        if device_id not in self.memory_pools:
            logger.error(f"No memory pool for device {device_id}")
            return None
            
        pool = self.memory_pools[device_id]
        
        if pool["free_mb"] < size_mb:
            logger.warning(f"Not enough free memory in pool for {device_id}. Requested: {size_mb}MB, Free: {pool['free_mb']}MB")
            return None
            
        # Allocate from pool
        allocation_id = f"mem_{secrets.token_hex(8)}"
        pool["used_mb"] += size_mb
        pool["free_mb"] -= size_mb
        self.memory_usage[device_id] += size_mb
        
        logger.debug(f"Allocated {size_mb}MB from pool for {device_id}, allocation_id: {allocation_id}")
        return allocation_id
        
    def deallocate_memory(self, device_id: str, allocation_id: str, size_mb: int):
        """Deallocate memory and return to pool."""
        if device_id in self.memory_pools:
            pool = self.memory_pools[device_id]
            pool["used_mb"] -= size_mb
            pool["free_mb"] += size_mb
            self.memory_usage[device_id] -= size_mb
            
            logger.debug(f"Deallocated {size_mb}MB to pool for {device_id}")
            
    def optimize_memory(self, device_id: str):
        """Optimize memory usage on a device."""
        if not self.optimization_enabled:
            return
            
        # In a real implementation, this would perform memory optimization
        # like garbage collection, memory defragmentation, etc.
        logger.info(f"Optimizing memory for {device_id}")
        
        # Force garbage collection
        gc.collect()
        
        # Reset memory pools if usage is too high
        if device_id in self.memory_pools:
            pool = self.memory_pools[device_id]
            usage_ratio = pool["used_mb"] / pool["size_mb"] if pool["size_mb"] > 0 else 0
            
            if usage_ratio > 0.9:  # If more than 90% of pool is used
                logger.info(f"Memory pool for {device_id} is highly utilized ({usage_ratio:.2%}), consider expansion")


class GPUAccelerationManager:
    """
    GPU acceleration manager for AI workloads with CUDA and OpenCL support,
    automatic GPU resource allocation, and memory management.
    """
    
    def __init__(self):
        self.resource_manager = GPUResourceManager()
        self.cuda_manager = CUDAManager()
        self.opencl_manager = OpenCLManager()
        self.memory_manager = GPUMemoryManager()
        self.workload_scheduler = None
        self.performance_monitors = {}
        self.framework_support = {
            GPUFramework.CUDA: self.cuda_manager.is_available,
            GPUFramework.OPENCL: self.opencl_manager.is_available
        }
        
        # Register available devices
        self._register_available_devices()
        
    def _register_available_devices(self):
        """Register all available GPU devices."""
        # Register CUDA devices
        for device in self.cuda_manager.devices:
            self.resource_manager.register_device(device)
            # Initialize memory pool for each device
            self.memory_manager.initialize_memory_pool(device.device_id, device.memory_size_mb // 2)
            
        # Register OpenCL devices
        for device in self.opencl_manager.devices:
            self.resource_manager.register_device(device)
            # Initialize memory pool for each device
            self.memory_manager.initialize_memory_pool(device.device_id, device.memory_size_mb // 2)
            
    def get_available_devices(self) -> List[GPUDeviceInfo]:
        """Get list of all available GPU devices."""
        return self.resource_manager.get_available_devices()
        
    def allocate_gpu_resources(
        self, 
        requested_memory_mb: int, 
        framework: GPUFramework = None,
        priority: int = 3,
        allocated_by: str = "system"
    ) -> Optional[Tuple[str, GPUAllocation]]:
        """
        Allocate GPU resources based on requirements.
        
        Returns:
            Tuple of (device_id, allocation) or None if allocation failed
        """
        available_devices = self.get_available_devices()
        
        if not available_devices:
            logger.error("No available GPU devices")
            return None
            
        # Filter by framework if specified
        if framework:
            available_devices = [dev for dev in available_devices if dev.framework == framework]
            
        if not available_devices:
            logger.error(f"No available GPU devices for framework {framework}")
            return None
            
        # Find the best device based on available memory and utilization
        best_device = None
        best_score = -1
        
        for device in available_devices:
            # Calculate a score based on available memory and current utilization
            available_memory = device.memory_size_mb - self.resource_manager.get_device_utilization(device.device_id) * device.memory_size_mb / 100
            
            if available_memory >= requested_memory_mb:
                # Prefer devices with lower utilization
                score = (available_memory / device.memory_size_mb) * (1 - device.utilization / 100)
                if score > best_score:
                    best_score = score
                    best_device = device
                    
        if not best_device:
            logger.error(f"No device has enough memory for request: {requested_memory_mb}MB")
            return None
            
        # Allocate memory
        allocation = self.resource_manager.allocate_memory(
            best_device.device_id, requested_memory_mb, allocated_by, priority
        )
        
        if allocation:
            # Also allocate from memory pool
            mem_allocation_id = self.memory_manager.allocate_memory(best_device.device_id, requested_memory_mb)
            if mem_allocation_id:
                allocation.metadata["memory_pool_allocation"] = mem_allocation_id
            else:
                logger.warning(f"Could not allocate from memory pool for {best_device.device_id}")
                
            return best_device.device_id, allocation
        else:
            return None
            
    def execute_on_gpu(
        self, 
        workload_type: str, 
        data: Union[np.ndarray, Any], 
        device_id: str = None,
        priority: int = 3
    ) -> str:
        """
        Execute a workload on GPU.
        
        Returns:
            Workload ID
        """
        workload_id = f"workload_{secrets.token_hex(8)}"
        
        # Determine data size
        if isinstance(data, np.ndarray):
            data_size_mb = data.nbytes / (1024 * 1024)
        else:
            # Estimate size for other data types
            data_size_mb = len(str(data).encode()) / (1024 * 1024)
            
        # If no device specified, find an appropriate one
        if not device_id:
            device_allocation = self.allocate_gpu_resources(
                int(data_size_mb * 1.5),  # Request 50% more for processing overhead
                priority=priority
            )
            if device_allocation:
                device_id = device_allocation[0]
            else:
                raise RuntimeError("Could not allocate GPU resources for workload")
                
        # Create workload
        workload = GPUWorkload(
            workload_id=workload_id,
            task_type=workload_type,
            data_size_mb=data_size_mb,
            estimated_runtime_ms=100,  # Placeholder
            priority=priority,
            allocated_device=device_id,
            submitted_at=datetime.now(),
            started_at=None,
            completed_at=None,
            status="queued",
            metrics={}
        )
        
        # Add to workload queue
        self.resource_manager.workload_queue.append(workload)
        
        logger.info(f"Queued workload {workload_id} on {device_id}")
        
        # Process the workload asynchronously
        asyncio.create_task(self._process_workload(workload, data))
        
        return workload_id
        
    async def _process_workload(self, workload: GPUWorkload, data: Any):
        """Process a workload on the GPU."""
        try:
            workload.status = "running"
            workload.started_at = datetime.now()
            
            device_id = workload.allocated_device
            
            # Transfer data to GPU
            if self.resource_manager.devices[device_id].framework == GPUFramework.CUDA:
                gpu_data = self.cuda_manager.transfer_to_gpu(data, device_id)
            else:
                # For this demo, we'll simulate OpenCL processing
                gpu_data = data  # Placeholder
                
            # Simulate GPU processing based on workload type
            result = await self._simulate_gpu_processing(workload.task_type, gpu_data)
            
            # Transfer result back from GPU
            if self.resource_manager.devices[device_id].framework == GPUFramework.CUDA:
                cpu_result = self.cuda_manager.transfer_from_gpu(result)
            else:
                cpu_result = result  # Placeholder
                
            workload.completed_at = datetime.now()
            workload.status = "completed"
            
            # Record performance metrics
            runtime = (workload.completed_at - workload.started_at).total_seconds() * 1000  # ms
            workload.metrics = {
                "runtime_ms": runtime,
                "throughput_mb_per_sec": workload.data_size_mb / (runtime / 1000) if runtime > 0 else 0,
                "device_utilization": self.resource_manager.get_device_utilization(device_id)
            }
            
            logger.info(f"Completed workload {workload.workload_id} in {runtime:.2f}ms")
            
        except Exception as e:
            workload.status = "failed"
            workload.completed_at = datetime.now()
            logger.error(f"Workload {workload.workload_id} failed: {str(e)}")
            
        finally:
            # Move from running to completed
            if workload.workload_id in self.resource_manager.running_workloads:
                del self.resource_manager.running_workloads[workload.workload_id]
                
    async def _simulate_gpu_processing(self, task_type: str, gpu_data: Any) -> Any:
        """Simulate GPU processing for different task types."""
        # Simulate processing time based on task type
        if task_type == "matrix_multiplication":
            # Simulate matrix multiplication
            if hasattr(gpu_data, 'shape') and len(gpu_data.shape) == 2:
                n = max(gpu_data.shape)
                # Simulate O(n^3) complexity
                processing_time = min(n * n * n / 1000000, 1.0)  # Cap at 1 second
                await asyncio.sleep(processing_time)
                # Return a dummy result
                import torch
                if isinstance(gpu_data, torch.Tensor):
                    return torch.matmul(gpu_data, gpu_data.T)
        elif task_type == "neural_network_inference":
            # Simulate neural network inference
            await asyncio.sleep(0.1)  # 100ms for inference
            # Return dummy result
            import torch
            if isinstance(gpu_data, torch.Tensor):
                return torch.sigmoid(gpu_data)
        else:
            # Default processing
            await asyncio.sleep(0.05)  # 50ms default
            return gpu_data
            
        return gpu_data  # Return original if no specific processing
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for GPU usage."""
        devices = self.get_available_devices()
        stats = {
            "total_devices": len(devices),
            "framework_support": self.framework_support,
            "device_utilization": {},
            "active_allocations": len([a for a in self.resource_manager.allocations.values() if a.status in ["allocated", "running"]]),
            "queued_workloads": len(self.resource_manager.workload_queue),
            "running_workloads": len(self.resource_manager.running_workloads),
            "completed_workloads": len([w for w in (self.resource_manager.workload_queue + list(self.resource_manager.running_workloads.values())) if w.status == "completed"])
        }
        
        for device in devices:
            stats["device_utilization"][device.device_id] = {
                "utilization": device.utilization,
                "memory_used_mb": device.memory_size_mb * device.utilization / 100,
                "temperature": device.temperature,
                "power_draw": device.power_draw
            }
            
        return stats
        
    def optimize_for_workload(self, workload_type: str) -> Optional[str]:
        """Optimize GPU selection for a specific workload type."""
        available_devices = self.get_available_devices()
        
        if not available_devices:
            return None
            
        # Different workloads may be better suited for different GPUs
        if workload_type in ["neural_network_training", "neural_network_inference"]:
            # Prefer newer GPUs with more memory for deep learning
            best_device = max(
                available_devices,
                key=lambda d: (d.memory_size_mb, d.cores if d.cores else 0),
                default=None
            )
        elif workload_type in ["matrix_multiplication", "linear_algebra"]:
            # Prefer GPUs with more cores for parallel computation
            best_device = max(
                available_devices,
                key=lambda d: (d.cores if d.cores else 0, d.clock_speed_mhz),
                default=None
            )
        else:
            # Default: pick device with most available memory
            best_device = max(
                available_devices,
                key=lambda d: d.memory_size_mb * (1 - d.utilization / 100),
                default=None
            )
            
        return best_device.device_id if best_device else None
        
    def monitor_device_temperature(self, device_id: str) -> float:
        """Monitor the temperature of a GPU device."""
        # In a real implementation, this would interface with GPU monitoring tools
        # For this demo, we'll simulate temperature readings
        import random
        simulated_temp = 45 + random.uniform(-5, 15)  # Base temp of 45°C ± variation
        return simulated_temp
        
    def cleanup_resources(self):
        """Clean up GPU resources."""
        # Deallocate all memory allocations
        for alloc_id in list(self.resource_manager.allocations.keys()):
            self.resource_manager.deallocate_memory(alloc_id)
            
        # Clear workload queues
        self.resource_manager.workload_queue.clear()
        self.resource_manager.running_workloads.clear()
        
        logger.info("Cleaned up GPU resources")


# Convenience function for easy use
def create_gpu_acceleration_manager() -> GPUAccelerationManager:
    """
    Convenience function to create a GPU acceleration manager.
    
    Returns:
        GPUAccelerationManager instance
    """
    return GPUAccelerationManager()