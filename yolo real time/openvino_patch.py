import os
import torch
from pathlib import Path

# Automatically patch OpenVINOBackend in Ultralytics YOLO to support custom multi-device configurations
try:
    from ultralytics.nn.backends.openvino import OpenVINOBackend
    import openvino as ov
    from ultralytics.utils import ARM64, LINUX, LOGGER

    def patched_load_model(self, weight: str | Path) -> None:
        LOGGER.info(f"Loading {weight} for OpenVINO inference (patched for MULTI device)...")
        core = ov.Core()
        # Enable disk caching to skip 30-second kernel compilation on subsequent runs
        cache_dir = Path(__file__).parent / "model_cache"
        cache_dir.mkdir(exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache_dir)})
        
        fallback_device = "CPU" if core.available_devices == ["CPU"] else "AUTO"
        
        device_name = fallback_device
        if isinstance(self.device, str):
            if self.device.startswith("intel:"):
                device_name = self.device[6:]
            else:
                device_name = self.device
            
            self.device = torch.device("cpu")
        
        w = Path(weight)
        if not w.is_file():
            w = next(w.glob("*.xml"))
        
        ov_model = core.read_model(model=str(w), weights=w.with_suffix(".bin"))
        if ov_model.get_parameters()[0].get_layout().empty:
            ov_model.get_parameters()[0].set_layout(ov.Layout("NCHW"))
        
        # Load metadata
        metadata_file = w.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML
            self.apply_metadata(YAML.load(metadata_file))
        
        # Set inference mode
        self.inference_mode = "CUMULATIVE_THROUGHPUT" if self.dynamic and self.batch > 1 else "LATENCY"
        config = {"PERFORMANCE_HINT": self.inference_mode}
        if LINUX and ARM64 and device_name == "CPU":
            config["EXECUTION_MODE_HINT"] = ov.properties.hint.ExecutionMode.ACCURACY
            config["INFERENCE_PRECISION_HINT"] = ov.Type.f32
        
        self.ov_compiled_model = core.compile_model(
            ov_model,
            device_name=device_name,
            config=config,
        )
        LOGGER.info(
            f"Using OpenVINO {self.inference_mode} mode for batch={self.batch} inference on "
            f"{', '.join(self.ov_compiled_model.get_property('EXECUTION_DEVICES'))}..."
        )
        self.input_name = self.ov_compiled_model.input().get_any_name()
        self.ov = ov

    OpenVINOBackend.load_model = patched_load_model
    LOGGER.info("OpenVINOBackend successfully patched to support MULTI device inference.")
except Exception as e:
    import sys
    print(f"Warning: Failed to patch OpenVINOBackend: {e}", file=sys.stderr)
