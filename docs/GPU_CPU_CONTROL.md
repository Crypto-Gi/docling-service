# GPU/CPU Mode Control Guide

**Quick Start**: Use the helper scripts to switch modes instantly!

## Quick Mode Switching

### Switch to CPU Mode (Stable, Slower)
```bash
./set-cpu-mode.sh
docker compose restart
```

### Switch to GPU Mode (Faster, More Memory)
```bash
./set-gpu-mode.sh
docker compose restart
```

## Manual Configuration

Edit your `.env` file:

```bash
# Force CPU-only mode (recommended for stability)
DOCLING_GPU_PREFERRED=false

# OR enable GPU mode (faster but needs more memory)
DOCLING_GPU_PREFERRED=true
```

Then restart:
```bash
docker compose restart
```

## When to Use Each Mode

### Use CPU Mode When:
✅ Getting OOM errors (exit code 137)  
✅ GPU has less than 4GB VRAM  
✅ System has less than 8GB RAM  
✅ Processing very large documents (>100 pages)  
✅ Running other GPU-intensive applications  
✅ Stability is more important than speed  

### Use GPU Mode When:
✅ GPU has 6GB+ VRAM  
✅ System has 16GB+ RAM  
✅ Processing many small documents  
✅ Speed is critical  
✅ GPU is dedicated to this service  

## Performance Comparison

| Document Size | CPU Mode | GPU Mode | Speed Difference |
|--------------|----------|----------|------------------|
| 10 pages     | ~15s     | ~5s      | 3x faster        |
| 50 pages     | ~90s     | ~20s     | 4.5x faster      |
| 100 pages    | ~180s    | ~40s     | 4.5x faster      |

**Note**: GPU mode is faster but requires more memory. CPU mode is slower but more stable.

## Your Current Setup

Based on `nvtop` output:
- **GPU**: NVIDIA GeForce GTX 1650 SUPER
- **GPU VRAM**: 4GB
- **Current Mode**: CPU (forced via .env)

### Recommendation for Your Hardware

**Use CPU mode** for now because:
- 4GB VRAM is the minimum for GPU mode
- Docling's AI models can use 3-4GB during processing
- CPU mode is more stable and won't crash

**Try GPU mode** if:
- No other applications are using the GPU
- You're processing small documents (<20 pages)
- You want to test if it works for your use case

## Troubleshooting

### Container Keeps Crashing (Exit 137)
**Solution**: Use CPU mode
```bash
./set-cpu-mode.sh
docker compose restart
```

### Processing is Too Slow
**Try GPU mode** (if you have enough memory):
```bash
./set-gpu-mode.sh
docker compose restart
```

### GPU Mode Works Sometimes, Crashes Other Times
**Solution**: Stick with CPU mode for reliability, or:
1. Close other GPU applications
2. Process smaller documents
3. Increase Docker memory limit

## Checking Current Mode

```bash
# Check .env setting
grep DOCLING_GPU_PREFERRED .env

# Check container logs for device selection
docker logs docling-service-docling-1 | grep -i "device\|cuda\|cpu"
```

## Advanced: Hybrid Approach

The service automatically falls back to CPU if GPU runs out of memory, but you can force CPU mode to avoid the retry overhead:

```bash
# .env file
DOCLING_GPU_PREFERRED=false  # Start with CPU, no GPU attempts
```

This is **recommended for production** to ensure consistent performance.

## Summary

| Setting | Speed | Stability | Memory Required |
|---------|-------|-----------|-----------------|
| `DOCLING_GPU_PREFERRED=false` | Slower | ✅ High | 4GB RAM |
| `DOCLING_GPU_PREFERRED=true` | Faster | ⚠️ Medium | 8GB RAM + 4GB VRAM |

**Current recommendation**: Keep CPU mode (`false`) for your GTX 1650 SUPER setup.
