//! Which seed-extension backend the run uses, chosen once per process.
//!
//! `BWA4_GPU=wgpu` (with the `gpu` feature) routes extension through the co-scheduler of
//! `bwa-gpu`: every worker's batch is shared between its own CPU kernel and one GPU service thread
//! that merges all workers' shares into shared launches. Anything else, or a machine with no usable
//! adapter, is the CPU kernel alone. The SAM is byte-identical in every case: which device runs an
//! extension job is scheduling, and each job is a pure function of its own inputs.
//!
//! Environment:
//! - `BWA4_GPU=wgpu` (alias `vulkan`): request the GPU.
//! - `BWA4_GPU_SPLIT=<0..1>`: fixed fraction of jobs sent to the GPU; unset means adaptive.
//! - `BWA4_GPU_SOFTWARE=1`: accept a CPU-emulated adapter (lavapipe). Exact but slow; for testing
//!   the GPU path on a machine without a GPU, which is how CI exercises it.

use bwa_neon::NeonBackend;

/// The extension backend of this run.
pub enum Backend {
    /// The CPU SIMD kernel alone.
    Cpu(NeonBackend),
    /// CPU and GPU sharing each batch.
    #[cfg(feature = "gpu")]
    Shared(bwa_gpu::CoSched<NeonBackend>),
}

/// The process-wide choice. Made on first use, from the environment; announces itself on stderr
/// only when a GPU was asked for, so a run that never mentions `BWA4_GPU` logs exactly as before.
pub fn backend() -> &'static Backend {
    static B: std::sync::OnceLock<Backend> = std::sync::OnceLock::new();
    B.get_or_init(choose)
}

#[cfg(not(feature = "gpu"))]
fn choose() -> Backend {
    if std::env::var_os("BWA4_GPU").is_some_and(|v| v != "off" && !v.is_empty()) {
        eprintln!("[M::gpu] BWA4_GPU ignored: this binary was built without the `gpu` feature");
    }
    Backend::Cpu(NeonBackend)
}

#[cfg(feature = "gpu")]
fn choose() -> Backend {
    use bwa_gpu::{CoSched, GpuRequest, GpuService, Split};
    use std::sync::Arc;
    match bwa_gpu::requested() {
        GpuRequest::Wgpu => {}
        GpuRequest::Off => return Backend::Cpu(NeonBackend),
        other => {
            eprintln!("[M::gpu] BWA4_GPU={other:?} is not available in this binary; CPU only");
            return Backend::Cpu(NeonBackend);
        }
    }
    let allow_sw = std::env::var_os("BWA4_GPU_SOFTWARE").is_some_and(|v| v == "1");
    let Some(dev) = bwa_wgpu::WgpuExtend::new(allow_sw) else {
        eprintln!(
            "[M::gpu] no usable wgpu adapter{}; CPU only",
            if allow_sw {
                ""
            } else {
                " (software adapters need BWA4_GPU_SOFTWARE=1)"
            }
        );
        return Backend::Cpu(NeonBackend);
    };
    let split = Split::from_env();
    let svc = GpuService::start(Arc::new(dev));
    eprintln!(
        "[M::gpu] seed extension shared with {} ; split {}",
        svc.device_name(),
        match split {
            Split::Fixed(f) => format!("{f}"),
            Split::Adaptive => "adaptive".into(),
        }
    );
    Backend::Shared(CoSched::new(NeonBackend, svc, split))
}

/// End-of-run line: how the work was actually divided. Silent on the CPU path.
pub fn dump() {
    #[cfg(feature = "gpu")]
    if let Backend::Shared(cs) = backend() {
        use std::sync::atomic::Ordering::Relaxed;
        let st = cs.service().stats();
        let (c, g) = cs.cells();
        let launches = st.launches.load(Relaxed).max(1);
        eprintln!(
            "[M::gpu] {} launches, {:.1} requests/launch, {:.1} k jobs/launch, device busy {:.2} s, \
             GPU share of cells {:.1} %, final split {:.3}, failures {}",
            st.launches.load(Relaxed),
            st.requests.load(Relaxed) as f64 / launches as f64,
            st.jobs.load(Relaxed) as f64 / launches as f64 / 1e3,
            st.busy_ns.load(Relaxed) as f64 / 1e9,
            100.0 * g as f64 / (c + g).max(1) as f64,
            cs.fraction(),
            st.failures.load(Relaxed),
        );
    }
}
