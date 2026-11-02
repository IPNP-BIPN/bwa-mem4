//! Portable GPU backend for seed extension (`ksw_extend2`), in WGSL through wgpu.
//!
//! # Why a portable backend next to the Metal one
//!
//! The Metal backend (`bwa-metal`) proved the kernel byte-identical on Apple Silicon, but the
//! machines that run production WGS are x86 Linux servers with discrete NVIDIA or AMD cards, and
//! the CUDA backend of issue #56 was never written for want of a machine to test it on. WGSL
//! compiled by wgpu reaches all of them through Vulkan (and Apple through Metal, Windows through
//! DX12) from ONE kernel source, compiled by the driver at run time: no CUDA toolkit, no Xcode, no
//! build script.
//!
//! It also has a property no vendor backend has: Mesa's **lavapipe** is a conforming Vulkan driver
//! that runs on the CPU. So the byte-identity gates of this crate execute the real shader, compiled
//! by the real SPIR-V path, on any Linux CI runner with `mesa-vulkan-drivers` installed. The kernel
//! is not "correct on the machine of whoever has a GPU"; it is correct on every push.
//!
//! # What is here
//!
//! - [`WgpuExtend`]: device, pipeline and persistent buffers. Implements [`DeviceKernel`] (the
//!   interface the co-scheduler in `bwa-gpu` drives) and [`SwBackend`] (so the project's four
//!   acceptance gates apply unchanged).
//! - Chunking against the device's buffer limits, so a batch of any size runs.
//!
//! # Software adapters
//!
//! A CPU-emulated adapter (lavapipe, WARP) is refused unless `allow_software` is set: it is exact
//! but slower than the aligner's own SIMD kernels, so in production it can only lose. Tests set it.
//!
//! [`DeviceKernel`]: bwa_mem4_gpu::DeviceKernel
//! [`SwBackend`]: bwa_mem4_extend::SwBackend

#![cfg_attr(not(feature = "wgpu"), allow(dead_code))]

/// The WGSL source, compiled by the driver at run time.
pub const EXTEND_WGSL: &str = include_str!("extend.wgsl");

/// Per-job geometry, `repr(C)` against the WGSL `struct EJob` (8 x 4 bytes, no vector members).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct GpuEJob {
    q_off: u32,
    q_len: u32,
    t_off: u32,
    t_len: u32,
    h0: i32,
    w: i32,
    _pad0: u32,
    _pad1: u32,
}

/// Per-job result, `repr(C)` against the WGSL `struct ERes`.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct GpuERes {
    score: i32,
    qle: i32,
    tle: i32,
    gtle: i32,
    gscore: i32,
    max_off: i32,
    _pad0: i32,
    _pad1: i32,
}

/// The uniform block, `repr(C)` against the WGSL `struct Params`.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct GpuParams {
    m: i32,
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    zdrop: i32,
    n_jobs: u32,
    grid_x: u32,
}

/// Invocations per workgroup; must match `@workgroup_size` in the shader.
const WG: u32 = 64;

/// Plain-old-data view of a `repr(C)` slice, for `Queue::write_buffer`.
fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: only called on the `repr(C)` integer structs above and on `i32`/`u8` slices, all of
    // which have no padding bytes and no invalid bit patterns.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[cfg(feature = "wgpu")]
mod backend {
    use super::*;
    use bwa_mem4_extend::{ExtendJob, ExtendResult, SwBackend};
    use bwa_mem4_gpu::{DeviceKernel, FlatBatch, FlatJob, Scoring};
    use std::sync::Mutex;

    /// Buffers kept across launches, grown to the largest batch seen and never shrunk: allocating
    /// per launch would cost more than a small launch computes.
    #[derive(Default)]
    struct Buffers {
        seqs: Option<wgpu::Buffer>,
        jobs: Option<wgpu::Buffer>,
        out: Option<wgpu::Buffer>,
        readback: Option<wgpu::Buffer>,
        eh_h: Option<wgpu::Buffer>,
        eh_e: Option<wgpu::Buffer>,
        mat: Option<wgpu::Buffer>,
        params: Option<wgpu::Buffer>,
    }

    fn ensure(
        dev: &wgpu::Device,
        slot: &mut Option<wgpu::Buffer>,
        need: u64,
        usage: wgpu::BufferUsages,
        label: &str,
    ) {
        let need = need.max(16).next_multiple_of(4);
        if slot.as_ref().is_none_or(|b| b.size() < need) {
            // Grow geometrically so a slowly growing batch size does not reallocate every launch.
            let size = slot
                .as_ref()
                .map_or(need, |b| need.max(b.size().saturating_mul(3) / 2))
                .next_multiple_of(4);
            *slot = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            }));
        }
    }

    /// A wgpu device with the extension pipeline compiled.
    pub struct WgpuExtend {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::ComputePipeline,
        layout: wgpu::BindGroupLayout,
        info: wgpu::AdapterInfo,
        /// Largest single storage binding the device accepts, in bytes.
        max_binding: u64,
        bufs: Mutex<Buffers>,
    }

    impl WgpuExtend {
        /// Open the best adapter and compile the kernel. `None` when there is no usable adapter, or
        /// only a software one and `allow_software` is false: the caller's cue to stay on the CPU.
        pub fn new(allow_software: bool) -> Option<Self> {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::PRIMARY,
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                    ..Default::default()
                }))
                .ok()?;
            let info = adapter.get_info();
            if info.device_type == wgpu::DeviceType::Cpu && !allow_software {
                return None;
            }
            // Take what the adapter offers for the two limits that bound batch size; the defaults
            // (128 MiB bindings) would chunk a production batch for no reason on a 24 GB card.
            let al = adapter.limits();
            let limits = wgpu::Limits {
                max_storage_buffer_binding_size: al.max_storage_buffer_binding_size,
                max_buffer_size: al.max_buffer_size,
                max_storage_buffers_per_shader_stage: al.max_storage_buffers_per_shader_stage,
                ..wgpu::Limits::downlevel_defaults()
            };
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("bwa-mem4 extend"),
                    required_limits: limits.clone(),
                    ..Default::default()
                }))
                .ok()?;
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("extend.wgsl"),
                source: wgpu::ShaderSource::Wgsl(EXTEND_WGSL.into()),
            });
            let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("extend"),
                entries: &[
                    storage(0, true),
                    storage(1, true),
                    storage(2, false),
                    storage(3, false),
                    storage(4, false),
                    storage(5, true),
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("extend"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("extend_fwd"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("extend_fwd"),
                compilation_options: Default::default(),
                cache: None,
            });
            let max_binding = limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size);
            Some(Self {
                device,
                queue,
                pipeline,
                layout,
                info,
                max_binding,
                bufs: Mutex::new(Buffers::default()),
            })
        }

        /// Whether the adapter is a CPU emulation (lavapipe, WARP).
        pub fn is_software(&self) -> bool {
            self.info.device_type == wgpu::DeviceType::Cpu
        }

        /// Largest number of jobs of query length `max_qlen` that one launch can hold: each DP rail
        /// is `n_jobs * (max_qlen + 1)` `i32`s and must fit one storage binding.
        fn max_jobs_per_launch(&self, max_qlen: usize) -> usize {
            let per_job = (max_qlen as u64 + 1) * 4;
            let by_rail = (self.max_binding / per_job).max(1);
            // 65 535 workgroups per grid dimension, and the 2-D grid covers the rest; this bound
            // keeps `gid` inside `u32` with a wide margin.
            by_rail.min(1 << 24) as usize
        }

        /// Run `jobs[range]` of `b` as one launch.
        fn launch(
            &self,
            b: &FlatBatch,
            jobs: &[FlatJob],
            s: &Scoring,
        ) -> Option<Vec<ExtendResult>> {
            let n = jobs.len();
            if n == 0 {
                return Some(Vec::new());
            }
            let max_qlen = jobs.iter().map(|j| j.q_len as usize).max().unwrap_or(0);
            let gjobs: Vec<GpuEJob> = jobs
                .iter()
                .map(|j| GpuEJob {
                    q_off: j.q_off,
                    q_len: j.q_len,
                    t_off: j.t_off,
                    t_len: j.t_len,
                    h0: j.h0,
                    w: j.w,
                    _pad0: 0,
                    _pad1: 0,
                })
                .collect();
            let groups = (n as u32).div_ceil(WG);
            let gx = groups.min(65_535);
            let gy = groups.div_ceil(gx);
            let params = GpuParams {
                m: s.m as i32,
                o_del: s.o_del,
                e_del: s.e_del,
                o_ins: s.o_ins,
                e_ins: s.e_ins,
                zdrop: s.zdrop,
                n_jobs: n as u32,
                grid_x: gx * WG,
            };
            let mat: Vec<i32> = s.mat[..s.m * s.m].iter().map(|&x| i32::from(x)).collect();
            let rail = (n as u64) * (max_qlen as u64 + 1) * 4;
            let out_bytes = (n * std::mem::size_of::<GpuERes>()) as u64;

            use wgpu::BufferUsages as U;
            let mut g = self.bufs.lock().ok()?;
            let d = &self.device;
            ensure(
                d,
                &mut g.seqs,
                b.bytes.len() as u64,
                U::STORAGE | U::COPY_DST,
                "seqs",
            );
            ensure(
                d,
                &mut g.jobs,
                as_bytes(&gjobs).len() as u64,
                U::STORAGE | U::COPY_DST,
                "jobs",
            );
            ensure(d, &mut g.out, out_bytes, U::STORAGE | U::COPY_SRC, "out");
            ensure(
                d,
                &mut g.readback,
                out_bytes,
                U::MAP_READ | U::COPY_DST,
                "readback",
            );
            ensure(d, &mut g.eh_h, rail, U::STORAGE, "eh_h");
            ensure(d, &mut g.eh_e, rail, U::STORAGE, "eh_e");
            ensure(
                d,
                &mut g.mat,
                (mat.len() * 4) as u64,
                U::STORAGE | U::COPY_DST,
                "mat",
            );
            ensure(d, &mut g.params, 32, U::UNIFORM | U::COPY_DST, "params");
            let (seqs, jb, out, rb, eh, ee, mb, pb) = (
                g.seqs.as_ref()?,
                g.jobs.as_ref()?,
                g.out.as_ref()?,
                g.readback.as_ref()?,
                g.eh_h.as_ref()?,
                g.eh_e.as_ref()?,
                g.mat.as_ref()?,
                g.params.as_ref()?,
            );
            self.queue.write_buffer(seqs, 0, &b.bytes);
            self.queue.write_buffer(jb, 0, as_bytes(&gjobs));
            self.queue.write_buffer(mb, 0, as_bytes(&mat));
            self.queue
                .write_buffer(pb, 0, as_bytes(std::slice::from_ref(&params)));

            fn whole(buf: &wgpu::Buffer, i: u32) -> wgpu::BindGroupEntry<'_> {
                wgpu::BindGroupEntry {
                    binding: i,
                    resource: buf.as_entire_binding(),
                }
            }
            let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("extend"),
                layout: &self.layout,
                entries: &[
                    whole(seqs, 0),
                    whole(jb, 1),
                    whole(out, 2),
                    whole(eh, 3),
                    whole(ee, 4),
                    whole(mb, 5),
                    whole(pb, 6),
                ],
            });
            let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("extend"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("extend"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            enc.copy_buffer_to_buffer(out, 0, rb, 0, out_bytes);
            self.queue.submit([enc.finish()]);

            let slice = rb.slice(..out_bytes);
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            d.poll(wgpu::PollType::wait_indefinitely()).ok()?;
            rx.recv().ok()?.ok()?;
            let res = {
                let view = slice.get_mapped_range().ok()?;
                let mut r = vec![GpuERes::default(); n];
                // SAFETY: `r` is `n` padding-free `repr(C)` structs and the mapped range holds
                // exactly `n * size_of::<GpuERes>()` bytes written by the kernel.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        view.as_ptr(),
                        r.as_mut_ptr() as *mut u8,
                        out_bytes as usize,
                    );
                }
                r
            };
            rb.unmap();
            Some(
                res.iter()
                    .map(|r| ExtendResult {
                        score: r.score,
                        qle: r.qle,
                        tle: r.tle,
                        gtle: r.gtle,
                        gscore: r.gscore,
                        max_off: r.max_off,
                    })
                    .collect(),
            )
        }
    }

    impl DeviceKernel for WgpuExtend {
        fn name(&self) -> &'static str {
            "wgpu"
        }

        fn device_name(&self) -> String {
            format!(
                "{} ({:?}, {:?})",
                self.info.name, self.info.backend, self.info.device_type
            )
        }

        fn run(&self, b: &FlatBatch, s: &Scoring) -> Option<Vec<ExtendResult>> {
            let cap = self.max_jobs_per_launch(b.max_qlen);
            let mut out = Vec::with_capacity(b.len());
            for chunk in b.jobs.chunks(cap) {
                out.extend(self.launch(b, chunk, s)?);
            }
            Some(out)
        }
    }

    impl SwBackend for WgpuExtend {
        fn name(&self) -> &'static str {
            "wgpu"
        }

        fn extend(
            &self,
            query: &[u8],
            target: &[u8],
            m: usize,
            mat: &[i8],
            o_del: i32,
            e_del: i32,
            o_ins: i32,
            e_ins: i32,
            w: i32,
            end_bonus: i32,
            zdrop: i32,
            h0: i32,
        ) -> ExtendResult {
            let job = [ExtendJob { query, target, h0 }];
            self.extend_batch(
                &job, m, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
            )
            .remove(0)
        }

        fn extend_batch(
            &self,
            jobs: &[ExtendJob],
            m: usize,
            mat: &[i8],
            o_del: i32,
            e_del: i32,
            o_ins: i32,
            e_ins: i32,
            w: i32,
            end_bonus: i32,
            zdrop: i32,
        ) -> Vec<ExtendResult> {
            if jobs.is_empty() {
                return Vec::new();
            }
            let s = Scoring {
                m,
                mat: mat[..m * m].to_vec(),
                o_del,
                e_del,
                o_ins,
                e_ins,
                zdrop,
            };
            let b = FlatBatch::from_jobs(jobs, &s, w, end_bonus);
            self.run(&b, &s).unwrap_or_else(|| {
                // A device that fails mid-run is answered by the reference, identically.
                bwa_mem4_extend::ScalarBackend.extend_batch(
                    jobs, m, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
                )
            })
        }
    }
}

#[cfg(feature = "wgpu")]
pub use backend::WgpuExtend;

/// Stand-in without the `wgpu` feature: construction always fails, so callers take their CPU path.
#[cfg(not(feature = "wgpu"))]
pub struct WgpuExtend;

#[cfg(not(feature = "wgpu"))]
impl WgpuExtend {
    /// Always `None`: no wgpu support compiled in.
    pub fn new(_allow_software: bool) -> Option<Self> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three `repr(C)` structs must match their WGSL declarations; a mismatch is silent
    /// corruption, not a compile error.
    #[test]
    fn structs_have_the_expected_layout() {
        assert_eq!(std::mem::size_of::<GpuEJob>(), 32);
        assert_eq!(std::mem::size_of::<GpuERes>(), 32);
        assert_eq!(std::mem::size_of::<GpuParams>(), 32);
    }

    /// Rule of every DP kernel in this project: no floating point. Checked on the source, since the
    /// WGSL is compiled by the driver and there is no artefact to disassemble at build time.
    #[test]
    fn kernel_has_no_floating_point() {
        let code: String = EXTEND_WGSL
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        for bad in ["f32", "f16", "f64", "vec2<f", "vec4<f"] {
            assert!(
                !code.contains(bad),
                "floating point type `{bad}` in the kernel"
            );
        }
    }

    /// The whole acceptance harness, unchanged, against the real shader on whatever adapter the host
    /// has, software included (lavapipe on CI). Skips, loudly, when there is no Vulkan at all.
    #[test]
    #[cfg(feature = "wgpu")]
    fn wgpu_extend_passes_every_backend_gate() {
        use bwa_mem4_extend::{
            assert_backend_batch_matches_scalar, assert_backend_batch_order_invariant,
            assert_backend_matches_scalar, assert_backend_saturation_boundary_matches_scalar,
            assert_backend_tie_rule_matches_scalar,
        };
        let Some(gpu) = WgpuExtend::new(true) else {
            eprintln!("skipping: no wgpu adapter (install mesa-vulkan-drivers for lavapipe)");
            return;
        };
        eprintln!("device: {}", bwa_mem4_gpu::DeviceKernel::device_name(&gpu));
        assert_backend_matches_scalar(&gpu);
        assert_backend_batch_matches_scalar(&gpu);
        assert_backend_tie_rule_matches_scalar(&gpu);
        assert_backend_saturation_boundary_matches_scalar(&gpu);
        assert_backend_batch_order_invariant(&gpu);
    }

    /// The co-scheduler over the real shader: every split passes the harness, including under eight
    /// concurrent callers whose requests the service merges into shared launches.
    #[test]
    #[cfg(feature = "wgpu")]
    fn wgpu_cosched_every_split_and_merged_launches() {
        use bwa_mem4_extend::{
            assert_backend_batch_matches_scalar, assert_backend_matches_scalar, ScalarBackend,
        };
        use bwa_mem4_gpu::{CoSched, GpuService, Split};
        use std::sync::Arc;
        let Some(gpu) = WgpuExtend::new(true) else {
            eprintln!("skipping: no wgpu adapter");
            return;
        };
        let svc = GpuService::start(Arc::new(gpu));
        for split in [Split::Fixed(0.5), Split::Fixed(1.0), Split::Adaptive] {
            let cs = CoSched::new(ScalarBackend, Arc::clone(&svc), split).with_min_device_jobs(0);
            assert_backend_matches_scalar(&cs);
            std::thread::scope(|sc| {
                for _ in 0..8 {
                    sc.spawn(|| assert_backend_batch_matches_scalar(&cs));
                }
            });
        }
        let st = svc.stats();
        use std::sync::atomic::Ordering::Relaxed;
        eprintln!(
            "launches {} requests {} failures {}",
            st.launches.load(Relaxed),
            st.requests.load(Relaxed),
            st.failures.load(Relaxed)
        );
        assert_eq!(st.failures.load(Relaxed), 0);
        assert!(st.launches.load(Relaxed) > 0);
        // Merging happened: fewer launches than requests.
        assert!(st.requests.load(Relaxed) > st.launches.load(Relaxed));
    }
}
