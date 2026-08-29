// Is the u8 rescue kernel bound by memory BANDWIDTH or by the rate at which it can ISSUE memory
// accesses? The two answers call for different fixes, and the shipped kernel sits on one of them.
//
// What the aligner measures (ROADMAP, "Ce qui borne vraiment le noyau GPU"):
//   u8 rails      50.31 Gcell/s   5 bytes/cell    252 GB/s (46% of 546)   252 G ops/s
//   32-bit rails  26.47 Gcell/s  20 bytes/cell    529 GB/s (97%)          132 G ops/s
// Same control flow, same five accesses per cell, so the 32-bit one is caught by bandwidth and the
// u8 one is not. That points at the access ISSUE rate, and the fix it implies is to make one access
// serve four jobs (a uchar4 rail), which divides ops by four without moving fewer bytes.
//
// That is a prediction, and it costs a day of kernel work to act on. This probe tests it in forty
// lines instead. Two kernels, the same recurrence and the same memory pattern as `rescue_fwd_u8`:
// three loads and two stores per cell on column-major rails. `mem1` runs one job per thread with
// `uchar` rails; `mem4` runs FOUR jobs per thread with `uchar4` rails, i.e. a quarter of the
// accesses for the same cells. Both are dispatched to compute the SAME number of cells.
//
// If mem4 is about twice mem1, the issue-rate reading is right and the uchar4 kernel is worth
// writing. If they are equal, it is not, and a day is saved.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static const char *SRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
// One job per thread, uchar rails: exactly the shipped kernel's access shape.
"kernel void mem1(device uchar *hp [[buffer(0)]],\n"
"                 device uchar *hc [[buffer(1)]],\n"
"                 device uchar *ev [[buffer(2)]],\n"
"                 constant uint &cols [[buffer(3)]],\n"
"                 constant uint &rows [[buffer(4)]],\n"
"                 constant uint &stride [[buffer(5)]],\n"
"                 uint gid [[thread_position_in_grid]]) {\n"
"  device uchar *p = hp, *c = hc;\n"
"  uchar f = 0, hd = 0;\n"
"  for (uint i = 0; i < rows; ++i) {\n"
"    for (uint j = 0; j < cols; ++j) {\n"
"      ulong o = (ulong)j * stride + gid;\n"
"      uchar e = ev[o];\n"
"      uchar h = max(max(subsat(addsat(hd, (uchar)5), (uchar)4), e), f);\n"
"      hd = p[o];\n"
"      c[o] = h;\n"
"      ev[o] = max(subsat(e, (uchar)1), subsat(h, (uchar)7));\n"
"      f     = max(subsat(f, (uchar)1), subsat(h, (uchar)7));\n"
"    }\n"
"    device uchar *t = p; p = c; c = t;\n"
"  }\n"
"}\n"
// Four jobs per thread, uchar4 rails: same cells, a quarter of the accesses.
"kernel void mem1g(device uchar *hp [[buffer(0)]],\n"
"                  device uchar *hc [[buffer(1)]],\n"
"                  device uchar *ev [[buffer(2)]],\n"
"                  constant uint &cols [[buffer(3)]],\n"
"                  constant uint &rows [[buffer(4)]],\n"
"                  constant uint &stride [[buffer(5)]],\n"
"                  device const uchar *seqs [[buffer(6)]],\n"
"                  uint gid [[thread_position_in_grid]]) {\n"
"  device uchar *p = hp, *c = hc;\n"
"  uchar f = 0, hd = 0;\n"
"  ulong qoff = (ulong)gid * cols;\n"   // each thread's own query, exactly as `j.q_off` gives it
"  for (uint i = 0; i < rows; ++i) {\n"
"    for (uint j = 0; j < cols; ++j) {\n"
"      ulong o = (ulong)j * stride + gid;\n"
"      uchar qb = seqs[qoff + j];\n"     // THE GATHER: 32 lanes read 32 scattered addresses
"      uchar e = ev[o];\n"
"      uchar h = max(max(subsat(addsat(hd, (uchar)(qb | 1)), (uchar)4), e), f);\n"
"      hd = p[o];\n"
"      c[o] = h;\n"
"      ev[o] = max(subsat(e, (uchar)1), subsat(h, (uchar)7));\n"
"      f     = max(subsat(f, (uchar)1), subsat(h, (uchar)7));\n"
"    }\n"
"    device uchar *t = p; p = c; c = t;\n"
"  }\n"
"}\n"
"kernel void mem4(device uchar4 *hp [[buffer(0)]],\n"
"                 device uchar4 *hc [[buffer(1)]],\n"
"                 device uchar4 *ev [[buffer(2)]],\n"
"                 constant uint &cols [[buffer(3)]],\n"
"                 constant uint &rows [[buffer(4)]],\n"
"                 constant uint &stride [[buffer(5)]],\n"
"                 uint gid [[thread_position_in_grid]]) {\n"
"  device uchar4 *p = hp, *c = hc;\n"
"  uchar4 f = 0, hd = 0;\n"
"  for (uint i = 0; i < rows; ++i) {\n"
"    for (uint j = 0; j < cols; ++j) {\n"
"      ulong o = (ulong)j * stride + gid;\n"
"      uchar4 e = ev[o];\n"
"      uchar4 h = max(max(subsat(addsat(hd, uchar4(5)), uchar4(4)), e), f);\n"
"      hd = p[o];\n"
"      c[o] = h;\n"
"      ev[o] = max(subsat(e, uchar4(1)), subsat(h, uchar4(7)));\n"
"      f     = max(subsat(f, uchar4(1)), subsat(h, uchar4(7)));\n"
"    }\n"
"    device uchar4 *t = p; p = c; c = t;\n"
"  }\n"
"}\n";

int main(void) { @autoreleasepool {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (!dev) { printf("no Metal device\n"); return 1; }
  NSError *err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:SRC]
                                         options:nil error:&err];
  if (!lib) { printf("compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
  id<MTLCommandQueue> q = [dev newCommandQueue];
  printf("device: %s\n", dev.name.UTF8String);

  // A rescue job's shape: a 150 bp query against a window of a few hundred bases.
  const uint32_t cols = 150, rows = 200;
  const uint32_t jobs = 1u << 18;          // total jobs, identical for both arms
  const size_t rail = (size_t)cols * jobs; // bytes per rail, identical for both arms

  id<MTLBuffer> b[3];
  for (int i = 0; i < 3; ++i)
    b[i] = [dev newBufferWithLength:rail options:MTLResourceStorageModePrivate];

  // A per-thread query, laid out job-major exactly as `seqs` is in the aligner: thread `g` reads
  // `seqs[g * cols + j]`, so the 32 lanes of a simdgroup read 32 addresses `cols` apart. That is a
  // gather, once per cell, and it is the one thing the real kernel does that `mem1` does not.
  id<MTLBuffer> qb = [dev newBufferWithLength:(size_t)cols * jobs
                                      options:MTLResourceStorageModePrivate];

  struct { const char *name; const char *fn; uint32_t threads; uint32_t stride; double per_iter; } arms[3] = {
    { "mem1  (uchar,  1 job/thread)",        "mem1",  jobs,     jobs,     1.0 },
    { "mem4  (uchar4, 4 jobs/thread)",       "mem4",  jobs / 4, jobs / 4, 4.0 },
    { "mem1g (uchar + query gather/cell)",   "mem1g", jobs,     jobs,     1.0 },
  };
  double gcell[3];
  for (int a = 0; a < 3; ++a) {
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:arms[a].fn]];
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!ps) { printf("pipeline failed: %s\n", err.localizedDescription.UTF8String); return 1; }
    uint32_t c = cols, r = rows, s = arms[a].stride;
    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
      [ce setComputePipelineState:ps];
      for (int i = 0; i < 3; ++i) [ce setBuffer:b[i] offset:0 atIndex:i];
      [ce setBytes:&c length:4 atIndex:3];
      [ce setBytes:&r length:4 atIndex:4];
      [ce setBytes:&s length:4 atIndex:5];
      if (a == 2) [ce setBuffer:qb offset:0 atIndex:6];
      [ce dispatchThreads:MTLSizeMake(arms[a].threads,1,1)
          threadsPerThreadgroup:MTLSizeMake(ps.maxTotalThreadsPerThreadgroup,1,1)];
      [ce endEncoding];
      NSDate *t0 = [NSDate date]; [cb commit]; [cb waitUntilCompleted];
      double dt = -[t0 timeIntervalSinceNow];
      if (dt < best) best = dt;
    }
    double cells = (double)arms[a].threads * rows * cols * arms[a].per_iter;
    gcell[a] = cells / best / 1e9;
    printf("%-30s %8.3f Gcell in %.4f s -> %6.2f Gcell/s, %6.1f GB/s, %6.1f G ops/s\n",
           arms[a].name, cells/1e9, best, gcell[a], gcell[a]*5.0, gcell[a]*5.0/arms[a].per_iter);
  }
  printf("\nuchar4 against scalar: %.2fx\n", gcell[1]/gcell[0]);
  printf("what the per-cell query gather costs: %.2fx\n", gcell[2]/gcell[0]);
  return 0;
} }
