#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static const char *SRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
// Semantics check: saturating add/sub on uchar4, at both clamps.
"kernel void sem(device uchar4 *o [[buffer(0)]]) {\n"
"  o[0] = addsat(uchar4(250,3,0,255), uchar4(10,0,0,1));\n"   // -> 255,3,0,255
"  o[1] = subsat(uchar4(3,255,0,10), uchar4(4,1,7,10));\n"    // -> 0,254,0,0
"}\n"
// One column of the rescue recurrence, uchar4 = 4 cells per thread, registers only.
// Loop-carried through h so nothing folds away; ITERS columns per thread.
"kernel void dp(device uchar4 *out [[buffer(0)]],\n"
"               constant uint &iters [[buffer(1)]],\n"
"               uint gid [[thread_position_in_grid]]) {\n"
"  uchar4 d = uchar4(gid & 63), e = uchar4(1), f = uchar4(2);\n"
"  uchar4 imax = 0, col = 0, jv = 0;\n"
"  uchar4 bias = uchar4(4), edel = uchar4(1), oedel = uchar4(7);\n"
"  uchar4 s = uchar4(5);\n"
"  for (uint i = 0; i < iters; ++i) {\n"
"    uchar4 diag = subsat(addsat(d, s), bias);\n"
"    uchar4 mfe  = max(diag, e);\n"
"    uchar4 h    = max(mfe, f);\n"
"    col  = select(col, jv, h > imax);\n"
"    imax = max(imax, h);\n"
"    uchar4 hg = subsat(h, oedel);\n"
"    e = max(subsat(e, edel), hg);\n"
"    f = max(subsat(f, edel), hg);\n"
"    d = h;\n"
"    jv += uchar4(1);\n"
"  }\n"
"  out[gid] = imax + col;\n"
"}\n";

int main(void) {
 @autoreleasepool {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  NSError *err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:SRC]
                                         options:[MTLCompileOptions new] error:&err];
  if (!lib) { printf("compile failed: %s\n", [[err localizedDescription] UTF8String]); return 1; }
  id<MTLCommandQueue> q = [dev newCommandQueue];

  // --- semantics ---
  id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"sem"] error:&err];
  id<MTLBuffer> sb = [dev newBufferWithLength:32 options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
  [ce setComputePipelineState:ps]; [ce setBuffer:sb offset:0 atIndex:0];
  [ce dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
  [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
  unsigned char *r = (unsigned char *)[sb contents];
  printf("addsat(250,3,0,255 + 10,0,0,1) = %u,%u,%u,%u   (want 255,3,0,255)\n", r[0],r[1],r[2],r[3]);
  printf("subsat(3,255,0,10 - 4,1,7,10)  = %u,%u,%u,%u   (want 0,254,0,0)\n", r[4],r[5],r[6],r[7]);

  // --- throughput ceiling: registers only, no memory in the loop ---
  id<MTLComputePipelineState> pd = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dp"] error:&err];
  printf("dp pipeline: maxTotalThreadsPerThreadgroup=%lu simdWidth=%lu\n",
         (unsigned long)pd.maxTotalThreadsPerThreadgroup, (unsigned long)pd.threadExecutionWidth);
  uint32_t threads = 1u << 20, iters = 4096;
  id<MTLBuffer> ob = [dev newBufferWithLength:threads*4 options:MTLResourceStorageModePrivate];
  id<MTLBuffer> ib = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
  *(uint32_t *)[ib contents] = iters;
  double best = 1e30;
  for (int rep = 0; rep < 5; ++rep) {
    cb = [q commandBuffer]; ce = [cb computeCommandEncoder];
    [ce setComputePipelineState:pd]; [ce setBuffer:ob offset:0 atIndex:0]; [ce setBuffer:ib offset:0 atIndex:1];
    [ce dispatchThreads:MTLSizeMake(threads,1,1)
       threadsPerThreadgroup:MTLSizeMake(pd.maxTotalThreadsPerThreadgroup,1,1)];
    [ce endEncoding];
    NSDate *t0 = [NSDate date]; [cb commit]; [cb waitUntilCompleted];
    double dt = -[t0 timeIntervalSinceNow];
    if (dt < best) best = dt;
  }
  double cells = (double)threads * (double)iters * 4.0;   // uchar4 = 4 cells per thread-iteration
  printf("dp kernel: %u threads x %u columns x 4 cells = %.3f Gcell in %.4f s -> %.1f Gcell/s\n",
         threads, iters, cells/1e9, best, cells/best/1e9);
  return 0;
 }
}
