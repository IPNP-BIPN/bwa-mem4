#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static int try_src(id<MTLDevice> dev, const char *label, const char *src) {
    @autoreleasepool {
        NSError *err = nil;
        MTLCompileOptions *o = [MTLCompileOptions new];
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src]
                                               options:o error:&err];
        if (lib) { printf("  OK    %s\n", label); return 1; }
        NSString *m = [err localizedDescription];
        // first line of the diagnostic only
        NSString *first = [[m componentsSeparatedByString:@"\n"] firstObject];
        printf("  FAIL  %-22s %s\n", label, [first UTF8String]);
        return 0;
    }
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("device: %s\n", [[dev name] UTF8String]);
        printf("MSL feature probe (issue #55 step 0)\n");
        try_src(dev, "addsat(uchar4)",
          "#include <metal_stdlib>\nusing namespace metal;\nkernel void k(device uchar4*o[[buffer(0)]]){o[0]=addsat(o[1],o[2]);}");
        try_src(dev, "subsat(uchar4)",
          "#include <metal_stdlib>\nusing namespace metal;\nkernel void k(device uchar4*o[[buffer(0)]]){o[0]=subsat(o[1],o[2]);}");
        try_src(dev, "add_sat(uchar4)",
          "#include <metal_stdlib>\nusing namespace metal;\nkernel void k(device uchar4*o[[buffer(0)]]){o[0]=add_sat(o[1],o[2]);}");
        try_src(dev, "sub_sat(uchar4)",
          "#include <metal_stdlib>\nusing namespace metal;\nkernel void k(device uchar4*o[[buffer(0)]]){o[0]=sub_sat(o[1],o[2]);}");
        try_src(dev, "addsat(uchar)",
          "#include <metal_stdlib>\nusing namespace metal;\nkernel void k(device uchar*o[[buffer(0)]]){o[0]=addsat(o[1],o[2]);}");
        try_src(dev, "max(uchar4)",
          "#include <metal_stdlib>\nusing namespace metal;\nkernel void k(device uchar4*o[[buffer(0)]]){o[0]=max(o[1],o[2]);}");
        try_src(dev, "addsat(ushort4)",
          "#include <metal_stdlib>\nusing namespace metal;\nkernel void k(device ushort4*o[[buffer(0)]]){o[0]=addsat(o[1],o[2]);}");
        try_src(dev, "subsat(ushort4)",
          "#include <metal_stdlib>\nusing namespace metal;\nkernel void k(device ushort4*o[[buffer(0)]]){o[0]=subsat(o[1],o[2]);}");
        return 0;
    }
}
