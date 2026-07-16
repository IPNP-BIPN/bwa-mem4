// Does the prefetch win survive thread contention?
// Models the real thing: ONE shared 16 GB array (as the SA is shared by every worker), each thread
// streaming its own independent random index list. Arm A: no prefetch. Arm B: prfm at distance 32.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <sys/mman.h>

#define D 32
#define NIDX 2000000L
#define SPAN_GB 16ull

static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }
static inline void pf(const void *p){ __asm__ volatile("prfm pldl1keep, [%0]" :: "r"(p)); }

static uint64_t *arr; static size_t N;
typedef struct { uint32_t *idx; int use_pf; double ns; } Job;

static void *run(void *a){
    Job *j = a; uint64_t s = 0;
    double t0 = now_ns();
    if (j->use_pf) {
        for (long i = 0; i < NIDX; i++) { if (i + D < NIDX) pf(&arr[j->idx[i+D]]); s += arr[j->idx[i]]; }
    } else {
        for (long i = 0; i < NIDX; i++) s += arr[j->idx[i]];
    }
    double t1 = now_ns();
    __asm__ volatile("" :: "r"(s));
    j->ns = (t1 - t0) / (double)NIDX;
    return NULL;
}

int main(void){
    size_t bytes = SPAN_GB << 30; N = bytes / 8;
    arr = mmap(NULL, bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
    if (arr == MAP_FAILED) { perror("mmap"); return 1; }
    fprintf(stderr, "faulting in %llu GB...\n", SPAN_GB);
    for (size_t i = 0; i < N; i += 2048) arr[i] = i;
    srandom(4242);
    printf("# ONE shared %llu GB array, per-thread independent random index streams\n", SPAN_GB);
    printf("# %-9s %-12s %-12s %s\n", "threads", "no_pf(ns)", "pf(ns)", "prefetch speedup");
    for (int nt = 1; nt <= 8; nt *= 2) {
        Job jobs[8]; pthread_t th[8];
        for (int t = 0; t < nt; t++) {
            jobs[t].idx = malloc(NIDX * 4);
            for (long i = 0; i < NIDX; i++)
                jobs[t].idx[i] = (uint32_t)((((uint64_t)random()<<31) ^ (uint64_t)random()) % N);
        }
        double res[2] = {1e18, 1e18};
        // Warm up at this thread count BEFORE any timing: the first pass over a freshly faulted
        // 16 GB mapping pays one-off costs (frequency ramp, page-table population, the memory
        // compressor settling) that would otherwise be charged entirely to whichever arm runs first.
        for (int t = 0; t < nt; t++) jobs[t].use_pf = 0;
        for (int t = 0; t < nt; t++) pthread_create(&th[t], NULL, run, &jobs[t]);
        for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
        // Interleave the arms, min-of-3. Never A,A,A then B,B,B.
        for (int rep = 0; rep < 3; rep++) {
            for (int arm = 0; arm < 2; arm++) {
                for (int t = 0; t < nt; t++) jobs[t].use_pf = arm;
                for (int t = 0; t < nt; t++) pthread_create(&th[t], NULL, run, &jobs[t]);
                double sum = 0;
                for (int t = 0; t < nt; t++) { pthread_join(th[t], NULL); sum += jobs[t].ns; }
                double v = sum / nt;
                if (v < res[arm]) res[arm] = v;
            }
        }
        printf("  %-9d %-12.1f %-12.1f %.2fx\n", nt, res[0], res[1], res[0]/res[1]);
        fflush(stdout);
        for (int t = 0; t < nt; t++) free(jobs[t].idx);
    }
    return 0;
}
