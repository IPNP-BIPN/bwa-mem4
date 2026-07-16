// Measures the TLB hierarchy directly, separated from the cache.
// Touch exactly ONE cache line per 16KB page, chasing a random cycle over PAGES.
// The DATA footprint stays small (npages * 128B) so it stays cache-resident far longer than the
// PAGE count stays TLB-resident. A latency cliff is therefore the TLB / page walker, not the cache.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <sys/mman.h>

#define PAGE 16384u
#define LINE 128u

static double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

typedef struct { void **head; long iters; double ns; } Job;

static void *chase(void *arg) {
    Job *j = arg;
    void **p = j->head;
    double t0 = now_ns();
    for (long i = 0; i < j->iters; i++) p = (void **)*p;   // serial dependent chain
    double t1 = now_ns();
    __asm__ volatile("" :: "r"(p));
    j->ns = (t1 - t0) / (double)j->iters;
    return NULL;
}

// Random cycle over npages, one slot per page. Returns the cycle head.
static void **build(size_t npages) {
    char *m = mmap(NULL, npages * (size_t)PAGE, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
    if (m == MAP_FAILED) return NULL;
    size_t *ord = malloc(npages * sizeof(size_t));
    for (size_t i = 0; i < npages; i++) ord[i] = i;
    for (size_t i = npages - 1; i > 0; i--) {
        size_t k = (size_t)(random() % (long)(i + 1));
        size_t t = ord[i]; ord[i] = ord[k]; ord[k] = t;
    }
    #define SLOT(pg) ((void **)(m + (pg) * (size_t)PAGE + ((pg) % (PAGE/LINE)) * (size_t)LINE))
    for (size_t i = 0; i < npages; i++) *SLOT(ord[i]) = (void *)SLOT(ord[(i + 1) % npages]);
    void **head = SLOT(ord[0]);
    free(ord);
    return head;
}

int main(int argc, char **argv) {
    int nthreads = argc > 1 ? atoi(argv[1]) : 1;
    srandom(12345);
    printf("# one line per 16KB page, random cycle over pages, %d thread(s)\n", nthreads);
    printf("# %-8s %-9s %-10s %s\n", "pages", "span(MB)", "data(KB)", "ns/access");
    for (size_t npages = 16; npages <= (1u<<18); npages *= 2) {
        Job jobs[16]; pthread_t th[16];
        for (int t = 0; t < nthreads; t++) {
            jobs[t].head = build(npages);       // each thread gets its OWN mapping
            if (!jobs[t].head) { printf("  mmap failed at %zu pages\n", npages); return 1; }
            jobs[t].iters = (long)npages * 4 < 200000 ? 200000 : (long)npages * 4;
        }
        for (int t = 0; t < nthreads; t++) chase(&jobs[t]);   // warm
        for (int t = 0; t < nthreads; t++) pthread_create(&th[t], NULL, chase, &jobs[t]);
        double sum = 0;
        for (int t = 0; t < nthreads; t++) { pthread_join(th[t], NULL); sum += jobs[t].ns; }
        printf("  %-8zu %-9.1f %-10.0f %.1f\n", npages, npages * PAGE / 1048576.0,
               npages * LINE / 1024.0, sum / nthreads);
        fflush(stdout);
    }
    return 0;
}
