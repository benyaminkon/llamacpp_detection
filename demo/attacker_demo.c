#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <sched.h>

#include <mastik/fr.h>
#include <mastik/low.h>
#include <mastik/util.h>

static void bind_cpu(int cpu_to_bind) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_to_bind, &set);
    sched_setaffinity(0, sizeof(set), &set);
}


int main(int argc, char **argv) {
    bind_cpu(0);  // bind to CPU 0
    // void *ptr = map_offset("/home/linuxbrew/.linuxbrew/lib/libllama.so", 0x15dd40);
    void *ptr = map_offset("/home/user/Projects/llamacpp_detection/.venv/lib/python3.10/site-packages/llama_cpp/lib/libllama.so", 0x16a4e0);
    if (ptr == NULL) {
        fprintf(stderr, "Could not locate decode_bytes\n");
        return 1;
    }

    printf("[*] monitoring decode_bytes %p\n", ptr);

    /* 2) FR setup */
    fr_t fr = fr_prepare();
    fr_monitor(fr, ptr);

    pid_t child = fork();
    if (child < 0) {
        perror("fork");
        return 1;
    }

    if (child == 0) {
        bind_cpu(7);  // bind child to CPU 1
        // execlp("python3",
        //    "python3",
        //    "src/victim.py",           // your script’s path
        //    "/home/user/Projects/llamacpp_detection/models/phi3-mini/Phi-3-mini-4k-instruct-q4.gguf",
        //    "/home/user/Projects/llamacpp_detection/alpaca_missing_tokens.txt",
        //    (char*)NULL);
        execlp("python3", "python3", "src/victim.py", (char*)NULL);
        perror("execlp");    // if we get here, exec failed
        exit(1);
    }
    printf("[*] spawned victim (pid %d)\n", child);

    // /* 5) sampling loop */
    // uint16_t res[1];
    // int lines=0;

    // for (;;) {
    //     fr_probe(fr, res);
    //     if (res[0] < 100) {
    //         printf("%4d: %s", ++lines, "Call decode_bytes\n");
    //     }
    //     delayloop(10000);  // delay to avoid flooding output
    // }
    
    uint16_t res[1];
    int lines=0;

    for (;; ) {
        fr_probe(fr, res);
        if (res[0] < 100) {
            printf("%d\n", res[0]);
            printf("%4d: %s", ++lines, "Call to func\n");
        }
        delayloop(10000);  // delay to avoid flooding output
    }


    /* 6) dump in sampling order */
    // FILE *out = fopen("results/raw_samples.txt", "w");
    // if (!out) {
    //     perror("fopen");
    //     kill(child, SIGKILL);
    //     return 1;
    // }
    // for (size_t i = 0; i < 1000000; i++) {
    //     fprintf(out, "%zu %u\n", i, res[i]);
    // }
    // fclose(out);
    // printf("[*] dumped %d samples to results/raw_samples.txt\n", lines);

    /* 7) cleanup */
    kill(child, SIGKILL);
    fr_release(fr);
    return 0;
}