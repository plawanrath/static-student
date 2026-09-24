/* Start-to-first-struct, measured from the parent with no interpreter in the path.
 *
 *   spawn_bench <launches> <warmup> -- <command> [args...]
 *
 * For each launch: take a monotonic timestamp, posix_spawn the child with its stdout on a pipe, read until the first
 * newline the child writes, and stop the clock there. That is the instant a deployment has its first parsed struct,
 * and it includes the loader, relocation, any engine's session setup, page faults on the weight array and the first
 * inference. The child is then reaped so the next launch starts clean. Timings are printed one per line in
 * microseconds, and the caller does the statistics. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

static double one_launch(char **argv) {
    int fds[2];
    if (pipe(fds) != 0) return -1;
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&fa, fds[0]);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    pid_t pid;
    int rc = posix_spawn(&pid, argv[0], &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(fds[1]);
    if (rc != 0) { close(fds[0]); fprintf(stderr, "spawn %s: %s\n", argv[0], strerror(rc)); return -1; }

    char buf[256];
    ssize_t got, total = 0;
    int saw_line = 0;
    while (!saw_line && (got = read(fds[0], buf, sizeof buf)) > 0) {
        total += got;
        for (ssize_t i = 0; i < got; i++) if (buf[i] == '\n') { saw_line = 1; break; }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    close(fds[0]);
    int status;
    waitpid(pid, &status, 0);
    if (!saw_line || total == 0) return -1;
    return (double)(t1.tv_sec - t0.tv_sec) * 1e6 + (double)(t1.tv_nsec - t0.tv_nsec) / 1e3;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <launches> <warmup> -- <command> [args...]\n", argv[0]); return 2; }
    long n = strtol(argv[1], NULL, 10), warm = strtol(argv[2], NULL, 10);
    char **cmd = &argv[4];
    for (long i = 0; i < warm; i++) if (one_launch(cmd) < 0) return 1;
    for (long i = 0; i < n; i++) {
        double us = one_launch(cmd);
        if (us < 0) { fprintf(stderr, "launch %ld failed\n", i); return 1; }
        printf("%.3f\n", us);
    }
    return 0;
}
