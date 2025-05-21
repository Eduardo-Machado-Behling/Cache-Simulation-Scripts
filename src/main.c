#include "args.h"
#include "report.h"

#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#ifndef THRDS_AMOUNT
#define THRDS_AMOUNT 4
#endif

void create_report(report_t *report, args_t *cmd, char *buff, int size) {
  PCRE2_SPTR pattern = (PCRE2_SPTR) R"(sim: \*\* simulation statistics \*\*)";
  PCRE2_SPTR subject = (PCRE2_SPTR)buff;
  PCRE2_SIZE subject_length = size;
  report->used = 0;

  strcpy(report->df[report->used++], cmd->argv[2]);
  strcpy(report->df[report->used++], cmd->argv[4]);
  strcpy(report->df[report->used++], cmd->argv[6]);
  strcpy(report->df[report->used++], cmd->argv[8]);
  strcpy(report->df[report->used++], cmd->argv[10]);
  strcpy(report->df[report->used++], cmd->argv[12]);

  {
    char *working = report->df[report->used++];
    int len = 0;
    for (size_t i = 13; i < cmd->argc; i++) {
      char *basename = strrchr(cmd->argv[i], '/');
      if (!basename++) {
        basename = cmd->argv[i];
      }
      len = strlen(basename);

      working = strcpy(working, basename);
      working[len] = i == cmd->argc - 1 ? '\0' : ';';
      working += len + 1;
    }
  }

  pcre2_code *re;
  PCRE2_SIZE erroffset;
  int errorcode;

  re = pcre2_compile(pattern, PCRE2_ZERO_TERMINATED, 0, &errorcode, &erroffset,
                     NULL);
  if (!re) {
    puts("PCRE2 compilation failed");
  }

  pcre2_match_data *match_data = pcre2_match_data_create_from_pattern(re, NULL);
  int rc =
      pcre2_match(re, subject, PCRE2_ZERO_TERMINATED, 0, 0, match_data, NULL);

  PCRE2_SIZE start_offset = 0;
  if (rc >= 0) {
    PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data);
    start_offset = ovector[1];
  }

  pcre2_match_data_free(match_data);
  pcre2_code_free(re);

  // Compile the pattern
  pattern = (PCRE2_SPTR) R"([\w.]+\s+([xa-fA-FkMGmg0-9\.]+)[\s#]+(.*))";
  re = pcre2_compile(pattern, PCRE2_ZERO_TERMINATED, 0, &errorcode, &erroffset,
                     NULL);
  if (re == NULL) {
    PCRE2_UCHAR buffer[256];
    pcre2_get_error_message(errorcode, buffer, sizeof(buffer));
    printf("PCRE2 compilation failed at offset %d: %s\n", (int)erroffset,
           buffer);
  }

  // Create match data block
  match_data = pcre2_match_data_create_from_pattern(re, NULL);

  // Find all matches (equivalent to Python's finditer)
  while (start_offset < subject_length) {
    rc = pcre2_match(re, subject, subject_length, start_offset, 0, match_data,
                     NULL);

    if (rc < 0) {
      if (rc == PCRE2_ERROR_NOMATCH) {
        // No more matches
        break;
      } else {
        // Error occurred
        printf("Matching error %d\n", rc);
        break;
      }
    }

    // Get the offset vector
    PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data);

    // Print captured groups (equivalent to match.group(1), match.group(2))
    if (rc >= 2) { // At least group 1 exists
      // Group 1
      PCRE2_SIZE start2 = ovector[2];
      PCRE2_SIZE end2 = ovector[3];
      snprintf(report->df[report->used++], sizeof(report->df[0]), "%.*s",
               (int)(end2 - start2), (char *)(subject + start2));
    }

    // Update start_offset for next iteration
    start_offset = ovector[1]; // End of current match

    // Prevent infinite loop on zero-length matches
    if (ovector[0] == ovector[1]) {
      if (start_offset >= subject_length)
        break;
      start_offset++;
    }
  }

  // Cleanup
  pcre2_match_data_free(match_data);
  pcre2_code_free(re);
}

void run(args_t *cmd, atomic_stack_t *stack, int fd) {
  int pipefd[2];
  pipe(pipefd);        // create a dedicated pipe per run
  char *exec_argv[21]; // 20 max + 1 for NULL

  printf("Running: ");
  for (int i = 0; i < cmd->argc; ++i) {
    exec_argv[i] = (char *)cmd->argv[i]; // cast OK if you're not modifying
    printf("%s ", exec_argv[i]);
  }
  puts("\n");
  exec_argv[cmd->argc] = NULL; // Null-terminate

  pid_t pid = fork();

  if (pid == 0) {
    // Child process
    close(pipefd[0]);
    dup2(pipefd[1], STDERR_FILENO); // Redirect stderr
    close(pipefd[1]);

    // Redirect stdout to /dev/null
    dup2(fd, STDOUT_FILENO);

    chdir("./benchmarks/vortex");
    execv("./../../sim-cache", exec_argv);
    perror("execl failed"); // If exec fails
    _exit(1);
  } else {
    // Parent process
    close(pipefd[1]); // Close unused write end
    char buffer[6000];
    char *work = buffer;
    ssize_t len;
    while ((len = read(pipefd[0], work,
                       (sizeof(buffer) - (work - buffer)) - 1)) > 0) {
      work += len;
    }
    *work = '\0';
    close(pipefd[0]); // Close read end after reading
    report_t report;
    create_report(&report, cmd, buffer, work - buffer);
    stack_push(stack, &report);
    printf("Done: ");
    for (int i = 0; i < cmd->argc; ++i) {
      exec_argv[i] = (char *)cmd->argv[i]; // cast OK if you're not modifying
      printf("%s ", exec_argv[i]);
    }
    puts("\n");
    waitpid(pid, NULL, 0);
  }
}

typedef struct thrd_args_t {
  atomic_stack_t *st;
  args_queue_t *queue;
  int id;
} thrd_args_t;

int running = 1;

void *work_thread(thrd_args_t *args) {
  char buff[128];
  snprintf(buff, sizeof(buff), "./Thread(%d).log", args->id);
  int fd = open(buff, O_WRONLY | O_CREAT, 0644);
  if (fd == -1) {
    perror("open");
    _exit(1);
  }

  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, SIGINT);                // Add SIGINT to the set
  pthread_sigmask(SIG_BLOCK, &set, NULL); // Block SIGINT in this thread
  while (args_queue_size(args->queue) && running) {
    stack_lock(args->st);
    run(args_queue_pop(args->queue), args->st, fd);
    stack_unlock(args->st);
    fsync(fd);
  }

  close(fd); // devnull no longer needed
}

atomic_stack_t *stack;
char time_str[16];
struct timespec start, end;
double elapsed;

void format_time(int total_seconds, char *buffer, size_t buffer_size) {
  int hours = total_seconds / 3600;
  int minutes = (total_seconds % 3600) / 60;
  int seconds = total_seconds % 60;

  snprintf(buffer, buffer_size, "%02d:%02d:%02d", hours, minutes, seconds);
}

void __save() {
  printf("exiting... saving...\n");
  running = 0;
  stack_to_csv(stack, "out.csv", 1);

  clock_gettime(CLOCK_MONOTONIC, &end);
  elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  format_time(elapsed, time_str, sizeof(time_str)); // 1:01:01
  printf("Total elapsed time %s for %ld simulations\n", time_str,
         stack_size(stack));
}

void save(int status) { exit(EXIT_FAILURE); }

int main(int argc, const char **argv, const char **envp) {
  {
    char buf[512] = "python3 gen_in.py";
    if (argc > 1) {
      snprintf(buf, sizeof(buf), "python3 gen_in.py \"%s\"", argv[1]);
    }
    system(buf);
  }

  args_queue_t *queue = parse_args("args.input");
  stack = create_stack(args_queue_size(queue) + 1);
  clock_gettime(CLOCK_MONOTONIC, &start);

  printf("Amount: %d\n", args_queue_size(queue));

  const int NUM_THREADS = THRDS_AMOUNT;
  pthread_t threads[NUM_THREADS];
  thrd_args_t threads_args[NUM_THREADS];
  int ids[NUM_THREADS];

  signal(SIGINT, save); // Register handler
  atexit(__save);

  for (int i = 0; i < NUM_THREADS; ++i) {
    threads_args[i].st = stack;
    threads_args[i].queue = queue;
    threads_args[i].id = i;

    pthread_create(&threads[i], NULL, (void *(*)(void *))work_thread,
                   threads_args + i);
  }

  while (args_queue_size(queue)) {
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    format_time(elapsed, time_str, sizeof(time_str)); // 1:01:01
    printf("[%s]: queued: %d, stacked: %ld\n", time_str, args_queue_size(queue),
           stack_size(stack));
    sleep(5);
    stack_to_csv(stack, "out.csv", 0);
  }

  for (int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(threads[i], NULL);
  }

  return EXIT_SUCCESS;
}
