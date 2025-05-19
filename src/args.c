#include "args.h"

#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct args_queue_t {
  args_t *data;
  int capacity;
  int used;

  atomic_int st;
};

static void parse_arg(args_t *args);

args_queue_t *parse_args(const char *path) {
  FILE *fp = fopen(path, "r");
  int fd = fileno(fp);
  struct stat file_stat;

  if (fd == -1) {
    fprintf(stderr, "[ERROR]: args file \"%s\" not openened\n\tREASON: %s\n",
            path, strerror(errno));
    return NULL;
  };

  args_queue_t *queue = (args_queue_t *)malloc(sizeof(args_queue_t));
  if (!queue) {
    fprintf(stderr,
            "[ERROR]: couldn't allocate %lluB memory, try stack "
            "allocated\n\tREASON: "
            "%s\n",
            sizeof(args_queue_t), strerror(errno));
    return NULL;
  }

  queue->used = 0;
  queue->st = 0;
  int filesize = -1;
  if (fstat(fd, &file_stat) == 0) {
    queue->capacity = file_stat.st_size / 8;
    filesize = file_stat.st_size;
  } else {
    queue->capacity = ARGS_DEFAULT_INIT_SIZE;
  }

  queue->data = (args_t *)malloc(sizeof(args_t) * queue->capacity);
  if (!queue->data) {
    fprintf(stderr,
            "[ERROR]: couldn't allocate %lluB memory!\n\tREASON: "
            "%s\n",
            queue->capacity, strerror(errno));
    return NULL;
  }

  char buff[2048];
  const char *delim = " \n";
  while (fgets(buff, sizeof(buff) - 1, fp)) {
    args_t *arg = queue->data + queue->used++;
    char *working = strtok(buff, delim);
    for (arg->argc = 0; working; arg->argc++) {
      strcpy(arg->argv[arg->argc], working);
      working = strtok(NULL, delim);
    }
  }

  queue->data = realloc(queue->data, sizeof(*queue->data) * queue->used);
  return queue;
}

args_t *args_queue_pop(args_queue_t *queue) {
  return queue->data + atomic_fetch_add(&queue->st, 1);
}

int args_queue_size(args_queue_t *queue) { return queue->used - queue->st; }
