#ifndef ARGS_H
#define ARGS_H

#include <stdint.h>

#ifndef ARGS_DEFAULT_INIT_SIZE
#define ARGS_DEFAULT_INIT_SIZE 8192
#endif

typedef struct args_t {
  int argc;
  char argv[20][255];
} args_t;

typedef struct args_queue_t args_queue_t;

args_queue_t *parse_args(const char *path);

args_t *args_queue_pop(args_queue_t *queue);
int args_queue_size(args_queue_t *queue);

#endif // ARGS_H