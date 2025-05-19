#ifndef ATOMIC_QUEUE_H
#define ATOMIC_QUEUE_H

#include <stddef.h>

#define COLS 40

extern const char cols_labels[35][22];

typedef struct report_t {
  char df[35][48];
  int used;
} report_t;

typedef struct atomic_stack_t atomic_stack_t;

atomic_stack_t *create_stack(size_t max_size);

report_t *stack_push(atomic_stack_t *stack);
report_t *stack_pop(atomic_stack_t *stack);
void stack_to_csv(atomic_stack_t *stack, const char *filepath);
size_t stack_size(atomic_stack_t *stack);

void stack_lock(atomic_stack_t *stack);
void stack_unlock(atomic_stack_t *stack);
#endif