#include "report.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

const char cols_labels[35][22] = {
    "-cache:il1",
    "-cache:dl1",
    "-cache:il2",
    "-cache:dl2",
    "-tlb:dtlb",
    "-tlb:itlb",
    "benchmark",
    "sim_num_insn        ",
    "sim_num_refs        ",
    "sim_elapsed_time    ",
    "sim_inst_rate       ",
    "ul1.accesses        ",
    "ul1.hits            ",
    "ul1.misses          ",
    "ul1.replacements    ",
    "ul1.writebacks      ",
    "ul1.invalidations   ",
    "ul1.miss_rate       ",
    "ul1.repl_rate       ",
    "ul1.wb_rate         ",
    "ul1.inv_rate        ",
    "ld_text_base        ",
    "ld_text_size        ",
    "ld_data_base        ",
    "ld_data_size        ",
    "ld_stack_base       ",
    "ld_stack_size       ",
    "ld_prog_entry       ",
    "ld_environ_base     ",
    "ld_target_big_endian",
    "mem.page_count      ",
    "mem.page_mem        ",
    "mem.ptab_misses     ",
    "mem.ptab_accesses   ",
    "mem.ptab_miss_rate  ",
};

typedef struct atomic_stack_t {
  report_t *data;
  atomic_int used;

  atomic_int lock;
} atomic_stack_t;

atomic_stack_t *create_stack(size_t max_size) {
  atomic_stack_t *stack = malloc(sizeof(stack));
  if (!stack)
    return NULL;

  atomic_store(&stack->used, 0);
  atomic_store(&stack->lock, 0);
  stack->data = malloc(max_size * sizeof(*stack->data));
  if (!stack->data) {
    free(stack);
    return NULL;
  }

  return stack;
}

report_t *stack_push(atomic_stack_t *stack) {
  return stack->data + atomic_fetch_add(&stack->used, 1);
}

void stack_lock(atomic_stack_t *stack) { atomic_fetch_add(&stack->lock, 1); }

void stack_unlock(atomic_stack_t *stack) { atomic_fetch_sub(&stack->lock, 1); }

report_t *stack_pop(atomic_stack_t *stack) {
  return stack->data + atomic_fetch_sub(&stack->used, 1);
}

size_t stack_size(atomic_stack_t *stack) { return stack->used; }

void stack_to_csv(atomic_stack_t *stack, const char *filepath) {
  char mode[2] = {0};
  int csv_exists = access(filepath, F_OK) == 0;
  if (csv_exists) {
    mode[0] = 'a';
  } else {
    mode[0] = 'w';
  }
  FILE *fp = fopen(filepath, mode);

  const size_t cols = sizeof(cols_labels) / sizeof(cols_labels[0]);

  if (!csv_exists) {
    for (size_t i = 0; i < cols; i++) {
      fputs(cols_labels[i], fp);
      fputc(i == cols - 1 ? '\n' : ',', fp);
    }
  }

  while (stack->lock) {
    printf("waiting... lock=%d\n", stack->lock);
    sleep(5);
  }

  for (size_t row = 0; row < stack->used; row++) {
    for (size_t i = 0; i < cols; i++) {
      fputs(stack->data[row].df[i], fp);
      fputc(i == cols - 1 ? '\n' : ',', fp);
    }
  }

  fclose(fp);
}