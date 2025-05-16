gcc2_compiled.:
___gnu_compiled_c:
	.global _compilers
.data
	.align 4
_compilers:
	.word	LC0
	.word	LC1
	.word	LC2
	.word	LC3
	.word	LC4
	.word	LC5
	.word	LC6
	.word	LC7
	.word	LC8
	.word	LC9
	.word	0
	.word	0
.text
	.align 8
LC9:
	.ascii "cpp %{nostdinc} %{C} %{v} %{D*} %{U*} %{I*} %{M*} %{trigraphs}         -undef -D__GNUC__ -$ %p %P        %c %{O:-D__OPTIMIZE__} %{traditional} %{pedantic}\11%{Wcomment*} %{Wtrigraphs} %{Wall} %C        %i %{!M*:%{!E:%{!pipe:%g.cpp}}}%{E:%{o*}}%{M*:%{o*}} |\12    %{!M*:%{!E:%{!S:as %{R} %{j} %{J} %{h} %{d2} %a                     %{c:%{o*}%{!o*:-o %w%b.o}}%{!c:-o %d%w%b.o}\11\11    %{!pipe:%g.s}\12 }}}\0"
	.align 8
LC8:
	.ascii ".S\0"
	.align 8
LC7:
	.ascii "%{!S:as %{R} %{j} %{J} %{h} %{d2} %a             %{c:%{o*}%{!o*:-o %w%b.o}}%{!c:-o %d%w%b.o} %i\12 }\0"
	.align 8
LC6:
	.ascii ".s\0"
	.align 8
LC5:
	.ascii "cc1 %i %1 %{!Q:-quiet} %{Y*} %{d*} %{m*} %{f*} %{a}\11%{g} %{O} %{W*} %{w} %{pedantic} %{ansi} %{traditional}\11%{v:-version} %{gg:-symout %g.sym} %{pg:-p} %{p}\11%{S:%{o*}%{!o*:-o %b.s}}%{!S:-o %{|!pipe:%g.s}} |\12    %{!S:as %{R} %{j} %{J} %{h} %{d2} %a %{gg:-G %g.sym}            %{c:%{o*}%{!o*:-o %w%b.o}}%{!c:-o %d%w%b.o} %{!pipe:%g.s}\12 }\0"
	.align 8
LC4:
	.ascii ".i\0"
	.align 8
LC3:
	.ascii "cpp -+ %{nostdinc} %{C} %{v} %{D*} %{U*} %{I*} %{M*}         -undef -D__GNUC__ %p %P        %c %{O:-D__OPTIMIZE__} %{traditional} %{pedantic}\11%{Wcomment*} %{Wtrigraphs} %{Wall} %C        %i %{!M*:%{!E:%{!pipe:%g.cpp}}}%{E:%{o*}}%{M*:%{o*}} |\12    %{!M*:%{!E:cc1plus %{!pipe:%g.cpp} %1\11\11   %{!Q:-quiet} -dumpbase %i %{Y*} %{d*} %{m*} %{f*} %{a}\11\11   %{g} %{O} %{W*} %{w} %{pedantic} %{traditional}\11\11   %{v:-version} %{gg:-symout %g.sym} %{pg:-p} %{p}\11\11   %{pg:%{fomit-frame-pointer:%e-pg and -fomit-frame-pointer are incompatible}}\11\11   %{S:%{o*}%{!o*:-o %b.s}}%{!S:-o %{|!pipe:%g.s}} |\12              %{!S:as %{R} %{j} %{J} %{h} %{d2} %a %{gg:-G %g.sym}\11\11      %{c:%{o*}%{!o*:-o %w%b.o}}%{!c:-o %d%w%b.o}                      %{!pipe:%g.s}\12 }}}\0"
	.align 8
LC2:
	.ascii ".cc\0"
	.align 8
LC1:
	.ascii "cpp %{nostdinc} %{C} %{v} %{D*} %{U*} %{I*} %{M*} %{trigraphs} -undef         -D__GNUC__ %{ansi:-trigraphs -$ -D__STRICT_ANSI__} %{!ansi:%p} %P        %c %{O:-D__OPTIMIZE__} %{traditional} %{pedantic}\11%{Wcomment*} %{Wtrigraphs} %{Wall} %C        %i %{!M*:%{!E:%{!pipe:%g.cpp}}}%{E:%{o*}}%{M*:%{o*}} |\12    %{!M*:%{!E:cc1 %{!pipe:%g.cpp} %1 \11\11   %{!Q:-quiet} -dumpbase %i %{Y*} %{d*} %{m*} %{f*} %{a}\11\11   %{g} %{O} %{W*} %{w} %{pedantic} %{ansi} %{traditional}\11\11   %{v:-version} %{gg:-symout %g.sym} %{pg:-p} %{p}\11\11   %{pg:%{fomit-frame-pointer:%e-pg and -fomit-frame-pointer are incompatible}}\11\11   %{S:%{o*}%{!o*:-o %b.s}}%{!S:-o %{|!pipe:%g.s}} |\12              %{!S:as %{R} %{j} %{J} %{h} %{d2} %a %{gg:-G %g.sym}\11\11      %{c:%{o*}%{!o*:-o %w%b.o}}%{!c:-o %d%w%b.o}                      %{!pipe:%g.s}\12 }}}\0"
	.align 8
LC0:
	.ascii ".c\0"
	.global _link_spec
	.align 8
LC10:
	.ascii "%{!c:%{!M*:%{!E:%{!S:ld %{o*} %l %{A} %{d} %{e*} %{N} %{n} %{r} %{s} %{S} %{T*} %{t} %{u*} %{X} %{x} %{z} %{y*} %{!nostdlib:%S}  %{L*} %o %{!nostdlib:gnulib%s %{g:-lg} %L}\12 }}}}\0"
.data
	.align 4
_link_spec:
	.word	LC10
.text
	.align 4
	.global _record_temp_file
	.proc	020
_record_temp_file:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	ld [%fp+68],%o0
	call _strlen,0
	nop
	add %o0,1,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	mov %o0,%l0
	mov %l0,%o0
	ld [%fp+68],%o1
	call _strcpy,0
	nop
	ld [%fp+72],%o0
	cmp %o0,0
	be L2
	nop
	mov 8,%o0
	call _xmalloc,0
	nop
	mov %o0,%l1
	sethi %hi(_always_delete_queue),%o0
	ld [%o0+%lo(_always_delete_queue)],%o1
	st %o1,[%l1+4]
	st %l0,[%l1]
	sethi %hi(_always_delete_queue),%o0
	st %l1,[%o0+%lo(_always_delete_queue)]
L2:
	ld [%fp+76],%o0
	cmp %o0,0
	be L3
	nop
	mov 8,%o0
	call _xmalloc,0
	nop
	mov %o0,%l1
	sethi %hi(_failure_delete_queue),%o0
	ld [%o0+%lo(_failure_delete_queue)],%o1
	st %o1,[%l1+4]
	st %l0,[%l1]
	sethi %hi(_failure_delete_queue),%o0
	st %l1,[%o0+%lo(_failure_delete_queue)]
L3:
L1:
	ret
	restore
	.align 4
	.global _delete_temp_files
	.proc	020
_delete_temp_files:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	sethi %hi(_always_delete_queue),%o0
	ld [%o0+%lo(_always_delete_queue)],%l0
L5:
	cmp %l0,0
	be L6
	nop
	ld [%l0],%o0
	call _unlink,0
	nop
L7:
	ld [%l0+4],%l0
	b L5
	nop
L6:
	ld [%fp+68],%o0
	cmp %o0,0
	bne L8
	nop
	sethi %hi(_failure_delete_queue),%o0
	ld [%o0+%lo(_failure_delete_queue)],%l0
L9:
	cmp %l0,0
	be L10
	nop
	ld [%l0],%o0
	call _unlink,0
	nop
L11:
	ld [%l0+4],%l0
	b L9
	nop
L10:
L8:
	sethi %hi(_always_delete_queue),%o0
	st %g0,[%o0+%lo(_always_delete_queue)]
	sethi %hi(_failure_delete_queue),%o0
	st %g0,[%o0+%lo(_failure_delete_queue)]
L4:
	ret
	restore
	.align 4
	.global _clear_failure_queue
	.proc	020
_clear_failure_queue:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	sethi %hi(_failure_delete_queue),%o0
	st %g0,[%o0+%lo(_failure_delete_queue)]
L12:
	ret
	restore
	.align 8
LC11:
	.ascii "/tmp/ccXXXXXX\0"
	.align 4
	.global _choose_temp_base
	.proc	020
_choose_temp_base:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	sethi %hi(LC11),%o0
	or %o0,%lo(LC11),%l0
	sethi %hi(_temp_filename),%l1
	mov %l0,%o0
	call _strlen,0
	nop
	add %o0,1,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	st %o0,[%l1+%lo(_temp_filename)]
	sethi %hi(_temp_filename),%o1
	ld [%o1+%lo(_temp_filename)],%o0
	mov %l0,%o1
	call _strcpy,0
	nop
	sethi %hi(_temp_filename),%o1
	ld [%o1+%lo(_temp_filename)],%o0
	call _mktemp,0
	nop
	sethi %hi(_temp_filename_length),%l1
	sethi %hi(_temp_filename),%o1
	ld [%o1+%lo(_temp_filename)],%o0
	call _strlen,0
	nop
	st %o0,[%l1+%lo(_temp_filename_length)]
L13:
	ret
	restore
	.global _user_exec_prefix
.data
	.align 4
_user_exec_prefix:
	.word	0
	.global _env_exec_prefix
	.align 4
_env_exec_prefix:
	.word	0
	.global _standard_exec_prefix
.text
	.align 8
LC12:
	.ascii "/usr/local/lib/gcc-\0"
.data
	.align 4
_standard_exec_prefix:
	.word	LC12
	.global _standard_exec_prefix_1
.text
	.align 8
LC13:
	.ascii "/usr/lib/gcc-\0"
.data
	.align 4
_standard_exec_prefix_1:
	.word	LC13
	.global _standard_startfile_prefix
.text
	.align 8
LC14:
	.ascii "/usr/local/lib/\0"
.data
	.align 4
_standard_startfile_prefix:
	.word	LC14
	.global _standard_startfile_prefix_1
.text
	.align 8
LC15:
	.ascii "/lib/\0"
.data
	.align 4
_standard_startfile_prefix_1:
	.word	LC15
	.global _standard_startfile_prefix_2
.text
	.align 8
LC16:
	.ascii "/usr/lib/\0"
.data
	.align 4
_standard_startfile_prefix_2:
	.word	LC16
.text
	.align 4
	.global _clear_args
	.proc	020
_clear_args:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	sethi %hi(_argbuf_index),%o0
	st %g0,[%o0+%lo(_argbuf_index)]
L14:
	ret
	restore
	.align 4
	.global _store_arg
	.proc	020
_store_arg:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	sethi %hi(_argbuf_index),%o0
	ld [%o0+%lo(_argbuf_index)],%o1
	add %o1,1,%o0
	sethi %hi(_argbuf_length),%o1
	ld [%o1+%lo(_argbuf_length)],%o2
	cmp %o0,%o2
	bne L16
	nop
	sethi %hi(_argbuf),%l0
	sethi %hi(_argbuf),%o0
	sethi %hi(_argbuf_length),%o1
	sethi %hi(_argbuf_length),%o2
	ld [%o2+%lo(_argbuf_length)],%o3
	mov %o3,%o2
	sll %o2,1,%o3
	mov %o3,%o2
	st %o2,[%o1+%lo(_argbuf_length)]
	mov %o2,%o1
	sll %o1,2,%o2
	ld [%o0+%lo(_argbuf)],%o0
	mov %o2,%o1
	call _realloc,0
	nop
	st %o0,[%l0+%lo(_argbuf)]
L16:
	sethi %hi(_argbuf_index),%o0
	ld [%o0+%lo(_argbuf_index)],%o1
	add %o1,1,%o2
	st %o2,[%o0+%lo(_argbuf_index)]
	sll %o1,2,%o0
	sethi %hi(_argbuf),%o2
	ld [%o2+%lo(_argbuf)],%o1
	add %o0,%o1,%o0
	ld [%fp+68],%o1
	st %o1,[%o0]
	sethi %hi(_argbuf_index),%o1
	ld [%o1+%lo(_argbuf_index)],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_argbuf),%o2
	ld [%o2+%lo(_argbuf)],%o1
	add %o0,%o1,%o0
	st %g0,[%o0]
	ld [%fp+72],%o0
	cmp %o0,0
	bne L18
	nop
	ld [%fp+76],%o0
	cmp %o0,0
	bne L18
	nop
	b L17
	nop
L18:
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	call _record_temp_file,0
	nop
L17:
L15:
	ret
	restore
	.align 4
	.proc	0102
_find_exec_file:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %g0,[%fp-20]
	sethi %hi(_standard_exec_prefix),%o1
	ld [%o1+%lo(_standard_exec_prefix)],%o0
	call _strlen,0
	nop
	st %o0,[%fp-28]
	sethi %hi(_user_exec_prefix),%o0
	ld [%o0+%lo(_user_exec_prefix)],%o1
	cmp %o1,0
	be L20
	nop
	sethi %hi(_user_exec_prefix),%o1
	ld [%o1+%lo(_user_exec_prefix)],%o0
	call _strlen,0
	nop
	ld [%fp-28],%o1
	cmp %o0,%o1
	bleu L20
	nop
	sethi %hi(_user_exec_prefix),%o1
	ld [%o1+%lo(_user_exec_prefix)],%o0
	call _strlen,0
	nop
	st %o0,[%fp-28]
L20:
	sethi %hi(_env_exec_prefix),%o0
	ld [%o0+%lo(_env_exec_prefix)],%o1
	cmp %o1,0
	be L21
	nop
	sethi %hi(_env_exec_prefix),%o1
	ld [%o1+%lo(_env_exec_prefix)],%o0
	call _strlen,0
	nop
	ld [%fp-28],%o1
	cmp %o0,%o1
	bleu L21
	nop
	sethi %hi(_env_exec_prefix),%o1
	ld [%o1+%lo(_env_exec_prefix)],%o0
	call _strlen,0
	nop
	st %o0,[%fp-28]
L21:
	sethi %hi(_standard_exec_prefix_1),%o1
	ld [%o1+%lo(_standard_exec_prefix_1)],%o0
	call _strlen,0
	nop
	ld [%fp-28],%o1
	cmp %o0,%o1
	bleu L22
	nop
	sethi %hi(_standard_exec_prefix_1),%o1
	ld [%o1+%lo(_standard_exec_prefix_1)],%o0
	call _strlen,0
	nop
	st %o0,[%fp-28]
L22:
	ld [%fp-28],%o0
	add %o0,1,%l0
	ld [%fp+68],%o0
	call _strlen,0
	nop
	add %l0,%o0,%o1
	st %o1,[%fp-28]
	ld [%fp-28],%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-24]
	sethi %hi(_user_exec_prefix),%o0
	ld [%o0+%lo(_user_exec_prefix)],%o1
	cmp %o1,0
	be L23
	nop
	sethi %hi(_user_exec_prefix),%o1
	ld [%fp-24],%o0
	ld [%o1+%lo(_user_exec_prefix)],%o1
	call _strcpy,0
	nop
	ld [%fp-24],%o0
	ld [%fp+68],%o1
	call _strcat,0
	nop
	ld [%fp-24],%o0
	mov 1,%o1
	call _access,0
	nop
	xor %o0,0,%o1
	subcc %g0,%o1,%g0
	subx %g0,-1,%o0
	st %o0,[%fp-20]
L23:
	ld [%fp-20],%o0
	cmp %o0,0
	bne L24
	nop
	sethi %hi(_env_exec_prefix),%o0
	ld [%o0+%lo(_env_exec_prefix)],%o1
	cmp %o1,0
	be L24
	nop
	sethi %hi(_env_exec_prefix),%o1
	ld [%fp-24],%o0
	ld [%o1+%lo(_env_exec_prefix)],%o1
	call _strcpy,0
	nop
	ld [%fp-24],%o0
	ld [%fp+68],%o1
	call _strcat,0
	nop
	ld [%fp-24],%o0
	mov 1,%o1
	call _access,0
	nop
	xor %o0,0,%o1
	subcc %g0,%o1,%g0
	subx %g0,-1,%o0
	st %o0,[%fp-20]
L24:
	ld [%fp-20],%o0
	cmp %o0,0
	bne L25
	nop
	sethi %hi(_standard_exec_prefix),%o1
	ld [%fp-24],%o0
	ld [%o1+%lo(_standard_exec_prefix)],%o1
	call _strcpy,0
	nop
	ld [%fp-24],%o0
	ld [%fp+68],%o1
	call _strcat,0
	nop
	ld [%fp-24],%o0
	mov 1,%o1
	call _access,0
	nop
	xor %o0,0,%o1
	subcc %g0,%o1,%g0
	subx %g0,-1,%o0
	st %o0,[%fp-20]
L25:
	ld [%fp-20],%o0
	cmp %o0,0
	bne L26
	nop
	sethi %hi(_standard_exec_prefix_1),%o1
	ld [%fp-24],%o0
	ld [%o1+%lo(_standard_exec_prefix_1)],%o1
	call _strcpy,0
	nop
	sethi %hi(_argbuf),%o0
	ld [%o0+%lo(_argbuf)],%o1
	ld [%fp-24],%o0
	ld [%o1],%o1
	call _strcat,0
	nop
	ld [%fp-24],%o0
	mov 1,%o1
	call _access,0
	nop
	xor %o0,0,%o1
	subcc %g0,%o1,%g0
	subx %g0,-1,%o0
	st %o0,[%fp-20]
L26:
	ld [%fp-20],%o0
	cmp %o0,0
	be L27
	nop
	ld [%fp-24],%i0
	b L19
	nop
	b L28
	nop
L27:
	mov 0,%i0
	b L19
	nop
L28:
L19:
	ret
	restore
	.align 8
LC17:
	.ascii "pipe\0"
	.align 8
LC18:
	.ascii "vfork\0"
	.align 4
	.proc	04
_pexecute:
	!#PROLOGUE# 0
	save %sp,-136,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	sethi %hi(_last_pipe_input),%o0
	ld [%o0+%lo(_last_pipe_input)],%o1
	st %o1,[%fp-36]
	mov 1,%o0
	st %o0,[%fp-40]
	ld [%fp+80],%o0
	cmp %o0,0
	be L30
	nop
	add %fp,-32,%o1
	mov %o1,%o0
	call _pipe,0
	nop
	cmp %o0,0
	bge L31
	nop
	sethi %hi(LC17),%o1
	or %o1,%lo(LC17),%o0
	call _pfatal_with_name,0
	nop
L31:
	ld [%fp-28],%o0
	st %o0,[%fp-40]
	sethi %hi(_last_pipe_input),%o0
	ld [%fp-32],%o1
	st %o1,[%o0+%lo(_last_pipe_input)]
	b L32
	nop
L30:
	sethi %hi(_last_pipe_input),%o0
	st %g0,[%o0+%lo(_last_pipe_input)]
L32:
	call _fork,0
	nop
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	cmp %o0,-1
	be L34
	nop
	cmp %o0,0
	be L35
	nop
	b L39
	nop
L34:
	sethi %hi(LC18),%o1
	or %o1,%lo(LC18),%o0
	call _pfatal_with_name,0
	nop
	b L33
	nop
L35:
	ld [%fp-36],%o0
	cmp %o0,0
	be L36
	nop
	mov 0,%o0
	call _close,0
	nop
	ld [%fp-36],%o0
	call _dup,0
	nop
	ld [%fp-36],%o0
	call _close,0
	nop
L36:
	ld [%fp-40],%o0
	cmp %o0,1
	be L37
	nop
	mov 1,%o0
	call _close,0
	nop
	ld [%fp-40],%o0
	call _dup,0
	nop
	ld [%fp-40],%o0
	call _close,0
	nop
L37:
	sethi %hi(_last_pipe_input),%o0
	ld [%o0+%lo(_last_pipe_input)],%o1
	cmp %o1,0
	be L38
	nop
	sethi %hi(_last_pipe_input),%o1
	ld [%o1+%lo(_last_pipe_input)],%o0
	call _close,0
	nop
L38:
	ld [%fp+72],%o0
	ld [%fp+76],%o1
	ld [%fp+68],%o2
	call %o2,0
	nop
	ld [%fp+72],%o0
	call _perror_exec,0
	nop
	mov -1,%o0
	call _exit,0
	nop
L39:
	ld [%fp-36],%o0
	cmp %o0,0
	be L40
	nop
	ld [%fp-36],%o0
	call _close,0
	nop
L40:
	ld [%fp-40],%o0
	cmp %o0,1
	be L41
	nop
	ld [%fp-40],%o0
	call _close,0
	nop
L41:
	ld [%fp-20],%i0
	b L29
	nop
L33:
L29:
	ret
	restore
	.align 8
LC19:
	.ascii "|\0"
	.align 8
LC20:
	.ascii " %s\0"
	.align 8
LC21:
	.ascii " |\0"
	.align 8
LC22:
	.ascii "\12\0"
	.align 8
LC23:
	.ascii "Program %s got fatal signal %d.\0"
	.align 4
	.global _execute
	.proc	04
_execute:
	!#PROLOGUE# 0
	save %sp,-160,%sp
	!#PROLOGUE# 1
	nop
	mov 1,%o0
	st %o0,[%fp-28]
	st %g0,[%fp-20]
L44:
	sethi %hi(_argbuf_index),%o0
	ld [%fp-20],%o1
	ld [%o0+%lo(_argbuf_index)],%o0
	cmp %o1,%o0
	bge L45
	nop
	ld [%fp-20],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_argbuf),%o1
	ld [%o1+%lo(_argbuf)],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC19),%o2
	or %o2,%lo(LC19),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L47
	nop
	ld [%fp-28],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-28]
L47:
L46:
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L44
	nop
L45:
	ld [%fp-28],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	add %o2,7,%o0
	add %sp,92,%o2
	sub %o2,%sp,%o1
	add %o0,%o1,%o2
	mov %o2,%o0
	add %o0,7,%o0
	srl %o0,3,%o1
	mov %o1,%o0
	sll %o0,3,%o1
	sub %sp,%o1,%sp
	add %sp,92,%o1
	mov %o1,%o0
	add %o0,7,%o0
	srl %o0,3,%o1
	mov %o1,%o0
	sll %o0,3,%o1
	st %o1,[%fp-36]
	ld [%fp-36],%o0
	sethi %hi(_argbuf),%o2
	ld [%o2+%lo(_argbuf)],%o1
	ld [%o1],%o2
	st %o2,[%o0]
	ld [%fp-36],%o0
	sethi %hi(_argbuf),%o1
	ld [%o1+%lo(_argbuf)],%o2
	st %o2,[%o0+4]
	ld [%fp-36],%o1
	ld [%o1],%o0
	call _find_exec_file,0
	nop
	st %o0,[%fp-32]
	ld [%fp-32],%o0
	cmp %o0,0
	be L48
	nop
	ld [%fp-36],%o0
	ld [%o0+4],%o1
	ld [%fp-32],%o0
	st %o0,[%o1]
L48:
	nop
	mov 1,%o0
	st %o0,[%fp-28]
	st %g0,[%fp-20]
L49:
	sethi %hi(_argbuf_index),%o0
	ld [%fp-20],%o1
	ld [%o0+%lo(_argbuf_index)],%o0
	cmp %o1,%o0
	bge L50
	nop
	ld [%fp-20],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_argbuf),%o1
	ld [%o1+%lo(_argbuf)],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC19),%o2
	or %o2,%lo(LC19),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L52
	nop
	ld [%fp-20],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_argbuf),%o2
	ld [%o2+%lo(_argbuf)],%o1
	add %o0,%o1,%o0
	st %g0,[%o0]
	ld [%fp-28],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	ld [%fp-36],%o1
	add %o0,%o1,%o0
	ld [%fp-20],%o1
	mov %o1,%o2
	sll %o2,2,%o1
	sethi %hi(_argbuf),%o3
	ld [%o3+%lo(_argbuf)],%o2
	add %o1,%o2,%o1
	ld [%o1+4],%o2
	st %o2,[%o0]
	ld [%fp-28],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	ld [%fp-36],%o1
	add %o0,%o1,%o0
	sethi %hi(_argbuf),%o1
	ld [%fp-20],%o2
	mov %o2,%o3
	sll %o3,2,%o2
	add %o2,4,%o3
	ld [%o1+%lo(_argbuf)],%o1
	add %o1,%o3,%o2
	st %o2,[%o0+4]
	ld [%fp-28],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	ld [%fp-36],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	call _find_exec_file,0
	nop
	st %o0,[%fp-32]
	ld [%fp-32],%o0
	cmp %o0,0
	be L53
	nop
	ld [%fp-28],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	ld [%fp-36],%o1
	add %o0,%o1,%o0
	ld [%o0+4],%o1
	ld [%fp-32],%o0
	st %o0,[%o1]
L53:
	ld [%fp-28],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-28]
L52:
L51:
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L49
	nop
L50:
	sethi %hi(_argbuf_index),%o1
	ld [%o1+%lo(_argbuf_index)],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_argbuf),%o2
	ld [%o2+%lo(_argbuf)],%o1
	add %o0,%o1,%o0
	st %g0,[%o0]
	sethi %hi(_vflag),%o0
	ldub [%o0+%lo(_vflag)],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L54
	nop
	st %g0,[%fp-20]
L55:
	ld [%fp-20],%o0
	ld [%fp-28],%o1
	cmp %o0,%o1
	bge L56
	nop
	ld [%fp-20],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	ld [%fp-36],%o1
	add %o0,%o1,%o0
	ld [%o0+4],%o1
	st %o1,[%fp-40]
L58:
	ld [%fp-40],%o0
	ld [%o0],%o1
	cmp %o1,0
	be L59
	nop
	ld [%fp-40],%o2
	sethi %hi(__iob+40),%o1
	or %o1,%lo(__iob+40),%o0
	sethi %hi(LC20),%o3
	or %o3,%lo(LC20),%o1
	ld [%o2],%o2
	call _fprintf,0
	nop
L60:
	ld [%fp-40],%o1
	add %o1,4,%o0
	mov %o0,%o1
	st %o1,[%fp-40]
	b L58
	nop
L59:
	ld [%fp-20],%o1
	add %o1,1,%o0
	ld [%fp-28],%o1
	cmp %o0,%o1
	be L61
	nop
	sethi %hi(__iob+40),%o1
	or %o1,%lo(__iob+40),%o0
	sethi %hi(LC21),%o2
	or %o2,%lo(LC21),%o1
	call _fprintf,0
	nop
L61:
	sethi %hi(__iob+40),%o1
	or %o1,%lo(__iob+40),%o0
	sethi %hi(LC22),%o2
	or %o2,%lo(LC22),%o1
	call _fprintf,0
	nop
L57:
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L55
	nop
L56:
	sethi %hi(__iob+40),%o1
	or %o1,%lo(__iob+40),%o0
	call _fflush,0
	nop
L54:
	sethi %hi(_last_pipe_input),%o0
	st %g0,[%o0+%lo(_last_pipe_input)]
	st %g0,[%fp-20]
L62:
	ld [%fp-20],%o0
	ld [%fp-28],%o1
	cmp %o0,%o1
	bge L63
	nop
	ld [%fp-20],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	ld [%fp-36],%o1
	add %o0,%o1,%o0
	ld [%o0+4],%o1
	ld [%o1],%o0
	st %o0,[%fp-40]
	ld [%fp-20],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	ld [%fp-36],%o1
	add %o0,%o1,%o0
	ld [%fp-40],%o1
	ld [%o0],%o0
	cmp %o1,%o0
	be L65
	nop
	sethi %hi(_execv),%o1
	or %o1,%lo(_execv),%o0
	b L66
	nop
L65:
	sethi %hi(_execvp),%o1
	or %o1,%lo(_execvp),%o0
L66:
	ld [%fp-20],%o1
	mov %o1,%o3
	sll %o3,1,%o2
	add %o2,%o1,%o2
	sll %o2,2,%o1
	ld [%fp-36],%o3
	add %o1,%o3,%o2
	ld [%fp-20],%o3
	add %o3,1,%o1
	ld [%fp-28],%o3
	cmp %o1,%o3
	bl,a L79
	mov 1,%o3
	mov 0,%o3
L79:
	ld [%fp-40],%o1
	ld [%o2+4],%o2
	call _pexecute,0
	nop
	ld [%fp-20],%o1
	mov %o1,%o3
	sll %o3,1,%o2
	add %o2,%o1,%o2
	sll %o2,2,%o1
	ld [%fp-36],%o2
	add %o1,%o2,%o1
	st %o0,[%o1+8]
	ld [%fp-20],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	ld [%fp-36],%o1
	add %o0,%o1,%o0
	ld [%fp-40],%o1
	ld [%o0],%o0
	cmp %o1,%o0
	be L67
	nop
	ld [%fp-40],%o0
	call _free,0
	nop
L67:
L64:
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L62
	nop
L63:
	st %g0,[%fp-44]
	st %g0,[%fp-20]
L68:
	ld [%fp-20],%o0
	ld [%fp-28],%o1
	cmp %o0,%o1
	bge L69
	nop
	add %fp,-48,%o1
	mov %o1,%o0
	call _wait,0
	nop
	st %o0,[%fp-52]
	ld [%fp-52],%o0
	cmp %o0,0
	bge L71
	nop
	call _abort,0
	nop
L71:
	ld [%fp-48],%o0
	cmp %o0,0
	be L72
	nop
	st %g0,[%fp-60]
L73:
	ld [%fp-60],%o0
	ld [%fp-28],%o1
	cmp %o0,%o1
	bge L74
	nop
	ld [%fp-60],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	ld [%fp-36],%o1
	add %o0,%o1,%o0
	ld [%o0+8],%o1
	ld [%fp-52],%o0
	cmp %o1,%o0
	bne L76
	nop
	ld [%fp-60],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	ld [%fp-36],%o1
	add %o0,%o1,%o0
	ld [%o0],%o1
	st %o1,[%fp-56]
L76:
L75:
	ld [%fp-60],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-60]
	b L73
	nop
L74:
	ld [%fp-48],%o1
	and %o1,127,%o0
	cmp %o0,0
	be L77
	nop
	ld [%fp-48],%o0
	and %o0,127,%o2
	sethi %hi(LC23),%o1
	or %o1,%lo(LC23),%o0
	ld [%fp-56],%o1
	call _fatal,0
	nop
L77:
	ld [%fp-48],%o0
	sethi %hi(65280),%o2
	or %o2,%lo(65280),%o1
	and %o0,%o1,%o0
	sra %o0,8,%o1
	cmp %o1,0
	ble L78
	nop
	mov -1,%o0
	st %o0,[%fp-44]
L78:
L72:
L70:
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L68
	nop
L69:
	ld [%fp-44],%i0
	b L43
	nop
L43:
	ret
	restore
	.align 8
LC24:
	.ascii "GCC_EXEC_PREFIX\0"
	.align 8
LC25:
	.ascii "Tdata\0"
	.align 4
	.global _process_command
	.proc	020
_process_command:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	sethi %hi(_n_switches),%o0
	st %g0,[%o0+%lo(_n_switches)]
	sethi %hi(_n_infiles),%o0
	st %g0,[%o0+%lo(_n_infiles)]
	sethi %hi(LC24),%o1
	or %o1,%lo(LC24),%o0
	call _getenv,0
	nop
	sethi %hi(_env_exec_prefix),%o1
	st %o0,[%o1+%lo(_env_exec_prefix)]
	mov 1,%l0
L81:
	ld [%fp+68],%o0
	cmp %l0,%o0
	bge L82
	nop
	mov %l0,%o0
	sll %o0,2,%o1
	ld [%fp+72],%o2
	add %o1,%o2,%o0
	ld [%o0],%o1
	ldub [%o1],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,45
	bne L84
	nop
	mov %l0,%o0
	sll %o0,2,%o1
	ld [%fp+72],%o2
	add %o1,%o2,%o0
	mov 1,%o1
	ld [%o0],%o2
	add %o1,%o2,%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,108
	be L84
	nop
	mov %l0,%o0
	sll %o0,2,%o1
	ld [%fp+72],%o2
	add %o1,%o2,%o0
	ld [%o0],%o1
	add %o1,1,%l1
	ldub [%l1],%o0
	sll %o0,24,%o1
	sra %o1,24,%l2
	cmp %l2,66
	be L86
	nop
	cmp %l2,118
	be L87
	nop
	b L88
	nop
L86:
	sethi %hi(_user_exec_prefix),%o0
	add %l1,1,%o1
	st %o1,[%o0+%lo(_user_exec_prefix)]
	b L85
	nop
L87:
	sethi %hi(_vflag),%o1
	sethi %hi(_vflag),%o0
	sethi %hi(_vflag),%o1
	ldub [%o1+%lo(_vflag)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	stb %o2,[%o0+%lo(_vflag)]
	sethi %hi(_n_switches),%o1
	sethi %hi(_n_switches),%o0
	sethi %hi(_n_switches),%o1
	ld [%o1+%lo(_n_switches)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_n_switches)]
	b L85
	nop
L88:
	sethi %hi(_n_switches),%o1
	sethi %hi(_n_switches),%o0
	sethi %hi(_n_switches),%o1
	ld [%o1+%lo(_n_switches)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_n_switches)]
	cmp %l2,68
	be L90
	nop
	cmp %l2,85
	be L90
	nop
	cmp %l2,111
	be L90
	nop
	cmp %l2,101
	be L90
	nop
	cmp %l2,84
	be L90
	nop
	cmp %l2,117
	be L90
	nop
	cmp %l2,73
	be L90
	nop
	cmp %l2,89
	be L90
	nop
	cmp %l2,109
	be L90
	nop
	cmp %l2,76
	be L90
	nop
	b L89
	nop
L90:
	ldub [%l1+1],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	bne L89
	nop
	add %l0,1,%l0
	b L91
	nop
L89:
	mov %l1,%o0
	sethi %hi(LC25),%o2
	or %o2,%lo(LC25),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L92
	nop
	add %l0,1,%l0
L92:
L91:
L85:
	b L94
	nop
L84:
	sethi %hi(_n_infiles),%o1
	sethi %hi(_n_infiles),%o0
	sethi %hi(_n_infiles),%o1
	ld [%o1+%lo(_n_infiles)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_n_infiles)]
L94:
L83:
	add %l0,1,%l0
	b L81
	nop
L82:
	sethi %hi(_switches),%l1
	sethi %hi(_n_switches),%o0
	ld [%o0+%lo(_n_switches)],%o1
	add %o1,1,%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	mov %o2,%o0
	call _xmalloc,0
	nop
	st %o0,[%l1+%lo(_switches)]
	sethi %hi(_infiles),%l1
	sethi %hi(_n_infiles),%o0
	ld [%o0+%lo(_n_infiles)],%o1
	add %o1,1,%o0
	mov %o0,%o1
	sll %o1,2,%o2
	mov %o2,%o0
	call _xmalloc,0
	nop
	st %o0,[%l1+%lo(_infiles)]
	sethi %hi(_n_switches),%o0
	st %g0,[%o0+%lo(_n_switches)]
	sethi %hi(_n_infiles),%o0
	st %g0,[%o0+%lo(_n_infiles)]
	mov 1,%l0
L95:
	ld [%fp+68],%o0
	cmp %l0,%o0
	bge L96
	nop
	mov %l0,%o0
	sll %o0,2,%o1
	ld [%fp+72],%o2
	add %o1,%o2,%o0
	ld [%o0],%o1
	ldub [%o1],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,45
	bne L98
	nop
	mov %l0,%o0
	sll %o0,2,%o1
	ld [%fp+72],%o2
	add %o1,%o2,%o0
	mov 1,%o1
	ld [%o0],%o2
	add %o1,%o2,%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,108
	be L98
	nop
	mov %l0,%o0
	sll %o0,2,%o1
	ld [%fp+72],%o2
	add %o1,%o2,%o0
	ld [%o0],%o1
	add %o1,1,%l1
	ldub [%l1],%o0
	sll %o0,24,%o1
	sra %o1,24,%l2
	cmp %l2,66
	bne L99
	nop
	b L97
	nop
L99:
	sethi %hi(_n_switches),%o1
	ld [%o1+%lo(_n_switches)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	st %l1,[%o0]
	cmp %l2,68
	be L103
	nop
	cmp %l2,85
	be L103
	nop
	cmp %l2,111
	be L103
	nop
	cmp %l2,101
	be L103
	nop
	cmp %l2,84
	be L103
	nop
	cmp %l2,117
	be L103
	nop
	cmp %l2,73
	be L103
	nop
	cmp %l2,89
	be L103
	nop
	cmp %l2,109
	be L103
	nop
	cmp %l2,76
	be L103
	nop
	b L102
	nop
L103:
	ldub [%l1+1],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	bne L102
	nop
	b L101
	nop
L102:
	mov %l1,%o0
	sethi %hi(LC25),%o2
	or %o2,%lo(LC25),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L100
	nop
	b L101
	nop
L101:
	sethi %hi(_n_switches),%o1
	ld [%o1+%lo(_n_switches)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	add %l0,1,%l0
	mov %l0,%o1
	sll %o1,2,%o2
	ld [%fp+72],%o3
	add %o2,%o3,%o1
	ld [%o1],%o2
	st %o2,[%o0+4]
	b L104
	nop
L100:
	sethi %hi(_n_switches),%o1
	ld [%o1+%lo(_n_switches)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	st %g0,[%o0+4]
L104:
	sethi %hi(_n_switches),%o1
	ld [%o1+%lo(_n_switches)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	st %g0,[%o0+8]
	sethi %hi(_n_switches),%o1
	sethi %hi(_n_switches),%o0
	sethi %hi(_n_switches),%o1
	ld [%o1+%lo(_n_switches)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_n_switches)]
	b L105
	nop
L98:
	sethi %hi(_n_infiles),%o0
	ld [%o0+%lo(_n_infiles)],%o1
	add %o1,1,%o2
	st %o2,[%o0+%lo(_n_infiles)]
	sll %o1,2,%o0
	sethi %hi(_infiles),%o2
	ld [%o2+%lo(_infiles)],%o1
	add %o0,%o1,%o0
	mov %l0,%o1
	sll %o1,2,%o2
	ld [%fp+72],%o3
	add %o2,%o3,%o1
	ld [%o1],%o2
	st %o2,[%o0]
L105:
L97:
	add %l0,1,%l0
	b L95
	nop
L96:
	sethi %hi(_n_switches),%o1
	ld [%o1+%lo(_n_switches)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	st %g0,[%o0]
	sethi %hi(_n_infiles),%o1
	ld [%o1+%lo(_n_infiles)],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_infiles),%o2
	ld [%o2+%lo(_infiles)],%o1
	add %o0,%o1,%o0
	st %g0,[%o0]
L80:
	ret
	restore
	.align 4
	.global _do_spec
	.proc	04
_do_spec:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	call _clear_args,0
	nop
	sethi %hi(_arg_going),%o0
	st %g0,[%o0+%lo(_arg_going)]
	sethi %hi(_delete_this_arg),%o0
	st %g0,[%o0+%lo(_delete_this_arg)]
	sethi %hi(_this_is_output_file),%o0
	st %g0,[%o0+%lo(_this_is_output_file)]
	sethi %hi(_this_is_library_file),%o0
	st %g0,[%o0+%lo(_this_is_library_file)]
	ld [%fp+68],%o0
	mov 0,%o1
	call _do_spec_1,0
	nop
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	cmp %o0,0
	bne L107
	nop
	sethi %hi(_argbuf_index),%o0
	ld [%o0+%lo(_argbuf_index)],%o1
	cmp %o1,0
	ble L108
	nop
	sethi %hi(_argbuf_index),%o1
	ld [%o1+%lo(_argbuf_index)],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_argbuf),%o1
	ld [%o1+%lo(_argbuf)],%o2
	add %o0,%o2,%o1
	ld [%o1-4],%o0
	sethi %hi(LC19),%o2
	or %o2,%lo(LC19),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L108
	nop
	sethi %hi(_argbuf_index),%o1
	sethi %hi(_argbuf_index),%o0
	sethi %hi(_argbuf_index),%o1
	ld [%o1+%lo(_argbuf_index)],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_argbuf_index)]
L108:
	sethi %hi(_argbuf_index),%o0
	ld [%o0+%lo(_argbuf_index)],%o1
	cmp %o1,0
	ble L109
	nop
	call _execute,0
	nop
	st %o0,[%fp-20]
L109:
L107:
	ld [%fp-20],%i0
	b L106
	nop
L106:
	ret
	restore
	.align 8
LC26:
	.ascii "Invalid specification!  Bug in cc.\0"
	.align 8
LC27:
	.ascii "%s\0"
	.align 8
LC28:
	.ascii "\0"
	.align 8
LC29:
	.ascii "%{m68000:-mc68010}%{mc68000:-mc68010}%{!mc68000:%{!m68000:-mc68020}}\0"
	.align 8
LC30:
	.ascii "%{funsigned-char:-D__CHAR_UNSIGNED__}\0"
	.align 8
LC31:
	.ascii "%{!msoft-float:%{mfpa:-D__HAVE_FPA__ }%{!mfpa:-D__HAVE_68881__ }}%{!ansi:%{m68000:-Dmc68010}%{mc68000:-Dmc68010}%{!mc68000:%{!m68000:-Dmc68020}}}\0"
	.align 8
LC32:
	.ascii "%{!p:%{!pg:-lc}}%{p:-lc_p}%{pg:-lc_p} %{a:/usr/lib/bb_link.o} \0"
	.align 8
LC33:
	.ascii "-Dmc68000 -Dsun -Dunix\0"
	.align 8
LC34:
	.ascii "-D\0"
	.align 8
LC35:
	.ascii "%{pg:gcrt0.o%s}%{!pg:%{p:mcrt0.o%s}%{!p:crt0.o%s}}\11   %{mfpa:Wcrt1.o%s}\11\11\11\11\11   %{msoft-float:Fcrt1.o%s}\11\11\11\11   %{!mfpa:%{!msoft-float:Mcrt1.o%s}}\0"
	.align 4
	.global _do_spec_1
	.proc	04
_do_spec_1:
	!#PROLOGUE# 0
	save %sp,-176,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%l0
L111:
	ldub [%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%l1
	mov %l1,%o0
	add %l0,1,%l0
	cmp %o0,0
	be L112
	nop
	ld [%fp+72],%o0
	cmp %o0,0
	bne L201
	nop
	mov %l1,%o0
	b L202
	nop
L201:
	mov 97,%o0
L202:
	cmp %o0,37
	be L150
	nop
	cmp %o0,37
	bg L203
	nop
	cmp %o0,32
	be L142
	nop
	cmp %o0,32
	bg L197
	nop
	cmp %o0,9
	be L141
	nop
	cmp %o0,10
	be L114
	nop
	b L197
	nop
L203:
	cmp %o0,124
	be L131
	nop
	b L197
	nop
L114:
	sethi %hi(_arg_going),%o0
	ld [%o0+%lo(_arg_going)],%o1
	cmp %o1,0
	be L115
	nop
	sethi %hi(_obstack),%o1
	or %o1,%lo(_obstack),%o0
	st %o0,[%fp-24]
	ld [%fp-24],%o0
	ld [%o0+12],%o1
	add %o1,1,%o0
	ld [%fp-24],%o1
	ld [%o1+16],%o2
	cmp %o0,%o2
	bleu L116
	nop
	ld [%fp-24],%o0
	mov 1,%o1
	call __obstack_newchunk,0
	nop
	b L117
	nop
L116:
L117:
	ld [%fp-24],%o0
	ld [%o0+12],%o1
	add %o1,1,%o2
	st %o2,[%o0+12]
	stb %g0,[%o1]
	sethi %hi(_obstack),%o1
	or %o1,%lo(_obstack),%o0
	st %o0,[%fp-28]
	ld [%fp-28],%o0
	ld [%o0+8],%o1
	st %o1,[%fp-32]
	ld [%fp-28],%o0
	ld [%fp-28],%o1
	ld [%fp-28],%o2
	ld [%o1+12],%o1
	ld [%o2+24],%o2
	add %o1,%o2,%o1
	ld [%fp-28],%o2
	ld [%o2+24],%o3
	xnor %g0,%o3,%o2
	and %o1,%o2,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	ld [%fp-28],%o0
	ld [%fp-28],%o1
	ld [%o0+12],%o0
	ld [%o1+4],%o1
	sub %o0,%o1,%o0
	ld [%fp-28],%o1
	ld [%fp-28],%o2
	ld [%o1+16],%o1
	ld [%o2+4],%o2
	sub %o1,%o2,%o1
	cmp %o0,%o1
	ble L118
	nop
	ld [%fp-28],%o0
	ld [%fp-28],%o1
	ld [%o1+16],%o2
	st %o2,[%o0+12]
	mov %o2,%o0
	b L119
	nop
L118:
	mov 0,%o0
L119:
	ld [%fp-28],%o0
	ld [%fp-28],%o1
	ld [%o1+12],%o2
	st %o2,[%o0+8]
	ld [%fp-32],%o0
	st %o0,[%fp-20]
	sethi %hi(_this_is_library_file),%o0
	ld [%o0+%lo(_this_is_library_file)],%o1
	cmp %o1,0
	be L120
	nop
	ld [%fp-20],%o0
	call _find_file,0
	nop
	st %o0,[%fp-20]
L120:
	sethi %hi(_delete_this_arg),%o1
	sethi %hi(_this_is_output_file),%o2
	ld [%fp-20],%o0
	ld [%o1+%lo(_delete_this_arg)],%o1
	ld [%o2+%lo(_this_is_output_file)],%o2
	call _store_arg,0
	nop
	sethi %hi(_this_is_output_file),%o0
	ld [%o0+%lo(_this_is_output_file)],%o1
	cmp %o1,0
	be L121
	nop
	sethi %hi(_input_file_number),%o1
	ld [%o1+%lo(_input_file_number)],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_outfiles),%o2
	ld [%o2+%lo(_outfiles)],%o1
	add %o0,%o1,%o0
	ld [%fp-20],%o1
	st %o1,[%o0]
L121:
L115:
	sethi %hi(_arg_going),%o0
	st %g0,[%o0+%lo(_arg_going)]
	sethi %hi(_argbuf_index),%o0
	ld [%o0+%lo(_argbuf_index)],%o1
	cmp %o1,0
	ble L122
	nop
	sethi %hi(_argbuf_index),%o1
	ld [%o1+%lo(_argbuf_index)],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_argbuf),%o1
	ld [%o1+%lo(_argbuf)],%o2
	add %o0,%o2,%o1
	ld [%o1-4],%o0
	sethi %hi(LC19),%o2
	or %o2,%lo(LC19),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L122
	nop
	st %g0,[%fp-32]
L123:
	sethi %hi(_n_switches),%o0
	ld [%fp-32],%o1
	ld [%o0+%lo(_n_switches)],%o0
	cmp %o1,%o0
	bge L124
	nop
	ld [%fp-32],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o1
	ld [%o1+%lo(_switches)],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC17),%o2
	or %o2,%lo(LC17),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L126
	nop
	b L124
	nop
L126:
L125:
	ld [%fp-32],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-32]
	b L123
	nop
L124:
	sethi %hi(_n_switches),%o0
	ld [%fp-32],%o1
	ld [%o0+%lo(_n_switches)],%o0
	cmp %o1,%o0
	bge L127
	nop
	ld [%fp-32],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	mov 1,%o1
	st %o1,[%o0+8]
	b L113
	nop
	b L128
	nop
L127:
	sethi %hi(_argbuf_index),%o1
	sethi %hi(_argbuf_index),%o0
	sethi %hi(_argbuf_index),%o1
	ld [%o1+%lo(_argbuf_index)],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_argbuf_index)]
L128:
L122:
	sethi %hi(_argbuf_index),%o0
	ld [%o0+%lo(_argbuf_index)],%o1
	cmp %o1,0
	ble L129
	nop
	call _execute,0
	nop
	st %o0,[%fp-32]
	ld [%fp-32],%o0
	cmp %o0,0
	be L130
	nop
	ld [%fp-32],%i0
	b L110
	nop
L130:
L129:
	call _clear_args,0
	nop
	sethi %hi(_arg_going),%o0
	st %g0,[%o0+%lo(_arg_going)]
	sethi %hi(_delete_this_arg),%o0
	st %g0,[%o0+%lo(_delete_this_arg)]
	sethi %hi(_this_is_output_file),%o0
	st %g0,[%o0+%lo(_this_is_output_file)]
	sethi %hi(_this_is_library_file),%o0
	st %g0,[%o0+%lo(_this_is_library_file)]
	b L113
	nop
L131:
	sethi %hi(_arg_going),%o0
	ld [%o0+%lo(_arg_going)],%o1
	cmp %o1,0
	be L132
	nop
	sethi %hi(_obstack),%o1
	or %o1,%lo(_obstack),%o0
	st %o0,[%fp-28]
	ld [%fp-28],%o0
	ld [%o0+12],%o1
	add %o1,1,%o0
	ld [%fp-28],%o1
	ld [%o1+16],%o2
	cmp %o0,%o2
	bleu L133
	nop
	ld [%fp-28],%o0
	mov 1,%o1
	call __obstack_newchunk,0
	nop
	b L134
	nop
L133:
L134:
	ld [%fp-28],%o0
	ld [%o0+12],%o1
	add %o1,1,%o2
	st %o2,[%o0+12]
	stb %g0,[%o1]
	sethi %hi(_obstack),%o1
	or %o1,%lo(_obstack),%o0
	st %o0,[%fp-24]
	ld [%fp-24],%o0
	ld [%o0+8],%o1
	st %o1,[%fp-36]
	ld [%fp-24],%o0
	ld [%fp-24],%o1
	ld [%fp-24],%o2
	ld [%o1+12],%o1
	ld [%o2+24],%o2
	add %o1,%o2,%o1
	ld [%fp-24],%o2
	ld [%o2+24],%o3
	xnor %g0,%o3,%o2
	and %o1,%o2,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	ld [%fp-24],%o0
	ld [%fp-24],%o1
	ld [%o0+12],%o0
	ld [%o1+4],%o1
	sub %o0,%o1,%o0
	ld [%fp-24],%o1
	ld [%fp-24],%o2
	ld [%o1+16],%o1
	ld [%o2+4],%o2
	sub %o1,%o2,%o1
	cmp %o0,%o1
	ble L135
	nop
	ld [%fp-24],%o0
	ld [%fp-24],%o1
	ld [%o1+16],%o2
	st %o2,[%o0+12]
	mov %o2,%o0
	b L136
	nop
L135:
	mov 0,%o0
L136:
	ld [%fp-24],%o0
	ld [%fp-24],%o1
	ld [%o1+12],%o2
	st %o2,[%o0+8]
	ld [%fp-36],%o0
	st %o0,[%fp-20]
	sethi %hi(_this_is_library_file),%o0
	ld [%o0+%lo(_this_is_library_file)],%o1
	cmp %o1,0
	be L137
	nop
	ld [%fp-20],%o0
	call _find_file,0
	nop
	st %o0,[%fp-20]
L137:
	sethi %hi(_delete_this_arg),%o1
	sethi %hi(_this_is_output_file),%o2
	ld [%fp-20],%o0
	ld [%o1+%lo(_delete_this_arg)],%o1
	ld [%o2+%lo(_this_is_output_file)],%o2
	call _store_arg,0
	nop
	sethi %hi(_this_is_output_file),%o0
	ld [%o0+%lo(_this_is_output_file)],%o1
	cmp %o1,0
	be L138
	nop
	sethi %hi(_input_file_number),%o1
	ld [%o1+%lo(_input_file_number)],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_outfiles),%o2
	ld [%o2+%lo(_outfiles)],%o1
	add %o0,%o1,%o0
	ld [%fp-20],%o1
	st %o1,[%o0]
L138:
L132:
	sethi %hi(_obstack),%o1
	or %o1,%lo(_obstack),%o0
	st %o0,[%fp-36]
	ld [%fp-36],%o0
	ld [%o0+12],%o1
	add %o1,1,%o0
	ld [%fp-36],%o1
	ld [%o1+16],%o2
	cmp %o0,%o2
	bleu L139
	nop
	ld [%fp-36],%o0
	mov 1,%o1
	call __obstack_newchunk,0
	nop
	b L140
	nop
L139:
L140:
	ld [%fp-36],%o1
	ld [%o1+12],%o0
	add %o0,1,%o2
	st %o2,[%o1+12]
	mov %l1,%o1
	stb %o1,[%o0]
	sethi %hi(_arg_going),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_arg_going)]
	b L113
	nop
L141:
L142:
	sethi %hi(_arg_going),%o0
	ld [%o0+%lo(_arg_going)],%o1
	cmp %o1,0
	be L143
	nop
	sethi %hi(_obstack),%o1
	or %o1,%lo(_obstack),%o0
	st %o0,[%fp-28]
	ld [%fp-28],%o0
	ld [%o0+12],%o1
	add %o1,1,%o0
	ld [%fp-28],%o1
	ld [%o1+16],%o2
	cmp %o0,%o2
	bleu L144
	nop
	ld [%fp-28],%o0
	mov 1,%o1
	call __obstack_newchunk,0
	nop
	b L145
	nop
L144:
L145:
	ld [%fp-28],%o0
	ld [%o0+12],%o1
	add %o1,1,%o2
	st %o2,[%o0+12]
	stb %g0,[%o1]
	sethi %hi(_obstack),%o1
	or %o1,%lo(_obstack),%o0
	st %o0,[%fp-24]
	ld [%fp-24],%o0
	ld [%o0+8],%o1
	st %o1,[%fp-40]
	ld [%fp-24],%o0
	ld [%fp-24],%o1
	ld [%fp-24],%o2
	ld [%o1+12],%o1
	ld [%o2+24],%o2
	add %o1,%o2,%o1
	ld [%fp-24],%o2
	ld [%o2+24],%o3
	xnor %g0,%o3,%o2
	and %o1,%o2,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	ld [%fp-24],%o0
	ld [%fp-24],%o1
	ld [%o0+12],%o0
	ld [%o1+4],%o1
	sub %o0,%o1,%o0
	ld [%fp-24],%o1
	ld [%fp-24],%o2
	ld [%o1+16],%o1
	ld [%o2+4],%o2
	sub %o1,%o2,%o1
	cmp %o0,%o1
	ble L146
	nop
	ld [%fp-24],%o0
	ld [%fp-24],%o1
	ld [%o1+16],%o2
	st %o2,[%o0+12]
	mov %o2,%o0
	b L147
	nop
L146:
	mov 0,%o0
L147:
	ld [%fp-24],%o0
	ld [%fp-24],%o1
	ld [%o1+12],%o2
	st %o2,[%o0+8]
	ld [%fp-40],%o0
	st %o0,[%fp-20]
	sethi %hi(_this_is_library_file),%o0
	ld [%o0+%lo(_this_is_library_file)],%o1
	cmp %o1,0
	be L148
	nop
	ld [%fp-20],%o0
	call _find_file,0
	nop
	st %o0,[%fp-20]
L148:
	sethi %hi(_delete_this_arg),%o1
	sethi %hi(_this_is_output_file),%o2
	ld [%fp-20],%o0
	ld [%o1+%lo(_delete_this_arg)],%o1
	ld [%o2+%lo(_this_is_output_file)],%o2
	call _store_arg,0
	nop
	sethi %hi(_this_is_output_file),%o0
	ld [%o0+%lo(_this_is_output_file)],%o1
	cmp %o1,0
	be L149
	nop
	sethi %hi(_input_file_number),%o1
	ld [%o1+%lo(_input_file_number)],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_outfiles),%o2
	ld [%o2+%lo(_outfiles)],%o1
	add %o0,%o1,%o0
	ld [%fp-20],%o1
	st %o1,[%o0]
L149:
L143:
	sethi %hi(_arg_going),%o0
	st %g0,[%o0+%lo(_arg_going)]
	sethi %hi(_delete_this_arg),%o0
	st %g0,[%o0+%lo(_delete_this_arg)]
	sethi %hi(_this_is_output_file),%o0
	st %g0,[%o0+%lo(_this_is_output_file)]
	sethi %hi(_this_is_library_file),%o0
	st %g0,[%o0+%lo(_this_is_library_file)]
	b L113
	nop
L150:
	ldub [%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%l1
	mov %l1,%o0
	add %l0,1,%l0
	mov 123,%o1
	cmp %o1,%o0
	blu L195
	nop
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(L196),%o2
	or %o2,%lo(L196),%o1
	ld [%o0+%o1],%o0
	jmp %o0
	nop
L196:
	.word	L152
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L174
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L177
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L180
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L182
	.word	L195
	.word	L195
	.word	L195
	.word	L184
	.word	L195
	.word	L195
	.word	L194
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L195
	.word	L178
	.word	L153
	.word	L179
	.word	L156
	.word	L157
	.word	L195
	.word	L160
	.word	L195
	.word	L163
	.word	L195
	.word	L195
	.word	L181
	.word	L195
	.word	L195
	.word	L166
	.word	L183
	.word	L195
	.word	L195
	.word	L170
	.word	L195
	.word	L195
	.word	L195
	.word	L171
	.word	L195
	.word	L195
	.word	L195
	.word	L172
L152:
	sethi %hi(LC26),%o1
	or %o1,%lo(LC26),%o0
	call _fatal,0
	nop
L153:
	sethi %hi(_obstack),%o1
	or %o1,%lo(_obstack),%o0
	st %o0,[%fp-40]
	sethi %hi(_basename_length),%o0
	ld [%o0+%lo(_basename_length)],%o1
	st %o1,[%fp-28]
	ld [%fp-40],%o0
	ld [%o0+12],%o1
	ld [%fp-28],%o2
	add %o1,%o2,%o0
	ld [%fp-40],%o1
	ld [%o1+16],%o2
	cmp %o0,%o2
	bleu L154
	nop
	ld [%fp-40],%o0
	ld [%fp-28],%o1
	call __obstack_newchunk,0
	nop
	b L155
	nop
L154:
	mov 0,%o0
L155:
	ld [%fp-40],%o0
	sethi %hi(_input_basename),%o1
	ld [%o0+12],%o0
	ld [%o1+%lo(_input_basename)],%o1
	ld [%fp-28],%o2
	call _memcpy,0
	nop
	mov %o0,%o1
	ld [%fp-40],%o0
	ld [%fp-40],%o1
	ld [%o1+12],%o2
	ld [%fp-28],%o3
	add %o2,%o3,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	sethi %hi(_arg_going),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_arg_going)]
	b L151
	nop
L156:
	sethi %hi(_delete_this_arg),%o0
	mov 2,%o1
	st %o1,[%o0+%lo(_delete_this_arg)]
	b L151
	nop
L157:
	st %l0,[%fp-24]
L158:
	ldub [%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	be L159
	nop
	ldub [%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,10
	be L159
	nop
	add %l0,1,%l0
	b L158
	nop
L159:
	ld [%fp-24],%o1
	sub %l0,%o1,%o0
	add %o0,1,%o1
	add %o1,7,%o0
	add %sp,92,%o2
	sub %o2,%sp,%o1
	add %o0,%o1,%o2
	mov %o2,%o0
	add %o0,7,%o0
	srl %o0,3,%o1
	mov %o1,%o0
	sll %o0,3,%o1
	sub %sp,%o1,%sp
	add %sp,92,%o1
	mov %o1,%o0
	add %o0,7,%o0
	srl %o0,3,%o1
	mov %o1,%o0
	sll %o0,3,%o1
	st %o1,[%fp-44]
	ld [%fp-24],%o0
	sub %l0,%o0,%o2
	ld [%fp-44],%o0
	ld [%fp-24],%o1
	call _strncpy,0
	nop
	sethi %hi(LC27),%o1
	or %o1,%lo(LC27),%o0
	ld [%fp-44],%o1
	call _error,0
	nop
	mov -1,%i0
	b L110
	nop
	b L151
	nop
L160:
	sethi %hi(_obstack),%o1
	or %o1,%lo(_obstack),%o0
	st %o0,[%fp-48]
	sethi %hi(_temp_filename_length),%o0
	ld [%o0+%lo(_temp_filename_length)],%o1
	st %o1,[%fp-52]
	ld [%fp-48],%o0
	ld [%o0+12],%o1
	ld [%fp-52],%o2
	add %o1,%o2,%o0
	ld [%fp-48],%o1
	ld [%o1+16],%o2
	cmp %o0,%o2
	bleu L161
	nop
	ld [%fp-48],%o0
	ld [%fp-52],%o1
	call __obstack_newchunk,0
	nop
	b L162
	nop
L161:
	mov 0,%o0
L162:
	ld [%fp-48],%o0
	sethi %hi(_temp_filename),%o1
	ld [%o0+12],%o0
	ld [%o1+%lo(_temp_filename)],%o1
	ld [%fp-52],%o2
	call _memcpy,0
	nop
	mov %o0,%o1
	ld [%fp-48],%o0
	ld [%fp-48],%o1
	ld [%o1+12],%o2
	ld [%fp-52],%o3
	add %o2,%o3,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	sethi %hi(_delete_this_arg),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_delete_this_arg)]
	sethi %hi(_arg_going),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_arg_going)]
	b L151
	nop
L163:
	sethi %hi(_obstack),%o1
	or %o1,%lo(_obstack),%o0
	st %o0,[%fp-56]
	sethi %hi(_input_filename_length),%o0
	ld [%o0+%lo(_input_filename_length)],%o1
	st %o1,[%fp-60]
	ld [%fp-56],%o0
	ld [%o0+12],%o1
	ld [%fp-60],%o2
	add %o1,%o2,%o0
	ld [%fp-56],%o1
	ld [%o1+16],%o2
	cmp %o0,%o2
	bleu L164
	nop
	ld [%fp-56],%o0
	ld [%fp-60],%o1
	call __obstack_newchunk,0
	nop
	b L165
	nop
L164:
	mov 0,%o0
L165:
	ld [%fp-56],%o0
	sethi %hi(_input_filename),%o1
	ld [%o0+12],%o0
	ld [%o1+%lo(_input_filename)],%o1
	ld [%fp-60],%o2
	call _memcpy,0
	nop
	mov %o0,%o1
	ld [%fp-56],%o0
	ld [%fp-56],%o1
	ld [%o1+12],%o2
	ld [%fp-60],%o3
	add %o2,%o3,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	sethi %hi(_arg_going),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_arg_going)]
	b L151
	nop
L166:
	mov 0,%l2
L167:
	sethi %hi(_n_infiles),%o0
	ld [%o0+%lo(_n_infiles)],%o1
	cmp %l2,%o1
	bge L168
	nop
	mov %l2,%o1
	sll %o1,2,%o0
	sethi %hi(_outfiles),%o1
	ld [%o1+%lo(_outfiles)],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	mov 0,%o1
	mov 0,%o2
	call _store_arg,0
	nop
L169:
	add %l2,1,%l2
	b L167
	nop
L168:
	b L151
	nop
L170:
	sethi %hi(_this_is_library_file),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_this_is_library_file)]
	b L151
	nop
L171:
	sethi %hi(_this_is_output_file),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_this_is_output_file)]
	b L151
	nop
L172:
	mov %l0,%o0
	call _handle_braces,0
	nop
	mov %o0,%l0
	cmp %l0,0
	bne L173
	nop
	mov -1,%i0
	b L110
	nop
L173:
	b L151
	nop
L174:
	sethi %hi(_obstack),%o1
	or %o1,%lo(_obstack),%o0
	st %o0,[%fp-64]
	ld [%fp-64],%o0
	ld [%o0+12],%o1
	add %o1,1,%o0
	ld [%fp-64],%o1
	ld [%o1+16],%o2
	cmp %o0,%o2
	bleu L175
	nop
	ld [%fp-64],%o0
	mov 1,%o1
	call __obstack_newchunk,0
	nop
	b L176
	nop
L175:
L176:
	ld [%fp-64],%o1
	ld [%o1+12],%o0
	add %o0,1,%o2
	st %o2,[%o1+12]
	mov 37,%o1
	stb %o1,[%o0]
	b L151
	nop
L177:
	sethi %hi(LC28),%o1
	or %o1,%lo(LC28),%o0
	mov 0,%o1
	call _do_spec_1,0
	nop
	b L151
	nop
L178:
	sethi %hi(LC29),%o1
	or %o1,%lo(LC29),%o0
	mov 0,%o1
	call _do_spec_1,0
	nop
	b L151
	nop
L179:
	sethi %hi(LC30),%o1
	or %o1,%lo(LC30),%o0
	mov 0,%o1
	call _do_spec_1,0
	nop
	b L151
	nop
L180:
	sethi %hi(LC31),%o1
	or %o1,%lo(LC31),%o0
	mov 0,%o1
	call _do_spec_1,0
	nop
	b L151
	nop
L181:
	sethi %hi(LC28),%o1
	or %o1,%lo(LC28),%o0
	mov 0,%o1
	call _do_spec_1,0
	nop
	b L151
	nop
L182:
	sethi %hi(LC32),%o1
	or %o1,%lo(LC32),%o0
	mov 0,%o1
	call _do_spec_1,0
	nop
	b L151
	nop
L183:
	sethi %hi(LC33),%o1
	or %o1,%lo(LC33),%o0
	mov 0,%o1
	call _do_spec_1,0
	nop
	b L151
	nop
L184:
	sethi %hi(LC33),%o1
	or %o1,%lo(LC33),%o0
	call _strlen,0
	nop
	mov %o0,%o1
	sll %o1,1,%o0
	add %o0,1,%o1
	add %o1,7,%o0
	add %sp,92,%o2
	sub %o2,%sp,%o1
	add %o0,%o1,%o2
	mov %o2,%o0
	add %o0,7,%o0
	srl %o0,3,%o1
	mov %o1,%o0
	sll %o0,3,%o1
	sub %sp,%o1,%sp
	add %sp,92,%o1
	mov %o1,%o0
	add %o0,7,%o0
	srl %o0,3,%o1
	mov %o1,%o0
	sll %o0,3,%o1
	st %o1,[%fp-68]
	ld [%fp-68],%o0
	st %o0,[%fp-72]
	sethi %hi(LC33),%o1
	or %o1,%lo(LC33),%o0
	st %o0,[%fp-76]
L185:
	ld [%fp-76],%o0
	sethi %hi(LC34),%o2
	or %o2,%lo(LC34),%o1
	mov 2,%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L187
	nop
	ld [%fp-68],%o0
	add %o0,1,%o1
	st %o1,[%fp-68]
	mov 45,%o1
	stb %o1,[%o0]
	ld [%fp-68],%o0
	add %o0,1,%o1
	st %o1,[%fp-68]
	mov 68,%o1
	stb %o1,[%o0]
	ld [%fp-68],%o0
	add %o0,1,%o1
	st %o1,[%fp-68]
	mov 95,%o1
	stb %o1,[%o0]
	ld [%fp-68],%o0
	add %o0,1,%o1
	st %o1,[%fp-68]
	mov 95,%o1
	stb %o1,[%o0]
	ld [%fp-76],%o0
	add %o0,2,%o1
	st %o1,[%fp-76]
	b L188
	nop
L187:
	ld [%fp-76],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,32
	be L190
	nop
	ld [%fp-76],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	bne L189
	nop
	b L190
	nop
L190:
	ld [%fp-68],%o0
	add %o0,1,%o1
	st %o1,[%fp-68]
	mov 95,%o1
	stb %o1,[%o0]
	ld [%fp-68],%o0
	add %o0,1,%o1
	st %o1,[%fp-68]
	mov 95,%o1
	stb %o1,[%o0]
	ld [%fp-76],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	bne L191
	nop
	b L186
	nop
	b L192
	nop
L191:
	ld [%fp-68],%o0
	add %o0,1,%o1
	st %o1,[%fp-68]
	ld [%fp-76],%o1
	add %o1,1,%o2
	st %o2,[%fp-76]
	ldub [%o1],%o1
	stb %o1,[%o0]
L192:
	b L193
	nop
L189:
	ld [%fp-68],%o0
	add %o0,1,%o1
	st %o1,[%fp-68]
	ld [%fp-76],%o1
	add %o1,1,%o2
	st %o2,[%fp-76]
	ldub [%o1],%o1
	stb %o1,[%o0]
L193:
L188:
	b L185
	nop
L186:
	ld [%fp-68],%o0
	stb %g0,[%o0]
	ld [%fp-72],%o0
	mov 0,%o1
	call _do_spec_1,0
	nop
	b L151
	nop
L194:
	sethi %hi(LC35),%o1
	or %o1,%lo(LC35),%o0
	mov 0,%o1
	call _do_spec_1,0
	nop
	b L151
	nop
L195:
	call _abort,0
	nop
L151:
	b L113
	nop
L197:
	sethi %hi(_obstack),%o1
	or %o1,%lo(_obstack),%o0
	st %o0,[%fp-76]
	ld [%fp-76],%o0
	ld [%o0+12],%o1
	add %o1,1,%o0
	ld [%fp-76],%o1
	ld [%o1+16],%o2
	cmp %o0,%o2
	bleu L198
	nop
	ld [%fp-76],%o0
	mov 1,%o1
	call __obstack_newchunk,0
	nop
	b L199
	nop
L198:
L199:
	ld [%fp-76],%o1
	ld [%o1+12],%o0
	add %o0,1,%o2
	st %o2,[%o1+12]
	mov %l1,%o1
	stb %o1,[%o0]
	sethi %hi(_arg_going),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_arg_going)]
L113:
	b L111
	nop
L112:
	mov 0,%i0
	b L110
	nop
L110:
	ret
	restore
	.align 8
LC36:
	.ascii "-\0"
	.align 4
	.global _handle_braces
	.proc	0102
_handle_braces:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	mov %i0,%l0
	st %g0,[%fp-24]
	st %g0,[%fp-28]
	ldub [%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,124
	bne L205
	nop
	mov 1,%o0
	st %o0,[%fp-24]
	add %l0,1,%l0
L205:
	ldub [%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,33
	bne L206
	nop
	mov 1,%o0
	st %o0,[%fp-28]
	add %l0,1,%l0
L206:
	st %l0,[%fp-20]
L207:
	ldub [%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,58
	be L208
	nop
	ldub [%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,125
	be L208
	nop
	add %l0,1,%l0
	b L207
	nop
L208:
	ldub [%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,125
	be L209
	nop
	mov 1,%l2
	add %l0,1,%l1
L210:
	cmp %l2,0
	ble L211
	nop
	ldub [%l1],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,123
	bne L212
	nop
	add %l2,1,%l2
	b L213
	nop
L212:
	ldub [%l1],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,125
	bne L214
	nop
	add %l2,-1,%l2
	b L215
	nop
L214:
	ldub [%l1],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	bne L216
	nop
	call _abort,0
	nop
L216:
L215:
L213:
	add %l1,1,%l1
	b L210
	nop
L211:
	b L217
	nop
L209:
	add %l0,1,%l1
L217:
	ldub [%l0-1],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,42
	bne L218
	nop
	ldub [%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,125
	bne L218
	nop
	add %l0,-1,%l0
	mov 0,%l2
L219:
	sethi %hi(_n_switches),%o0
	ld [%o0+%lo(_n_switches)],%o1
	cmp %l2,%o1
	bge L220
	nop
	mov %l2,%o0
	sll %o0,1,%o1
	add %o1,%l2,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	ld [%fp-20],%o1
	sub %l0,%o1,%o2
	ld [%o0],%o0
	ld [%fp-20],%o1
	call _strncmp,0
	nop
	cmp %o0,0
	bne L222
	nop
	mov %l2,%o0
	call _give_switch,0
	nop
L222:
L221:
	add %l2,1,%l2
	b L219
	nop
L220:
	b L223
	nop
L218:
	st %g0,[%fp-32]
	ldub [%l0-1],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,42
	bne L224
	nop
	mov 0,%l2
L225:
	sethi %hi(_n_switches),%o0
	ld [%o0+%lo(_n_switches)],%o1
	cmp %l2,%o1
	bge L226
	nop
	mov %l2,%o0
	sll %o0,1,%o1
	add %o1,%l2,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	ld [%fp-20],%o2
	sub %l0,%o2,%o1
	add %o1,-1,%o2
	ld [%o0],%o0
	ld [%fp-20],%o1
	call _strncmp,0
	nop
	cmp %o0,0
	bne L228
	nop
	mov %l2,%o0
	sll %o0,1,%o1
	add %o1,%l2,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	mov 1,%o1
	st %o1,[%o0+8]
	mov 1,%o0
	st %o0,[%fp-32]
L228:
L227:
	add %l2,1,%l2
	b L225
	nop
L226:
	b L229
	nop
L224:
	mov 0,%l2
L230:
	sethi %hi(_n_switches),%o0
	ld [%o0+%lo(_n_switches)],%o1
	cmp %l2,%o1
	bge L231
	nop
	mov %l2,%o0
	sll %o0,1,%o1
	add %o1,%l2,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	ld [%fp-20],%o1
	sub %l0,%o1,%o2
	ld [%o0],%o0
	ld [%fp-20],%o1
	call _strncmp,0
	nop
	cmp %o0,0
	bne L233
	nop
	mov %l2,%o0
	sll %o0,1,%o1
	add %o1,%l2,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	ld [%fp-20],%o2
	sub %l0,%o2,%o1
	ld [%o0],%o2
	add %o1,%o2,%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	bne L233
	nop
	mov %l2,%o0
	sll %o0,1,%o1
	add %o1,%l2,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	mov 1,%o1
	st %o1,[%o0+8]
	mov 1,%o0
	st %o0,[%fp-32]
	b L231
	nop
L233:
L232:
	add %l2,1,%l2
	b L230
	nop
L231:
L229:
	ld [%fp-32],%o0
	ld [%fp-28],%o1
	cmp %o0,%o1
	be L234
	nop
	ldub [%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,125
	bne L235
	nop
	mov %l2,%o0
	call _give_switch,0
	nop
	b L236
	nop
L235:
	add %l0,1,%o0
	sub %l1,%l0,%o1
	add %o1,-2,%o2
	mov %o2,%o1
	call _save_string,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	mov 0,%o1
	call _do_spec_1,0
	nop
	cmp %o0,0
	bge L237
	nop
	mov 0,%i0
	b L204
	nop
L237:
L236:
	b L238
	nop
L234:
	ld [%fp-24],%o0
	cmp %o0,0
	be L239
	nop
	sethi %hi(LC36),%o1
	or %o1,%lo(LC36),%o0
	call _do_spec_1,0
	nop
L239:
L238:
L223:
	mov %l1,%i0
	b L204
	nop
L204:
	ret
	restore
	.align 8
LC37:
	.ascii " \0"
	.align 4
	.global _give_switch
	.proc	04
_give_switch:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	sethi %hi(LC36),%o1
	or %o1,%lo(LC36),%o0
	mov 0,%o1
	call _do_spec_1,0
	nop
	ld [%fp+68],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o1
	ld [%o1+%lo(_switches)],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	mov 1,%o1
	call _do_spec_1,0
	nop
	sethi %hi(LC37),%o1
	or %o1,%lo(LC37),%o0
	mov 0,%o1
	call _do_spec_1,0
	nop
	ld [%fp+68],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	ld [%o0+4],%o1
	cmp %o1,0
	be L241
	nop
	ld [%fp+68],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o1
	ld [%o1+%lo(_switches)],%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	mov 1,%o1
	call _do_spec_1,0
	nop
	sethi %hi(LC37),%o1
	or %o1,%lo(LC37),%o0
	mov 0,%o1
	call _do_spec_1,0
	nop
L241:
	ld [%fp+68],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	mov 1,%o1
	st %o1,[%o0+8]
L240:
	ret
	restore
	.align 8
LC38:
	.ascii "./\0"
	.align 4
	.global _find_file
	.proc	0102
_find_file:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %g0,[%fp-28]
	sethi %hi(_standard_exec_prefix),%o1
	ld [%o1+%lo(_standard_exec_prefix)],%o0
	call _strlen,0
	nop
	st %o0,[%fp-20]
	sethi %hi(_user_exec_prefix),%o0
	ld [%o0+%lo(_user_exec_prefix)],%o1
	cmp %o1,0
	be L243
	nop
	sethi %hi(_user_exec_prefix),%o1
	ld [%o1+%lo(_user_exec_prefix)],%o0
	call _strlen,0
	nop
	ld [%fp-20],%o1
	cmp %o0,%o1
	bleu L243
	nop
	sethi %hi(_user_exec_prefix),%o1
	ld [%o1+%lo(_user_exec_prefix)],%o0
	call _strlen,0
	nop
	st %o0,[%fp-20]
L243:
	sethi %hi(_env_exec_prefix),%o0
	ld [%o0+%lo(_env_exec_prefix)],%o1
	cmp %o1,0
	be L244
	nop
	sethi %hi(_env_exec_prefix),%o1
	ld [%o1+%lo(_env_exec_prefix)],%o0
	call _strlen,0
	nop
	ld [%fp-20],%o1
	cmp %o0,%o1
	bleu L244
	nop
	sethi %hi(_env_exec_prefix),%o1
	ld [%o1+%lo(_env_exec_prefix)],%o0
	call _strlen,0
	nop
	st %o0,[%fp-20]
L244:
	sethi %hi(_standard_exec_prefix),%o1
	ld [%o1+%lo(_standard_exec_prefix)],%o0
	call _strlen,0
	nop
	ld [%fp-20],%o1
	cmp %o0,%o1
	bleu L245
	nop
	sethi %hi(_standard_exec_prefix),%o1
	ld [%o1+%lo(_standard_exec_prefix)],%o0
	call _strlen,0
	nop
	st %o0,[%fp-20]
L245:
	sethi %hi(_standard_exec_prefix_1),%o1
	ld [%o1+%lo(_standard_exec_prefix_1)],%o0
	call _strlen,0
	nop
	ld [%fp-20],%o1
	cmp %o0,%o1
	bleu L246
	nop
	sethi %hi(_standard_exec_prefix_1),%o1
	ld [%o1+%lo(_standard_exec_prefix_1)],%o0
	call _strlen,0
	nop
	st %o0,[%fp-20]
L246:
	sethi %hi(_standard_startfile_prefix),%o1
	ld [%o1+%lo(_standard_startfile_prefix)],%o0
	call _strlen,0
	nop
	ld [%fp-20],%o1
	cmp %o0,%o1
	bleu L247
	nop
	sethi %hi(_standard_startfile_prefix),%o1
	ld [%o1+%lo(_standard_startfile_prefix)],%o0
	call _strlen,0
	nop
	st %o0,[%fp-20]
L247:
	sethi %hi(_standard_startfile_prefix_1),%o1
	ld [%o1+%lo(_standard_startfile_prefix_1)],%o0
	call _strlen,0
	nop
	ld [%fp-20],%o1
	cmp %o0,%o1
	bleu L248
	nop
	sethi %hi(_standard_startfile_prefix_1),%o1
	ld [%o1+%lo(_standard_startfile_prefix_1)],%o0
	call _strlen,0
	nop
	st %o0,[%fp-20]
L248:
	sethi %hi(_standard_startfile_prefix_2),%o1
	ld [%o1+%lo(_standard_startfile_prefix_2)],%o0
	call _strlen,0
	nop
	ld [%fp-20],%o1
	cmp %o0,%o1
	bleu L249
	nop
	sethi %hi(_standard_startfile_prefix_2),%o1
	ld [%o1+%lo(_standard_startfile_prefix_2)],%o0
	call _strlen,0
	nop
	st %o0,[%fp-20]
L249:
	ld [%fp-20],%o0
	add %o0,1,%l0
	ld [%fp+68],%o0
	call _strlen,0
	nop
	add %l0,%o0,%o1
	st %o1,[%fp-20]
	ld [%fp-20],%o1
	add %o1,7,%o0
	add %sp,92,%o2
	sub %o2,%sp,%o1
	add %o0,%o1,%o2
	mov %o2,%o0
	add %o0,7,%o0
	srl %o0,3,%o1
	mov %o1,%o0
	sll %o0,3,%o1
	sub %sp,%o1,%sp
	add %sp,92,%o1
	mov %o1,%o0
	add %o0,7,%o0
	srl %o0,3,%o1
	mov %o1,%o0
	sll %o0,3,%o1
	st %o1,[%fp-24]
	sethi %hi(_user_exec_prefix),%o0
	ld [%o0+%lo(_user_exec_prefix)],%o1
	cmp %o1,0
	be L250
	nop
	sethi %hi(_user_exec_prefix),%o1
	ld [%fp-24],%o0
	ld [%o1+%lo(_user_exec_prefix)],%o1
	call _strcpy,0
	nop
	ld [%fp-24],%o0
	ld [%fp+68],%o1
	call _strcat,0
	nop
	ld [%fp-24],%o0
	mov 4,%o1
	call _access,0
	nop
	xor %o0,0,%o1
	subcc %g0,%o1,%g0
	subx %g0,-1,%o0
	st %o0,[%fp-28]
L250:
	ld [%fp-28],%o0
	cmp %o0,0
	bne L251
	nop
	sethi %hi(_env_exec_prefix),%o0
	ld [%o0+%lo(_env_exec_prefix)],%o1
	cmp %o1,0
	be L251
	nop
	sethi %hi(_env_exec_prefix),%o1
	ld [%fp-24],%o0
	ld [%o1+%lo(_env_exec_prefix)],%o1
	call _strcpy,0
	nop
	ld [%fp-24],%o0
	ld [%fp+68],%o1
	call _strcat,0
	nop
	ld [%fp-24],%o0
	mov 4,%o1
	call _access,0
	nop
	xor %o0,0,%o1
	subcc %g0,%o1,%g0
	subx %g0,-1,%o0
	st %o0,[%fp-28]
L251:
	ld [%fp-28],%o0
	cmp %o0,0
	bne L252
	nop
	sethi %hi(_standard_exec_prefix),%o1
	ld [%fp-24],%o0
	ld [%o1+%lo(_standard_exec_prefix)],%o1
	call _strcpy,0
	nop
	ld [%fp-24],%o0
	ld [%fp+68],%o1
	call _strcat,0
	nop
	ld [%fp-24],%o0
	mov 4,%o1
	call _access,0
	nop
	xor %o0,0,%o1
	subcc %g0,%o1,%g0
	subx %g0,-1,%o0
	st %o0,[%fp-28]
L252:
	ld [%fp-28],%o0
	cmp %o0,0
	bne L253
	nop
	sethi %hi(_standard_exec_prefix_1),%o1
	ld [%fp-24],%o0
	ld [%o1+%lo(_standard_exec_prefix_1)],%o1
	call _strcpy,0
	nop
	ld [%fp-24],%o0
	ld [%fp+68],%o1
	call _strcat,0
	nop
	ld [%fp-24],%o0
	mov 4,%o1
	call _access,0
	nop
	xor %o0,0,%o1
	subcc %g0,%o1,%g0
	subx %g0,-1,%o0
	st %o0,[%fp-28]
L253:
	ld [%fp-28],%o0
	cmp %o0,0
	bne L254
	nop
	sethi %hi(_standard_startfile_prefix),%o1
	ld [%fp-24],%o0
	ld [%o1+%lo(_standard_startfile_prefix)],%o1
	call _strcpy,0
	nop
	ld [%fp-24],%o0
	ld [%fp+68],%o1
	call _strcat,0
	nop
	ld [%fp-24],%o0
	mov 4,%o1
	call _access,0
	nop
	xor %o0,0,%o1
	subcc %g0,%o1,%g0
	subx %g0,-1,%o0
	st %o0,[%fp-28]
L254:
	ld [%fp-28],%o0
	cmp %o0,0
	bne L255
	nop
	sethi %hi(_standard_startfile_prefix_1),%o1
	ld [%fp-24],%o0
	ld [%o1+%lo(_standard_startfile_prefix_1)],%o1
	call _strcpy,0
	nop
	ld [%fp-24],%o0
	ld [%fp+68],%o1
	call _strcat,0
	nop
	ld [%fp-24],%o0
	mov 4,%o1
	call _access,0
	nop
	xor %o0,0,%o1
	subcc %g0,%o1,%g0
	subx %g0,-1,%o0
	st %o0,[%fp-28]
L255:
	ld [%fp-28],%o0
	cmp %o0,0
	bne L256
	nop
	sethi %hi(_standard_startfile_prefix_2),%o1
	ld [%fp-24],%o0
	ld [%o1+%lo(_standard_startfile_prefix_2)],%o1
	call _strcpy,0
	nop
	ld [%fp-24],%o0
	ld [%fp+68],%o1
	call _strcat,0
	nop
	ld [%fp-24],%o0
	mov 4,%o1
	call _access,0
	nop
	xor %o0,0,%o1
	subcc %g0,%o1,%g0
	subx %g0,-1,%o0
	st %o0,[%fp-28]
L256:
	ld [%fp-28],%o0
	cmp %o0,0
	bne L257
	nop
	ld [%fp-24],%o0
	sethi %hi(LC38),%o2
	or %o2,%lo(LC38),%o1
	call _strcpy,0
	nop
	ld [%fp-24],%o0
	ld [%fp+68],%o1
	call _strcat,0
	nop
	ld [%fp-24],%o0
	mov 4,%o1
	call _access,0
	nop
	xor %o0,0,%o1
	subcc %g0,%o1,%g0
	subx %g0,-1,%o0
	st %o0,[%fp-28]
L257:
	ld [%fp-28],%o0
	cmp %o0,0
	be L258
	nop
	ld [%fp-24],%o0
	call _strlen,0
	nop
	mov %o0,%o1
	ld [%fp-24],%o0
	call _save_string,0
	nop
	mov %o0,%i0
	b L242
	nop
L258:
	ld [%fp+68],%i0
	b L242
	nop
L242:
	ret
	restore
	.align 4
	.global _fatal_error
	.proc	020
_fatal_error:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	mov 0,%o1
	call _signal,0
	nop
	mov 0,%o0
	call _delete_temp_files,0
	nop
	call _getpid,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp+68],%o1
	call _kill,0
	nop
L259:
	ret
	restore
	.align 8
LC39:
	.ascii "gcc version %s\12\0"
	.align 8
LC40:
	.ascii "No input files specified.\0"
	.align 8
LC41:
	.ascii "%s: linker input file unused since linking not done\0"
	.align 8
LC42:
	.ascii "unrecognized option `-%s'\0"
	.align 4
	.global _main
	.proc	04
_main:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	call ___main,0
	nop
	st %g0,[%fp-24]
	st %g0,[%fp-28]
	sethi %hi(_programname),%o0
	ld [%fp+72],%o1
	ld [%o1],%o2
	st %o2,[%o0+%lo(_programname)]
	mov 2,%o0
	mov 1,%o1
	call _signal,0
	nop
	cmp %o0,1
	be L261
	nop
	mov 2,%o0
	sethi %hi(_fatal_error),%o2
	or %o2,%lo(_fatal_error),%o1
	call _signal,0
	nop
L261:
	mov 1,%o0
	mov 1,%o1
	call _signal,0
	nop
	cmp %o0,1
	be L262
	nop
	mov 1,%o0
	sethi %hi(_fatal_error),%o2
	or %o2,%lo(_fatal_error),%o1
	call _signal,0
	nop
L262:
	mov 15,%o0
	mov 1,%o1
	call _signal,0
	nop
	cmp %o0,1
	be L263
	nop
	mov 15,%o0
	sethi %hi(_fatal_error),%o2
	or %o2,%lo(_fatal_error),%o1
	call _signal,0
	nop
L263:
	sethi %hi(_argbuf_length),%o0
	mov 10,%o1
	st %o1,[%o0+%lo(_argbuf_length)]
	sethi %hi(_argbuf),%l1
	sethi %hi(_argbuf_length),%o0
	ld [%o0+%lo(_argbuf_length)],%o1
	mov %o1,%o0
	sll %o0,2,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	st %o0,[%l1+%lo(_argbuf)]
	sethi %hi(_obstack),%o1
	or %o1,%lo(_obstack),%o0
	mov 0,%o1
	mov 0,%o2
	sethi %hi(_xmalloc),%o4
	or %o4,%lo(_xmalloc),%o3
	sethi %hi(_free),%o5
	or %o5,%lo(_free),%o4
	call __obstack_begin,0
	nop
	call _choose_temp_base,0
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	call _process_command,0
	nop
	sethi %hi(_vflag),%o0
	ldub [%o0+%lo(_vflag)],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L264
	nop
	sethi %hi(_version_string),%o2
	sethi %hi(__iob+40),%o1
	or %o1,%lo(__iob+40),%o0
	sethi %hi(LC39),%o3
	or %o3,%lo(LC39),%o1
	ld [%o2+%lo(_version_string)],%o2
	call _fprintf,0
	nop
	sethi %hi(_n_infiles),%o0
	ld [%o0+%lo(_n_infiles)],%o1
	cmp %o1,0
	bne L265
	nop
	mov 0,%o0
	call _exit,0
	nop
L265:
L264:
	sethi %hi(_n_infiles),%o0
	ld [%o0+%lo(_n_infiles)],%o1
	cmp %o1,0
	bne L266
	nop
	sethi %hi(LC40),%o1
	or %o1,%lo(LC40),%o0
	call _fatal,0
	nop
L266:
	sethi %hi(_outfiles),%l1
	sethi %hi(_n_infiles),%o0
	ld [%o0+%lo(_n_infiles)],%o1
	mov %o1,%o0
	sll %o0,2,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	st %o0,[%l1+%lo(_outfiles)]
	sethi %hi(_outfiles),%o0
	sethi %hi(_n_infiles),%o1
	ld [%o1+%lo(_n_infiles)],%o2
	mov %o2,%o1
	sll %o1,2,%o2
	ld [%o0+%lo(_outfiles)],%o0
	mov 0,%o1
	call _memset,0
	nop
	mov 0,%l0
L267:
	sethi %hi(_n_infiles),%o0
	ld [%o0+%lo(_n_infiles)],%o1
	cmp %l0,%o1
	bge L268
	nop
	sethi %hi(_input_filename),%o0
	mov %l0,%o2
	sll %o2,2,%o1
	sethi %hi(_infiles),%o3
	ld [%o3+%lo(_infiles)],%o2
	add %o1,%o2,%o1
	ld [%o1],%o2
	st %o2,[%o0+%lo(_input_filename)]
	sethi %hi(_input_filename_length),%l2
	sethi %hi(_input_filename),%o1
	ld [%o1+%lo(_input_filename)],%o0
	call _strlen,0
	nop
	st %o0,[%l2+%lo(_input_filename_length)]
	sethi %hi(_input_file_number),%o0
	st %l0,[%o0+%lo(_input_file_number)]
	mov %l0,%o1
	sll %o1,2,%o0
	sethi %hi(_outfiles),%o2
	ld [%o2+%lo(_outfiles)],%o1
	add %o0,%o1,%o0
	sethi %hi(_input_filename),%o1
	ld [%o1+%lo(_input_filename)],%o2
	st %o2,[%o0]
	sethi %hi(_compilers),%o0
	or %o0,%lo(_compilers),%l1
L270:
	ld [%l1+4],%o0
	cmp %o0,0
	be L271
	nop
	ld [%l1],%o0
	call _strlen,0
	nop
	sethi %hi(_input_filename_length),%o1
	ld [%o1+%lo(_input_filename_length)],%o2
	cmp %o0,%o2
	bgeu L273
	nop
	mov %l0,%o1
	sll %o1,2,%o0
	sethi %hi(_infiles),%o2
	ld [%o2+%lo(_infiles)],%o1
	add %o0,%o1,%o0
	sethi %hi(_input_filename_length),%o1
	ld [%o0],%o0
	ld [%o1+%lo(_input_filename_length)],%o1
	add %o0,%o1,%l2
	ld [%l1],%o0
	call _strlen,0
	nop
	sub %l2,%o0,%o1
	ld [%l1],%o0
	call _strcmp,0
	nop
	cmp %o0,0
	bne L273
	nop
	sethi %hi(_input_basename),%o0
	sethi %hi(_input_filename),%o1
	ld [%o1+%lo(_input_filename)],%o2
	st %o2,[%o0+%lo(_input_basename)]
	sethi %hi(_input_filename),%o0
	ld [%o0+%lo(_input_filename)],%l2
L274:
	ldub [%l2],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	be L275
	nop
	ldub [%l2],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,47
	bne L277
	nop
	sethi %hi(_input_basename),%o0
	add %l2,1,%o1
	st %o1,[%o0+%lo(_input_basename)]
L277:
L276:
	add %l2,1,%l2
	b L274
	nop
L275:
	sethi %hi(_basename_length),%l3
	sethi %hi(_input_filename_length),%l4
	ld [%l1],%o0
	call _strlen,0
	nop
	ld [%l4+%lo(_input_filename_length)],%o1
	sub %o1,%o0,%o0
	sethi %hi(_input_basename),%o1
	sethi %hi(_input_filename),%o2
	ld [%o1+%lo(_input_basename)],%o1
	ld [%o2+%lo(_input_filename)],%o2
	sub %o1,%o2,%o1
	sub %o0,%o1,%o0
	st %o0,[%l3+%lo(_basename_length)]
	ld [%l1+4],%o0
	call _do_spec,0
	nop
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	cmp %o0,0
	bge L278
	nop
	mov 1,%o0
	st %o0,[%fp-28]
L278:
	b L271
	nop
L273:
L272:
	add %l1,8,%l1
	b L270
	nop
L271:
	ld [%l1+4],%o0
	cmp %o0,0
	bne L279
	nop
	ld [%fp-24],%o0
	cmp %o0,0
	be L279
	nop
	sethi %hi(_input_filename),%o1
	sethi %hi(LC41),%o2
	or %o2,%lo(LC41),%o0
	ld [%o1+%lo(_input_filename)],%o1
	call _error,0
	nop
L279:
	call _clear_failure_queue,0
	nop
L269:
	add %l0,1,%l0
	b L267
	nop
L268:
	ld [%fp-24],%o0
	cmp %o0,0
	bne L280
	nop
	ld [%fp-28],%o0
	cmp %o0,0
	bne L280
	nop
	sethi %hi(_link_spec),%o1
	ld [%o1+%lo(_link_spec)],%o0
	call _do_spec,0
	nop
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	cmp %o0,0
	bge L281
	nop
	mov 1,%o0
	st %o0,[%fp-28]
L281:
L280:
	call _validate_all_switches,0
	nop
	mov 0,%l0
L282:
	sethi %hi(_n_switches),%o0
	ld [%o0+%lo(_n_switches)],%o1
	cmp %l0,%o1
	bge L283
	nop
	mov %l0,%o0
	sll %o0,1,%o1
	add %o1,%l0,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	ld [%o0+8],%o1
	cmp %o1,0
	bne L285
	nop
	mov %l0,%o0
	sll %o0,1,%o1
	add %o1,%l0,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o1
	ld [%o1+%lo(_switches)],%o2
	add %o0,%o2,%o1
	sethi %hi(LC42),%o2
	or %o2,%lo(LC42),%o0
	ld [%o1],%o1
	call _error,0
	nop
L285:
L284:
	add %l0,1,%l0
	b L282
	nop
L283:
	ld [%fp-28],%o0
	xor %o0,0,%o2
	subcc %g0,%o2,%g0
	subx %g0,-1,%o1
	mov %o1,%o0
	call _delete_temp_files,0
	nop
	ld [%fp-28],%o0
	call _exit,0
	nop
L260:
	ret
	restore
	.align 8
LC43:
	.ascii "Virtual memory full.\0"
	.align 4
	.global _xmalloc
	.proc	04
_xmalloc:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	call _malloc,0
	nop
	mov %o0,%l0
	cmp %l0,0
	bne L287
	nop
	sethi %hi(LC43),%o1
	or %o1,%lo(LC43),%o0
	call _fatal,0
	nop
L287:
	mov %l0,%i0
	b L286
	nop
L286:
	ret
	restore
	.align 4
	.global _xrealloc
	.proc	04
_xrealloc:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	call _realloc,0
	nop
	mov %o0,%l0
	cmp %l0,0
	bne L289
	nop
	sethi %hi(LC43),%o1
	or %o1,%lo(LC43),%o0
	call _fatal,0
	nop
L289:
	mov %l0,%i0
	b L288
	nop
L288:
	ret
	restore
	.align 4
	.global _concat
	.proc	0102
_concat:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	ld [%fp+68],%o0
	call _strlen,0
	nop
	st %o0,[%fp-20]
	ld [%fp+72],%o0
	call _strlen,0
	nop
	st %o0,[%fp-24]
	ld [%fp+76],%o0
	call _strlen,0
	nop
	st %o0,[%fp-28]
	ld [%fp-20],%o0
	ld [%fp-24],%o1
	add %o0,%o1,%o0
	ld [%fp-28],%o1
	add %o0,%o1,%o0
	add %o0,1,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-32]
	ld [%fp-32],%o0
	ld [%fp+68],%o1
	call _strcpy,0
	nop
	ld [%fp-32],%o0
	ld [%fp-20],%o2
	add %o0,%o2,%o1
	mov %o1,%o0
	ld [%fp+72],%o1
	call _strcpy,0
	nop
	ld [%fp-32],%o0
	ld [%fp-20],%o1
	add %o0,%o1,%o0
	ld [%fp-24],%o2
	add %o0,%o2,%o1
	mov %o1,%o0
	ld [%fp+76],%o1
	call _strcpy,0
	nop
	ld [%fp-32],%o0
	ld [%fp-20],%o1
	add %o0,%o1,%o0
	ld [%fp-24],%o1
	add %o0,%o1,%o0
	ld [%fp-28],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
	ld [%fp-32],%i0
	b L290
	nop
L290:
	ret
	restore
	.align 4
	.global _save_string
	.proc	0102
_save_string:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+72],%o0
	add %o0,1,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	mov %o0,%l0
	mov %l0,%o0
	ld [%fp+68],%o1
	ld [%fp+72],%o2
	call _memcpy,0
	nop
	ld [%fp+72],%o1
	add %l0,%o1,%o0
	stb %g0,[%o0]
	mov %l0,%i0
	b L291
	nop
L291:
	ret
	restore
	.align 8
LC44:
	.ascii "%s: \0"
	.align 8
LC45:
	.ascii "cannot open %s\0"
	.align 4
	.global _pfatal_with_name
	.proc	04
_pfatal_with_name:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	sethi %hi(_errno),%o0
	sethi %hi(_sys_nerr),%o1
	ld [%o0+%lo(_errno)],%o0
	ld [%o1+%lo(_sys_nerr)],%o1
	cmp %o0,%o1
	bge L293
	nop
	sethi %hi(_errno),%o1
	ld [%o1+%lo(_errno)],%o0
	mov %o0,%o2
	sll %o2,2,%o1
	sethi %hi(_sys_errlist),%o0
	or %o0,%lo(_sys_errlist),%o2
	sethi %hi(LC44),%o3
	or %o3,%lo(LC44),%o0
	ld [%o1+%o2],%o1
	sethi %hi(LC28),%o3
	or %o3,%lo(LC28),%o2
	call _concat,0
	nop
	st %o0,[%fp-20]
	b L294
	nop
L293:
	sethi %hi(LC45),%o1
	or %o1,%lo(LC45),%o0
	st %o0,[%fp-20]
L294:
	ld [%fp-20],%o0
	ld [%fp+68],%o1
	call _fatal,0
	nop
L292:
	ret
	restore
	.align 4
	.global _perror_with_name
	.proc	04
_perror_with_name:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	sethi %hi(_errno),%o0
	sethi %hi(_sys_nerr),%o1
	ld [%o0+%lo(_errno)],%o0
	ld [%o1+%lo(_sys_nerr)],%o1
	cmp %o0,%o1
	bge L296
	nop
	sethi %hi(_errno),%o1
	ld [%o1+%lo(_errno)],%o0
	mov %o0,%o2
	sll %o2,2,%o1
	sethi %hi(_sys_errlist),%o0
	or %o0,%lo(_sys_errlist),%o2
	sethi %hi(LC44),%o3
	or %o3,%lo(LC44),%o0
	ld [%o1+%o2],%o1
	sethi %hi(LC28),%o3
	or %o3,%lo(LC28),%o2
	call _concat,0
	nop
	st %o0,[%fp-20]
	b L297
	nop
L296:
	sethi %hi(LC45),%o1
	or %o1,%lo(LC45),%o0
	st %o0,[%fp-20]
L297:
	ld [%fp-20],%o0
	ld [%fp+68],%o1
	call _error,0
	nop
L295:
	ret
	restore
	.align 8
LC46:
	.ascii "installation problem, cannot exec %s: \0"
	.align 8
LC47:
	.ascii "installation problem, cannot exec %s\0"
	.align 4
	.global _perror_exec
	.proc	04
_perror_exec:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	sethi %hi(_errno),%o0
	sethi %hi(_sys_nerr),%o1
	ld [%o0+%lo(_errno)],%o0
	ld [%o1+%lo(_sys_nerr)],%o1
	cmp %o0,%o1
	bge L299
	nop
	sethi %hi(_errno),%o1
	ld [%o1+%lo(_errno)],%o0
	mov %o0,%o2
	sll %o2,2,%o1
	sethi %hi(_sys_errlist),%o0
	or %o0,%lo(_sys_errlist),%o2
	sethi %hi(LC46),%o3
	or %o3,%lo(LC46),%o0
	ld [%o1+%o2],%o1
	sethi %hi(LC28),%o3
	or %o3,%lo(LC28),%o2
	call _concat,0
	nop
	st %o0,[%fp-20]
	b L300
	nop
L299:
	sethi %hi(LC47),%o1
	or %o1,%lo(LC47),%o0
	st %o0,[%fp-20]
L300:
	ld [%fp-20],%o0
	ld [%fp+68],%o1
	call _error,0
	nop
L298:
	ret
	restore
	.align 4
	.global _fatal
	.proc	04
_fatal:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	call _error,0
	nop
	mov 0,%o0
	call _delete_temp_files,0
	nop
	mov 1,%o0
	call _exit,0
	nop
L301:
	ret
	restore
	.align 4
	.global _error
	.proc	04
_error:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	sethi %hi(_programname),%o2
	sethi %hi(__iob+40),%o1
	or %o1,%lo(__iob+40),%o0
	sethi %hi(LC44),%o3
	or %o3,%lo(LC44),%o1
	ld [%o2+%lo(_programname)],%o2
	call _fprintf,0
	nop
	sethi %hi(__iob+40),%o1
	or %o1,%lo(__iob+40),%o0
	ld [%fp+68],%o1
	ld [%fp+72],%o2
	ld [%fp+76],%o3
	call _fprintf,0
	nop
	sethi %hi(__iob+40),%o1
	or %o1,%lo(__iob+40),%o0
	sethi %hi(LC22),%o2
	or %o2,%lo(LC22),%o1
	call _fprintf,0
	nop
L302:
	ret
	restore
	.align 4
	.global _validate_all_switches
	.proc	020
_validate_all_switches:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	sethi %hi(_compilers),%o1
	or %o1,%lo(_compilers),%o0
	st %o0,[%fp-20]
L304:
	ld [%fp-20],%o0
	ld [%o0+4],%o1
	cmp %o1,0
	be L305
	nop
	ld [%fp-20],%o0
	ld [%o0+4],%l0
L307:
	ldub [%l0],%l1
	mov %l1,%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	add %l0,1,%l0
	cmp %o0,0
	be L308
	nop
	sll %l1,24,%o1
	sra %o1,24,%o0
	cmp %o0,37
	bne L309
	nop
	ldub [%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,123
	bne L309
	nop
	add %l0,1,%o1
	mov %o1,%o0
	call _validate_switches,0
	nop
L309:
	b L307
	nop
L308:
L306:
	ld [%fp-20],%o1
	add %o1,8,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L304
	nop
L305:
	sethi %hi(_link_spec),%o0
	ld [%o0+%lo(_link_spec)],%l0
L310:
	ldub [%l0],%l1
	mov %l1,%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	add %l0,1,%l0
	cmp %o0,0
	be L311
	nop
	sll %l1,24,%o1
	sra %o1,24,%o0
	cmp %o0,37
	bne L312
	nop
	ldub [%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,123
	bne L312
	nop
	add %l0,1,%o1
	mov %o1,%o0
	call _validate_switches,0
	nop
L312:
	b L310
	nop
L311:
L303:
	ret
	restore
	.align 4
	.global _validate_switches
	.proc	020
_validate_switches:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%l0
	ldub [%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,124
	bne L314
	nop
	add %l0,1,%l0
L314:
	ldub [%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,33
	bne L315
	nop
	add %l0,1,%l0
L315:
	st %l0,[%fp-20]
L316:
	ldub [%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,58
	be L317
	nop
	ldub [%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,125
	be L317
	nop
	add %l0,1,%l0
	b L316
	nop
L317:
	ldub [%l0-1],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,42
	bne L318
	nop
	add %l0,-1,%l0
	mov 0,%l1
L319:
	sethi %hi(_n_switches),%o0
	ld [%o0+%lo(_n_switches)],%o1
	cmp %l1,%o1
	bge L320
	nop
	mov %l1,%o0
	sll %o0,1,%o1
	add %o1,%l1,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	ld [%fp-20],%o1
	sub %l0,%o1,%o2
	ld [%o0],%o0
	ld [%fp-20],%o1
	call _strncmp,0
	nop
	cmp %o0,0
	bne L322
	nop
	mov %l1,%o0
	sll %o0,1,%o1
	add %o1,%l1,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	mov 1,%o1
	st %o1,[%o0+8]
L322:
L321:
	add %l1,1,%l1
	b L319
	nop
L320:
	b L323
	nop
L318:
	mov 0,%l1
L324:
	sethi %hi(_n_switches),%o0
	ld [%o0+%lo(_n_switches)],%o1
	cmp %l1,%o1
	bge L325
	nop
	mov %l1,%o0
	sll %o0,1,%o1
	add %o1,%l1,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	ld [%fp-20],%o1
	sub %l0,%o1,%o2
	ld [%o0],%o0
	ld [%fp-20],%o1
	call _strncmp,0
	nop
	cmp %o0,0
	bne L327
	nop
	mov %l1,%o0
	sll %o0,1,%o1
	add %o1,%l1,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	ld [%fp-20],%o2
	sub %l0,%o2,%o1
	ld [%o0],%o2
	add %o1,%o2,%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	bne L327
	nop
	mov %l1,%o0
	sll %o0,1,%o1
	add %o1,%l1,%o1
	sll %o1,2,%o0
	sethi %hi(_switches),%o2
	ld [%o2+%lo(_switches)],%o1
	add %o0,%o1,%o0
	mov 1,%o1
	st %o1,[%o0+8]
L327:
L326:
	add %l1,1,%l1
	b L324
	nop
L325:
L323:
L313:
	ret
	restore
	.global _obstack
	.common _obstack,40,"bss"
	.global _temp_filename
	.common _temp_filename,8,"bss"
	.global _temp_filename_length
	.common _temp_filename_length,8,"bss"
	.global _always_delete_queue
	.common _always_delete_queue,8,"bss"
	.global _failure_delete_queue
	.common _failure_delete_queue,8,"bss"
	.global _argbuf
	.common _argbuf,8,"bss"
	.global _argbuf_length
	.common _argbuf_length,8,"bss"
	.global _argbuf_index
	.common _argbuf_index,8,"bss"
	.global _vflag
	.common _vflag,8,"bss"
	.global _programname
	.common _programname,8,"bss"
	.global _last_pipe_input
	.common _last_pipe_input,8,"bss"
	.global _switches
	.common _switches,8,"bss"
	.global _n_switches
	.common _n_switches,8,"bss"
	.global _infiles
	.common _infiles,8,"bss"
	.global _n_infiles
	.common _n_infiles,8,"bss"
	.global _outfiles
	.common _outfiles,8,"bss"
	.global _input_filename
	.common _input_filename,8,"bss"
	.global _input_file_number
	.common _input_file_number,8,"bss"
	.global _input_filename_length
	.common _input_filename_length,8,"bss"
	.global _basename_length
	.common _basename_length,8,"bss"
	.global _input_basename
	.common _input_basename,8,"bss"
	.global _arg_going
	.common _arg_going,8,"bss"
	.global _delete_this_arg
	.common _delete_this_arg,8,"bss"
	.global _this_is_output_file
	.common _this_is_output_file,8,"bss"
	.global _this_is_library_file
	.common _this_is_library_file,8,"bss"
