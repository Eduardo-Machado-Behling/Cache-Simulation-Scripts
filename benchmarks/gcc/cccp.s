gcc2_compiled.:
___gnu_compiled_c:
.data
	.align 4
_lint:
	.word	0
	.align 4
_put_out_comments:
	.word	0
	.align 4
_no_trigraphs:
	.word	0
	.align 4
_print_deps:
	.word	0
	.align 4
_print_include_names:
	.word	0
	.align 4
_dump_macros:
	.word	0
	.align 4
_debug_output:
	.word	0
	.align 4
_inhibit_warnings:
	.word	0
	.align 4
_warn_import:
	.word	1
	.align 4
_done_initializing:
	.word	0
	.align 4
_indepth:
	.word	-1
	.align 4
_system_include_depth:
	.word	0
	.align 4
_include_defaults_array:
	.word	LC0
	.word	1
	.word	LC1
	.word	0
	.word	LC2
	.word	0
	.word	LC3
	.word	0
	.word	0
	.word	0
.text
	.align 8
LC3:
	.ascii "/usr/include\0"
	.align 8
LC2:
	.ascii "/usr/local/include\0"
	.align 8
LC1:
	.ascii "/usr/local/bin\0"
	.align 8
LC0:
	.ascii "/usr\0"
.data
	.align 4
_include_defaults:
	.word	_include_defaults_array
	.align 4
_include:
	.word	0
	.align 4
_first_bracket_include:
	.word	0
	.align 4
_first_system_include:
	.word	0
	.align 4
_last_include:
	.word	0
	.align 4
_after_include:
	.word	0
	.align 4
_last_after_include:
	.word	0
	.align 4
_dont_repeat_files:
	.word	0
	.align 4
_all_include_files:
	.word	0
	.align 4
_stringlist_tailp:
	.word	_stringlist
	.align 8
_rest_extension:
	.ascii "...\0"
.text
	.align 8
LC4:
	.ascii "-Dunix -D__osf__ -D__alpha -D__alpha__ -D_LONGLONG -DSYSTYPE_BSD  -D_SYSTYPE_BSD\0"
.data
	.align 4
_predefs:
	.word	LC4
	.align 4
_directive_table:
	.word	6
	.word	_do_define
	.word	LC5
	.word	1
	.byte	0
	.byte	1
	.skip 2
	.word	2
	.word	_do_if
	.word	LC6
	.word	7
	.skip 4
	.word	5
	.word	_do_xifdef
	.word	LC7
	.word	5
	.skip 4
	.word	6
	.word	_do_xifdef
	.word	LC8
	.word	6
	.skip 4
	.word	5
	.word	_do_endif
	.word	LC9
	.word	15
	.skip 4
	.word	4
	.word	_do_else
	.word	LC10
	.word	8
	.skip 4
	.word	4
	.word	_do_elif
	.word	LC11
	.word	10
	.skip 4
	.word	4
	.word	_do_line
	.word	LC12
	.word	12
	.skip 4
	.word	7
	.word	_do_include
	.word	LC13
	.word	2
	.byte	1
	.skip 3
	.word	12
	.word	_do_include
	.word	LC14
	.word	3
	.byte	1
	.skip 3
	.word	6
	.word	_do_include
	.word	LC15
	.word	4
	.byte	1
	.skip 3
	.word	5
	.word	_do_undef
	.word	LC16
	.word	11
	.skip 4
	.word	5
	.word	_do_error
	.word	LC17
	.word	13
	.skip 4
	.word	7
	.word	_do_warning
	.word	LC18
	.word	14
	.skip 4
	.word	6
	.word	_do_pragma
	.word	LC19
	.word	9
	.byte	0
	.byte	0
	.byte	1
	.skip 1
	.word	5
	.word	_do_ident
	.word	LC20
	.word	17
	.byte	0
	.byte	0
	.byte	1
	.skip 1
	.word	6
	.word	_do_assert
	.word	LC21
	.word	18
	.skip 4
	.word	8
	.word	_do_unassert
	.word	LC22
	.word	19
	.skip 4
	.word	-1
	.word	0
	.word	LC23
	.word	35
	.skip 4
.text
	.align 8
LC23:
	.ascii "\0"
	.align 8
LC22:
	.ascii "unassert\0"
	.align 8
LC21:
	.ascii "assert\0"
	.align 8
LC20:
	.ascii "ident\0"
	.align 8
LC19:
	.ascii "pragma\0"
	.align 8
LC18:
	.ascii "warning\0"
	.align 8
LC17:
	.ascii "error\0"
	.align 8
LC16:
	.ascii "undef\0"
	.align 8
LC15:
	.ascii "import\0"
	.align 8
LC14:
	.ascii "include_next\0"
	.align 8
LC13:
	.ascii "include\0"
	.align 8
LC12:
	.ascii "line\0"
	.align 8
LC11:
	.ascii "elif\0"
	.align 8
LC10:
	.ascii "else\0"
	.align 8
LC9:
	.ascii "endif\0"
	.align 8
LC8:
	.ascii "ifndef\0"
	.align 8
LC7:
	.ascii "ifdef\0"
	.align 8
LC6:
	.ascii "if\0"
	.align 8
LC5:
	.ascii "define\0"
.data
	.align 4
_errors:
	.word	0
	.align 4
_if_stack:
	.word	0
.text
	.align 8
LC24:
	.ascii "Usage: %s [switches] input output\0"
	.align 8
LC25:
	.ascii "-include\0"
	.align 8
LC26:
	.ascii "Filename missing after -include option\0"
	.align 8
LC27:
	.ascii "-imacros\0"
	.align 8
LC28:
	.ascii "Filename missing after -imacros option\0"
	.align 8
LC29:
	.ascii "-iprefix\0"
	.align 8
LC30:
	.ascii "Filename missing after -iprefix option\0"
	.align 8
LC31:
	.ascii "-idirafter\0"
	.align 8
LC32:
	.ascii "Directory name missing after -idirafter option\0"
	.align 8
LC33:
	.ascii "Output filename specified twice\0"
	.align 8
LC34:
	.ascii "Filename missing after -o option\0"
	.align 8
LC35:
	.ascii "-\0"
	.align 8
LC36:
	.ascii "-pedantic\0"
	.align 8
LC37:
	.ascii "-pedantic-errors\0"
	.align 8
LC38:
	.ascii "-pcp\0"
	.align 8
LC39:
	.ascii "w\0"
	.align 8
LC40:
	.ascii "-traditional\0"
	.align 8
LC41:
	.ascii "-trigraphs\0"
	.align 8
LC42:
	.ascii "-lang-c\0"
	.align 8
LC43:
	.ascii "-lang-c++\0"
	.align 8
LC44:
	.ascii "-lang-objc\0"
	.align 8
LC45:
	.ascii "-lang-objc++\0"
	.align 8
LC46:
	.ascii "-lang-asm\0"
	.align 8
LC47:
	.ascii "-lint\0"
	.align 8
LC48:
	.ascii "-Wtrigraphs\0"
	.align 8
LC49:
	.ascii "-Wno-trigraphs\0"
	.align 8
LC50:
	.ascii "-Wcomment\0"
	.align 8
LC51:
	.ascii "-Wno-comment\0"
	.align 8
LC52:
	.ascii "-Wcomments\0"
	.align 8
LC53:
	.ascii "-Wno-comments\0"
	.align 8
LC54:
	.ascii "-Wtraditional\0"
	.align 8
LC55:
	.ascii "-Wno-traditional\0"
	.align 8
LC56:
	.ascii "-Wimport\0"
	.align 8
LC57:
	.ascii "-Wno-import\0"
	.align 8
LC58:
	.ascii "-Werror\0"
	.align 8
LC59:
	.ascii "-Wno-error\0"
	.align 8
LC60:
	.ascii "-Wall\0"
	.align 8
LC61:
	.ascii "-M\0"
	.align 8
LC62:
	.ascii "-MM\0"
	.align 8
LC63:
	.ascii "-MD\0"
	.align 8
LC64:
	.ascii "-MMD\0"
	.align 8
LC65:
	.ascii "GNU CPP version %s\0"
	.align 8
LC66:
	.ascii "\12\0"
	.align 8
LC67:
	.ascii "Macro name missing after -D option\0"
	.align 8
LC68:
	.ascii "Assertion missing after -A option\0"
	.align 8
LC69:
	.ascii "-A\0"
	.align 8
LC70:
	.ascii "Macro name missing after -U option\0"
	.align 8
LC71:
	.ascii "Directory name missing after -I option\0"
	.align 8
LC72:
	.ascii "-nostdinc\0"
	.align 8
LC73:
	.ascii "-nostdinc++\0"
	.align 8
LC74:
	.ascii "-noprecomp\0"
	.align 8
LC75:
	.ascii "Invalid option `%s'\0"
	.align 8
LC76:
	.ascii "CPATH\0"
	.align 8
LC77:
	.ascii "C_INCLUDE_PATH\0"
	.align 8
LC78:
	.ascii "CPLUS_INCLUDE_PATH\0"
	.align 8
LC79:
	.ascii "OBJC_INCLUDE_PATH\0"
	.align 8
LC80:
	.ascii "OBJCPLUS_INCLUDE_PATH\0"
	.align 8
LC81:
	.ascii ".\0"
	.align 8
LC82:
	.ascii "/include\0"
	.align 8
LC83:
	.ascii "SUNPRO_DEPENDENCIES\0"
	.align 8
LC84:
	.ascii "DEPENDENCIES_OUTPUT\0"
	.align 8
LC85:
	.ascii ":\0"
	.align 8
LC86:
	.ascii "-: \0"
	.align 8
LC87:
	.ascii ".o : \0"
	.align 8
LC88:
	.ascii " \0"
	.align 8
LC89:
	.ascii "stdout\0"
	.align 8
LC90:
	.ascii "file does not end in newline\0"
	.align 8
LC91:
	.ascii "a\0"
	.align 8
LC92:
	.ascii "I/O error on output\0"
	.align 4
	.global _main
	.proc	04
_main:
	!#PROLOGUE# 0
	save %sp,-272,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	call ___main,0
	nop
	ld [%fp+68],%o0
	mov %o0,%o1
	sll %o1,2,%o2
	mov %o2,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-52]
	ld [%fp+68],%o0
	mov %o0,%o1
	sll %o1,2,%o2
	mov %o2,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-56]
	ld [%fp+68],%o0
	mov %o0,%o1
	sll %o1,2,%o2
	mov %o2,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-60]
	ld [%fp+68],%o0
	mov %o0,%o1
	sll %o1,2,%o2
	mov %o2,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-64]
	ld [%fp+68],%o0
	mov %o0,%o1
	sll %o1,2,%o2
	mov %o2,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-68]
	ld [%fp+68],%o0
	mov %o0,%o1
	sll %o1,2,%o2
	mov %o2,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-72]
	st %g0,[%fp-76]
	st %g0,[%fp-80]
	st %g0,[%fp-84]
	st %g0,[%fp-88]
	st %g0,[%fp-92]
	st %g0,[%fp-96]
	st %g0,[%fp-100]
	st %g0,[%fp-104]
	add %fp,-112,%o1
	mov 3,%o0
	call _getrlimit,0
	nop
	ld [%fp-108],%o0
	st %o0,[%fp-112]
	add %fp,-112,%o1
	mov 3,%o0
	call _setrlimit,0
	nop
	sethi %hi(_progname),%o0
	ld [%fp+72],%o1
	ld [%o1],%o2
	st %o2,[%o0+%lo(_progname)]
	st %g0,[%fp-28]
	st %g0,[%fp-32]
	sethi %hi(_dollars_in_ident),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_dollars_in_ident)]
	call _initialize_char_syntax,0
	nop
	sethi %hi(_dollars_in_ident),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_dollars_in_ident)]
	sethi %hi(_no_line_commands),%o0
	st %g0,[%o0+%lo(_no_line_commands)]
	sethi %hi(_no_trigraphs),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_no_trigraphs)]
	sethi %hi(_dump_macros),%o0
	st %g0,[%o0+%lo(_dump_macros)]
	sethi %hi(_no_output),%o0
	st %g0,[%o0+%lo(_no_output)]
	sethi %hi(_cplusplus),%o0
	st %g0,[%o0+%lo(_cplusplus)]
	sethi %hi(_cplusplus_comments),%o0
	st %g0,[%o0+%lo(_cplusplus_comments)]
	ld [%fp+68],%o0
	mov %o0,%o1
	sll %o1,2,%o2
	ld [%fp-52],%o0
	mov %o2,%o1
	call _bzero,0
	nop
	ld [%fp+68],%o0
	mov %o0,%o1
	sll %o1,2,%o2
	ld [%fp-56],%o0
	mov %o2,%o1
	call _bzero,0
	nop
	ld [%fp+68],%o0
	mov %o0,%o1
	sll %o1,2,%o2
	ld [%fp-60],%o0
	mov %o2,%o1
	call _bzero,0
	nop
	ld [%fp+68],%o0
	mov %o0,%o1
	sll %o1,2,%o2
	ld [%fp-64],%o0
	mov %o2,%o1
	call _bzero,0
	nop
	ld [%fp+68],%o0
	mov %o0,%o1
	sll %o1,2,%o2
	ld [%fp-68],%o0
	mov %o2,%o1
	call _bzero,0
	nop
	mov 1,%o0
	st %o0,[%fp-44]
L2:
	ld [%fp-44],%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	bge L3
	nop
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	ld [%o0],%o1
	ldub [%o1],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,45
	be L5
	nop
	ld [%fp-32],%o0
	cmp %o0,0
	be L6
	nop
	ld [%fp+72],%o1
	sethi %hi(LC24),%o2
	or %o2,%lo(LC24),%o0
	ld [%o1],%o1
	call _fatal,0
	nop
	b L7
	nop
L6:
	ld [%fp-28],%o0
	cmp %o0,0
	be L8
	nop
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	ld [%o0],%o1
	st %o1,[%fp-32]
	b L9
	nop
L8:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	ld [%o0],%o1
	st %o1,[%fp-28]
L9:
L7:
	b L10
	nop
L5:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	mov 1,%o1
	ld [%o0],%o2
	add %o1,%o2,%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	mov 119,%o1
	cmp %o1,%o0
	blu L148
	nop
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(L149),%o2
	or %o2,%lo(L149),%o1
	ld [%o0+%o1],%o0
	jmp %o0
	nop
L149:
	.word	L144
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L129
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L53
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L111
	.word	L148
	.word	L126
	.word	L106
	.word	L127
	.word	L148
	.word	L148
	.word	L105
	.word	L130
	.word	L148
	.word	L148
	.word	L148
	.word	L81
	.word	L148
	.word	L148
	.word	L128
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L121
	.word	L148
	.word	L55
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L148
	.word	L92
	.word	L148
	.word	L148
	.word	L102
	.word	L148
	.word	L12
	.word	L148
	.word	L148
	.word	L46
	.word	L148
	.word	L137
	.word	L27
	.word	L31
	.word	L148
	.word	L148
	.word	L148
	.word	L41
	.word	L143
	.word	L104
	.word	L54
L12:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC25),%o2
	or %o2,%lo(LC25),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L13
	nop
	ld [%fp-44],%o1
	add %o1,1,%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	bne L14
	nop
	sethi %hi(LC26),%o1
	or %o1,%lo(LC26),%o0
	call _fatal,0
	nop
	b L15
	nop
L14:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-68],%o1
	add %o0,%o1,%o0
	ld [%fp-44],%o1
	mov %o1,%o2
	sll %o2,2,%o1
	ld [%fp+72],%o2
	add %o1,%o2,%o1
	ld [%o1+4],%o2
	st %o2,[%o0]
	ld [%fp-44],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-44]
L15:
L13:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC27),%o2
	or %o2,%lo(LC27),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L16
	nop
	ld [%fp-44],%o1
	add %o1,1,%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	bne L17
	nop
	sethi %hi(LC28),%o1
	or %o1,%lo(LC28),%o0
	call _fatal,0
	nop
	b L18
	nop
L17:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-52],%o1
	add %o0,%o1,%o0
	ld [%fp-44],%o1
	mov %o1,%o2
	sll %o2,2,%o1
	ld [%fp+72],%o2
	add %o1,%o2,%o1
	ld [%o1+4],%o2
	st %o2,[%o0]
	ld [%fp-44],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-44]
L18:
L16:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC29),%o2
	or %o2,%lo(LC29),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L19
	nop
	ld [%fp-44],%o1
	add %o1,1,%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	bne L20
	nop
	sethi %hi(LC30),%o1
	or %o1,%lo(LC30),%o0
	call _fatal,0
	nop
	b L21
	nop
L20:
	sethi %hi(_include_prefix),%o0
	ld [%fp-44],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%fp-44]
	mov %o2,%o1
	sll %o1,2,%o2
	ld [%fp+72],%o3
	add %o2,%o3,%o1
	ld [%o1],%o2
	st %o2,[%o0+%lo(_include_prefix)]
L21:
L19:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC31),%o2
	or %o2,%lo(LC31),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L22
	nop
	mov 12,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-116]
	ld [%fp-116],%o0
	st %g0,[%o0]
	ld [%fp-116],%o0
	st %g0,[%o0+8]
	ld [%fp-44],%o1
	add %o1,1,%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	bne L23
	nop
	sethi %hi(LC32),%o1
	or %o1,%lo(LC32),%o0
	call _fatal,0
	nop
	b L24
	nop
L23:
	ld [%fp-116],%o0
	ld [%fp-44],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%fp-44]
	mov %o2,%o1
	sll %o1,2,%o2
	ld [%fp+72],%o3
	add %o2,%o3,%o1
	ld [%o1],%o2
	st %o2,[%o0+4]
L24:
	sethi %hi(_after_include),%o0
	ld [%o0+%lo(_after_include)],%o1
	cmp %o1,0
	bne L25
	nop
	sethi %hi(_after_include),%o0
	ld [%fp-116],%o1
	st %o1,[%o0+%lo(_after_include)]
	b L26
	nop
L25:
	sethi %hi(_last_after_include),%o1
	ld [%o1+%lo(_last_after_include)],%o0
	ld [%fp-116],%o1
	st %o1,[%o0]
L26:
	sethi %hi(_last_after_include),%o0
	ld [%fp-116],%o1
	st %o1,[%o0+%lo(_last_after_include)]
L22:
	b L11
	nop
L27:
	ld [%fp-32],%o0
	cmp %o0,0
	be L28
	nop
	sethi %hi(LC33),%o1
	or %o1,%lo(LC33),%o0
	call _fatal,0
	nop
L28:
	ld [%fp-44],%o1
	add %o1,1,%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	bne L29
	nop
	sethi %hi(LC34),%o1
	or %o1,%lo(LC34),%o0
	call _fatal,0
	nop
L29:
	ld [%fp-44],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-44]
	mov %o1,%o0
	sll %o0,2,%o1
	ld [%fp+72],%o2
	add %o1,%o2,%o0
	ld [%o0],%o1
	st %o1,[%fp-32]
	ld [%fp-32],%o0
	sethi %hi(LC35),%o2
	or %o2,%lo(LC35),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L30
	nop
	sethi %hi(LC23),%o1
	or %o1,%lo(LC23),%o0
	st %o0,[%fp-32]
L30:
	b L11
	nop
L31:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC36),%o2
	or %o2,%lo(LC36),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L32
	nop
	sethi %hi(_pedantic),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_pedantic)]
	b L33
	nop
L32:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC37),%o2
	or %o2,%lo(LC37),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L34
	nop
	sethi %hi(_pedantic),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_pedantic)]
	sethi %hi(_pedantic_errors),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_pedantic_errors)]
	b L35
	nop
L34:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC38),%o2
	or %o2,%lo(LC38),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L36
	nop
	ld [%fp-44],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-44]
	mov %o1,%o0
	sll %o0,2,%o1
	ld [%fp+72],%o2
	add %o1,%o2,%o0
	ld [%o0],%o1
	st %o1,[%fp-116]
	sethi %hi(_pcp_outfile),%l0
	ld [%fp-116],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,45
	bne L39
	nop
	mov 1,%o0
	ld [%fp-116],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	bne L39
	nop
	b L37
	nop
L39:
	ld [%fp-116],%o0
	sethi %hi(LC39),%o2
	or %o2,%lo(LC39),%o1
	call _fopen,0
	nop
	mov %o0,%l1
	b L38
	nop
L37:
	sethi %hi(__iob+32),%o0
	mov 18,%o1
	or %o0,%lo(__iob+32),%o2
	add %o1,%o2,%o0
	lduh [%o0],%o1
	sll %o1,16,%o0
	sra %o0,16,%o1
	mov %o1,%o0
	call _dup,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	sethi %hi(LC39),%o2
	or %o2,%lo(LC39),%o1
	call _fdopen,0
	nop
	mov %o0,%l1
L38:
	st %l1,[%l0+%lo(_pcp_outfile)]
	sethi %hi(_pcp_outfile),%o0
	ld [%o0+%lo(_pcp_outfile)],%o1
	cmp %o1,0
	bne L40
	nop
	ld [%fp-116],%o0
	call _pfatal_with_name,0
	nop
L40:
	sethi %hi(_no_precomp),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_no_precomp)]
L36:
L35:
L33:
	b L11
	nop
L41:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC40),%o2
	or %o2,%lo(LC40),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L42
	nop
	sethi %hi(_traditional),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_traditional)]
	sethi %hi(_dollars_in_ident),%o0
	ld [%o0+%lo(_dollars_in_ident)],%o1
	cmp %o1,0
	ble L43
	nop
	sethi %hi(_dollars_in_ident),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_dollars_in_ident)]
L43:
	b L44
	nop
L42:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC41),%o2
	or %o2,%lo(LC41),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L45
	nop
	sethi %hi(_no_trigraphs),%o0
	st %g0,[%o0+%lo(_no_trigraphs)]
L45:
L44:
	b L11
	nop
L46:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC42),%o2
	or %o2,%lo(LC42),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L47
	nop
	sethi %hi(_cplusplus),%o0
	st %g0,[%o0+%lo(_cplusplus)]
	sethi %hi(_cplusplus_comments),%o0
	st %g0,[%o0+%lo(_cplusplus_comments)]
	sethi %hi(_objc),%o0
	st %g0,[%o0+%lo(_objc)]
L47:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC43),%o2
	or %o2,%lo(LC43),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L48
	nop
	sethi %hi(_cplusplus),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_cplusplus)]
	sethi %hi(_cplusplus_comments),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_cplusplus_comments)]
	sethi %hi(_objc),%o0
	st %g0,[%o0+%lo(_objc)]
L48:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC44),%o2
	or %o2,%lo(LC44),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L49
	nop
	sethi %hi(_objc),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_objc)]
	sethi %hi(_cplusplus),%o0
	st %g0,[%o0+%lo(_cplusplus)]
	sethi %hi(_cplusplus_comments),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_cplusplus_comments)]
L49:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC45),%o2
	or %o2,%lo(LC45),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L50
	nop
	sethi %hi(_objc),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_objc)]
	sethi %hi(_cplusplus),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_cplusplus)]
	sethi %hi(_cplusplus_comments),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_cplusplus_comments)]
L50:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC46),%o2
	or %o2,%lo(LC46),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L51
	nop
	sethi %hi(_lang_asm),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_lang_asm)]
L51:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC47),%o2
	or %o2,%lo(LC47),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L52
	nop
	sethi %hi(_lint),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_lint)]
L52:
	b L11
	nop
L53:
	sethi %hi(_cplusplus),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_cplusplus)]
	sethi %hi(_cplusplus_comments),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_cplusplus_comments)]
	b L11
	nop
L54:
	sethi %hi(_inhibit_warnings),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_inhibit_warnings)]
	b L11
	nop
L55:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC48),%o2
	or %o2,%lo(LC48),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L56
	nop
	sethi %hi(_warn_trigraphs),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_warn_trigraphs)]
	b L57
	nop
L56:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC49),%o2
	or %o2,%lo(LC49),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L58
	nop
	sethi %hi(_warn_trigraphs),%o0
	st %g0,[%o0+%lo(_warn_trigraphs)]
	b L59
	nop
L58:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC50),%o2
	or %o2,%lo(LC50),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L60
	nop
	sethi %hi(_warn_comments),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_warn_comments)]
	b L61
	nop
L60:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC51),%o2
	or %o2,%lo(LC51),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L62
	nop
	sethi %hi(_warn_comments),%o0
	st %g0,[%o0+%lo(_warn_comments)]
	b L63
	nop
L62:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC52),%o2
	or %o2,%lo(LC52),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L64
	nop
	sethi %hi(_warn_comments),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_warn_comments)]
	b L65
	nop
L64:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC53),%o2
	or %o2,%lo(LC53),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L66
	nop
	sethi %hi(_warn_comments),%o0
	st %g0,[%o0+%lo(_warn_comments)]
	b L67
	nop
L66:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC54),%o2
	or %o2,%lo(LC54),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L68
	nop
	sethi %hi(_warn_stringify),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_warn_stringify)]
	b L69
	nop
L68:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC55),%o2
	or %o2,%lo(LC55),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L70
	nop
	sethi %hi(_warn_stringify),%o0
	st %g0,[%o0+%lo(_warn_stringify)]
	b L71
	nop
L70:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC56),%o2
	or %o2,%lo(LC56),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L72
	nop
	sethi %hi(_warn_import),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_warn_import)]
	b L73
	nop
L72:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC57),%o2
	or %o2,%lo(LC57),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L74
	nop
	sethi %hi(_warn_import),%o0
	st %g0,[%o0+%lo(_warn_import)]
	b L75
	nop
L74:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC58),%o2
	or %o2,%lo(LC58),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L76
	nop
	sethi %hi(_warnings_are_errors),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_warnings_are_errors)]
	b L77
	nop
L76:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC59),%o2
	or %o2,%lo(LC59),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L78
	nop
	sethi %hi(_warnings_are_errors),%o0
	st %g0,[%o0+%lo(_warnings_are_errors)]
	b L79
	nop
L78:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC60),%o2
	or %o2,%lo(LC60),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L80
	nop
	sethi %hi(_warn_trigraphs),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_warn_trigraphs)]
	sethi %hi(_warn_comments),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_warn_comments)]
L80:
L79:
L77:
L75:
L73:
L71:
L69:
L67:
L65:
L63:
L61:
L59:
L57:
	b L11
	nop
L81:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC61),%o2
	or %o2,%lo(LC61),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L82
	nop
	sethi %hi(_print_deps),%o0
	mov 2,%o1
	st %o1,[%o0+%lo(_print_deps)]
	b L83
	nop
L82:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC62),%o2
	or %o2,%lo(LC62),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L84
	nop
	sethi %hi(_print_deps),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_print_deps)]
	b L85
	nop
L84:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC63),%o2
	or %o2,%lo(LC63),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L86
	nop
	sethi %hi(_print_deps),%o0
	mov 2,%o1
	st %o1,[%o0+%lo(_print_deps)]
	b L87
	nop
L86:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC64),%o2
	or %o2,%lo(LC64),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L88
	nop
	sethi %hi(_print_deps),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_print_deps)]
L88:
L87:
L85:
L83:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC63),%o2
	or %o2,%lo(LC63),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	be L90
	nop
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC64),%o2
	or %o2,%lo(LC64),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L89
	nop
	b L90
	nop
L90:
	ld [%fp-44],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-44]
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	ld [%o0],%o1
	st %o1,[%fp-96]
	b L91
	nop
L89:
	sethi %hi(__iob+32),%o1
	or %o1,%lo(__iob+32),%o0
	st %o0,[%fp-100]
	mov 1,%o0
	st %o0,[%fp-92]
L91:
	b L11
	nop
L92:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	ld [%o0],%o1
	add %o1,2,%o0
	st %o0,[%fp-120]
L93:
	ld [%fp-120],%o0
	add %o0,1,%o1
	st %o1,[%fp-120]
	ldub [%o0],%o0
	stb %o0,[%fp-121]
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	be L94
	nop
	ldub [%fp-121],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,77
	be L96
	nop
	cmp %o0,77
	bg L101
	nop
	cmp %o0,68
	be L98
	nop
	b L100
	nop
L101:
	cmp %o0,78
	be L97
	nop
	b L100
	nop
L96:
	sethi %hi(_dump_macros),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_dump_macros)]
	sethi %hi(_no_output),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_no_output)]
	b L95
	nop
L97:
	sethi %hi(_dump_macros),%o0
	mov 2,%o1
	st %o1,[%o0+%lo(_dump_macros)]
	b L95
	nop
L98:
	sethi %hi(_dump_macros),%o0
	mov 3,%o1
	st %o1,[%o0+%lo(_dump_macros)]
	b L95
	nop
L100:
L95:
	b L93
	nop
L94:
	b L11
	nop
L102:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	mov 2,%o1
	ld [%o0],%o2
	add %o1,%o2,%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,51
	bne L103
	nop
	sethi %hi(_debug_output),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_debug_output)]
L103:
	b L11
	nop
L104:
	sethi %hi(_version_string),%o2
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC65),%o3
	or %o3,%lo(LC65),%o1
	ld [%o2+%lo(_version_string)],%o2
	call _fprintf,0
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC66),%o2
	or %o2,%lo(LC66),%o1
	call _fprintf,0
	nop
	b L11
	nop
L105:
	sethi %hi(_print_include_names),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_print_include_names)]
	b L11
	nop
L106:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	mov 2,%o1
	ld [%o0],%o2
	add %o1,%o2,%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L107
	nop
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	ld [%o0],%o1
	add %o1,2,%o0
	st %o0,[%fp-120]
	b L108
	nop
L107:
	ld [%fp-44],%o1
	add %o1,1,%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	bne L109
	nop
	sethi %hi(LC67),%o1
	or %o1,%lo(LC67),%o0
	call _fatal,0
	nop
	b L110
	nop
L109:
	ld [%fp-44],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-44]
	mov %o1,%o0
	sll %o0,2,%o1
	ld [%fp+72],%o2
	add %o1,%o2,%o0
	ld [%o0],%o1
	st %o1,[%fp-120]
L110:
L108:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-56],%o1
	add %o0,%o1,%o0
	ld [%fp-120],%o1
	st %o1,[%o0]
	b L11
	nop
L111:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	mov 2,%o1
	ld [%o0],%o2
	add %o1,%o2,%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L112
	nop
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	ld [%o0],%o1
	add %o1,2,%o0
	st %o0,[%fp-128]
	b L113
	nop
L112:
	ld [%fp-44],%o1
	add %o1,1,%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	bne L114
	nop
	sethi %hi(LC68),%o1
	or %o1,%lo(LC68),%o0
	call _fatal,0
	nop
	b L115
	nop
L114:
	ld [%fp-44],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-44]
	mov %o1,%o0
	sll %o0,2,%o1
	ld [%fp+72],%o2
	add %o1,%o2,%o0
	ld [%o0],%o1
	st %o1,[%fp-128]
L115:
L113:
	ld [%fp-128],%o0
	sethi %hi(LC35),%o2
	or %o2,%lo(LC35),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L116
	nop
	mov 1,%o0
	st %o0,[%fp-76]
	st %g0,[%fp-132]
L117:
	ld [%fp-132],%o0
	ld [%fp-44],%o1
	cmp %o0,%o1
	bge L118
	nop
	ld [%fp-132],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-56],%o1
	add %o0,%o1,%o0
	ld [%fp-132],%o1
	mov %o1,%o2
	sll %o2,2,%o1
	ld [%fp-64],%o2
	add %o1,%o2,%o1
	st %g0,[%o1]
	st %g0,[%o0]
L119:
	ld [%fp-132],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-132]
	b L117
	nop
L118:
	b L120
	nop
L116:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-64],%o1
	add %o0,%o1,%o0
	ld [%fp-128],%o1
	st %o1,[%o0]
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-72],%o1
	add %o0,%o1,%o0
	sethi %hi(LC69),%o2
	or %o2,%lo(LC69),%o1
	st %o1,[%o0]
L120:
	b L11
	nop
L121:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	mov 2,%o1
	ld [%o0],%o2
	add %o1,%o2,%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L122
	nop
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-60],%o1
	add %o0,%o1,%o0
	ld [%fp-44],%o1
	mov %o1,%o2
	sll %o2,2,%o1
	ld [%fp+72],%o2
	add %o1,%o2,%o1
	ld [%o1],%o2
	add %o2,2,%o1
	st %o1,[%o0]
	b L123
	nop
L122:
	ld [%fp-44],%o1
	add %o1,1,%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	bne L124
	nop
	sethi %hi(LC70),%o1
	or %o1,%lo(LC70),%o0
	call _fatal,0
	nop
	b L125
	nop
L124:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-60],%o1
	add %o0,%o1,%o0
	ld [%fp-44],%o1
	mov %o1,%o2
	sll %o2,2,%o1
	ld [%fp+72],%o2
	add %o1,%o2,%o1
	ld [%o1+4],%o2
	st %o2,[%o0]
	ld [%fp-44],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-44]
L125:
L123:
	b L11
	nop
L126:
	sethi %hi(_put_out_comments),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_put_out_comments)]
	b L11
	nop
L127:
	b L11
	nop
L128:
	sethi %hi(_no_line_commands),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_no_line_commands)]
	b L11
	nop
L129:
	sethi %hi(_dollars_in_ident),%o0
	st %g0,[%o0+%lo(_dollars_in_ident)]
	b L11
	nop
L130:
	sethi %hi(_ignore_srcdir),%o0
	ld [%o0+%lo(_ignore_srcdir)],%o1
	cmp %o1,0
	bne L131
	nop
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	ld [%o0],%o2
	add %o2,2,%o1
	mov %o1,%o0
	sethi %hi(LC35),%o2
	or %o2,%lo(LC35),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L131
	nop
	sethi %hi(_ignore_srcdir),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_ignore_srcdir)]
	sethi %hi(_first_bracket_include),%o0
	st %g0,[%o0+%lo(_first_bracket_include)]
	b L132
	nop
L131:
	mov 12,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-132]
	ld [%fp-132],%o0
	st %g0,[%o0]
	ld [%fp-132],%o0
	st %g0,[%o0+8]
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	mov 2,%o1
	ld [%o0],%o2
	add %o1,%o2,%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L133
	nop
	ld [%fp-132],%o0
	ld [%fp-44],%o1
	mov %o1,%o2
	sll %o2,2,%o1
	ld [%fp+72],%o2
	add %o1,%o2,%o1
	ld [%o1],%o2
	add %o2,2,%o1
	st %o1,[%o0+4]
	b L134
	nop
L133:
	ld [%fp-44],%o1
	add %o1,1,%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	bne L135
	nop
	sethi %hi(LC71),%o1
	or %o1,%lo(LC71),%o0
	call _fatal,0
	nop
	b L136
	nop
L135:
	ld [%fp-132],%o0
	ld [%fp-44],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%fp-44]
	mov %o2,%o1
	sll %o1,2,%o2
	ld [%fp+72],%o3
	add %o2,%o3,%o1
	ld [%o1],%o2
	st %o2,[%o0+4]
L136:
L134:
	ld [%fp-132],%o0
	ld [%fp-132],%o1
	call _append_include_chain,0
	nop
L132:
	b L11
	nop
L137:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC72),%o2
	or %o2,%lo(LC72),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L138
	nop
	mov 1,%o0
	st %o0,[%fp-80]
	b L139
	nop
L138:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC73),%o2
	or %o2,%lo(LC73),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L140
	nop
	mov 1,%o0
	st %o0,[%fp-84]
	b L141
	nop
L140:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(LC74),%o2
	or %o2,%lo(LC74),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L142
	nop
	sethi %hi(_no_precomp),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_no_precomp)]
L142:
L141:
L139:
	b L11
	nop
L143:
	mov 1,%o0
	st %o0,[%fp-76]
	b L11
	nop
L144:
	ld [%fp-28],%o0
	cmp %o0,0
	bne L145
	nop
	sethi %hi(LC23),%o1
	or %o1,%lo(LC23),%o0
	st %o0,[%fp-28]
	b L11
	nop
	b L146
	nop
L145:
	ld [%fp-32],%o0
	cmp %o0,0
	bne L147
	nop
	sethi %hi(LC23),%o1
	or %o1,%lo(LC23),%o0
	st %o0,[%fp-32]
	b L11
	nop
L147:
L146:
L148:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	sethi %hi(LC75),%o2
	or %o2,%lo(LC75),%o0
	ld [%o1],%o1
	call _fatal,0
	nop
L11:
L10:
L4:
	ld [%fp-44],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-44]
	b L2
	nop
L3:
	sethi %hi(LC76),%o1
	or %o1,%lo(LC76),%o0
	call _getenv,0
	nop
	st %o0,[%fp-36]
	ld [%fp-36],%o0
	cmp %o0,0
	be L150
	nop
	ld [%fp-80],%o0
	cmp %o0,0
	bne L150
	nop
	ld [%fp-36],%o0
	call _path_include,0
	nop
L150:
	call _initialize_char_syntax,0
	nop
	mov 10,%o0
	call _xmalloc,0
	nop
	sethi %hi(_outbuf),%o1
	mov 20,%o2
	or %o1,%lo(_outbuf),%o3
	add %o2,%o3,%o1
	st %o0,[%o1]
	sethi %hi(_outbuf),%o0
	mov 24,%o1
	or %o0,%lo(_outbuf),%o2
	add %o1,%o2,%o0
	sethi %hi(_outbuf+20),%o1
	ld [%o1+%lo(_outbuf+20)],%o2
	st %o2,[%o0]
	sethi %hi(_outbuf),%o0
	mov 16,%o1
	or %o0,%lo(_outbuf),%o2
	add %o1,%o2,%o0
	mov 10,%o1
	st %o1,[%o0]
	sethi %hi(_indepth),%o0
	sethi %hi(_indepth),%o1
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o3
	add %o3,1,%o2
	mov %o2,%o0
	st %o0,[%o1+%lo(_indepth)]
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-48]
	ld [%fp-28],%o0
	cmp %o0,0
	bne L151
	nop
	sethi %hi(LC23),%o1
	or %o1,%lo(LC23),%o0
	st %o0,[%fp-28]
L151:
	ld [%fp-48],%o0
	ld [%fp-48],%o1
	ld [%fp-28],%o2
	st %o2,[%o1]
	st %o2,[%o0+4]
	ld [%fp-48],%o0
	st %g0,[%o0+12]
	ld [%fp-48],%o0
	sethi %hi(_outbuf),%o2
	or %o2,%lo(_outbuf),%o1
	call _initialize_builtins,0
	nop
	ld [%fp-76],%o0
	cmp %o0,0
	bne L152
	nop
	sethi %hi(_predefs),%o1
	ld [%o1+%lo(_predefs)],%o0
	call _strlen,0
	nop
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
	st %o1,[%fp-132]
	sethi %hi(_predefs),%o1
	ld [%fp-132],%o0
	ld [%o1+%lo(_predefs)],%o1
	call _strcpy,0
	nop
L153:
	ld [%fp-132],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L154
	nop
L155:
	ld [%fp-132],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,32
	be L157
	nop
	ld [%fp-132],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,9
	be L157
	nop
	b L156
	nop
L157:
	ld [%fp-132],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-132]
	b L155
	nop
L156:
	ld [%fp-132],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,45
	bne L158
	nop
	mov 1,%o0
	ld [%fp-132],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,68
	bne L158
	nop
	ld [%fp-132],%o0
	add %o0,2,%o1
	st %o1,[%fp-128]
L159:
	ld [%fp-132],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L160
	nop
	ld [%fp-132],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,32
	be L160
	nop
	ld [%fp-132],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,9
	be L160
	nop
	ld [%fp-132],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-132]
	b L159
	nop
L160:
	ld [%fp-132],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L161
	nop
	ld [%fp-132],%o0
	add %o0,1,%o1
	st %o1,[%fp-132]
	stb %g0,[%o0]
L161:
	sethi %hi(_debug_output),%o0
	ld [%o0+%lo(_debug_output)],%o1
	cmp %o1,0
	be L162
	nop
	ld [%fp-48],%o0
	sethi %hi(_outbuf),%o2
	or %o2,%lo(_outbuf),%o1
	mov 0,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
L162:
	ld [%fp-128],%o0
	sethi %hi(_outbuf),%o2
	or %o2,%lo(_outbuf),%o1
	call _make_definition,0
	nop
L163:
	ld [%fp-132],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,32
	be L165
	nop
	ld [%fp-132],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,9
	be L165
	nop
	b L164
	nop
L165:
	ld [%fp-132],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-132]
	b L163
	nop
L164:
	b L166
	nop
L158:
	ld [%fp-132],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,45
	bne L167
	nop
	mov 1,%o0
	ld [%fp-132],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,65
	bne L167
	nop
	ld [%fp-132],%o0
	add %o0,2,%o1
	st %o1,[%fp-120]
	ld [%fp-120],%o0
	st %o0,[%fp-116]
L168:
	ld [%fp-116],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L169
	nop
	ld [%fp-116],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,32
	be L169
	nop
	ld [%fp-116],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,9
	be L169
	nop
	ld [%fp-116],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,40
	be L169
	nop
	ld [%fp-116],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-116]
	b L168
	nop
L169:
	ld [%fp-116],%o0
	st %o0,[%fp-136]
L170:
	ld [%fp-136],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L171
	nop
	ld [%fp-136],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,32
	be L172
	nop
	ld [%fp-136],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,9
	be L172
	nop
	b L171
	nop
L172:
	ld [%fp-136],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-136]
	b L170
	nop
L171:
	ld [%fp-136],%o0
	add %o0,1,%o1
	st %o1,[%fp-136]
	ldub [%o0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,40
	be L173
	nop
	call _abort,0
	nop
L173:
	nop
L174:
	ld [%fp-136],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L175
	nop
	ld [%fp-136],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,32
	be L176
	nop
	ld [%fp-136],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,9
	be L176
	nop
	b L175
	nop
L176:
	ld [%fp-136],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-136]
	b L174
	nop
L175:
	ld [%fp-136],%o0
	st %o0,[%fp-140]
L177:
	ld [%fp-140],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L178
	nop
	ld [%fp-140],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,32
	be L178
	nop
	ld [%fp-140],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,9
	be L178
	nop
	ld [%fp-140],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,41
	be L178
	nop
	ld [%fp-140],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-140]
	b L177
	nop
L178:
	ld [%fp-140],%o0
	st %o0,[%fp-144]
L179:
	ld [%fp-144],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L180
	nop
	ld [%fp-144],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,32
	be L181
	nop
	ld [%fp-144],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,9
	be L181
	nop
	b L180
	nop
L181:
	ld [%fp-144],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-144]
	b L179
	nop
L180:
	ld [%fp-144],%o0
	add %o0,1,%o1
	st %o1,[%fp-144]
	ldub [%o0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,41
	be L182
	nop
	call _abort,0
	nop
L182:
	ld [%fp-144],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L183
	nop
	ld [%fp-144],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,32
	be L183
	nop
	ld [%fp-144],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,9
	be L183
	nop
	call _abort,0
	nop
L183:
	ld [%fp-144],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	st %o0,[%fp-148]
	ld [%fp-144],%o0
	stb %g0,[%o0]
	sethi %hi(LC69),%o1
	or %o1,%lo(LC69),%o0
	ld [%fp-120],%o1
	call _make_assertion,0
	nop
	ld [%fp-144],%o0
	ldub [%fp-145],%o1
	stb %o1,[%o0]
	ld [%fp-144],%o0
	st %o0,[%fp-132]
L184:
	ld [%fp-132],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,32
	be L186
	nop
	ld [%fp-132],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,9
	be L186
	nop
	b L185
	nop
L186:
	ld [%fp-132],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-132]
	b L184
	nop
L185:
	b L187
	nop
L167:
	call _abort,0
	nop
L187:
L166:
	b L153
	nop
L154:
L152:
	nop
	mov 1,%o0
	st %o0,[%fp-44]
L188:
	ld [%fp-44],%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	bge L189
	nop
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-60],%o1
	add %o0,%o1,%o0
	ld [%o0],%o1
	cmp %o1,0
	be L191
	nop
	sethi %hi(_debug_output),%o0
	ld [%o0+%lo(_debug_output)],%o1
	cmp %o1,0
	be L192
	nop
	ld [%fp-48],%o0
	sethi %hi(_outbuf),%o2
	or %o2,%lo(_outbuf),%o1
	mov 0,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
L192:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-60],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(_outbuf),%o2
	or %o2,%lo(_outbuf),%o1
	call _make_undef,0
	nop
L191:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-56],%o1
	add %o0,%o1,%o0
	ld [%o0],%o1
	cmp %o1,0
	be L193
	nop
	sethi %hi(_debug_output),%o0
	ld [%o0+%lo(_debug_output)],%o1
	cmp %o1,0
	be L194
	nop
	ld [%fp-48],%o0
	sethi %hi(_outbuf),%o2
	or %o2,%lo(_outbuf),%o1
	mov 0,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
L194:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-56],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	sethi %hi(_outbuf),%o2
	or %o2,%lo(_outbuf),%o1
	call _make_definition,0
	nop
L193:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-64],%o1
	add %o0,%o1,%o0
	ld [%o0],%o1
	cmp %o1,0
	be L195
	nop
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-72],%o1
	add %o0,%o1,%o0
	ld [%fp-44],%o1
	mov %o1,%o2
	sll %o2,2,%o1
	ld [%fp-64],%o2
	add %o1,%o2,%o1
	ld [%o0],%o0
	ld [%o1],%o1
	call _make_assertion,0
	nop
L195:
L190:
	ld [%fp-44],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-44]
	b L188
	nop
L189:
	sethi %hi(_done_initializing),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_done_initializing)]
	st %g0,[%fp-148]
	sethi %hi(_objc),%o0
	ld [%o0+%lo(_objc)],%o1
	sll %o1,1,%o0
	sethi %hi(_cplusplus),%o1
	ld [%o1+%lo(_cplusplus)],%o2
	add %o0,%o2,%o0
	cmp %o0,1
	be L198
	nop
	cmp %o0,1
	bg L203
	nop
	cmp %o0,0
	be L197
	nop
	b L202
	nop
L203:
	cmp %o0,2
	be L199
	nop
	cmp %o0,3
	be L200
	nop
	b L202
	nop
L197:
	sethi %hi(LC77),%o1
	or %o1,%lo(LC77),%o0
	call _getenv,0
	nop
	st %o0,[%fp-148]
	b L196
	nop
L198:
	sethi %hi(LC78),%o1
	or %o1,%lo(LC78),%o0
	call _getenv,0
	nop
	st %o0,[%fp-148]
	b L196
	nop
L199:
	sethi %hi(LC79),%o1
	or %o1,%lo(LC79),%o0
	call _getenv,0
	nop
	st %o0,[%fp-148]
	b L196
	nop
L200:
	sethi %hi(LC80),%o1
	or %o1,%lo(LC80),%o0
	call _getenv,0
	nop
	st %o0,[%fp-148]
	b L196
	nop
L202:
L196:
	ld [%fp-148],%o0
	cmp %o0,0
	be L204
	nop
	ld [%fp-148],%o0
	call _strlen,0
	nop
	add %o0,2,%o1
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
	st %o1,[%fp-144]
	mov 1,%o0
	st %o0,[%fp-140]
	ld [%fp-148],%o0
	st %o0,[%fp-136]
L205:
	ld [%fp-136],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L206
	nop
	ld [%fp-136],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,58
	bne L208
	nop
	ld [%fp-140],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-140]
L208:
L207:
	ld [%fp-136],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-136]
	b L205
	nop
L206:
	sethi %hi(_include_defaults),%l0
	ld [%fp-140],%o0
	mov %o0,%o1
	sll %o1,3,%o0
	add %o0,40,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	st %o0,[%l0+%lo(_include_defaults)]
	ld [%fp-148],%o0
	st %o0,[%fp-128]
	st %o0,[%fp-136]
	st %g0,[%fp-140]
L209:
	ld [%fp-128],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,58
	be L212
	nop
	ld [%fp-128],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	bne L211
	nop
	b L212
	nop
L212:
	ld [%fp-128],%o0
	ld [%fp-136],%o1
	sub %o0,%o1,%o2
	ld [%fp-144],%o0
	ld [%fp-136],%o1
	call _strncpy,0
	nop
	ld [%fp-128],%o0
	ld [%fp-136],%o1
	cmp %o0,%o1
	bne L213
	nop
	ld [%fp-144],%o0
	sethi %hi(LC81),%o2
	or %o2,%lo(LC81),%o1
	call _strcpy,0
	nop
	b L214
	nop
L213:
	ld [%fp-128],%o0
	ld [%fp-136],%o1
	sub %o0,%o1,%o0
	ld [%fp-144],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
L214:
	ld [%fp-144],%o0
	call _savestring,0
	nop
	ld [%fp-140],%o1
	mov %o1,%o2
	sll %o2,3,%o1
	sethi %hi(_include_defaults),%o3
	ld [%o3+%lo(_include_defaults)],%o2
	add %o1,%o2,%o1
	st %o0,[%o1]
	ld [%fp-140],%o0
	mov %o0,%o1
	sll %o1,3,%o0
	sethi %hi(_include_defaults),%o2
	ld [%o2+%lo(_include_defaults)],%o1
	add %o0,%o1,%o0
	sethi %hi(_cplusplus),%o1
	ld [%o1+%lo(_cplusplus)],%o2
	st %o2,[%o0+4]
	ld [%fp-140],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-140]
	ld [%fp-128],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	bne L215
	nop
	b L210
	nop
L215:
	ld [%fp-128],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-136]
	st %o1,[%fp-128]
	b L216
	nop
L211:
	ld [%fp-128],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-128]
L216:
	b L209
	nop
L210:
	sethi %hi(_include_defaults),%o0
	ld [%fp-140],%o1
	mov %o1,%o2
	sll %o2,3,%o1
	ld [%o0+%lo(_include_defaults)],%o0
	add %o1,%o0,%o1
	sethi %hi(_include_defaults_array),%o2
	or %o2,%lo(_include_defaults_array),%o0
	mov 40,%o2
	call _bcopy,0
	nop
L204:
	sethi %hi(_first_system_include),%o0
	st %g0,[%o0+%lo(_first_system_include)]
	ld [%fp-80],%o0
	cmp %o0,0
	bne L217
	nop
	sethi %hi(_include_defaults),%o0
	ld [%o0+%lo(_include_defaults)],%o1
	st %o1,[%fp-140]
	sethi %hi(_include_prefix),%o0
	ld [%o0+%lo(_include_prefix)],%o1
	st %o1,[%fp-136]
	sethi %hi(LC1),%o1
	or %o1,%lo(LC1),%o0
	call _savestring,0
	nop
	st %o0,[%fp-128]
	st %g0,[%fp-120]
	ld [%fp-128],%o0
	call _strlen,0
	nop
	ld [%fp-128],%o1
	add %o0,%o1,%o0
	add %o0,-8,%o1
	mov %o1,%o0
	sethi %hi(LC82),%o2
	or %o2,%lo(LC82),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L218
	nop
	ld [%fp-128],%o0
	call _strlen,0
	nop
	add %o0,-7,%o1
	st %o1,[%fp-120]
	ld [%fp-128],%o0
	ld [%fp-120],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
L218:
	ld [%fp-136],%o0
	cmp %o0,0
	be L219
	nop
	ld [%fp-120],%o0
	cmp %o0,0
	be L219
	nop
	sethi %hi(_include_defaults),%o0
	ld [%o0+%lo(_include_defaults)],%o1
	st %o1,[%fp-140]
L220:
	ld [%fp-140],%o0
	ld [%o0],%o1
	cmp %o1,0
	be L221
	nop
	ld [%fp-140],%o0
	ld [%o0+4],%o1
	cmp %o1,0
	be L224
	nop
	sethi %hi(_cplusplus),%o0
	ld [%o0+%lo(_cplusplus)],%o1
	cmp %o1,0
	be L223
	nop
	ld [%fp-84],%o0
	cmp %o0,0
	bne L223
	nop
	b L224
	nop
L224:
	ld [%fp-140],%o1
	ld [%o1],%o0
	ld [%fp-128],%o1
	ld [%fp-120],%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L225
	nop
	mov 12,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-116]
	ld [%fp-136],%o0
	call _strlen,0
	nop
	mov %o0,%l0
	ld [%fp-140],%o1
	ld [%o1],%o0
	call _strlen,0
	nop
	add %l0,%o0,%o1
	ld [%fp-120],%o0
	sub %o1,%o0,%o1
	st %o1,[%fp-152]
	ld [%fp-152],%o0
	add %o0,1,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-156]
	ld [%fp-156],%o0
	ld [%fp-136],%o1
	call _strcpy,0
	nop
	ld [%fp-140],%o0
	ld [%o0],%o1
	ld [%fp-120],%o0
	add %o1,%o0,%o1
	ld [%fp-156],%o0
	call _strcat,0
	nop
	ld [%fp-116],%o0
	ld [%fp-156],%o1
	st %o1,[%o0+4]
	ld [%fp-116],%o0
	st %g0,[%o0+8]
	ld [%fp-116],%o0
	ld [%fp-116],%o1
	call _append_include_chain,0
	nop
	sethi %hi(_first_system_include),%o0
	ld [%o0+%lo(_first_system_include)],%o1
	cmp %o1,0
	bne L226
	nop
	sethi %hi(_first_system_include),%o0
	ld [%fp-116],%o1
	st %o1,[%o0+%lo(_first_system_include)]
L226:
L225:
L223:
L222:
	ld [%fp-140],%o1
	add %o1,8,%o0
	mov %o0,%o1
	st %o1,[%fp-140]
	b L220
	nop
L221:
L219:
	nop
	sethi %hi(_include_defaults),%o0
	ld [%o0+%lo(_include_defaults)],%o1
	st %o1,[%fp-140]
L227:
	ld [%fp-140],%o0
	ld [%o0],%o1
	cmp %o1,0
	be L228
	nop
	ld [%fp-140],%o0
	ld [%o0+4],%o1
	cmp %o1,0
	be L231
	nop
	sethi %hi(_cplusplus),%o0
	ld [%o0+%lo(_cplusplus)],%o1
	cmp %o1,0
	be L230
	nop
	ld [%fp-84],%o0
	cmp %o0,0
	bne L230
	nop
	b L231
	nop
L231:
	mov 12,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-152]
	ld [%fp-152],%o0
	st %g0,[%o0+8]
	ld [%fp-152],%o0
	ld [%fp-140],%o1
	ld [%o1],%o2
	st %o2,[%o0+4]
	ld [%fp-152],%o0
	ld [%fp-152],%o1
	call _append_include_chain,0
	nop
	sethi %hi(_first_system_include),%o0
	ld [%o0+%lo(_first_system_include)],%o1
	cmp %o1,0
	bne L232
	nop
	sethi %hi(_first_system_include),%o0
	ld [%fp-152],%o1
	st %o1,[%o0+%lo(_first_system_include)]
L232:
L230:
L229:
	ld [%fp-140],%o1
	add %o1,8,%o0
	mov %o0,%o1
	st %o1,[%fp-140]
	b L227
	nop
L228:
L217:
	sethi %hi(_after_include),%o0
	sethi %hi(_last_after_include),%o1
	ld [%o0+%lo(_after_include)],%o0
	ld [%o1+%lo(_last_after_include)],%o1
	call _append_include_chain,0
	nop
	sethi %hi(_first_system_include),%o0
	ld [%o0+%lo(_first_system_include)],%o1
	cmp %o1,0
	bne L233
	nop
	sethi %hi(_first_system_include),%o0
	sethi %hi(_after_include),%o1
	ld [%o1+%lo(_after_include)],%o2
	st %o2,[%o0+%lo(_first_system_include)]
L233:
	sethi %hi(_no_output),%o1
	sethi %hi(_no_output),%o0
	sethi %hi(_no_output),%o1
	ld [%o1+%lo(_no_output)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_no_output)]
	mov 1,%o0
	st %o0,[%fp-44]
L234:
	ld [%fp-44],%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	bge L235
	nop
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-52],%o1
	add %o0,%o1,%o0
	ld [%o0],%o1
	cmp %o1,0
	be L237
	nop
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-52],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	mov 0,%o1
	mov 438,%o2
	call _open,0
	nop
	st %o0,[%fp-156]
	ld [%fp-156],%o0
	cmp %o0,0
	bge L238
	nop
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-52],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	call _perror_with_name,0
	nop
	mov 33,%i0
	b L1
	nop
L238:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-52],%o2
	add %o0,%o2,%o1
	ld [%fp-156],%o0
	ld [%o1],%o1
	sethi %hi(_outbuf),%o3
	or %o3,%lo(_outbuf),%o2
	mov 0,%o3
	mov 0,%o4
	call _finclude,0
	nop
L237:
L236:
	ld [%fp-44],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-44]
	b L234
	nop
L235:
	sethi %hi(_no_output),%o1
	sethi %hi(_no_output),%o0
	sethi %hi(_no_output),%o1
	ld [%o1+%lo(_no_output)],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_no_output)]
	ld [%fp-28],%o0
	cmp %o0,0
	be L240
	nop
	ld [%fp-28],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	bne L239
	nop
	b L240
	nop
L240:
	sethi %hi(LC23),%o1
	or %o1,%lo(LC23),%o0
	st %o0,[%fp-28]
	st %g0,[%fp-40]
	b L241
	nop
L239:
	ld [%fp-28],%o0
	mov 0,%o1
	mov 438,%o2
	call _open,0
	nop
	st %o0,[%fp-40]
	ld [%fp-40],%o0
	cmp %o0,0
	bge L242
	nop
	b L243
	nop
L242:
L241:
	sethi %hi(_print_deps),%o0
	ld [%o0+%lo(_print_deps)],%o1
	cmp %o1,0
	bne L244
	nop
	sethi %hi(LC83),%o1
	or %o1,%lo(LC83),%o0
	call _getenv,0
	nop
	cmp %o0,0
	bne L245
	nop
	sethi %hi(LC84),%o1
	or %o1,%lo(LC84),%o0
	call _getenv,0
	nop
	cmp %o0,0
	bne L245
	nop
	b L244
	nop
L245:
	sethi %hi(LC84),%o1
	or %o1,%lo(LC84),%o0
	call _getenv,0
	nop
	st %o0,[%fp-152]
	ld [%fp-152],%o0
	cmp %o0,0
	bne L246
	nop
	sethi %hi(LC83),%o1
	or %o1,%lo(LC83),%o0
	call _getenv,0
	nop
	st %o0,[%fp-152]
	sethi %hi(_print_deps),%o0
	mov 2,%o1
	st %o1,[%o0+%lo(_print_deps)]
	b L247
	nop
L246:
	sethi %hi(_print_deps),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_print_deps)]
L247:
	ld [%fp-152],%o0
	st %o0,[%fp-140]
L248:
	ld [%fp-140],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L249
	nop
	ld [%fp-140],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,32
	be L249
	nop
	ld [%fp-140],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-140]
	b L248
	nop
L249:
	ld [%fp-140],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L250
	nop
	ld [%fp-140],%o0
	add %o0,1,%o1
	st %o1,[%fp-104]
	ld [%fp-140],%o0
	ld [%fp-152],%o1
	sub %o0,%o1,%o0
	add %o0,1,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-116]
	ld [%fp-140],%o0
	ld [%fp-152],%o1
	sub %o0,%o1,%o2
	ld [%fp-152],%o0
	ld [%fp-116],%o1
	call _bcopy,0
	nop
	ld [%fp-140],%o0
	ld [%fp-152],%o1
	sub %o0,%o1,%o0
	ld [%fp-116],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
	b L251
	nop
L250:
	st %g0,[%fp-104]
	ld [%fp-152],%o0
	st %o0,[%fp-116]
L251:
	ld [%fp-116],%o0
	st %o0,[%fp-96]
L244:
	sethi %hi(_print_deps),%o0
	ld [%o0+%lo(_print_deps)],%o1
	cmp %o1,0
	be L252
	nop
	sethi %hi(_deps_allocated_size),%o0
	mov 200,%o1
	st %o1,[%o0+%lo(_deps_allocated_size)]
	sethi %hi(_deps_allocated_size),%o1
	ld [%o1+%lo(_deps_allocated_size)],%o0
	call _xmalloc,0
	nop
	sethi %hi(_deps_buffer),%o1
	st %o0,[%o1+%lo(_deps_buffer)]
	sethi %hi(_deps_buffer),%o1
	ld [%o1+%lo(_deps_buffer)],%o0
	stb %g0,[%o0]
	sethi %hi(_deps_size),%o0
	st %g0,[%o0+%lo(_deps_size)]
	sethi %hi(_deps_column),%o0
	st %g0,[%o0+%lo(_deps_column)]
	ld [%fp-104],%o0
	cmp %o0,0
	be L253
	nop
	ld [%fp-104],%o0
	mov 0,%o1
	call _deps_output,0
	nop
	sethi %hi(LC85),%o1
	or %o1,%lo(LC85),%o0
	mov 0,%o1
	call _deps_output,0
	nop
	b L254
	nop
L253:
	ld [%fp-28],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	bne L255
	nop
	sethi %hi(LC86),%o1
	or %o1,%lo(LC86),%o0
	mov 0,%o1
	call _deps_output,0
	nop
	b L256
	nop
L255:
	ld [%fp-28],%o0
	st %o0,[%fp-160]
	ld [%fp-160],%o0
	st %o0,[%fp-164]
L257:
	ld [%fp-164],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L258
	nop
	ld [%fp-164],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,47
	bne L259
	nop
	ld [%fp-164],%o0
	add %o0,1,%o1
	st %o1,[%fp-160]
L259:
	ld [%fp-164],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-164]
	b L257
	nop
L258:
	ld [%fp-160],%o0
	call _strlen,0
	nop
	st %o0,[%fp-140]
	ld [%fp-140],%o0
	ld [%fp-160],%o1
	add %o0,%o1,%o0
	ldub [%o0-2],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,46
	bne L260
	nop
	ld [%fp-140],%o0
	ld [%fp-160],%o1
	add %o0,%o1,%o0
	ldub [%o0-1],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,99
	bne L260
	nop
	ld [%fp-140],%o0
	add %o0,-2,%o1
	ld [%fp-160],%o0
	call _deps_output,0
	nop
	b L261
	nop
L260:
	ld [%fp-140],%o0
	ld [%fp-160],%o1
	add %o0,%o1,%o0
	ldub [%o0-2],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,46
	bne L262
	nop
	ld [%fp-140],%o0
	ld [%fp-160],%o1
	add %o0,%o1,%o0
	ldub [%o0-1],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,67
	bne L262
	nop
	ld [%fp-140],%o0
	add %o0,-2,%o1
	ld [%fp-160],%o0
	call _deps_output,0
	nop
	b L263
	nop
L262:
	ld [%fp-140],%o0
	ld [%fp-160],%o1
	add %o0,%o1,%o0
	ldub [%o0-3],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,46
	bne L264
	nop
	ld [%fp-140],%o0
	ld [%fp-160],%o1
	add %o0,%o1,%o0
	ldub [%o0-2],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,99
	bne L264
	nop
	ld [%fp-140],%o0
	ld [%fp-160],%o1
	add %o0,%o1,%o0
	ldub [%o0-1],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,99
	bne L264
	nop
	ld [%fp-140],%o0
	add %o0,-3,%o1
	ld [%fp-160],%o0
	call _deps_output,0
	nop
	b L265
	nop
L264:
	ld [%fp-140],%o0
	ld [%fp-160],%o1
	add %o0,%o1,%o0
	ldub [%o0-2],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,46
	bne L266
	nop
	ld [%fp-140],%o0
	ld [%fp-160],%o1
	add %o0,%o1,%o0
	ldub [%o0-1],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,115
	bne L266
	nop
	ld [%fp-140],%o0
	add %o0,-2,%o1
	ld [%fp-160],%o0
	call _deps_output,0
	nop
	b L267
	nop
L266:
	ld [%fp-140],%o0
	ld [%fp-160],%o1
	add %o0,%o1,%o0
	ldub [%o0-2],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,46
	bne L268
	nop
	ld [%fp-140],%o0
	ld [%fp-160],%o1
	add %o0,%o1,%o0
	ldub [%o0-1],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,83
	bne L268
	nop
	ld [%fp-140],%o0
	add %o0,-2,%o1
	ld [%fp-160],%o0
	call _deps_output,0
	nop
	b L269
	nop
L268:
	ld [%fp-140],%o0
	ld [%fp-160],%o1
	add %o0,%o1,%o0
	ldub [%o0-2],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,46
	bne L270
	nop
	ld [%fp-140],%o0
	ld [%fp-160],%o1
	add %o0,%o1,%o0
	ldub [%o0-1],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,109
	bne L270
	nop
	ld [%fp-140],%o0
	add %o0,-2,%o1
	ld [%fp-160],%o0
	call _deps_output,0
	nop
	b L271
	nop
L270:
	ld [%fp-160],%o0
	mov 0,%o1
	call _deps_output,0
	nop
L271:
L269:
L267:
L265:
L263:
L261:
	sethi %hi(LC87),%o1
	or %o1,%lo(LC87),%o0
	mov 0,%o1
	call _deps_output,0
	nop
	ld [%fp-28],%o0
	mov 0,%o1
	call _deps_output,0
	nop
	sethi %hi(LC88),%o1
	or %o1,%lo(LC88),%o0
	mov 0,%o1
	call _deps_output,0
	nop
L256:
L254:
L252:
	add %fp,-20,%o1
	add %fp,-24,%o2
	ld [%fp-40],%o0
	call _file_size_and_mode,0
	nop
	ld [%fp-48],%o0
	ld [%fp-48],%o1
	ld [%fp-28],%o2
	st %o2,[%o1]
	st %o2,[%o0+4]
	ld [%fp-48],%o0
	mov 1,%o1
	st %o1,[%o0+12]
	ld [%fp-48],%o0
	stb %g0,[%o0+40]
	ld [%fp-20],%o0
	sethi %hi(61440),%o1
	and %o0,%o1,%o0
	sethi %hi(32768),%o1
	cmp %o0,%o1
	be L272
	nop
	mov 2000,%o0
	st %o0,[%fp-140]
	st %g0,[%fp-164]
	ld [%fp-140],%o0
	add %o0,2,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	ld [%fp-48],%o1
	st %o0,[%o1+20]
	ld [%fp-48],%o0
	ld [%o0+20],%o1
	st %o1,[%fp-172]
L273:
	ld [%fp-140],%o0
	ld [%fp-164],%o1
	sub %o0,%o1,%o2
	ld [%fp-40],%o0
	ld [%fp-172],%o1
	call _read,0
	nop
	st %o0,[%fp-168]
	ld [%fp-168],%o0
	cmp %o0,0
	bge L276
	nop
	b L243
	nop
L276:
	ld [%fp-168],%o0
	cmp %o0,0
	bne L277
	nop
	b L274
	nop
L277:
	ld [%fp-164],%o0
	ld [%fp-168],%o1
	add %o0,%o1,%o0
	st %o0,[%fp-164]
	ld [%fp-172],%o0
	ld [%fp-168],%o1
	add %o0,%o1,%o0
	st %o0,[%fp-172]
	ld [%fp-140],%o0
	ld [%fp-164],%o1
	cmp %o0,%o1
	bne L278
	nop
	ld [%fp-140],%o0
	mov %o0,%o1
	sll %o1,1,%o0
	st %o0,[%fp-140]
	ld [%fp-48],%o0
	ld [%fp-140],%o2
	add %o2,2,%o1
	ld [%o0+20],%o0
	call _xrealloc,0
	nop
	ld [%fp-48],%o1
	st %o0,[%o1+20]
	ld [%fp-48],%o0
	ld [%o0+20],%o1
	ld [%fp-164],%o0
	add %o1,%o0,%o1
	st %o1,[%fp-172]
L278:
L275:
	b L273
	nop
L274:
	ld [%fp-48],%o0
	ld [%fp-164],%o1
	st %o1,[%o0+16]
	b L279
	nop
L272:
	ld [%fp-48],%o0
	st %g0,[%o0+16]
	ld [%fp-24],%o0
	add %o0,2,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	ld [%fp-48],%o1
	st %o0,[%o1+20]
L280:
	ld [%fp-24],%o0
	cmp %o0,0
	ble L281
	nop
	ld [%fp-48],%o0
	ld [%fp-48],%o1
	ld [%o0+20],%o0
	ld [%o1+16],%o2
	add %o0,%o2,%o1
	ld [%fp-40],%o0
	ld [%fp-24],%o2
	call _read,0
	nop
	st %o0,[%fp-164]
	ld [%fp-164],%o0
	cmp %o0,0
	bg L282
	nop
	ld [%fp-164],%o0
	cmp %o0,0
	bne L283
	nop
	b L281
	nop
L283:
	b L243
	nop
L282:
	ld [%fp-48],%o0
	ld [%fp-48],%o1
	ld [%o1+16],%o2
	ld [%fp-164],%o1
	add %o2,%o1,%o2
	st %o2,[%o0+16]
	ld [%fp-24],%o0
	ld [%fp-164],%o1
	sub %o0,%o1,%o0
	st %o0,[%fp-24]
	b L280
	nop
L281:
L279:
	ld [%fp-48],%o0
	ld [%fp-48],%o1
	ld [%o1+20],%o2
	st %o2,[%o0+24]
	ld [%fp-48],%o0
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o2
	st %o2,[%o0+32]
	ld [%fp-48],%o0
	ld [%o0+16],%o1
	cmp %o1,0
	ble L286
	nop
	ld [%fp-48],%o0
	ld [%fp-48],%o1
	ld [%o1+16],%o2
	ld [%o0+20],%o1
	add %o2,%o1,%o0
	ldub [%o0-1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L285
	nop
	b L286
	nop
L286:
	ld [%fp-48],%o0
	ld [%o0+16],%o1
	cmp %o1,1
	ble L284
	nop
	ld [%fp-48],%o0
	ld [%fp-48],%o1
	ld [%o1+16],%o2
	ld [%o0+20],%o1
	add %o2,%o1,%o0
	ldub [%o0-2],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	be L285
	nop
	b L284
	nop
L285:
	ld [%fp-48],%o0
	ld [%fp-48],%o2
	ld [%o2+16],%o1
	add %o1,1,%o3
	st %o3,[%o2+16]
	ld [%o0+20],%o2
	add %o1,%o2,%o0
	mov 10,%o1
	stb %o1,[%o0]
	mov 1,%o0
	st %o0,[%fp-88]
L284:
	ld [%fp-48],%o0
	ld [%fp-48],%o1
	ld [%o0+20],%o0
	ld [%o1+16],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
	sethi %hi(_no_trigraphs),%o0
	ld [%o0+%lo(_no_trigraphs)],%o1
	cmp %o1,0
	bne L287
	nop
	ld [%fp-48],%o0
	call _trigraph_pcp,0
	nop
L287:
	ld [%fp-32],%o0
	cmp %o0,0
	be L289
	nop
	ld [%fp-32],%o0
	sethi %hi(LC23),%o2
	or %o2,%lo(LC23),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L288
	nop
	b L289
	nop
L289:
	sethi %hi(LC89),%o1
	or %o1,%lo(LC89),%o0
	st %o0,[%fp-32]
	b L290
	nop
L288:
	ld [%fp-32],%o0
	sethi %hi(LC39),%o2
	or %o2,%lo(LC39),%o1
	sethi %hi(__iob+32),%o3
	or %o3,%lo(__iob+32),%o2
	call _freopen,0
	nop
	cmp %o0,0
	bne L291
	nop
	ld [%fp-32],%o0
	call _pfatal_with_name,0
	nop
L291:
L290:
	ld [%fp-48],%o0
	sethi %hi(_outbuf),%o2
	or %o2,%lo(_outbuf),%o1
	mov 0,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
	mov 1,%o0
	st %o0,[%fp-44]
L292:
	ld [%fp-44],%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	bge L293
	nop
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-68],%o1
	add %o0,%o1,%o0
	ld [%o0],%o1
	cmp %o1,0
	be L295
	nop
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-68],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	mov 0,%o1
	mov 438,%o2
	call _open,0
	nop
	st %o0,[%fp-164]
	ld [%fp-164],%o0
	cmp %o0,0
	bge L296
	nop
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-68],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	call _perror_with_name,0
	nop
	mov 33,%i0
	b L1
	nop
L296:
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp-68],%o2
	add %o0,%o2,%o1
	ld [%fp-164],%o0
	ld [%o1],%o1
	sethi %hi(_outbuf),%o3
	or %o3,%lo(_outbuf),%o2
	mov 0,%o3
	mov 0,%o4
	call _finclude,0
	nop
L295:
L294:
	ld [%fp-44],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-44]
	b L292
	nop
L293:
	sethi %hi(_outbuf),%o1
	or %o1,%lo(_outbuf),%o0
	mov 0,%o1
	call _rescan,0
	nop
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L297
	nop
	ld [%fp-88],%o0
	cmp %o0,0
	be L297
	nop
	sethi %hi(LC90),%o1
	or %o1,%lo(LC90),%o0
	call _pedwarn,0
	nop
L297:
	sethi %hi(_dump_macros),%o0
	ld [%o0+%lo(_dump_macros)],%o1
	cmp %o1,1
	bne L298
	nop
	call _dump_all_macros,0
	nop
	b L299
	nop
L298:
	ld [%fp-92],%o0
	cmp %o0,0
	bne L300
	nop
	call _write_output,0
	nop
L300:
L299:
	sethi %hi(_print_deps),%o0
	ld [%o0+%lo(_print_deps)],%o1
	cmp %o1,0
	be L301
	nop
	sethi %hi(_errors),%o0
	ld [%o0+%lo(_errors)],%o1
	cmp %o1,0
	bne L302
	nop
	ld [%fp-96],%o0
	cmp %o0,0
	be L303
	nop
	ld [%fp-96],%o0
	sethi %hi(LC91),%o2
	or %o2,%lo(LC91),%o1
	call _fopen,0
	nop
	st %o0,[%fp-100]
	ld [%fp-100],%o0
	cmp %o0,0
	bne L303
	nop
	ld [%fp-96],%o0
	call _pfatal_with_name,0
	nop
L303:
	sethi %hi(_deps_buffer),%o1
	ld [%o1+%lo(_deps_buffer)],%o0
	ld [%fp-100],%o1
	call _fputs,0
	nop
	ld [%fp-100],%o1
	ld [%fp-100],%o0
	ld [%fp-100],%o1
	ld [%o1],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0]
	cmp %o2,0
	bge L304
	nop
	mov 10,%o0
	ld [%fp-100],%o1
	call __flsbuf,0
	nop
	b L305
	nop
L304:
	ld [%fp-100],%o1
	ld [%o1+4],%o0
	add %o0,1,%o2
	st %o2,[%o1+4]
	mov 10,%o1
	stb %o1,[%o0]
L305:
	ld [%fp-96],%o0
	cmp %o0,0
	be L306
	nop
	ld [%fp-100],%o0
	lduh [%o0+16],%o1
	and %o1,32,%o0
	sll %o0,16,%o1
	sra %o1,16,%o0
	cmp %o0,0
	bne L308
	nop
	ld [%fp-100],%o0
	call _fclose,0
	nop
	cmp %o0,0
	bne L308
	nop
	b L307
	nop
L308:
	sethi %hi(LC92),%o1
	or %o1,%lo(LC92),%o0
	call _fatal,0
	nop
L307:
L306:
L302:
L301:
	sethi %hi(__iob+32),%o0
	mov 16,%o1
	or %o0,%lo(__iob+32),%o2
	add %o1,%o2,%o0
	lduh [%o0],%o1
	and %o1,32,%o0
	sll %o0,16,%o1
	sra %o1,16,%o0
	cmp %o0,0
	bne L310
	nop
	sethi %hi(__iob+32),%o1
	or %o1,%lo(__iob+32),%o0
	call _fclose,0
	nop
	cmp %o0,0
	bne L310
	nop
	b L309
	nop
L310:
	sethi %hi(LC92),%o1
	or %o1,%lo(LC92),%o0
	call _fatal,0
	nop
L309:
	sethi %hi(_errors),%o0
	ld [%o0+%lo(_errors)],%o1
	cmp %o1,0
	be L311
	nop
	mov 33,%o0
	call _exit,0
	nop
L311:
	mov 0,%o0
	call _exit,0
	nop
L243:
	ld [%fp-28],%o0
	call _pfatal_with_name,0
	nop
	mov 0,%i0
	b L1
	nop
L1:
	ret
	restore
	.align 4
	.proc	020
_path_include:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L313
	nop
L314:
	ld [%fp-20],%o0
	st %o0,[%fp-24]
L316:
	ld [%fp-24],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L317
	nop
	ld [%fp-24],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,58
	be L317
	nop
	ld [%fp-24],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
	b L316
	nop
L317:
	ld [%fp-20],%o0
	ld [%fp-24],%o1
	cmp %o0,%o1
	bne L318
	nop
	mov 2,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-28]
	ld [%fp-28],%o0
	mov 46,%o1
	stb %o1,[%o0]
	mov 1,%o0
	ld [%fp-28],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
	b L319
	nop
L318:
	ld [%fp-24],%o0
	ld [%fp-20],%o1
	sub %o0,%o1,%o0
	add %o0,1,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-28]
	ld [%fp-24],%o0
	ld [%fp-20],%o1
	sub %o0,%o1,%o2
	ld [%fp-20],%o0
	ld [%fp-28],%o1
	call _bcopy,0
	nop
	ld [%fp-24],%o0
	ld [%fp-20],%o1
	sub %o0,%o1,%o0
	ld [%fp-28],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
L319:
	mov 12,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-32]
	ld [%fp-32],%o0
	st %g0,[%o0]
	ld [%fp-32],%o0
	st %g0,[%o0+8]
	ld [%fp-32],%o0
	ld [%fp-28],%o1
	st %o1,[%o0+4]
	ld [%fp-32],%o0
	ld [%fp-32],%o1
	call _append_include_chain,0
	nop
	ld [%fp-24],%o0
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	bne L320
	nop
	b L315
	nop
L320:
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L314
	nop
L315:
L313:
L312:
	ret
	restore
	.align 8
LC93:
	.ascii "%d trigraph(s) encountered\0"
	.align 4
	.proc	020
_trigraph_pcp:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	ld [%o0+20],%l3
	mov %l3,%o0
	mov %o0,%l2
	mov %o0,%l1
L322:
	mov %l3,%o0
	mov 63,%o1
	call _index,0
	nop
	mov %o0,%l3
	mov %l3,%o0
	cmp %o0,0
	be L323
	nop
	add %l3,1,%l3
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,63
	be L324
	nop
	b L322
	nop
L324:
	add %l3,1,%l3
	ldub [%l3],%o0
	and %o0,0xff,%o1
	add %o1,-33,%o0
	mov 30,%o1
	cmp %o1,%o0
	blu L336
	nop
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(L337),%o2
	or %o2,%lo(L337),%o1
	ld [%o0+%o1],%o0
	jmp %o0
	nop
L337:
	.word	L332
	.word	L336
	.word	L336
	.word	L336
	.word	L336
	.word	L336
	.word	L330
	.word	L327
	.word	L329
	.word	L336
	.word	L336
	.word	L336
	.word	L334
	.word	L336
	.word	L328
	.word	L336
	.word	L336
	.word	L336
	.word	L336
	.word	L336
	.word	L336
	.word	L336
	.word	L336
	.word	L336
	.word	L336
	.word	L336
	.word	L336
	.word	L331
	.word	L326
	.word	L333
	.word	L335
L326:
	mov 35,%l0
	b L325
	nop
L327:
	mov 91,%l0
	b L325
	nop
L328:
	mov 92,%l0
	b L325
	nop
L329:
	mov 93,%l0
	b L325
	nop
L330:
	mov 94,%l0
	b L325
	nop
L331:
	mov 123,%l0
	b L325
	nop
L332:
	mov 124,%l0
	b L325
	nop
L333:
	mov 125,%l0
	b L325
	nop
L334:
	mov 126,%l0
	b L325
	nop
L335:
	add %l3,-1,%l3
	b L322
	nop
L336:
	b L322
	nop
L325:
	sub %l3,%l1,%o0
	add %o0,-2,%o1
	st %o1,[%fp-20]
	cmp %l2,%l1
	be L338
	nop
	ld [%fp-20],%o0
	cmp %o0,0
	ble L338
	nop
	mov %l1,%o0
	mov %l2,%o1
	ld [%fp-20],%o2
	call _bcopy,0
	nop
L338:
	ld [%fp-20],%o0
	add %l2,%o0,%l2
	stb %l0,[%l2]
	add %l2,1,%l2
	add %l3,1,%l3
	mov %l3,%l1
	b L322
	nop
L323:
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o1+20],%o2
	sub %l1,%o2,%o1
	ld [%o0+16],%o0
	sub %o0,%o1,%o1
	st %o1,[%fp-20]
	cmp %l2,%l1
	be L339
	nop
	ld [%fp-20],%o0
	cmp %o0,0
	ble L339
	nop
	mov %l1,%o0
	mov %l2,%o1
	ld [%fp-20],%o2
	call _bcopy,0
	nop
L339:
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	sub %l1,%l2,%o2
	ld [%o1+16],%o1
	sub %o1,%o2,%o2
	st %o2,[%o0+16]
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o0+20],%o0
	ld [%o1+16],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
	sethi %hi(_warn_trigraphs),%o0
	ld [%o0+%lo(_warn_trigraphs)],%o1
	cmp %o1,0
	be L340
	nop
	cmp %l1,%l2
	be L340
	nop
	sub %l1,%l2,%o1
	mov %o1,%o0
	srl %o0,31,%o1
	add %o0,%o1,%o0
	sra %o0,1,%o1
	sethi %hi(LC93),%o2
	or %o2,%lo(LC93),%o0
	call _warning,0
	nop
L340:
L321:
	ret
	restore
	.align 4
	.proc	020
_newline_fix:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	mov 0,%o1
L342:
	ldub [%o0],%o3
	and %o3,0xff,%o2
	cmp %o2,92
	bne L344
	nop
	ldub [%o0+1],%o3
	and %o3,0xff,%o2
	cmp %o2,10
	bne L345
	nop
	add %o0,2,%o0
	add %o1,1,%o1
	b L346
	nop
L345:
	ldub [%o0+1],%o3
	and %o3,0xff,%o2
	cmp %o2,13
	bne L347
	nop
	ldub [%o0+2],%o3
	and %o3,0xff,%o2
	cmp %o2,10
	bne L347
	nop
	add %o0,3,%o0
	add %o1,1,%o1
	b L348
	nop
L347:
	b L343
	nop
L348:
L346:
	b L349
	nop
L344:
	b L343
	nop
L349:
	b L342
	nop
L343:
	cmp %o1,0
	be L351
	nop
	ldub [%o0],%o3
	and %o3,0xff,%o2
	cmp %o2,47
	be L350
	nop
	ldub [%o0],%o3
	and %o3,0xff,%o2
	cmp %o2,42
	bne L351
	nop
	b L350
	nop
L351:
	b L341
	nop
L350:
	nop
L352:
	ldub [%o0],%o3
	and %o3,0xff,%o2
	cmp %o2,42
	be L354
	nop
	ldub [%o0],%o3
	and %o3,0xff,%o2
	cmp %o2,47
	be L354
	nop
	b L353
	nop
L354:
	ld [%fp+68],%o2
	add %o2,1,%o3
	st %o3,[%fp+68]
	ldub [%o0],%o3
	stb %o3,[%o2]
	add %o0,1,%o0
	b L352
	nop
L353:
	nop
L355:
	mov %o1,%o2
	add %o1,-1,%o1
	cmp %o2,0
	ble L356
	nop
	ld [%fp+68],%o2
	add %o2,1,%o3
	st %o3,[%fp+68]
	mov 92,%o3
	stb %o3,[%o2]
	ld [%fp+68],%o2
	add %o2,1,%o3
	st %o3,[%fp+68]
	mov 10,%o3
	stb %o3,[%o2]
	b L355
	nop
L356:
L341:
	ret
	restore
	.align 4
	.proc	020
_name_newline_fix:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	mov 0,%o1
L358:
	ldub [%o0],%o3
	and %o3,0xff,%o2
	cmp %o2,92
	bne L360
	nop
	ldub [%o0+1],%o3
	and %o3,0xff,%o2
	cmp %o2,10
	bne L361
	nop
	add %o0,2,%o0
	add %o1,1,%o1
	b L362
	nop
L361:
	ldub [%o0+1],%o3
	and %o3,0xff,%o2
	cmp %o2,13
	bne L363
	nop
	ldub [%o0+2],%o3
	and %o3,0xff,%o2
	cmp %o2,10
	bne L363
	nop
	add %o0,3,%o0
	add %o1,1,%o1
	b L364
	nop
L363:
	b L359
	nop
L364:
L362:
	b L365
	nop
L360:
	b L359
	nop
L365:
	b L358
	nop
L359:
	cmp %o1,0
	be L367
	nop
	ldub [%o0],%o3
	and %o3,0xff,%o2
	sethi %hi(_is_idchar),%o4
	or %o4,%lo(_is_idchar),%o3
	ldub [%o2+%o3],%o4
	and %o4,0xff,%o2
	cmp %o2,0
	bne L366
	nop
	b L367
	nop
L367:
	b L357
	nop
L366:
	nop
L368:
	ldub [%o0],%o3
	and %o3,0xff,%o2
	sethi %hi(_is_idchar),%o4
	or %o4,%lo(_is_idchar),%o3
	ldub [%o2+%o3],%o4
	and %o4,0xff,%o2
	cmp %o2,0
	be L369
	nop
	ld [%fp+68],%o2
	add %o2,1,%o3
	st %o3,[%fp+68]
	ldub [%o0],%o3
	stb %o3,[%o2]
	add %o0,1,%o0
	b L368
	nop
L369:
	nop
L370:
	mov %o1,%o2
	add %o1,-1,%o1
	cmp %o2,0
	ble L371
	nop
	ld [%fp+68],%o2
	add %o2,1,%o3
	st %o3,[%fp+68]
	mov 92,%o3
	stb %o3,[%o2]
	ld [%fp+68],%o2
	add %o2,1,%o3
	st %o3,[%fp+68]
	mov 10,%o3
	stb %o3,[%o2]
	b L370
	nop
L371:
L357:
	ret
	restore
	.align 8
LC94:
	.ascii "NOTREACHED\0"
	.align 8
LC95:
	.ascii "ARGSUSED\0"
	.align 8
LC96:
	.ascii "LINTLIBRARY\0"
	.align 8
LC97:
	.ascii "VARARGS\0"
	.align 4
	.proc	0102
_get_lintcmd:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	mov %i0,%l1
	mov %i1,%l0
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	st %i4,[%fp+84]
	ld [%fp+80],%o0
	st %g0,[%o0]
L373:
L376:
	ldub [%l1],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L377
	nop
	add %l1,1,%l1
	b L376
	nop
L377:
L375:
	b L374
	nop
	b L373
	nop
L374:
	cmp %l1,%l0
	blu L378
	nop
	mov 0,%i0
	b L372
	nop
L378:
	sub %l0,%l1,%o0
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	cmp %o0,9
	ble L379
	nop
	mov %l1,%o0
	sethi %hi(LC94),%o2
	or %o2,%lo(LC94),%o1
	mov 10,%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L379
	nop
	ld [%fp+84],%o0
	mov 10,%o1
	st %o1,[%o0]
	sethi %hi(LC94),%o0
	or %o0,%lo(LC94),%i0
	b L372
	nop
L379:
	ld [%fp-20],%o0
	cmp %o0,7
	ble L380
	nop
	mov %l1,%o0
	sethi %hi(LC95),%o2
	or %o2,%lo(LC95),%o1
	mov 8,%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L380
	nop
	ld [%fp+84],%o0
	mov 8,%o1
	st %o1,[%o0]
	sethi %hi(LC95),%o0
	or %o0,%lo(LC95),%i0
	b L372
	nop
L380:
	ld [%fp-20],%o0
	cmp %o0,10
	ble L381
	nop
	mov %l1,%o0
	sethi %hi(LC96),%o2
	or %o2,%lo(LC96),%o1
	mov 11,%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L381
	nop
	ld [%fp+84],%o0
	mov 11,%o1
	st %o1,[%o0]
	sethi %hi(LC96),%o0
	or %o0,%lo(LC96),%i0
	b L372
	nop
L381:
	ld [%fp-20],%o0
	cmp %o0,6
	ble L382
	nop
	mov %l1,%o0
	sethi %hi(LC97),%o2
	or %o2,%lo(LC97),%o1
	mov 7,%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L382
	nop
	ld [%fp+84],%o0
	mov 7,%o1
	st %o1,[%o0]
	add %l1,7,%l1
	ld [%fp-20],%o0
	add %o0,-7,%o1
	st %o1,[%fp-20]
	ld [%fp-20],%o0
	cmp %o0,0
	be L384
	nop
	sethi %hi(___lc_ctype),%o1
	ld [%o1+%lo(___lc_ctype)],%o0
	ld [%o0+28],%o1
	cmp %o1,0
	bne L385
	nop
	sethi %hi(___lc_ctype),%o1
	ld [%o1+%lo(___lc_ctype)],%o0
	ldub [%l1],%o2
	and %o2,0xff,%o1
	mov %o1,%o2
	sll %o2,2,%o1
	ld [%o0+64],%o2
	add %o1,%o2,%o0
	ld [%o0],%o1
	and %o1,16,%o0
	cmp %o0,0
	bne L383
	nop
	b L384
	nop
L385:
	sethi %hi(___lc_ctype),%o0
	ld [%o0+%lo(___lc_ctype)],%o3
	ldub [%l1],%o1
	and %o1,0xff,%o0
	sethi %hi(___lc_ctype),%o2
	mov 16,%o1
	ld [%o2+%lo(___lc_ctype)],%o2
	ld [%o3+28],%o3
	call %o3,0
	nop
	cmp %o0,0
	bne L383
	nop
	b L384
	nop
L386:
L384:
	sethi %hi(LC97),%o0
	or %o0,%lo(LC97),%i0
	b L372
	nop
L383:
	nop
	ld [%fp+76],%o0
	mov %l1,%o1
	st %o1,[%o0]
	mov %o1,%l2
L387:
	cmp %l2,%l0
	bgeu L388
	nop
	sethi %hi(___lc_ctype),%o1
	ld [%o1+%lo(___lc_ctype)],%o0
	ld [%o0+28],%o1
	cmp %o1,0
	bne L390
	nop
	sethi %hi(___lc_ctype),%o1
	ld [%o1+%lo(___lc_ctype)],%o0
	ldub [%l2],%o2
	and %o2,0xff,%o1
	mov %o1,%o2
	sll %o2,2,%o1
	ld [%o0+64],%o2
	add %o1,%o2,%o0
	ld [%o0],%o1
	and %o1,16,%o0
	cmp %o0,0
	bne L391
	nop
	b L388
	nop
L390:
	sethi %hi(___lc_ctype),%o0
	ld [%o0+%lo(___lc_ctype)],%o3
	ldub [%l2],%o1
	and %o1,0xff,%o0
	sethi %hi(___lc_ctype),%o2
	mov 16,%o1
	ld [%o2+%lo(___lc_ctype)],%o2
	ld [%o3+28],%o3
	call %o3,0
	nop
	cmp %o0,0
	bne L391
	nop
	b L388
	nop
L391:
L389:
	add %l2,1,%l2
	b L387
	nop
L388:
	ld [%fp+80],%o0
	ld [%fp+76],%o1
	ld [%o1],%o2
	sub %l2,%o2,%o1
	st %o1,[%o0]
	sethi %hi(LC97),%o0
	or %o0,%lo(LC97),%i0
	b L372
	nop
L382:
	mov 0,%i0
	b L372
	nop
L372:
	ret
	restore
	.align 8
LC98:
	.ascii "unterminated string or character constant\0"
	.align 8
LC99:
	.ascii "possible real start of unterminated constant\0"
	.align 8
LC100:
	.ascii "#pragma lint \0"
	.align 8
LC101:
	.ascii "`/*' within comment\0"
	.align 8
LC102:
	.ascii "unterminated comment\0"
	.align 8
LC103:
	.ascii "unterminated `#%s' conditional\0"
	.align 4
	.proc	020
_rescan:
	!#PROLOGUE# 0
	save %sp,-184,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	mov 0,%l1
	mov 0,%l2
	st %g0,[%fp-24]
	st %g0,[%fp-28]
	st %g0,[%fp-36]
	sethi %hi(_no_output),%o0
	ld [%o0+%lo(_no_output)],%o1
	cmp %o1,0
	be L393
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	ld [%o0+%o1],%o0
	cmp %o0,0
	be L393
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o1
	or %o1,%lo(_instack),%o2
	add %o0,%o2,%o1
	mov %o1,%o0
	mov 1,%o1
	call _skip_if_group,0
	nop
L393:
	ld [%fp+68],%o0
	ld [%o0+24],%l5
L394:
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	ld [%o0+24],%l3
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o0+20],%o0
	ld [%o1+16],%o1
	add %o0,%o1,%l4
	ld [%fp+68],%o0
	st %l5,[%o0+24]
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%fp+68],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	sub %l4,%l3,%o1
	cmp %o0,%o1
	bg L397
	nop
	sub %l4,%l3,%o1
	ld [%fp+68],%o0
	call _grow_outbuf,0
	nop
	b L398
	nop
L397:
L398:
	st %g0,[%fp-40]
	ld [%fp+68],%o0
	ld [%o0+24],%l5
L396:
	b L395
	nop
	b L394
	nop
L395:
	st %l3,[%fp-40]
	ldub [%l4],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L399
	nop
	call _abort,0
	nop
L399:
	nop
L400:
	ldub [%l3],%l0
	add %l3,1,%l3
	stb %l0,[%l5]
	add %l5,1,%l5
	and %l0,0xff,%o0
	mov 122,%o1
	cmp %o1,%o0
	blu L646
	nop
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(L710),%o2
	or %o2,%lo(L710),%o1
	ld [%o0+%o1],%o0
	jmp %o0
	nop
L710:
	.word	L631
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L614
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L458
	.word	L408
	.word	L532
	.word	L646
	.word	L646
	.word	L459
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L494
	.word	L535
	.word	L535
	.word	L535
	.word	L535
	.word	L535
	.word	L535
	.word	L535
	.word	L535
	.word	L535
	.word	L535
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L646
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L588
	.word	L646
	.word	L403
	.word	L646
	.word	L646
	.word	L561
	.word	L646
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
	.word	L562
L403:
	cmp %l3,%l4
	blu L404
	nop
	b L402
	nop
L404:
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L405
	nop
	add %l3,1,%l3
	ld [%fp-20],%o1
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	add %l5,-1,%l5
	b L402
	nop
L405:
	cmp %l1,0
	ble L406
	nop
	b L407
	nop
L406:
	ldub [%l3],%o0
	stb %o0,[%l5]
	add %l3,1,%l3
	add %l5,1,%l5
	b L402
	nop
L408:
	sethi %hi(_assertions_flag),%o0
	ld [%o0+%lo(_assertions_flag)],%o1
	cmp %o1,0
	be L409
	nop
L410:
L413:
	ldub [%l3],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L414
	nop
	add %l3,1,%l3
	b L413
	nop
L414:
L412:
	b L411
	nop
	b L410
	nop
L411:
	nop
L415:
	ldub [%l3],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L416
	nop
	ldub [%l3],%o0
	stb %o0,[%l5]
	add %l3,1,%l3
	add %l5,1,%l5
	b L415
	nop
L416:
	nop
L417:
L420:
	ldub [%l3],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L421
	nop
	add %l3,1,%l3
	b L420
	nop
L421:
L419:
	b L418
	nop
	b L417
	nop
L418:
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,40
	bne L422
	nop
	ld [%fp-20],%o0
	st %l3,[%o0+24]
	ld [%fp-20],%o0
	call _skip_paren_group,0
	nop
	ld [%fp-20],%o0
	ld [%o0+24],%o1
	sub %o1,%l3,%o2
	mov %l3,%o0
	mov %l5,%o1
	call _bcopy,0
	nop
	ld [%fp-20],%o0
	ld [%o0+24],%o1
	sub %o1,%l3,%o0
	add %l5,%o0,%l5
	ld [%fp-20],%o0
	ld [%o0+24],%l3
L422:
L409:
	ld [%fp-20],%o0
	ld [%o0+28],%o1
	cmp %o1,0
	be L423
	nop
	b L424
	nop
L423:
	ld [%fp-20],%o0
	ld [%o0],%o1
	cmp %o1,0
	bne L425
	nop
	ld [%fp-20],%o0
	ld [%fp-40],%o1
	ld [%o0+20],%o0
	cmp %o1,%o0
	bne L425
	nop
	b L424
	nop
L425:
	cmp %l1,0
	be L426
	nop
	b L407
	nop
L426:
	ld [%fp-40],%o0
	cmp %o0,0
	bne L427
	nop
	b L424
	nop
L427:
	ld [%fp-40],%o0
	st %o0,[%fp-44]
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	bne L428
	nop
L429:
	ld [%fp-44],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L431
	nop
	ld [%fp-44],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-44]
	b L432
	nop
L431:
	ld [%fp-44],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L433
	nop
	mov 1,%o0
	ld [%fp-44],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L433
	nop
	ld [%fp-44],%o0
	add %o0,2,%o1
	st %o1,[%fp-44]
	b L434
	nop
L433:
	ld [%fp-44],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L435
	nop
	mov 1,%o0
	ld [%fp-44],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L435
	nop
	ld [%fp-44],%o0
	add %o0,2,%o1
	st %o1,[%fp-44]
L436:
	ld [%fp-44],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L438
	nop
	mov 1,%o0
	ld [%fp-44],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L438
	nop
	b L437
	nop
L438:
	ld [%fp-44],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-44]
	b L436
	nop
L437:
	ld [%fp-44],%o0
	add %o0,2,%o1
	st %o1,[%fp-44]
	b L439
	nop
L435:
	sethi %hi(_cplusplus_comments),%o0
	ld [%o0+%lo(_cplusplus_comments)],%o1
	cmp %o1,0
	be L440
	nop
	ld [%fp-44],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L440
	nop
	mov 1,%o0
	ld [%fp-44],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L440
	nop
	ld [%fp-44],%o0
	add %o0,2,%o1
	st %o1,[%fp-44]
L441:
	ld [%fp-44],%o0
	add %o0,1,%o1
	st %o1,[%fp-44]
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	be L442
	nop
	b L441
	nop
L442:
	b L443
	nop
L440:
	b L430
	nop
L443:
L439:
L434:
L432:
	b L429
	nop
L430:
L428:
	ld [%fp-44],%o1
	add %o1,1,%o0
	cmp %o0,%l3
	be L444
	nop
	b L424
	nop
L444:
	add %l5,-1,%l5
	ld [%fp-20],%o0
	st %l3,[%o0+24]
	ld [%fp+68],%o0
	st %l5,[%o0+24]
	ld [%fp-20],%o0
	ld [%fp+68],%o1
	call _handle_directive,0
	nop
	cmp %o0,0
	bne L445
	nop
	sethi %hi(_no_output),%o0
	ld [%o0+%lo(_no_output)],%o1
	cmp %o1,0
	be L446
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	ld [%o0+%o1],%o0
	cmp %o0,0
	be L446
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o1
	or %o1,%lo(_instack),%o2
	add %o0,%o2,%o1
	mov %o1,%o0
	mov 1,%o1
	call _skip_if_group,0
	nop
L447:
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	ld [%o0+24],%l3
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o0+20],%o0
	ld [%o1+16],%o1
	add %o0,%o1,%l4
	ld [%fp+68],%o0
	st %l5,[%o0+24]
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%fp+68],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	sub %l4,%l3,%o1
	cmp %o0,%o1
	bg L450
	nop
	sub %l4,%l3,%o1
	ld [%fp+68],%o0
	call _grow_outbuf,0
	nop
	b L451
	nop
L450:
L451:
	st %g0,[%fp-40]
	ld [%fp+68],%o0
	ld [%o0+24],%l5
L449:
	b L448
	nop
	b L447
	nop
L448:
	st %l3,[%fp-40]
	b L402
	nop
L446:
	add %l5,1,%l5
	b L424
	nop
L445:
	sethi %hi(_no_output),%o0
	ld [%o0+%lo(_no_output)],%o1
	cmp %o1,0
	be L452
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	ld [%o0+%o1],%o0
	cmp %o0,0
	be L452
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o1
	or %o1,%lo(_instack),%o2
	add %o0,%o2,%o1
	mov %o1,%o0
	mov 1,%o1
	call _skip_if_group,0
	nop
L452:
	ld [%fp+68],%o0
	ld [%o0+24],%l5
L453:
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	ld [%o0+24],%l3
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o0+20],%o0
	ld [%o1+16],%o1
	add %o0,%o1,%l4
	ld [%fp+68],%o0
	st %l5,[%o0+24]
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%fp+68],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	sub %l4,%l3,%o1
	cmp %o0,%o1
	bg L456
	nop
	sub %l4,%l3,%o1
	ld [%fp+68],%o0
	call _grow_outbuf,0
	nop
	b L457
	nop
L456:
L457:
	st %g0,[%fp-40]
	ld [%fp+68],%o0
	ld [%o0+24],%l5
L455:
	b L454
	nop
	b L453
	nop
L454:
	st %l3,[%fp-40]
	b L402
	nop
L458:
L459:
	cmp %l1,0
	be L460
	nop
	b L407
	nop
L460:
	ld [%fp-20],%o0
	ld [%o0+12],%o1
	st %o1,[%fp-32]
L461:
	cmp %l3,%l4
	blu L463
	nop
	ld [%fp-20],%o0
	ld [%o0+28],%o1
	cmp %o1,0
	be L464
	nop
L465:
	ld [%fp-20],%o0
	ld [%o0+28],%o1
	mov 31,%o0
	st %o0,[%o1+12]
	ld [%fp-20],%o0
	ld [%o0+36],%o1
	cmp %o1,0
	be L468
	nop
	ld [%fp-20],%o1
	ld [%o1+36],%o0
	call _free,0
	nop
L468:
	sethi %hi(_indepth),%o1
	sethi %hi(_indepth),%o0
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_indepth)]
L467:
	b L466
	nop
	b L465
	nop
L466:
	nop
L469:
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	ld [%o0+24],%l3
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o0+20],%o0
	ld [%o1+16],%o1
	add %o0,%o1,%l4
	ld [%fp+68],%o0
	st %l5,[%o0+24]
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%fp+68],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	sub %l4,%l3,%o1
	cmp %o0,%o1
	bg L472
	nop
	sub %l4,%l3,%o1
	ld [%fp+68],%o0
	call _grow_outbuf,0
	nop
	b L473
	nop
L472:
L473:
	st %g0,[%fp-40]
	ld [%fp+68],%o0
	ld [%o0+24],%l5
L471:
	b L470
	nop
	b L469
	nop
L470:
	b L461
	nop
L464:
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	bne L474
	nop
	ld [%fp-32],%o0
	call _line_for_error,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	sethi %hi(LC98),%o2
	or %o2,%lo(LC98),%o1
	call _error_with_line,0
	nop
	ld [%fp-36],%o0
	sethi %hi(LC99),%o2
	or %o2,%lo(LC99),%o1
	call _error_with_line,0
	nop
	st %g0,[%fp-36]
L474:
	b L462
	nop
L463:
	ldub [%l3],%o0
	stb %o0,[%l5]
	add %l5,1,%l5
	ldub [%l3],%o1
	and %o1,0xff,%o0
	add %l3,1,%l3
	cmp %o0,39
	be L489
	nop
	cmp %o0,39
	bg L493
	nop
	cmp %o0,10
	be L476
	nop
	cmp %o0,34
	be L488
	nop
	b L492
	nop
L493:
	cmp %o0,92
	be L482
	nop
	b L492
	nop
L476:
	ld [%fp-20],%o1
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	ld [%fp+68],%o1
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	be L477
	nop
	st %l3,[%fp-40]
	b L478
	nop
L477:
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	bne L480
	nop
	and %l0,0xff,%o0
	cmp %o0,39
	be L480
	nop
	b L479
	nop
L480:
	ld [%fp-32],%o0
	call _line_for_error,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	sethi %hi(LC98),%o2
	or %o2,%lo(LC98),%o1
	call _error_with_line,0
	nop
	b L478
	nop
L479:
	ld [%fp-36],%o0
	cmp %o0,0
	bne L481
	nop
	ld [%fp-20],%o0
	ld [%o0+12],%o1
	add %o1,-1,%o0
	st %o0,[%fp-36]
L481:
	b L475
	nop
L482:
	cmp %l3,%l4
	blu L483
	nop
	b L475
	nop
L483:
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L484
	nop
	add %l5,-1,%l5
	add %l3,1,%l3
	ld [%fp-20],%o1
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	b L485
	nop
L484:
L486:
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L487
	nop
	ldub [%l3+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L487
	nop
	add %l3,2,%l3
	ld [%fp-20],%o1
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	b L486
	nop
L487:
	ldub [%l3],%o0
	stb %o0,[%l5]
	add %l3,1,%l3
	add %l5,1,%l5
L485:
	b L475
	nop
L488:
L489:
	ldub [%l3-1],%o1
	and %o1,0xff,%o0
	and %l0,0xff,%o1
	cmp %o0,%o1
	bne L490
	nop
	b L478
	nop
L490:
	b L475
	nop
L492:
L475:
	b L461
	nop
L462:
	nop
L478:
	b L402
	nop
L494:
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L495
	nop
	ldub [%l3+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L495
	nop
	mov %l3,%o0
	call _newline_fix,0
	nop
L495:
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	be L496
	nop
	sethi %hi(_cplusplus_comments),%o0
	ld [%o0+%lo(_cplusplus_comments)],%o1
	cmp %o1,0
	be L497
	nop
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L497
	nop
	b L496
	nop
L497:
	b L424
	nop
L496:
	ld [%fp-20],%o0
	ld [%o0+28],%o1
	cmp %o1,0
	be L498
	nop
	b L424
	nop
L498:
	cmp %l1,0
	be L499
	nop
	b L407
	nop
L499:
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L500
	nop
	ld [%fp-20],%o0
	ld [%o0+12],%o1
	st %o1,[%fp-32]
	add %l3,-1,%l3
	add %l5,-1,%l5
	sethi %hi(_put_out_comments),%o0
	ld [%o0+%lo(_put_out_comments)],%o1
	cmp %o1,0
	bne L501
	nop
	mov 32,%o0
	stb %o0,[%l5]
	add %l5,1,%l5
	b L502
	nop
L501:
	mov 47,%o0
	stb %o0,[%l5]
	add %l5,1,%l5
	mov 47,%o0
	stb %o0,[%l5]
	add %l5,1,%l5
L502:
	add %l3,2,%o0
	st %o0,[%fp-44]
L503:
	cmp %l3,%l4
	bgeu L504
	nop
	ldub [%l3],%o1
	and %o1,0xff,%o0
	add %l3,1,%l3
	cmp %o0,10
	bne L505
	nop
	add %l3,-1,%l3
	sethi %hi(_put_out_comments),%o0
	ld [%o0+%lo(_put_out_comments)],%o1
	cmp %o1,0
	be L506
	nop
	ld [%fp-44],%o0
	sub %l3,%o0,%o2
	ld [%fp-44],%o0
	mov %l5,%o1
	call _bcopy,0
	nop
	ld [%fp-44],%o1
	sub %l3,%o1,%o0
	add %l5,%o0,%l5
L506:
	b L504
	nop
L505:
	b L503
	nop
L504:
	b L402
	nop
L500:
	ld [%fp-20],%o0
	ld [%o0+12],%o1
	st %o1,[%fp-32]
	add %l3,1,%l3
	sethi %hi(_lint),%o0
	ld [%o0+%lo(_lint)],%o1
	cmp %o1,0
	be L507
	nop
	add %fp,-44,%o2
	add %fp,-52,%o3
	add %fp,-48,%o4
	mov %l3,%o0
	mov %l4,%o1
	call _get_lintcmd,0
	nop
	st %o0,[%fp-56]
	ld [%fp-56],%o0
	cmp %o0,0
	be L508
	nop
	mov 10,%o0
	stb %o0,[%l5-1]
	sethi %hi(LC100),%o1
	or %o1,%lo(LC100),%o0
	mov %l5,%o1
	mov 13,%o2
	call _bcopy,0
	nop
	add %l5,13,%l5
	ld [%fp-56],%o0
	mov %l5,%o1
	ld [%fp-48],%o2
	call _bcopy,0
	nop
	ld [%fp-48],%o0
	add %l5,%o0,%l5
	ld [%fp-52],%o0
	cmp %o0,0
	be L509
	nop
	mov 32,%o0
	stb %o0,[%l5]
	add %l5,1,%l5
	ld [%fp-44],%o0
	mov %l5,%o1
	ld [%fp-52],%o2
	call _bcopy,0
	nop
	ld [%fp-52],%o0
	add %l5,%o0,%l5
L509:
	ld [%fp-20],%o0
	ld [%fp+68],%o1
	mov 0,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
	mov 32,%o0
	stb %o0,[%l5]
	add %l5,1,%l5
	mov 47,%o0
	stb %o0,[%l5]
	add %l5,1,%l5
L508:
L507:
	sethi %hi(_put_out_comments),%o0
	ld [%o0+%lo(_put_out_comments)],%o1
	cmp %o1,0
	bne L510
	nop
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	be L511
	nop
	add %l5,-1,%l5
	b L512
	nop
L511:
	mov 32,%o0
	stb %o0,[%l5-1]
L512:
	b L513
	nop
L510:
	mov 42,%o0
	stb %o0,[%l5]
	add %l5,1,%l5
L513:
	st %l3,[%fp-60]
L514:
	cmp %l3,%l4
	bgeu L515
	nop
	ldub [%l3],%o1
	and %o1,0xff,%o0
	add %l3,1,%l3
	cmp %o0,42
	be L519
	nop
	cmp %o0,42
	bg L528
	nop
	cmp %o0,10
	be L524
	nop
	b L527
	nop
L528:
	cmp %o0,47
	be L517
	nop
	b L527
	nop
L517:
	sethi %hi(_warn_comments),%o0
	ld [%o0+%lo(_warn_comments)],%o1
	cmp %o1,0
	be L518
	nop
	cmp %l3,%l4
	bgeu L518
	nop
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L518
	nop
	sethi %hi(LC101),%o1
	or %o1,%lo(LC101),%o0
	call _warning,0
	nop
L518:
	b L516
	nop
L519:
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L520
	nop
	ldub [%l3+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L520
	nop
	mov %l3,%o0
	call _newline_fix,0
	nop
L520:
	cmp %l3,%l4
	bgeu L522
	nop
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	be L522
	nop
	b L521
	nop
L522:
	b L523
	nop
L521:
	b L516
	nop
L524:
	ld [%fp-20],%o1
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	sethi %hi(_put_out_comments),%o0
	ld [%o0+%lo(_put_out_comments)],%o1
	cmp %o1,0
	bne L525
	nop
	mov 10,%o0
	stb %o0,[%l5]
	add %l5,1,%l5
L525:
	ld [%fp+68],%o1
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
L527:
L516:
	b L514
	nop
L515:
	nop
L523:
	cmp %l3,%l4
	blu L529
	nop
	ld [%fp-32],%o0
	call _line_for_error,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	sethi %hi(LC102),%o2
	or %o2,%lo(LC102),%o1
	call _error_with_line,0
	nop
	b L530
	nop
L529:
	add %l3,1,%l3
	sethi %hi(_put_out_comments),%o0
	ld [%o0+%lo(_put_out_comments)],%o1
	cmp %o1,0
	be L531
	nop
	ld [%fp-60],%o0
	sub %l3,%o0,%o2
	ld [%fp-60],%o0
	mov %l5,%o1
	call _bcopy,0
	nop
	ld [%fp-60],%o1
	sub %l3,%o1,%o0
	add %l5,%o0,%l5
L531:
L530:
	b L402
	nop
L532:
	sethi %hi(_dollars_in_ident),%o0
	ld [%o0+%lo(_dollars_in_ident)],%o1
	cmp %o1,0
	bne L533
	nop
	b L424
	nop
L533:
	b L534
	nop
L535:
L536:
L537:
L538:
L539:
L540:
L541:
L542:
L543:
L544:
	cmp %l1,0
	bne L545
	nop
L546:
	cmp %l3,%l4
	bgeu L547
	nop
L548:
	cmp %l3,%l4
	bgeu L549
	nop
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L549
	nop
	ldub [%l3+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L549
	nop
	ld [%fp-20],%o1
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	add %l3,2,%l3
	b L548
	nop
L549:
	ldub [%l3],%l0
	add %l3,1,%l3
	sethi %hi(___lc_ctype),%o1
	ld [%o1+%lo(___lc_ctype)],%o0
	ld [%o0+28],%o1
	cmp %o1,0
	bne L551
	nop
	sethi %hi(___lc_ctype),%o1
	ld [%o1+%lo(___lc_ctype)],%o0
	and %l0,0xff,%o1
	mov %o1,%o2
	sll %o2,2,%o1
	ld [%o0+64],%o2
	add %o1,%o2,%o0
	ld [%o0],%o1
	and %o1,2,%o0
	cmp %o0,0
	bne L550
	nop
	b L552
	nop
L551:
	sethi %hi(___lc_ctype),%o0
	ld [%o0+%lo(___lc_ctype)],%o3
	and %l0,0xff,%o0
	sethi %hi(___lc_ctype),%o2
	mov 2,%o1
	ld [%o2+%lo(___lc_ctype)],%o2
	ld [%o3+28],%o3
	call %o3,0
	nop
	cmp %o0,0
	bne L550
	nop
	b L552
	nop
L552:
	and %l0,0xff,%o0
	cmp %o0,46
	bne L553
	nop
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,46
	be L553
	nop
	b L550
	nop
L553:
	and %l0,0xff,%o0
	cmp %o0,95
	be L550
	nop
	add %l3,-1,%l3
	b L547
	nop
L550:
	stb %l0,[%l5]
	add %l5,1,%l5
	and %l0,0xff,%o0
	cmp %o0,101
	be L555
	nop
	and %l0,0xff,%o0
	cmp %o0,69
	be L555
	nop
	b L554
	nop
L555:
L556:
	cmp %l3,%l4
	bgeu L557
	nop
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L557
	nop
	ldub [%l3+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L557
	nop
	ld [%fp-20],%o1
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	add %l3,2,%l3
	b L556
	nop
L557:
	cmp %l3,%l4
	bgeu L558
	nop
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,43
	be L559
	nop
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,45
	be L559
	nop
	b L558
	nop
L559:
	ldub [%l3],%o0
	stb %o0,[%l5]
	add %l3,1,%l3
	add %l5,1,%l5
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	be L560
	nop
	b L547
	nop
L560:
L558:
L554:
	b L546
	nop
L547:
	b L402
	nop
L545:
L561:
L562:
L563:
L564:
L565:
L566:
L567:
L568:
L569:
L570:
L571:
L572:
L573:
L574:
L575:
L576:
L577:
L578:
L579:
L580:
L581:
L582:
L583:
L584:
L585:
L586:
L587:
L588:
L589:
L590:
L591:
L592:
L593:
L594:
L595:
L596:
L597:
L598:
L599:
L600:
L601:
L602:
L603:
L604:
L605:
L606:
L607:
L608:
L609:
L610:
L611:
L612:
L613:
	nop
L534:
	add %l1,1,%l1
	sll %l2,2,%l2
	and %l0,0xff,%o0
	add %l2,%o0,%l2
	b L402
	nop
L614:
	ld [%fp-20],%o0
	ld [%o0+28],%o1
	cmp %o1,0
	be L615
	nop
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,45
	bne L616
	nop
	ld [%fp-28],%o0
	cmp %o0,0
	bne L617
	nop
	mov 0,%l1
	mov 0,%l2
L617:
	add %l3,1,%l3
	ld [%fp+72],%o0
	cmp %o0,0
	bne L618
	nop
	add %l5,-1,%l5
	b L619
	nop
L618:
	mov 45,%o0
	stb %o0,[%l5]
	add %l5,1,%l5
L619:
	b L620
	nop
L616:
	ldub [%l3],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_space),%o2
	or %o2,%lo(_is_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L621
	nop
	cmp %l1,0
	ble L622
	nop
	b L407
	nop
L622:
	ld [%fp+72],%o0
	cmp %o0,0
	bne L623
	nop
	ldub [%l3],%o0
	stb %o0,[%l5-1]
	add %l3,1,%l3
	ldub [%l5-1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L624
	nop
	ld [%fp+68],%o1
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
L624:
	b L625
	nop
L623:
	ldub [%l3],%o0
	stb %o0,[%l5]
	add %l3,1,%l3
	add %l5,1,%l5
L625:
	b L626
	nop
L621:
	call _abort,0
	nop
L626:
L620:
	b L402
	nop
L615:
	cmp %l1,0
	ble L627
	nop
	b L407
	nop
L627:
	st %l3,[%fp-40]
	ld [%fp-20],%o1
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	ld [%fp+68],%o1
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	ld [%fp-20],%o0
	ld [%fp+68],%o1
	ld [%o0+12],%o0
	ld [%o1+12],%o1
	cmp %o0,%o1
	be L628
	nop
	ld [%fp+68],%o0
	st %l5,[%o0+24]
	ld [%fp-20],%o0
	ld [%fp+68],%o1
	mov 1,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%fp+68],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	ld [%fp-20],%o1
	ld [%fp-20],%o2
	ld [%fp-20],%o3
	ld [%o2+24],%o2
	ld [%o3+20],%o3
	sub %o2,%o3,%o2
	ld [%o1+16],%o3
	sub %o3,%o2,%o1
	cmp %o0,%o1
	bg L629
	nop
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%fp-20],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o0
	sub %o0,%o1,%o1
	ld [%fp+68],%o0
	call _grow_outbuf,0
	nop
	b L630
	nop
L629:
L630:
	ld [%fp+68],%o0
	ld [%o0+24],%l5
L628:
	b L402
	nop
L631:
	cmp %l3,%l4
	bgu L632
	nop
	b L424
	nop
L632:
	ld [%fp-20],%o0
	ld [%o0+28],%o1
	cmp %o1,0
	be L633
	nop
	add %l5,-1,%l5
	add %l3,-1,%l3
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	be L634
	nop
	cmp %l1,0
	be L634
	nop
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o1
	add %o1,-1,%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o0,%o1,%o2
	ld [%o2+24],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	bne L634
	nop
	mov 1,%o0
	st %o0,[%fp-24]
	b L424
	nop
L634:
	nop
L635:
	ld [%fp-20],%o0
	ld [%o0+28],%o1
	mov 31,%o0
	st %o0,[%o1+12]
	ld [%fp-20],%o0
	ld [%o0+36],%o1
	cmp %o1,0
	be L638
	nop
	ld [%fp-20],%o1
	ld [%o1+36],%o0
	call _free,0
	nop
L638:
	sethi %hi(_indepth),%o1
	sethi %hi(_indepth),%o0
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_indepth)]
L637:
	b L636
	nop
	b L635
	nop
L636:
	nop
L639:
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	ld [%o0+24],%l3
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o0+20],%o0
	ld [%o1+16],%o1
	add %o0,%o1,%l4
	ld [%fp+68],%o0
	st %l5,[%o0+24]
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%fp+68],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	sub %l4,%l3,%o1
	cmp %o0,%o1
	bg L642
	nop
	sub %l4,%l3,%o1
	ld [%fp+68],%o0
	call _grow_outbuf,0
	nop
	b L643
	nop
L642:
L643:
	st %g0,[%fp-40]
	ld [%fp+68],%o0
	ld [%o0+24],%l5
L641:
	b L640
	nop
	b L639
	nop
L640:
	b L402
	nop
L633:
	cmp %l1,0
	bne L644
	nop
	add %l5,-1,%l5
	add %l3,-1,%l3
	ld [%fp+68],%o0
	st %l5,[%o0+24]
	ld [%fp-20],%o0
	st %l3,[%o0+24]
	b L645
	nop
L644:
	nop
L407:
	add %l3,-1,%l3
	add %l5,-1,%l5
	mov 1,%o0
	st %o0,[%fp-24]
L646:
	nop
L424:
	cmp %l1,0
	ble L647
	nop
	sethi %hi(_pcp_outfile),%o0
	ld [%o0+%lo(_pcp_outfile)],%o1
	cmp %o1,0
	be L649
	nop
	sethi %hi(_pcp_inside_if),%o0
	ld [%o0+%lo(_pcp_inside_if)],%o1
	cmp %o1,0
	bne L649
	nop
	b L648
	nop
L649:
L650:
	nop
	sethi %hi(2147483647),%o1
	or %o1,%lo(2147483647),%o0
	and %l2,%o0,%o1
	mov %o1,%o0
	mov 1403,%o1
	call .rem,0
	nop
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_hashtab),%o2
	or %o2,%lo(_hashtab),%o1
	ld [%o0+%o1],%l6
L651:
	cmp %l6,0
	be L652
	nop
	ld [%l6+16],%o0
	cmp %o0,%l1
	bne L654
	nop
	mov %l1,%l7
	ld [%l6+20],%i0
	sub %l5,%l7,%i1
	ld [%fp-24],%o0
	cmp %o0,0
	bne L655
	nop
	add %i1,-1,%i1
L655:
	nop
L656:
	ldub [%i0],%o1
	and %o1,0xff,%o0
	ldub [%i1],%o2
	and %o2,0xff,%o1
	add %i1,1,%i1
	add %i0,1,%i0
	cmp %o0,%o1
	be L659
	nop
	b L660
	nop
L659:
L658:
	add %l7,-1,%l7
	cmp %l7,0
	be L657
	nop
	b L656
	nop
L657:
	ld [%fp-24],%o0
	cmp %o0,0
	bne L661
	nop
	add %l3,-1,%l3
	add %l5,-1,%l5
L661:
	ld [%fp+68],%o0
	ld [%o0+20],%o1
	sub %l5,%o1,%o0
	sub %o0,%l1,%o1
	st %o1,[%fp-64]
	ld [%fp+68],%o0
	ld [%o0+12],%o1
	st %o1,[%fp-68]
	ld [%l6+12],%o0
	cmp %o0,34
	bne L662
	nop
	mov %l6,%o0
	call _pcstring_used,0
	nop
	b L652
	nop
L662:
	ld [%l6+12],%o0
	xor %o0,32,%o1
	subcc %g0,%o1,%g0
	subx %g0,-1,%o0
	st %o0,[%fp-72]
	ld [%fp-72],%o0
	cmp %o0,0
	be L663
	nop
	ld [%fp+72],%o0
	cmp %o0,0
	be L664
	nop
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%fp+68],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	sub %l4,%l3,%o1
	add %o1,2,%o2
	cmp %o0,%o2
	bg L665
	nop
	sub %l4,%l3,%o0
	add %o0,2,%o1
	ld [%fp+68],%o0
	call _grow_outbuf,0
	nop
	b L666
	nop
L665:
L666:
	mov 10,%o0
	stb %o0,[%l5]
	add %l5,1,%l5
	mov 45,%o0
	stb %o0,[%l5]
	add %l5,1,%l5
L664:
	b L652
	nop
L663:
	ld [%l6+12],%o0
	cmp %o0,31
	be L668
	nop
	ld [%l6+12],%o0
	cmp %o0,32
	be L668
	nop
	b L667
	nop
L668:
	ld [%l6+24],%o0
	ld [%o0],%o1
	cmp %o1,0
	bl L667
	nop
	st %l3,[%fp-76]
	st %l5,[%fp-80]
	ld [%fp-20],%o0
	ld [%o0+12],%o1
	st %o1,[%fp-84]
	ld [%fp+68],%o0
	ld [%o0+12],%o1
	st %o1,[%fp-88]
L669:
	cmp %l3,%l4
	bne L671
	nop
	ld [%fp-20],%o0
	ld [%o0+28],%o1
	cmp %o1,0
	be L671
	nop
L672:
	ld [%fp-20],%o0
	ld [%o0+28],%o1
	mov 31,%o0
	st %o0,[%o1+12]
	ld [%fp-20],%o0
	ld [%o0+36],%o1
	cmp %o1,0
	be L675
	nop
	ld [%fp-20],%o1
	ld [%o1+36],%o0
	call _free,0
	nop
L675:
	sethi %hi(_indepth),%o1
	sethi %hi(_indepth),%o0
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_indepth)]
L674:
	b L673
	nop
	b L672
	nop
L673:
	nop
L676:
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	ld [%o0+24],%l3
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o0+20],%o0
	ld [%o1+16],%o1
	add %o0,%o1,%l4
	ld [%fp+68],%o0
	st %l5,[%o0+24]
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%fp+68],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	sub %l4,%l3,%o1
	cmp %o0,%o1
	bg L679
	nop
	sub %l4,%l3,%o1
	ld [%fp+68],%o0
	call _grow_outbuf,0
	nop
	b L680
	nop
L679:
L680:
	st %g0,[%fp-40]
	ld [%fp+68],%o0
	ld [%o0+24],%l5
L678:
	b L677
	nop
	b L676
	nop
L677:
	st %l3,[%fp-76]
	st %l5,[%fp-80]
	ld [%fp-20],%o0
	ld [%o0+12],%o1
	st %o1,[%fp-84]
	ld [%fp+68],%o0
	ld [%o0+12],%o1
	st %o1,[%fp-88]
	b L681
	nop
L671:
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L682
	nop
	add %l3,1,%o0
	cmp %o0,%l4
	be L682
	nop
	ldub [%l3+1],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L682
	nop
	sethi %hi(_put_out_comments),%o0
	ld [%o0+%lo(_put_out_comments)],%o1
	cmp %o1,0
	be L683
	nop
	mov 47,%o0
	stb %o0,[%l5]
	add %l5,1,%l5
	mov 42,%o0
	stb %o0,[%l5]
	add %l5,1,%l5
	b L684
	nop
L683:
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	bne L685
	nop
	mov 32,%o0
	stb %o0,[%l5]
	add %l5,1,%l5
L685:
L684:
	add %l3,2,%l3
L686:
	add %l3,1,%o0
	cmp %o0,%l4
	be L687
	nop
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L688
	nop
	ldub [%l3+1],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L688
	nop
	b L687
	nop
L688:
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L689
	nop
	ld [%fp-20],%o1
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	ld [%fp+68],%o1
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
L689:
	sethi %hi(_put_out_comments),%o0
	ld [%o0+%lo(_put_out_comments)],%o1
	cmp %o1,0
	be L690
	nop
	ldub [%l3],%o0
	stb %o0,[%l5]
	add %l3,1,%l3
	add %l5,1,%l5
	b L691
	nop
L690:
	add %l3,1,%l3
L691:
	b L686
	nop
L687:
	add %l3,2,%l3
	sethi %hi(_put_out_comments),%o0
	ld [%o0+%lo(_put_out_comments)],%o1
	cmp %o1,0
	be L692
	nop
	mov 42,%o0
	stb %o0,[%l5]
	add %l5,1,%l5
	mov 47,%o0
	stb %o0,[%l5]
	add %l5,1,%l5
L692:
	b L693
	nop
L682:
	ldub [%l3],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_space),%o2
	or %o2,%lo(_is_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L694
	nop
	ldub [%l3],%o0
	stb %o0,[%l5]
	add %l3,1,%l3
	add %l5,1,%l5
	ldub [%l3-1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L695
	nop
	ld [%fp-20],%o0
	ld [%o0+28],%o1
	cmp %o1,0
	bne L696
	nop
	ld [%fp-20],%o1
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	ld [%fp+68],%o1
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	b L697
	nop
L696:
	ld [%fp+72],%o0
	cmp %o0,0
	bne L698
	nop
	add %l5,-1,%l5
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,45
	bne L699
	nop
	add %l3,1,%l3
	b L700
	nop
L699:
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L701
	nop
	ld [%fp+68],%o1
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
L701:
	ldub [%l3],%o0
	stb %o0,[%l5]
	add %l3,1,%l3
	add %l5,1,%l5
L700:
	b L702
	nop
L698:
	ldub [%l3],%o0
	stb %o0,[%l5]
	add %l3,1,%l3
	add %l5,1,%l5
L702:
L697:
L695:
	b L703
	nop
L694:
	b L670
	nop
L703:
L693:
L681:
	b L669
	nop
L670:
	ldub [%l3],%o1
	and %o1,0xff,%o0
	cmp %o0,40
	be L704
	nop
	ld [%fp-76],%l3
	ld [%fp-80],%l5
	ld [%fp-20],%o0
	ld [%fp-84],%o1
	st %o1,[%o0+12]
	ld [%fp+68],%o0
	ld [%fp-88],%o1
	st %o1,[%o0+12]
	b L652
	nop
L704:
L667:
	ld [%fp+68],%o0
	ld [%o0+20],%o1
	ld [%fp-64],%o0
	add %o1,%o0,%l5
	ld [%fp+68],%o0
	ld [%fp-68],%o1
	st %o1,[%o0+12]
	ld [%fp-20],%o0
	st %l3,[%o0+24]
	ld [%fp+68],%o0
	st %l5,[%o0+24]
	mov %l6,%o0
	ld [%fp+68],%o1
	call _macroexpand,0
	nop
	ld [%fp+68],%o0
	ld [%o0+24],%l5
L705:
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	ld [%o0+24],%l3
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o0+20],%o0
	ld [%o1+16],%o1
	add %o0,%o1,%l4
	ld [%fp+68],%o0
	st %l5,[%o0+24]
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%fp+68],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	sub %l4,%l3,%o1
	cmp %o0,%o1
	bg L708
	nop
	sub %l4,%l3,%o1
	ld [%fp+68],%o0
	call _grow_outbuf,0
	nop
	b L709
	nop
L708:
L709:
	st %g0,[%fp-40]
	ld [%fp+68],%o0
	ld [%o0+24],%l5
L707:
	b L706
	nop
	b L705
	nop
L706:
	b L652
	nop
L654:
	nop
L660:
L653:
	ld [%l6],%l6
	b L651
	nop
L652:
L648:
	mov 0,%l2
	mov 0,%l1
	st %g0,[%fp-24]
	st %g0,[%fp-28]
L647:
L402:
	b L400
	nop
L401:
	nop
L645:
	sethi %hi(_if_stack),%o0
	ld [%fp-20],%o1
	ld [%o0+%lo(_if_stack)],%o0
	ld [%o1+32],%o1
	cmp %o0,%o1
	be L711
	nop
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	ld [%o0+20],%o1
	add %o1,-5,%o0
	mov 5,%o1
	cmp %o1,%o0
	blu L719
	nop
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(L718),%o2
	or %o2,%lo(L718),%o1
	ld [%o0+%o1],%o0
	jmp %o0
	nop
L718:
	.word	L714
	.word	L715
	.word	L713
	.word	L716
	.word	L719
	.word	L717
L713:
	sethi %hi(LC6),%o1
	or %o1,%lo(LC6),%o0
	st %o0,[%fp-88]
	b L712
	nop
L714:
	sethi %hi(LC7),%o1
	or %o1,%lo(LC7),%o0
	st %o0,[%fp-88]
	b L712
	nop
L715:
	sethi %hi(LC8),%o1
	or %o1,%lo(LC8),%o0
	st %o0,[%fp-88]
	b L712
	nop
L716:
	sethi %hi(LC10),%o1
	or %o1,%lo(LC10),%o0
	st %o0,[%fp-88]
	b L712
	nop
L717:
	sethi %hi(LC11),%o1
	or %o1,%lo(LC11),%o0
	st %o0,[%fp-88]
	b L712
	nop
L719:
L712:
	sethi %hi(_if_stack),%o0
	ld [%o0+%lo(_if_stack)],%o1
	ld [%o1+8],%o0
	call _line_for_error,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	sethi %hi(LC103),%o2
	or %o2,%lo(LC103),%o1
	ld [%fp-88],%o2
	call _error_with_line,0
	nop
L711:
	sethi %hi(_if_stack),%o0
	ld [%fp-20],%o1
	ld [%o1+32],%o2
	st %o2,[%o0+%lo(_if_stack)]
L392:
	ret
	restore
	.align 8
LC104:
	.ascii "macro or `#include' recursion too deep\0"
	.align 4
	.proc	010
_expand_to_temp_buffer:
	!#PROLOGUE# 0
	save %sp,-176,%sp
	!#PROLOGUE# 1
	ld [%fp+64],%l0
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	ld [%fp+72],%o0
	ld [%fp+68],%o1
	sub %o0,%o1,%o0
	st %o0,[%fp-68]
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o1
	st %o1,[%fp-76]
	sethi %hi(_assertions_flag),%o0
	ld [%o0+%lo(_assertions_flag)],%o1
	st %o1,[%fp-80]
	sethi %hi(_assertions_flag),%o0
	ld [%fp+80],%o1
	st %o1,[%o0+%lo(_assertions_flag)]
	ld [%fp-68],%o0
	cmp %o0,0
	bge L721
	nop
	call _abort,0
	nop
L721:
	ld [%fp-68],%o0
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
	st %o1,[%fp-72]
	ld [%fp+68],%o0
	ld [%fp-72],%o1
L722:
	ld [%fp+72],%o2
	cmp %o0,%o2
	be L723
	nop
	ldub [%o0],%o2
	stb %o2,[%o1]
	add %o0,1,%o0
	add %o1,1,%o1
	b L722
	nop
L723:
	ld [%fp-72],%o0
	ld [%fp-68],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
	ld [%fp-68],%o0
	mov %o0,%o1
	sll %o1,1,%o0
	add %o0,100,%o1
	st %o1,[%fp-48]
	ld [%fp-48],%o0
	call _xmalloc,0
	nop
	mov %o0,%o1
	st %o1,[%fp-44]
	st %o1,[%fp-40]
	st %g0,[%fp-64]
	st %g0,[%fp-36]
	st %g0,[%fp-28]
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o1
	cmp %o1,198
	ble L724
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o0,%o1,%o2
	ld [%o2+12],%o0
	call _line_for_error,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	sethi %hi(LC104),%o2
	or %o2,%lo(LC104),%o1
	call _error_with_line,0
	nop
	ld [%fp-64],%o0
	st %o0,[%l0]
	ld [%fp-60],%o0
	st %o0,[%l0+4]
	ld [%fp-56],%o0
	st %o0,[%l0+8]
	ld [%fp-52],%o0
	st %o0,[%l0+12]
	ld [%fp-48],%o0
	st %o0,[%l0+16]
	ld [%fp-44],%o0
	st %o0,[%l0+20]
	ld [%fp-40],%o0
	st %o0,[%l0+24]
	ld [%fp-36],%o0
	st %o0,[%l0+28]
	ld [%fp-32],%o0
	st %o0,[%l0+32]
	ld [%fp-28],%o0
	st %o0,[%l0+36]
	ld [%fp-24],%o0
	st %o0,[%l0+40]
	b L720
	nop
L724:
	sethi %hi(_indepth),%o1
	sethi %hi(_indepth),%o0
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_indepth)]
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o0,%o1,%l1
	st %g0,[%l1]
	st %g0,[%l1+4]
	stb %g0,[%l1+40]
	st %g0,[%l1+28]
	st %g0,[%l1+36]
	ld [%fp-68],%o0
	st %o0,[%l1+16]
	ld [%fp-72],%o0
	st %o0,[%l1+24]
	st %o0,[%l1+20]
	sethi %hi(_if_stack),%o0
	ld [%o0+%lo(_if_stack)],%o1
	st %o1,[%l1+32]
	mov 1,%o0
	st %o0,[%fp-52]
	mov 1,%o0
	st %o0,[%l1+12]
	add %fp,-64,%o1
	mov %o1,%o0
	ld [%fp+76],%o1
	call _rescan,0
	nop
	sethi %hi(_indepth),%o1
	sethi %hi(_indepth),%o0
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_indepth)]
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o1
	ld [%fp-76],%o0
	cmp %o1,%o0
	be L725
	nop
	call _abort,0
	nop
L725:
	ld [%fp-40],%o0
	ld [%fp-44],%o1
	sub %o0,%o1,%o0
	st %o0,[%fp-48]
	sethi %hi(_assertions_flag),%o0
	ld [%fp-80],%o1
	st %o1,[%o0+%lo(_assertions_flag)]
	ld [%fp-64],%o0
	st %o0,[%l0]
	ld [%fp-60],%o0
	st %o0,[%l0+4]
	ld [%fp-56],%o0
	st %o0,[%l0+8]
	ld [%fp-52],%o0
	st %o0,[%l0+12]
	ld [%fp-48],%o0
	st %o0,[%l0+16]
	ld [%fp-44],%o0
	st %o0,[%l0+20]
	ld [%fp-40],%o0
	st %o0,[%l0+24]
	ld [%fp-36],%o0
	st %o0,[%l0+28]
	ld [%fp-32],%o0
	st %o0,[%l0+32]
	ld [%fp-28],%o0
	st %o0,[%l0+36]
	ld [%fp-24],%o0
	st %o0,[%l0+40]
	b L720
	nop
L720:
	mov %l0,%i0
	jmp %i7+12
	restore
	.align 8
LC105:
	.ascii "%s in preprocessing directive\0"
	.align 8
LC106:
	.ascii "formfeed\0"
	.align 8
LC107:
	.ascii "vertical tab\0"
.data
	.align 4
_line_directive_table.18:
	.word	4
	.word	_do_line
	.word	LC12
	.word	12
	.skip 4
.text
	.align 8
LC108:
	.ascii "`#' followed by integer\0"
	.align 8
LC109:
	.ascii "invalid preprocessor directive\0"
	.align 8
LC110:
	.ascii "invalid preprocessor directive name\0"
	.align 8
LC111:
	.ascii "__NeXT__\0"
	.align 4
	.proc	04
_handle_directive:
	!#PROLOGUE# 0
	save %sp,-160,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %g0,[%fp-24]
	ld [%fp+68],%o0
	ld [%o0+24],%l0
	sethi %hi(_directive_start),%o0
	add %l0,-1,%o1
	st %o1,[%o0+%lo(_directive_start)]
L727:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L729
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,12
	be L731
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,11
	be L731
	nop
	b L730
	nop
L731:
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L730
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,12
	bne L732
	nop
	sethi %hi(LC106),%o0
	or %o0,%lo(LC106),%o1
	b L733
	nop
L732:
	sethi %hi(LC107),%o0
	or %o0,%lo(LC107),%o1
L733:
	sethi %hi(LC105),%o2
	or %o2,%lo(LC105),%o0
	call _pedwarn,0
	nop
L730:
	add %l0,1,%l0
	b L734
	nop
L729:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L735
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L735
	nop
	ld [%fp+68],%o0
	st %l0,[%o0+24]
	ld [%fp+68],%o0
	add %o0,12,%o1
	ld [%fp+68],%o0
	mov 0,%o2
	call _skip_to_end_of_comment,0
	nop
	ld [%fp+68],%o0
	ld [%o0+24],%l0
	b L736
	nop
L735:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L737
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L737
	nop
	add %l0,2,%l0
	ld [%fp+68],%o1
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	b L738
	nop
L737:
	b L728
	nop
L738:
L736:
L734:
	b L727
	nop
L728:
	mov %l0,%l1
L739:
	ldub [%l1],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L741
	nop
	add %l1,1,%l1
	b L742
	nop
L741:
	ldub [%l1],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L743
	nop
	ldub [%l1+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L743
	nop
	mov %l1,%o0
	call _name_newline_fix,0
	nop
L743:
	ldub [%l1],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L744
	nop
	add %l1,1,%l1
	b L745
	nop
L744:
	b L740
	nop
L745:
L742:
	b L739
	nop
L740:
	sub %l1,%l0,%l3
	st %l0,[%fp-28]
	st %l1,[%fp-32]
	cmp %l3,0
	bne L746
	nop
	ld [%fp-32],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L746
	nop
	ld [%fp+68],%o0
	ld [%fp-32],%o1
	st %o1,[%o0+24]
	mov 1,%i0
	b L726
	nop
L746:
	cmp %l3,0
	be L748
	nop
	ld [%fp-28],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idstart),%o2
	or %o2,%lo(_is_idstart),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	bne L747
	nop
	b L748
	nop
L748:
	ld [%fp-28],%o0
	st %o0,[%fp-36]
L749:
	ld [%fp-36],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L750
	nop
	ld [%fp-36],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bleu L752
	nop
	ld [%fp-36],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,57
	bgu L752
	nop
	b L751
	nop
L752:
	b L750
	nop
L751:
	ld [%fp-36],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-36]
	b L749
	nop
L750:
	ld [%fp-36],%o0
	ld [%fp-28],%o1
	cmp %o0,%o1
	be L753
	nop
	ld [%fp-36],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	bne L753
	nop
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L754
	nop
	sethi %hi(LC108),%o1
	or %o1,%lo(LC108),%o0
	call _pedwarn,0
	nop
L754:
	ld [%fp-28],%o0
	st %o0,[%fp-32]
	sethi %hi(_line_directive_table.18),%o0
	or %o0,%lo(_line_directive_table.18),%l2
	b L755
	nop
L753:
	ld [%fp-36],%o0
	ld [%fp-28],%o1
	cmp %o0,%o1
	bne L756
	nop
L757:
	ld [%fp-36],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,35
	be L759
	nop
	ld [%fp-36],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	bne L759
	nop
	b L758
	nop
L759:
	ld [%fp-36],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-36]
	b L757
	nop
L758:
	ld [%fp-36],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L760
	nop
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L761
	nop
	sethi %hi(_lang_asm),%o0
	ld [%o0+%lo(_lang_asm)],%o1
	cmp %o1,0
	bne L761
	nop
	sethi %hi(LC109),%o1
	or %o1,%lo(LC109),%o0
	call _warning,0
	nop
L761:
	mov 0,%i0
	b L726
	nop
L760:
L756:
	sethi %hi(_lang_asm),%o0
	ld [%o0+%lo(_lang_asm)],%o1
	cmp %o1,0
	bne L762
	nop
	sethi %hi(LC110),%o1
	or %o1,%lo(LC110),%o0
	call _error,0
	nop
L762:
	mov 0,%i0
	b L726
	nop
L747:
	nop
	sethi %hi(_directive_table),%o0
	or %o0,%lo(_directive_table),%l2
L763:
	ld [%l2],%o0
	cmp %o0,0
	ble L764
	nop
	ld [%l2],%o0
	cmp %o0,%l3
	bne L766
	nop
	ld [%l2+8],%o0
	ld [%fp-28],%o1
	mov %l3,%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L766
	nop
	st %g0,[%fp-44]
L755:
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o0+20],%o0
	ld [%o1+16],%o1
	add %o0,%o1,%l5
	st %g0,[%fp-36]
	mov 0,%o0
	sethi %hi(_traditional),%o1
	ld [%o1+%lo(_traditional)],%o2
	cmp %o2,0
	be L767
	nop
	ldub [%l2+17],%o1
	sll %o1,24,%o2
	sra %o2,24,%o1
	cmp %o1,0
	be L767
	nop
	mov 1,%o0
L767:
	st %o0,[%fp-48]
	ld [%l2+12],%o0
	cmp %o0,4
	bne L768
	nop
	sethi %hi(_objc),%o0
	ld [%o0+%lo(_objc)],%o1
	cmp %o1,0
	bne L768
	nop
	sethi %hi(LC111),%o1
	or %o1,%lo(LC111),%o0
	mov -1,%o1
	mov -1,%o2
	call _lookup,0
	nop
	cmp %o0,0
	bne L768
	nop
	b L764
	nop
L768:
	ld [%fp-32],%l0
	mov %l0,%o0
	mov %o0,%l4
L769:
	cmp %l0,%l5
	bgeu L770
	nop
	ldub [%l0],%l6
	add %l0,1,%l0
	and %l6,0xff,%o0
	cmp %o0,47
	be L784
	nop
	cmp %o0,47
	bg L799
	nop
	cmp %o0,34
	be L776
	nop
	cmp %o0,34
	bg L800
	nop
	cmp %o0,10
	be L796
	nop
	cmp %o0,10
	bl L798
	nop
	cmp %o0,12
	bg L798
	nop
	b L792
	nop
L800:
	cmp %o0,39
	be L775
	nop
	b L798
	nop
L799:
	cmp %o0,60
	be L780
	nop
	cmp %o0,92
	be L772
	nop
	b L798
	nop
L772:
	cmp %l0,%l5
	bgeu L773
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L774
	nop
	ld [%fp+68],%o1
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	mov 1,%o0
	st %o0,[%fp-24]
L774:
	add %l0,1,%l0
L773:
	b L771
	nop
L775:
L776:
	add %l0,-1,%o0
	ld [%fp+68],%o2
	ld [%fp+68],%o1
	add %o1,12,%o3
	add %fp,-24,%o4
	add %fp,-36,%o5
	mov %l5,%o1
	ld [%o2+12],%o2
	call _skip_quoted_string,0
	nop
	mov %o0,%l0
	ld [%fp-36],%o0
	cmp %o0,0
	be L777
	nop
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	be L778
	nop
	ld [%fp+68],%o0
	st %l0,[%o0+24]
	b L779
	nop
L778:
	ld [%fp+68],%o0
	st %l0,[%o0+24]
	mov 1,%i0
	b L726
	nop
L777:
	b L771
	nop
L780:
	ldub [%l2+16],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	bne L781
	nop
	b L771
	nop
L781:
	nop
L782:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L783
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,62
	be L783
	nop
	add %l0,1,%l0
	b L782
	nop
L783:
	b L771
	nop
L784:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L785
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L785
	nop
	mov %l0,%o0
	call _newline_fix,0
	nop
L785:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	be L787
	nop
	sethi %hi(_cplusplus_comments),%o0
	ld [%o0+%lo(_cplusplus_comments)],%o1
	cmp %o1,0
	be L786
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	be L787
	nop
	b L786
	nop
L787:
	add %l0,-1,%o0
	st %o0,[%fp-52]
	ld [%fp+68],%o0
	add %l0,1,%o1
	st %o1,[%o0+24]
	ld [%fp+68],%o0
	add %o0,12,%o1
	ld [%fp+68],%o0
	mov 0,%o2
	call _skip_to_end_of_comment,0
	nop
	ld [%fp+68],%o0
	ld [%o0+24],%l0
	cmp %l0,%l5
	be L789
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	be L789
	nop
	b L788
	nop
L789:
	ld [%fp-52],%l0
	b L779
	nop
L788:
	ld [%fp-48],%o0
	cmp %o0,0
	bne L790
	nop
	ld [%fp-24],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
L790:
L786:
	b L771
	nop
L791:
L792:
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L793
	nop
	and %l6,0xff,%o0
	cmp %o0,12
	bne L794
	nop
	sethi %hi(LC106),%o0
	or %o0,%lo(LC106),%o1
	b L795
	nop
L794:
	sethi %hi(LC107),%o0
	or %o0,%lo(LC107),%o1
L795:
	sethi %hi(LC105),%o2
	or %o2,%lo(LC105),%o0
	call _pedwarn,0
	nop
L793:
	b L771
	nop
L796:
	add %l0,-1,%l0
	ld [%fp+68],%o0
	st %l0,[%o0+24]
	b L779
	nop
L798:
L771:
	b L769
	nop
L770:
	ld [%fp+68],%o0
	st %l0,[%o0+24]
L779:
	ld [%fp+68],%o0
	ld [%o0+24],%o1
	st %o1,[%fp-20]
	sethi %hi(_no_output),%o0
	ld [%o0+%lo(_no_output)],%o1
	cmp %o1,0
	bne L801
	nop
	ldub [%l2+18],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	be L801
	nop
	sethi %hi(_put_out_comments),%o0
	ld [%o0+%lo(_put_out_comments)],%o1
	cmp %o1,0
	be L801
	nop
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%fp+72],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	ld [%l2],%o2
	add %o2,2,%o1
	cmp %o0,%o1
	bg L802
	nop
	ld [%l2],%o0
	add %o0,2,%o1
	ld [%fp+72],%o0
	call _grow_outbuf,0
	nop
	b L803
	nop
L802:
L803:
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%o0+24],%o0
	ld [%o1+20],%o1
	cmp %o0,%o1
	bleu L804
	nop
	ld [%fp+72],%o0
	mov -1,%o1
	ld [%o0+24],%o2
	add %o1,%o2,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	be L804
	nop
	ld [%fp+72],%o1
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	ld [%fp+72],%o1
	ld [%o1+24],%o0
	add %o0,1,%o2
	st %o2,[%o1+24]
	mov 10,%o1
	stb %o1,[%o0]
L804:
	ld [%fp+72],%o1
	ld [%o1+24],%o0
	add %o0,1,%o2
	st %o2,[%o1+24]
	mov 35,%o1
	stb %o1,[%o0]
	ld [%fp+72],%o1
	ld [%l2+8],%o0
	ld [%o1+24],%o1
	ld [%l2],%o2
	call _bcopy,0
	nop
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%o1+24],%o2
	ld [%l2],%o1
	add %o2,%o1,%o2
	st %o2,[%o0+24]
	sub %l0,%l4,%o0
	st %o0,[%fp-52]
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%fp+72],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	ld [%fp-52],%o1
	cmp %o0,%o1
	bg L805
	nop
	ld [%fp+72],%o0
	ld [%fp-52],%o1
	call _grow_outbuf,0
	nop
	b L806
	nop
L805:
L806:
	ld [%fp+72],%o1
	mov %l4,%o0
	ld [%o1+24],%o1
	ld [%fp-52],%o2
	call _bcopy,0
	nop
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%o1+24],%o2
	ld [%fp-52],%o1
	add %o2,%o1,%o2
	st %o2,[%o0+24]
L807:
	ld [%fp-52],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-52]
	cmp %o1,0
	bl L808
	nop
	ld [%fp-52],%o1
	add %l4,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L809
	nop
	ld [%fp+72],%o1
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
L809:
	b L807
	nop
L808:
	add %fp,-40,%o0
	st %o0,[%fp-44]
L801:
	ld [%fp-24],%o0
	cmp %o0,0
	be L810
	nop
	mov %l4,%l6
	sub %l0,%l4,%o0
	add %o0,5,%o1
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
	add %sp,92,%l1
	mov %l1,%o0
	add %o0,7,%o0
	srl %o0,3,%o1
	mov %o1,%o0
	sll %o0,3,%o1
	mov %o1,%l1
	mov %l1,%l4
L811:
	cmp %l6,%l0
	bgeu L812
	nop
	ldub [%l6],%l7
	add %l6,1,%l6
	stb %l7,[%l1]
	add %l1,1,%l1
	and %l7,0xff,%o0
	cmp %o0,47
	be L847
	nop
	cmp %o0,47
	bg L858
	nop
	cmp %o0,34
	be L840
	nop
	cmp %o0,34
	bg L859
	nop
	cmp %o0,10
	be L814
	nop
	b L857
	nop
L859:
	cmp %o0,39
	be L839
	nop
	b L857
	nop
L858:
	cmp %o0,60
	be L815
	nop
	cmp %o0,92
	be L821
	nop
	b L857
	nop
L814:
	call _abort,0
	nop
	b L813
	nop
L815:
	ldub [%l2+16],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	bne L816
	nop
	b L813
	nop
L816:
	nop
L817:
	cmp %l6,%l0
	bgeu L818
	nop
	and %l7,0xff,%o0
	cmp %o0,62
	be L818
	nop
	ldub [%l6],%l7
	add %l6,1,%l6
	and %l7,0xff,%o0
	cmp %o0,92
	bne L819
	nop
	cmp %l6,%l0
	bgeu L819
	nop
	ldub [%l6],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L819
	nop
	add %l6,1,%l6
	b L820
	nop
L819:
	stb %l7,[%l1]
	add %l1,1,%l1
L820:
	b L817
	nop
L818:
	b L813
	nop
L821:
	ldub [%l6],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L822
	nop
	add %l6,1,%l6
	add %l1,-1,%l1
	cmp %l1,%l4
	be L823
	nop
	ldub [%l1-1],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_space),%o2
	or %o2,%lo(_is_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L823
	nop
L824:
	cmp %l1,%l4
	be L825
	nop
	ldub [%l1-1],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_space),%o2
	or %o2,%lo(_is_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L825
	nop
	add %l1,-1,%l1
	b L824
	nop
L825:
	add %l1,1,%l1
L826:
L829:
	ldub [%l6],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L830
	nop
	add %l6,1,%l6
	b L829
	nop
L830:
L828:
	b L827
	nop
	b L826
	nop
L827:
	b L831
	nop
L823:
	ldub [%l6],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_space),%o2
	or %o2,%lo(_is_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L832
	nop
	ldub [%l6],%o0
	stb %o0,[%l1]
	add %l6,1,%l6
	add %l1,1,%l1
L833:
L836:
	ldub [%l6],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L837
	nop
	add %l6,1,%l6
	b L836
	nop
L837:
L835:
	b L834
	nop
	b L833
	nop
L834:
L832:
L831:
	b L838
	nop
L822:
	ldub [%l6],%o0
	stb %o0,[%l1]
	add %l6,1,%l6
	add %l1,1,%l1
L838:
	b L813
	nop
L839:
L840:
	add %l6,-1,%o0
	ld [%fp+68],%o2
	mov %l0,%o1
	ld [%o2+12],%o2
	mov 0,%o3
	mov 0,%o4
	mov 0,%o5
	call _skip_quoted_string,0
	nop
	mov %o0,%i0
L841:
	cmp %l6,%i0
	be L842
	nop
	ldub [%l6],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L843
	nop
	add %l6,1,%l6
	ldub [%l6],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	be L844
	nop
	mov 92,%o0
	stb %o0,[%l1]
	add %l1,1,%l1
	b L845
	nop
L844:
	add %l6,1,%l6
L845:
	b L846
	nop
L843:
	ldub [%l6],%o0
	stb %o0,[%l1]
	add %l6,1,%l6
	add %l1,1,%l1
L846:
	b L841
	nop
L842:
	b L813
	nop
L847:
	ldub [%l6],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	be L849
	nop
	sethi %hi(_cplusplus_comments),%o0
	ld [%o0+%lo(_cplusplus_comments)],%o1
	cmp %o1,0
	be L848
	nop
	ldub [%l6],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	be L849
	nop
	b L848
	nop
L849:
	ld [%fp+68],%o0
	add %l6,1,%o1
	st %o1,[%o0+24]
	ld [%fp+68],%o0
	ld [%fp-44],%o1
	mov 0,%o2
	call _skip_to_end_of_comment,0
	nop
	ld [%fp-48],%o0
	cmp %o0,0
	be L850
	nop
L851:
	ld [%fp+68],%o0
	ld [%o0+24],%o1
	cmp %l6,%o1
	be L852
	nop
	ldub [%l6],%o0
	stb %o0,[%l1]
	add %l6,1,%l6
	add %l1,1,%l1
	b L851
	nop
L852:
	b L853
	nop
L850:
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	be L854
	nop
	add %l1,-1,%l1
	b L855
	nop
L854:
	mov 32,%o0
	stb %o0,[%l1-1]
L855:
L853:
	ld [%fp+68],%o0
	ld [%o0+24],%l6
L848:
L857:
L813:
	b L811
	nop
L812:
	stb %g0,[%l1]
	b L860
	nop
L810:
	mov %l0,%l1
L860:
	ld [%fp+68],%o0
	ld [%fp-20],%o1
	st %o1,[%o0+24]
	sethi %hi(_no_output),%o0
	ld [%o0+%lo(_no_output)],%o1
	cmp %o1,0
	bne L861
	nop
	ld [%fp-44],%o0
	cmp %o0,0
	bne L861
	nop
	ldub [%l2+18],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	bne L862
	nop
	ld [%l2+12],%o0
	cmp %o0,1
	bne L861
	nop
	sethi %hi(_dump_macros),%o0
	ld [%o0+%lo(_dump_macros)],%o1
	cmp %o1,2
	be L862
	nop
	sethi %hi(_dump_macros),%o0
	ld [%o0+%lo(_dump_macros)],%o1
	cmp %o1,3
	be L862
	nop
	b L861
	nop
L862:
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%fp+72],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	ld [%l2],%o2
	add %o2,1,%o1
	cmp %o0,%o1
	bg L863
	nop
	ld [%l2],%o0
	add %o0,1,%o1
	ld [%fp+72],%o0
	call _grow_outbuf,0
	nop
	b L864
	nop
L863:
L864:
	ld [%fp+72],%o1
	ld [%o1+24],%o0
	add %o0,1,%o2
	st %o2,[%o1+24]
	mov 35,%o1
	stb %o1,[%o0]
	ld [%fp+72],%o1
	ld [%l2+8],%o0
	ld [%o1+24],%o1
	ld [%l2],%o2
	call _bcopy,0
	nop
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%o1+24],%o2
	ld [%l2],%o1
	add %o2,%o1,%o2
	st %o2,[%o0+24]
	ldub [%l2+18],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	bne L866
	nop
	sethi %hi(_dump_macros),%o0
	ld [%o0+%lo(_dump_macros)],%o1
	cmp %o1,3
	be L866
	nop
	b L865
	nop
L866:
	sub %l1,%l4,%o0
	st %o0,[%fp-56]
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%fp+72],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	ld [%fp-56],%o1
	cmp %o0,%o1
	bg L867
	nop
	ld [%fp+72],%o0
	ld [%fp-56],%o1
	call _grow_outbuf,0
	nop
	b L868
	nop
L867:
L868:
	ld [%fp+72],%o1
	mov %l4,%o0
	ld [%o1+24],%o1
	ld [%fp-56],%o2
	call _bcopy,0
	nop
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%o1+24],%o2
	ld [%fp-56],%o1
	add %o2,%o1,%o2
	st %o2,[%o0+24]
	b L869
	nop
L865:
	ld [%l2+12],%o0
	cmp %o0,1
	bne L870
	nop
	sethi %hi(_dump_macros),%o0
	ld [%o0+%lo(_dump_macros)],%o1
	cmp %o1,2
	bne L870
	nop
	st %l4,[%fp-60]
L871:
L874:
	ld [%fp-60],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L875
	nop
	ld [%fp-60],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-60]
	b L874
	nop
L875:
L873:
	b L872
	nop
	b L871
	nop
L872:
	ld [%fp-60],%o0
	st %o0,[%fp-64]
L876:
	ld [%fp-60],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L877
	nop
	ld [%fp-60],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-60]
	b L876
	nop
L877:
	ld [%fp-60],%o0
	ld [%fp-64],%o1
	sub %o0,%o1,%o0
	st %o0,[%fp-56]
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%fp+72],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	ld [%fp-56],%o2
	add %o2,1,%o1
	cmp %o0,%o1
	bg L878
	nop
	ld [%fp-56],%o0
	add %o0,1,%o1
	ld [%fp+72],%o0
	call _grow_outbuf,0
	nop
	b L879
	nop
L878:
L879:
	ld [%fp+72],%o1
	ld [%o1+24],%o0
	add %o0,1,%o2
	st %o2,[%o1+24]
	mov 32,%o1
	stb %o1,[%o0]
	ld [%fp+72],%o1
	ld [%fp-64],%o0
	ld [%o1+24],%o1
	ld [%fp-56],%o2
	call _bcopy,0
	nop
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%o1+24],%o2
	ld [%fp-56],%o1
	add %o2,%o1,%o2
	st %o2,[%o0+24]
L870:
L869:
L861:
	mov %l4,%o0
	mov %l1,%o1
	ld [%fp+72],%o2
	mov %l2,%o3
	ld [%l2+4],%o4
	call %o4,0
	nop
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%fp+72],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	ld [%fp+68],%o1
	ld [%fp+68],%o2
	ld [%fp+68],%o3
	ld [%o2+24],%o2
	ld [%o3+20],%o3
	sub %o2,%o3,%o2
	ld [%o1+16],%o3
	sub %o3,%o2,%o1
	cmp %o0,%o1
	bg L880
	nop
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%fp+68],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o0
	sub %o0,%o1,%o1
	ld [%fp+72],%o0
	call _grow_outbuf,0
	nop
	b L881
	nop
L880:
L881:
	mov 1,%i0
	b L726
	nop
L766:
L765:
	add %l2,20,%l2
	b L763
	nop
L764:
	mov 0,%i0
	b L726
	nop
L726:
	ret
	restore

	.reserve _timebuf.21,8,"bss"
	.align 4
	.proc	0110
_timestamp:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	sethi %hi(_timebuf.21),%o0
	ld [%o0+%lo(_timebuf.21)],%o1
	cmp %o1,0
	bne L883
	nop
	mov 0,%o0
	call _time,0
	nop
	st %o0,[%fp-20]
	add %fp,-20,%o1
	mov %o1,%o0
	call _localtime,0
	nop
	sethi %hi(_timebuf.21),%o1
	st %o0,[%o1+%lo(_timebuf.21)]
L883:
	sethi %hi(_timebuf.21),%o0
	ld [%o0+%lo(_timebuf.21)],%i0
	b L882
	nop
L882:
	ret
	restore
.data
	.align 4
_monthnames:
	.word	LC112
	.word	LC113
	.word	LC114
	.word	LC115
	.word	LC116
	.word	LC117
	.word	LC118
	.word	LC119
	.word	LC120
	.word	LC121
	.word	LC122
	.word	LC123
.text
	.align 8
LC123:
	.ascii "Dec\0"
	.align 8
LC122:
	.ascii "Nov\0"
	.align 8
LC121:
	.ascii "Oct\0"
	.align 8
LC120:
	.ascii "Sep\0"
	.align 8
LC119:
	.ascii "Aug\0"
	.align 8
LC118:
	.ascii "Jul\0"
	.align 8
LC117:
	.ascii "Jun\0"
	.align 8
LC116:
	.ascii "May\0"
	.align 8
LC115:
	.ascii "Apr\0"
	.align 8
LC114:
	.ascii "Mar\0"
	.align 8
LC113:
	.ascii "Feb\0"
	.align 8
LC112:
	.ascii "Jan\0"
	.align 8
LC124:
	.ascii "Predefined macro `%s' used inside `#if' during precompilation\0"
	.align 8
LC125:
	.ascii "cccp error: not in any file?!\0"
	.align 8
LC126:
	.ascii "\"%s\"\0"
	.align 8
LC127:
	.ascii "\"\"\0"
	.align 8
LC128:
	.ascii "%d\0"
	.align 8
LC129:
	.ascii "long unsigned int\0"
	.align 8
LC130:
	.ascii "%s\0"
	.align 8
LC131:
	.ascii "long int\0"
	.align 8
LC132:
	.ascii "short unsigned int\0"
	.align 8
LC133:
	.ascii "#define %s %d\12\0"
	.align 8
LC134:
	.ascii "\"%s %2d %4d\"\0"
	.align 8
LC135:
	.ascii "\"%02d:%02d:%02d\"\0"
	.align 8
LC136:
	.ascii " 0 \0"
	.align 8
LC137:
	.ascii "#define %s\12\0"
	.align 8
LC138:
	.ascii " 1 \0"
	.align 8
LC139:
	.ascii "#undef \0"
	.align 8
LC140:
	.ascii "`defined' without an identifier\0"
	.align 8
LC141:
	.ascii "cccp error: invalid special hash type\0"
	.align 4
	.proc	020
_special_symbol:
	!#PROLOGUE# 0
	save %sp,-152,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %g0,[%fp-36]
	st %g0,[%fp-44]
	sethi %hi(_pcp_outfile),%o0
	ld [%o0+%lo(_pcp_outfile)],%o1
	cmp %o1,0
	be L885
	nop
	sethi %hi(_pcp_inside_if),%o0
	ld [%o0+%lo(_pcp_inside_if)],%o1
	cmp %o1,0
	be L885
	nop
	ld [%fp+68],%o0
	ld [%o0+12],%o1
	cmp %o1,33
	be L885
	nop
	ld [%fp+68],%o0
	ld [%o0+12],%o1
	cmp %o1,30
	be L885
	nop
	ld [%fp+68],%o1
	sethi %hi(LC124),%o2
	or %o2,%lo(LC124),%o0
	ld [%o1+20],%o1
	call _error,0
	nop
L885:
	nop
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o1
	st %o1,[%fp-24]
L886:
	ld [%fp-24],%o0
	cmp %o0,0
	bl L887
	nop
	ld [%fp-24],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	ld [%o0+%o1],%o0
	cmp %o0,0
	be L889
	nop
	ld [%fp-24],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-36]
	b L887
	nop
L889:
L888:
	ld [%fp-24],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
	b L886
	nop
L887:
	ld [%fp-36],%o0
	cmp %o0,0
	bne L890
	nop
	sethi %hi(LC125),%o1
	or %o1,%lo(LC125),%o0
	call _error,0
	nop
	b L884
	nop
L890:
	ld [%fp+68],%o0
	ld [%o0+12],%o1
	add %o1,-20,%o0
	mov 13,%o1
	cmp %o1,%o0
	blu L945
	nop
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(L946),%o2
	or %o2,%lo(L946),%o1
	ld [%o0+%o1],%o0
	jmp %o0
	nop
L946:
	.word	L909
	.word	L910
	.word	L892
	.word	L892
	.word	L898
	.word	L903
	.word	L904
	.word	L905
	.word	L906
	.word	L911
	.word	L907
	.word	L945
	.word	L945
	.word	L914
L892:
L893:
	ld [%fp+68],%o0
	ld [%o0+12],%o1
	cmp %o1,22
	bne L894
	nop
	ld [%fp-36],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-48]
	b L895
	nop
L894:
	sethi %hi(_instack+4),%o0
	ld [%o0+%lo(_instack+4)],%o1
	st %o1,[%fp-48]
L895:
	ld [%fp-48],%o0
	cmp %o0,0
	be L896
	nop
	ld [%fp-48],%o0
	call _strlen,0
	nop
	add %o0,3,%o1
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
	st %o1,[%fp-20]
	ld [%fp-20],%o0
	sethi %hi(LC126),%o2
	or %o2,%lo(LC126),%o1
	ld [%fp-48],%o2
	call _sprintf,0
	nop
	b L897
	nop
L896:
	sethi %hi(LC127),%o1
	or %o1,%lo(LC127),%o0
	st %o0,[%fp-20]
L897:
	b L891
	nop
L898:
	st %g0,[%fp-32]
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o1
	st %o1,[%fp-24]
L899:
	ld [%fp-24],%o0
	cmp %o0,0
	bl L900
	nop
	ld [%fp-24],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	ld [%o0+%o1],%o0
	cmp %o0,0
	be L902
	nop
	ld [%fp-32],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-32]
L902:
L901:
	ld [%fp-24],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
	b L899
	nop
L900:
	add %sp,92,%o1
	sub %o1,%sp,%o0
	add %o0,15,%o1
	mov %o1,%o0
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
	st %o1,[%fp-20]
	ld [%fp-32],%o0
	add %o0,-1,%o2
	ld [%fp-20],%o0
	sethi %hi(LC128),%o3
	or %o3,%lo(LC128),%o1
	call _sprintf,0
	nop
	b L891
	nop
L903:
	sethi %hi(_version_string),%o1
	ld [%o1+%lo(_version_string)],%o0
	call _strlen,0
	nop
	add %o0,3,%o1
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
	st %o1,[%fp-20]
	sethi %hi(_version_string),%o2
	ld [%fp-20],%o0
	sethi %hi(LC126),%o3
	or %o3,%lo(LC126),%o1
	ld [%o2+%lo(_version_string)],%o2
	call _sprintf,0
	nop
	b L891
	nop
L904:
	sethi %hi(LC129),%o1
	or %o1,%lo(LC129),%o0
	call _strlen,0
	nop
	add %o0,3,%o1
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
	st %o1,[%fp-20]
	ld [%fp-20],%o0
	sethi %hi(LC130),%o2
	or %o2,%lo(LC130),%o1
	sethi %hi(LC129),%o3
	or %o3,%lo(LC129),%o2
	call _sprintf,0
	nop
	b L891
	nop
L905:
	sethi %hi(LC131),%o1
	or %o1,%lo(LC131),%o0
	call _strlen,0
	nop
	add %o0,3,%o1
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
	st %o1,[%fp-20]
	ld [%fp-20],%o0
	sethi %hi(LC130),%o2
	or %o2,%lo(LC130),%o1
	sethi %hi(LC131),%o3
	or %o3,%lo(LC131),%o2
	call _sprintf,0
	nop
	b L891
	nop
L906:
	sethi %hi(LC132),%o1
	or %o1,%lo(LC132),%o0
	call _strlen,0
	nop
	add %o0,3,%o1
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
	st %o1,[%fp-20]
	ld [%fp-20],%o0
	sethi %hi(LC130),%o2
	or %o2,%lo(LC130),%o1
	sethi %hi(LC132),%o3
	or %o3,%lo(LC132),%o2
	call _sprintf,0
	nop
	b L891
	nop
L907:
	add %sp,92,%o1
	sub %o1,%sp,%o0
	add %o0,23,%o1
	mov %o1,%o0
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
	st %o1,[%fp-20]
	ld [%fp+68],%o2
	ld [%fp-20],%o0
	sethi %hi(LC128),%o3
	or %o3,%lo(LC128),%o1
	ld [%o2+24],%o2
	call _sprintf,0
	nop
	sethi %hi(_pcp_inside_if),%o0
	ld [%o0+%lo(_pcp_inside_if)],%o1
	cmp %o1,0
	be L908
	nop
	sethi %hi(_pcp_outfile),%o0
	ld [%o0+%lo(_pcp_outfile)],%o1
	cmp %o1,0
	be L908
	nop
	sethi %hi(_pcp_outfile),%o0
	ld [%fp+68],%o2
	ld [%fp+68],%o3
	ld [%o0+%lo(_pcp_outfile)],%o0
	sethi %hi(LC133),%o4
	or %o4,%lo(LC133),%o1
	ld [%o2+20],%o2
	ld [%o3+24],%o3
	call _fprintf,0
	nop
L908:
	b L891
	nop
L909:
	add %sp,92,%o1
	sub %o1,%sp,%o0
	add %o0,17,%o1
	mov %o1,%o0
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
	st %o1,[%fp-20]
	ld [%fp-36],%o2
	ld [%fp-20],%o0
	sethi %hi(LC128),%o3
	or %o3,%lo(LC128),%o1
	ld [%o2+12],%o2
	call _sprintf,0
	nop
	b L891
	nop
L910:
L911:
	add %sp,92,%o1
	sub %o1,%sp,%o0
	add %o0,27,%o1
	mov %o1,%o0
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
	st %o1,[%fp-20]
	call _timestamp,0
	nop
	st %o0,[%fp-40]
	ld [%fp+68],%o0
	ld [%o0+12],%o1
	cmp %o1,21
	bne L912
	nop
	ld [%fp-40],%o0
	ld [%o0+16],%o1
	mov %o1,%o0
	sll %o0,2,%o2
	sethi %hi(_monthnames),%o0
	or %o0,%lo(_monthnames),%o3
	ld [%fp-40],%o4
	ld [%fp-40],%o0
	ld [%o0+20],%o1
	add %o1,1900,%o5
	ld [%fp-20],%o0
	sethi %hi(LC134),%g2
	or %g2,%lo(LC134),%o1
	ld [%o2+%o3],%o2
	ld [%o4+12],%o3
	mov %o5,%o4
	call _sprintf,0
	nop
	b L913
	nop
L912:
	ld [%fp-40],%o2
	ld [%fp-40],%o3
	ld [%fp-40],%o4
	ld [%fp-20],%o0
	sethi %hi(LC135),%o5
	or %o5,%lo(LC135),%o1
	ld [%o2+8],%o2
	ld [%o3+4],%o3
	ld [%o4],%o4
	call _sprintf,0
	nop
L913:
	b L891
	nop
L914:
	sethi %hi(LC136),%o1
	or %o1,%lo(LC136),%o0
	st %o0,[%fp-20]
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-36]
L915:
L918:
	ld [%fp-36],%o0
	ld [%o0+24],%o1
	ldub [%o1],%o2
	and %o2,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L919
	nop
	ld [%fp-36],%o1
	ld [%fp-36],%o0
	ld [%fp-36],%o1
	ld [%o1+24],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+24]
	b L918
	nop
L919:
L917:
	b L916
	nop
	b L915
	nop
L916:
	ld [%fp-36],%o0
	ld [%o0+24],%o1
	ldub [%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,40
	bne L920
	nop
	ld [%fp-44],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-44]
	ld [%fp-36],%o1
	ld [%fp-36],%o0
	ld [%fp-36],%o1
	ld [%o1+24],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+24]
L921:
L924:
	ld [%fp-36],%o0
	ld [%o0+24],%o1
	ldub [%o1],%o2
	and %o2,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L925
	nop
	ld [%fp-36],%o1
	ld [%fp-36],%o0
	ld [%fp-36],%o1
	ld [%o1+24],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+24]
	b L924
	nop
L925:
L923:
	b L922
	nop
	b L921
	nop
L922:
L920:
	ld [%fp-36],%o0
	ld [%o0+24],%o1
	ldub [%o1],%o2
	and %o2,0xff,%o0
	sethi %hi(_is_idstart),%o2
	or %o2,%lo(_is_idstart),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	bne L926
	nop
	b L927
	nop
L926:
	ld [%fp-36],%o1
	ld [%o1+24],%o0
	mov -1,%o1
	mov -1,%o2
	call _lookup,0
	nop
	mov %o0,%o1
	st %o1,[%fp+68]
	cmp %o1,0
	be L928
	nop
	sethi %hi(_pcp_outfile),%o0
	ld [%o0+%lo(_pcp_outfile)],%o1
	cmp %o1,0
	be L929
	nop
	sethi %hi(_pcp_inside_if),%o0
	ld [%o0+%lo(_pcp_inside_if)],%o1
	cmp %o1,0
	be L929
	nop
	ld [%fp+68],%o0
	ld [%o0+24],%o1
	ld [%o1+8],%o0
	cmp %o0,0
	be L929
	nop
	sethi %hi(_pcp_outfile),%o0
	ld [%fp+68],%o2
	ld [%o0+%lo(_pcp_outfile)],%o0
	sethi %hi(LC137),%o3
	or %o3,%lo(LC137),%o1
	ld [%o2+20],%o2
	call _fprintf,0
	nop
L929:
	sethi %hi(LC138),%o1
	or %o1,%lo(LC138),%o0
	st %o0,[%fp-20]
	b L930
	nop
L928:
	sethi %hi(_pcp_outfile),%o0
	ld [%o0+%lo(_pcp_outfile)],%o1
	cmp %o1,0
	be L931
	nop
	sethi %hi(_pcp_inside_if),%o0
	ld [%o0+%lo(_pcp_inside_if)],%o1
	cmp %o1,0
	be L931
	nop
	ld [%fp-36],%o0
	ld [%o0+24],%o1
	st %o1,[%fp-52]
	sethi %hi(_pcp_outfile),%o1
	ld [%o1+%lo(_pcp_outfile)],%o0
	sethi %hi(LC139),%o2
	or %o2,%lo(LC139),%o1
	call _fprintf,0
	nop
L932:
	ld [%fp-52],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L933
	nop
	ld [%fp-52],%o0
	add %o0,1,%o1
	st %o1,[%fp-52]
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_pcp_outfile),%o1
	ld [%o1+%lo(_pcp_outfile)],%o1
	call _fputc,0
	nop
	b L932
	nop
L933:
	sethi %hi(_pcp_outfile),%o1
	ld [%o1+%lo(_pcp_outfile)],%o0
	sethi %hi(_pcp_outfile),%o1
	ld [%o1+%lo(_pcp_outfile)],%o0
	sethi %hi(_pcp_outfile),%o2
	ld [%o2+%lo(_pcp_outfile)],%o1
	ld [%o1],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0]
	cmp %o2,0
	bge L934
	nop
	sethi %hi(_pcp_outfile),%o1
	mov 10,%o0
	ld [%o1+%lo(_pcp_outfile)],%o1
	call __flsbuf,0
	nop
	b L935
	nop
L934:
	sethi %hi(_pcp_outfile),%o0
	ld [%o0+%lo(_pcp_outfile)],%o1
	ld [%o1+4],%o0
	add %o0,1,%o2
	st %o2,[%o1+4]
	mov 10,%o1
	stb %o1,[%o0]
L935:
L931:
L930:
	nop
L936:
	ld [%fp-36],%o0
	ld [%o0+24],%o1
	ldub [%o1],%o2
	and %o2,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L937
	nop
	ld [%fp-36],%o1
	ld [%fp-36],%o0
	ld [%fp-36],%o1
	ld [%o1+24],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+24]
	b L936
	nop
L937:
	nop
L938:
L941:
	ld [%fp-36],%o0
	ld [%o0+24],%o1
	ldub [%o1],%o2
	and %o2,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L942
	nop
	ld [%fp-36],%o1
	ld [%fp-36],%o0
	ld [%fp-36],%o1
	ld [%o1+24],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+24]
	b L941
	nop
L942:
L940:
	b L939
	nop
	b L938
	nop
L939:
	ld [%fp-44],%o0
	cmp %o0,0
	be L943
	nop
	ld [%fp-36],%o0
	ld [%o0+24],%o1
	ldub [%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,41
	be L944
	nop
	b L927
	nop
L944:
	ld [%fp-36],%o1
	ld [%fp-36],%o0
	ld [%fp-36],%o1
	ld [%o1+24],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+24]
L943:
	b L891
	nop
L927:
	sethi %hi(LC140),%o1
	or %o1,%lo(LC140),%o0
	call _error,0
	nop
	b L891
	nop
L945:
	sethi %hi(LC141),%o1
	or %o1,%lo(LC141),%o0
	call _error,0
	nop
	call _abort,0
	nop
L891:
	ld [%fp-20],%o0
	call _strlen,0
	nop
	st %o0,[%fp-28]
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%fp+72],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	ld [%fp-28],%o1
	cmp %o0,%o1
	bg L947
	nop
	ld [%fp+72],%o0
	ld [%fp-28],%o1
	call _grow_outbuf,0
	nop
	b L948
	nop
L947:
L948:
	ld [%fp+72],%o1
	ld [%fp-20],%o0
	ld [%o1+24],%o1
	ld [%fp-28],%o2
	call _bcopy,0
	nop
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%o1+24],%o2
	ld [%fp-28],%o1
	add %o2,%o1,%o2
	st %o2,[%o0+24]
	b L884
	nop
L884:
	ret
	restore
.data
	.align 4
_import_warning.26:
	.word	0
.text
	.align 8
LC142:
	.ascii "using `#import' is not recommended\0"
	.align 8
LC143:
	.ascii "The fact that a certain header file need not be processed more than once\12\0"
	.align 8
LC144:
	.ascii "should be indicated in the header file, not where it is used.\12\0"
	.align 8
LC145:
	.ascii "The best way to do this is with a conditional of this form:\12\12\0"
	.align 8
LC146:
	.ascii "  #ifndef _FOO_H_INCLUDED\12\0"
	.align 8
LC147:
	.ascii "  #define _FOO_H_INCLUDED\12\0"
	.align 8
LC148:
	.ascii "  ... <real contents of file> ...\12\0"
	.align 8
LC149:
	.ascii "  #endif /* Not _FOO_H_INCLUDED */\12\12\0"
	.align 8
LC150:
	.ascii "Then users can use `#include' any number of times.\12\0"
	.align 8
LC151:
	.ascii "GNU C automatically avoids processing the file more than once\12\0"
	.align 8
LC152:
	.ascii "when it is equipped with such a conditional.\12\0"
	.align 8
LC153:
	.ascii "`#import' expects \"fname\" or <fname>\0"
	.align 8
LC154:
	.ascii "`#include' expects \"fname\" or <fname>\0"
	.align 8
LC155:
	.ascii "/\0"
	.align 8
LC156:
	.ascii "No include path in which to find %s\0"
	.align 8
LC157:
	.ascii "%s\12\0"
	.align 8
LC158:
	.ascii "%s%d\0"
	.align 4
	.proc	04
_do_include:
	!#PROLOGUE# 0
	save %sp,-408,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	ld [%fp+80],%o0
	ld [%o0+12],%o1
	xor %o1,4,%o2
	subcc %g0,%o2,%g0
	subx %g0,-1,%o0
	st %o0,[%fp-20]
	ld [%fp+80],%o0
	ld [%o0+12],%o1
	xor %o1,3,%o2
	subcc %g0,%o2,%g0
	subx %g0,-1,%o0
	st %o0,[%fp-24]
	sethi %hi(_include),%o0
	ld [%o0+%lo(_include)],%o1
	st %o1,[%fp-48]
	st %g0,[%fp-68]
	st %g0,[%fp-80]
	st %g0,[%fp-132]
	mov -1,%o0
	st %o0,[%fp-136]
	mov -1,%o0
	st %o0,[%fp-76]
	ld [%fp-20],%o0
	cmp %o0,0
	be L950
	nop
	sethi %hi(_warn_import),%o0
	ld [%o0+%lo(_warn_import)],%o1
	cmp %o1,0
	be L950
	nop
	sethi %hi(_inhibit_warnings),%o0
	ld [%o0+%lo(_inhibit_warnings)],%o1
	cmp %o1,0
	bne L950
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o0,%o1,%o0
	ldub [%o0+40],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	bne L950
	nop
	sethi %hi(_import_warning.26),%o0
	ld [%o0+%lo(_import_warning.26)],%o1
	cmp %o1,0
	bne L950
	nop
	sethi %hi(_import_warning.26),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_import_warning.26)]
	sethi %hi(LC142),%o1
	or %o1,%lo(LC142),%o0
	call _warning,0
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC143),%o2
	or %o2,%lo(LC143),%o1
	call _fprintf,0
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC144),%o2
	or %o2,%lo(LC144),%o1
	call _fprintf,0
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC145),%o2
	or %o2,%lo(LC145),%o1
	call _fprintf,0
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC146),%o2
	or %o2,%lo(LC146),%o1
	call _fprintf,0
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC147),%o2
	or %o2,%lo(LC147),%o1
	call _fprintf,0
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC148),%o2
	or %o2,%lo(LC148),%o1
	call _fprintf,0
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC149),%o2
	or %o2,%lo(LC149),%o1
	call _fprintf,0
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC150),%o2
	or %o2,%lo(LC150),%o1
	call _fprintf,0
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC151),%o2
	or %o2,%lo(LC151),%o1
	call _fprintf,0
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC152),%o2
	or %o2,%lo(LC152),%o1
	call _fprintf,0
	nop
L950:
	nop
L951:
	ld [%fp+68],%o0
	st %o0,[%fp-40]
L952:
L955:
	ld [%fp-40],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L956
	nop
	ld [%fp-40],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-40]
	b L955
	nop
L956:
L954:
	b L953
	nop
	b L952
	nop
L953:
	nop
L957:
	ld [%fp+72],%o0
	ld [%fp-40],%o1
	cmp %o0,%o1
	be L958
	nop
	mov -1,%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L958
	nop
	ld [%fp+72],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp+72]
	b L957
	nop
L958:
	ld [%fp-40],%o0
	add %o0,1,%o1
	st %o1,[%fp-40]
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,34
	be L960
	nop
	cmp %o0,60
	be L979
	nop
	b L984
	nop
L960:
	ld [%fp-40],%o0
	st %o0,[%fp-156]
	ld [%fp+72],%o0
	ld [%fp-40],%o1
	sub %o0,%o1,%o0
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
	st %o1,[%fp-40]
	ld [%fp-40],%o0
	st %o0,[%fp-44]
L961:
	ld [%fp-156],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	be L962
	nop
L963:
	ld [%fp-156],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	be L964
	nop
	ld [%fp-156],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,34
	be L964
	nop
	ld [%fp-44],%o0
	add %o0,1,%o1
	st %o1,[%fp-44]
	ld [%fp-156],%o1
	add %o1,1,%o2
	st %o2,[%fp-156]
	ldub [%o1],%o1
	stb %o1,[%o0]
	b L963
	nop
L964:
	ld [%fp-156],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-156]
	ld [%fp-156],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	bne L965
	nop
	b L962
	nop
L965:
	nop
L966:
	ld [%fp-156],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	be L967
	nop
	ld [%fp-156],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L967
	nop
	ld [%fp-156],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-156]
	b L966
	nop
L967:
	ld [%fp-156],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	be L968
	nop
	ld [%fp-156],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,34
	bne L968
	nop
	ld [%fp-156],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-156]
	b L969
	nop
L968:
	b L970
	nop
L969:
	b L961
	nop
L962:
	ld [%fp-44],%o0
	add %o0,1,%o1
	st %o1,[%fp-44]
	stb %g0,[%o0]
	sethi %hi(_ignore_srcdir),%o0
	ld [%o0+%lo(_ignore_srcdir)],%o1
	cmp %o1,0
	be L971
	nop
	b L959
	nop
L971:
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-152]
L972:
	ld [%fp-152],%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	cmp %o0,%o1
	blu L973
	nop
	ld [%fp-152],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-164]
	cmp %o1,0
	be L975
	nop
	ld [%fp-48],%o0
	st %o0,[%fp-64]
	add %fp,-64,%o0
	st %o0,[%fp-48]
	ld [%fp-164],%o0
	mov 47,%o1
	call _rindex,0
	nop
	st %o0,[%fp-160]
	ld [%fp-160],%o0
	cmp %o0,0
	be L976
	nop
	ld [%fp-160],%o0
	ld [%fp-164],%o1
	sub %o0,%o1,%o0
	st %o0,[%fp-156]
	ld [%fp-156],%o0
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
	st %o1,[%fp-60]
	ld [%fp-60],%o0
	ld [%fp-164],%o1
	ld [%fp-156],%o2
	call _strncpy,0
	nop
	ld [%fp-60],%o0
	ld [%fp-156],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
	sethi %hi(_max_include_len),%o0
	ld [%fp-156],%o1
	ld [%o0+%lo(_max_include_len)],%o0
	cmp %o1,%o0
	ble L977
	nop
	sethi %hi(_max_include_len),%o0
	ld [%fp-156],%o1
	st %o1,[%o0+%lo(_max_include_len)]
L977:
	b L978
	nop
L976:
	st %g0,[%fp-60]
L978:
	b L973
	nop
L975:
L974:
	ld [%fp-152],%o1
	add %o1,-44,%o0
	mov %o0,%o1
	st %o1,[%fp-152]
	b L972
	nop
L973:
	b L959
	nop
L979:
	ld [%fp-40],%o0
	st %o0,[%fp-44]
L980:
	ld [%fp-44],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	be L981
	nop
	ld [%fp-44],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,62
	be L981
	nop
	ld [%fp-44],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-44]
	b L980
	nop
L981:
	ld [%fp-44],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,62
	bne L982
	nop
	ld [%fp-44],%o1
	add %o1,1,%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	bne L982
	nop
	mov 1,%o0
	st %o0,[%fp-132]
	sethi %hi(_first_bracket_include),%o0
	ld [%o0+%lo(_first_bracket_include)],%o1
	cmp %o1,0
	be L983
	nop
	sethi %hi(_first_bracket_include),%o0
	ld [%o0+%lo(_first_bracket_include)],%o1
	st %o1,[%fp-48]
L983:
	b L959
	nop
L982:
	b L970
	nop
L984:
	nop
L970:
	ld [%fp-80],%o0
	cmp %o0,0
	be L985
	nop
	ld [%fp-20],%o0
	cmp %o0,0
	be L986
	nop
	sethi %hi(LC153),%o1
	or %o1,%lo(LC153),%o0
	call _error,0
	nop
	b L987
	nop
L986:
	sethi %hi(LC154),%o1
	or %o1,%lo(LC154),%o0
	call _error,0
	nop
L987:
	mov 0,%i0
	b L949
	nop
	b L988
	nop
L985:
	add %fp,-128,%o0
	st %o0,[%sp+64]
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_to_temp_buffer,0
	nop
	unimp 44
	ld [%fp-104],%o0
	ld [%fp-108],%o1
	sub %o0,%o1,%o0
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
	st %o1,[%fp+68]
	ld [%fp-104],%o0
	ld [%fp-108],%o1
	sub %o0,%o1,%o2
	ld [%fp-108],%o0
	ld [%fp+68],%o1
	call _bcopy,0
	nop
	ld [%fp-104],%o0
	ld [%fp-108],%o1
	sub %o0,%o1,%o0
	ld [%fp+68],%o1
	add %o1,%o0,%o0
	st %o0,[%fp+72]
	ld [%fp-108],%o0
	call _free,0
	nop
	ld [%fp-80],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-80]
	b L951
	nop
L988:
L959:
	ld [%fp-24],%o0
	cmp %o0,0
	be L990
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-164]
L991:
	ld [%fp-164],%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	cmp %o0,%o1
	blu L992
	nop
	ld [%fp-164],%o0
	ld [%o0],%o1
	cmp %o1,0
	be L994
	nop
	ld [%fp-164],%o0
	ld [%o0+8],%o1
	cmp %o1,0
	be L995
	nop
	ld [%fp-164],%o0
	ld [%o0+8],%o1
	ld [%o1],%o0
	st %o0,[%fp-48]
L995:
	b L992
	nop
L994:
L993:
	ld [%fp-164],%o1
	add %o1,-44,%o0
	mov %o0,%o1
	st %o1,[%fp-164]
	b L991
	nop
L992:
L990:
	ld [%fp-44],%o0
	ld [%fp-40],%o1
	sub %o0,%o1,%o0
	st %o0,[%fp-72]
	sethi %hi(_max_include_len),%o0
	ld [%o0+%lo(_max_include_len)],%o1
	ld [%fp-72],%o2
	add %o1,%o2,%o0
	add %o0,2,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-28]
	ld [%fp-40],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L996
	nop
	ld [%fp-28],%o0
	ld [%fp-40],%o1
	ld [%fp-72],%o2
	call _strncpy,0
	nop
	ld [%fp-28],%o0
	ld [%fp-72],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
	ld [%fp-28],%o0
	call _redundant_include_p,0
	nop
	cmp %o0,0
	be L997
	nop
	mov 0,%i0
	b L949
	nop
L997:
	ld [%fp-20],%o0
	cmp %o0,0
	be L998
	nop
	ld [%fp-28],%o0
	call _lookup_import,0
	nop
	st %o0,[%fp-76]
	b L999
	nop
L998:
	ld [%fp-28],%o0
	mov 0,%o1
	mov 438,%o2
	call _open,0
	nop
	st %o0,[%fp-76]
L999:
	ld [%fp-76],%o0
	cmp %o0,-2
	bne L1000
	nop
	mov 0,%i0
	b L949
	nop
L1000:
	b L1001
	nop
L996:
	ld [%fp-48],%o0
	st %o0,[%fp-68]
L1002:
	ld [%fp-68],%o0
	cmp %o0,0
	be L1003
	nop
	ld [%fp-68],%o0
	ld [%o0+4],%o1
	cmp %o1,0
	be L1005
	nop
	ld [%fp-68],%o0
	ld [%o0+4],%o1
	ldub [%o1],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	bne L1006
	nop
	b L1004
	nop
L1006:
	ld [%fp-68],%o1
	ld [%fp-28],%o0
	ld [%o1+4],%o1
	call _strcpy,0
	nop
	ld [%fp-28],%o0
	sethi %hi(LC155),%o2
	or %o2,%lo(LC155),%o1
	call _strcat,0
	nop
	ld [%fp-28],%o0
	call _strlen,0
	nop
	ld [%fp-72],%o1
	add %o0,%o1,%o0
	ld [%fp-28],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
	b L1007
	nop
L1005:
	ld [%fp-28],%o0
	stb %g0,[%o0]
L1007:
	ld [%fp-28],%o0
	ld [%fp-40],%o1
	ld [%fp-72],%o2
	call _strncat,0
	nop
	ld [%fp-20],%o0
	cmp %o0,0
	be L1008
	nop
	ld [%fp-28],%o0
	call _lookup_import,0
	nop
	st %o0,[%fp-76]
	b L1009
	nop
L1008:
	ld [%fp-28],%o0
	mov 0,%o1
	mov 438,%o2
	call _open,0
	nop
	st %o0,[%fp-76]
L1009:
	ld [%fp-76],%o0
	cmp %o0,-2
	bne L1010
	nop
	mov 0,%i0
	b L949
	nop
L1010:
	ld [%fp-28],%o0
	call _redundant_include_p,0
	nop
	cmp %o0,0
	be L1011
	nop
	ld [%fp-76],%o0
	call _close,0
	nop
	mov 0,%i0
	b L949
	nop
L1011:
	ld [%fp-76],%o0
	cmp %o0,0
	bl L1012
	nop
	b L1003
	nop
L1012:
L1004:
	ld [%fp-68],%o0
	ld [%o0],%o1
	st %o1,[%fp-68]
	b L1002
	nop
L1003:
L1001:
	ld [%fp-76],%o0
	cmp %o0,0
	bge L1013
	nop
	ld [%fp-28],%o0
	ld [%fp-40],%o1
	ld [%fp-72],%o2
	call _strncpy,0
	nop
	ld [%fp-28],%o0
	ld [%fp-72],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
	ld [%fp-48],%o0
	cmp %o0,0
	be L1014
	nop
	ld [%fp-28],%o0
	call _error_from_errno,0
	nop
	b L1015
	nop
L1014:
	sethi %hi(LC156),%o1
	or %o1,%lo(LC156),%o0
	ld [%fp-28],%o1
	call _error,0
	nop
L1015:
	sethi %hi(_print_deps),%o0
	mov 0,%o1
	ld [%fp-132],%o2
	cmp %o2,0
	bne L1018
	nop
	sethi %hi(_system_include_depth),%o2
	ld [%o2+%lo(_system_include_depth)],%o3
	cmp %o3,0
	bg L1018
	nop
	b L1017
	nop
L1018:
	mov 1,%o1
L1017:
	ld [%o0+%lo(_print_deps)],%o0
	cmp %o0,%o1
	ble L1016
	nop
	sethi %hi(LC23),%o1
	or %o1,%lo(LC23),%o0
	mov 0,%o1
	call _deps_output,0
	nop
	ld [%fp-132],%o0
	cmp %o0,0
	be L1019
	nop
	ld [%fp-48],%o0
	st %o0,[%fp-68]
L1020:
	ld [%fp-68],%o0
	cmp %o0,0
	be L1021
	nop
	ld [%fp-68],%o0
	ld [%o0+4],%o1
	cmp %o1,0
	be L1023
	nop
	ld [%fp-68],%o0
	ld [%o0+4],%o1
	ldub [%o1],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	bne L1024
	nop
	b L1022
	nop
L1024:
	ld [%fp-68],%o1
	ld [%o1+4],%o0
	mov 0,%o1
	call _deps_output,0
	nop
	sethi %hi(LC155),%o1
	or %o1,%lo(LC155),%o0
	mov 0,%o1
	call _deps_output,0
	nop
	b L1021
	nop
L1023:
L1022:
	ld [%fp-68],%o0
	ld [%o0],%o1
	st %o1,[%fp-68]
	b L1020
	nop
L1021:
L1019:
	ld [%fp-40],%o0
	ld [%fp-72],%o1
	call _deps_output,0
	nop
	sethi %hi(LC88),%o1
	or %o1,%lo(LC88),%o0
	mov 0,%o1
	call _deps_output,0
	nop
L1016:
	b L1025
	nop
L1013:
	sethi %hi(_dont_repeat_files),%o0
	ld [%o0+%lo(_dont_repeat_files)],%o1
	st %o1,[%fp-164]
L1026:
	ld [%fp-164],%o0
	cmp %o0,0
	be L1027
	nop
	ld [%fp-164],%o1
	ld [%o1+4],%o0
	ld [%fp-28],%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L1029
	nop
	ld [%fp-76],%o0
	call _close,0
	nop
	mov 0,%i0
	b L949
	nop
L1029:
L1028:
	ld [%fp-164],%o0
	ld [%o0],%o1
	st %o1,[%fp-164]
	b L1026
	nop
L1027:
	nop
	sethi %hi(_all_include_files),%o0
	ld [%o0+%lo(_all_include_files)],%o1
	st %o1,[%fp-164]
L1030:
	ld [%fp-164],%o0
	cmp %o0,0
	be L1031
	nop
	ld [%fp-164],%o1
	ld [%o1+4],%o0
	ld [%fp-28],%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L1033
	nop
	b L1031
	nop
L1033:
L1032:
	ld [%fp-164],%o0
	ld [%o0],%o1
	st %o1,[%fp-164]
	b L1030
	nop
L1031:
	ld [%fp-164],%o0
	cmp %o0,0
	bne L1034
	nop
	mov 12,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-164]
	ld [%fp-164],%o0
	st %g0,[%o0+8]
	ld [%fp-164],%o0
	sethi %hi(_all_include_files),%o1
	ld [%o1+%lo(_all_include_files)],%o2
	st %o2,[%o0]
	sethi %hi(_all_include_files),%o0
	ld [%fp-164],%o1
	st %o1,[%o0+%lo(_all_include_files)]
	ld [%fp-28],%o0
	call _savestring,0
	nop
	ld [%fp-164],%o1
	st %o0,[%o1+4]
	sethi %hi(_print_deps),%o0
	mov 0,%o1
	ld [%fp-132],%o2
	cmp %o2,0
	bne L1037
	nop
	sethi %hi(_system_include_depth),%o2
	ld [%o2+%lo(_system_include_depth)],%o3
	cmp %o3,0
	bg L1037
	nop
	b L1036
	nop
L1037:
	mov 1,%o1
L1036:
	ld [%o0+%lo(_print_deps)],%o0
	cmp %o0,%o1
	ble L1035
	nop
	sethi %hi(LC23),%o1
	or %o1,%lo(LC23),%o0
	mov 0,%o1
	call _deps_output,0
	nop
	ld [%fp-28],%o0
	mov 0,%o1
	call _deps_output,0
	nop
	sethi %hi(LC88),%o1
	or %o1,%lo(LC88),%o0
	mov 0,%o1
	call _deps_output,0
	nop
L1035:
L1034:
	sethi %hi(_print_include_names),%o0
	ld [%o0+%lo(_print_include_names)],%o1
	cmp %o1,0
	be L1038
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC157),%o2
	or %o2,%lo(LC157),%o1
	ld [%fp-28],%o2
	call _fprintf,0
	nop
L1038:
	ld [%fp-132],%o0
	cmp %o0,0
	be L1039
	nop
	sethi %hi(_system_include_depth),%o1
	sethi %hi(_system_include_depth),%o0
	sethi %hi(_system_include_depth),%o1
	ld [%o1+%lo(_system_include_depth)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_system_include_depth)]
L1039:
	ld [%fp-76],%o0
	ld [%fp-28],%o1
	call _add_import,0
	nop
	ld [%fp-28],%o0
	call _strlen,0
	nop
	add %o0,30,%o1
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
	st %o1,[%fp-32]
	st %g0,[%fp-140]
	st %g0,[%fp-148]
	add %fp,-240,%o1
	ld [%fp-76],%o0
	call _fstat,0
	nop
	sethi %hi(_no_precomp),%o0
	ld [%o0+%lo(_no_precomp)],%o1
	cmp %o1,0
	bne L1040
	nop
L1041:
	ld [%fp-148],%o3
	add %o3,1,%o0
	st %o0,[%fp-148]
	ld [%fp-32],%o0
	sethi %hi(LC158),%o2
	or %o2,%lo(LC158),%o1
	ld [%fp-28],%o2
	call _sprintf,0
	nop
	ld [%fp-32],%o0
	mov 0,%o1
	mov 438,%o2
	call _open,0
	nop
	st %o0,[%fp-136]
	ld [%fp-136],%o0
	cmp %o0,-1
	be L1044
	nop
	add %fp,-312,%o1
	ld [%fp-136],%o0
	call _fstat,0
	nop
	add %fp,-240,%o1
	add %o1,4,%o0
	add %fp,-312,%o1
	add %o1,4,%o2
	mov %o2,%o1
	mov 4,%o2
	call _bcmp,0
	nop
	cmp %o0,0
	bne L1046
	nop
	ld [%fp-240],%o0
	ld [%fp-312],%o1
	cmp %o0,%o1
	bne L1046
	nop
	b L1045
	nop
L1046:
	add %fp,-144,%o2
	ld [%fp-136],%o0
	ld [%fp-28],%o1
	call _check_precompiled,0
	nop
	st %o0,[%fp-140]
	ld [%fp-136],%o0
	call _close,0
	nop
	b L1047
	nop
L1045:
	ld [%fp-136],%o0
	call _close,0
	nop
	b L1042
	nop
L1047:
L1044:
L1043:
	ld [%fp-136],%o0
	cmp %o0,-1
	be L1042
	nop
	ld [%fp-140],%o0
	cmp %o0,0
	bne L1042
	nop
	b L1041
	nop
L1042:
L1040:
	ld [%fp-140],%o0
	cmp %o0,0
	be L1048
	nop
	ld [%fp-32],%o0
	call _strlen,0
	nop
	add %o0,1,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-36]
	ld [%fp-36],%o0
	ld [%fp-32],%o1
	call _strcpy,0
	nop
	ld [%fp-140],%o0
	ld [%fp-144],%o1
	ld [%fp-28],%o2
	ld [%fp+76],%o3
	call _pcfinclude,0
	nop
	b L1049
	nop
L1048:
	ld [%fp-28],%o0
	call _is_system_include,0
	nop
	mov %o0,%o3
	ld [%fp-76],%o0
	ld [%fp-28],%o1
	ld [%fp+76],%o2
	ld [%fp-68],%o4
	call _finclude,0
	nop
L1049:
	ld [%fp-132],%o0
	cmp %o0,0
	be L1050
	nop
	sethi %hi(_system_include_depth),%o1
	sethi %hi(_system_include_depth),%o0
	sethi %hi(_system_include_depth),%o1
	ld [%o1+%lo(_system_include_depth)],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_system_include_depth)]
L1050:
L1025:
	mov 0,%i0
	b L949
	nop
L949:
	ret
	restore
	.align 4
	.proc	04
_redundant_include_p:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	sethi %hi(_all_include_files),%o0
	ld [%o0+%lo(_all_include_files)],%o1
	st %o1,[%fp-20]
L1052:
	ld [%fp-20],%o0
	cmp %o0,0
	be L1053
	nop
	ld [%fp-20],%o1
	ld [%fp+68],%o0
	ld [%o1+4],%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L1055
	nop
	ld [%fp-20],%o0
	ld [%o0+8],%o1
	cmp %o1,0
	be L1055
	nop
	ld [%fp-20],%o1
	ld [%o1+8],%o0
	mov -1,%o1
	mov -1,%o2
	call _lookup,0
	nop
	cmp %o0,0
	be L1055
	nop
	mov 1,%i0
	b L1051
	nop
L1055:
L1054:
	ld [%fp-20],%o0
	ld [%o0],%o1
	st %o1,[%fp-20]
	b L1052
	nop
L1053:
	mov 0,%i0
	b L1051
	nop
L1051:
	ret
	restore
	.align 4
	.proc	04
_is_system_include:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	mov %i0,%l0
	sethi %hi(_first_system_include),%o0
	ld [%o0+%lo(_first_system_include)],%o1
	st %o1,[%fp-20]
L1057:
	ld [%fp-20],%o0
	cmp %o0,0
	be L1058
	nop
	ld [%fp-20],%o0
	ld [%o0+4],%o1
	cmp %o1,0
	be L1060
	nop
	ld [%fp-20],%o0
	ld [%o0+4],%l1
	mov %l1,%o0
	call _strlen,0
	nop
	mov %o0,%l2
	mov %l1,%o0
	mov %l0,%o1
	mov %l2,%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L1061
	nop
	ldub [%l0+%l2],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,47
	bne L1061
	nop
	mov 1,%i0
	b L1056
	nop
L1061:
L1060:
L1059:
	ld [%fp-20],%o0
	ld [%o0],%o1
	st %o1,[%fp-20]
	b L1057
	nop
L1058:
	mov 0,%i0
	b L1056
	nop
L1056:
	ret
	restore
	.align 4
	.proc	020
_finclude:
	!#PROLOGUE# 0
	save %sp,-144,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	st %i4,[%fp+84]
	st %g0,[%fp-36]
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o1
	cmp %o1,198
	ble L1063
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o0,%o1,%o2
	ld [%o2+12],%o0
	call _line_for_error,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	sethi %hi(LC104),%o2
	or %o2,%lo(LC104),%o1
	call _error_with_line,0
	nop
	b L1062
	nop
L1063:
	add %fp,-20,%o1
	add %fp,-24,%o2
	ld [%fp+68],%o0
	call _file_size_and_mode,0
	nop
	cmp %o0,0
	bge L1064
	nop
	ld [%fp+72],%o0
	call _perror_with_name,0
	nop
	ld [%fp+68],%o0
	call _close,0
	nop
	b L1062
	nop
L1064:
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack+44),%o2
	or %o2,%lo(_instack+44),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-32]
	ld [%fp-32],%o0
	mov 44,%o1
	call _bzero,0
	nop
	ld [%fp-32],%o0
	ld [%fp-32],%o1
	ld [%fp+72],%o2
	st %o2,[%o1]
	st %o2,[%o0+4]
	ld [%fp-32],%o0
	st %g0,[%o0+16]
	ld [%fp-32],%o0
	mov 1,%o1
	st %o1,[%o0+12]
	ld [%fp-32],%o0
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o2
	st %o2,[%o0+32]
	ld [%fp-32],%o0
	ldub [%fp+83],%o1
	stb %o1,[%o0+40]
	ld [%fp-32],%o0
	ld [%fp+84],%o1
	st %o1,[%o0+8]
	ld [%fp-20],%o0
	sethi %hi(61440),%o1
	and %o0,%o1,%o0
	sethi %hi(32768),%o1
	cmp %o0,%o1
	bne L1065
	nop
	ld [%fp-24],%o0
	add %o0,2,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	ld [%fp-32],%o1
	st %o0,[%o1+20]
	ld [%fp-32],%o0
	ld [%fp-32],%o1
	ld [%o1+20],%o2
	st %o2,[%o0+24]
L1066:
	ld [%fp-24],%o0
	cmp %o0,0
	ble L1067
	nop
	ld [%fp-32],%o0
	ld [%fp-32],%o1
	ld [%o0+20],%o0
	ld [%o1+16],%o2
	add %o0,%o2,%o1
	ld [%fp+68],%o0
	ld [%fp-24],%o2
	call _read,0
	nop
	st %o0,[%fp-28]
	ld [%fp-28],%o0
	cmp %o0,0
	bg L1068
	nop
	ld [%fp-28],%o0
	cmp %o0,0
	bne L1069
	nop
	b L1067
	nop
L1069:
	b L1070
	nop
L1068:
	ld [%fp-32],%o0
	ld [%fp-32],%o1
	ld [%o1+16],%o2
	ld [%fp-28],%o1
	add %o2,%o1,%o2
	st %o2,[%o0+16]
	ld [%fp-24],%o0
	ld [%fp-28],%o1
	sub %o0,%o1,%o0
	st %o0,[%fp-24]
	b L1066
	nop
L1067:
	b L1071
	nop
L1065:
	mov 2000,%o0
	st %o0,[%fp-48]
	st %g0,[%fp-24]
	ld [%fp-48],%o0
	add %o0,2,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-44]
	ld [%fp-32],%o0
	ld [%fp-44],%o1
	st %o1,[%o0+20]
	ld [%fp-44],%o0
	st %o0,[%fp-40]
L1072:
	ld [%fp-48],%o0
	ld [%fp-24],%o1
	sub %o0,%o1,%o2
	ld [%fp+68],%o0
	ld [%fp-40],%o1
	call _read,0
	nop
	st %o0,[%fp-28]
	ld [%fp-28],%o0
	cmp %o0,0
	bge L1075
	nop
	b L1070
	nop
L1075:
	ld [%fp-28],%o0
	cmp %o0,0
	bne L1076
	nop
	b L1073
	nop
L1076:
	ld [%fp-24],%o0
	ld [%fp-28],%o1
	add %o0,%o1,%o0
	st %o0,[%fp-24]
	ld [%fp-40],%o0
	ld [%fp-28],%o1
	add %o0,%o1,%o0
	st %o0,[%fp-40]
	ld [%fp-48],%o0
	ld [%fp-24],%o1
	cmp %o0,%o1
	bne L1077
	nop
	ld [%fp-48],%o0
	mov %o0,%o1
	sll %o1,1,%o0
	st %o0,[%fp-48]
	ld [%fp-48],%o0
	add %o0,2,%o1
	ld [%fp-44],%o0
	call _xrealloc,0
	nop
	st %o0,[%fp-44]
	ld [%fp-32],%o0
	ld [%fp-44],%o1
	st %o1,[%o0+20]
	ld [%fp-44],%o0
	ld [%fp-24],%o1
	add %o0,%o1,%o0
	st %o0,[%fp-40]
L1077:
L1074:
	b L1072
	nop
L1073:
	ld [%fp-32],%o0
	ld [%fp-32],%o1
	ld [%o1+20],%o2
	st %o2,[%o0+24]
	ld [%fp-32],%o0
	ld [%fp-24],%o1
	st %o1,[%o0+16]
L1071:
	ld [%fp+68],%o0
	call _close,0
	nop
	sethi %hi(_indepth),%o1
	sethi %hi(_indepth),%o0
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_indepth)]
	sethi %hi(_input_file_stack_tick),%o1
	sethi %hi(_input_file_stack_tick),%o0
	sethi %hi(_input_file_stack_tick),%o1
	ld [%o1+%lo(_input_file_stack_tick)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_input_file_stack_tick)]
	sethi %hi(_no_trigraphs),%o0
	ld [%o0+%lo(_no_trigraphs)],%o1
	cmp %o1,0
	bne L1078
	nop
	ld [%fp-32],%o0
	call _trigraph_pcp,0
	nop
L1078:
	ld [%fp-32],%o0
	ld [%o0+16],%o1
	cmp %o1,0
	ble L1081
	nop
	ld [%fp-32],%o0
	ld [%fp-32],%o1
	ld [%o1+16],%o2
	ld [%o0+20],%o1
	add %o2,%o1,%o0
	ldub [%o0-1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L1080
	nop
	b L1081
	nop
L1081:
	ld [%fp-32],%o0
	ld [%o0+16],%o1
	cmp %o1,1
	ble L1079
	nop
	ld [%fp-32],%o0
	ld [%fp-32],%o1
	ld [%o1+16],%o2
	ld [%o0+20],%o1
	add %o2,%o1,%o0
	ldub [%o0-2],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	be L1080
	nop
	b L1079
	nop
L1080:
	ld [%fp-32],%o0
	ld [%fp-32],%o2
	ld [%o2+16],%o1
	add %o1,1,%o3
	st %o3,[%o2+16]
	ld [%o0+20],%o2
	add %o1,%o2,%o0
	mov 10,%o1
	stb %o1,[%o0]
	mov 1,%o0
	st %o0,[%fp-36]
L1079:
	ld [%fp-32],%o0
	ld [%fp-32],%o1
	ld [%o0+20],%o0
	ld [%o1+16],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
	ld [%fp-32],%o0
	ld [%fp+76],%o1
	mov 0,%o2
	mov 1,%o3
	call _output_line_command,0
	nop
	ld [%fp+76],%o0
	mov 0,%o1
	call _rescan,0
	nop
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L1082
	nop
	ld [%fp-36],%o0
	cmp %o0,0
	be L1082
	nop
	sethi %hi(LC90),%o1
	or %o1,%lo(LC90),%o0
	call _pedwarn,0
	nop
L1082:
	sethi %hi(_indepth),%o1
	sethi %hi(_indepth),%o0
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_indepth)]
	sethi %hi(_input_file_stack_tick),%o1
	sethi %hi(_input_file_stack_tick),%o0
	sethi %hi(_input_file_stack_tick),%o1
	ld [%o1+%lo(_input_file_stack_tick)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_input_file_stack_tick)]
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o1
	or %o1,%lo(_instack),%o2
	add %o0,%o2,%o1
	mov %o1,%o0
	ld [%fp+76],%o1
	mov 0,%o2
	mov 2,%o3
	call _output_line_command,0
	nop
	ld [%fp-32],%o1
	ld [%o1+20],%o0
	call _free,0
	nop
	b L1062
	nop
L1070:
	ld [%fp+72],%o0
	call _perror_with_name,0
	nop
	ld [%fp+68],%o0
	call _close,0
	nop
	ld [%fp-32],%o1
	ld [%o1+20],%o0
	call _free,0
	nop
L1062:
	ret
	restore
	.align 4
	.proc	020
_record_control_macro:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	sethi %hi(_all_include_files),%o0
	ld [%o0+%lo(_all_include_files)],%o1
	st %o1,[%fp-20]
L1084:
	ld [%fp-20],%o0
	cmp %o0,0
	be L1085
	nop
	ld [%fp-20],%o1
	ld [%o1+4],%o0
	ld [%fp+68],%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L1087
	nop
	ld [%fp-20],%o0
	ld [%fp+72],%o1
	st %o1,[%o0+8]
	b L1083
	nop
L1087:
L1086:
	ld [%fp-20],%o0
	ld [%o0],%o1
	st %o1,[%fp-20]
	b L1084
	nop
L1085:
	call _abort,0
	nop
L1083:
	ret
	restore
	.align 4
	.proc	04
_import_hash:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %g0,[%fp-20]
L1089:
	ld [%fp+68],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L1090
	nop
	ld [%fp+68],%o0
	add %o0,1,%o1
	st %o1,[%fp+68]
	ldub [%o0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	ld [%fp-20],%o1
	add %o1,%o0,%o0
	st %o0,[%fp-20]
	b L1089
	nop
L1090:
	ld [%fp-20],%o0
	mov 31,%o1
	call .rem,0
	nop
	mov %o0,%i0
	b L1088
	nop
L1088:
	ret
	restore
	.align 4
	.proc	04
_lookup_import:
	!#PROLOGUE# 0
	save %sp,-208,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	call _import_hash,0
	nop
	st %o0,[%fp-28]
	ld [%fp-28],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_import_hash_table),%o2
	or %o2,%lo(_import_hash_table),%o1
	ld [%o0+%o1],%o0
	st %o0,[%fp-20]
L1092:
	ld [%fp-20],%o0
	cmp %o0,0
	be L1093
	nop
	ld [%fp-20],%o1
	ld [%fp+68],%o0
	ld [%o1],%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L1094
	nop
	mov -2,%i0
	b L1091
	nop
L1094:
	ld [%fp-20],%o0
	ld [%o0+12],%o1
	st %o1,[%fp-20]
	b L1092
	nop
L1093:
	ld [%fp+68],%o0
	mov 0,%o1
	mov 438,%o2
	call _open,0
	nop
	st %o0,[%fp-108]
	ld [%fp-108],%o0
	cmp %o0,0
	bge L1095
	nop
	ld [%fp-108],%i0
	b L1091
	nop
L1095:
	add %fp,-104,%o1
	ld [%fp-108],%o0
	call _fstat,0
	nop
	st %g0,[%fp-24]
L1096:
	ld [%fp-24],%o0
	cmp %o0,30
	bg L1097
	nop
	ld [%fp-24],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_import_hash_table),%o2
	or %o2,%lo(_import_hash_table),%o1
	ld [%o0+%o1],%o0
	st %o0,[%fp-20]
L1099:
	ld [%fp-20],%o0
	cmp %o0,0
	be L1100
	nop
	ld [%fp-20],%o1
	add %o1,4,%o0
	add %fp,-104,%o1
	add %o1,4,%o2
	mov %o2,%o1
	mov 4,%o2
	call _bcmp,0
	nop
	cmp %o0,0
	bne L1101
	nop
	ld [%fp-20],%o0
	ld [%o0+8],%o1
	ld [%fp-104],%o0
	cmp %o1,%o0
	bne L1101
	nop
	ld [%fp-108],%o0
	call _close,0
	nop
	mov -2,%i0
	b L1091
	nop
L1101:
	ld [%fp-20],%o0
	ld [%o0+12],%o1
	st %o1,[%fp-20]
	b L1099
	nop
L1100:
L1098:
	ld [%fp-24],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
	b L1096
	nop
L1097:
	ld [%fp-108],%i0
	b L1091
	nop
L1091:
	ret
	restore
	.align 4
	.proc	020
_add_import:
	!#PROLOGUE# 0
	save %sp,-192,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+72],%o0
	call _import_hash,0
	nop
	st %o0,[%fp-24]
	add %fp,-96,%o1
	ld [%fp+68],%o0
	call _fstat,0
	nop
	mov 16,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-20]
	ld [%fp+72],%o0
	call _strlen,0
	nop
	add %o0,1,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	ld [%fp-20],%o1
	st %o0,[%o1]
	ld [%fp-20],%o1
	ld [%o1],%o0
	ld [%fp+72],%o1
	call _strcpy,0
	nop
	add %fp,-96,%o1
	add %o1,4,%o0
	ld [%fp-20],%o2
	add %o2,4,%o1
	mov 4,%o2
	call _bcopy,0
	nop
	ld [%fp-20],%o0
	ld [%fp-96],%o1
	st %o1,[%o0+8]
	ld [%fp-20],%o0
	ld [%fp-24],%o1
	mov %o1,%o2
	sll %o2,2,%o1
	sethi %hi(_import_hash_table),%o3
	or %o3,%lo(_import_hash_table),%o2
	ld [%o1+%o2],%o1
	st %o1,[%o0+12]
	ld [%fp-24],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_import_hash_table),%o2
	or %o2,%lo(_import_hash_table),%o1
	ld [%fp-20],%o2
	st %o2,[%o0+%o1]
L1102:
	ret
	restore
	.align 4
	.proc	0102
_check_precompiled:
	!#PROLOGUE# 0
	save %sp,-144,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %g0,[%fp-28]
	sethi %hi(_pcp_outfile),%o0
	ld [%o0+%lo(_pcp_outfile)],%o1
	cmp %o1,0
	be L1104
	nop
	mov 0,%i0
	b L1103
	nop
L1104:
	add %fp,-20,%o1
	add %fp,-24,%o2
	ld [%fp+68],%o0
	call _file_size_and_mode,0
	nop
	cmp %o0,0
	bge L1105
	nop
	mov 0,%i0
	b L1103
	nop
L1105:
	ld [%fp-20],%o0
	sethi %hi(61440),%o1
	and %o0,%o1,%o0
	sethi %hi(32768),%o1
	cmp %o0,%o1
	bne L1106
	nop
	ld [%fp-24],%o0
	add %o0,2,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-32]
L1107:
	ld [%fp-24],%o0
	cmp %o0,0
	ble L1108
	nop
	ld [%fp-32],%o0
	ld [%fp-28],%o2
	add %o0,%o2,%o1
	ld [%fp+68],%o0
	ld [%fp-24],%o2
	call _read,0
	nop
	st %o0,[%fp-40]
	ld [%fp-40],%o0
	cmp %o0,0
	bge L1109
	nop
	b L1110
	nop
L1109:
	ld [%fp-40],%o0
	cmp %o0,0
	bne L1111
	nop
	b L1108
	nop
L1111:
	ld [%fp-28],%o0
	ld [%fp-40],%o1
	add %o0,%o1,%o0
	st %o0,[%fp-28]
	ld [%fp-24],%o0
	ld [%fp-40],%o1
	sub %o0,%o1,%o0
	st %o0,[%fp-24]
	b L1107
	nop
L1108:
	b L1112
	nop
L1106:
	call _abort,0
	nop
L1112:
	ld [%fp-28],%o0
	cmp %o0,0
	ble L1113
	nop
	ld [%fp-28],%o0
	ld [%fp-32],%o1
	add %o0,%o1,%o0
	ldub [%o0-1],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,10
	be L1113
	nop
	ld [%fp-28],%o0
	add %o0,1,%o1
	st %o1,[%fp-28]
	ld [%fp-32],%o1
	add %o0,%o1,%o0
	mov 10,%o1
	stb %o1,[%o0]
L1113:
	ld [%fp-32],%o0
	ld [%fp-28],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
	ld [%fp+76],%o0
	ld [%fp-32],%o1
	ld [%fp-28],%o2
	add %o1,%o2,%o1
	st %o1,[%o0]
	ld [%fp-32],%o0
	call _check_preconditions,0
	nop
	cmp %o0,0
	bne L1114
	nop
	b L1110
	nop
L1114:
	nop
	ld [%fp-32],%o0
	st %o0,[%fp-44]
L1115:
	ld [%fp-44],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L1116
	nop
L1117:
	ld [%fp-44],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-44]
	b L1115
	nop
L1116:
	ld [%fp-44],%o1
	add %o1,1,%o0
	mov %o0,%i0
	b L1103
	nop
L1110:
	ld [%fp-32],%o0
	call _free,0
	nop
	mov 0,%i0
	b L1103
	nop
L1103:
	ret
	restore
	.align 8
LC159:
	.ascii "Bad format encountered while reading precompiled file\0"
	.align 4
	.proc	04
_check_preconditions:
	!#PROLOGUE# 0
	save %sp,-144,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
L1119:
	ld [%fp+68],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L1120
	nop
	ld [%fp+68],%o0
	mov 10,%o1
	call _index,0
	nop
	st %o0,[%fp-36]
	ld [%fp+68],%o0
	add %o0,1,%o1
	st %o1,[%fp+68]
	ldub [%o0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,35
	be L1121
	nop
	sethi %hi(LC159),%o1
	or %o1,%lo(LC159),%o0
	call _error,0
	nop
	mov 0,%i0
	b L1118
	nop
L1121:
	ld [%fp+68],%o0
	sethi %hi(LC5),%o2
	or %o2,%lo(LC5),%o1
	mov 6,%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L1122
	nop
	ld [%fp+68],%o0
	add %o0,6,%o1
	st %o1,[%fp+68]
	add %fp,-32,%o0
	st %o0,[%sp+64]
	ld [%fp+68],%o0
	ld [%fp-36],%o1
	mov 0,%o2
	call _create_definition,0
	nop
	unimp 12
	ld [%fp-32],%o0
	cmp %o0,0
	bne L1123
	nop
	call _abort,0
	nop
L1123:
	ld [%fp-28],%o0
	ld [%fp-24],%o1
	mov -1,%o2
	call _lookup,0
	nop
	mov %o0,%o1
	st %o1,[%fp-40]
	cmp %o1,0
	be L1125
	nop
	ld [%fp-40],%o0
	ld [%o0+12],%o1
	cmp %o1,31
	be L1126
	nop
	ld [%fp-40],%o0
	ld [%o0+12],%o1
	cmp %o1,30
	bne L1125
	nop
	b L1126
	nop
L1126:
	ld [%fp-40],%o0
	ld [%o0+12],%o1
	cmp %o1,31
	bne L1124
	nop
	ld [%fp-40],%o1
	ld [%fp-32],%o0
	ld [%o1+24],%o1
	call _compare_defs,0
	nop
	cmp %o0,0
	bne L1124
	nop
	ld [%fp-32],%o0
	ld [%o0+4],%o1
	cmp %o1,2
	bne L1125
	nop
	ld [%fp-32],%o0
	ld [%o0+12],%o1
	ldub [%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,10
	bne L1125
	nop
	ld [%fp-32],%o0
	mov 1,%o1
	ld [%o0+12],%o2
	add %o1,%o2,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,32
	bne L1125
	nop
	b L1124
	nop
L1125:
	mov 0,%i0
	b L1118
	nop
L1124:
	b L1127
	nop
L1122:
	ld [%fp+68],%o0
	sethi %hi(LC16),%o2
	or %o2,%lo(LC16),%o1
	mov 5,%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L1128
	nop
	ld [%fp+68],%o0
	add %o0,5,%o1
	st %o1,[%fp+68]
L1129:
	ld [%fp+68],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1130
	nop
	ld [%fp+68],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp+68]
	b L1129
	nop
L1130:
	ld [%fp+68],%o0
	st %o0,[%fp-40]
L1131:
	ld [%fp+68],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1132
	nop
	ld [%fp+68],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp+68]
	b L1131
	nop
L1132:
	ld [%fp+68],%o0
	ld [%fp-40],%o1
	sub %o0,%o1,%o0
	st %o0,[%fp-44]
	ld [%fp-40],%o0
	ld [%fp-44],%o1
	mov -1,%o2
	call _lookup,0
	nop
	cmp %o0,0
	be L1133
	nop
	mov 0,%i0
	b L1118
	nop
L1133:
	b L1134
	nop
L1128:
	sethi %hi(LC159),%o1
	or %o1,%lo(LC159),%o0
	call _error,0
	nop
	mov 0,%i0
	b L1118
	nop
L1134:
L1127:
	ld [%fp-36],%o0
	add %o0,1,%o1
	st %o1,[%fp+68]
	b L1119
	nop
L1120:
	mov 1,%i0
	b L1118
	nop
L1118:
	ret
	restore
	.align 4
	.proc	020
_pcfinclude:
	!#PROLOGUE# 0
	save %sp,-192,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	ld [%fp+68],%o0
	st %o0,[%fp-72]
	ld [%fp-72],%o0
	add %o0,1,%o1
	st %o1,[%fp-72]
	ldub [%o0],%o0
	and %o0,0xff,%o1
	st %o1,[%fp-68]
	ld [%fp-68],%o1
	sll %o1,8,%o0
	ld [%fp-72],%o1
	add %o1,1,%o2
	st %o2,[%fp-72]
	ldub [%o1],%o2
	and %o2,0xff,%o1
	or %o0,%o1,%o0
	st %o0,[%fp-68]
	ld [%fp-68],%o1
	sll %o1,8,%o0
	ld [%fp-72],%o1
	add %o1,1,%o2
	st %o2,[%fp-72]
	ldub [%o1],%o2
	and %o2,0xff,%o1
	or %o0,%o1,%o0
	st %o0,[%fp-68]
	ld [%fp-68],%o1
	sll %o1,8,%o0
	ld [%fp-72],%o1
	add %o1,1,%o2
	st %o2,[%fp-72]
	ldub [%o1],%o2
	and %o2,0xff,%o1
	or %o0,%o1,%o0
	st %o0,[%fp-68]
L1136:
	ld [%fp-68],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-68]
	cmp %o1,-1
	be L1137
	nop
	ld [%fp-72],%o1
	and %o1,3,%o0
	cmp %o0,0
	be L1138
	nop
	ld [%fp-72],%o1
	add %o1,4,%o0
	ld [%fp-72],%o2
	and %o2,3,%o1
	sub %o0,%o1,%o0
	st %o0,[%fp-72]
L1138:
	ld [%fp-72],%o0
	st %o0,[%fp-84]
	ld [%fp-72],%o1
	add %o1,28,%o0
	mov %o0,%o1
	st %o1,[%fp-72]
	st %o1,[%fp-76]
L1139:
	ld [%fp-72],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L1140
	nop
L1141:
	ld [%fp-72],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-72]
	b L1139
	nop
L1140:
	ld [%fp-72],%o1
	add %o1,1,%o0
	st %o0,[%fp-72]
	add %fp,-64,%o0
	st %o0,[%sp+64]
	ld [%fp-76],%o0
	mov 0,%o2
	mov 0,%o3
	call _expand_to_temp_buffer,0
	nop
	unimp 44
	ld [%fp-84],%o0
	ld [%fp-44],%o1
	st %o1,[%o0]
	ld [%fp-84],%o0
	ld [%fp-48],%o1
	st %o1,[%o0+4]
	ld [%fp-84],%o0
	st %g0,[%o0+8]
	ld [%fp-84],%o0
	ld [%fp+76],%o1
	st %o1,[%o0+16]
	ld [%fp-84],%o0
	sethi %hi(_outbuf+24),%o1
	sethi %hi(_outbuf+20),%o2
	ld [%o1+%lo(_outbuf+24)],%o1
	ld [%o2+%lo(_outbuf+20)],%o2
	sub %o1,%o2,%o1
	st %o1,[%o0+24]
	ld [%fp-84],%o0
	st %g0,[%o0+20]
	sethi %hi(_stringlist_tailp),%o1
	ld [%o1+%lo(_stringlist_tailp)],%o0
	ld [%fp-84],%o1
	st %o1,[%o0]
	sethi %hi(_stringlist_tailp),%o0
	ld [%fp-84],%o1
	add %o1,20,%o2
	st %o2,[%o0+%lo(_stringlist_tailp)]
	ld [%fp-72],%o0
	add %o0,1,%o1
	st %o1,[%fp-72]
	ldub [%o0],%o0
	and %o0,0xff,%o1
	st %o1,[%fp-88]
	ld [%fp-88],%o1
	sll %o1,8,%o0
	ld [%fp-72],%o1
	add %o1,1,%o2
	st %o2,[%fp-72]
	ldub [%o1],%o2
	and %o2,0xff,%o1
	or %o0,%o1,%o0
	st %o0,[%fp-88]
	ld [%fp-88],%o1
	sll %o1,8,%o0
	ld [%fp-72],%o1
	add %o1,1,%o2
	st %o2,[%fp-72]
	ldub [%o1],%o2
	and %o2,0xff,%o1
	or %o0,%o1,%o0
	st %o0,[%fp-88]
	ld [%fp-88],%o1
	sll %o1,8,%o0
	ld [%fp-72],%o1
	add %o1,1,%o2
	st %o2,[%fp-72]
	ldub [%o1],%o2
	and %o2,0xff,%o1
	or %o0,%o1,%o0
	st %o0,[%fp-88]
	ld [%fp-88],%o0
	cmp %o0,-1
	bne L1142
	nop
	ld [%fp-84],%o0
	mov 1,%o1
	st %o1,[%o0+8]
	b L1143
	nop
L1142:
	nop
L1144:
	ld [%fp-88],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-88]
	cmp %o1,-1
	be L1145
	nop
	ld [%fp-72],%o0
	st %o0,[%fp-92]
	ld [%fp-72],%o0
	add %o0,8,%o1
	st %o1,[%fp-72]
	ld [%fp-72],%o0
	call _strlen,0
	nop
	ld [%fp-72],%o1
	add %o1,%o0,%o0
	st %o0,[%fp-80]
	ld [%fp-92],%o0
	ld [%fp-84],%o1
	st %o1,[%o0]
	add %fp,-64,%o0
	st %o0,[%sp+64]
	ld [%fp-72],%o0
	ld [%fp-80],%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_to_temp_buffer,0
	nop
	unimp 44
	ld [%fp-44],%o0
	st %o0,[%fp-40]
L1147:
	ld [%fp-40],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1148
	nop
	ld [%fp-40],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-40]
	b L1147
	nop
L1148:
	ld [%fp-40],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idstart),%o2
	or %o2,%lo(_is_idstart),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1150
	nop
	ld [%fp-44],%o0
	ld [%fp-48],%o1
	add %o0,%o1,%o0
	ld [%fp-40],%o1
	cmp %o1,%o0
	be L1150
	nop
	b L1149
	nop
L1150:
	ld [%fp-84],%o0
	mov 1,%o1
	st %o1,[%o0+8]
	b L1146
	nop
L1149:
	ld [%fp-40],%o0
	mov -1,%o1
	mov -1,%o2
	call _lookup,0
	nop
	st %o0,[%fp-96]
	ld [%fp-96],%o0
	cmp %o0,0
	bne L1151
	nop
	ld [%fp-92],%o0
	st %g0,[%o0+4]
	ld [%fp-40],%o0
	mov -1,%o1
	mov 34,%o2
	mov 0,%o3
	ld [%fp-92],%o4
	mov -1,%o5
	call _install,0
	nop
	b L1152
	nop
L1151:
	ld [%fp-96],%o0
	ld [%o0+12],%o1
	cmp %o1,34
	bne L1153
	nop
	ld [%fp-92],%o0
	ld [%fp-96],%o1
	ld [%o1+24],%o2
	st %o2,[%o0+4]
	ld [%fp-96],%o0
	ld [%fp-92],%o1
	st %o1,[%o0+24]
	b L1154
	nop
L1153:
	ld [%fp-84],%o0
	mov 1,%o1
	st %o1,[%o0+8]
L1154:
L1152:
L1146:
	ld [%fp-44],%o0
	call _free,0
	nop
	ld [%fp-80],%o0
	add %o0,1,%o1
	st %o1,[%fp-72]
	b L1144
	nop
L1145:
L1143:
	b L1136
	nop
L1137:
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o1
	or %o1,%lo(_instack),%o2
	add %o0,%o2,%o1
	mov %o1,%o0
	ld [%fp+80],%o1
	mov 0,%o2
	mov 1,%o3
	call _output_line_command,0
	nop
L1135:
	ret
	restore
	.align 4
	.proc	020
_pcstring_used:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	ld [%o0+24],%o1
	st %o1,[%fp-20]
L1156:
	ld [%fp-20],%o0
	cmp %o0,0
	be L1157
	nop
	ld [%fp-20],%o0
	ld [%o0],%o1
	mov 1,%o0
	st %o0,[%o1+8]
L1158:
	ld [%fp-20],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-20]
	b L1156
	nop
L1157:
	ld [%fp+68],%o0
	call _delete_macro,0
	nop
L1155:
	ret
	restore
	.align 8
LC160:
	.ascii "\12# %d \"%s\"\12\0"
	.align 4
	.proc	020
_write_output:
	!#PROLOGUE# 0
	save %sp,-136,%sp
	!#PROLOGUE# 1
	mov 80,%o0
	st %o0,[%fp-28]
	ld [%fp-28],%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-32]
	sethi %hi(_outbuf+20),%o0
	ld [%o0+%lo(_outbuf+20)],%o1
	st %o1,[%fp-24]
	sethi %hi(_stringlist),%o0
	ld [%o0+%lo(_stringlist)],%o1
	st %o1,[%fp-20]
L1160:
	sethi %hi(_outbuf+24),%o0
	ld [%fp-24],%o1
	ld [%o0+%lo(_outbuf+24)],%o0
	cmp %o1,%o0
	blu L1162
	nop
	ld [%fp-20],%o0
	cmp %o0,0
	bne L1162
	nop
	b L1161
	nop
L1162:
	ld [%fp-20],%o0
	cmp %o0,0
	be L1163
	nop
	sethi %hi(_outbuf+20),%o0
	ld [%fp-24],%o1
	ld [%o0+%lo(_outbuf+20)],%o2
	sub %o1,%o2,%o0
	ld [%fp-20],%o1
	ld [%o1+24],%o2
	cmp %o0,%o2
	bne L1163
	nop
	ld [%fp-20],%o0
	ld [%o0+8],%o1
	cmp %o1,0
	be L1164
	nop
	ld [%fp-20],%o1
	ld [%o1+16],%o0
	call _strlen,0
	nop
	st %o0,[%fp-36]
	ld [%fp-36],%o0
	ld [%fp-28],%o1
	cmp %o0,%o1
	ble L1165
	nop
	ld [%fp-28],%o0
	mov %o0,%o1
	sll %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-28]
	ld [%fp-32],%o0
	call _xrealloc,0
	nop
	st %o0,[%fp-32]
L1165:
	ld [%fp-20],%o2
	ld [%fp-20],%o3
	ld [%fp-32],%o0
	sethi %hi(LC160),%o4
	or %o4,%lo(LC160),%o1
	ld [%o2+12],%o2
	ld [%o3+16],%o3
	call _sprintf,0
	nop
	sethi %hi(__iob+32),%o0
	mov 18,%o1
	or %o0,%lo(__iob+32),%o2
	add %o1,%o2,%o0
	lduh [%o0],%o1
	sll %o1,16,%o0
	sra %o0,16,%l0
	ld [%fp-32],%o0
	call _strlen,0
	nop
	mov %o0,%o2
	mov %l0,%o0
	ld [%fp-32],%o1
	call _write,0
	nop
	sethi %hi(__iob+32),%o0
	mov 18,%o1
	or %o0,%lo(__iob+32),%o2
	add %o1,%o2,%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	sra %o2,16,%o0
	ld [%fp-20],%o1
	ld [%fp-20],%o2
	ld [%o1],%o1
	ld [%o2+4],%o2
	call _write,0
	nop
L1164:
	ld [%fp-20],%o0
	ld [%o0+20],%o1
	st %o1,[%fp-20]
	b L1166
	nop
L1163:
	ld [%fp-20],%o0
	cmp %o0,0
	be L1167
	nop
	ld [%fp-20],%o0
	sethi %hi(_outbuf+20),%o1
	ld [%fp-24],%o2
	ld [%o1+%lo(_outbuf+20)],%o3
	sub %o2,%o3,%o1
	ld [%o0+24],%o2
	sub %o2,%o1,%o0
	b L1168
	nop
L1167:
	sethi %hi(_outbuf+24),%o1
	ld [%o1+%lo(_outbuf+24)],%o2
	ld [%fp-24],%o1
	sub %o2,%o1,%o0
L1168:
	st %o0,[%fp-36]
	sethi %hi(__iob+32),%o0
	mov 18,%o1
	or %o0,%lo(__iob+32),%o2
	add %o1,%o2,%o0
	lduh [%o0],%o1
	sll %o1,16,%o0
	sra %o0,16,%o1
	mov %o1,%o0
	ld [%fp-24],%o1
	ld [%fp-36],%o2
	call _write,0
	nop
	ld [%fp-24],%o0
	ld [%fp-36],%o1
	add %o0,%o1,%o0
	st %o0,[%fp-24]
L1166:
	b L1160
	nop
L1161:
L1159:
	ret
	restore
	.align 4
	.proc	020
_pass_thru_directive:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	ld [%fp+80],%o0
	ld [%o0],%l0
	ld [%fp+76],%o0
	ld [%fp+76],%o1
	ld [%fp+76],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	ld [%fp+72],%o1
	ld [%fp+68],%o2
	sub %o1,%o2,%o1
	add %o1,1,%o2
	add %l0,%o2,%o1
	cmp %o0,%o1
	bgu L1170
	nop
	ld [%fp+72],%o0
	ld [%fp+68],%o1
	sub %o0,%o1,%o0
	add %o0,1,%o1
	add %l0,%o1,%o2
	ld [%fp+76],%o0
	mov %o2,%o1
	call _grow_outbuf,0
	nop
	b L1171
	nop
L1170:
L1171:
	ld [%fp+76],%o1
	ld [%o1+24],%o0
	add %o0,1,%o2
	st %o2,[%o1+24]
	mov 35,%o1
	stb %o1,[%o0]
	ld [%fp+80],%o0
	ld [%fp+76],%o1
	ld [%o0+8],%o0
	ld [%o1+24],%o1
	mov %l0,%o2
	call _bcopy,0
	nop
	ld [%fp+76],%o0
	ld [%fp+76],%o1
	ld [%o1+24],%o2
	add %o2,%l0,%o1
	st %o1,[%o0+24]
	ld [%fp+72],%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	be L1172
	nop
	ld [%fp+68],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,32
	be L1172
	nop
	ld [%fp+76],%o1
	ld [%o1+24],%o0
	add %o0,1,%o2
	st %o2,[%o1+24]
	mov 32,%o1
	stb %o1,[%o0]
L1172:
	ld [%fp+76],%o1
	ld [%fp+72],%o0
	ld [%fp+68],%o3
	sub %o0,%o3,%o2
	ld [%fp+68],%o0
	ld [%o1+24],%o1
	call _bcopy,0
	nop
	ld [%fp+76],%o0
	ld [%fp+76],%o1
	ld [%fp+72],%o2
	ld [%fp+68],%o3
	sub %o2,%o3,%o2
	ld [%o1+24],%o1
	add %o1,%o2,%o2
	st %o2,[%o0+24]
L1169:
	ret
	restore
	.align 8
LC161:
	.ascii "macro\0"
	.align 8
LC162:
	.ascii "another parameter follows `%s'\0"
	.align 8
LC163:
	.ascii "invalid character in macro parameter name\0"
	.align 8
LC164:
	.ascii "badly punctuated parameter list in `#define'\0"
	.align 8
LC165:
	.ascii "unterminated parameter list in `#define'\0"
	.align 8
LC166:
	.ascii "duplicate argument name `%s' in `#define'\0"
	.align 4
	.proc	010
_create_definition:
	!#PROLOGUE# 0
	save %sp,-184,%sp
	!#PROLOGUE# 1
	ld [%fp+64],%l0
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o0,%o1,%o0
	ld [%o0+12],%o1
	st %o1,[%fp-32]
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o0,%o1,%o0
	ld [%o0+4],%o1
	st %o1,[%fp-36]
	st %g0,[%fp-40]
	st %g0,[%fp-48]
	ld [%fp+68],%o0
	st %o0,[%fp-20]
L1174:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1175
	nop
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L1174
	nop
L1175:
	ld [%fp-20],%o0
	st %o0,[%fp-24]
	ld [%fp-20],%o0
	sethi %hi(LC161),%o2
	or %o2,%lo(LC161),%o1
	call _check_macro_name,0
	nop
	st %o0,[%fp-28]
	ld [%fp-20],%o0
	ld [%fp-28],%o1
	add %o0,%o1,%o0
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,40
	bne L1176
	nop
	st %g0,[%fp-68]
	st %g0,[%fp-72]
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
L1177:
L1180:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1181
	nop
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L1180
	nop
L1181:
L1179:
	b L1178
	nop
	b L1177
	nop
L1178:
	nop
L1182:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,41
	be L1183
	nop
	add %sp,92,%o1
	sub %o1,%sp,%o0
	add %o0,27,%o1
	mov %o1,%o0
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
	st %o1,[%fp-76]
	ld [%fp-76],%o0
	ld [%fp-20],%o1
	st %o1,[%o0+4]
	ld [%fp-76],%o0
	ld [%fp-68],%o1
	st %o1,[%o0]
	ld [%fp-76],%o0
	ld [%fp-72],%o1
	add %o1,1,%o2
	st %o2,[%fp-72]
	st %o1,[%o0+12]
	ld [%fp-76],%o0
	stb %g0,[%o0+16]
	ld [%fp-76],%o0
	st %o0,[%fp-68]
	ld [%fp-40],%o0
	cmp %o0,0
	be L1184
	nop
	sethi %hi(LC162),%o1
	or %o1,%lo(LC162),%o0
	sethi %hi(_rest_extension),%o2
	or %o2,%lo(_rest_extension),%o1
	call _pedwarn,0
	nop
L1184:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idstart),%o2
	or %o2,%lo(_is_idstart),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	bne L1185
	nop
	sethi %hi(LC163),%o1
	or %o1,%lo(LC163),%o0
	call _pedwarn,0
	nop
L1185:
	nop
L1186:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1187
	nop
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	ld [%fp+72],%o0
	ld [%fp-20],%o1
	sub %o0,%o1,%o0
	cmp %o0,3
	bleu L1188
	nop
	sethi %hi(_rest_extension),%o1
	or %o1,%lo(_rest_extension),%o0
	ld [%fp-20],%o1
	mov 3,%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L1188
	nop
	mov 1,%o0
	st %o0,[%fp-40]
	ld [%fp-76],%o0
	mov 1,%o1
	stb %o1,[%o0+16]
	b L1187
	nop
L1188:
	b L1186
	nop
L1187:
	ld [%fp-76],%o0
	ld [%fp-76],%o1
	ld [%fp-20],%o2
	ld [%o1+4],%o1
	sub %o2,%o1,%o2
	st %o2,[%o0+8]
	ld [%fp-40],%o0
	cmp %o0,1
	bne L1189
	nop
	ld [%fp-20],%o0
	add %o0,3,%o1
	st %o1,[%fp-20]
L1189:
	ld [%fp-48],%o1
	add %o1,2,%o0
	ld [%fp-76],%o1
	ld [%o1+8],%o2
	add %o0,%o2,%o0
	st %o0,[%fp-48]
L1190:
L1193:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1194
	nop
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L1193
	nop
L1194:
L1192:
	b L1191
	nop
	b L1190
	nop
L1191:
	ld [%fp-76],%o0
	ld [%o0+8],%o1
	cmp %o1,0
	be L1196
	nop
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,44
	be L1195
	nop
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,41
	bne L1196
	nop
	b L1195
	nop
L1196:
	sethi %hi(LC164),%o1
	or %o1,%lo(LC164),%o0
	call _error,0
	nop
	b L1197
	nop
L1195:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,44
	bne L1198
	nop
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
L1199:
L1202:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1203
	nop
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L1202
	nop
L1203:
L1201:
	b L1200
	nop
	b L1199
	nop
L1200:
L1198:
	ld [%fp-20],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	blu L1204
	nop
	sethi %hi(LC165),%o1
	or %o1,%lo(LC165),%o0
	call _error,0
	nop
	b L1197
	nop
L1204:
	ld [%fp-76],%o0
	ld [%o0],%o1
	st %o1,[%fp-80]
L1205:
	ld [%fp-80],%o0
	cmp %o0,0
	be L1206
	nop
	ld [%fp-76],%o0
	ld [%fp-80],%o1
	ld [%o0+8],%o0
	ld [%o1+8],%o1
	cmp %o0,%o1
	bne L1208
	nop
	ld [%fp-76],%o0
	ld [%fp-80],%o1
	ld [%fp-76],%o2
	ld [%o0+4],%o0
	ld [%o1+4],%o1
	ld [%o2+8],%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L1208
	nop
	ld [%fp-76],%o0
	ld [%o0+8],%o2
	add %o2,1,%o1
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
	st %o1,[%fp-84]
	ld [%fp-76],%o1
	ld [%fp-76],%o2
	ld [%fp-84],%o0
	ld [%o1+4],%o1
	ld [%o2+8],%o2
	call _strncpy,0
	nop
	ld [%fp-76],%o0
	ld [%fp-84],%o1
	ld [%o0+8],%o2
	add %o1,%o2,%o0
	stb %g0,[%o0]
	sethi %hi(LC166),%o1
	or %o1,%lo(LC166),%o0
	ld [%fp-84],%o1
	call _error,0
	nop
	b L1197
	nop
L1208:
L1207:
	ld [%fp-80],%o0
	ld [%o0],%o1
	st %o1,[%fp-80]
	b L1205
	nop
L1206:
	b L1182
	nop
L1183:
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	ld [%fp-20],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	bgeu L1209
	nop
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,32
	be L1210
	nop
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,9
	be L1210
	nop
	b L1209
	nop
L1210:
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
L1209:
	ld [%fp-20],%o0
	ld [%fp+72],%o1
	ld [%fp-72],%o2
	ld [%fp-68],%o3
	call _collect_expansion,0
	nop
	st %o0,[%fp-44]
	ld [%fp-44],%o0
	ldub [%fp-37],%o1
	stb %o1,[%o0+24]
	ld [%fp-48],%o0
	add %o0,1,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	ld [%fp-44],%o1
	st %o0,[%o1+32]
	st %g0,[%fp-80]
	ld [%fp-68],%o0
	st %o0,[%fp-84]
L1211:
	ld [%fp-84],%o0
	cmp %o0,0
	be L1212
	nop
	ld [%fp-84],%o0
	ld [%fp-44],%o1
	ld [%o1+32],%o2
	ld [%fp-80],%o3
	add %o2,%o3,%o1
	ld [%fp-84],%o2
	ld [%o0+4],%o0
	ld [%o2+8],%o2
	call _bcopy,0
	nop
	ld [%fp-84],%o0
	ld [%fp-80],%o1
	ld [%o0+8],%o0
	add %o1,%o0,%o1
	st %o1,[%fp-80]
	ld [%fp-84],%o0
	ld [%o0],%o1
	cmp %o1,0
	be L1214
	nop
	ld [%fp-44],%o0
	ld [%fp-80],%o1
	add %o1,1,%o2
	st %o2,[%fp-80]
	ld [%o0+32],%o2
	add %o1,%o2,%o0
	mov 44,%o1
	stb %o1,[%o0]
	ld [%fp-44],%o0
	ld [%fp-80],%o1
	add %o1,1,%o2
	st %o2,[%fp-80]
	ld [%o0+32],%o2
	add %o1,%o2,%o0
	mov 32,%o1
	stb %o1,[%o0]
L1214:
L1213:
	ld [%fp-84],%o0
	ld [%o0],%o1
	st %o1,[%fp-84]
	b L1211
	nop
L1212:
	ld [%fp-44],%o0
	ld [%o0+32],%o1
	ld [%fp-80],%o2
	add %o1,%o2,%o0
	stb %g0,[%o0]
	b L1215
	nop
L1176:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1216
	nop
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
L1216:
	ld [%fp-20],%o0
	ld [%fp+72],%o1
	mov -1,%o2
	mov 0,%o3
	call _collect_expansion,0
	nop
	st %o0,[%fp-44]
	ld [%fp-44],%o0
	sethi %hi(LC23),%o2
	or %o2,%lo(LC23),%o1
	st %o1,[%o0+32]
L1215:
	ld [%fp-44],%o0
	ld [%fp-32],%o1
	st %o1,[%o0+16]
	ld [%fp-44],%o0
	ld [%fp-36],%o1
	st %o1,[%o0+20]
	ld [%fp-44],%o0
	ld [%fp+76],%o1
	xor %o1,0,%o2
	subcc %g0,%o2,%g0
	subx %g0,-1,%o1
	st %o1,[%o0+8]
	ld [%fp-44],%o0
	st %o0,[%fp-64]
	ld [%fp-24],%o0
	st %o0,[%fp-60]
	ld [%fp-28],%o0
	st %o0,[%fp-56]
	ld [%fp-64],%o0
	st %o0,[%l0]
	ld [%fp-60],%o0
	st %o0,[%l0+4]
	ld [%fp-56],%o0
	st %o0,[%l0+8]
	b L1173
	nop
L1197:
	st %g0,[%fp-64]
	ld [%fp-64],%o0
	st %o0,[%l0]
	ld [%fp-60],%o0
	st %o0,[%l0+4]
	ld [%fp-56],%o0
	st %o0,[%l0+8]
	b L1173
	nop
L1173:
	mov %l0,%i0
	jmp %i7+12
	restore
	.align 8
LC167:
	.ascii "' redefined\0"
	.align 8
LC168:
	.ascii "this is the location of the previous definition\0"
	.align 4
	.proc	04
_do_define:
	!#PROLOGUE# 0
	save %sp,-152,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	sethi %hi(_pcp_outfile),%o0
	ld [%o0+%lo(_pcp_outfile)],%o1
	cmp %o1,0
	be L1218
	nop
	ld [%fp+76],%o0
	cmp %o0,0
	be L1218
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	ld [%fp+80],%o3
	call _pass_thru_directive,0
	nop
L1218:
	add %fp,-40,%o0
	st %o0,[%sp+64]
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	call _create_definition,0
	nop
	unimp 12
	ld [%fp-40],%o0
	cmp %o0,0
	bne L1219
	nop
	b L1220
	nop
L1219:
	ld [%fp-36],%o0
	ld [%fp-32],%o1
	mov 1403,%o2
	call _hashf,0
	nop
	st %o0,[%fp-20]
	ld [%fp-36],%o0
	ld [%fp-32],%o1
	ld [%fp-20],%o2
	call _lookup,0
	nop
	mov %o0,%o1
	st %o1,[%fp-44]
	cmp %o1,0
	be L1221
	nop
	st %g0,[%fp-48]
	ld [%fp-44],%o0
	ld [%o0+12],%o1
	cmp %o1,34
	bne L1222
	nop
	mov 1,%o0
	st %o0,[%fp-48]
	b L1223
	nop
L1222:
	ld [%fp-44],%o0
	ld [%o0+12],%o1
	cmp %o1,31
	bne L1224
	nop
	ld [%fp-44],%o1
	ld [%fp-40],%o0
	ld [%o1+24],%o1
	call _compare_defs,0
	nop
	xor %o0,0,%o1
	subcc %g0,%o1,%g0
	subx %g0,-1,%o0
	st %o0,[%fp-48]
	b L1225
	nop
L1224:
	ld [%fp-44],%o0
	ld [%o0+12],%o1
	cmp %o1,30
	bne L1226
	nop
	sethi %hi(_done_initializing),%o0
	ld [%o0+%lo(_done_initializing)],%o1
	xor %o1,0,%o2
	subcc %g0,%o2,%g0
	subx %g0,-1,%o0
	st %o0,[%fp-48]
L1226:
L1225:
L1223:
	ld [%fp-48],%o0
	cmp %o0,0
	bne L1227
	nop
	sethi %hi(_debug_output),%o0
	ld [%o0+%lo(_debug_output)],%o1
	cmp %o1,0
	be L1228
	nop
	ld [%fp+76],%o0
	cmp %o0,0
	be L1228
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	ld [%fp+80],%o3
	call _pass_thru_directive,0
	nop
L1228:
	ld [%fp-32],%o0
	add %o0,22,%o1
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
	st %o1,[%fp-52]
	ld [%fp-52],%o0
	mov 96,%o1
	stb %o1,[%o0]
	ld [%fp-52],%o0
	add %o0,1,%o1
	ld [%fp-36],%o0
	ld [%fp-32],%o2
	call _bcopy,0
	nop
	ld [%fp-52],%o0
	ld [%fp-32],%o1
	add %o0,%o1,%o0
	add %o0,1,%o1
	mov %o1,%o0
	sethi %hi(LC167),%o2
	or %o2,%lo(LC167),%o1
	call _strcpy,0
	nop
	ld [%fp-52],%o0
	call _pedwarn,0
	nop
	ld [%fp-44],%o0
	ld [%o0+12],%o1
	cmp %o1,31
	bne L1229
	nop
	ld [%fp-44],%o1
	ld [%o1+24],%o0
	ld [%fp-44],%o1
	ld [%o1+24],%o2
	ld [%o0+20],%o0
	ld [%o2+16],%o1
	sethi %hi(LC168),%o3
	or %o3,%lo(LC168),%o2
	call _pedwarn_with_file_and_line,0
	nop
L1229:
L1227:
	ld [%fp-44],%o0
	mov 31,%o1
	st %o1,[%o0+12]
	ld [%fp-44],%o0
	ld [%fp-40],%o1
	st %o1,[%o0+24]
	b L1230
	nop
L1221:
	sethi %hi(_debug_output),%o0
	ld [%o0+%lo(_debug_output)],%o1
	cmp %o1,0
	be L1231
	nop
	ld [%fp+76],%o0
	cmp %o0,0
	be L1231
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	ld [%fp+80],%o3
	call _pass_thru_directive,0
	nop
L1231:
	ld [%fp-36],%o0
	ld [%fp-32],%o1
	mov 31,%o2
	mov 0,%o3
	ld [%fp-40],%o4
	ld [%fp-20],%o5
	call _install,0
	nop
L1230:
	mov 0,%i0
	b L1217
	nop
L1220:
	mov 1,%i0
	b L1217
	nop
L1217:
	ret
	restore
	.align 8
LC169:
	.ascii "invalid %s name\0"
	.align 8
LC170:
	.ascii "invalid %s name `%s'\0"
	.align 8
LC171:
	.ascii "defined\0"
	.align 8
LC172:
	.ascii "invalid %s name `defined'\0"
	.align 4
	.proc	04
_check_macro_name:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
	st %o0,[%fp-20]
L1233:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1234
	nop
L1235:
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L1233
	nop
L1234:
	ld [%fp-20],%o0
	ld [%fp+68],%o1
	sub %o0,%o1,%o0
	st %o0,[%fp-24]
	ld [%fp-24],%o0
	cmp %o0,0
	bne L1236
	nop
	sethi %hi(LC169),%o1
	or %o1,%lo(LC169),%o0
	ld [%fp+72],%o1
	call _error,0
	nop
	b L1237
	nop
L1236:
	ld [%fp+68],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idstart),%o2
	or %o2,%lo(_is_idstart),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	bne L1238
	nop
	ld [%fp-24],%o0
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
	st %o1,[%fp-28]
	ld [%fp+68],%o0
	ld [%fp-28],%o1
	ld [%fp-24],%o2
	call _bcopy,0
	nop
	ld [%fp-28],%o0
	ld [%fp-24],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
	sethi %hi(LC170),%o1
	or %o1,%lo(LC170),%o0
	ld [%fp+72],%o1
	ld [%fp-28],%o2
	call _error,0
	nop
	b L1239
	nop
L1238:
	ld [%fp+68],%o0
	sethi %hi(LC171),%o2
	or %o2,%lo(LC171),%o1
	mov 7,%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L1240
	nop
	ld [%fp-24],%o0
	cmp %o0,7
	bne L1240
	nop
	sethi %hi(LC172),%o1
	or %o1,%lo(LC172),%o0
	ld [%fp+72],%o1
	call _error,0
	nop
L1240:
L1239:
L1237:
	ld [%fp-24],%i0
	b L1232
	nop
L1232:
	ret
	restore
	.align 4
	.proc	04
_compare_defs:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
	ld [%o0+12],%l2
	ld [%fp+72],%o0
	ld [%o0+12],%l3
	mov 1,%o0
	st %o0,[%fp-20]
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%o0],%o0
	ld [%o1],%o1
	cmp %o0,%o1
	be L1242
	nop
	mov 1,%i0
	b L1241
	nop
L1242:
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%o0+32],%o0
	ld [%o1+32],%o1
	call _strcmp,0
	nop
	cmp %o0,0
	be L1243
	nop
	mov 1,%i0
	b L1241
	nop
L1243:
	nop
	ld [%fp+68],%o0
	ld [%o0+28],%l0
	ld [%fp+72],%o0
	ld [%o0+28],%l1
L1244:
	cmp %l0,0
	be L1245
	nop
	cmp %l1,0
	be L1245
	nop
	ld [%l0+8],%o0
	ld [%l1+8],%o1
	cmp %o0,%o1
	bne L1250
	nop
	mov %l2,%o0
	mov %l3,%o1
	ld [%l0+8],%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L1250
	nop
	b L1249
	nop
L1250:
	ld [%fp-20],%o0
	mov %l2,%o1
	ld [%l0+8],%o2
	mov %l3,%o3
	ld [%l1+8],%o4
	mov 0,%o5
	call _comp_def_part,0
	nop
	cmp %o0,0
	bne L1248
	nop
	b L1249
	nop
L1249:
	ld [%l0+12],%o0
	ld [%l1+12],%o1
	cmp %o0,%o1
	bne L1248
	nop
	ldub [%l0+4],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	ldub [%l1+4],%o1
	sll %o1,24,%o2
	sra %o2,24,%o1
	cmp %o0,%o1
	bne L1248
	nop
	ldub [%l0+5],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	ldub [%l1+5],%o1
	sll %o1,24,%o2
	sra %o2,24,%o1
	cmp %o0,%o1
	bne L1248
	nop
	ldub [%l0+6],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	ldub [%l1+6],%o1
	sll %o1,24,%o2
	sra %o2,24,%o1
	cmp %o0,%o1
	bne L1248
	nop
	b L1247
	nop
L1248:
	mov 1,%i0
	b L1241
	nop
L1247:
	st %g0,[%fp-20]
	ld [%l0+8],%o0
	add %l2,%o0,%l2
	ld [%l1+8],%o0
	add %l3,%o0,%l3
L1246:
	ld [%l0],%l0
	ld [%l1],%l1
	b L1244
	nop
L1245:
	cmp %l0,%l1
	be L1251
	nop
	mov 1,%i0
	b L1241
	nop
L1251:
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o1+12],%o2
	sub %l2,%o2,%o1
	ld [%o0+4],%o0
	sub %o0,%o1,%o2
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%o1+12],%o3
	sub %l3,%o3,%o1
	ld [%o0+4],%o0
	sub %o0,%o1,%o4
	ld [%fp-20],%o0
	mov %l2,%o1
	mov %l3,%o3
	mov 1,%o5
	call _comp_def_part,0
	nop
	cmp %o0,0
	be L1252
	nop
	mov 1,%i0
	b L1241
	nop
L1252:
	mov 0,%i0
	b L1241
	nop
L1241:
	ret
	restore
	.align 4
	.proc	04
_comp_def_part:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	st %i4,[%fp+84]
	st %i5,[%fp+88]
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	add %o1,%o2,%o0
	ld [%fp+80],%o2
	ld [%fp+84],%o3
	add %o2,%o3,%o1
	ld [%fp+68],%o2
	cmp %o2,0
	be L1254
	nop
L1255:
	ld [%fp+72],%o2
	cmp %o2,%o0
	be L1256
	nop
	ld [%fp+72],%o2
	ldub [%o2],%o3
	and %o3,0xff,%o2
	sethi %hi(_is_space),%o4
	or %o4,%lo(_is_space),%o3
	ldub [%o2+%o3],%o4
	and %o4,0xff,%o2
	cmp %o2,0
	be L1256
	nop
	ld [%fp+72],%o3
	add %o3,1,%o2
	mov %o2,%o3
	st %o3,[%fp+72]
	b L1255
	nop
L1256:
	nop
L1257:
	ld [%fp+80],%o2
	cmp %o2,%o1
	be L1258
	nop
	ld [%fp+80],%o2
	ldub [%o2],%o3
	and %o3,0xff,%o2
	sethi %hi(_is_space),%o4
	or %o4,%lo(_is_space),%o3
	ldub [%o2+%o3],%o4
	and %o4,0xff,%o2
	cmp %o2,0
	be L1258
	nop
	ld [%fp+80],%o3
	add %o3,1,%o2
	mov %o2,%o3
	st %o3,[%fp+80]
	b L1257
	nop
L1258:
L1254:
	ld [%fp+88],%o2
	cmp %o2,0
	be L1259
	nop
L1260:
	ld [%fp+72],%o2
	cmp %o2,%o0
	be L1261
	nop
	ldub [%o0-1],%o3
	and %o3,0xff,%o2
	sethi %hi(_is_space),%o4
	or %o4,%lo(_is_space),%o3
	ldub [%o2+%o3],%o4
	and %o4,0xff,%o2
	cmp %o2,0
	be L1261
	nop
	add %o0,-1,%o0
	b L1260
	nop
L1261:
	nop
L1262:
	ld [%fp+80],%o2
	cmp %o2,%o1
	be L1263
	nop
	ldub [%o1-1],%o3
	and %o3,0xff,%o2
	sethi %hi(_is_space),%o4
	or %o4,%lo(_is_space),%o3
	ldub [%o2+%o3],%o4
	and %o4,0xff,%o2
	cmp %o2,0
	be L1263
	nop
	add %o1,-1,%o1
	b L1262
	nop
L1263:
L1259:
	nop
L1264:
	ld [%fp+72],%o2
	cmp %o2,%o0
	be L1265
	nop
	ld [%fp+80],%o2
	cmp %o2,%o1
	be L1265
	nop
	ld [%fp+72],%o2
	ldub [%o2],%o3
	and %o3,0xff,%o2
	sethi %hi(_is_space),%o4
	or %o4,%lo(_is_space),%o3
	ldub [%o2+%o3],%o4
	and %o4,0xff,%o2
	cmp %o2,0
	be L1266
	nop
	ld [%fp+80],%o2
	ldub [%o2],%o3
	and %o3,0xff,%o2
	sethi %hi(_is_space),%o4
	or %o4,%lo(_is_space),%o3
	ldub [%o2+%o3],%o4
	and %o4,0xff,%o2
	cmp %o2,0
	be L1266
	nop
L1267:
	ld [%fp+72],%o2
	cmp %o2,%o0
	be L1268
	nop
	ld [%fp+72],%o2
	ldub [%o2],%o3
	and %o3,0xff,%o2
	sethi %hi(_is_space),%o4
	or %o4,%lo(_is_space),%o3
	ldub [%o2+%o3],%o4
	and %o4,0xff,%o2
	cmp %o2,0
	be L1268
	nop
	ld [%fp+72],%o3
	add %o3,1,%o2
	mov %o2,%o3
	st %o3,[%fp+72]
	b L1267
	nop
L1268:
	nop
L1269:
	ld [%fp+80],%o2
	cmp %o2,%o1
	be L1270
	nop
	ld [%fp+80],%o2
	ldub [%o2],%o3
	and %o3,0xff,%o2
	sethi %hi(_is_space),%o4
	or %o4,%lo(_is_space),%o3
	ldub [%o2+%o3],%o4
	and %o4,0xff,%o2
	cmp %o2,0
	be L1270
	nop
	ld [%fp+80],%o3
	add %o3,1,%o2
	mov %o2,%o3
	st %o3,[%fp+80]
	b L1269
	nop
L1270:
	b L1271
	nop
L1266:
	ld [%fp+72],%o3
	ld [%fp+80],%o2
	ldub [%o3],%o4
	and %o4,0xff,%o3
	ldub [%o2],%o4
	and %o4,0xff,%o2
	cmp %o3,%o2
	bne L1272
	nop
	ld [%fp+72],%o3
	add %o3,1,%o2
	mov %o2,%o3
	st %o3,[%fp+72]
	ld [%fp+80],%o3
	add %o3,1,%o2
	mov %o2,%o3
	st %o3,[%fp+80]
	b L1273
	nop
L1272:
	b L1265
	nop
L1273:
L1271:
	b L1264
	nop
L1265:
	mov 0,%o2
	ld [%fp+72],%o3
	cmp %o3,%o0
	bne L1275
	nop
	ld [%fp+80],%o3
	cmp %o3,%o1
	bne L1275
	nop
	b L1274
	nop
L1275:
	mov 1,%o2
L1274:
	mov %o2,%i0
	b L1253
	nop
L1253:
	ret
	restore
	.align 8
LC173:
	.ascii "`##' at start of macro definition\0"
	.align 8
LC174:
	.ascii "`##' at end of macro definition\0"
	.align 8
LC175:
	.ascii "`#' operator is not followed by a macro argument name\0"
	.align 8
LC176:
	.ascii "macro argument `%.*s' is stringified.\0"
	.align 8
LC177:
	.ascii "macro arg `%.*s' would be stringified with -traditional.\0"
	.align 8
LC178:
	.ascii "`#' operator should be followed by a macro argument name\0"
	.align 4
	.proc	0110
_collect_expansion:
	!#PROLOGUE# 0
	save %sp,-152,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	st %g0,[%fp-24]
	st %g0,[%fp-28]
	st %g0,[%fp-32]
	st %g0,[%fp-40]
	ld [%fp+72],%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	bgeu L1277
	nop
	call _abort,0
	nop
L1277:
	ld [%fp+72],%l1
	ld [%fp+68],%l0
L1278:
	cmp %l0,%l1
	bgeu L1279
	nop
	ldub [%l1-1],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_space),%o2
	or %o2,%lo(_is_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1279
	nop
	add %l1,-1,%l1
	b L1278
	nop
L1279:
	nop
L1280:
	cmp %l0,%l1
	bgeu L1281
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_space),%o2
	or %o2,%lo(_is_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1281
	nop
	add %l0,1,%l0
	b L1280
	nop
L1281:
	ld [%fp+72],%o1
	sub %o1,%l1,%o0
	mov %o0,%o1
	sll %o1,1,%o0
	ld [%fp+68],%o2
	sub %l0,%o2,%o1
	mov %o1,%o2
	sll %o2,1,%o1
	add %o1,36,%o2
	add %o0,%o2,%o0
	sub %l1,%l0,%o1
	add %o0,%o1,%o0
	add %o0,3,%o1
	st %o1,[%fp-36]
	mov 1,%o0
	ld [%fp-36],%o1
	call _xcalloc,0
	nop
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	ld [%fp+76],%o1
	st %o1,[%o0]
	ld [%fp-20],%o0
	ld [%fp-20],%o2
	add %o2,36,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	mov %o2,%l3
	mov %l3,%l2
	ld [%fp+68],%l0
L1282:
	cmp %l0,%l1
	bgeu L1283
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_space),%o2
	or %o2,%lo(_is_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1283
	nop
	mov 10,%o0
	stb %o0,[%l3]
	add %l3,1,%l3
	ldub [%l0],%o0
	stb %o0,[%l3]
	add %l0,1,%l0
	add %l3,1,%l3
	b L1282
	nop
L1283:
	sub %l1,%l0,%o0
	cmp %o0,1
	ble L1284
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,35
	bne L1284
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,35
	bne L1284
	nop
	sethi %hi(LC173),%o1
	or %o1,%lo(LC173),%o0
	call _error,0
	nop
	add %l0,2,%l0
L1284:
	nop
L1285:
	cmp %l0,%l1
	bgeu L1286
	nop
	st %g0,[%fp-44]
	ldub [%l0],%l4
	add %l0,1,%l0
	stb %l4,[%l3]
	add %l3,1,%l3
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	bne L1287
	nop
	and %l4,0xff,%o0
	cmp %o0,35
	be L1298
	nop
	cmp %o0,35
	bg L1320
	nop
	cmp %o0,34
	be L1290
	nop
	b L1319
	nop
L1320:
	cmp %o0,39
	be L1289
	nop
	cmp %o0,92
	be L1294
	nop
	b L1319
	nop
L1289:
L1290:
	ld [%fp-40],%o0
	cmp %o0,0
	be L1291
	nop
	and %l4,0xff,%o0
	ld [%fp-40],%o1
	cmp %o0,%o1
	bne L1292
	nop
	st %g0,[%fp-40]
L1292:
	b L1293
	nop
L1291:
	and %l4,0xff,%o0
	st %o0,[%fp-40]
L1293:
	b L1288
	nop
L1294:
	cmp %l0,%l1
	bgeu L1295
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,35
	bne L1295
	nop
	ld [%fp-40],%o0
	cmp %o0,0
	bne L1295
	nop
	add %l3,-1,%l3
	ldub [%l0],%o0
	stb %o0,[%l3]
	add %l0,1,%l0
	add %l3,1,%l3
	b L1296
	nop
L1295:
	cmp %l0,%l1
	bgeu L1297
	nop
	ld [%fp-40],%o0
	cmp %o0,0
	be L1297
	nop
	ldub [%l0],%o0
	stb %o0,[%l3]
	add %l0,1,%l0
	add %l3,1,%l3
L1297:
L1296:
	b L1288
	nop
L1298:
	ld [%fp-40],%o0
	cmp %o0,0
	be L1299
	nop
	b L1288
	nop
L1299:
	cmp %l0,%l1
	bgeu L1300
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,35
	bne L1300
	nop
	add %l3,-1,%l3
L1301:
	cmp %l3,%l2
	bleu L1302
	nop
	ldub [%l3-1],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1302
	nop
	add %l3,-1,%l3
	b L1301
	nop
L1302:
	add %l0,1,%l0
L1303:
L1306:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1307
	nop
	add %l0,1,%l0
	b L1306
	nop
L1307:
L1305:
	b L1304
	nop
	b L1303
	nop
L1304:
	st %l0,[%fp-28]
	cmp %l0,%l1
	bne L1308
	nop
	sethi %hi(LC174),%o1
	or %o1,%lo(LC174),%o0
	call _error,0
	nop
L1308:
	b L1309
	nop
L1300:
	add %l3,-1,%l3
L1310:
L1313:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1314
	nop
	add %l0,1,%l0
	b L1313
	nop
L1314:
L1312:
	b L1311
	nop
	b L1310
	nop
L1311:
	cmp %l0,%l1
	be L1316
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idstart),%o2
	or %o2,%lo(_is_idstart),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1316
	nop
	ld [%fp+76],%o0
	cmp %o0,0
	ble L1316
	nop
	b L1315
	nop
L1316:
	sethi %hi(LC175),%o1
	or %o1,%lo(LC175),%o0
	call _error,0
	nop
	b L1317
	nop
L1315:
	st %l0,[%fp-32]
L1317:
L1309:
	b L1288
	nop
L1319:
L1288:
	b L1321
	nop
L1287:
	and %l4,0xff,%o0
	cmp %o0,39
	be L1323
	nop
	cmp %o0,39
	bg L1339
	nop
	cmp %o0,34
	be L1324
	nop
	b L1338
	nop
L1339:
	cmp %o0,47
	be L1331
	nop
	cmp %o0,92
	be L1328
	nop
	b L1338
	nop
L1323:
L1324:
	ld [%fp-40],%o0
	cmp %o0,0
	be L1325
	nop
	and %l4,0xff,%o0
	ld [%fp-40],%o1
	cmp %o0,%o1
	bne L1326
	nop
	st %g0,[%fp-40]
L1326:
	b L1327
	nop
L1325:
	and %l4,0xff,%o0
	st %o0,[%fp-40]
L1327:
	b L1322
	nop
L1328:
	ld [%fp-40],%o0
	cmp %o0,0
	be L1329
	nop
	cmp %l0,%l1
	bgeu L1329
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	ld [%fp-40],%o1
	cmp %o0,%o1
	be L1330
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	be L1330
	nop
	b L1329
	nop
L1330:
	ldub [%l0],%o0
	stb %o0,[%l3]
	add %l0,1,%l0
	add %l3,1,%l3
	b L1285
	nop
L1329:
	b L1322
	nop
L1331:
	ld [%fp-40],%o0
	cmp %o0,0
	be L1332
	nop
	b L1322
	nop
L1332:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L1333
	nop
	add %l3,-1,%l3
	add %l0,1,%l0
L1334:
	cmp %l0,%l1
	bgeu L1335
	nop
	ldub [%l0-2],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L1336
	nop
	ldub [%l0-1],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L1336
	nop
	b L1335
	nop
L1336:
	add %l0,1,%l0
	b L1334
	nop
L1335:
L1333:
	b L1322
	nop
L1338:
L1322:
L1321:
	and %l4,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1340
	nop
	ld [%fp+76],%o0
	cmp %o0,0
	ble L1340
	nop
	add %l0,-1,%o0
	st %o0,[%fp-48]
	add %l3,-1,%l3
L1341:
	cmp %l0,%l1
	be L1342
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1342
	nop
	add %l0,1,%l0
	b L1341
	nop
L1342:
	ld [%fp-48],%o0
	sub %l0,%o0,%o1
	st %o1,[%fp-52]
	and %l4,0xff,%o0
	sethi %hi(_is_idstart),%o2
	or %o2,%lo(_is_idstart),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1343
	nop
	ld [%fp+80],%l5
L1344:
	cmp %l5,0
	be L1345
	nop
	ld [%l5+4],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	and %l4,0xff,%o1
	cmp %o0,%o1
	bne L1347
	nop
	ld [%l5+8],%o0
	ld [%fp-52],%o1
	cmp %o0,%o1
	bne L1347
	nop
	ld [%l5+4],%o0
	ld [%fp-48],%o1
	ld [%fp-52],%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L1347
	nop
	ld [%fp-40],%o0
	cmp %o0,0
	be L1348
	nop
	sethi %hi(_warn_stringify),%o0
	ld [%o0+%lo(_warn_stringify)],%o1
	cmp %o1,0
	be L1348
	nop
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	be L1349
	nop
	sethi %hi(LC176),%o1
	or %o1,%lo(LC176),%o0
	ld [%fp-52],%o1
	ld [%l5+4],%o2
	call _warning,0
	nop
	b L1350
	nop
L1349:
	sethi %hi(LC177),%o1
	or %o1,%lo(LC177),%o0
	ld [%fp-52],%o1
	ld [%l5+4],%o2
	call _warning,0
	nop
L1350:
L1348:
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	bne L1351
	nop
	ld [%fp-40],%o0
	cmp %o0,0
	be L1351
	nop
	b L1345
	nop
L1351:
	mov 16,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-56]
	ld [%fp-56],%o0
	st %g0,[%o0]
	ld [%fp-56],%o0
	ld [%fp-28],%o1
	ld [%fp-48],%o2
	xor %o1,%o2,%o3
	subcc %g0,%o3,%g0
	subx %g0,-1,%o1
	stb %o1,[%o0+5]
	ld [%fp-56],%o0
	stb %g0,[%o0+6]
	ld [%fp-56],%o0
	ldub [%l5+16],%o1
	stb %o1,[%o0+7]
	ld [%fp-56],%o0
	sethi %hi(_traditional),%o1
	ld [%o1+%lo(_traditional)],%o2
	cmp %o2,0
	be L1352
	nop
	ld [%fp-40],%o1
	xor %o1,0,%o3
	subcc %g0,%o3,%g0
	addx %g0,0,%o2
	mov %o2,%o1
	b L1353
	nop
L1352:
	ld [%fp-32],%o2
	ld [%fp-48],%o3
	xor %o2,%o3,%o4
	subcc %g0,%o4,%g0
	subx %g0,-1,%o2
	mov %o2,%o1
L1353:
	stb %o1,[%o0+4]
	ld [%fp-24],%o0
	cmp %o0,0
	bne L1354
	nop
	ld [%fp-20],%o0
	ld [%fp-56],%o1
	st %o1,[%o0+28]
	b L1355
	nop
L1354:
	ld [%fp-24],%o0
	ld [%fp-56],%o1
	st %o1,[%o0]
L1355:
	ld [%fp-56],%o0
	st %o0,[%fp-24]
	ld [%fp-56],%o0
	ld [%l5+12],%o1
	st %o1,[%o0+12]
	ld [%fp-56],%o0
	sub %l3,%l2,%o1
	st %o1,[%o0+8]
	mov %l0,%o0
L1356:
L1359:
	ldub [%o0],%o2
	and %o2,0xff,%o1
	sethi %hi(_is_hor_space),%o3
	or %o3,%lo(_is_hor_space),%o2
	ldub [%o1+%o2],%o3
	and %o3,0xff,%o1
	cmp %o1,0
	be L1360
	nop
	add %o0,1,%o0
	b L1359
	nop
L1360:
L1358:
	b L1357
	nop
	b L1356
	nop
L1357:
	add %o0,2,%o1
	cmp %o1,%l1
	bgu L1361
	nop
	ldub [%o0],%o2
	and %o2,0xff,%o1
	cmp %o1,35
	bne L1361
	nop
	ldub [%o0+1],%o2
	and %o2,0xff,%o1
	cmp %o1,35
	bne L1361
	nop
	ld [%fp-56],%o1
	mov 1,%o2
	stb %o2,[%o1+6]
L1361:
	mov %l3,%l2
	mov 1,%o0
	st %o0,[%fp-44]
	b L1345
	nop
L1347:
L1346:
	ld [%l5],%l5
	b L1344
	nop
L1345:
L1343:
	ld [%fp-44],%o0
	cmp %o0,0
	bne L1362
	nop
	mov %l0,%l5
	ld [%fp-48],%l0
L1363:
	cmp %l0,%l5
	be L1364
	nop
	ldub [%l0],%o0
	stb %o0,[%l3]
	add %l0,1,%l0
	add %l3,1,%l3
	b L1363
	nop
L1364:
	ld [%fp-32],%o0
	ld [%fp-48],%o1
	cmp %o0,%o1
	bne L1365
	nop
	sethi %hi(LC178),%o1
	or %o1,%lo(LC178),%o0
	call _error,0
	nop
L1365:
L1362:
L1340:
	b L1285
	nop
L1286:
	ld [%fp+72],%o0
	cmp %l1,%o0
	bgeu L1366
	nop
L1367:
	ld [%fp+72],%o0
	cmp %l1,%o0
	bgeu L1368
	nop
	ldub [%l1],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_space),%o2
	or %o2,%lo(_is_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1368
	nop
	mov 10,%o0
	stb %o0,[%l3]
	add %l3,1,%l3
	ldub [%l1],%o0
	stb %o0,[%l3]
	add %l1,1,%l1
	add %l3,1,%l3
	b L1367
	nop
L1368:
	b L1369
	nop
L1366:
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	bne L1370
	nop
	ld [%fp-40],%o0
	cmp %o0,0
	bne L1370
	nop
	mov 10,%o0
	stb %o0,[%l3]
	add %l3,1,%l3
	mov 32,%o0
	stb %o0,[%l3]
	add %l3,1,%l3
L1370:
L1369:
	stb %g0,[%l3]
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o1+12],%o2
	sub %l3,%o2,%o1
	st %o1,[%o0+4]
	ld [%fp-20],%o0
	ld [%o0+4],%o1
	add %o1,1,%o0
	ld [%fp-36],%o1
	cmp %o0,%o1
	ble L1371
	nop
	call _abort,0
	nop
L1371:
	ld [%fp-20],%i0
	b L1276
	nop
L1276:
	ret
	restore
	.align 8
LC179:
	.ascii "ANSI C does not allow `#assert'\0"
	.align 8
LC180:
	.ascii "assertion\0"
	.align 8
LC181:
	.ascii "missing token-sequence in `#assert'\0"
	.align 8
LC182:
	.ascii "empty token-sequence in `#assert'\0"
	.align 8
LC183:
	.ascii "`defined' redefined as assertion\0"
	.align 4
	.proc	04
_do_assert:
	!#PROLOGUE# 0
	save %sp,-144,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	st %g0,[%fp-32]
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L1373
	nop
	sethi %hi(_done_initializing),%o0
	ld [%o0+%lo(_done_initializing)],%o1
	cmp %o1,0
	be L1373
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o0,%o1,%o0
	ldub [%o0+40],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	bne L1373
	nop
	sethi %hi(LC179),%o1
	or %o1,%lo(LC179),%o0
	call _pedwarn,0
	nop
L1373:
	ld [%fp+68],%o0
	st %o0,[%fp-20]
L1374:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1375
	nop
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L1374
	nop
L1375:
	ld [%fp-20],%o0
	st %o0,[%fp-24]
	ld [%fp-20],%o0
	sethi %hi(LC180),%o2
	or %o2,%lo(LC180),%o1
	call _check_macro_name,0
	nop
	st %o0,[%fp-28]
	ld [%fp-20],%o0
	ld [%fp-28],%o1
	add %o0,%o1,%o0
	st %o0,[%fp-20]
L1376:
L1379:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1380
	nop
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L1379
	nop
L1380:
L1378:
	b L1377
	nop
	b L1376
	nop
L1377:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,40
	be L1381
	nop
	sethi %hi(LC181),%o1
	or %o1,%lo(LC181),%o0
	call _error,0
	nop
	mov 1,%i0
	b L1372
	nop
L1381:
	st %g0,[%fp-36]
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
L1382:
L1385:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1386
	nop
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L1385
	nop
L1386:
L1384:
	b L1383
	nop
	b L1382
	nop
L1383:
	add %fp,-20,%o0
	add %fp,-36,%o2
	ld [%fp+72],%o1
	call _read_token_list,0
	nop
	st %o0,[%fp-32]
	ld [%fp-36],%o0
	cmp %o0,0
	be L1387
	nop
	mov 1,%i0
	b L1372
	nop
L1387:
	ld [%fp-32],%o0
	cmp %o0,0
	bne L1388
	nop
	sethi %hi(LC182),%o1
	or %o1,%lo(LC182),%o0
	call _error,0
	nop
	mov 1,%i0
	b L1372
	nop
L1388:
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
L1389:
L1392:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1393
	nop
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L1392
	nop
L1393:
L1391:
	b L1390
	nop
	b L1389
	nop
L1390:
	ld [%fp-24],%o0
	ld [%fp-28],%o1
	mov 37,%o2
	call _hashf,0
	nop
	st %o0,[%fp-40]
	mov 8,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-44]
	ld [%fp-24],%o0
	ld [%fp-28],%o1
	ld [%fp-40],%o2
	call _assertion_lookup,0
	nop
	st %o0,[%fp-36]
	ld [%fp-36],%o0
	cmp %o0,0
	bne L1394
	nop
	ld [%fp-28],%o0
	cmp %o0,7
	bne L1395
	nop
	ld [%fp-24],%o0
	sethi %hi(LC171),%o2
	or %o2,%lo(LC171),%o1
	ld [%fp-28],%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L1395
	nop
	sethi %hi(LC183),%o1
	or %o1,%lo(LC183),%o0
	call _error,0
	nop
L1395:
	ld [%fp-24],%o0
	ld [%fp-28],%o1
	ld [%fp-40],%o2
	call _assertion_install,0
	nop
	st %o0,[%fp-36]
L1394:
	ld [%fp-44],%o0
	ld [%fp-32],%o1
	st %o1,[%o0+4]
	ld [%fp-44],%o0
	ld [%fp-36],%o1
	ld [%o1+20],%o2
	st %o2,[%o0]
	ld [%fp-36],%o0
	ld [%fp-44],%o1
	st %o1,[%o0+20]
	mov 0,%i0
	b L1372
	nop
L1372:
	ret
	restore
	.align 8
LC184:
	.ascii "ANSI C does not allow `#unassert'\0"
	.align 8
LC185:
	.ascii "empty token list in `#unassert'\0"
	.align 4
	.proc	04
_do_unassert:
	!#PROLOGUE# 0
	save %sp,-152,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	st %g0,[%fp-32]
	st %g0,[%fp-36]
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L1397
	nop
	sethi %hi(_done_initializing),%o0
	ld [%o0+%lo(_done_initializing)],%o1
	cmp %o1,0
	be L1397
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o0,%o1,%o0
	ldub [%o0+40],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	bne L1397
	nop
	sethi %hi(LC184),%o1
	or %o1,%lo(LC184),%o0
	call _pedwarn,0
	nop
L1397:
	ld [%fp+68],%o0
	st %o0,[%fp-20]
L1398:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1399
	nop
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L1398
	nop
L1399:
	ld [%fp-20],%o0
	st %o0,[%fp-24]
	ld [%fp-20],%o0
	sethi %hi(LC180),%o2
	or %o2,%lo(LC180),%o1
	call _check_macro_name,0
	nop
	st %o0,[%fp-28]
	ld [%fp-20],%o0
	ld [%fp-28],%o1
	add %o0,%o1,%o0
	st %o0,[%fp-20]
L1400:
L1403:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1404
	nop
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L1403
	nop
L1404:
L1402:
	b L1401
	nop
	b L1400
	nop
L1401:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,40
	bne L1405
	nop
	st %g0,[%fp-40]
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
L1406:
L1409:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1410
	nop
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L1409
	nop
L1410:
L1408:
	b L1407
	nop
	b L1406
	nop
L1407:
	add %fp,-20,%o0
	add %fp,-40,%o2
	ld [%fp+72],%o1
	call _read_token_list,0
	nop
	st %o0,[%fp-32]
	ld [%fp-40],%o0
	cmp %o0,0
	be L1411
	nop
	mov 1,%i0
	b L1396
	nop
L1411:
	ld [%fp-32],%o0
	cmp %o0,0
	bne L1412
	nop
	sethi %hi(LC185),%o1
	or %o1,%lo(LC185),%o0
	call _error,0
	nop
	mov 1,%i0
	b L1396
	nop
L1412:
	mov 1,%o0
	st %o0,[%fp-36]
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
L1413:
L1416:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1417
	nop
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L1416
	nop
L1417:
L1415:
	b L1414
	nop
	b L1413
	nop
L1414:
L1405:
	ld [%fp-24],%o0
	ld [%fp-28],%o1
	mov 37,%o2
	call _hashf,0
	nop
	st %o0,[%fp-44]
	ld [%fp-24],%o0
	ld [%fp-28],%o1
	ld [%fp-44],%o2
	call _assertion_lookup,0
	nop
	st %o0,[%fp-40]
	ld [%fp-40],%o0
	cmp %o0,0
	bne L1418
	nop
	mov 1,%i0
	b L1396
	nop
L1418:
	ld [%fp-36],%o0
	cmp %o0,0
	bne L1419
	nop
	ld [%fp-40],%o0
	ld [%o0+20],%o1
	st %o1,[%fp-48]
L1420:
	ld [%fp-48],%o0
	cmp %o0,0
	be L1421
	nop
	ld [%fp-48],%o0
	ld [%o0],%o1
	st %o1,[%fp-56]
	ld [%fp-48],%o1
	ld [%o1+4],%o0
	call _free_token_list,0
	nop
	ld [%fp-48],%o0
	call _free,0
	nop
L1422:
	ld [%fp-56],%o0
	st %o0,[%fp-48]
	b L1420
	nop
L1421:
	ld [%fp-40],%o0
	call _delete_assertion,0
	nop
	b L1423
	nop
L1419:
	ld [%fp-40],%o0
	ld [%o0+20],%o1
	st %o1,[%fp-48]
	st %g0,[%fp-52]
L1424:
	ld [%fp-48],%o0
	cmp %o0,0
	be L1425
	nop
	ld [%fp-48],%o0
	ld [%o0],%o1
	st %o1,[%fp-56]
	ld [%fp-48],%o1
	ld [%o1+4],%o0
	ld [%fp-32],%o1
	call _compare_token_lists,0
	nop
	cmp %o0,0
	be L1426
	nop
	ld [%fp-52],%o0
	cmp %o0,0
	be L1427
	nop
	ld [%fp-52],%o0
	ld [%fp-56],%o1
	st %o1,[%o0]
	b L1428
	nop
L1427:
	ld [%fp-40],%o0
	ld [%fp-48],%o1
	ld [%o1],%o2
	st %o2,[%o0+20]
L1428:
	ld [%fp-48],%o1
	ld [%o1+4],%o0
	call _free_token_list,0
	nop
	ld [%fp-48],%o0
	call _free,0
	nop
	b L1429
	nop
L1426:
	ld [%fp-48],%o0
	st %o0,[%fp-52]
L1429:
	ld [%fp-56],%o0
	st %o0,[%fp-48]
	b L1424
	nop
L1425:
L1423:
	mov 0,%i0
	b L1396
	nop
L1396:
	ret
	restore
	.align 8
LC186:
	.ascii "ANSI C does not allow testing assertions\0"
	.align 4
	.global _check_assertion
	.proc	04
_check_assertion:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	mov 37,%o2
	call _hashf,0
	nop
	st %o0,[%fp-24]
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L1431
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o0,%o1,%o0
	ldub [%o0+40],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	bne L1431
	nop
	sethi %hi(LC186),%o1
	or %o1,%lo(LC186),%o0
	call _pedwarn,0
	nop
L1431:
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%fp-24],%o2
	call _assertion_lookup,0
	nop
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	cmp %o0,0
	bne L1432
	nop
	mov 0,%i0
	b L1430
	nop
L1432:
	ld [%fp+76],%o0
	cmp %o0,0
	bne L1433
	nop
	mov 1,%i0
	b L1430
	nop
L1433:
	ld [%fp-20],%o0
	ld [%o0+20],%o1
	st %o1,[%fp-28]
L1434:
	ld [%fp-28],%o0
	cmp %o0,0
	be L1435
	nop
	ld [%fp-28],%o1
	ld [%o1+4],%o0
	ld [%fp+80],%o1
	call _compare_token_lists,0
	nop
	cmp %o0,0
	be L1436
	nop
	mov 1,%i0
	b L1430
	nop
L1436:
	ld [%fp-28],%o0
	ld [%o0],%o1
	st %o1,[%fp-28]
	b L1434
	nop
L1435:
	mov 0,%i0
	b L1430
	nop
L1430:
	ret
	restore
	.align 4
	.proc	04
_compare_token_lists:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
L1438:
	ld [%fp+68],%o0
	cmp %o0,0
	be L1439
	nop
	ld [%fp+72],%o0
	cmp %o0,0
	be L1439
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%o0+8],%o0
	ld [%o1+8],%o1
	cmp %o0,%o1
	be L1440
	nop
	mov 0,%i0
	b L1437
	nop
L1440:
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%fp+68],%o2
	ld [%o0+4],%o0
	ld [%o1+4],%o1
	ld [%o2+8],%o2
	call _strncmp,0
	nop
	cmp %o0,0
	be L1441
	nop
	mov 0,%i0
	b L1437
	nop
L1441:
	ld [%fp+68],%o0
	ld [%o0],%o1
	st %o1,[%fp+68]
	ld [%fp+72],%o0
	ld [%o0],%o1
	st %o1,[%fp+72]
	b L1438
	nop
L1439:
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	xor %o0,%o1,%o2
	subcc %g0,%o2,%g0
	subx %g0,-1,%o0
	mov %o0,%i0
	b L1437
	nop
L1437:
	ret
	restore
	.align 8
LC187:
	.ascii "unterminated token sequence in `#assert' or `#unassert'\0"
	.align 4
	.proc	0110
_read_token_list:
	!#PROLOGUE# 0
	save %sp,-136,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %g0,[%fp-20]
	ld [%fp+68],%o0
	ld [%o0],%o1
	st %o1,[%fp-24]
	mov 1,%o0
	st %o0,[%fp-28]
	ld [%fp+76],%o0
	st %g0,[%o0]
L1443:
	ld [%fp-28],%o0
	cmp %o0,0
	ble L1444
	nop
	st %g0,[%fp-36]
	ld [%fp-24],%o0
	st %o0,[%fp-40]
	ld [%fp-24],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,40
	bne L1445
	nop
	ld [%fp-24],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
	ld [%fp-28],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-28]
	b L1446
	nop
L1445:
	ld [%fp-24],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,41
	bne L1447
	nop
	ld [%fp-28],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-28]
	ld [%fp-28],%o0
	cmp %o0,0
	bne L1448
	nop
	b L1444
	nop
L1448:
	ld [%fp-24],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
	b L1449
	nop
L1447:
	ld [%fp-24],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,34
	be L1451
	nop
	ld [%fp-24],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,39
	be L1451
	nop
	b L1450
	nop
L1451:
	add %fp,-36,%o5
	ld [%fp-24],%o0
	ld [%fp+72],%o1
	mov 0,%o2
	mov 0,%o3
	mov 0,%o4
	call _skip_quoted_string,0
	nop
	st %o0,[%fp-24]
	b L1452
	nop
L1450:
	nop
L1453:
	ld [%fp-24],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	bne L1454
	nop
	ld [%fp-24],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,40
	be L1454
	nop
	ld [%fp-24],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,41
	be L1454
	nop
	ld [%fp-24],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,34
	be L1454
	nop
	ld [%fp-24],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,39
	be L1454
	nop
	ld [%fp-24],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	be L1454
	nop
	ld [%fp-24],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
	b L1453
	nop
L1454:
L1452:
L1449:
L1446:
	mov 20,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-32]
	ld [%fp-24],%o0
	ld [%fp-40],%o1
	sub %o0,%o1,%o0
	add %o0,1,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	ld [%fp-32],%o1
	st %o0,[%o1+4]
	ld [%fp-32],%o1
	ld [%fp-24],%o0
	ld [%fp-40],%o3
	sub %o0,%o3,%o2
	ld [%fp-40],%o0
	ld [%o1+4],%o1
	call _bcopy,0
	nop
	ld [%fp-32],%o0
	ld [%fp-24],%o1
	ld [%fp-40],%o2
	sub %o1,%o2,%o1
	ld [%o0+4],%o2
	add %o1,%o2,%o0
	stb %g0,[%o0]
	ld [%fp-32],%o0
	ld [%fp-20],%o1
	st %o1,[%o0]
	ld [%fp-32],%o0
	st %o0,[%fp-20]
	ld [%fp-32],%o0
	ld [%fp-24],%o1
	ld [%fp-40],%o2
	sub %o1,%o2,%o1
	st %o1,[%o0+8]
L1455:
L1458:
	ld [%fp-24],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1459
	nop
	ld [%fp-24],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
	b L1458
	nop
L1459:
L1457:
	b L1456
	nop
	b L1455
	nop
L1456:
	ld [%fp-24],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	blu L1460
	nop
	sethi %hi(LC187),%o1
	or %o1,%lo(LC187),%o0
	call _error,0
	nop
	ld [%fp+76],%o0
	mov -1,%o1
	st %o1,[%o0]
	mov 0,%i0
	b L1442
	nop
L1460:
	b L1443
	nop
L1444:
	ld [%fp+68],%o0
	ld [%fp-24],%o1
	st %o1,[%o0]
	mov 0,%o0
	ld [%fp-20],%o1
L1461:
	cmp %o1,0
	be L1462
	nop
	ld [%o1],%o2
	st %o0,[%o1]
	mov %o1,%o0
L1463:
	mov %o2,%o1
	b L1461
	nop
L1462:
	mov %o0,%i0
	b L1442
	nop
L1442:
	ret
	restore
	.align 4
	.proc	020
_free_token_list:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
L1465:
	ld [%fp+68],%o0
	cmp %o0,0
	be L1466
	nop
	ld [%fp+68],%o0
	ld [%o0],%o1
	st %o1,[%fp-20]
	ld [%fp+68],%o1
	ld [%o1+4],%o0
	call _free,0
	nop
	ld [%fp+68],%o0
	call _free,0
	nop
	ld [%fp-20],%o0
	st %o0,[%fp+68]
	b L1465
	nop
L1466:
L1464:
	ret
	restore
	.align 4
	.proc	0110
_assertion_install:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	ld [%fp+72],%o0
	add %o0,25,%l1
	mov %l1,%o0
	call _xmalloc,0
	nop
	mov %o0,%l0
	ld [%fp+76],%l2
	mov %l2,%o1
	sll %o1,2,%o0
	sethi %hi(_assertion_hashtab),%o2
	or %o2,%lo(_assertion_hashtab),%o1
	add %o1,%o0,%o0
	st %o0,[%l0+8]
	mov %l2,%o1
	sll %o1,2,%o0
	sethi %hi(_assertion_hashtab),%o2
	or %o2,%lo(_assertion_hashtab),%o1
	ld [%o0+%o1],%o0
	st %o0,[%l0]
	mov %l2,%o1
	sll %o1,2,%o0
	sethi %hi(_assertion_hashtab),%o2
	or %o2,%lo(_assertion_hashtab),%o1
	st %l0,[%o0+%o1]
	st %g0,[%l0+4]
	ld [%l0],%o0
	cmp %o0,0
	be L1468
	nop
	ld [%l0],%o0
	st %l0,[%o0+4]
L1468:
	ld [%fp+72],%o0
	st %o0,[%l0+12]
	st %g0,[%l0+20]
	add %l0,24,%o0
	st %o0,[%l0+16]
	ld [%l0+16],%l3
	ld [%fp+68],%l4
	mov 0,%l1
L1469:
	ld [%fp+72],%o0
	cmp %l1,%o0
	bge L1470
	nop
	ldub [%l4],%o0
	stb %o0,[%l3]
	add %l4,1,%l4
	add %l3,1,%l3
L1471:
	add %l1,1,%l1
	b L1469
	nop
L1470:
	ld [%l0+16],%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
	mov %l0,%i0
	b L1467
	nop
L1467:
	ret
	restore
	.align 4
	.proc	0110
_assertion_lookup:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	ld [%fp+76],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_assertion_hashtab),%o2
	or %o2,%lo(_assertion_hashtab),%o1
	ld [%o0+%o1],%l1
L1473:
	cmp %l1,0
	be L1474
	nop
	ld [%l1+12],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	bne L1475
	nop
	ld [%l1+16],%o0
	ld [%fp+68],%o1
	ld [%fp+72],%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L1475
	nop
	mov %l1,%i0
	b L1472
	nop
L1475:
	ld [%l1],%l1
	b L1473
	nop
L1474:
	mov 0,%i0
	b L1472
	nop
L1472:
	ret
	restore
	.align 4
	.proc	020
_delete_assertion:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	cmp %o1,0
	be L1477
	nop
	ld [%fp+68],%o1
	ld [%o1+4],%o0
	ld [%fp+68],%o1
	ld [%o1],%o2
	st %o2,[%o0]
L1477:
	ld [%fp+68],%o0
	ld [%o0],%o1
	cmp %o1,0
	be L1478
	nop
	ld [%fp+68],%o1
	ld [%o1],%o0
	ld [%fp+68],%o1
	ld [%o1+4],%o2
	st %o2,[%o0+4]
L1478:
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	ld [%fp+68],%o0
	ld [%o1],%o1
	cmp %o0,%o1
	bne L1479
	nop
	ld [%fp+68],%o1
	ld [%o1+8],%o0
	ld [%fp+68],%o1
	ld [%o1],%o2
	st %o2,[%o0]
L1479:
	ld [%fp+68],%o0
	call _free,0
	nop
L1476:
	ret
	restore
	.align 8
LC188:
	.ascii "invalid format `#line' command\0"

	.reserve _fname_table.85,152,"bss"
	.align 4
	.proc	04
_do_line:
	!#PROLOGUE# 0
	save %sp,-192,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-20]
	st %g0,[%fp-80]
	add %fp,-72,%o0
	st %o0,[%sp+64]
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_to_temp_buffer,0
	nop
	unimp 44
	ld [%fp-52],%l0
L1481:
L1484:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1485
	nop
	add %l0,1,%l0
	b L1484
	nop
L1485:
L1483:
	b L1482
	nop
	b L1481
	nop
L1482:
	sethi %hi(___lc_ctype),%o1
	ld [%o1+%lo(___lc_ctype)],%o0
	ld [%o0+28],%o1
	cmp %o1,0
	bne L1487
	nop
	sethi %hi(___lc_ctype),%o1
	ld [%o1+%lo(___lc_ctype)],%o0
	ldub [%l0],%o2
	and %o2,0xff,%o1
	mov %o1,%o2
	sll %o2,2,%o1
	ld [%o0+64],%o2
	add %o1,%o2,%o0
	ld [%o0],%o1
	and %o1,16,%o0
	cmp %o0,0
	bne L1486
	nop
	b L1488
	nop
L1487:
	sethi %hi(___lc_ctype),%o0
	ld [%o0+%lo(___lc_ctype)],%o3
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(___lc_ctype),%o2
	mov 16,%o1
	ld [%o2+%lo(___lc_ctype)],%o2
	ld [%o3+28],%o3
	call %o3,0
	nop
	cmp %o0,0
	bne L1486
	nop
	b L1488
	nop
L1488:
	sethi %hi(LC188),%o1
	or %o1,%lo(LC188),%o0
	call _error,0
	nop
	mov 0,%i0
	b L1480
	nop
L1486:
	mov %l0,%o0
	call _atoi,0
	nop
	add %o0,-1,%o1
	st %o1,[%fp-76]
L1489:
	sethi %hi(___lc_ctype),%o1
	ld [%o1+%lo(___lc_ctype)],%o0
	ld [%o0+28],%o1
	cmp %o1,0
	bne L1491
	nop
	sethi %hi(___lc_ctype),%o1
	ld [%o1+%lo(___lc_ctype)],%o0
	ldub [%l0],%o2
	and %o2,0xff,%o1
	mov %o1,%o2
	sll %o2,2,%o1
	ld [%o0+64],%o2
	add %o1,%o2,%o0
	ld [%o0],%o1
	and %o1,16,%o0
	cmp %o0,0
	bne L1492
	nop
	b L1490
	nop
L1491:
	sethi %hi(___lc_ctype),%o0
	ld [%o0+%lo(___lc_ctype)],%o3
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(___lc_ctype),%o2
	mov 16,%o1
	ld [%o2+%lo(___lc_ctype)],%o2
	ld [%o3+28],%o3
	call %o3,0
	nop
	cmp %o0,0
	bne L1492
	nop
	b L1490
	nop
L1492:
	add %l0,1,%l0
	b L1489
	nop
L1490:
	nop
L1493:
L1496:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1497
	nop
	add %l0,1,%l0
	b L1496
	nop
L1497:
L1495:
	b L1494
	nop
	b L1493
	nop
L1494:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,34
	bne L1498
	nop
	add %l0,1,%l0
	st %l0,[%fp-92]
L1499:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L1500
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,34
	be L1500
	nop
	add %l0,1,%l0
	b L1499
	nop
L1500:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,34
	be L1501
	nop
	sethi %hi(LC188),%o1
	or %o1,%lo(LC188),%o0
	call _error,0
	nop
	mov 0,%i0
	b L1480
	nop
L1501:
	ld [%fp-92],%o0
	sub %l0,%o0,%o1
	st %o1,[%fp-96]
	add %l0,1,%l0
L1502:
L1505:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1506
	nop
	add %l0,1,%l0
	b L1505
	nop
L1506:
L1504:
	b L1503
	nop
	b L1502
	nop
L1503:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L1507
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,49
	bne L1508
	nop
	mov 1,%o0
	st %o0,[%fp-80]
	b L1509
	nop
L1508:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,50
	bne L1510
	nop
	mov 2,%o0
	st %o0,[%fp-80]
	b L1511
	nop
L1510:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,51
	bne L1512
	nop
	ld [%fp-20],%o0
	mov 1,%o1
	stb %o1,[%o0+40]
	b L1513
	nop
L1512:
	sethi %hi(LC188),%o1
	or %o1,%lo(LC188),%o0
	call _error,0
	nop
	mov 0,%i0
	b L1480
	nop
L1513:
L1511:
L1509:
	add %l0,1,%l0
L1514:
L1517:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1518
	nop
	add %l0,1,%l0
	b L1517
	nop
L1518:
L1516:
	b L1515
	nop
	b L1514
	nop
L1515:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,51
	bne L1519
	nop
	ld [%fp-20],%o0
	mov 1,%o1
	stb %o1,[%o0+40]
	add %l0,1,%l0
L1520:
L1523:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1524
	nop
	add %l0,1,%l0
	b L1523
	nop
L1524:
L1522:
	b L1521
	nop
	b L1520
	nop
L1521:
L1519:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L1525
	nop
	sethi %hi(LC188),%o1
	or %o1,%lo(LC188),%o0
	call _error,0
	nop
	mov 0,%i0
	b L1480
	nop
L1525:
L1507:
	ld [%fp-92],%o0
	ld [%fp-96],%o1
	mov 37,%o2
	call _hashf,0
	nop
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_fname_table.85),%o2
	or %o2,%lo(_fname_table.85),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-88]
	ld [%fp-88],%o0
	ld [%o0],%o1
	st %o1,[%fp-84]
L1526:
	ld [%fp-84],%o0
	cmp %o0,0
	be L1527
	nop
	ld [%fp-84],%o0
	ld [%o0+16],%o1
	ld [%fp-96],%o0
	cmp %o1,%o0
	bne L1529
	nop
	ld [%fp-84],%o1
	ld [%o1+24],%o0
	ld [%fp-92],%o1
	ld [%fp-96],%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L1529
	nop
	ld [%fp-20],%o0
	ld [%fp-84],%o1
	ld [%o1+24],%o2
	st %o2,[%o0+4]
	b L1527
	nop
L1529:
L1528:
	ld [%fp-84],%o0
	ld [%o0],%o1
	st %o1,[%fp-84]
	b L1526
	nop
L1527:
	ld [%fp-84],%o0
	cmp %o0,0
	bne L1530
	nop
	ld [%fp-96],%o0
	add %o0,29,%o1
	mov 1,%o0
	call _xcalloc,0
	nop
	st %o0,[%fp-84]
	ld [%fp-84],%o0
	ld [%fp-88],%o1
	ld [%o1],%o2
	st %o2,[%o0]
	ld [%fp-88],%o0
	ld [%fp-84],%o1
	st %o1,[%o0]
	ld [%fp-84],%o0
	ld [%fp-96],%o1
	st %o1,[%o0+16]
	ld [%fp-20],%o0
	ld [%fp-84],%o1
	ld [%fp-84],%o3
	add %o3,28,%o2
	mov %o2,%o3
	st %o3,[%o1+24]
	st %o3,[%o0+4]
	ld [%fp-84],%o1
	ld [%fp-92],%o0
	ld [%o1+24],%o1
	ld [%fp-96],%o2
	call _bcopy,0
	nop
L1530:
	b L1531
	nop
L1498:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L1532
	nop
	sethi %hi(LC188),%o1
	or %o1,%lo(LC188),%o0
	call _error,0
	nop
	mov 0,%i0
	b L1480
	nop
L1532:
L1531:
	ld [%fp-20],%o0
	ld [%fp-76],%o1
	st %o1,[%o0+12]
	ld [%fp-20],%o0
	ld [%fp+76],%o1
	mov 0,%o2
	ld [%fp-80],%o3
	call _output_line_command,0
	nop
	ld [%fp+76],%o0
	ld [%fp+76],%o1
	ld [%fp+76],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	ld [%fp-20],%o1
	ld [%fp-20],%o2
	ld [%fp-20],%o3
	ld [%o2+24],%o2
	ld [%o3+20],%o3
	sub %o2,%o3,%o2
	ld [%o1+16],%o3
	sub %o3,%o2,%o1
	cmp %o0,%o1
	bg L1533
	nop
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%fp-20],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o0
	sub %o0,%o1,%o1
	ld [%fp+76],%o0
	call _grow_outbuf,0
	nop
	b L1534
	nop
L1533:
L1534:
	mov 0,%i0
	b L1480
	nop
L1480:
	ret
	restore
	.align 8
LC189:
	.ascii "undefining `%s'\0"
	.align 8
LC190:
	.ascii "garbage after `#undef' directive\0"
	.align 4
	.proc	04
_do_undef:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	ld [%fp+68],%o0
	st %o0,[%fp-28]
	sethi %hi(_pcp_outfile),%o0
	ld [%o0+%lo(_pcp_outfile)],%o1
	cmp %o1,0
	be L1536
	nop
	ld [%fp+76],%o0
	cmp %o0,0
	be L1536
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	ld [%fp+80],%o3
	call _pass_thru_directive,0
	nop
L1536:
	nop
L1537:
L1540:
	ld [%fp+68],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1541
	nop
	ld [%fp+68],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp+68]
	b L1540
	nop
L1541:
L1539:
	b L1538
	nop
	b L1537
	nop
L1538:
	ld [%fp+68],%o0
	sethi %hi(LC161),%o2
	or %o2,%lo(LC161),%o1
	call _check_macro_name,0
	nop
	st %o0,[%fp-20]
L1542:
	ld [%fp+68],%o0
	ld [%fp-20],%o1
	mov -1,%o2
	call _lookup,0
	nop
	mov %o0,%o1
	st %o1,[%fp-24]
	cmp %o1,0
	be L1543
	nop
	sethi %hi(_debug_output),%o0
	ld [%o0+%lo(_debug_output)],%o1
	cmp %o1,0
	be L1544
	nop
	ld [%fp+76],%o0
	cmp %o0,0
	be L1544
	nop
	ld [%fp-28],%o0
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	ld [%fp+80],%o3
	call _pass_thru_directive,0
	nop
L1544:
	ld [%fp-24],%o0
	ld [%o0+12],%o1
	cmp %o1,31
	be L1545
	nop
	ld [%fp-24],%o1
	sethi %hi(LC189),%o2
	or %o2,%lo(LC189),%o0
	ld [%o1+20],%o1
	call _warning,0
	nop
L1545:
	ld [%fp-24],%o0
	call _delete_macro,0
	nop
	b L1542
	nop
L1543:
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L1546
	nop
	ld [%fp+68],%o0
	ld [%fp-20],%o1
	add %o0,%o1,%o0
	st %o0,[%fp+68]
L1547:
L1550:
	ld [%fp+68],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1551
	nop
	ld [%fp+68],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp+68]
	b L1550
	nop
L1551:
L1549:
	b L1548
	nop
	b L1547
	nop
L1548:
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	be L1552
	nop
	sethi %hi(LC190),%o1
	or %o1,%lo(LC190),%o0
	call _pedwarn,0
	nop
L1552:
L1546:
	mov 0,%i0
	b L1535
	nop
L1535:
	ret
	restore
	.align 8
LC191:
	.ascii "#error %s\0"
	.align 4
	.proc	04
_do_error:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	ld [%fp+72],%o0
	ld [%fp+68],%o1
	sub %o0,%o1,%o0
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	add %o0,1,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-24]
	ld [%fp+68],%o0
	ld [%fp-24],%o1
	ld [%fp-20],%o2
	call _bcopy,0
	nop
	ld [%fp-24],%o0
	ld [%fp-20],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
L1554:
L1557:
	ld [%fp-24],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1558
	nop
	ld [%fp-24],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
	b L1557
	nop
L1558:
L1556:
	b L1555
	nop
	b L1554
	nop
L1555:
	sethi %hi(LC191),%o1
	or %o1,%lo(LC191),%o0
	ld [%fp-24],%o1
	call _error,0
	nop
	mov 33,%o0
	call _exit,0
	nop
	mov 0,%i0
	b L1553
	nop
L1553:
	ret
	restore
	.align 8
LC192:
	.ascii "#warning %s\0"
	.align 4
	.proc	04
_do_warning:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	ld [%fp+72],%o0
	ld [%fp+68],%o1
	sub %o0,%o1,%o0
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	add %o0,1,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-24]
	ld [%fp+68],%o0
	ld [%fp-24],%o1
	ld [%fp-20],%o2
	call _bcopy,0
	nop
	ld [%fp-24],%o0
	ld [%fp-20],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
L1560:
L1563:
	ld [%fp-24],%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1564
	nop
	ld [%fp-24],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
	b L1563
	nop
L1564:
L1562:
	b L1561
	nop
	b L1560
	nop
L1561:
	sethi %hi(LC192),%o1
	or %o1,%lo(LC192),%o0
	ld [%fp-24],%o1
	call _warning,0
	nop
	mov 0,%i0
	b L1559
	nop
L1559:
	ret
	restore
	.align 4
	.proc	04
_do_once:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %g0,[%fp-24]
	nop
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o1
	st %o1,[%fp-20]
L1566:
	ld [%fp-20],%o0
	cmp %o0,0
	bl L1567
	nop
	ld [%fp-20],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	ld [%o0+%o1],%o0
	cmp %o0,0
	be L1569
	nop
	ld [%fp-20],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-24]
	b L1567
	nop
L1569:
L1568:
	ld [%fp-20],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L1566
	nop
L1567:
	ld [%fp-24],%o0
	cmp %o0,0
	be L1570
	nop
	mov 12,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-28]
	ld [%fp-28],%o0
	sethi %hi(_dont_repeat_files),%o1
	ld [%o1+%lo(_dont_repeat_files)],%o2
	st %o2,[%o0]
	sethi %hi(_dont_repeat_files),%o0
	ld [%fp-28],%o1
	st %o1,[%o0+%lo(_dont_repeat_files)]
	ld [%fp-24],%o1
	ld [%o1],%o0
	call _savestring,0
	nop
	ld [%fp-28],%o1
	st %o0,[%o1+4]
	ld [%fp-28],%o0
	st %g0,[%o0+8]
L1570:
	mov 0,%i0
	b L1565
	nop
L1565:
	ret
	restore
	.align 8
LC193:
	.ascii "ANSI C does not allow `#ident'\0"
	.align 4
	.proc	04
_do_ident:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L1572
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o0,%o1,%o0
	ldub [%o0+40],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	bne L1572
	nop
	sethi %hi(LC193),%o1
	or %o1,%lo(LC193),%o0
	call _pedwarn,0
	nop
L1572:
	mov 0,%i0
	b L1571
	nop
L1571:
	ret
	restore
	.align 8
LC194:
	.ascii "once\0"
	.align 8
LC195:
	.ascii "`#pragma once' is obsolete\0"
	.align 8
LC196:
	.ascii "implementation\0"
	.align 8
LC197:
	.ascii "`#pragma implementation' for `%s' appears after file is included\0"
	.align 4
	.proc	04
_do_pragma:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
L1574:
	ld [%fp+68],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,32
	be L1576
	nop
	ld [%fp+68],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,9
	be L1576
	nop
	b L1575
	nop
L1576:
	ld [%fp+68],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp+68]
	b L1574
	nop
L1575:
	ld [%fp+68],%o0
	sethi %hi(LC194),%o2
	or %o2,%lo(LC194),%o1
	mov 4,%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L1577
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o0,%o1,%o0
	ldub [%o0+40],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	bne L1578
	nop
	sethi %hi(LC195),%o1
	or %o1,%lo(LC195),%o0
	call _warning,0
	nop
L1578:
	call _do_once,0
	nop
L1577:
	ld [%fp+68],%o0
	sethi %hi(LC196),%o2
	or %o2,%lo(LC196),%o1
	mov 14,%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L1579
	nop
	ld [%fp+68],%o0
	add %o0,14,%o1
	st %o1,[%fp-24]
L1580:
L1583:
	ld [%fp-24],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1584
	nop
	ld [%fp-24],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
	b L1583
	nop
L1584:
L1582:
	b L1581
	nop
	b L1580
	nop
L1581:
	ld [%fp-24],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	be L1586
	nop
	ld [%fp-24],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,34
	bne L1586
	nop
	b L1585
	nop
L1586:
	mov 0,%i0
	b L1573
	nop
L1585:
	ld [%fp-24],%o0
	add %o0,1,%o1
	st %o1,[%fp-28]
	ld [%fp-28],%o0
	mov 34,%o1
	call _index,0
	nop
	mov %o0,%o1
	st %o1,[%fp-24]
	cmp %o1,0
	be L1587
	nop
	ld [%fp-24],%o0
	stb %g0,[%o0]
L1587:
	nop
	sethi %hi(_all_include_files),%o0
	ld [%o0+%lo(_all_include_files)],%o1
	st %o1,[%fp-20]
L1588:
	ld [%fp-20],%o0
	cmp %o0,0
	be L1589
	nop
	ld [%fp-20],%o1
	ld [%o1+4],%o0
	mov 47,%o1
	call _rindex,0
	nop
	st %o0,[%fp-32]
	ld [%fp-32],%o0
	cmp %o0,0
	be L1591
	nop
	ld [%fp-32],%o1
	add %o1,1,%o0
	b L1592
	nop
L1591:
	ld [%fp-20],%o1
	ld [%o1+4],%o0
L1592:
	st %o0,[%fp-32]
	ld [%fp-32],%o0
	cmp %o0,0
	be L1593
	nop
	ld [%fp-32],%o0
	ld [%fp-28],%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L1593
	nop
	sethi %hi(LC197),%o1
	or %o1,%lo(LC197),%o0
	ld [%fp-28],%o1
	call _warning,0
	nop
L1593:
L1590:
	ld [%fp-20],%o0
	ld [%o0],%o1
	st %o1,[%fp-20]
	b L1588
	nop
L1589:
L1579:
	mov 0,%i0
	b L1573
	nop
L1573:
	ret
	restore
	.align 8
LC198:
	.ascii "ANSI C does not allow `#sccs'\0"
	.align 4
	.proc	04
_do_sccs:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L1595
	nop
	sethi %hi(LC198),%o1
	or %o1,%lo(LC198),%o0
	call _pedwarn,0
	nop
L1595:
	mov 0,%i0
	b L1594
	nop
L1594:
	ret
	restore
	.align 4
	.proc	04
_do_if:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-24]
	ld [%fp+72],%o0
	ld [%fp+68],%o2
	sub %o0,%o2,%o1
	ld [%fp+68],%o0
	call _eval_if_expression,0
	nop
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	xor %o0,0,%o2
	subcc %g0,%o2,%g0
	subx %g0,-1,%o1
	ld [%fp-24],%o0
	mov 7,%o2
	mov 0,%o3
	call _conditional_skip,0
	nop
	mov 0,%i0
	b L1596
	nop
L1596:
	ret
	restore
	.align 8
LC199:
	.ascii "`#elif' not within a conditional\0"
	.align 8
LC200:
	.ascii "`#elif' after `#else'\0"
	.align 8
LC201:
	.ascii " (matches line %d\0"
	.align 8
LC202:
	.ascii ", file %s\0"
	.align 8
LC203:
	.ascii ")\12\0"
	.align 4
	.proc	04
_do_elif:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-24]
	sethi %hi(_if_stack),%o0
	sethi %hi(_indepth),%o2
	ld [%o2+%lo(_indepth)],%o1
	mov %o1,%o3
	sll %o3,1,%o2
	add %o2,%o1,%o2
	sll %o2,2,%o3
	sub %o3,%o1,%o3
	sll %o3,2,%o1
	sethi %hi(_instack),%o3
	or %o3,%lo(_instack),%o2
	add %o1,%o2,%o1
	ld [%o0+%lo(_if_stack)],%o0
	ld [%o1+32],%o1
	cmp %o0,%o1
	bne L1598
	nop
	sethi %hi(LC199),%o1
	or %o1,%lo(LC199),%o0
	call _error,0
	nop
	mov 0,%i0
	b L1597
	nop
	b L1599
	nop
L1598:
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	ld [%o0+20],%o1
	cmp %o1,7
	be L1600
	nop
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	ld [%o0+20],%o1
	cmp %o1,10
	be L1600
	nop
	sethi %hi(LC200),%o1
	or %o1,%lo(LC200),%o0
	call _error,0
	nop
	sethi %hi(_if_stack),%o0
	ld [%o0+%lo(_if_stack)],%o2
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC201),%o3
	or %o3,%lo(LC201),%o1
	ld [%o2+8],%o2
	call _fprintf,0
	nop
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	ld [%o0+4],%o1
	cmp %o1,0
	be L1601
	nop
	ld [%fp-24],%o0
	ld [%o0],%o1
	cmp %o1,0
	be L1601
	nop
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	ld [%fp-24],%o1
	ld [%o0+4],%o0
	ld [%o1+4],%o1
	call _strcmp,0
	nop
	cmp %o0,0
	be L1601
	nop
	sethi %hi(_if_stack),%o0
	ld [%o0+%lo(_if_stack)],%o2
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC202),%o3
	or %o3,%lo(LC202),%o1
	ld [%o2+4],%o2
	call _fprintf,0
	nop
L1601:
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC203),%o2
	or %o2,%lo(LC203),%o1
	call _fprintf,0
	nop
L1600:
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	mov 10,%o1
	st %o1,[%o0+20]
L1599:
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	ld [%o0+12],%o1
	cmp %o1,0
	be L1602
	nop
	ld [%fp-24],%o0
	mov 0,%o1
	call _skip_if_group,0
	nop
	b L1603
	nop
L1602:
	ld [%fp+72],%o0
	ld [%fp+68],%o2
	sub %o0,%o2,%o1
	ld [%fp+68],%o0
	call _eval_if_expression,0
	nop
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	cmp %o0,0
	bne L1604
	nop
	ld [%fp-24],%o0
	mov 0,%o1
	call _skip_if_group,0
	nop
	b L1605
	nop
L1604:
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	sethi %hi(_if_stack),%o2
	ld [%o2+%lo(_if_stack)],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	ld [%fp-24],%o0
	ld [%fp+76],%o1
	mov 1,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
L1605:
L1603:
	mov 0,%i0
	b L1597
	nop
L1597:
	ret
	restore
	.align 4
	.proc	04
_eval_if_expression:
	!#PROLOGUE# 0
	save %sp,-168,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	sethi %hi(LC171),%o1
	or %o1,%lo(LC171),%o0
	mov -1,%o1
	mov 33,%o2
	mov 0,%o3
	mov 0,%o4
	mov -1,%o5
	call _install,0
	nop
	st %o0,[%fp-68]
	sethi %hi(_pcp_inside_if),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_pcp_inside_if)]
	ld [%fp+68],%o0
	ld [%fp+72],%o2
	add %o0,%o2,%o1
	add %fp,-64,%o0
	st %o0,[%sp+64]
	ld [%fp+68],%o0
	mov 0,%o2
	mov 1,%o3
	call _expand_to_temp_buffer,0
	nop
	unimp 44
	sethi %hi(_pcp_inside_if),%o0
	st %g0,[%o0+%lo(_pcp_inside_if)]
	ld [%fp-68],%o0
	call _delete_macro,0
	nop
	ld [%fp-44],%o0
	call _parse_c_expression,0
	nop
	st %o0,[%fp-72]
	ld [%fp-44],%o0
	call _free,0
	nop
	ld [%fp-72],%i0
	b L1606
	nop
L1606:
	ret
	restore
	.align 8
LC204:
	.ascii "`#%s' with no argument\0"
	.align 8
LC205:
	.ascii "`#%s' argument starts with punctuation\0"
	.align 8
LC206:
	.ascii "`#%s' argument starts with a digit\0"
	.align 8
LC207:
	.ascii "garbage at end of `#%s' argument\0"
	.align 4
	.proc	04
_do_xifdef:
	!#PROLOGUE# 0
	save %sp,-152,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-24]
	st %g0,[%fp-32]
	st %g0,[%fp-36]
	ld [%fp-24],%o0
	ld [%o0],%o1
	cmp %o1,0
	be L1608
	nop
	ld [%fp+80],%o0
	ld [%o0+12],%o1
	cmp %o1,6
	bne L1608
	nop
	ld [%fp-24],%o0
	ld [%o0+20],%o1
	st %o1,[%fp-40]
L1609:
	sethi %hi(_directive_start),%o0
	ld [%fp-40],%o1
	ld [%o0+%lo(_directive_start)],%o0
	cmp %o1,%o0
	be L1610
	nop
	ld [%fp-40],%o0
	add %o0,1,%o1
	st %o1,[%fp-40]
	ldub [%o0],%o0
	stb %o0,[%fp-41]
	ldub [%fp-41],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	sethi %hi(_is_space),%o2
	or %o2,%lo(_is_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1611
	nop
	b L1612
	nop
L1611:
	ldub [%fp-41],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,47
	bne L1613
	nop
	ld [%fp-24],%o0
	ld [%fp-40],%o1
	ld [%o0+24],%o0
	cmp %o1,%o0
	be L1613
	nop
	ld [%fp-40],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L1613
	nop
	ld [%fp-24],%o0
	ld [%o0+24],%o1
	st %o1,[%fp-52]
	ld [%fp-24],%o0
	ld [%fp-40],%o1
	add %o1,1,%o2
	st %o2,[%o0+24]
	add %fp,-48,%o1
	ld [%fp-24],%o0
	mov 1,%o2
	call _skip_to_end_of_comment,0
	nop
	st %o0,[%fp-40]
	ld [%fp-24],%o0
	ld [%fp-52],%o1
	st %o1,[%o0+24]
	b L1614
	nop
L1613:
	b L1615
	nop
L1614:
L1612:
	b L1609
	nop
L1610:
	mov 1,%o0
	st %o0,[%fp-32]
L1615:
L1608:
	nop
L1616:
L1619:
	ld [%fp+68],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1620
	nop
	ld [%fp+68],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp+68]
	b L1619
	nop
L1620:
L1618:
	b L1617
	nop
	b L1616
	nop
L1617:
	nop
L1621:
	ld [%fp+72],%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	be L1622
	nop
	mov -1,%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1622
	nop
	ld [%fp+72],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp+72]
	b L1621
	nop
L1622:
	nop
	ld [%fp+68],%o0
	st %o0,[%fp-28]
L1623:
	ld [%fp-28],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1624
	nop
L1625:
	ld [%fp-28],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-28]
	b L1623
	nop
L1624:
	ld [%fp-28],%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	bne L1626
	nop
	ld [%fp+80],%o0
	ld [%o0+12],%o1
	xor %o1,5,%o2
	subcc %g0,%o2,%g0
	subx %g0,-1,%o0
	st %o0,[%fp-20]
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	bne L1627
	nop
	ld [%fp-28],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	bne L1628
	nop
	sethi %hi(LC204),%o1
	or %o1,%lo(LC204),%o0
	b L1629
	nop
L1628:
	sethi %hi(LC205),%o1
	or %o1,%lo(LC205),%o0
L1629:
	ld [%fp+80],%o1
	ld [%o1+8],%o1
	call _pedwarn,0
	nop
L1627:
	b L1630
	nop
L1626:
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L1631
	nop
	ld [%fp+68],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bleu L1631
	nop
	ld [%fp+68],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,57
	bgu L1631
	nop
	ld [%fp+80],%o1
	sethi %hi(LC206),%o2
	or %o2,%lo(LC206),%o0
	ld [%o1+8],%o1
	call _pedwarn,0
	nop
	b L1632
	nop
L1631:
	ld [%fp-28],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	be L1633
	nop
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	bne L1633
	nop
	ld [%fp+80],%o1
	sethi %hi(LC207),%o2
	or %o2,%lo(LC207),%o0
	ld [%o1+8],%o1
	call _pedwarn,0
	nop
L1633:
L1632:
	ld [%fp-28],%o0
	ld [%fp+68],%o2
	sub %o0,%o2,%o1
	ld [%fp+68],%o0
	mov -1,%o2
	call _lookup,0
	nop
	st %o0,[%fp-52]
	sethi %hi(_pcp_outfile),%o0
	ld [%o0+%lo(_pcp_outfile)],%o1
	cmp %o1,0
	be L1634
	nop
	ld [%fp-52],%o0
	cmp %o0,0
	be L1635
	nop
	ld [%fp-52],%o0
	ld [%o0+24],%o1
	ld [%o1+8],%o0
	cmp %o0,0
	be L1635
	nop
	sethi %hi(_pcp_outfile),%o0
	ld [%fp-52],%o2
	ld [%o0+%lo(_pcp_outfile)],%o0
	sethi %hi(LC137),%o3
	or %o3,%lo(LC137),%o1
	ld [%o2+20],%o2
	call _fprintf,0
	nop
	b L1636
	nop
L1635:
	ld [%fp+68],%o0
	st %o0,[%fp-48]
	sethi %hi(_pcp_outfile),%o1
	ld [%o1+%lo(_pcp_outfile)],%o0
	sethi %hi(LC139),%o2
	or %o2,%lo(LC139),%o1
	call _fprintf,0
	nop
L1637:
	ld [%fp-48],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1638
	nop
	ld [%fp-48],%o0
	add %o0,1,%o1
	st %o1,[%fp-48]
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_pcp_outfile),%o1
	ld [%o1+%lo(_pcp_outfile)],%o1
	call _fputc,0
	nop
	b L1637
	nop
L1638:
	sethi %hi(_pcp_outfile),%o1
	ld [%o1+%lo(_pcp_outfile)],%o0
	sethi %hi(_pcp_outfile),%o1
	ld [%o1+%lo(_pcp_outfile)],%o0
	sethi %hi(_pcp_outfile),%o2
	ld [%o2+%lo(_pcp_outfile)],%o1
	ld [%o1],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0]
	cmp %o2,0
	bge L1639
	nop
	sethi %hi(_pcp_outfile),%o1
	mov 10,%o0
	ld [%o1+%lo(_pcp_outfile)],%o1
	call __flsbuf,0
	nop
	b L1640
	nop
L1639:
	sethi %hi(_pcp_outfile),%o0
	ld [%o0+%lo(_pcp_outfile)],%o1
	ld [%o1+4],%o0
	add %o0,1,%o2
	st %o2,[%o1+4]
	mov 10,%o1
	stb %o1,[%o0]
L1640:
L1636:
L1634:
	ld [%fp-52],%o0
	xor %o0,0,%o1
	subcc %g0,%o1,%g0
	subx %g0,-1,%l0
	ld [%fp+80],%o0
	ld [%o0+12],%o1
	xor %o1,6,%o2
	subcc %g0,%o2,%g0
	subx %g0,-1,%o0
	xor %o0,%l0,%o0
	st %o0,[%fp-20]
	ld [%fp-32],%o0
	cmp %o0,0
	be L1641
	nop
	ld [%fp-20],%o0
	cmp %o0,0
	bne L1641
	nop
	ld [%fp-28],%o0
	ld [%fp+68],%o1
	sub %o0,%o1,%o0
	add %o0,1,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-36]
	ld [%fp-28],%o0
	ld [%fp+68],%o1
	sub %o0,%o1,%o2
	ld [%fp+68],%o0
	ld [%fp-36],%o1
	call _bcopy,0
	nop
	ld [%fp-28],%o0
	ld [%fp+68],%o1
	sub %o0,%o1,%o0
	ld [%fp-36],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
L1641:
L1630:
	ld [%fp-24],%o0
	ld [%fp-20],%o1
	mov 7,%o2
	ld [%fp-36],%o3
	call _conditional_skip,0
	nop
	mov 0,%i0
	b L1607
	nop
L1607:
	ret
	restore
	.align 4
	.proc	020
_conditional_skip:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	mov 1,%o0
	mov 24,%o1
	call _xcalloc,0
	nop
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	ld [%fp+68],%o1
	ld [%o1+4],%o2
	st %o2,[%o0+4]
	ld [%fp-20],%o0
	ld [%fp+68],%o1
	ld [%o1+12],%o2
	st %o2,[%o0+8]
	ld [%fp-20],%o0
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o2
	st %o2,[%o0]
	ld [%fp-20],%o0
	ld [%fp+80],%o1
	st %o1,[%o0+16]
	sethi %hi(_if_stack),%o0
	ld [%fp-20],%o1
	st %o1,[%o0+%lo(_if_stack)]
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	ld [%fp+76],%o1
	st %o1,[%o0+20]
	ld [%fp+72],%o0
	cmp %o0,0
	be L1643
	nop
	ld [%fp+68],%o0
	mov 0,%o1
	call _skip_if_group,0
	nop
	b L1642
	nop
	b L1644
	nop
L1643:
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	sethi %hi(_if_stack),%o2
	ld [%o2+%lo(_if_stack)],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	ld [%fp+68],%o0
	sethi %hi(_outbuf),%o2
	or %o2,%lo(_outbuf),%o1
	mov 1,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
L1644:
L1642:
	ret
	restore
	.align 8
LC208:
	.ascii "`#%s' not within a conditional\0"
	.align 8
LC209:
	.ascii "`#else' or `#elif' after `#else'\0"
	.align 4
	.proc	020
_skip_if_group:
	!#PROLOGUE# 0
	save %sp,-136,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
	ld [%o0+24],%l0
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o0+20],%o0
	ld [%o1+16],%o1
	add %o0,%o1,%l2
	sethi %hi(_if_stack),%o0
	ld [%o0+%lo(_if_stack)],%o1
	st %o1,[%fp-24]
	st %l0,[%fp-28]
L1646:
	cmp %l0,%l2
	bgeu L1647
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	add %l0,1,%l0
	cmp %o0,39
	be L1654
	nop
	cmp %o0,39
	bg L1739
	nop
	cmp %o0,34
	be L1653
	nop
	cmp %o0,34
	bg L1740
	nop
	cmp %o0,10
	be L1658
	nop
	b L1738
	nop
L1740:
	cmp %o0,35
	be L1659
	nop
	b L1738
	nop
L1739:
	cmp %o0,47
	be L1649
	nop
	cmp %o0,92
	be L1655
	nop
	b L1738
	nop
L1649:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L1650
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L1650
	nop
	mov %l0,%o0
	call _newline_fix,0
	nop
L1650:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	be L1652
	nop
	sethi %hi(_cplusplus_comments),%o0
	ld [%o0+%lo(_cplusplus_comments)],%o1
	cmp %o1,0
	be L1651
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	be L1652
	nop
	b L1651
	nop
L1652:
	ld [%fp+68],%o0
	add %l0,1,%l0
	st %l0,[%o0+24]
	ld [%fp+68],%o0
	add %o0,12,%o1
	ld [%fp+68],%o0
	mov 0,%o2
	call _skip_to_end_of_comment,0
	nop
	mov %o0,%l0
L1651:
	b L1648
	nop
L1653:
L1654:
	add %l0,-1,%o0
	ld [%fp+68],%o2
	ld [%fp+68],%o1
	add %o1,12,%o3
	mov %l2,%o1
	ld [%o2+12],%o2
	mov 0,%o4
	mov 0,%o5
	call _skip_quoted_string,0
	nop
	mov %o0,%l0
	b L1648
	nop
L1655:
	cmp %l0,%l2
	bgeu L1656
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L1657
	nop
	ld [%fp+68],%o1
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
L1657:
	add %l0,1,%l0
L1656:
	b L1648
	nop
L1658:
	ld [%fp+68],%o1
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	st %l0,[%fp-28]
	b L1648
	nop
L1659:
	ld [%fp+68],%o0
	add %l0,-1,%o1
	st %o1,[%o0+24]
	ld [%fp-28],%o0
	cmp %o0,0
	bne L1660
	nop
	b L1648
	nop
L1660:
	ld [%fp-28],%l0
L1661:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1663
	nop
	add %l0,1,%l0
	b L1664
	nop
L1663:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L1665
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L1665
	nop
	add %l0,2,%l0
	b L1666
	nop
L1665:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L1667
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L1667
	nop
	add %l0,2,%l0
L1668:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L1670
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L1670
	nop
	b L1669
	nop
L1670:
	add %l0,1,%l0
	b L1668
	nop
L1669:
	add %l0,2,%l0
	b L1671
	nop
L1667:
	sethi %hi(_cplusplus_comments),%o0
	ld [%o0+%lo(_cplusplus_comments)],%o1
	cmp %o1,0
	be L1672
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L1672
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L1672
	nop
	add %l0,2,%l0
L1673:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	add %l0,1,%l0
	cmp %o0,10
	be L1674
	nop
	b L1673
	nop
L1674:
	b L1675
	nop
L1672:
	b L1662
	nop
L1675:
L1671:
L1666:
L1664:
	b L1661
	nop
L1662:
	ld [%fp+68],%o0
	ld [%o0+24],%o1
	cmp %l0,%o1
	be L1676
	nop
	ld [%fp+68],%o0
	ld [%o0+24],%o1
	add %o1,1,%l0
	b L1648
	nop
L1676:
	ld [%fp+68],%o0
	ld [%o0+24],%o1
	add %o1,1,%l0
L1677:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1679
	nop
	add %l0,1,%l0
	b L1680
	nop
L1679:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L1681
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L1681
	nop
	add %l0,2,%l0
	b L1682
	nop
L1681:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L1683
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L1683
	nop
	add %l0,2,%l0
L1684:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L1686
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L1686
	nop
	b L1685
	nop
L1686:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L1687
	nop
	ld [%fp+68],%o1
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
L1687:
	add %l0,1,%l0
	b L1684
	nop
L1685:
	add %l0,2,%l0
	b L1688
	nop
L1683:
	sethi %hi(_cplusplus_comments),%o0
	ld [%o0+%lo(_cplusplus_comments)],%o1
	cmp %o1,0
	be L1689
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L1689
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L1689
	nop
	add %l0,2,%l0
L1690:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	add %l0,1,%l0
	cmp %o0,10
	be L1691
	nop
	b L1690
	nop
L1691:
	b L1692
	nop
L1689:
	b L1678
	nop
L1692:
L1688:
L1682:
L1680:
	b L1677
	nop
L1678:
	mov %l0,%l1
L1693:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1695
	nop
	add %l0,1,%l0
	b L1696
	nop
L1695:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L1697
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L1697
	nop
	mov %l0,%o0
	call _name_newline_fix,0
	nop
L1697:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1698
	nop
	add %l0,1,%l0
	b L1699
	nop
L1698:
	b L1694
	nop
L1699:
L1696:
	b L1693
	nop
L1694:
	sub %l0,%l1,%l3
	st %l1,[%fp-32]
	st %l0,[%fp-36]
	cmp %l3,0
	bne L1700
	nop
	ld [%fp-36],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L1700
	nop
	b L1646
	nop
L1700:
	cmp %l3,0
	be L1702
	nop
	ld [%fp-32],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idstart),%o2
	or %o2,%lo(_is_idstart),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	bne L1701
	nop
	b L1702
	nop
L1702:
	ld [%fp-32],%o0
	st %o0,[%fp-40]
L1703:
	ld [%fp-40],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1704
	nop
	ld [%fp-40],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bleu L1706
	nop
	ld [%fp-40],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,57
	bgu L1706
	nop
	b L1705
	nop
L1706:
	b L1704
	nop
L1705:
	ld [%fp-40],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-40]
	b L1703
	nop
L1704:
	ld [%fp-40],%o0
	ld [%fp-32],%o1
	cmp %o0,%o1
	be L1707
	nop
	ld [%fp-40],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	bne L1707
	nop
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L1708
	nop
	sethi %hi(LC108),%o1
	or %o1,%lo(LC108),%o0
	call _pedwarn,0
	nop
L1708:
	b L1646
	nop
L1707:
	ld [%fp-40],%o0
	ld [%fp-32],%o1
	cmp %o0,%o1
	bne L1709
	nop
L1710:
	ld [%fp-40],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,35
	be L1712
	nop
	ld [%fp-40],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	bne L1712
	nop
	b L1711
	nop
L1712:
	ld [%fp-40],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-40]
	b L1710
	nop
L1711:
	ld [%fp-40],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L1713
	nop
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L1714
	nop
	sethi %hi(_lang_asm),%o0
	ld [%o0+%lo(_lang_asm)],%o1
	cmp %o1,0
	bne L1714
	nop
	sethi %hi(LC109),%o1
	or %o1,%lo(LC109),%o0
	call _pedwarn,0
	nop
L1714:
	b L1646
	nop
L1713:
L1709:
	sethi %hi(_lang_asm),%o0
	ld [%o0+%lo(_lang_asm)],%o1
	cmp %o1,0
	bne L1715
	nop
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L1715
	nop
	sethi %hi(LC110),%o1
	or %o1,%lo(LC110),%o0
	call _pedwarn,0
	nop
L1715:
	b L1646
	nop
L1701:
	nop
	sethi %hi(_directive_table),%o1
	or %o1,%lo(_directive_table),%o0
	st %o0,[%fp-20]
L1716:
	ld [%fp-20],%o0
	ld [%o0],%o1
	cmp %o1,0
	bl L1717
	nop
	ld [%fp-20],%o0
	ld [%o0],%o1
	cmp %l3,%o1
	bne L1719
	nop
	ld [%fp-20],%o1
	ld [%fp-20],%o2
	mov %l1,%o0
	ld [%o1+8],%o1
	ld [%o2],%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L1719
	nop
	ld [%fp+72],%o0
	cmp %o0,0
	be L1720
	nop
	b L1645
	nop
L1720:
	ld [%fp-20],%o0
	ld [%o0+12],%o1
	add %o1,-5,%o0
	mov 10,%o1
	cmp %o1,%o0
	blu L1735
	nop
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(L1734),%o2
	or %o2,%lo(L1734),%o1
	ld [%o0+%o1],%o0
	jmp %o0
	nop
L1734:
	.word	L1723
	.word	L1723
	.word	L1723
	.word	L1725
	.word	L1735
	.word	L1728
	.word	L1735
	.word	L1735
	.word	L1735
	.word	L1735
	.word	L1726
L1722:
L1723:
L1724:
	mov 1,%o0
	mov 24,%o1
	call _xcalloc,0
	nop
	st %o0,[%fp-40]
	ld [%fp-40],%o0
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o2
	st %o2,[%o0]
	sethi %hi(_if_stack),%o0
	ld [%fp-40],%o1
	st %o1,[%o0+%lo(_if_stack)]
	ld [%fp-40],%o0
	ld [%fp+68],%o1
	ld [%o1+12],%o2
	st %o2,[%o0+8]
	ld [%fp-40],%o0
	ld [%fp+68],%o1
	ld [%o1+4],%o2
	st %o2,[%o0+4]
	ld [%fp-40],%o0
	ld [%fp-20],%o1
	ld [%o1+12],%o2
	st %o2,[%o0+20]
	b L1721
	nop
L1725:
L1726:
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L1727
	nop
	sethi %hi(_if_stack),%o0
	ld [%o0+%lo(_if_stack)],%o1
	ld [%fp-24],%o0
	cmp %o1,%o0
	be L1727
	nop
	mov %l0,%o0
	call _validate_else,0
	nop
L1727:
L1728:
	sethi %hi(_if_stack),%o0
	sethi %hi(_indepth),%o2
	ld [%o2+%lo(_indepth)],%o1
	mov %o1,%o3
	sll %o3,1,%o2
	add %o2,%o1,%o2
	sll %o2,2,%o3
	sub %o3,%o1,%o3
	sll %o3,2,%o1
	sethi %hi(_instack),%o3
	or %o3,%lo(_instack),%o2
	add %o1,%o2,%o1
	ld [%o0+%lo(_if_stack)],%o0
	ld [%o1+32],%o1
	cmp %o0,%o1
	bne L1729
	nop
	ld [%fp-20],%o1
	sethi %hi(LC208),%o2
	or %o2,%lo(LC208),%o0
	ld [%o1+8],%o1
	call _error,0
	nop
	b L1721
	nop
	b L1730
	nop
L1729:
	sethi %hi(_if_stack),%o0
	ld [%o0+%lo(_if_stack)],%o1
	ld [%fp-24],%o0
	cmp %o1,%o0
	bne L1731
	nop
	b L1645
	nop
L1731:
L1730:
	ld [%fp-20],%o0
	ld [%o0+12],%o1
	cmp %o1,15
	be L1732
	nop
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	ld [%o0+20],%o1
	cmp %o1,8
	bne L1733
	nop
	sethi %hi(LC209),%o1
	or %o1,%lo(LC209),%o0
	call _error,0
	nop
L1733:
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	ld [%fp-20],%o1
	ld [%o1+12],%o2
	st %o2,[%o0+20]
	b L1721
	nop
L1732:
	sethi %hi(_if_stack),%o0
	ld [%o0+%lo(_if_stack)],%o1
	st %o1,[%fp-40]
	sethi %hi(_if_stack),%o0
	sethi %hi(_if_stack),%o2
	ld [%o2+%lo(_if_stack)],%o1
	ld [%o1],%o2
	st %o2,[%o0+%lo(_if_stack)]
	ld [%fp-40],%o0
	call _free,0
	nop
	b L1721
	nop
L1735:
L1721:
	b L1717
	nop
L1719:
L1718:
	ld [%fp-20],%o1
	add %o1,20,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L1716
	nop
L1717:
	ld [%fp-20],%o0
	ld [%o0],%o1
	cmp %o1,0
	bge L1736
	nop
	sethi %hi(_lang_asm),%o0
	ld [%o0+%lo(_lang_asm)],%o1
	cmp %o1,0
	bne L1736
	nop
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L1736
	nop
	sethi %hi(LC110),%o1
	or %o1,%lo(LC110),%o0
	call _pedwarn,0
	nop
L1736:
L1738:
L1648:
	b L1646
	nop
L1647:
	ld [%fp+68],%o0
	st %l0,[%o0+24]
L1645:
	ret
	restore
	.align 8
LC210:
	.ascii "text following `#else' violates ANSI standard\0"
	.align 8
LC211:
	.ascii "`#else' not within a conditional\0"
	.align 8
LC212:
	.ascii "`#else' after `#else'\0"
	.align 4
	.proc	04
_do_else:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-20]
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L1742
	nop
L1743:
L1746:
	ld [%fp+68],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1747
	nop
	ld [%fp+68],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp+68]
	b L1746
	nop
L1747:
L1745:
	b L1744
	nop
	b L1743
	nop
L1744:
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	be L1748
	nop
	sethi %hi(LC210),%o1
	or %o1,%lo(LC210),%o0
	call _pedwarn,0
	nop
L1748:
L1742:
	sethi %hi(_if_stack),%o0
	sethi %hi(_indepth),%o2
	ld [%o2+%lo(_indepth)],%o1
	mov %o1,%o3
	sll %o3,1,%o2
	add %o2,%o1,%o2
	sll %o2,2,%o3
	sub %o3,%o1,%o3
	sll %o3,2,%o1
	sethi %hi(_instack),%o3
	or %o3,%lo(_instack),%o2
	add %o1,%o2,%o1
	ld [%o0+%lo(_if_stack)],%o0
	ld [%o1+32],%o1
	cmp %o0,%o1
	bne L1749
	nop
	sethi %hi(LC211),%o1
	or %o1,%lo(LC211),%o0
	call _error,0
	nop
	mov 0,%i0
	b L1741
	nop
	b L1750
	nop
L1749:
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	st %g0,[%o0+16]
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	ld [%o0+20],%o1
	cmp %o1,7
	be L1751
	nop
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	ld [%o0+20],%o1
	cmp %o1,10
	be L1751
	nop
	sethi %hi(LC212),%o1
	or %o1,%lo(LC212),%o0
	call _error,0
	nop
	sethi %hi(_if_stack),%o0
	ld [%o0+%lo(_if_stack)],%o2
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC201),%o3
	or %o3,%lo(LC201),%o1
	ld [%o2+8],%o2
	call _fprintf,0
	nop
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	ld [%fp-20],%o1
	ld [%o0+4],%o0
	ld [%o1+4],%o1
	call _strcmp,0
	nop
	cmp %o0,0
	be L1752
	nop
	sethi %hi(_if_stack),%o0
	ld [%o0+%lo(_if_stack)],%o2
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC202),%o3
	or %o3,%lo(LC202),%o1
	ld [%o2+4],%o2
	call _fprintf,0
	nop
L1752:
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC203),%o2
	or %o2,%lo(LC203),%o1
	call _fprintf,0
	nop
L1751:
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	mov 8,%o1
	st %o1,[%o0+20]
L1750:
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	ld [%o0+12],%o1
	cmp %o1,0
	be L1753
	nop
	ld [%fp-20],%o0
	mov 0,%o1
	call _skip_if_group,0
	nop
	b L1754
	nop
L1753:
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o0
	sethi %hi(_if_stack),%o2
	ld [%o2+%lo(_if_stack)],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	ld [%fp-20],%o0
	ld [%fp+76],%o1
	mov 1,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
L1754:
	mov 0,%i0
	b L1741
	nop
L1741:
	ret
	restore
	.align 8
LC213:
	.ascii "text following `#endif' violates ANSI standard\0"
	.align 8
LC214:
	.ascii "unbalanced `#endif'\0"
	.align 4
	.proc	04
_do_endif:
	!#PROLOGUE# 0
	save %sp,-144,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	sethi %hi(_pedantic),%o0
	ld [%o0+%lo(_pedantic)],%o1
	cmp %o1,0
	be L1756
	nop
L1757:
L1760:
	ld [%fp+68],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1761
	nop
	ld [%fp+68],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp+68]
	b L1760
	nop
L1761:
L1759:
	b L1758
	nop
	b L1757
	nop
L1758:
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	be L1762
	nop
	sethi %hi(LC213),%o1
	or %o1,%lo(LC213),%o0
	call _pedwarn,0
	nop
L1762:
L1756:
	sethi %hi(_if_stack),%o0
	sethi %hi(_indepth),%o2
	ld [%o2+%lo(_indepth)],%o1
	mov %o1,%o3
	sll %o3,1,%o2
	add %o2,%o1,%o2
	sll %o2,2,%o3
	sub %o3,%o1,%o3
	sll %o3,2,%o1
	sethi %hi(_instack),%o3
	or %o3,%lo(_instack),%o2
	add %o1,%o2,%o1
	ld [%o0+%lo(_if_stack)],%o0
	ld [%o1+32],%o1
	cmp %o0,%o1
	bne L1763
	nop
	sethi %hi(LC214),%o1
	or %o1,%lo(LC214),%o0
	call _error,0
	nop
	b L1764
	nop
L1763:
	sethi %hi(_if_stack),%o0
	ld [%o0+%lo(_if_stack)],%o1
	st %o1,[%fp-20]
	sethi %hi(_if_stack),%o0
	sethi %hi(_if_stack),%o2
	ld [%o2+%lo(_if_stack)],%o1
	ld [%o1],%o2
	st %o2,[%o0+%lo(_if_stack)]
	ld [%fp-20],%o0
	ld [%o0+16],%o1
	cmp %o1,0
	be L1765
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-24]
	ld [%fp-24],%o0
	ld [%o0+24],%o1
	st %o1,[%fp-28]
	ld [%fp-24],%o0
	ld [%fp-24],%o1
	ld [%o0+20],%o0
	ld [%o1+16],%o1
	add %o0,%o1,%o0
	st %o0,[%fp-32]
L1766:
	ld [%fp-28],%o0
	ld [%fp-32],%o1
	cmp %o0,%o1
	be L1767
	nop
	ld [%fp-28],%o0
	add %o0,1,%o1
	st %o1,[%fp-28]
	ldub [%o0],%o0
	stb %o0,[%fp-33]
	ldub [%fp-33],%o1
	and %o1,0xff,%o0
	cmp %o0,32
	be L1769
	nop
	cmp %o0,32
	bg L1777
	nop
	cmp %o0,10
	bg L1774
	nop
	cmp %o0,9
	bl L1774
	nop
	b L1770
	nop
L1777:
	cmp %o0,47
	be L1772
	nop
	b L1774
	nop
L1769:
L1770:
L1771:
	b L1768
	nop
L1772:
	ld [%fp-28],%o0
	ld [%fp-32],%o1
	cmp %o0,%o1
	be L1773
	nop
	ld [%fp-28],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L1773
	nop
	ld [%fp-24],%o0
	ld [%o0+24],%o1
	st %o1,[%fp-44]
	ld [%fp-24],%o0
	ld [%fp-28],%o1
	add %o1,1,%o2
	st %o2,[%o0+24]
	add %fp,-40,%o1
	ld [%fp-24],%o0
	mov 1,%o2
	call _skip_to_end_of_comment,0
	nop
	st %o0,[%fp-28]
	ld [%fp-24],%o0
	ld [%fp-44],%o1
	st %o1,[%o0+24]
L1773:
	b L1768
	nop
L1774:
	b L1775
	nop
L1768:
	b L1766
	nop
L1767:
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o1
	cmp %o1,0
	be L1778
	nop
	ld [%fp-24],%o0
	ld [%fp-20],%o1
	ld [%o0],%o0
	ld [%o1+16],%o1
	call _record_control_macro,0
	nop
L1778:
	nop
L1775:
L1765:
	ld [%fp-20],%o0
	call _free,0
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o1
	or %o1,%lo(_instack),%o2
	add %o0,%o2,%o1
	mov %o1,%o0
	ld [%fp+76],%o1
	mov 1,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
L1764:
	mov 0,%i0
	b L1755
	nop
L1755:
	ret
	restore
	.align 8
LC215:
	.ascii "text following `#else' or `#endif' violates ANSI standard\0"
	.align 4
	.proc	020
_validate_else:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	mov %i0,%l0
L1780:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L1782
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L1782
	nop
	add %l0,2,%l0
L1782:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_hor_space),%o2
	or %o2,%lo(_is_hor_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1783
	nop
	add %l0,1,%l0
	b L1784
	nop
L1783:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L1785
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L1786
	nop
	ldub [%l0+2],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L1786
	nop
	add %l0,1,%o1
	mov %o1,%o0
	call _newline_fix,0
	nop
L1786:
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L1787
	nop
	add %l0,2,%l0
L1788:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L1789
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L1790
	nop
	ldub [%l0+2],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L1790
	nop
	add %l0,1,%o1
	mov %o1,%o0
	call _newline_fix,0
	nop
L1790:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L1791
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L1791
	nop
	add %l0,2,%l0
	b L1789
	nop
L1791:
	add %l0,1,%l0
	b L1788
	nop
L1789:
	b L1792
	nop
L1787:
	sethi %hi(_cplusplus_comments),%o0
	ld [%o0+%lo(_cplusplus_comments)],%o1
	cmp %o1,0
	be L1793
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L1793
	nop
	add %l0,2,%l0
L1794:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L1795
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	add %l0,1,%l0
	cmp %o0,10
	be L1795
	nop
	b L1794
	nop
L1795:
L1793:
L1792:
	b L1796
	nop
L1785:
	b L1781
	nop
L1796:
L1784:
	b L1780
	nop
L1781:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L1797
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	be L1797
	nop
	sethi %hi(LC215),%o1
	or %o1,%lo(LC215),%o0
	call _pedwarn,0
	nop
L1797:
L1779:
	ret
	restore
	.align 4
	.proc	0114
_skip_to_end_of_comment:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	mov %i0,%l0
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	ld [%l0+20],%o0
	ld [%l0+16],%o1
	add %o0,%o1,%l1
	ld [%l0+24],%l2
	sethi %hi(_outbuf),%o1
	or %o1,%lo(_outbuf),%o0
	st %o0,[%fp-20]
	mov 0,%o0
	sethi %hi(_put_out_comments),%o1
	ld [%o1+%lo(_put_out_comments)],%o2
	cmp %o2,0
	be L1799
	nop
	ld [%fp+72],%o1
	cmp %o1,0
	bne L1799
	nop
	mov 1,%o0
L1799:
	st %o0,[%fp-24]
	ld [%fp-24],%o0
	cmp %o0,0
	be L1800
	nop
	ld [%fp-20],%o1
	ld [%o1+24],%o0
	add %o0,1,%o2
	st %o2,[%o1+24]
	mov 47,%o1
	stb %o1,[%o0]
	ld [%fp-20],%o1
	ld [%o1+24],%o0
	add %o0,1,%o2
	st %o2,[%o1+24]
	mov 42,%o1
	stb %o1,[%o0]
L1800:
	sethi %hi(_cplusplus_comments),%o0
	ld [%o0+%lo(_cplusplus_comments)],%o1
	cmp %o1,0
	be L1801
	nop
	ldub [%l2-1],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L1801
	nop
	ld [%fp-24],%o0
	cmp %o0,0
	be L1802
	nop
L1803:
	cmp %l2,%l1
	bgeu L1804
	nop
	ld [%fp-20],%o1
	ld [%o1+24],%o0
	add %o0,1,%o2
	st %o2,[%o1+24]
	ldub [%l2],%o1
	stb %o1,[%o0]
	and %o1,0xff,%o0
	add %l2,1,%l2
	cmp %o0,10
	bne L1805
	nop
	add %l2,-1,%l2
	b L1804
	nop
L1805:
	b L1803
	nop
L1804:
	ld [%fp-20],%o0
	mov -1,%o1
	ld [%o0+24],%o2
	add %o1,%o2,%o0
	mov 42,%o1
	stb %o1,[%o0]
	ld [%fp-20],%o1
	ld [%o1+24],%o0
	add %o0,1,%o2
	st %o2,[%o1+24]
	mov 47,%o1
	stb %o1,[%o0]
	ld [%fp-20],%o1
	ld [%o1+24],%o0
	add %o0,1,%o2
	st %o2,[%o1+24]
	mov 10,%o1
	stb %o1,[%o0]
	b L1806
	nop
L1802:
L1807:
	cmp %l2,%l1
	bgeu L1808
	nop
	ldub [%l2],%o1
	and %o1,0xff,%o0
	add %l2,1,%l2
	cmp %o0,10
	bne L1809
	nop
	add %l2,-1,%l2
	b L1808
	nop
L1809:
	b L1807
	nop
L1808:
L1806:
	st %l2,[%l0+24]
	mov %l2,%i0
	b L1798
	nop
L1801:
	nop
L1810:
	cmp %l2,%l1
	bgeu L1811
	nop
	ld [%fp-24],%o0
	cmp %o0,0
	be L1812
	nop
	ld [%fp-20],%o1
	ld [%o1+24],%o0
	add %o0,1,%o2
	st %o2,[%o1+24]
	ldub [%l2],%o1
	stb %o1,[%o0]
L1812:
	ldub [%l2],%o1
	and %o1,0xff,%o0
	add %l2,1,%l2
	cmp %o0,42
	be L1819
	nop
	cmp %o0,42
	bg L1825
	nop
	cmp %o0,10
	be L1816
	nop
	b L1824
	nop
L1825:
	cmp %o0,47
	be L1814
	nop
	b L1824
	nop
L1814:
	sethi %hi(_warn_comments),%o0
	ld [%o0+%lo(_warn_comments)],%o1
	cmp %o1,0
	be L1815
	nop
	ld [%fp+76],%o0
	cmp %o0,0
	bne L1815
	nop
	cmp %l2,%l1
	bgeu L1815
	nop
	ldub [%l2],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L1815
	nop
	sethi %hi(LC101),%o1
	or %o1,%lo(LC101),%o0
	call _warning,0
	nop
L1815:
	b L1813
	nop
L1816:
	ld [%fp+72],%o0
	cmp %o0,0
	be L1817
	nop
	ld [%fp+72],%o1
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%o1],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0]
L1817:
	ld [%fp-24],%o0
	cmp %o0,0
	be L1818
	nop
	ld [%fp-20],%o1
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
L1818:
	b L1813
	nop
L1819:
	ldub [%l2],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L1820
	nop
	ldub [%l2+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L1820
	nop
	mov %l2,%o0
	call _newline_fix,0
	nop
L1820:
	ldub [%l2],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L1821
	nop
	ld [%fp-24],%o0
	cmp %o0,0
	be L1822
	nop
	ld [%fp-20],%o1
	ld [%o1+24],%o0
	add %o0,1,%o2
	st %o2,[%o1+24]
	mov 47,%o1
	stb %o1,[%o0]
L1822:
	add %l2,1,%l2
	st %l2,[%l0+24]
	mov %l2,%i0
	b L1798
	nop
L1821:
	b L1813
	nop
L1824:
L1813:
	b L1810
	nop
L1811:
	st %l2,[%l0+24]
	mov %l2,%i0
	b L1798
	nop
L1798:
	ret
	restore
	.align 8
LC216:
	.ascii "unterminated character constant\0"
	.align 4
	.proc	0114
_skip_quoted_string:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	mov %i0,%l1
	mov %i1,%l0
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	st %i4,[%fp+84]
	st %i5,[%fp+88]
	ldub [%l1],%l3
	add %l1,1,%l1
L1827:
	cmp %l1,%l0
	blu L1829
	nop
	ld [%fp+76],%o0
	call _line_for_error,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	sethi %hi(LC98),%o2
	or %o2,%lo(LC98),%o1
	call _error_with_line,0
	nop
	ld [%fp+88],%o0
	cmp %o0,0
	be L1830
	nop
	ld [%fp+88],%o0
	mov 1,%o1
	st %o1,[%o0]
L1830:
	b L1828
	nop
L1829:
	ldub [%l1],%l2
	add %l1,1,%l1
	and %l2,0xff,%o0
	cmp %o0,92
	bne L1831
	nop
L1832:
	ldub [%l1],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L1833
	nop
	ldub [%l1+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L1833
	nop
	ld [%fp+84],%o0
	cmp %o0,0
	be L1834
	nop
	ld [%fp+84],%o0
	mov 1,%o1
	st %o1,[%o0]
L1834:
	ld [%fp+80],%o0
	cmp %o0,0
	be L1835
	nop
	ld [%fp+80],%o1
	ld [%fp+80],%o0
	ld [%fp+80],%o1
	ld [%o1],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0]
L1835:
	add %l1,2,%l1
	b L1832
	nop
L1833:
	ldub [%l1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L1836
	nop
	ld [%fp+80],%o0
	cmp %o0,0
	be L1836
	nop
	ld [%fp+84],%o0
	cmp %o0,0
	be L1837
	nop
	ld [%fp+84],%o0
	mov 1,%o1
	st %o1,[%o0]
L1837:
	ld [%fp+80],%o1
	ld [%fp+80],%o0
	ld [%fp+80],%o1
	ld [%o1],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0]
L1836:
	add %l1,1,%l1
	b L1838
	nop
L1831:
	and %l2,0xff,%o0
	cmp %o0,10
	bne L1839
	nop
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	be L1840
	nop
	add %l1,-1,%l1
	ld [%fp+88],%o0
	cmp %o0,0
	be L1841
	nop
	ld [%fp+88],%o0
	mov 1,%o1
	st %o1,[%o0]
L1841:
	b L1828
	nop
L1840:
	and %l3,0xff,%o0
	cmp %o0,39
	bne L1842
	nop
	ld [%fp+76],%o0
	call _line_for_error,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	sethi %hi(LC216),%o2
	or %o2,%lo(LC216),%o1
	call _error_with_line,0
	nop
	add %l1,-1,%l1
	ld [%fp+88],%o0
	cmp %o0,0
	be L1843
	nop
	ld [%fp+88],%o0
	mov 1,%o1
	st %o1,[%o0]
L1843:
	b L1828
	nop
L1842:
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	be L1844
	nop
	ld [%fp+88],%o0
	cmp %o0,0
	be L1845
	nop
	ld [%fp+88],%o0
	mov 1,%o1
	st %o1,[%o0]
L1845:
	b L1828
	nop
L1844:
	ld [%fp+80],%o0
	cmp %o0,0
	be L1846
	nop
	ld [%fp+80],%o1
	ld [%fp+80],%o0
	ld [%fp+80],%o1
	ld [%o1],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0]
L1846:
	b L1847
	nop
L1839:
	and %l2,0xff,%o0
	and %l3,0xff,%o1
	cmp %o0,%o1
	bne L1848
	nop
	b L1828
	nop
L1848:
L1847:
L1838:
	b L1827
	nop
L1828:
	mov %l1,%i0
	b L1826
	nop
L1826:
	ret
	restore
	.align 4
	.proc	0114
_skip_paren_group:
	!#PROLOGUE# 0
	save %sp,-136,%sp
	!#PROLOGUE# 1
	mov %i0,%l0
	ld [%l0+20],%o0
	ld [%l0+16],%o1
	add %o0,%o1,%o0
	st %o0,[%fp-20]
	ld [%l0+24],%o0
	st %o0,[%fp-24]
	st %g0,[%fp-28]
	st %g0,[%fp-32]
L1850:
	ld [%fp-24],%o0
	ld [%fp-20],%o1
	cmp %o0,%o1
	be L1851
	nop
	ld [%fp-24],%o0
	add %o0,1,%o1
	st %o1,[%fp-24]
	ldub [%o0],%o0
	and %o0,0xff,%o1
	st %o1,[%fp-36]
	ld [%fp-36],%o1
	add %o1,-34,%o0
	mov 13,%o1
	cmp %o1,%o0
	blu L1862
	nop
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(L1861),%o2
	or %o2,%lo(L1861),%o1
	ld [%o0+%o1],%o0
	jmp %o0
	nop
L1861:
	.word	L1858
	.word	L1862
	.word	L1862
	.word	L1862
	.word	L1862
	.word	L1859
	.word	L1853
	.word	L1854
	.word	L1862
	.word	L1862
	.word	L1862
	.word	L1862
	.word	L1862
	.word	L1856
L1853:
	ld [%fp-28],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-28]
	b L1852
	nop
L1854:
	ld [%fp-28],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-28]
	ld [%fp-28],%o0
	cmp %o0,0
	bne L1855
	nop
	ld [%fp-24],%o0
	st %o0,[%l0+24]
	mov %o0,%i0
	b L1849
	nop
L1855:
	b L1852
	nop
L1856:
	ld [%fp-24],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L1857
	nop
	ld [%fp-24],%o0
	st %o0,[%l0+24]
	add %fp,-32,%o1
	mov %l0,%o0
	mov 0,%o2
	call _skip_to_end_of_comment,0
	nop
	st %o0,[%fp-24]
	ld [%l0+24],%o0
	st %o0,[%fp-24]
L1857:
L1858:
L1859:
	st %g0,[%fp-40]
	ld [%fp-24],%o1
	add %o1,-1,%o0
	add %fp,-40,%o5
	ld [%fp-20],%o1
	mov 0,%o2
	mov 0,%o3
	mov 0,%o4
	call _skip_quoted_string,0
	nop
	st %o0,[%fp-24]
	ld [%fp-40],%o0
	cmp %o0,0
	be L1860
	nop
	ld [%fp-24],%o0
	st %o0,[%l0+24]
	mov %o0,%i0
	b L1849
	nop
L1860:
	b L1852
	nop
L1862:
L1852:
	b L1850
	nop
L1851:
	ld [%fp-24],%o0
	st %o0,[%l0+24]
	ld [%fp-24],%i0
	b L1849
	nop
L1849:
	ret
	restore
	.align 8
LC217:
	.ascii "# %d \"%s\"\0"
	.align 8
LC218:
	.ascii " 1\0"
	.align 8
LC219:
	.ascii " 2\0"
	.align 8
LC220:
	.ascii " 3\0"
	.align 4
	.proc	020
_output_line_command:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	sethi %hi(_no_line_commands),%o0
	ld [%o0+%lo(_no_line_commands)],%o1
	cmp %o1,0
	bne L1865
	nop
	ld [%fp+68],%o0
	ld [%o0],%o1
	cmp %o1,0
	be L1865
	nop
	sethi %hi(_no_output),%o0
	ld [%o0+%lo(_no_output)],%o1
	cmp %o1,0
	bne L1865
	nop
	b L1864
	nop
L1865:
	ld [%fp+72],%o0
	ld [%fp+68],%o1
	ld [%o1+12],%o2
	st %o2,[%o0+12]
	b L1863
	nop
L1864:
	ld [%fp+76],%o0
	cmp %o0,0
	be L1866
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%o0+12],%o0
	ld [%o1+12],%o1
	cmp %o0,%o1
	bne L1867
	nop
	b L1863
	nop
L1867:
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%o0+12],%o0
	ld [%o1+12],%o1
	cmp %o0,%o1
	ble L1868
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%o1+12],%o2
	add %o2,8,%o1
	ld [%o0+12],%o0
	cmp %o0,%o1
	bge L1868
	nop
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%fp+72],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	cmp %o0,10
	bg L1869
	nop
	ld [%fp+72],%o0
	mov 10,%o1
	call _grow_outbuf,0
	nop
	b L1870
	nop
L1869:
L1870:
	nop
L1871:
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%o0+12],%o0
	ld [%o1+12],%o1
	cmp %o0,%o1
	ble L1872
	nop
	ld [%fp+72],%o1
	ld [%o1+24],%o0
	add %o0,1,%o2
	st %o2,[%o1+24]
	mov 10,%o1
	stb %o1,[%o0]
	ld [%fp+72],%o1
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	b L1871
	nop
L1872:
	b L1863
	nop
L1868:
L1866:
	ld [%fp+68],%o0
	ld [%o0+12],%o1
	cmp %o1,0
	bne L1873
	nop
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o0+24],%o0
	ld [%o1+20],%o1
	sub %o0,%o1,%o0
	ld [%fp+68],%o1
	ld [%o1+16],%o2
	cmp %o0,%o2
	bge L1873
	nop
	ld [%fp+68],%o0
	ld [%o0+24],%o1
	ldub [%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,10
	bne L1873
	nop
	ld [%fp+68],%o1
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o1+12],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+12]
	ld [%fp+68],%o1
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o1+24],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+24]
L1873:
	ld [%fp+68],%o1
	ld [%o1+4],%o0
	call _strlen,0
	nop
	add %o0,100,%o1
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
	ld [%fp+68],%o2
	ld [%fp+68],%o3
	ld [%fp-24],%o0
	sethi %hi(LC217),%o4
	or %o4,%lo(LC217),%o1
	ld [%o2+12],%o2
	ld [%o3+4],%o3
	call _sprintf,0
	nop
	ld [%fp+80],%o0
	cmp %o0,0
	be L1874
	nop
	ld [%fp+80],%o0
	cmp %o0,1
	bne L1875
	nop
	sethi %hi(LC218),%o0
	or %o0,%lo(LC218),%o1
	b L1876
	nop
L1875:
	sethi %hi(LC219),%o0
	or %o0,%lo(LC219),%o1
L1876:
	ld [%fp-24],%o0
	call _strcat,0
	nop
L1874:
	ld [%fp+68],%o0
	ldub [%o0+40],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L1877
	nop
	ld [%fp-24],%o0
	sethi %hi(LC220),%o2
	or %o2,%lo(LC220),%o1
	call _strcat,0
	nop
L1877:
	ld [%fp-24],%o0
	call _strlen,0
	nop
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	add %o0,1,%o1
	st %o1,[%fp-20]
	ld [%fp-24],%o1
	add %o0,%o1,%o0
	mov 10,%o1
	stb %o1,[%o0]
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%fp+72],%o2
	ld [%o1+24],%o1
	ld [%o2+20],%o2
	sub %o1,%o2,%o1
	ld [%o0+16],%o2
	sub %o2,%o1,%o0
	ld [%fp-20],%o2
	add %o2,1,%o1
	cmp %o0,%o1
	bg L1878
	nop
	ld [%fp-20],%o0
	add %o0,1,%o1
	ld [%fp+72],%o0
	call _grow_outbuf,0
	nop
	b L1879
	nop
L1878:
L1879:
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%o0+24],%o0
	ld [%o1+20],%o1
	cmp %o0,%o1
	bleu L1880
	nop
	ld [%fp+72],%o0
	mov -1,%o1
	ld [%o0+24],%o2
	add %o1,%o2,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	be L1880
	nop
	ld [%fp+72],%o1
	ld [%o1+24],%o0
	add %o0,1,%o2
	st %o2,[%o1+24]
	mov 10,%o1
	stb %o1,[%o0]
L1880:
	ld [%fp+72],%o1
	ld [%fp-24],%o0
	ld [%o1+24],%o1
	ld [%fp-20],%o2
	call _bcopy,0
	nop
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%o1+24],%o2
	ld [%fp-20],%o1
	add %o2,%o1,%o2
	st %o2,[%o0+24]
	ld [%fp+72],%o0
	ld [%fp+68],%o1
	ld [%o1+12],%o2
	st %o2,[%o0+12]
L1863:
	ret
	restore
	.align 8
LC221:
	.ascii "arguments given to macro `%s'\0"
	.align 8
LC222:
	.ascii "macro `%s' used without args\0"
	.align 8
LC223:
	.ascii "macro `%s' used with just one arg\0"
	.align 8
LC224:
	.ascii "macro `%s' used with only %d args\0"
	.align 8
LC225:
	.ascii "macro `%s' used with too many (%d) args\0"
	.align 8
LC226:
	.ascii "\\%03o\0"
	.align 4
	.proc	020
_macroexpand:
	!#PROLOGUE# 0
	save %sp,-160,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
	ld [%o0+24],%o1
	st %o1,[%fp-24]
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o0,%o1,%o0
	ld [%o0+12],%o1
	st %o1,[%fp-32]
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o1
	cmp %o1,198
	ble L1882
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o0,%o1,%o2
	ld [%o2+12],%o0
	call _line_for_error,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	sethi %hi(LC104),%o2
	or %o2,%lo(LC104),%o1
	call _error_with_line,0
	nop
	b L1881
	nop
L1882:
	ld [%fp+68],%o0
	ld [%o0+12],%o1
	cmp %o1,31
	be L1883
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	call _special_symbol,0
	nop
	b L1881
	nop
L1883:
	sethi %hi(_pcp_inside_if),%o0
	ld [%o0+%lo(_pcp_inside_if)],%o1
	cmp %o1,0
	be L1884
	nop
	sethi %hi(_pcp_outfile),%o0
	ld [%o0+%lo(_pcp_outfile)],%o1
	cmp %o1,0
	be L1884
	nop
	ld [%fp-24],%o0
	ld [%o0+8],%o1
	cmp %o1,0
	be L1884
	nop
	sethi %hi(_pcp_outfile),%o1
	ld [%fp+68],%o0
	ld [%o1+%lo(_pcp_outfile)],%o1
	call _dump_single_macro,0
	nop
L1884:
	ld [%fp-24],%o0
	ld [%o0],%o1
	st %o1,[%fp-20]
	ld [%fp-20],%o0
	cmp %o0,0
	bl L1885
	nop
	st %g0,[%fp-48]
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	sll %o1,5,%o2
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
	st %o1,[%fp-44]
	mov 0,%l1
L1886:
	ld [%fp-20],%o0
	cmp %l1,%o0
	bge L1887
	nop
	mov %l1,%o0
	sll %o0,5,%o1
	ld [%fp-44],%o2
	add %o1,%o2,%o0
	mov %l1,%o1
	sll %o1,5,%o2
	ld [%fp-44],%o3
	add %o2,%o3,%o1
	sethi %hi(LC23),%o2
	or %o2,%lo(LC23),%o3
	st %o3,[%o1+4]
	st %o3,[%o0]
	mov %l1,%o0
	sll %o0,5,%o1
	ld [%fp-44],%o2
	add %o1,%o2,%o0
	mov %l1,%o1
	sll %o1,5,%o2
	ld [%fp-44],%o3
	add %o2,%o3,%o1
	mov %l1,%o2
	sll %o2,5,%o3
	ld [%fp-44],%o4
	add %o3,%o4,%o2
	st %g0,[%o2+16]
	st %g0,[%o1+12]
	st %g0,[%o0+8]
	mov %l1,%o0
	sll %o0,5,%o1
	ld [%fp-44],%o2
	add %o1,%o2,%o0
	mov %l1,%o1
	sll %o1,5,%o2
	ld [%fp-44],%o3
	add %o2,%o3,%o1
	st %g0,[%o1+24]
	st %g0,[%o0+20]
	mov %l1,%o0
	sll %o0,5,%o1
	ld [%fp-44],%o2
	add %o1,%o2,%o0
	stb %g0,[%o0+30]
L1888:
	add %l1,1,%l1
	b L1886
	nop
L1887:
	mov 0,%l1
	st %g0,[%fp-36]
L1889:
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o1
	or %o1,%lo(_instack),%o2
	add %o0,%o2,%o1
	sethi %hi(_instack),%o0
	sethi %hi(_indepth),%o2
	ld [%o2+%lo(_indepth)],%o1
	mov %o1,%o3
	sll %o3,1,%o2
	add %o2,%o1,%o2
	sll %o2,2,%o3
	sub %o3,%o1,%o3
	sll %o3,2,%o1
	or %o0,%lo(_instack),%o2
	add %o1,%o2,%o0
	sethi %hi(_indepth),%o2
	ld [%o2+%lo(_indepth)],%o1
	mov %o1,%o3
	sll %o3,1,%o2
	add %o2,%o1,%o2
	sll %o2,2,%o3
	sub %o3,%o1,%o3
	sll %o3,2,%o1
	sethi %hi(_instack),%o3
	or %o3,%lo(_instack),%o2
	add %o1,%o2,%o1
	ld [%o1+24],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+24]
	ld [%fp-36],%o0
	cmp %o0,0
	be L1892
	nop
	b L1891
	nop
L1892:
	ld [%fp-20],%o0
	cmp %l1,%o0
	bl L1894
	nop
	ld [%fp-20],%o0
	cmp %o0,0
	bne L1893
	nop
	cmp %l1,0
	bne L1893
	nop
	b L1894
	nop
L1894:
	ld [%fp-20],%o1
	add %o1,-1,%o0
	cmp %l1,%o0
	bne L1895
	nop
	ld [%fp-24],%o0
	ldub [%o0+24],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L1895
	nop
	mov 1,%o0
	st %o0,[%fp-36]
L1895:
	mov %l1,%o0
	sll %o0,5,%o1
	ld [%fp-44],%o0
	add %o1,%o0,%o1
	mov %o1,%o0
	ld [%fp-36],%o1
	call _macarg,0
	nop
	st %o0,[%fp-48]
	b L1896
	nop
L1893:
	mov 0,%o0
	mov 0,%o1
	call _macarg,0
	nop
	st %o0,[%fp-48]
L1896:
	ld [%fp-48],%o0
	cmp %o0,0
	be L1897
	nop
	ld [%fp-32],%o0
	call _line_for_error,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp-48],%o1
	call _error_with_line,0
	nop
	b L1890
	nop
L1897:
	add %l1,1,%l1
L1891:
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o0,%o1,%o2
	ld [%o2+24],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,41
	be L1890
	nop
	b L1889
	nop
L1890:
	cmp %l1,1
	bne L1898
	nop
	ld [%fp-44],%o1
	ld [%o1],%o0
	ld [%fp-44],%o2
	ld [%o2+8],%o3
	add %o0,%o3,%o1
L1899:
	cmp %o0,%o1
	be L1900
	nop
	ldub [%o0],%o3
	and %o3,0xff,%o2
	sethi %hi(_is_space),%o4
	or %o4,%lo(_is_space),%o3
	ldub [%o2+%o3],%o4
	and %o4,0xff,%o2
	cmp %o2,0
	be L1900
	nop
	add %o0,1,%o0
	b L1899
	nop
L1900:
	cmp %o0,%o1
	bne L1901
	nop
	mov 0,%l1
L1901:
L1898:
	st %g0,[%fp-40]
	ld [%fp-20],%o0
	cmp %o0,0
	bne L1902
	nop
	cmp %l1,0
	ble L1902
	nop
	ld [%fp-48],%o0
	cmp %o0,0
	bne L1903
	nop
	ld [%fp+68],%o1
	sethi %hi(LC221),%o2
	or %o2,%lo(LC221),%o0
	ld [%o1+20],%o1
	call _error,0
	nop
L1903:
	b L1904
	nop
L1902:
	ld [%fp-20],%o0
	cmp %l1,%o0
	bge L1905
	nop
	ld [%fp-20],%o0
	cmp %o0,1
	bne L1906
	nop
	cmp %l1,0
	bne L1906
	nop
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	be L1906
	nop
	b L1907
	nop
L1906:
	ld [%fp-20],%o1
	add %o1,-1,%o0
	cmp %l1,%o0
	bne L1908
	nop
	ld [%fp-24],%o0
	ldub [%o0+24],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L1908
	nop
	mov 1,%o0
	st %o0,[%fp-40]
	b L1909
	nop
L1908:
	ld [%fp-48],%o0
	cmp %o0,0
	be L1910
	nop
	b L1911
	nop
L1910:
	cmp %l1,0
	bne L1912
	nop
	ld [%fp+68],%o1
	sethi %hi(LC222),%o2
	or %o2,%lo(LC222),%o0
	ld [%o1+20],%o1
	call _error,0
	nop
	b L1913
	nop
L1912:
	cmp %l1,1
	bne L1914
	nop
	ld [%fp+68],%o1
	sethi %hi(LC223),%o2
	or %o2,%lo(LC223),%o0
	ld [%o1+20],%o1
	call _error,0
	nop
	b L1915
	nop
L1914:
	ld [%fp+68],%o1
	sethi %hi(LC224),%o2
	or %o2,%lo(LC224),%o0
	ld [%o1+20],%o1
	mov %l1,%o2
	call _error,0
	nop
L1915:
L1913:
L1911:
L1909:
L1907:
	b L1916
	nop
L1905:
	ld [%fp-20],%o0
	cmp %l1,%o0
	ble L1917
	nop
	ld [%fp-48],%o0
	cmp %o0,0
	bne L1918
	nop
	ld [%fp+68],%o1
	sethi %hi(LC225),%o2
	or %o2,%lo(LC225),%o0
	ld [%o1+20],%o1
	mov %l1,%o2
	call _error,0
	nop
L1918:
L1917:
L1916:
L1904:
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o1
	or %o1,%lo(_instack),%o2
	add %o0,%o2,%o1
	sethi %hi(_instack),%o0
	sethi %hi(_indepth),%o2
	ld [%o2+%lo(_indepth)],%o1
	mov %o1,%o3
	sll %o3,1,%o2
	add %o2,%o1,%o2
	sll %o2,2,%o3
	sub %o3,%o1,%o3
	sll %o3,2,%o1
	or %o0,%lo(_instack),%o2
	add %o1,%o2,%o0
	sethi %hi(_indepth),%o2
	ld [%o2+%lo(_indepth)],%o1
	mov %o1,%o3
	sll %o3,1,%o2
	add %o2,%o1,%o2
	sll %o2,2,%o3
	sub %o3,%o1,%o3
	sll %o3,2,%o1
	sethi %hi(_instack),%o3
	or %o3,%lo(_instack),%o2
	add %o1,%o2,%o1
	ld [%o1+24],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+24]
	ld [%fp-20],%o0
	cmp %o0,0
	bne L1919
	nop
	ld [%fp-24],%o0
	ld [%o0+12],%l0
	ld [%fp-24],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-28]
	b L1920
	nop
L1919:
	ld [%fp-24],%o0
	ld [%o0+12],%l2
	ld [%fp-24],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-28]
	ld [%fp-24],%o0
	ld [%o0+28],%l5
L1921:
	cmp %l5,0
	be L1922
	nop
	ldub [%l5+4],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	be L1924
	nop
	ld [%l5+12],%o0
	mov %o0,%o1
	sll %o1,5,%o0
	ld [%fp-44],%o1
	add %o0,%o1,%o0
	ld [%fp-28],%o1
	ld [%o0+16],%o0
	add %o1,%o0,%o1
	st %o1,[%fp-28]
	b L1925
	nop
L1924:
	ldub [%l5+5],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	bne L1927
	nop
	ldub [%l5+6],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	bne L1927
	nop
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	bne L1927
	nop
	b L1926
	nop
L1927:
	ld [%l5+12],%o0
	mov %o0,%o1
	sll %o1,5,%o0
	ld [%fp-44],%o1
	add %o0,%o1,%o0
	ld [%fp-28],%o1
	ld [%o0+8],%o0
	add %o1,%o0,%o1
	st %o1,[%fp-28]
	b L1928
	nop
L1926:
	ld [%l5+12],%o0
	mov %o0,%o1
	sll %o1,5,%o0
	ld [%fp-44],%o1
	add %o0,%o1,%o0
	ld [%fp-28],%o1
	ld [%o0+12],%o0
	add %o1,%o0,%o1
	st %o1,[%fp-28]
L1928:
L1925:
	ld [%l5+12],%o0
	mov %o0,%o1
	sll %o1,5,%o0
	ld [%fp-44],%o1
	add %o0,%o1,%o0
	ldub [%o0+30],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,9
	bg L1929
	nop
	ld [%l5+12],%o0
	mov %o0,%o1
	sll %o1,5,%o0
	ld [%fp-44],%o1
	add %o0,%o1,%o0
	ld [%l5+12],%o1
	mov %o1,%o0
	sll %o0,5,%o1
	ld [%fp-44],%o2
	add %o1,%o2,%o0
	ld [%l5+12],%o1
	mov %o1,%o2
	sll %o2,5,%o1
	ld [%fp-44],%o2
	add %o1,%o2,%o1
	ldub [%o1+30],%o2
	add %o2,1,%o1
	mov %o1,%o2
	stb %o2,[%o0+30]
L1929:
L1923:
	ld [%l5],%l5
	b L1921
	nop
L1922:
	ld [%fp-28],%o0
	add %o0,1,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	mov %o0,%l0
	mov 0,%l4
	mov 0,%l3
	mov 0,%l6
	ld [%fp-24],%o0
	ld [%o0+28],%l5
L1930:
	cmp %l5,0
	be L1931
	nop
	ld [%l5+12],%o0
	mov %o0,%o1
	sll %o1,5,%o0
	ld [%fp-44],%o1
	add %o0,%o1,%l7
	mov 0,%l1
L1933:
	ld [%l5+8],%o0
	cmp %l1,%o0
	bge L1934
	nop
	ld [%fp-40],%o0
	cmp %o0,0
	be L1937
	nop
	ldub [%l5+7],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	be L1938
	nop
	ldub [%l5+5],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	bne L1936
	nop
	b L1938
	nop
L1938:
	cmp %l6,0
	be L1937
	nop
	ldub [%l6+7],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	be L1937
	nop
	ldub [%l6+6],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	bne L1936
	nop
	b L1937
	nop
L1937:
	add %l0,%l4,%o0
	ldub [%l2+%l3],%o1
	stb %o1,[%o0]
	add %l4,1,%l4
L1936:
L1935:
	add %l1,1,%l1
	add %l3,1,%l3
	b L1933
	nop
L1934:
	ldub [%l5+4],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	be L1939
	nop
	ld [%l7+8],%o0
	st %o0,[%fp-52]
	st %g0,[%fp-56]
	st %g0,[%fp-60]
	mov 0,%l1
L1940:
	ld [%fp-52],%o0
	cmp %l1,%o0
	bge L1941
	nop
	ld [%l7],%o1
	add %l1,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	st %o0,[%fp-64]
	sethi %hi(_is_space),%o1
	or %o1,%lo(_is_space),%o0
	ld [%fp-64],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L1941
	nop
	add %l1,1,%l1
	b L1940
	nop
L1941:
	nop
L1942:
	ld [%fp-52],%o0
	cmp %l1,%o0
	bge L1943
	nop
	ld [%fp-52],%o0
	ld [%l7],%o1
	add %o0,%o1,%o0
	ldub [%o0-1],%o1
	and %o1,0xff,%o0
	st %o0,[%fp-64]
	sethi %hi(_is_space),%o1
	or %o1,%lo(_is_space),%o0
	ld [%fp-64],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L1943
	nop
	ld [%fp-52],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-52]
	b L1942
	nop
L1943:
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	bne L1944
	nop
	add %l0,%l4,%o0
	mov 34,%o1
	stb %o1,[%o0]
	add %l4,1,%l4
L1944:
	nop
L1945:
	ld [%fp-52],%o0
	cmp %l1,%o0
	bge L1946
	nop
	ld [%l7],%o1
	add %l1,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	st %o0,[%fp-64]
	ld [%fp-64],%o0
	cmp %o0,10
	bne L1948
	nop
	ld [%l7],%o1
	add %l1,%o1,%o0
	ldub [%o0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	be L1948
	nop
	add %l1,1,%l1
	b L1947
	nop
L1948:
	ld [%fp-60],%o0
	cmp %o0,0
	bne L1949
	nop
	ld [%fp-64],%o0
	cmp %o0,10
	bne L1950
	nop
	ld [%l7],%o1
	add %l1,%o1,%o0
	ldub [%o0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	be L1951
	nop
	b L1949
	nop
L1950:
	sethi %hi(_is_space),%o1
	or %o1,%lo(_is_space),%o0
	ld [%fp-64],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	bne L1951
	nop
	b L1949
	nop
L1951:
L1952:
	ld [%fp-64],%o0
	cmp %o0,10
	bne L1954
	nop
	ld [%l7],%o1
	add %l1,%o1,%o0
	ldub [%o0+1],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_space),%o2
	or %o2,%lo(_is_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1954
	nop
	add %l1,2,%l1
	b L1955
	nop
L1954:
	ld [%fp-64],%o0
	cmp %o0,10
	be L1956
	nop
	sethi %hi(_is_space),%o1
	or %o1,%lo(_is_space),%o0
	ld [%fp-64],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L1956
	nop
	add %l1,1,%l1
	b L1957
	nop
L1956:
	b L1953
	nop
L1957:
L1955:
	ld [%l7],%o1
	add %l1,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	st %o0,[%fp-64]
	b L1952
	nop
L1953:
	add %l1,-1,%l1
	mov 32,%o0
	st %o0,[%fp-64]
L1949:
	ld [%fp-56],%o0
	cmp %o0,0
	be L1958
	nop
	st %g0,[%fp-56]
	b L1959
	nop
L1958:
	ld [%fp-64],%o0
	cmp %o0,92
	bne L1960
	nop
	mov 1,%o0
	st %o0,[%fp-56]
L1960:
	ld [%fp-60],%o0
	cmp %o0,0
	be L1961
	nop
	ld [%fp-64],%o0
	ld [%fp-60],%o1
	cmp %o0,%o1
	bne L1962
	nop
	st %g0,[%fp-60]
L1962:
	b L1963
	nop
L1961:
	ld [%fp-64],%o0
	cmp %o0,34
	be L1965
	nop
	ld [%fp-64],%o0
	cmp %o0,39
	be L1965
	nop
	b L1964
	nop
L1965:
	ld [%fp-64],%o0
	st %o0,[%fp-60]
L1964:
L1963:
L1959:
	ld [%fp-64],%o0
	cmp %o0,34
	be L1967
	nop
	ld [%fp-60],%o0
	cmp %o0,0
	be L1966
	nop
	ld [%fp-64],%o0
	cmp %o0,92
	be L1967
	nop
	b L1966
	nop
L1967:
	add %l0,%l4,%o0
	mov 92,%o1
	stb %o1,[%o0]
	add %l4,1,%l4
L1966:
	sethi %hi(___lc_ctype),%o1
	ld [%o1+%lo(___lc_ctype)],%o0
	ld [%o0+28],%o1
	cmp %o1,0
	bne L1969
	nop
	sethi %hi(___lc_ctype),%o1
	ld [%o1+%lo(___lc_ctype)],%o0
	ld [%fp-64],%o1
	mov %o1,%o2
	sll %o2,2,%o1
	ld [%o0+64],%o2
	add %o1,%o2,%o0
	ld [%o0],%o1
	and %o1,128,%o0
	cmp %o0,0
	bne L1970
	nop
	b L1968
	nop
L1969:
	sethi %hi(___lc_ctype),%o0
	ld [%o0+%lo(___lc_ctype)],%o3
	sethi %hi(___lc_ctype),%o2
	ld [%fp-64],%o0
	mov 128,%o1
	ld [%o2+%lo(___lc_ctype)],%o2
	ld [%o3+28],%o3
	call %o3,0
	nop
	cmp %o0,0
	bne L1970
	nop
	b L1968
	nop
L1970:
	add %l0,%l4,%o0
	ldub [%fp-61],%o1
	stb %o1,[%o0]
	add %l4,1,%l4
	b L1971
	nop
L1968:
	add %l0,%l4,%o1
	mov %o1,%o0
	sethi %hi(LC226),%o2
	or %o2,%lo(LC226),%o1
	ld [%fp-64],%o2
	call _sprintf,0
	nop
	add %l4,4,%l4
L1971:
L1947:
	add %l1,1,%l1
	b L1945
	nop
L1946:
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	bne L1972
	nop
	add %l0,%l4,%o0
	mov 34,%o1
	stb %o1,[%o0]
	add %l4,1,%l4
L1972:
	b L1973
	nop
L1939:
	ldub [%l5+5],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	bne L1975
	nop
	ldub [%l5+6],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	bne L1975
	nop
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	bne L1975
	nop
	b L1974
	nop
L1975:
	ld [%l7],%o0
	st %o0,[%fp-60]
	ld [%fp-60],%o0
	ld [%l7+8],%o1
	add %o0,%o1,%o0
	st %o0,[%fp-56]
	ldub [%l5+5],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	be L1976
	nop
L1977:
	ld [%fp-60],%o0
	ld [%fp-56],%o1
	cmp %o0,%o1
	be L1978
	nop
	ld [%fp-60],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_space),%o2
	or %o2,%lo(_is_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1978
	nop
	ld [%fp-60],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-60]
	b L1977
	nop
L1978:
	nop
L1979:
	ld [%fp-60],%o0
	ld [%fp-56],%o1
	cmp %o0,%o1
	be L1980
	nop
	ld [%fp-60],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1980
	nop
	add %l0,%l4,%o0
	ld [%fp-60],%o1
	add %o1,1,%o2
	st %o2,[%fp-60]
	ldub [%o1],%o1
	stb %o1,[%o0]
	add %l4,1,%l4
	b L1979
	nop
L1980:
	ld [%fp-60],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L1981
	nop
	mov 1,%o0
	ld [%fp-60],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,45
	bne L1981
	nop
	ld [%fp-60],%o0
	add %o0,2,%o1
	st %o1,[%fp-60]
L1981:
L1976:
	ldub [%l5+6],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	be L1982
	nop
L1983:
	ld [%fp-60],%o0
	ld [%fp-56],%o1
	cmp %o0,%o1
	be L1984
	nop
	mov -1,%o0
	ld [%fp-56],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_space),%o2
	or %o2,%lo(_is_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L1985
	nop
	ld [%fp-56],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-56]
	b L1986
	nop
L1985:
	mov -1,%o0
	ld [%fp-56],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,45
	bne L1987
	nop
	ld [%fp-56],%o0
	add %o0,-1,%o1
	st %o1,[%fp-52]
L1988:
	ld [%fp-52],%o0
	ld [%fp-60],%o1
	cmp %o0,%o1
	be L1989
	nop
	mov -1,%o0
	ld [%fp-52],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L1989
	nop
	ld [%fp-52],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-52]
	b L1988
	nop
L1989:
	ld [%fp-52],%o1
	add %o1,1,%o0
	ld [%fp-56],%o1
	sub %o1,%o0,%o0
	and %o0,1,%o1
	cmp %o1,0
	be L1990
	nop
	ld [%fp-56],%o0
	add %o0,-2,%o1
	st %o1,[%fp-56]
	b L1991
	nop
L1990:
	b L1984
	nop
L1991:
	b L1992
	nop
L1987:
	b L1984
	nop
L1992:
L1986:
	b L1983
	nop
L1984:
L1982:
	add %l0,%l4,%o1
	ld [%fp-56],%o0
	ld [%fp-60],%o3
	sub %o0,%o3,%o2
	ld [%fp-60],%o0
	call _bcopy,0
	nop
	ld [%fp-56],%o0
	ld [%fp-60],%o1
	sub %o0,%o1,%o0
	add %l4,%o0,%l4
	b L1993
	nop
L1974:
	add %l0,%l4,%o1
	ld [%l7+4],%o0
	ld [%l7+12],%o2
	call _bcopy,0
	nop
	ld [%l7+12],%o0
	add %l4,%o0,%l4
	ldub [%l7+30],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,1
	ble L1994
	nop
	ldub [%l7+28],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	ble L1994
	nop
	mov 1,%o0
	stb %o0,[%l7+30]
	ld [%l7+4],%o0
	ld [%l7+12],%o1
	call _delete_newlines,0
	nop
	st %o0,[%l7+12]
L1994:
L1993:
L1973:
	ld [%fp-28],%o0
	cmp %l4,%o0
	ble L1995
	nop
	call _abort,0
	nop
L1995:
L1932:
	mov %l5,%l6
	ld [%l5],%l5
	b L1930
	nop
L1931:
	nop
	mov %l3,%l1
L1996:
	ld [%fp-24],%o0
	ld [%o0+4],%o1
	cmp %l1,%o1
	bge L1997
	nop
	ldub [%l2+%l1],%o1
	and %o1,0xff,%o0
	cmp %o0,41
	bne L1999
	nop
	st %g0,[%fp-40]
L1999:
	ld [%fp-40],%o0
	cmp %o0,0
	be L2001
	nop
	cmp %l6,0
	be L2001
	nop
	ldub [%l6+7],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	be L2001
	nop
	ldub [%l6+6],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,0
	bne L2000
	nop
	b L2001
	nop
L2001:
	add %l0,%l4,%o0
	ldub [%l2+%l1],%o1
	stb %o1,[%o0]
	add %l4,1,%l4
L2000:
L1998:
	add %l1,1,%l1
	b L1996
	nop
L1997:
	stb %g0,[%l0+%l4]
	st %l4,[%fp-28]
	mov 0,%l1
L2002:
	ld [%fp-20],%o0
	cmp %l1,%o0
	bge L2003
	nop
	mov %l1,%o0
	sll %o0,5,%o1
	ld [%fp-44],%o2
	add %o1,%o2,%o0
	ld [%o0+20],%o1
	cmp %o1,0
	be L2005
	nop
	mov %l1,%o0
	sll %o0,5,%o1
	ld [%fp-44],%o0
	add %o1,%o0,%o1
	ld [%o1+20],%o0
	call _free,0
	nop
L2005:
	mov %l1,%o0
	sll %o0,5,%o1
	ld [%fp-44],%o2
	add %o1,%o2,%o0
	ld [%o0+24],%o1
	cmp %o1,0
	be L2006
	nop
	mov %l1,%o0
	sll %o0,5,%o1
	ld [%fp-44],%o0
	add %o1,%o0,%o1
	ld [%o1+24],%o0
	call _free,0
	nop
L2006:
L2004:
	add %l1,1,%l1
	b L2002
	nop
L2003:
L1920:
	b L2007
	nop
L1885:
	ld [%fp-24],%o0
	ld [%o0+12],%l0
	ld [%fp-24],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-28]
L2007:
	sethi %hi(_indepth),%o1
	sethi %hi(_indepth),%o2
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o4
	add %o4,1,%o3
	mov %o3,%o1
	st %o1,[%o2+%lo(_indepth)]
	mov %o1,%o3
	sll %o3,1,%o2
	add %o2,%o1,%o2
	sll %o2,2,%o3
	sub %o3,%o1,%o3
	sll %o3,2,%o1
	sethi %hi(_instack),%o3
	or %o3,%lo(_instack),%o2
	add %o1,%o2,%o0
	st %g0,[%o0]
	st %g0,[%o0+4]
	st %g0,[%o0+12]
	st %l0,[%o0+20]
	ld [%fp-28],%o1
	st %o1,[%o0+16]
	st %l0,[%o0+24]
	ld [%fp-20],%o1
	cmp %o1,0
	ble L2008
	nop
	mov %l0,%o1
	b L2009
	nop
L2008:
	mov 0,%o1
L2009:
	st %o1,[%o0+36]
	ld [%fp+68],%o1
	st %o1,[%o0+28]
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o2
	st %o2,[%o0+32]
	stb %g0,[%o0+40]
	sethi %hi(_traditional),%o1
	ld [%o1+%lo(_traditional)],%o2
	cmp %o2,0
	bne L2010
	nop
	ld [%fp+68],%o1
	mov 32,%o2
	st %o2,[%o1+12]
L2010:
L1881:
	ret
	restore
	.align 8
LC227:
	.ascii "unterminated macro call\0"
	.align 4
	.proc	0102
_macarg:
	!#PROLOGUE# 0
	save %sp,-200,%sp
	!#PROLOGUE# 1
	mov %i0,%l0
	st %i1,[%fp+72]
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-20]
	st %g0,[%fp-24]
	st %g0,[%fp-28]
	st %g0,[%fp-32]
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%fp-20],%o2
	ld [%o1+20],%o1
	ld [%o2+16],%o2
	add %o1,%o2,%o1
	add %fp,-24,%o2
	add %fp,-28,%o3
	add %fp,-32,%o4
	ld [%o0+24],%o0
	ld [%fp+72],%o5
	call _macarg1,0
	nop
	st %o0,[%fp-36]
	ld [%fp-20],%o0
	ld [%o0],%o1
	cmp %o1,0
	be L2013
	nop
	ld [%fp-28],%o0
	cmp %o0,0
	bne L2012
	nop
	ld [%fp-32],%o0
	cmp %o0,0
	bne L2012
	nop
	b L2013
	nop
L2013:
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o0+20],%o0
	ld [%o1+16],%o1
	add %o0,%o1,%o0
	ld [%fp-36],%o1
	cmp %o1,%o0
	be L2012
	nop
	cmp %l0,0
	be L2014
	nop
	ld [%fp-20],%o0
	ld [%o0+24],%o1
	st %o1,[%l0]
	ld [%fp-20],%o0
	ld [%fp-36],%o1
	ld [%o0+24],%o0
	sub %o1,%o0,%o1
	st %o1,[%l0+8]
	ldub [%fp-25],%o0
	stb %o0,[%l0+28]
L2014:
	ld [%fp-20],%o0
	ld [%fp-36],%o1
	st %o1,[%o0+24]
	b L2015
	nop
L2012:
	ld [%fp-20],%o0
	ld [%fp-36],%o1
	ld [%o0+24],%o0
	sub %o1,%o0,%o1
	st %o1,[%fp-40]
	ld [%fp-28],%o0
	st %o0,[%fp-44]
	ld [%fp-40],%o0
	ld [%fp-44],%o1
	add %o0,%o1,%o0
	add %o0,1,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-48]
	st %g0,[%fp-52]
	ld [%fp-20],%o1
	ld [%o1+24],%o0
	ld [%fp-48],%o1
	ld [%fp-40],%o2
	call _bcopy,0
	nop
	ld [%fp-20],%o0
	ld [%fp-36],%o1
	st %o1,[%o0+24]
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o1+12],%o2
	ld [%fp-28],%o1
	add %o2,%o1,%o2
	st %o2,[%o0+12]
L2016:
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o0+20],%o0
	ld [%o1+16],%o1
	add %o0,%o1,%o0
	ld [%fp-36],%o1
	cmp %o1,%o0
	bne L2017
	nop
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o0,%o1,%o0
	ld [%o0+28],%o1
	cmp %o1,0
	bne L2018
	nop
	ld [%fp-48],%o0
	call _free,0
	nop
	sethi %hi(LC227),%o0
	or %o0,%lo(LC227),%i0
	b L2011
	nop
L2018:
	ld [%fp-20],%o0
	ld [%o0+28],%o1
	mov 31,%o0
	st %o0,[%o1+12]
	ld [%fp-20],%o0
	ld [%o0+36],%o1
	cmp %o1,0
	be L2019
	nop
	ld [%fp-20],%o1
	ld [%o1+36],%o0
	call _free,0
	nop
L2019:
	sethi %hi(_indepth),%o0
	sethi %hi(_indepth),%o1
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o3
	add %o3,-1,%o2
	mov %o2,%o0
	st %o0,[%o1+%lo(_indepth)]
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-20]
	st %g0,[%fp-28]
	st %g0,[%fp-32]
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%fp-20],%o2
	ld [%o1+20],%o1
	ld [%o2+16],%o2
	add %o1,%o2,%o1
	add %fp,-24,%o2
	add %fp,-28,%o3
	add %fp,-32,%o4
	ld [%o0+24],%o0
	ld [%fp+72],%o5
	call _macarg1,0
	nop
	st %o0,[%fp-36]
	ld [%fp-40],%o0
	st %o0,[%fp-52]
	ld [%fp-20],%o0
	ld [%fp-36],%o1
	ld [%o0+24],%o2
	sub %o1,%o2,%o0
	ld [%fp-40],%o1
	add %o1,%o0,%o0
	st %o0,[%fp-40]
	ld [%fp-44],%o0
	ld [%fp-28],%o1
	add %o0,%o1,%o0
	st %o0,[%fp-44]
	ld [%fp-40],%o0
	ld [%fp-44],%o1
	add %o0,%o1,%o0
	add %o0,1,%o1
	ld [%fp-48],%o0
	call _xrealloc,0
	nop
	st %o0,[%fp-48]
	ld [%fp-20],%o0
	ld [%fp-48],%o1
	ld [%fp-40],%o2
	add %o1,%o2,%o1
	ld [%fp-20],%o2
	ld [%fp-36],%o3
	ld [%o2+24],%o4
	sub %o3,%o4,%o2
	sub %o1,%o2,%o1
	ld [%fp-20],%o2
	ld [%fp-36],%o3
	ld [%o2+24],%o4
	sub %o3,%o4,%o2
	ld [%o0+24],%o0
	call _bcopy,0
	nop
	ld [%fp-20],%o0
	ld [%fp-36],%o1
	st %o1,[%o0+24]
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%o1+12],%o2
	ld [%fp-28],%o1
	add %o2,%o1,%o2
	st %o2,[%o0+12]
	b L2016
	nop
L2017:
	cmp %l0,0
	be L2020
	nop
	ld [%fp-48],%o0
	st %o0,[%l0]
	ld [%fp-40],%o0
	st %o0,[%l0+8]
	ld [%fp-48],%o0
	st %o0,[%l0+20]
	ldub [%fp-25],%o0
	stb %o0,[%l0+28]
	ldub [%fp-29],%o0
	stb %o0,[%l0+29]
	ld [%fp-28],%o0
	cmp %o0,0
	bne L2022
	nop
	ld [%fp-32],%o0
	cmp %o0,0
	bne L2022
	nop
	b L2021
	nop
L2022:
	ld [%fp-20],%o0
	ld [%o0],%o1
	cmp %o1,0
	be L2021
	nop
	ld [%l0],%o0
	ld [%fp-52],%o1
	add %o0,%o1,%o0
	ld [%l0+8],%o1
	ld [%fp-52],%o2
	sub %o1,%o2,%o1
	ld [%fp-28],%o2
	call _discard_comments,0
	nop
	ld [%fp-52],%o1
	add %o1,%o0,%o0
	st %o0,[%l0+8]
L2021:
	ld [%l0],%o0
	ld [%l0+8],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
	ld [%fp-40],%o0
	ld [%fp-44],%o1
	add %o0,%o1,%o0
	ld [%l0+8],%o1
	cmp %o1,%o0
	ble L2023
	nop
	call _abort,0
	nop
L2023:
L2020:
L2015:
	cmp %l0,0
	be L2024
	nop
	ld [%l0],%o0
	ld [%l0+8],%o2
	add %o0,%o2,%o1
	add %fp,-104,%o0
	st %o0,[%sp+64]
	ld [%l0],%o0
	mov 1,%o2
	mov 0,%o3
	call _expand_to_temp_buffer,0
	nop
	unimp 44
	ld [%fp-84],%o0
	st %o0,[%l0+4]
	ld [%fp-88],%o0
	st %o0,[%l0+12]
	ld [%fp-84],%o0
	st %o0,[%l0+24]
	ld [%l0],%l1
	ld [%l0+8],%o0
	add %l1,%o0,%l2
L2025:
	cmp %l1,%l2
	be L2026
	nop
	ldub [%l1],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_space),%o2
	or %o2,%lo(_is_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L2026
	nop
	add %l1,1,%l1
	b L2025
	nop
L2026:
	nop
L2027:
	cmp %l1,%l2
	be L2028
	nop
	ldub [%l2-1],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_space),%o2
	or %o2,%lo(_is_space),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L2028
	nop
	add %l2,-1,%l2
	b L2027
	nop
L2028:
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	bne L2029
	nop
	mov 2,%l3
	b L2030
	nop
L2029:
	mov 0,%l3
L2030:
	nop
L2031:
	cmp %l1,%l2
	be L2032
	nop
	ldub [%l1],%l4
	add %l1,1,%l1
	add %l3,1,%l3
	and %l4,0xff,%o0
	cmp %o0,34
	be L2034
	nop
	and %l4,0xff,%o0
	cmp %o0,92
	be L2034
	nop
	b L2033
	nop
L2034:
	add %l3,1,%l3
	b L2035
	nop
L2033:
	sethi %hi(___lc_ctype),%o1
	ld [%o1+%lo(___lc_ctype)],%o0
	ld [%o0+28],%o1
	cmp %o1,0
	bne L2037
	nop
	sethi %hi(___lc_ctype),%o1
	ld [%o1+%lo(___lc_ctype)],%o0
	and %l4,0xff,%o1
	mov %o1,%o2
	sll %o2,2,%o1
	ld [%o0+64],%o2
	add %o1,%o2,%o0
	ld [%o0],%o1
	and %o1,128,%o0
	cmp %o0,0
	bne L2036
	nop
	b L2038
	nop
L2037:
	sethi %hi(___lc_ctype),%o0
	ld [%o0+%lo(___lc_ctype)],%o3
	and %l4,0xff,%o0
	sethi %hi(___lc_ctype),%o2
	mov 128,%o1
	ld [%o2+%lo(___lc_ctype)],%o2
	ld [%o3+28],%o3
	call %o3,0
	nop
	cmp %o0,0
	bne L2036
	nop
	b L2038
	nop
L2038:
	add %l3,3,%l3
L2036:
L2035:
	b L2031
	nop
L2032:
	st %l3,[%l0+16]
L2024:
	mov 0,%i0
	b L2011
	nop
L2011:
	ret
	restore
	.align 4
	.proc	0114
_macarg1:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	mov %i1,%l0
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	st %i4,[%fp+84]
	st %i5,[%fp+88]
	ld [%fp+68],%l1
L2040:
	cmp %l1,%l0
	bgeu L2041
	nop
	ldub [%l1],%o1
	and %o1,0xff,%o0
	cmp %o0,41
	be L2044
	nop
	cmp %o0,41
	bg L2078
	nop
	cmp %o0,39
	be L2062
	nop
	cmp %o0,39
	bg L2043
	nop
	cmp %o0,10
	be L2049
	nop
	cmp %o0,34
	be L2063
	nop
	b L2077
	nop
L2078:
	cmp %o0,47
	be L2050
	nop
	cmp %o0,47
	bg L2079
	nop
	cmp %o0,44
	be L2074
	nop
	b L2077
	nop
L2079:
	cmp %o0,92
	be L2046
	nop
	b L2077
	nop
L2043:
	ld [%fp+76],%o1
	ld [%fp+76],%o0
	ld [%fp+76],%o1
	ld [%o1],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0]
	b L2042
	nop
L2044:
	ld [%fp+76],%o1
	ld [%fp+76],%o0
	ld [%fp+76],%o1
	ld [%o1],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0]
	cmp %o2,0
	bge L2045
	nop
	mov %l1,%i0
	b L2039
	nop
L2045:
	b L2042
	nop
L2046:
	add %l1,1,%o0
	cmp %o0,%l0
	bgeu L2047
	nop
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	be L2047
	nop
	add %l1,1,%l1
	ldub [%l1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L2048
	nop
	ld [%fp+80],%o1
	ld [%fp+80],%o0
	ld [%fp+80],%o1
	ld [%o1],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0]
L2048:
L2047:
	b L2042
	nop
L2049:
	ld [%fp+80],%o1
	ld [%fp+80],%o0
	ld [%fp+80],%o1
	ld [%o1],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0]
	b L2042
	nop
L2050:
	ldub [%l1+1],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L2051
	nop
	ldub [%l1+2],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L2051
	nop
	add %l1,1,%o1
	mov %o1,%o0
	call _newline_fix,0
	nop
L2051:
	sethi %hi(_cplusplus_comments),%o0
	ld [%o0+%lo(_cplusplus_comments)],%o1
	cmp %o1,0
	be L2052
	nop
	ldub [%l1+1],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L2052
	nop
	ld [%fp+84],%o0
	mov 1,%o1
	st %o1,[%o0]
	add %l1,2,%l1
L2053:
	cmp %l1,%l0
	bgeu L2054
	nop
	ldub [%l1],%o1
	and %o1,0xff,%o0
	add %l1,1,%l1
	cmp %o0,10
	be L2054
	nop
	b L2053
	nop
L2054:
	ld [%fp+80],%o1
	ld [%fp+80],%o0
	ld [%fp+80],%o1
	ld [%o1],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0]
	b L2042
	nop
L2052:
	ldub [%l1+1],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L2056
	nop
	add %l1,1,%o0
	cmp %o0,%l0
	bgeu L2056
	nop
	b L2055
	nop
L2056:
	b L2042
	nop
L2055:
	ld [%fp+84],%o0
	mov 1,%o1
	st %o1,[%o0]
	add %l1,2,%l1
L2057:
	add %l1,1,%o0
	cmp %o0,%l0
	bgeu L2058
	nop
	ldub [%l1],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L2059
	nop
	ldub [%l1+1],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L2059
	nop
	ldub [%l1+2],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L2059
	nop
	add %l1,1,%o1
	mov %o1,%o0
	call _newline_fix,0
	nop
L2059:
	ldub [%l1],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L2060
	nop
	ldub [%l1+1],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L2060
	nop
	b L2058
	nop
L2060:
	ldub [%l1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L2061
	nop
	ld [%fp+80],%o1
	ld [%fp+80],%o0
	ld [%fp+80],%o1
	ld [%o1],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0]
L2061:
	add %l1,1,%l1
	b L2057
	nop
L2058:
	b L2042
	nop
L2062:
L2063:
	ldub [%l1],%o0
	and %o0,0xff,%o1
	st %o1,[%fp-20]
	add %l1,1,%l1
L2064:
	add %l1,1,%o0
	cmp %o0,%l0
	bgeu L2065
	nop
	ldub [%l1],%o1
	and %o1,0xff,%o0
	ld [%fp-20],%o1
	cmp %o0,%o1
	be L2065
	nop
	ldub [%l1],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L2067
	nop
	add %l1,1,%l1
	ldub [%l1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L2068
	nop
	ld [%fp+80],%o1
	ld [%fp+80],%o0
	ld [%fp+80],%o1
	ld [%o1],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0]
L2068:
	nop
L2069:
	ldub [%l1],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L2070
	nop
	ldub [%l1+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L2070
	nop
	add %l1,2,%l1
	b L2069
	nop
L2070:
	b L2071
	nop
L2067:
	ldub [%l1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L2072
	nop
	ld [%fp+80],%o1
	ld [%fp+80],%o0
	ld [%fp+80],%o1
	ld [%o1],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0]
	ld [%fp-20],%o0
	cmp %o0,39
	bne L2073
	nop
	b L2065
	nop
L2073:
L2072:
L2071:
L2066:
	add %l1,1,%l1
	b L2064
	nop
L2065:
	b L2042
	nop
L2074:
	ld [%fp+76],%o0
	ld [%o0],%o1
	cmp %o1,0
	bne L2075
	nop
	ld [%fp+88],%o0
	cmp %o0,0
	bne L2075
	nop
	mov %l1,%i0
	b L2039
	nop
L2075:
	b L2042
	nop
L2077:
L2042:
	add %l1,1,%l1
	b L2040
	nop
L2041:
	mov %l1,%i0
	b L2039
	nop
L2039:
	ret
	restore
	.align 4
	.proc	04
_discard_comments:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	ld [%fp+76],%o0
	cmp %o0,0
	ble L2081
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	add %o0,%o1,%l0
	ld [%fp+76],%o0
	add %l0,%o0,%l1
	ld [%fp+68],%l2
L2082:
	cmp %l2,%l0
	be L2083
	nop
	add %l1,-1,%l1
	add %l0,-1,%l0
	ldub [%l0],%o0
	stb %o0,[%l1]
	b L2082
	nop
L2083:
L2081:
	ld [%fp+68],%o0
	ld [%fp+76],%o1
	add %o0,%o1,%l0
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	add %o0,%o1,%l2
	ld [%fp+76],%o0
	add %l2,%o0,%l2
	ld [%fp+68],%l1
L2084:
	cmp %l0,%l2
	bgeu L2085
	nop
	ldub [%l0],%o0
	and %o0,0xff,%l3
	mov %l3,%o1
	mov %o1,%o0
	stb %o0,[%l1]
	add %l0,1,%l0
	add %l1,1,%l1
	cmp %l3,39
	be L2101
	nop
	cmp %l3,39
	bg L2112
	nop
	cmp %l3,10
	be L2087
	nop
	cmp %l3,34
	be L2102
	nop
	b L2111
	nop
L2112:
	cmp %l3,47
	be L2090
	nop
	cmp %l3,92
	be L2088
	nop
	b L2111
	nop
L2087:
	mov 10,%o0
	stb %o0,[%l1]
	add %l1,1,%l1
	b L2086
	nop
L2088:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L2089
	nop
	add %l1,-1,%l1
	add %l0,1,%l0
L2089:
	b L2086
	nop
L2090:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L2091
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L2091
	nop
	mov %l0,%o0
	call _newline_fix,0
	nop
L2091:
	sethi %hi(_cplusplus_comments),%o0
	ld [%o0+%lo(_cplusplus_comments)],%o1
	cmp %o1,0
	be L2092
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L2092
	nop
	add %l1,-1,%l1
	add %l0,1,%l0
L2093:
	cmp %l0,%l2
	bgeu L2094
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	add %l0,1,%l0
	cmp %o0,10
	be L2094
	nop
	b L2093
	nop
L2094:
	b L2086
	nop
L2092:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L2096
	nop
	add %l0,1,%o0
	cmp %o0,%l2
	bgeu L2096
	nop
	b L2095
	nop
L2096:
	b L2086
	nop
L2095:
	add %l1,-1,%l1
	add %l0,1,%l0
L2097:
	add %l0,1,%o0
	cmp %o0,%l2
	bgeu L2098
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L2099
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L2099
	nop
	ldub [%l0+2],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L2099
	nop
	add %l0,1,%o1
	mov %o1,%o0
	call _newline_fix,0
	nop
L2099:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,42
	bne L2100
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,47
	bne L2100
	nop
	b L2098
	nop
L2100:
	add %l0,1,%l0
	b L2097
	nop
L2098:
	add %l0,2,%l0
	b L2086
	nop
L2101:
L2102:
	st %l3,[%fp-20]
L2103:
	cmp %l0,%l2
	bgeu L2104
	nop
	ldub [%l0],%o0
	and %o0,0xff,%l3
	mov %l3,%o1
	mov %o1,%o0
	stb %o0,[%l1]
	add %l0,1,%l0
	add %l1,1,%l1
	ld [%fp-20],%o0
	cmp %l3,%o0
	bne L2105
	nop
	b L2104
	nop
L2105:
	cmp %l3,10
	bne L2106
	nop
	ld [%fp-20],%o0
	cmp %o0,39
	bne L2106
	nop
	b L2104
	nop
L2106:
	cmp %l3,92
	bne L2107
	nop
	cmp %l0,%l2
	bgeu L2107
	nop
L2108:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L2109
	nop
	ldub [%l0+1],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L2109
	nop
	add %l0,2,%l0
	b L2108
	nop
L2109:
	ldub [%l0],%o0
	stb %o0,[%l1]
	add %l0,1,%l0
	add %l1,1,%l1
L2107:
	b L2103
	nop
L2104:
	b L2086
	nop
L2111:
L2086:
	b L2084
	nop
L2085:
	ld [%fp+68],%o1
	sub %l1,%o1,%o0
	mov %o0,%i0
	b L2080
	nop
L2080:
	ret
	restore
	.align 4
	.proc	04
_delete_newlines:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
	ld [%fp+68],%o4
	ld [%fp+72],%o5
	add %o4,%o5,%o2
	ld [%fp+68],%o1
L2114:
	cmp %o0,%o2
	bgeu L2115
	nop
	ldub [%o0],%o4
	and %o4,0xff,%o3
	mov %o3,%o5
	mov %o5,%o4
	stb %o4,[%o1]
	add %o0,1,%o0
	add %o1,1,%o1
	cmp %o3,34
	be L2120
	nop
	cmp %o3,34
	bg L2127
	nop
	cmp %o3,10
	be L2117
	nop
	b L2126
	nop
L2127:
	cmp %o3,39
	be L2119
	nop
	b L2126
	nop
L2117:
	ldub [%o0],%o5
	and %o5,0xff,%o4
	cmp %o4,10
	bne L2118
	nop
	add %o0,1,%o0
	add %o1,-1,%o1
L2118:
	b L2116
	nop
L2119:
L2120:
	st %o3,[%fp-20]
L2121:
	cmp %o0,%o2
	bgeu L2122
	nop
	ldub [%o0],%o4
	and %o4,0xff,%o3
	mov %o3,%o5
	mov %o5,%o4
	stb %o4,[%o1]
	add %o0,1,%o0
	add %o1,1,%o1
	ld [%fp-20],%o4
	cmp %o3,%o4
	bne L2123
	nop
	b L2122
	nop
L2123:
	cmp %o3,10
	bne L2124
	nop
	ld [%fp-20],%o4
	cmp %o4,39
	bne L2124
	nop
	b L2122
	nop
L2124:
	b L2121
	nop
L2122:
	b L2116
	nop
L2126:
L2116:
	b L2114
	nop
L2115:
	ld [%fp+68],%o5
	sub %o1,%o5,%o4
	mov %o4,%i0
	b L2113
	nop
L2113:
	ret
	restore
	.align 8
LC228:
	.ascii "%s:%d: \0"
	.align 4
	.global _error
	.proc	020
_error:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	st %g0,[%fp-24]
	call _print_containing_files,0
	nop
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o1
	st %o1,[%fp-20]
L2129:
	ld [%fp-20],%o0
	cmp %o0,0
	bl L2130
	nop
	ld [%fp-20],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	ld [%o0+%o1],%o0
	cmp %o0,0
	be L2132
	nop
	ld [%fp-20],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-24]
	b L2130
	nop
L2132:
L2131:
	ld [%fp-20],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L2129
	nop
L2130:
	ld [%fp-24],%o0
	cmp %o0,0
	be L2133
	nop
	ld [%fp-24],%o2
	ld [%fp-24],%o3
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC228),%o4
	or %o4,%lo(LC228),%o1
	ld [%o2+4],%o2
	ld [%o3+12],%o3
	call _fprintf,0
	nop
L2133:
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	ld [%fp+68],%o1
	ld [%fp+72],%o2
	ld [%fp+76],%o3
	ld [%fp+80],%o4
	call _fprintf,0
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC66),%o2
	or %o2,%lo(LC66),%o1
	call _fprintf,0
	nop
	sethi %hi(_errors),%o1
	sethi %hi(_errors),%o0
	sethi %hi(_errors),%o1
	ld [%o1+%lo(_errors)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_errors)]
L2128:
	ret
	restore
	.align 8
LC229:
	.ascii "%s: %s\12\0"
	.align 8
LC230:
	.ascii "%s: undocumented I/O error\12\0"
	.align 4
	.proc	020
_error_from_errno:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %g0,[%fp-24]
	call _print_containing_files,0
	nop
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o1
	st %o1,[%fp-20]
L2135:
	ld [%fp-20],%o0
	cmp %o0,0
	bl L2136
	nop
	ld [%fp-20],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	ld [%o0+%o1],%o0
	cmp %o0,0
	be L2138
	nop
	ld [%fp-20],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-24]
	b L2136
	nop
L2138:
L2137:
	ld [%fp-20],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L2135
	nop
L2136:
	ld [%fp-24],%o0
	cmp %o0,0
	be L2139
	nop
	ld [%fp-24],%o2
	ld [%fp-24],%o3
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC228),%o4
	or %o4,%lo(LC228),%o1
	ld [%o2+4],%o2
	ld [%o3+12],%o3
	call _fprintf,0
	nop
L2139:
	sethi %hi(_errno),%o0
	sethi %hi(_sys_nerr),%o1
	ld [%o0+%lo(_errno)],%o0
	ld [%o1+%lo(_sys_nerr)],%o1
	cmp %o0,%o1
	bge L2140
	nop
	sethi %hi(_errno),%o1
	ld [%o1+%lo(_errno)],%o0
	mov %o0,%o1
	sll %o1,2,%o3
	sethi %hi(_sys_errlist),%o0
	or %o0,%lo(_sys_errlist),%o4
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC229),%o2
	or %o2,%lo(LC229),%o1
	ld [%fp+68],%o2
	ld [%o3+%o4],%o3
	call _fprintf,0
	nop
	b L2141
	nop
L2140:
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC230),%o2
	or %o2,%lo(LC230),%o1
	ld [%fp+68],%o2
	call _fprintf,0
	nop
L2141:
	sethi %hi(_errors),%o1
	sethi %hi(_errors),%o0
	sethi %hi(_errors),%o1
	ld [%o1+%lo(_errors)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_errors)]
L2134:
	ret
	restore
	.align 8
LC231:
	.ascii "warning: \0"
	.align 4
	.global _warning
	.proc	020
_warning:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	st %g0,[%fp-24]
	sethi %hi(_inhibit_warnings),%o0
	ld [%o0+%lo(_inhibit_warnings)],%o1
	cmp %o1,0
	be L2143
	nop
	b L2142
	nop
L2143:
	sethi %hi(_warnings_are_errors),%o0
	ld [%o0+%lo(_warnings_are_errors)],%o1
	cmp %o1,0
	be L2144
	nop
	sethi %hi(_errors),%o1
	sethi %hi(_errors),%o0
	sethi %hi(_errors),%o1
	ld [%o1+%lo(_errors)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_errors)]
L2144:
	call _print_containing_files,0
	nop
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o1
	st %o1,[%fp-20]
L2145:
	ld [%fp-20],%o0
	cmp %o0,0
	bl L2146
	nop
	ld [%fp-20],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	ld [%o0+%o1],%o0
	cmp %o0,0
	be L2148
	nop
	ld [%fp-20],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-24]
	b L2146
	nop
L2148:
L2147:
	ld [%fp-20],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L2145
	nop
L2146:
	ld [%fp-24],%o0
	cmp %o0,0
	be L2149
	nop
	ld [%fp-24],%o2
	ld [%fp-24],%o3
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC228),%o4
	or %o4,%lo(LC228),%o1
	ld [%o2+4],%o2
	ld [%o3+12],%o3
	call _fprintf,0
	nop
L2149:
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC231),%o2
	or %o2,%lo(LC231),%o1
	call _fprintf,0
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	ld [%fp+68],%o1
	ld [%fp+72],%o2
	ld [%fp+76],%o3
	ld [%fp+80],%o4
	call _fprintf,0
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC66),%o2
	or %o2,%lo(LC66),%o1
	call _fprintf,0
	nop
L2142:
	ret
	restore
	.align 4
	.proc	020
_error_with_line:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	st %i4,[%fp+84]
	st %g0,[%fp-24]
	call _print_containing_files,0
	nop
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o1
	st %o1,[%fp-20]
L2151:
	ld [%fp-20],%o0
	cmp %o0,0
	bl L2152
	nop
	ld [%fp-20],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	ld [%o0+%o1],%o0
	cmp %o0,0
	be L2154
	nop
	ld [%fp-20],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-24]
	b L2152
	nop
L2154:
L2153:
	ld [%fp-20],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L2151
	nop
L2152:
	ld [%fp-24],%o0
	cmp %o0,0
	be L2155
	nop
	ld [%fp-24],%o2
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC228),%o3
	or %o3,%lo(LC228),%o1
	ld [%o2+4],%o2
	ld [%fp+68],%o3
	call _fprintf,0
	nop
L2155:
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	ld [%fp+80],%o3
	ld [%fp+84],%o4
	call _fprintf,0
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC66),%o2
	or %o2,%lo(LC66),%o1
	call _fprintf,0
	nop
	sethi %hi(_errors),%o1
	sethi %hi(_errors),%o0
	sethi %hi(_errors),%o1
	ld [%o1+%lo(_errors)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_errors)]
L2150:
	ret
	restore
	.align 4
	.global _pedwarn
	.proc	020
_pedwarn:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	sethi %hi(_pedantic_errors),%o0
	ld [%o0+%lo(_pedantic_errors)],%o1
	cmp %o1,0
	be L2157
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	ld [%fp+80],%o3
	call _error,0
	nop
	b L2158
	nop
L2157:
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	ld [%fp+80],%o3
	call _warning,0
	nop
L2158:
L2156:
	ret
	restore
	.align 4
	.proc	020
_pedwarn_with_file_and_line:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	st %i4,[%fp+84]
	st %i5,[%fp+88]
	sethi %hi(_pedantic_errors),%o0
	ld [%o0+%lo(_pedantic_errors)],%o1
	cmp %o1,0
	bne L2160
	nop
	sethi %hi(_inhibit_warnings),%o0
	ld [%o0+%lo(_inhibit_warnings)],%o1
	cmp %o1,0
	be L2160
	nop
	b L2159
	nop
L2160:
	ld [%fp+68],%o0
	cmp %o0,0
	be L2161
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC228),%o2
	or %o2,%lo(LC228),%o1
	ld [%fp+68],%o2
	ld [%fp+72],%o3
	call _fprintf,0
	nop
L2161:
	sethi %hi(_pedantic_errors),%o0
	ld [%o0+%lo(_pedantic_errors)],%o1
	cmp %o1,0
	bne L2163
	nop
	sethi %hi(_warnings_are_errors),%o0
	ld [%o0+%lo(_warnings_are_errors)],%o1
	cmp %o1,0
	bne L2163
	nop
	b L2162
	nop
L2163:
	sethi %hi(_errors),%o1
	sethi %hi(_errors),%o0
	sethi %hi(_errors),%o1
	ld [%o1+%lo(_errors)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_errors)]
L2162:
	sethi %hi(_pedantic_errors),%o0
	ld [%o0+%lo(_pedantic_errors)],%o1
	cmp %o1,0
	bne L2164
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC231),%o2
	or %o2,%lo(LC231),%o1
	call _fprintf,0
	nop
L2164:
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	ld [%fp+76],%o1
	ld [%fp+80],%o2
	ld [%fp+84],%o3
	ld [%fp+88],%o4
	call _fprintf,0
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC66),%o2
	or %o2,%lo(LC66),%o1
	call _fprintf,0
	nop
L2159:
	ret
	restore
	.align 8
LC232:
	.ascii "In file included\0"
	.align 8
LC233:
	.ascii ",\0"
	.align 8
LC234:
	.ascii " from %s:%d\0"
	.align 8
LC235:
	.ascii ":\12\0"
	.align 4
	.proc	020
_print_containing_files:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %g0,[%fp-20]
	mov 1,%o0
	st %o0,[%fp-28]
	sethi %hi(_last_error_tick),%o0
	sethi %hi(_input_file_stack_tick),%o1
	ld [%o0+%lo(_last_error_tick)],%o0
	ld [%o1+%lo(_input_file_stack_tick)],%o1
	cmp %o0,%o1
	bne L2166
	nop
	b L2165
	nop
L2166:
	nop
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o1
	st %o1,[%fp-24]
L2167:
	ld [%fp-24],%o0
	cmp %o0,0
	bl L2168
	nop
	ld [%fp-24],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	ld [%o0+%o1],%o0
	cmp %o0,0
	be L2170
	nop
	ld [%fp-24],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-20]
	b L2168
	nop
L2170:
L2169:
	ld [%fp-24],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
	b L2167
	nop
L2168:
	ld [%fp-20],%o0
	cmp %o0,0
	bne L2171
	nop
	b L2165
	nop
L2171:
	nop
	ld [%fp-24],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
L2172:
	ld [%fp-24],%o0
	cmp %o0,0
	bl L2173
	nop
	ld [%fp-24],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	ld [%o0+%o1],%o0
	cmp %o0,0
	be L2175
	nop
	ld [%fp-24],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-20]
	ld [%fp-28],%o0
	cmp %o0,0
	be L2176
	nop
	st %g0,[%fp-28]
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC232),%o2
	or %o2,%lo(LC232),%o1
	call _fprintf,0
	nop
	b L2177
	nop
L2176:
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC233),%o2
	or %o2,%lo(LC233),%o1
	call _fprintf,0
	nop
L2177:
	ld [%fp-20],%o2
	ld [%fp-20],%o3
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC234),%o4
	or %o4,%lo(LC234),%o1
	ld [%o2+4],%o2
	ld [%o3+12],%o3
	call _fprintf,0
	nop
L2175:
L2174:
	ld [%fp-24],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
	b L2172
	nop
L2173:
	ld [%fp-28],%o0
	cmp %o0,0
	bne L2178
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC235),%o2
	or %o2,%lo(LC235),%o1
	call _fprintf,0
	nop
L2178:
	sethi %hi(_last_error_tick),%o0
	sethi %hi(_input_file_stack_tick),%o1
	ld [%o1+%lo(_input_file_stack_tick)],%o2
	st %o2,[%o0+%lo(_last_error_tick)]
L2165:
	ret
	restore
	.align 4
	.proc	04
_line_for_error:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	st %o0,[%fp-24]
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o1
	st %o1,[%fp-20]
L2180:
	ld [%fp-20],%o0
	cmp %o0,0
	bl L2181
	nop
	ld [%fp-20],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	ld [%o0+%o1],%o0
	cmp %o0,0
	be L2183
	nop
	ld [%fp-24],%i0
	b L2179
	nop
L2183:
	ld [%fp-20],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	ld [%fp-20],%o0
	cmp %o0,0
	bge L2184
	nop
	mov 0,%i0
	b L2179
	nop
L2184:
	ld [%fp-20],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o0,%o1,%o0
	ld [%o0+12],%o1
	st %o1,[%fp-24]
L2182:
	b L2180
	nop
L2181:
	call _abort,0
	nop
	mov 0,%i0
	b L2179
	nop
L2179:
	ret
	restore
	.align 4
	.proc	04
_grow_outbuf:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	mov %i0,%l1
	mov %i1,%l0
	ld [%l1+24],%o0
	ld [%l1+20],%o1
	sub %o0,%o1,%o0
	ld [%l1+16],%o1
	sub %o1,%o0,%o0
	cmp %o0,%l0
	ble L2186
	nop
	mov 0,%i0
	b L2185
	nop
L2186:
	ld [%l1+16],%o0
	mov %o0,%o1
	sll %o1,1,%o0
	st %o0,[%l1+16]
	mov %l0,%o1
	sll %o1,1,%o0
	add %o0,%l0,%o0
	mov %o0,%o1
	srl %o1,31,%o0
	add %o1,%o0,%o1
	sra %o1,1,%o0
	ld [%l1+24],%o1
	ld [%l1+20],%o2
	sub %o1,%o2,%o1
	add %o0,%o1,%o0
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	ld [%l1+16],%o1
	cmp %o0,%o1
	ble L2187
	nop
	ld [%fp-20],%o0
	st %o0,[%l1+16]
L2187:
	ld [%l1+20],%o0
	ld [%l1+16],%o1
	call _xrealloc,0
	nop
	mov %o0,%l2
	mov %l2,%o0
	cmp %o0,0
	bne L2188
	nop
	call _memory_full,0
	nop
L2188:
	ld [%l1+24],%o0
	ld [%l1+20],%o1
	sub %o0,%o1,%o0
	add %l2,%o0,%o1
	st %o1,[%l1+24]
	st %l2,[%l1+20]
	mov 0,%i0
	b L2185
	nop
L2185:
	ret
	restore
	.align 4
	.proc	0110
_install:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	st %i4,[%fp+84]
	st %i5,[%fp+88]
	ld [%fp+72],%o0
	cmp %o0,0
	bge L2190
	nop
	ld [%fp+68],%l3
L2191:
	ldub [%l3],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L2192
	nop
	add %l3,1,%l3
	b L2191
	nop
L2192:
	ld [%fp+68],%o0
	sub %l3,%o0,%o1
	st %o1,[%fp+72]
L2190:
	ld [%fp+88],%o0
	cmp %o0,0
	bge L2193
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	mov 1403,%o2
	call _hashf,0
	nop
	st %o0,[%fp+88]
L2193:
	ld [%fp+72],%o0
	add %o0,29,%l1
	mov %l1,%o0
	call _xmalloc,0
	nop
	mov %o0,%l0
	ld [%fp+88],%l2
	mov %l2,%o1
	sll %o1,2,%o0
	sethi %hi(_hashtab),%o2
	or %o2,%lo(_hashtab),%o1
	add %o1,%o0,%o0
	st %o0,[%l0+8]
	mov %l2,%o1
	sll %o1,2,%o0
	sethi %hi(_hashtab),%o2
	or %o2,%lo(_hashtab),%o1
	ld [%o0+%o1],%o0
	st %o0,[%l0]
	mov %l2,%o1
	sll %o1,2,%o0
	sethi %hi(_hashtab),%o2
	or %o2,%lo(_hashtab),%o1
	st %l0,[%o0+%o1]
	st %g0,[%l0+4]
	ld [%l0],%o0
	cmp %o0,0
	be L2194
	nop
	ld [%l0],%o0
	st %l0,[%o0+4]
L2194:
	ld [%fp+76],%o0
	st %o0,[%l0+12]
	ld [%fp+72],%o0
	st %o0,[%l0+16]
	ld [%l0+12],%o0
	cmp %o0,30
	bne L2195
	nop
	ld [%fp+80],%o0
	st %o0,[%l0+24]
	b L2196
	nop
L2195:
	ld [%fp+84],%o0
	st %o0,[%l0+24]
L2196:
	add %l0,28,%o0
	st %o0,[%l0+20]
	ld [%l0+20],%l3
	ld [%fp+68],%l4
	mov 0,%l1
L2197:
	ld [%fp+72],%o0
	cmp %l1,%o0
	bge L2198
	nop
	ldub [%l4],%o0
	stb %o0,[%l3]
	add %l4,1,%l4
	add %l3,1,%l3
L2199:
	add %l1,1,%l1
	b L2197
	nop
L2198:
	ld [%l0+20],%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
	mov %l0,%i0
	b L2189
	nop
L2189:
	ret
	restore
	.align 4
	.global _lookup
	.proc	0110
_lookup:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	ld [%fp+72],%o0
	cmp %o0,0
	bge L2201
	nop
	ld [%fp+68],%l0
L2202:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L2203
	nop
L2204:
	add %l0,1,%l0
	b L2202
	nop
L2203:
	ld [%fp+68],%o0
	sub %l0,%o0,%o1
	st %o1,[%fp+72]
L2201:
	ld [%fp+76],%o0
	cmp %o0,0
	bge L2205
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	mov 1403,%o2
	call _hashf,0
	nop
	st %o0,[%fp+76]
L2205:
	ld [%fp+76],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_hashtab),%o2
	or %o2,%lo(_hashtab),%o1
	ld [%o0+%o1],%l1
L2206:
	cmp %l1,0
	be L2207
	nop
	ld [%l1+16],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	bne L2208
	nop
	ld [%l1+20],%o0
	ld [%fp+68],%o1
	ld [%fp+72],%o2
	call _strncmp,0
	nop
	cmp %o0,0
	bne L2208
	nop
	mov %l1,%i0
	b L2200
	nop
L2208:
	ld [%l1],%l1
	b L2206
	nop
L2207:
	mov 0,%i0
	b L2200
	nop
L2200:
	ret
	restore
	.align 4
	.proc	020
_delete_macro:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	cmp %o1,0
	be L2210
	nop
	ld [%fp+68],%o1
	ld [%o1+4],%o0
	ld [%fp+68],%o1
	ld [%o1],%o2
	st %o2,[%o0]
L2210:
	ld [%fp+68],%o0
	ld [%o0],%o1
	cmp %o1,0
	be L2211
	nop
	ld [%fp+68],%o1
	ld [%o1],%o0
	ld [%fp+68],%o1
	ld [%o1+4],%o2
	st %o2,[%o0+4]
L2211:
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	ld [%fp+68],%o0
	ld [%o1],%o1
	cmp %o0,%o1
	bne L2212
	nop
	ld [%fp+68],%o1
	ld [%o1+8],%o0
	ld [%fp+68],%o1
	ld [%o1],%o2
	st %o2,[%o0]
L2212:
	ld [%fp+68],%o0
	call _free,0
	nop
L2209:
	ret
	restore
	.align 4
	.proc	04
_hashf:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	mov %i0,%l0
	mov %i1,%l1
	st %i2,[%fp+76]
	mov 0,%l2
L2214:
	add %l1,-1,%l1
	cmp %l1,-1
	be L2215
	nop
	sll %l2,2,%l2
	ldub [%l0],%o1
	and %o1,0xff,%o0
	add %l2,%o0,%l2
	add %l0,1,%l0
	b L2214
	nop
L2215:
	sethi %hi(2147483647),%o1
	or %o1,%lo(2147483647),%o0
	and %l2,%o0,%o1
	mov %o1,%o0
	ld [%fp+76],%o1
	call .rem,0
	nop
	mov %o0,%i0
	b L2213
	nop
L2213:
	ret
	restore
	.align 8
LC236:
	.ascii "#define %s\0"
	.align 8
LC237:
	.ascii "(\0"
	.align 8
LC238:
	.ascii ", \0"
	.align 8
LC239:
	.ascii ")\0"
	.align 8
LC240:
	.ascii " #\0"
	.align 8
LC241:
	.ascii " ## \0"
	.align 4
	.proc	020
_dump_single_macro:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	mov %i0,%l0
	st %i1,[%fp+72]
	ld [%l0+24],%l1
	ld [%fp+72],%o0
	sethi %hi(LC236),%o2
	or %o2,%lo(LC236),%o1
	ld [%l0+20],%o2
	call _fprintf,0
	nop
	ld [%l1],%o0
	cmp %o0,0
	bl L2217
	nop
	ld [%fp+72],%o0
	sethi %hi(LC237),%o2
	or %o2,%lo(LC237),%o1
	call _fprintf,0
	nop
	st %g0,[%fp-32]
L2218:
	ld [%fp-32],%o0
	ld [%l1],%o1
	cmp %o0,%o1
	bge L2219
	nop
	mov %l1,%o0
	ld [%fp-32],%o1
	ld [%fp+72],%o2
	call _dump_arg_n,0
	nop
	ld [%fp-32],%o1
	add %o1,1,%o0
	ld [%l1],%o1
	cmp %o0,%o1
	bge L2221
	nop
	ld [%fp+72],%o0
	sethi %hi(LC238),%o2
	or %o2,%lo(LC238),%o1
	call _fprintf,0
	nop
L2221:
L2220:
	ld [%fp-32],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-32]
	b L2218
	nop
L2219:
	ld [%fp+72],%o0
	sethi %hi(LC239),%o2
	or %o2,%lo(LC239),%o1
	call _fprintf,0
	nop
L2217:
	ld [%fp+72],%o0
	sethi %hi(LC88),%o2
	or %o2,%lo(LC88),%o1
	call _fprintf,0
	nop
	st %g0,[%fp-24]
	st %g0,[%fp-28]
	ld [%l1+28],%o0
	st %o0,[%fp-20]
L2222:
	ld [%fp-20],%o0
	cmp %o0,0
	be L2223
	nop
	ld [%fp-20],%o2
	ld [%l1+12],%o0
	ld [%fp-24],%o1
	ld [%o2+8],%o2
	ld [%fp+72],%o3
	call _dump_defn_1,0
	nop
	ld [%fp-20],%o0
	ld [%o0+8],%o1
	cmp %o1,0
	be L2225
	nop
	st %g0,[%fp-28]
L2225:
	ld [%fp-20],%o0
	ld [%fp-24],%o1
	ld [%o0+8],%o0
	add %o1,%o0,%o1
	st %o1,[%fp-24]
	ld [%fp-20],%o0
	ldub [%o0+4],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L2226
	nop
	ld [%fp+72],%o0
	sethi %hi(LC240),%o2
	or %o2,%lo(LC240),%o1
	call _fprintf,0
	nop
L2226:
	ld [%fp-20],%o0
	ldub [%o0+5],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L2227
	nop
	ld [%fp-28],%o0
	cmp %o0,0
	bne L2227
	nop
	ld [%fp+72],%o0
	sethi %hi(LC241),%o2
	or %o2,%lo(LC241),%o1
	call _fprintf,0
	nop
L2227:
	st %g0,[%fp-28]
	ld [%fp-20],%o1
	mov %l1,%o0
	ld [%o1+12],%o1
	ld [%fp+72],%o2
	call _dump_arg_n,0
	nop
	ld [%fp-20],%o0
	ldub [%o0+6],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,0
	be L2228
	nop
	ld [%fp+72],%o0
	sethi %hi(LC241),%o2
	or %o2,%lo(LC241),%o1
	call _fprintf,0
	nop
	mov 1,%o0
	st %o0,[%fp-28]
L2228:
L2224:
	ld [%fp-20],%o0
	ld [%o0],%o1
	st %o1,[%fp-20]
	b L2222
	nop
L2223:
	ld [%l1+4],%o0
	ld [%fp-24],%o1
	sub %o0,%o1,%o2
	ld [%l1+12],%o0
	ld [%fp-24],%o1
	ld [%fp+72],%o3
	call _dump_defn_1,0
	nop
	ld [%fp+72],%o0
	sethi %hi(LC66),%o2
	or %o2,%lo(LC66),%o1
	call _fprintf,0
	nop
L2216:
	ret
	restore
	.align 4
	.proc	020
_dump_all_macros:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	nop
	st %g0,[%fp-20]
L2230:
	ld [%fp-20],%o0
	cmp %o0,1402
	bg L2231
	nop
	ld [%fp-20],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_hashtab),%o2
	or %o2,%lo(_hashtab),%o1
	ld [%o0+%o1],%l0
L2233:
	cmp %l0,0
	be L2234
	nop
	ld [%l0+12],%o0
	cmp %o0,31
	bne L2236
	nop
	mov %l0,%o0
	sethi %hi(__iob+32),%o2
	or %o2,%lo(__iob+32),%o1
	call _dump_single_macro,0
	nop
L2236:
L2235:
	ld [%l0],%l0
	b L2233
	nop
L2234:
L2232:
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L2230
	nop
L2231:
L2229:
	ret
	restore
	.align 4
	.proc	020
_dump_defn_1:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	st %o0,[%fp-20]
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	add %o0,%o1,%o0
	ld [%fp+76],%o1
	add %o0,%o1,%o0
	st %o0,[%fp-24]
L2238:
	ld [%fp-20],%o0
	ld [%fp-24],%o1
	cmp %o0,%o1
	bgeu L2239
	nop
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	be L2240
	nop
	ld [%fp+80],%o1
	ld [%fp+80],%o0
	ld [%fp+80],%o1
	ld [%o1],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0]
	cmp %o2,0
	bge L2241
	nop
	ld [%fp-20],%o0
	ldub [%o0],%o2
	and %o2,0xff,%o1
	mov %o1,%o0
	ld [%fp+80],%o1
	call __flsbuf,0
	nop
	b L2242
	nop
L2241:
	ld [%fp+80],%o1
	ld [%o1+4],%o0
	add %o0,1,%o2
	st %o2,[%o1+4]
	ld [%fp-20],%o1
	ldub [%o1],%o2
	stb %o2,[%o0]
L2242:
	b L2243
	nop
L2240:
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,34
	be L2245
	nop
	ld [%fp-20],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,39
	be L2245
	nop
	b L2244
	nop
L2245:
	ld [%fp-20],%o0
	ld [%fp-24],%o1
	mov 0,%o2
	mov 0,%o3
	mov 0,%o4
	mov 0,%o5
	call _skip_quoted_string,0
	nop
	st %o0,[%fp-28]
	ld [%fp-28],%o0
	ld [%fp-20],%o2
	sub %o0,%o2,%o1
	ld [%fp-20],%o0
	mov 1,%o2
	ld [%fp+80],%o3
	call _fwrite,0
	nop
	ld [%fp-28],%o0
	add %o0,-1,%o1
	st %o1,[%fp-20]
L2244:
L2243:
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L2238
	nop
L2239:
L2237:
	ret
	restore
	.align 4
	.proc	020
_dump_arg_n:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	ld [%fp+68],%o0
	ld [%o0+32],%l0
L2247:
	ld [%fp+72],%o1
	add %o1,1,%o0
	ld [%fp+68],%o1
	ld [%o1],%o2
	cmp %o0,%o2
	bge L2248
	nop
	mov %l0,%o0
	mov 32,%o1
	call _index,0
	nop
	add %o0,1,%l0
	ld [%fp+72],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp+72]
	b L2247
	nop
L2248:
	nop
L2249:
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L2250
	nop
	ldub [%l0],%o1
	and %o1,0xff,%o0
	cmp %o0,44
	be L2250
	nop
	ld [%fp+76],%o1
	ld [%fp+76],%o0
	ld [%fp+76],%o1
	ld [%o1],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0]
	cmp %o2,0
	bge L2251
	nop
	ldub [%l0],%o0
	and %o0,0xff,%o1
	mov %o1,%o0
	ld [%fp+76],%o1
	call __flsbuf,0
	nop
	b L2252
	nop
L2251:
	ld [%fp+76],%o1
	ld [%o1+4],%o0
	add %o0,1,%o2
	st %o2,[%o1+4]
	ldub [%l0],%o1
	stb %o1,[%o0]
L2252:
	add %l0,1,%l0
	b L2249
	nop
L2250:
L2246:
	ret
	restore
	.align 4
	.proc	020
_initialize_char_syntax:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	nop
	mov 97,%o0
L2254:
	cmp %o0,122
	bg L2255
	nop
	sethi %hi(_is_idchar-32),%o2
	or %o2,%lo(_is_idchar-32),%o1
	mov 1,%o2
	stb %o2,[%o0+%o1]
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	mov 1,%o2
	stb %o2,[%o0+%o1]
	sethi %hi(_is_idstart-32),%o2
	or %o2,%lo(_is_idstart-32),%o1
	mov 1,%o2
	stb %o2,[%o0+%o1]
	sethi %hi(_is_idstart),%o2
	or %o2,%lo(_is_idstart),%o1
	mov 1,%o2
	stb %o2,[%o0+%o1]
L2256:
	add %o0,1,%o0
	b L2254
	nop
L2255:
	nop
	mov 48,%o0
L2257:
	cmp %o0,57
	bg L2258
	nop
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	mov 1,%o2
	stb %o2,[%o0+%o1]
L2259:
	add %o0,1,%o0
	b L2257
	nop
L2258:
	sethi %hi(_is_idchar),%o1
	mov 95,%o2
	or %o1,%lo(_is_idchar),%o3
	add %o2,%o3,%o1
	mov 1,%o2
	stb %o2,[%o1]
	sethi %hi(_is_idstart),%o1
	mov 95,%o2
	or %o1,%lo(_is_idstart),%o3
	add %o2,%o3,%o1
	mov 1,%o2
	stb %o2,[%o1]
	sethi %hi(_is_idchar),%o1
	mov 36,%o2
	or %o1,%lo(_is_idchar),%o3
	add %o2,%o3,%o1
	sethi %hi(_dollars_in_ident),%o2
	mov 3,%o3
	or %o2,%lo(_dollars_in_ident),%o4
	add %o3,%o4,%o2
	ldub [%o2],%o3
	stb %o3,[%o1]
	sethi %hi(_is_idstart),%o1
	mov 36,%o2
	or %o1,%lo(_is_idstart),%o3
	add %o2,%o3,%o1
	sethi %hi(_dollars_in_ident),%o2
	mov 3,%o3
	or %o2,%lo(_dollars_in_ident),%o4
	add %o3,%o4,%o2
	ldub [%o2],%o3
	stb %o3,[%o1]
	sethi %hi(_is_hor_space),%o1
	mov 32,%o2
	or %o1,%lo(_is_hor_space),%o3
	add %o2,%o3,%o1
	mov 1,%o2
	stb %o2,[%o1]
	sethi %hi(_is_hor_space),%o1
	mov 9,%o2
	or %o1,%lo(_is_hor_space),%o3
	add %o2,%o3,%o1
	mov 1,%o2
	stb %o2,[%o1]
	sethi %hi(_is_hor_space),%o1
	mov 11,%o2
	or %o1,%lo(_is_hor_space),%o3
	add %o2,%o3,%o1
	mov 1,%o2
	stb %o2,[%o1]
	sethi %hi(_is_hor_space),%o1
	mov 12,%o2
	or %o1,%lo(_is_hor_space),%o3
	add %o2,%o3,%o1
	mov 1,%o2
	stb %o2,[%o1]
	sethi %hi(_is_hor_space),%o1
	mov 13,%o2
	or %o1,%lo(_is_hor_space),%o3
	add %o2,%o3,%o1
	mov 1,%o2
	stb %o2,[%o1]
	sethi %hi(_is_space),%o1
	mov 32,%o2
	or %o1,%lo(_is_space),%o3
	add %o2,%o3,%o1
	mov 1,%o2
	stb %o2,[%o1]
	sethi %hi(_is_space),%o1
	mov 9,%o2
	or %o1,%lo(_is_space),%o3
	add %o2,%o3,%o1
	mov 1,%o2
	stb %o2,[%o1]
	sethi %hi(_is_space),%o1
	mov 11,%o2
	or %o1,%lo(_is_space),%o3
	add %o2,%o3,%o1
	mov 1,%o2
	stb %o2,[%o1]
	sethi %hi(_is_space),%o1
	mov 12,%o2
	or %o1,%lo(_is_space),%o3
	add %o2,%o3,%o1
	mov 1,%o2
	stb %o2,[%o1]
	sethi %hi(_is_space),%o1
	mov 10,%o2
	or %o1,%lo(_is_space),%o3
	add %o2,%o3,%o1
	mov 1,%o2
	stb %o2,[%o1]
	sethi %hi(_is_space),%o1
	mov 13,%o2
	or %o1,%lo(_is_space),%o3
	add %o2,%o3,%o1
	mov 1,%o2
	stb %o2,[%o1]
L2253:
	ret
	restore
	.align 8
LC242:
	.ascii "__LINE__\0"
	.align 8
LC243:
	.ascii "__DATE__\0"
	.align 8
LC244:
	.ascii "__FILE__\0"
	.align 8
LC245:
	.ascii "__BASE_FILE__\0"
	.align 8
LC246:
	.ascii "__INCLUDE_LEVEL__\0"
	.align 8
LC247:
	.ascii "__VERSION__\0"
	.align 8
LC248:
	.ascii "__SIZE_TYPE__\0"
	.align 8
LC249:
	.ascii "__PTRDIFF_TYPE__ \0"
	.align 8
LC250:
	.ascii "__WCHAR_TYPE__\0"
	.align 8
LC251:
	.ascii "__TIME__\0"
	.align 8
LC252:
	.ascii "__STDC__\0"
	.align 8
LC253:
	.ascii "__OBJC__\0"
	.align 8
LC254:
	.ascii " __BASE_FILE__ \"%s\"\12\0"
	.align 8
LC255:
	.ascii " __VERSION__ \"%s\"\12\0"
	.align 8
LC256:
	.ascii " __SIZE_TYPE__ %s\12\0"
	.align 8
LC257:
	.ascii " __PTRDIFF_TYPE__ %s\12\0"
	.align 8
LC258:
	.ascii " __WCHAR_TYPE__ %s\12\0"
	.align 8
LC259:
	.ascii " __DATE__ \"%s %2d %4d\"\12\0"
	.align 8
LC260:
	.ascii " __TIME__ \"%02d:%02d:%02d\"\12\0"
	.align 8
LC261:
	.ascii " __STDC__ 1\0"
	.align 8
LC262:
	.ascii " __OBJC__ 1\0"
	.align 4
	.proc	020
_initialize_builtins:
	!#PROLOGUE# 0
	save %sp,-2168,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	sethi %hi(LC242),%o1
	or %o1,%lo(LC242),%o0
	mov -1,%o1
	mov 20,%o2
	mov 0,%o3
	mov 0,%o4
	mov -1,%o5
	call _install,0
	nop
	sethi %hi(LC243),%o1
	or %o1,%lo(LC243),%o0
	mov -1,%o1
	mov 21,%o2
	mov 0,%o3
	mov 0,%o4
	mov -1,%o5
	call _install,0
	nop
	sethi %hi(LC244),%o1
	or %o1,%lo(LC244),%o0
	mov -1,%o1
	mov 22,%o2
	mov 0,%o3
	mov 0,%o4
	mov -1,%o5
	call _install,0
	nop
	sethi %hi(LC245),%o1
	or %o1,%lo(LC245),%o0
	mov -1,%o1
	mov 23,%o2
	mov 0,%o3
	mov 0,%o4
	mov -1,%o5
	call _install,0
	nop
	sethi %hi(LC246),%o1
	or %o1,%lo(LC246),%o0
	mov -1,%o1
	mov 24,%o2
	mov 0,%o3
	mov 0,%o4
	mov -1,%o5
	call _install,0
	nop
	sethi %hi(LC247),%o1
	or %o1,%lo(LC247),%o0
	mov -1,%o1
	mov 25,%o2
	mov 0,%o3
	mov 0,%o4
	mov -1,%o5
	call _install,0
	nop
	sethi %hi(LC248),%o1
	or %o1,%lo(LC248),%o0
	mov -1,%o1
	mov 26,%o2
	mov 0,%o3
	mov 0,%o4
	mov -1,%o5
	call _install,0
	nop
	sethi %hi(LC249),%o1
	or %o1,%lo(LC249),%o0
	mov -1,%o1
	mov 27,%o2
	mov 0,%o3
	mov 0,%o4
	mov -1,%o5
	call _install,0
	nop
	sethi %hi(LC250),%o1
	or %o1,%lo(LC250),%o0
	mov -1,%o1
	mov 28,%o2
	mov 0,%o3
	mov 0,%o4
	mov -1,%o5
	call _install,0
	nop
	sethi %hi(LC251),%o1
	or %o1,%lo(LC251),%o0
	mov -1,%o1
	mov 29,%o2
	mov 0,%o3
	mov 0,%o4
	mov -1,%o5
	call _install,0
	nop
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	bne L2261
	nop
	sethi %hi(LC252),%o1
	or %o1,%lo(LC252),%o0
	mov -1,%o1
	mov 30,%o2
	mov 1,%o3
	mov 0,%o4
	mov -1,%o5
	call _install,0
	nop
L2261:
	sethi %hi(_objc),%o0
	ld [%o0+%lo(_objc)],%o1
	cmp %o1,0
	be L2262
	nop
	sethi %hi(LC253),%o1
	or %o1,%lo(LC253),%o0
	mov -1,%o1
	mov 30,%o2
	mov 1,%o3
	mov 0,%o4
	mov -1,%o5
	call _install,0
	nop
L2262:
	sethi %hi(_debug_output),%o0
	ld [%o0+%lo(_debug_output)],%o1
	cmp %o1,0
	be L2263
	nop
	sethi %hi(_directive_table),%o0
	or %o0,%lo(_directive_table),%l0
	call _timestamp,0
	nop
	st %o0,[%fp-2068]
	add %fp,-2064,%o0
	sethi %hi(_instack+4),%o2
	sethi %hi(LC254),%o3
	or %o3,%lo(LC254),%o1
	ld [%o2+%lo(_instack+4)],%o2
	call _sprintf,0
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	mov 0,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
	add %fp,-2064,%l1
	add %fp,-2064,%l2
	add %fp,-2064,%o1
	mov %o1,%o0
	call _strlen,0
	nop
	add %l2,%o0,%o1
	mov %l1,%o0
	ld [%fp+72],%o2
	mov %l0,%o3
	call _pass_thru_directive,0
	nop
	add %fp,-2064,%o0
	sethi %hi(_version_string),%o2
	sethi %hi(LC255),%o3
	or %o3,%lo(LC255),%o1
	ld [%o2+%lo(_version_string)],%o2
	call _sprintf,0
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	mov 0,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
	add %fp,-2064,%l1
	add %fp,-2064,%l2
	add %fp,-2064,%o1
	mov %o1,%o0
	call _strlen,0
	nop
	add %l2,%o0,%o1
	mov %l1,%o0
	ld [%fp+72],%o2
	mov %l0,%o3
	call _pass_thru_directive,0
	nop
	add %fp,-2064,%o1
	mov %o1,%o0
	sethi %hi(LC256),%o2
	or %o2,%lo(LC256),%o1
	sethi %hi(LC129),%o3
	or %o3,%lo(LC129),%o2
	call _sprintf,0
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	mov 0,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
	add %fp,-2064,%l1
	add %fp,-2064,%l2
	add %fp,-2064,%o1
	mov %o1,%o0
	call _strlen,0
	nop
	add %l2,%o0,%o1
	mov %l1,%o0
	ld [%fp+72],%o2
	mov %l0,%o3
	call _pass_thru_directive,0
	nop
	add %fp,-2064,%o1
	mov %o1,%o0
	sethi %hi(LC257),%o2
	or %o2,%lo(LC257),%o1
	sethi %hi(LC131),%o3
	or %o3,%lo(LC131),%o2
	call _sprintf,0
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	mov 0,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
	add %fp,-2064,%l1
	add %fp,-2064,%l2
	add %fp,-2064,%o1
	mov %o1,%o0
	call _strlen,0
	nop
	add %l2,%o0,%o1
	mov %l1,%o0
	ld [%fp+72],%o2
	mov %l0,%o3
	call _pass_thru_directive,0
	nop
	add %fp,-2064,%o1
	mov %o1,%o0
	sethi %hi(LC258),%o2
	or %o2,%lo(LC258),%o1
	sethi %hi(LC132),%o3
	or %o3,%lo(LC132),%o2
	call _sprintf,0
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	mov 0,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
	add %fp,-2064,%l1
	add %fp,-2064,%l2
	add %fp,-2064,%o1
	mov %o1,%o0
	call _strlen,0
	nop
	add %l2,%o0,%o1
	mov %l1,%o0
	ld [%fp+72],%o2
	mov %l0,%o3
	call _pass_thru_directive,0
	nop
	add %fp,-2064,%o0
	ld [%fp-2068],%o1
	ld [%o1+16],%o2
	mov %o2,%o1
	sll %o1,2,%o2
	sethi %hi(_monthnames),%o1
	or %o1,%lo(_monthnames),%o3
	ld [%fp-2068],%o4
	ld [%fp-2068],%o1
	ld [%o1+20],%g2
	add %g2,1900,%o5
	sethi %hi(LC259),%g2
	or %g2,%lo(LC259),%o1
	ld [%o2+%o3],%o2
	ld [%o4+12],%o3
	mov %o5,%o4
	call _sprintf,0
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	mov 0,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
	add %fp,-2064,%l1
	add %fp,-2064,%l2
	add %fp,-2064,%o1
	mov %o1,%o0
	call _strlen,0
	nop
	add %l2,%o0,%o1
	mov %l1,%o0
	ld [%fp+72],%o2
	mov %l0,%o3
	call _pass_thru_directive,0
	nop
	add %fp,-2064,%o0
	ld [%fp-2068],%o2
	ld [%fp-2068],%o3
	ld [%fp-2068],%o4
	sethi %hi(LC260),%o5
	or %o5,%lo(LC260),%o1
	ld [%o2+8],%o2
	ld [%o3+4],%o3
	ld [%o4],%o4
	call _sprintf,0
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	mov 0,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
	add %fp,-2064,%l1
	add %fp,-2064,%l2
	add %fp,-2064,%o1
	mov %o1,%o0
	call _strlen,0
	nop
	add %l2,%o0,%o1
	mov %l1,%o0
	ld [%fp+72],%o2
	mov %l0,%o3
	call _pass_thru_directive,0
	nop
	sethi %hi(_traditional),%o0
	ld [%o0+%lo(_traditional)],%o1
	cmp %o1,0
	bne L2264
	nop
	add %fp,-2064,%o1
	mov %o1,%o0
	sethi %hi(LC261),%o2
	or %o2,%lo(LC261),%o1
	call _sprintf,0
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	mov 0,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
	add %fp,-2064,%l1
	add %fp,-2064,%l2
	add %fp,-2064,%o1
	mov %o1,%o0
	call _strlen,0
	nop
	add %l2,%o0,%o1
	mov %l1,%o0
	ld [%fp+72],%o2
	mov %l0,%o3
	call _pass_thru_directive,0
	nop
L2264:
	sethi %hi(_objc),%o0
	ld [%o0+%lo(_objc)],%o1
	cmp %o1,0
	be L2265
	nop
	add %fp,-2064,%o1
	mov %o1,%o0
	sethi %hi(LC262),%o2
	or %o2,%lo(LC262),%o1
	call _sprintf,0
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	mov 0,%o2
	mov 0,%o3
	call _output_line_command,0
	nop
	add %fp,-2064,%l1
	add %fp,-2064,%l2
	add %fp,-2064,%o1
	mov %o1,%o0
	call _strlen,0
	nop
	add %l2,%o0,%o1
	mov %l1,%o0
	ld [%fp+72],%o2
	mov %l0,%o3
	call _pass_thru_directive,0
	nop
L2265:
L2263:
L2260:
	ret
	restore
	.align 8
LC263:
	.ascii "malformed option `-D %s'\0"
	.align 8
LC264:
	.ascii "*Initialization*\0"
	.align 4
	.proc	020
_make_definition:
	!#PROLOGUE# 0
	save %sp,-136,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
	st %o0,[%fp-28]
	ld [%fp+68],%o0
	st %o0,[%fp-32]
	ld [%fp-32],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idstart),%o2
	or %o2,%lo(_is_idstart),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	bne L2267
	nop
	sethi %hi(LC263),%o1
	or %o1,%lo(LC263),%o0
	ld [%fp+68],%o1
	call _error,0
	nop
	b L2266
	nop
L2267:
	nop
L2268:
	ld [%fp-32],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-32]
	ldub [%o1],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L2269
	nop
	b L2268
	nop
L2269:
	ld [%fp-32],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	bne L2270
	nop
	ld [%fp-32],%o0
	ld [%fp-28],%o1
	sub %o0,%o1,%o0
	add %o0,4,%o1
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
	st %o1,[%fp-28]
	ld [%fp-28],%o0
	ld [%fp+68],%o1
	call _strcpy,0
	nop
	ld [%fp-28],%o0
	sethi %hi(LC218),%o2
	or %o2,%lo(LC218),%o1
	call _strcat,0
	nop
	b L2271
	nop
L2270:
	ld [%fp-32],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,61
	be L2272
	nop
	sethi %hi(LC263),%o1
	or %o1,%lo(LC263),%o0
	ld [%fp+68],%o1
	call _error,0
	nop
	b L2266
	nop
	b L2273
	nop
L2272:
	ld [%fp+68],%o0
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
	st %o1,[%fp-28]
	ld [%fp-32],%o0
	ld [%fp+68],%o1
	sub %o0,%o1,%o2
	ld [%fp-28],%o0
	ld [%fp+68],%o1
	call _strncpy,0
	nop
	ld [%fp-32],%o0
	ld [%fp+68],%o1
	sub %o0,%o1,%o0
	ld [%fp-28],%o1
	add %o0,%o1,%o0
	mov 32,%o1
	stb %o1,[%o0]
	ld [%fp-32],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-32]
	ld [%fp-32],%o0
	ld [%fp+68],%o1
	sub %o0,%o1,%o0
	ld [%fp-28],%o1
	add %o1,%o0,%o0
	st %o0,[%fp-36]
L2274:
	ld [%fp-32],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L2275
	nop
	ld [%fp-32],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L2276
	nop
	mov 1,%o0
	ld [%fp-32],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L2276
	nop
	ld [%fp-32],%o0
	add %o0,2,%o1
	st %o1,[%fp-32]
	b L2277
	nop
L2276:
	ld [%fp-32],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L2278
	nop
	ld [%fp-36],%o0
	add %o0,1,%o1
	st %o1,[%fp-36]
	mov 10,%o1
	stb %o1,[%o0]
	ld [%fp-36],%o0
	add %o0,1,%o1
	st %o1,[%fp-36]
	mov 10,%o1
	stb %o1,[%o0]
	ld [%fp-32],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-32]
	b L2279
	nop
L2278:
	ld [%fp-36],%o0
	add %o0,1,%o1
	st %o1,[%fp-36]
	ld [%fp-32],%o1
	add %o1,1,%o2
	st %o2,[%fp-32]
	ldub [%o1],%o1
	stb %o1,[%o0]
L2279:
L2277:
	b L2274
	nop
L2275:
	ld [%fp-36],%o0
	stb %g0,[%o0]
L2273:
L2271:
	sethi %hi(_indepth),%o0
	sethi %hi(_indepth),%o1
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o3
	add %o3,1,%o2
	mov %o2,%o0
	st %o0,[%o1+%lo(_indepth)]
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	sethi %hi(LC264),%o2
	or %o2,%lo(LC264),%o3
	st %o3,[%o1]
	st %o3,[%o0+4]
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%fp-28],%o2
	st %o2,[%o1+24]
	st %o2,[%o0+20]
	ld [%fp-20],%l0
	ld [%fp-28],%o0
	call _strlen,0
	nop
	st %o0,[%l0+16]
	ld [%fp-20],%o0
	mov 1,%o1
	st %o1,[%o0+12]
	ld [%fp-20],%o0
	st %g0,[%o0+28]
	ld [%fp-20],%o0
	st %g0,[%o0+36]
	ld [%fp-20],%o0
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o2
	st %o2,[%o0+32]
	ld [%fp-20],%o0
	stb %g0,[%o0+40]
	sethi %hi(_directive_table),%o1
	or %o1,%lo(_directive_table),%o0
	st %o0,[%fp-24]
L2280:
	ld [%fp-24],%o0
	ld [%o0+12],%o1
	cmp %o1,1
	be L2281
	nop
L2282:
	ld [%fp-24],%o1
	add %o1,20,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
	b L2280
	nop
L2281:
	ld [%fp-28],%o0
	call _strlen,0
	nop
	ld [%fp-28],%o2
	add %o0,%o2,%o1
	ld [%fp-28],%o0
	ld [%fp+72],%o2
	ld [%fp-24],%o3
	call _do_define,0
	nop
	sethi %hi(_indepth),%o1
	sethi %hi(_indepth),%o0
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_indepth)]
L2266:
	ret
	restore
	.align 8
LC265:
	.ascii "*undef*\0"
	.align 4
	.proc	020
_make_undef:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	sethi %hi(_indepth),%o0
	sethi %hi(_indepth),%o1
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o3
	add %o3,1,%o2
	mov %o2,%o0
	st %o0,[%o1+%lo(_indepth)]
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	sethi %hi(LC265),%o2
	or %o2,%lo(LC265),%o3
	st %o3,[%o1]
	st %o3,[%o0+4]
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%fp+68],%o2
	st %o2,[%o1+24]
	st %o2,[%o0+20]
	ld [%fp-20],%l0
	ld [%fp+68],%o0
	call _strlen,0
	nop
	st %o0,[%l0+16]
	ld [%fp-20],%o0
	mov 1,%o1
	st %o1,[%o0+12]
	ld [%fp-20],%o0
	st %g0,[%o0+28]
	ld [%fp-20],%o0
	st %g0,[%o0+36]
	ld [%fp-20],%o0
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o2
	st %o2,[%o0+32]
	ld [%fp-20],%o0
	stb %g0,[%o0+40]
	sethi %hi(_directive_table),%o1
	or %o1,%lo(_directive_table),%o0
	st %o0,[%fp-24]
L2284:
	ld [%fp-24],%o0
	ld [%o0+12],%o1
	cmp %o1,11
	be L2285
	nop
L2286:
	ld [%fp-24],%o1
	add %o1,20,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
	b L2284
	nop
L2285:
	ld [%fp+68],%o0
	call _strlen,0
	nop
	ld [%fp+68],%o2
	add %o0,%o2,%o1
	ld [%fp+68],%o0
	ld [%fp+72],%o2
	ld [%fp-24],%o3
	call _do_undef,0
	nop
	sethi %hi(_indepth),%o1
	sethi %hi(_indepth),%o0
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_indepth)]
L2283:
	ret
	restore
	.align 8
LC266:
	.ascii "malformed option `%s %s'\0"
	.align 4
	.proc	020
_make_assertion:
	!#PROLOGUE# 0
	save %sp,-136,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+72],%o0
	call _strlen,0
	nop
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
	st %o1,[%fp-28]
	ld [%fp-28],%o0
	ld [%fp+72],%o1
	call _strcpy,0
	nop
	ld [%fp-28],%o0
	st %o0,[%fp-36]
	st %o0,[%fp-32]
L2288:
	ld [%fp-32],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L2289
	nop
	ld [%fp-32],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,92
	bne L2290
	nop
	mov 1,%o0
	ld [%fp-32],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,10
	bne L2290
	nop
	ld [%fp-32],%o0
	add %o0,2,%o1
	st %o1,[%fp-32]
	b L2291
	nop
L2290:
	ld [%fp-36],%o0
	add %o0,1,%o1
	st %o1,[%fp-36]
	ld [%fp-32],%o1
	add %o1,1,%o2
	st %o2,[%fp-32]
	ldub [%o1],%o1
	stb %o1,[%o0]
L2291:
	b L2288
	nop
L2289:
	ld [%fp-36],%o0
	stb %g0,[%o0]
	ld [%fp-28],%o0
	st %o0,[%fp-32]
	ld [%fp-32],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idstart),%o2
	or %o2,%lo(_is_idstart),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	bne L2292
	nop
	sethi %hi(LC266),%o1
	or %o1,%lo(LC266),%o0
	ld [%fp+68],%o1
	ld [%fp+72],%o2
	call _error,0
	nop
	b L2287
	nop
L2292:
	nop
L2293:
	ld [%fp-32],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-32]
	ldub [%o1],%o1
	and %o1,0xff,%o0
	sethi %hi(_is_idchar),%o2
	or %o2,%lo(_is_idchar),%o1
	ldub [%o0+%o1],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L2294
	nop
	b L2293
	nop
L2294:
	nop
L2295:
	ld [%fp-32],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,32
	be L2297
	nop
	ld [%fp-32],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,9
	be L2297
	nop
	b L2296
	nop
L2297:
	ld [%fp-32],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-32]
	b L2295
	nop
L2296:
	ld [%fp-32],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L2298
	nop
	ld [%fp-32],%o0
	ldub [%o0],%o1
	and %o1,0xff,%o0
	cmp %o0,40
	be L2298
	nop
	sethi %hi(LC266),%o1
	or %o1,%lo(LC266),%o0
	ld [%fp+68],%o1
	ld [%fp+72],%o2
	call _error,0
	nop
	b L2287
	nop
L2298:
	sethi %hi(_indepth),%o0
	sethi %hi(_indepth),%o1
	sethi %hi(_indepth),%o0
	ld [%o0+%lo(_indepth)],%o3
	add %o3,1,%o2
	mov %o2,%o0
	st %o0,[%o1+%lo(_indepth)]
	mov %o0,%o2
	sll %o2,1,%o1
	add %o1,%o0,%o1
	sll %o1,2,%o2
	sub %o2,%o0,%o2
	sll %o2,2,%o0
	sethi %hi(_instack),%o2
	or %o2,%lo(_instack),%o1
	add %o1,%o0,%o0
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	sethi %hi(LC264),%o2
	or %o2,%lo(LC264),%o3
	st %o3,[%o1]
	st %o3,[%o0+4]
	ld [%fp-20],%o0
	ld [%fp-20],%o1
	ld [%fp-28],%o2
	st %o2,[%o1+24]
	st %o2,[%o0+20]
	ld [%fp-20],%l0
	ld [%fp-28],%o0
	call _strlen,0
	nop
	st %o0,[%l0+16]
	ld [%fp-20],%o0
	mov 1,%o1
	st %o1,[%o0+12]
	ld [%fp-20],%o0
	st %g0,[%o0+28]
	ld [%fp-20],%o0
	st %g0,[%o0+36]
	ld [%fp-20],%o0
	sethi %hi(_if_stack),%o1
	ld [%o1+%lo(_if_stack)],%o2
	st %o2,[%o0+32]
	ld [%fp-20],%o0
	stb %g0,[%o0+40]
	sethi %hi(_directive_table),%o1
	or %o1,%lo(_directive_table),%o0
	st %o0,[%fp-24]
L2299:
	ld [%fp-24],%o0
	ld [%o0+12],%o1
	cmp %o1,18
	be L2300
	nop
L2301:
	ld [%fp-24],%o1
	add %o1,20,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
	b L2299
	nop
L2300:
	ld [%fp-28],%o0
	call _strlen,0
	nop
	ld [%fp-28],%o2
	add %o0,%o2,%o1
	ld [%fp-28],%o0
	mov 0,%o2
	ld [%fp-24],%o3
	call _do_assert,0
	nop
	sethi %hi(_indepth),%o1
	sethi %hi(_indepth),%o0
	sethi %hi(_indepth),%o1
	ld [%o1+%lo(_indepth)],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_indepth)]
L2287:
	ret
	restore
	.align 4
	.proc	020
_append_include_chain:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
	cmp %o0,0
	be L2304
	nop
	ld [%fp+72],%o0
	cmp %o0,0
	bne L2303
	nop
	b L2304
	nop
L2304:
	b L2302
	nop
L2303:
	sethi %hi(_include),%o0
	ld [%o0+%lo(_include)],%o1
	cmp %o1,0
	bne L2305
	nop
	sethi %hi(_include),%o0
	ld [%fp+68],%o1
	st %o1,[%o0+%lo(_include)]
	b L2306
	nop
L2305:
	sethi %hi(_last_include),%o1
	ld [%o1+%lo(_last_include)],%o0
	ld [%fp+68],%o1
	st %o1,[%o0]
L2306:
	sethi %hi(_first_bracket_include),%o0
	ld [%o0+%lo(_first_bracket_include)],%o1
	cmp %o1,0
	bne L2307
	nop
	sethi %hi(_first_bracket_include),%o0
	ld [%fp+68],%o1
	st %o1,[%o0+%lo(_first_bracket_include)]
L2307:
	nop
	ld [%fp+68],%o0
	st %o0,[%fp-20]
L2308:
	ld [%fp-20],%o1
	ld [%o1+4],%o0
	call _strlen,0
	nop
	st %o0,[%fp-24]
	sethi %hi(_max_include_len),%o0
	ld [%fp-24],%o1
	ld [%o0+%lo(_max_include_len)],%o0
	cmp %o1,%o0
	ble L2311
	nop
	sethi %hi(_max_include_len),%o0
	ld [%fp-24],%o1
	st %o1,[%o0+%lo(_max_include_len)]
L2311:
	ld [%fp-20],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	bne L2312
	nop
	b L2309
	nop
L2312:
L2310:
	ld [%fp-20],%o0
	ld [%o0],%o1
	st %o1,[%fp-20]
	b L2308
	nop
L2309:
	ld [%fp+72],%o0
	st %g0,[%o0]
	sethi %hi(_last_include),%o0
	ld [%fp+72],%o1
	st %o1,[%o0+%lo(_last_include)]
L2302:
	ret
	restore
	.align 8
LC267:
	.ascii "\\\12  \0"
	.align 4
	.proc	020
_deps_output:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+72],%o0
	cmp %o0,0
	bne L2314
	nop
	ld [%fp+68],%o0
	call _strlen,0
	nop
	st %o0,[%fp+72]
L2314:
	ld [%fp+72],%o0
	cmp %o0,0
	bne L2315
	nop
	sethi %hi(_deps_column),%o0
	ld [%o0+%lo(_deps_column)],%o1
	cmp %o1,0
	be L2315
	nop
	sethi %hi(_deps_column),%o0
	ld [%fp+72],%o1
	ld [%o0+%lo(_deps_column)],%o2
	add %o1,%o2,%o0
	cmp %o0,75
	bleu L2315
	nop
	sethi %hi(LC267),%o1
	or %o1,%lo(LC267),%o0
	mov 0,%o1
	call _deps_output,0
	nop
	sethi %hi(_deps_column),%o0
	st %g0,[%o0+%lo(_deps_column)]
L2315:
	sethi %hi(_deps_size),%o0
	ld [%o0+%lo(_deps_size)],%o1
	ld [%fp+72],%o0
	add %o1,%o0,%o1
	add %o1,1,%o0
	sethi %hi(_deps_allocated_size),%o1
	ld [%o1+%lo(_deps_allocated_size)],%o2
	cmp %o0,%o2
	bleu L2316
	nop
	sethi %hi(_deps_allocated_size),%o0
	sethi %hi(_deps_size),%o1
	ld [%o1+%lo(_deps_size)],%o2
	ld [%fp+72],%o3
	add %o2,%o3,%o1
	add %o1,50,%o2
	st %o2,[%o0+%lo(_deps_allocated_size)]
	sethi %hi(_deps_allocated_size),%o0
	sethi %hi(_deps_allocated_size),%o1
	ld [%o1+%lo(_deps_allocated_size)],%o2
	mov %o2,%o1
	sll %o1,1,%o2
	st %o2,[%o0+%lo(_deps_allocated_size)]
	sethi %hi(_deps_buffer),%o0
	sethi %hi(_deps_allocated_size),%o1
	ld [%o0+%lo(_deps_buffer)],%o0
	ld [%o1+%lo(_deps_allocated_size)],%o1
	call _xrealloc,0
	nop
	sethi %hi(_deps_buffer),%o1
	st %o0,[%o1+%lo(_deps_buffer)]
L2316:
	sethi %hi(_deps_buffer),%o0
	sethi %hi(_deps_size),%o1
	ld [%o0+%lo(_deps_buffer)],%o0
	ld [%o1+%lo(_deps_size)],%o2
	add %o0,%o2,%o1
	ld [%fp+68],%o0
	ld [%fp+72],%o2
	call _bcopy,0
	nop
	sethi %hi(_deps_size),%o0
	sethi %hi(_deps_size),%o1
	ld [%o1+%lo(_deps_size)],%o2
	ld [%fp+72],%o1
	add %o2,%o1,%o2
	st %o2,[%o0+%lo(_deps_size)]
	sethi %hi(_deps_column),%o0
	sethi %hi(_deps_column),%o1
	ld [%o1+%lo(_deps_column)],%o2
	ld [%fp+72],%o1
	add %o2,%o1,%o2
	st %o2,[%o0+%lo(_deps_column)]
	sethi %hi(_deps_buffer),%o1
	ld [%o1+%lo(_deps_buffer)],%o0
	sethi %hi(_deps_size),%o2
	ld [%o2+%lo(_deps_size)],%o1
	add %o0,%o1,%o0
	stb %g0,[%o0]
L2313:
	ret
	restore
	.align 8
LC268:
	.ascii "%s: \0"
	.align 4
	.proc	020
_fatal:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	sethi %hi(_progname),%o2
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC268),%o3
	or %o3,%lo(LC268),%o1
	ld [%o2+%lo(_progname)],%o2
	call _fprintf,0
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	ld [%fp+68],%o1
	ld [%fp+72],%o2
	call _fprintf,0
	nop
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC66),%o2
	or %o2,%lo(LC66),%o1
	call _fprintf,0
	nop
	mov 33,%o0
	call _exit,0
	nop
L2317:
	ret
	restore
	.align 8
LC269:
	.ascii "Internal gcc abort.\0"
	.align 4
	.global _fancy_abort
	.proc	020
_fancy_abort:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	sethi %hi(LC269),%o1
	or %o1,%lo(LC269),%o0
	call _fatal,0
	nop
L2318:
	ret
	restore
	.align 4
	.proc	020
_perror_with_name:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	sethi %hi(_progname),%o2
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC268),%o3
	or %o3,%lo(LC268),%o1
	ld [%o2+%lo(_progname)],%o2
	call _fprintf,0
	nop
	sethi %hi(_errno),%o0
	sethi %hi(_sys_nerr),%o1
	ld [%o0+%lo(_errno)],%o0
	ld [%o1+%lo(_sys_nerr)],%o1
	cmp %o0,%o1
	bge L2320
	nop
	sethi %hi(_errno),%o1
	ld [%o1+%lo(_errno)],%o0
	mov %o0,%o1
	sll %o1,2,%o3
	sethi %hi(_sys_errlist),%o0
	or %o0,%lo(_sys_errlist),%o4
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC229),%o2
	or %o2,%lo(LC229),%o1
	ld [%fp+68],%o2
	ld [%o3+%o4],%o3
	call _fprintf,0
	nop
	b L2321
	nop
L2320:
	sethi %hi(__iob+64),%o1
	or %o1,%lo(__iob+64),%o0
	sethi %hi(LC230),%o2
	or %o2,%lo(LC230),%o1
	ld [%fp+68],%o2
	call _fprintf,0
	nop
L2321:
	sethi %hi(_errors),%o1
	sethi %hi(_errors),%o0
	sethi %hi(_errors),%o1
	ld [%o1+%lo(_errors)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_errors)]
L2319:
	ret
	restore
	.align 4
	.proc	020
_pfatal_with_name:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	call _perror_with_name,0
	nop
	mov 33,%o0
	call _exit,0
	nop
L2322:
	ret
	restore
	.align 8
LC270:
	.ascii "Memory exhausted.\0"
	.align 4
	.proc	020
_memory_full:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	sethi %hi(LC270),%o1
	or %o1,%lo(LC270),%o0
	call _fatal,0
	nop
L2323:
	ret
	restore
	.align 4
	.global _xmalloc
	.proc	0102
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
	be L2325
	nop
	mov %l0,%i0
	b L2324
	nop
L2325:
	call _memory_full,0
	nop
	mov 0,%i0
	b L2324
	nop
L2324:
	ret
	restore
	.align 4
	.proc	0102
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
	be L2327
	nop
	mov %l0,%i0
	b L2326
	nop
L2327:
	call _memory_full,0
	nop
	mov 0,%i0
	b L2326
	nop
L2326:
	ret
	restore
	.align 4
	.proc	0102
_xcalloc:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	call .umul,0
	nop
	mov %o0,%l0
	mov %l0,%o0
	call _malloc,0
	nop
	mov %o0,%l1
	cmp %l1,0
	be L2329
	nop
	cmp %l0,100
	bleu L2330
	nop
	mov %l1,%o0
	mov %l0,%o1
	call _bzero,0
	nop
	b L2331
	nop
L2330:
	mov %l1,%o0
	add %l1,%l0,%o1
	add %o1,-4,%o1
	add %l0,-4,%o2
L2332:
	cmp %o0,%o1
	bgeu L2333
	nop
	st %g0,[%o0]
	add %o0,4,%o0
	b L2332
	nop
L2333:
	cmp %o2,0
	bge L2334
	nop
	mov 0,%o2
L2334:
	nop
L2335:
	cmp %o2,%l0
	bgeu L2336
	nop
	add %l1,%o2,%o3
	stb %g0,[%o3]
	add %o2,1,%o2
	b L2335
	nop
L2336:
L2331:
	mov %l1,%i0
	b L2328
	nop
L2329:
	call _memory_full,0
	nop
	mov 0,%i0
	b L2328
	nop
L2328:
	ret
	restore
	.align 4
	.proc	0102
_savestring:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	call _strlen,0
	nop
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	add %o0,1,%o1
	mov %o1,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-24]
	ld [%fp-24],%o0
	ld [%fp+68],%o1
	call _strcpy,0
	nop
	ld [%fp-24],%i0
	b L2337
	nop
L2337:
	ret
	restore
	.align 4
	.proc	04
_file_size_and_mode:
	!#PROLOGUE# 0
	save %sp,-184,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	add %fp,-88,%o1
	ld [%fp+68],%o0
	call _fstat,0
	nop
	cmp %o0,0
	bge L2339
	nop
	mov -1,%i0
	b L2338
	nop
L2339:
	ld [%fp+72],%o0
	cmp %o0,0
	be L2340
	nop
	ld [%fp+72],%o0
	ld [%fp-80],%o1
	st %o1,[%o0]
L2340:
	ld [%fp+76],%o0
	cmp %o0,0
	be L2341
	nop
	ld [%fp+76],%o0
	ld [%fp-60],%o1
	st %o1,[%o0]
L2341:
	mov 0,%i0
	b L2338
	nop
L2338:
	ret
	restore

	.reserve _progname,8,"bss"

	.reserve _cplusplus,8,"bss"

	.reserve _cplusplus_comments,8,"bss"

	.reserve _objc,8,"bss"

	.reserve _lang_asm,8,"bss"

	.reserve _max_include_len,8,"bss"

	.reserve _no_line_commands,8,"bss"

	.reserve _pcp_outfile,8,"bss"

	.reserve _pcp_inside_if,8,"bss"

	.reserve _no_precomp,8,"bss"
	.global _pedantic
	.common _pedantic,8,"bss"

	.reserve _pedantic_errors,8,"bss"

	.reserve _warn_comments,8,"bss"

	.reserve _warn_stringify,8,"bss"

	.reserve _warn_trigraphs,8,"bss"

	.reserve _warnings_are_errors,8,"bss"
	.global _traditional
	.common _traditional,8,"bss"

	.reserve _no_output,8,"bss"

	.reserve _instack,8800,"bss"

	.reserve _last_error_tick,8,"bss"

	.reserve _input_file_stack_tick,8,"bss"

	.reserve _outbuf,48,"bss"

	.reserve _include_prefix,8,"bss"

	.reserve _stringlist,8,"bss"

	.reserve _hashtab,5616,"bss"

	.reserve _assertion_hashtab,152,"bss"

	.reserve _assertions_flag,8,"bss"
	.global _directive_start
	.common _directive_start,8,"bss"
	.global _is_idchar
	.common _is_idchar,256,"bss"
	.global _is_idstart
	.common _is_idstart,256,"bss"
	.global _is_hor_space
	.common _is_hor_space,256,"bss"

	.reserve _is_space,256,"bss"

	.reserve _dollars_in_ident,8,"bss"

	.reserve _deps_buffer,8,"bss"

	.reserve _deps_allocated_size,8,"bss"

	.reserve _deps_size,8,"bss"

	.reserve _deps_column,8,"bss"

	.reserve _ignore_srcdir,8,"bss"

	.reserve _import_hash_table,128,"bss"
