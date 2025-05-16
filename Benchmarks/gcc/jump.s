gcc2_compiled.:
___gnu_compiled_c:
.text
	.align 4
	.global _jump_optimize
	.proc	020
_jump_optimize:
	!#PROLOGUE# 0
	save %sp,-176,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	mov 1,%o0
	st %o0,[%fp-24]
	st %g0,[%fp-28]
	ld [%fp+68],%l0
L2:
	cmp %l0,0
	be L3
	nop
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,17
	bne L5
	nop
	st %g0,[%l0+20]
L5:
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,14
	bne L6
	nop
	st %g0,[%l0+32]
L6:
	ld [%l0+4],%o0
	ld [%fp-28],%o1
	cmp %o0,%o1
	ble L7
	nop
	ld [%l0+4],%o0
	st %o0,[%fp-28]
L7:
L4:
	ld [%l0+12],%l0
	b L2
	nop
L3:
	ld [%fp-28],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-28]
	sethi %hi(_jump_chain),%o0
	ld [%fp-28],%o1
	mov %o1,%o2
	sll %o2,2,%o3
	add %o3,7,%o1
	add %sp,100,%o3
	sub %o3,%sp,%o2
	add %o1,%o2,%o3
	mov %o3,%o1
	add %o1,7,%o1
	srl %o1,3,%o2
	mov %o2,%o1
	sll %o1,3,%o2
	sub %sp,%o2,%sp
	add %sp,100,%o2
	mov %o2,%o1
	add %o1,7,%o1
	srl %o1,3,%o2
	mov %o2,%o1
	sll %o1,3,%o2
	st %o2,[%o0+%lo(_jump_chain)]
	sethi %hi(_jump_chain),%o0
	ld [%fp-28],%o1
	mov %o1,%o3
	sll %o3,2,%o2
	ld [%o0+%lo(_jump_chain)],%o0
	mov 0,%o1
	call _memset,0
	nop
	ld [%fp+68],%l0
L8:
	cmp %l0,0
	be L9
	nop
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,16
	bne L11
	nop
	ld [%l0+12],%l0
L12:
	cmp %l0,0
	be L13
	nop
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,17
	be L13
	nop
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,18
	bne L14
	nop
	ld [%l0+20],%o0
	cmp %o0,-6
	be L14
	nop
	ld [%l0+12],%l0
	b L15
	nop
L14:
	mov %l0,%o0
	call _delete_insn,0
	nop
	mov %o0,%l0
L15:
	b L12
	nop
L13:
	b L16
	nop
L11:
	ld [%l0+12],%l0
L16:
L10:
	b L8
	nop
L9:
	nop
	ld [%fp+68],%l0
L17:
	cmp %l0,0
	be L18
	nop
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,14
	bne L20
	nop
	ld [%l0],%o1
	and %o1,16,%o0
	cmp %o0,0
	bne L20
	nop
	ld [%l0+16],%o0
	mov %l0,%o1
	ld [%fp+72],%o2
	call _mark_jump_label,0
	nop
	ld [%l0+32],%o0
	cmp %o0,0
	be L21
	nop
	mov %l0,%o0
	call _simplejump_p,0
	nop
	cmp %o0,0
	be L21
	nop
	ld [%l0+4],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_jump_chain),%o2
	ld [%o2+%lo(_jump_chain)],%o1
	add %o0,%o1,%o0
	ld [%l0+32],%o1
	ld [%o1+4],%o2
	mov %o2,%o3
	sll %o3,2,%o1
	sethi %hi(_jump_chain),%o3
	ld [%o3+%lo(_jump_chain)],%o2
	add %o1,%o2,%o1
	ld [%o1],%o2
	st %o2,[%o0]
	ld [%l0+32],%o0
	ld [%o0+4],%o1
	mov %o1,%o2
	sll %o2,2,%o0
	sethi %hi(_jump_chain),%o2
	ld [%o2+%lo(_jump_chain)],%o1
	add %o0,%o1,%o0
	st %l0,[%o0]
L21:
	ld [%l0+16],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,29
	bne L22
	nop
	ld [%l0+4],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_jump_chain),%o2
	ld [%o2+%lo(_jump_chain)],%o1
	add %o0,%o1,%o0
	sethi %hi(_jump_chain),%o2
	ld [%o2+%lo(_jump_chain)],%o1
	ld [%o1],%o2
	st %o2,[%o0]
	sethi %hi(_jump_chain),%o1
	ld [%o1+%lo(_jump_chain)],%o0
	st %l0,[%o0]
L22:
L20:
L19:
	ld [%l0+12],%l0
	b L17
	nop
L18:
	st %g0,[%fp-32]
	ld [%fp+68],%l0
L23:
	cmp %l0,0
	be L24
	nop
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,17
	bne L26
	nop
	ld [%l0+20],%o0
	cmp %o0,0
	bne L26
	nop
	mov %l0,%o0
	call _delete_insn,0
	nop
	mov %o0,%l0
	b L27
	nop
L26:
	st %l0,[%fp-32]
	ld [%l0+12],%l0
L27:
L25:
	b L23
	nop
L24:
	sethi %hi(_optimize),%o0
	ld [%o0+%lo(_optimize)],%o1
	cmp %o1,0
	bne L28
	nop
	ld [%fp-32],%l0
L29:
	cmp %l0,0
	be L30
	nop
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,17
	be L31
	nop
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,14
	bne L32
	nop
	ld [%l0+16],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,29
	be L31
	nop
	b L32
	nop
L32:
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,18
	bne L30
	nop
	ld [%l0+20],%o0
	cmp %o0,-6
	bne L31
	nop
	b L30
	nop
L31:
	ld [%l0+8],%l0
	b L29
	nop
L30:
	cmp %l0,0
	be L33
	nop
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,18
	bne L33
	nop
	ld [%l0+20],%o0
	cmp %o0,-6
	bne L33
	nop
	ld [%l0],%o1
	and %o1,16,%o0
	cmp %o0,0
	bne L33
	nop
	sethi %hi(_current_function_returns_null),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_current_function_returns_null)]
L33:
	nop
	ld [%fp+68],%l0
L34:
	cmp %l0,0
	be L35
	nop
	ld [%l0],%o0
	and %o0,-17,%o1
	st %o1,[%l0]
L36:
	ld [%l0+12],%l0
	b L34
	nop
L35:
	b L1
	nop
L28:
	ld [%fp+76],%o0
	cmp %o0,0
	be L37
	nop
	ld [%fp+68],%l0
L38:
	cmp %l0,0
	be L39
	nop
	ld [%l0+12],%l1
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,13
	bne L41
	nop
	ld [%l0+16],%l2
	lduh [%l2],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,26
	be L43
	nop
	lduh [%l2],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,27
	be L43
	nop
	b L42
	nop
L43:
	mov %l0,%o0
	call _delete_insn,0
	nop
	b L44
	nop
L42:
	lduh [%l2],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,25
	bne L45
	nop
	ld [%l2+4],%o0
	ld [%l2+8],%o1
	cmp %o0,%o1
	be L46
	nop
	ld [%l2+4],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,37
	bne L45
	nop
	ld [%l2+8],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,37
	bne L45
	nop
	ld [%l2+8],%o0
	ld [%l2+4],%o1
	call _rtx_equal_p,0
	nop
	cmp %o0,0
	bne L46
	nop
	b L45
	nop
L46:
	ld [%l2+4],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,37
	bne L47
	nop
	ld [%l2+4],%o0
	ld [%o0],%o1
	and %o1,16,%o0
	cmp %o0,0
	bne L45
	nop
	b L47
	nop
L47:
	ld [%l2+8],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,37
	bne L48
	nop
	ld [%l2+8],%o0
	ld [%o0],%o1
	and %o1,16,%o0
	cmp %o0,0
	bne L45
	nop
	b L48
	nop
L48:
	mov %l0,%o0
	call _delete_insn,0
	nop
	b L49
	nop
L45:
	lduh [%l2],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,25
	bne L50
	nop
	ld [%l2+8],%o0
	call _true_regnum,0
	nop
	st %o0,[%fp-36]
	ld [%l2+4],%o0
	call _true_regnum,0
	nop
	st %o0,[%fp-40]
	ld [%fp-36],%o0
	ld [%fp-40],%o1
	cmp %o0,%o1
	bne L51
	nop
	ld [%fp-36],%o0
	cmp %o0,0
	bl L51
	nop
	mov %l0,%o0
	call _delete_insn,0
	nop
	b L52
	nop
L51:
	ld [%fp-36],%o0
	cmp %o0,0
	bl L53
	nop
	ld [%fp-40],%o0
	cmp %o0,0
	bl L53
	nop
	ld [%l2+8],%o0
	ld [%o0],%o1
	srl %o1,8,%o0
	and %o0,255,%o1
	st %o1,[%sp+92]
	mov 0,%o0
	mov %l0,%o1
	mov 0,%o2
	ld [%fp-36],%o3
	mov 0,%o4
	ld [%fp-40],%o5
	call _find_equiv_reg,0
	nop
	st %o0,[%fp-44]
	ld [%fp-44],%o0
	cmp %o0,0
	be L54
	nop
	ld [%fp-44],%o1
	ld [%l2+4],%o0
	ldub [%o1+2],%o2
	and %o2,0xff,%o1
	ldub [%o0+2],%o2
	and %o2,0xff,%o0
	cmp %o1,%o0
	bne L54
	nop
	mov %l0,%o0
	call _delete_insn,0
	nop
L54:
L53:
L52:
L50:
L49:
L44:
L41:
	mov %l1,%l0
L40:
	b L38
	nop
L39:
L37:
	mov 1,%o0
	st %o0,[%fp-20]
L55:
	ld [%fp-20],%o0
	cmp %o0,0
	be L56
	nop
	st %g0,[%fp-20]
	ld [%fp+68],%l0
L57:
	cmp %l0,0
	be L58
	nop
	ld [%l0+12],%l1
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,14
	bne L60
	nop
	ld [%l0+16],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,23
	bne L61
	nop
	ld [%l0+16],%o0
	mov 0,%o1
	ld [%fp+76],%o2
	call _tension_vector_labels,0
	nop
	ld [%fp-20],%o1
	or %o1,%o0,%o0
	st %o0,[%fp-20]
L61:
	ld [%l0+16],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,24
	bne L62
	nop
	ld [%l0+16],%o0
	mov 1,%o1
	ld [%fp+76],%o2
	call _tension_vector_labels,0
	nop
	ld [%fp-20],%o1
	or %o1,%o0,%o0
	st %o0,[%fp-20]
L62:
L60:
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,14
	bne L63
	nop
	ld [%l0+32],%o0
	cmp %o0,0
	be L63
	nop
	ld [%l0+32],%o0
	call _prev_real_insn,0
	nop
	mov %o0,%l2
	cmp %l2,%l0
	bne L64
	nop
	mov %l0,%o0
	call _condjump_p,0
	nop
	cmp %o0,0
	be L64
	nop
	mov %l0,%o0
	call _delete_jump,0
	nop
	mov 1,%o0
	st %o0,[%fp-20]
	b L65
	nop
L64:
	mov %l0,%o0
	call _next_real_insn,0
	nop
	st %o0,[%fp-44]
	ld [%fp-44],%o0
	cmp %o0,0
	be L66
	nop
	ld [%fp-44],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,14
	bne L66
	nop
	mov %l0,%o0
	call _condjump_p,0
	nop
	cmp %o0,0
	be L66
	nop
	ld [%fp-44],%o0
	call _simplejump_p,0
	nop
	cmp %o0,0
	be L66
	nop
	ld [%fp-44],%o0
	ld [%l0+32],%o1
	ld [%o0+32],%o0
	cmp %o1,%o0
	bne L66
	nop
	mov %l0,%o0
	call _delete_jump,0
	nop
	mov 1,%o0
	st %o0,[%fp-20]
	ld [%l0+12],%l1
	b L67
	nop
L66:
	mov %l0,%o0
	call _simplejump_p,0
	nop
	cmp %o0,0
	be L68
	nop
	ld [%l0+32],%o0
	call _next_real_insn,0
	nop
	st %o0,[%fp-44]
	ld [%fp-44],%o0
	cmp %o0,0
	be L68
	nop
	ld [%fp-44],%o0
	ld [%o0+16],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,29
	bne L68
	nop
	ld [%fp-44],%o0
	ld [%o0+16],%o1
	st %o1,[%l0+16]
	mov -1,%o0
	st %o0,[%l0+20]
	b L69
	nop
L68:
	cmp %l2,0
	be L70
	nop
	lduh [%l2],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,14
	bne L70
	nop
	mov %l2,%o0
	call _prev_real_insn,0
	nop
	cmp %o0,%l0
	bne L70
	nop
	mov %l0,%o0
	mov %l2,%o1
	call _no_labels_between_p,0
	nop
	cmp %o0,0
	be L70
	nop
	mov %l2,%o0
	call _simplejump_p,0
	nop
	cmp %o0,0
	be L70
	nop
	mov %l0,%o0
	call _condjump_p,0
	nop
	cmp %o0,0
	be L70
	nop
	ld [%l2+32],%o1
	ld [%l2+32],%o0
	ld [%l2+32],%o1
	ld [%o1+20],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+20]
	mov %l2,%o0
	call _delete_insn,0
	nop
	mov %l0,%o0
	ld [%l2+32],%o1
	call _invert_jump,0
	nop
	ld [%l2+32],%o1
	ld [%l2+32],%o0
	ld [%l2+32],%o1
	ld [%o1+20],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0+20]
	mov %l0,%l1
	mov 1,%o0
	st %o0,[%fp-20]
	b L71
	nop
L70:
	ld [%l0+32],%o0
	ld [%fp+76],%o1
	call _follow_jumps,0
	nop
	mov %o0,%l3
	ld [%l0+32],%o0
	cmp %l3,%o0
	be L72
	nop
	mov %l0,%o0
	mov %l3,%o1
	call _redirect_jump,0
	nop
	mov 1,%o0
	st %o0,[%fp-20]
	mov %l0,%l1
L72:
	mov %l0,%o0
	call _next_label,0
	nop
	st %o0,[%fp-40]
	ld [%fp-40],%o0
	cmp %o0,0
	be L73
	nop
	ld [%fp-40],%o0
	call _prev_real_insn,0
	nop
	b L74
	nop
L73:
	mov 0,%o0
L74:
	st %o0,[%fp-36]
	ld [%fp-24],%o0
	cmp %o0,0
	bne L75
	nop
	mov %l0,%o0
	call _condjump_p,0
	nop
	cmp %o0,0
	be L75
	nop
	ld [%l0+32],%o0
	ld [%fp-40],%o1
	cmp %o0,%o1
	bne L75
	nop
	ld [%fp-40],%o0
	ld [%o0+20],%o1
	cmp %o1,1
	bne L75
	nop
	ld [%fp-36],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,14
	bne L75
	nop
	ld [%fp-36],%o0
	call _simplejump_p,0
	nop
	cmp %o0,0
	be L75
	nop
	ld [%fp-40],%o0
	call _next_label,0
	nop
	st %o0,[%fp-48]
	ld [%fp-48],%o0
	cmp %o0,0
	be L76
	nop
	ld [%fp-48],%o0
	call _prev_real_insn,0
	nop
	b L77
	nop
L76:
	mov 0,%o0
L77:
	st %o0,[%fp-52]
	ld [%fp-36],%o0
	ld [%fp-52],%o1
	cmp %o0,%o1
	be L78
	nop
	ld [%fp-36],%o0
	ld [%o0+32],%o1
	ld [%fp-48],%o0
	cmp %o1,%o0
	bne L78
	nop
	ld [%fp-52],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,14
	bne L78
	nop
	ld [%fp-52],%o0
	ld [%o0+12],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,16
	bne L78
	nop
	ld [%l0+12],%o0
	st %o0,[%fp-56]
	ld [%fp-40],%o0
	ld [%o0+12],%o1
	st %o1,[%fp-60]
	ld [%fp-36],%o0
	ld [%o0+12],%o1
	st %o1,[%fp-64]
	ld [%fp-52],%o0
	ld [%o0+12],%o1
	st %o1,[%fp-68]
	ld [%fp-60],%o0
	st %o0,[%l0+12]
	ld [%fp-60],%o0
	st %l0,[%o0+8]
	ld [%fp-52],%o0
	ld [%fp-64],%o1
	st %o1,[%o0+12]
	ld [%fp-64],%o0
	ld [%fp-52],%o1
	st %o1,[%o0+8]
	ld [%fp-40],%o0
	ld [%fp-56],%o1
	st %o1,[%o0+12]
	ld [%fp-56],%o0
	ld [%fp-40],%o1
	st %o1,[%o0+8]
	ld [%fp-36],%o0
	ld [%fp-68],%o1
	st %o1,[%o0+12]
	ld [%fp-68],%o0
	ld [%fp-36],%o1
	st %o1,[%o0+8]
	mov %l0,%o0
	ld [%fp-40],%o1
	call _invert_jump,0
	nop
	mov 1,%o0
	st %o0,[%fp-20]
	b L59
	nop
L78:
L75:
	ld [%fp+72],%o0
	cmp %o0,0
	be L79
	nop
	mov %l0,%o0
	call _condjump_p,0
	nop
	cmp %o0,0
	be L79
	nop
	ld [%l0+32],%o0
	call _prev_real_insn,0
	nop
	st %o0,[%fp-60]
	ld [%fp-60],%o0
	cmp %o0,0
	be L80
	nop
	ld [%fp-60],%o0
	mov %l0,%o1
	call _jump_back_p,0
	nop
	cmp %o0,0
	bne L80
	nop
	st %g0,[%fp-60]
L80:
	st %g0,[%fp-68]
	ld [%fp-60],%o0
	cmp %o0,0
	be L81
	nop
	add %fp,-68,%o3
	add %fp,-64,%o4
	mov %l0,%o0
	ld [%fp-60],%o1
	mov 2,%o2
	call _find_cross_jump,0
	nop
L81:
	ld [%fp-68],%o0
	cmp %o0,0
	be L82
	nop
	mov %l0,%o0
	ld [%fp-68],%o1
	ld [%fp-64],%o2
	call _do_cross_jump,0
	nop
	mov 38,%o0
	mov 0,%o1
	ld [%l0+32],%o2
	call _gen_rtx,0
	nop
	ld [%l0+16],%o1
	st %o0,[%o1+8]
	mov %l0,%o0
	call _emit_barrier_after,0
	nop
	mov 1,%o0
	st %o0,[%fp-20]
	mov %l0,%l1
L82:
L79:
	ld [%fp+72],%o0
	cmp %o0,0
	be L83
	nop
	mov %l0,%o0
	call _simplejump_p,0
	nop
	cmp %o0,0
	be L83
	nop
	st %g0,[%fp-56]
	add %fp,-56,%o3
	add %fp,-52,%o4
	mov %l0,%o0
	ld [%l0+32],%o1
	mov 1,%o2
	call _find_cross_jump,0
	nop
	ld [%l0+32],%o0
	ld [%o0+4],%o1
	ld [%fp-28],%o0
	cmp %o1,%o0
	bge L84
	nop
	ld [%l0+32],%o0
	ld [%o0+4],%o1
	mov %o1,%o2
	sll %o2,2,%o0
	sethi %hi(_jump_chain),%o2
	ld [%o2+%lo(_jump_chain)],%o1
	add %o0,%o1,%o0
	ld [%o0],%o1
	st %o1,[%fp-72]
L85:
	ld [%fp-72],%o0
	cmp %o0,0
	be L86
	nop
	ld [%fp-56],%o0
	cmp %o0,0
	bne L86
	nop
	ld [%fp-72],%o0
	cmp %o0,%l0
	be L88
	nop
	ld [%fp-72],%o0
	ld [%o0+32],%o1
	ld [%l0+32],%o0
	cmp %o1,%o0
	bne L88
	nop
	ld [%fp-72],%o0
	ld [%o0],%o1
	and %o1,16,%o0
	cmp %o0,0
	bne L88
	nop
	add %fp,-56,%o3
	add %fp,-52,%o4
	mov %l0,%o0
	ld [%fp-72],%o1
	mov 2,%o2
	call _find_cross_jump,0
	nop
L88:
L87:
	ld [%fp-72],%o0
	ld [%o0+4],%o1
	mov %o1,%o2
	sll %o2,2,%o0
	sethi %hi(_jump_chain),%o2
	ld [%o2+%lo(_jump_chain)],%o1
	add %o0,%o1,%o0
	ld [%o0],%o1
	st %o1,[%fp-72]
	b L85
	nop
L86:
L84:
	ld [%fp-56],%o0
	cmp %o0,0
	be L89
	nop
	mov %l0,%o0
	ld [%fp-56],%o1
	ld [%fp-52],%o2
	call _do_cross_jump,0
	nop
	mov 1,%o0
	st %o0,[%fp-20]
	mov %l0,%l1
L89:
L83:
L71:
L69:
L67:
L65:
	b L90
	nop
L63:
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,14
	bne L91
	nop
	ld [%l0+16],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,29
	bne L91
	nop
	ld [%fp+72],%o0
	cmp %o0,0
	be L92
	nop
	st %g0,[%fp-72]
	sethi %hi(_jump_chain),%o1
	ld [%o1+%lo(_jump_chain)],%o0
	ld [%o0],%o1
	st %o1,[%fp-64]
L93:
	ld [%fp-64],%o0
	cmp %o0,0
	be L94
	nop
	ld [%fp-72],%o0
	cmp %o0,0
	bne L94
	nop
	ld [%fp-64],%o0
	cmp %o0,%l0
	be L96
	nop
	ld [%fp-64],%o0
	ld [%o0],%o1
	and %o1,16,%o0
	cmp %o0,0
	bne L96
	nop
	ld [%fp-64],%o0
	ld [%o0+16],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,29
	bne L96
	nop
	add %fp,-72,%o3
	add %fp,-68,%o4
	mov %l0,%o0
	ld [%fp-64],%o1
	mov 2,%o2
	call _find_cross_jump,0
	nop
L96:
L95:
	ld [%fp-64],%o0
	ld [%o0+4],%o1
	mov %o1,%o2
	sll %o2,2,%o0
	sethi %hi(_jump_chain),%o2
	ld [%o2+%lo(_jump_chain)],%o1
	add %o0,%o1,%o0
	ld [%o0],%o1
	st %o1,[%fp-64]
	b L93
	nop
L94:
	ld [%fp-72],%o0
	cmp %o0,0
	be L97
	nop
	mov %l0,%o0
	ld [%fp-72],%o1
	ld [%fp-68],%o2
	call _do_cross_jump,0
	nop
	mov 1,%o0
	st %o0,[%fp-20]
	mov %l0,%l1
L97:
L92:
L91:
L90:
L59:
	mov %l1,%l0
	b L57
	nop
L58:
	st %g0,[%fp-24]
	b L55
	nop
L56:
	ld [%fp-32],%l0
L98:
	cmp %l0,0
	be L99
	nop
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,17
	be L100
	nop
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,14
	bne L101
	nop
	ld [%l0+16],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,29
	be L100
	nop
	b L101
	nop
L101:
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,18
	bne L99
	nop
	ld [%l0+20],%o0
	cmp %o0,-6
	bne L100
	nop
	b L99
	nop
L100:
	ld [%l0+8],%l0
	b L98
	nop
L99:
	cmp %l0,0
	be L102
	nop
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,18
	bne L102
	nop
	ld [%l0+20],%o0
	cmp %o0,-6
	bne L102
	nop
	sethi %hi(_current_function_returns_null),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_current_function_returns_null)]
	mov %l0,%o0
	call _delete_insn,0
	nop
L102:
L1:
	ret
	restore
	.align 4
	.proc	020
_find_cross_jump:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	st %i4,[%fp+84]
	ld [%fp+68],%l0
	ld [%fp+72],%l1
	st %g0,[%fp-20]
	st %g0,[%fp-24]
	st %g0,[%fp-28]
	st %g0,[%fp-32]
	ld [%fp+80],%o0
	st %g0,[%o0]
	ld [%fp+84],%o0
	st %g0,[%o0]
L104:
	ld [%l0+8],%l0
L106:
	cmp %l0,0
	be L107
	nop
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,18
	bne L107
	nop
	ld [%l0+8],%l0
	b L106
	nop
L107:
	ld [%l1+8],%l1
L108:
	cmp %l1,0
	be L109
	nop
	lduh [%l1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,18
	be L110
	nop
	lduh [%l1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,17
	be L110
	nop
	b L109
	nop
L110:
	ld [%l1+8],%l1
	b L108
	nop
L109:
	cmp %l0,0
	bne L111
	nop
	b L105
	nop
L111:
	ld [%fp+68],%o0
	cmp %l1,%o0
	be L113
	nop
	ld [%fp+72],%o0
	cmp %l0,%o0
	be L113
	nop
	b L112
	nop
L113:
	b L105
	nop
L112:
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,17
	bne L114
	nop
	ld [%fp+76],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp+76]
	b L105
	nop
L114:
	cmp %l1,0
	be L116
	nop
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	lduh [%l1],%o1
	sll %o1,16,%o2
	srl %o2,16,%o1
	cmp %o0,%o1
	bne L116
	nop
	b L115
	nop
L116:
	b L105
	nop
L115:
	ld [%l0+16],%l2
	ld [%l1+16],%l3
	lduh [%l2],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	lduh [%l3],%o1
	sll %o1,16,%o2
	srl %o2,16,%o1
	cmp %o0,%o1
	bne L118
	nop
	mov %l2,%o0
	mov %l3,%o1
	call _rtx_renumbered_equal_p,0
	nop
	cmp %o0,0
	bne L117
	nop
	b L118
	nop
L118:
	mov %l2,%o0
	call _sets_cc0_p,0
	nop
	cmp %o0,0
	bne L120
	nop
	mov %l3,%o0
	call _sets_cc0_p,0
	nop
	cmp %o0,0
	bne L120
	nop
	b L119
	nop
L120:
	ld [%fp-28],%o0
	st %o0,[%fp-20]
	ld [%fp-32],%o0
	st %o0,[%fp-24]
	ld [%fp+76],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp+76]
L119:
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,14
	bne L121
	nop
	ld [%l0+32],%o0
	cmp %o0,0
	be L121
	nop
	ld [%l0+32],%o0
	call _prev_real_insn,0
	nop
	ld [%fp+68],%o1
	cmp %o0,%o1
	bne L121
	nop
	ld [%fp+76],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp+76]
L121:
	b L105
	nop
L117:
	lduh [%l2],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,26
	be L122
	nop
	lduh [%l2],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,27
	be L122
	nop
	ld [%fp-20],%o0
	st %o0,[%fp-28]
	ld [%fp-24],%o0
	st %o0,[%fp-32]
	st %l0,[%fp-20]
	st %l1,[%fp-24]
	ld [%fp+76],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp+76]
L122:
	b L104
	nop
L105:
	ld [%fp+76],%o0
	cmp %o0,0
	bg L123
	nop
	ld [%fp-20],%o0
	cmp %o0,0
	be L123
	nop
	ld [%fp+80],%o0
	ld [%fp-20],%o1
	st %o1,[%o0]
	ld [%fp+84],%o0
	ld [%fp-24],%o1
	st %o1,[%o0]
L123:
L103:
	ret
	restore
	.align 4
	.proc	020
_do_cross_jump:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	ld [%fp+76],%o0
	ld [%o0+8],%l0
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,17
	be L125
	nop
	call _gen_label_rtx,0
	nop
	mov %o0,%l0
	ld [%fp+76],%o1
	mov %l0,%o0
	ld [%o1+8],%o1
	call _emit_label_after,0
	nop
	st %g0,[%l0+20]
L125:
	ld [%fp+68],%o0
	ld [%o0+16],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,29
	bne L126
	nop
	mov %l0,%o0
	call _gen_jump,0
	nop
	ld [%fp+68],%o1
	st %o0,[%o1+16]
	ld [%fp+68],%o0
	mov -1,%o1
	st %o1,[%o0+20]
	ld [%fp+68],%o0
	st %l0,[%o0+32]
	ld [%l0+20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%l0+20]
	b L127
	nop
L126:
	ld [%fp+68],%o0
	mov %l0,%o1
	call _redirect_jump,0
	nop
L127:
	ld [%fp+72],%o0
	ld [%o0+8],%o1
	st %o1,[%fp+72]
L128:
	ld [%fp+72],%o0
	ld [%o0+12],%o1
	ld [%fp+68],%o0
	cmp %o1,%o0
	be L129
	nop
	ld [%fp+72],%o0
	ld [%o0+12],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,18
	be L130
	nop
	ld [%fp+72],%o1
	ld [%o1+12],%o0
	call _delete_insn,0
	nop
	b L131
	nop
L130:
	ld [%fp+72],%o0
	ld [%o0+12],%o1
	st %o1,[%fp+72]
L131:
	b L128
	nop
L129:
L124:
	ret
	restore
	.align 4
	.proc	04
_jump_back_p:
	!#PROLOGUE# 0
	save %sp,-136,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
	call _simplejump_p,0
	nop
	cmp %o0,0
	bne L134
	nop
	ld [%fp+68],%o0
	call _condjump_p,0
	nop
	cmp %o0,0
	be L134
	nop
	ld [%fp+72],%o0
	call _simplejump_p,0
	nop
	cmp %o0,0
	bne L134
	nop
	b L133
	nop
L134:
	mov 0,%i0
	b L132
	nop
L133:
	ld [%fp+68],%o1
	ld [%o1+32],%o0
	call _prev_real_insn,0
	nop
	ld [%fp+72],%o1
	cmp %o1,%o0
	be L135
	nop
	mov 0,%i0
	b L132
	nop
L135:
	ld [%fp+68],%o0
	call _prev_real_insn,0
	nop
	st %o0,[%fp-28]
	ld [%fp-28],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,13
	bne L137
	nop
	ld [%fp-28],%o0
	ld [%o0+16],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,25
	bne L137
	nop
	ld [%fp-28],%o0
	ld [%o0+16],%o1
	sethi %hi(_cc0_rtx),%o0
	ld [%o1+4],%o1
	ld [%o0+%lo(_cc0_rtx)],%o0
	cmp %o1,%o0
	bne L137
	nop
	ld [%fp-28],%o0
	ld [%o0+16],%o1
	ld [%o1+8],%o0
	ld [%o0],%o1
	srl %o1,8,%o0
	and %o0,255,%o1
	mov %o1,%o2
	sll %o2,2,%o0
	sethi %hi(_mode_class),%o2
	or %o2,%lo(_mode_class),%o1
	ld [%o0+%o1],%o0
	cmp %o0,1
	be L136
	nop
	ld [%fp-28],%o0
	ld [%o0+16],%o1
	ld [%o1+8],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,43
	bne L137
	nop
	ld [%fp-28],%o0
	ld [%o0+16],%o1
	ld [%o1+8],%o0
	ld [%o0+4],%o1
	ld [%o1],%o2
	srl %o2,8,%o0
	and %o0,255,%o1
	mov %o1,%o2
	sll %o2,2,%o0
	sethi %hi(_mode_class),%o2
	or %o2,%lo(_mode_class),%o1
	ld [%o0+%o1],%o0
	cmp %o0,1
	bne L137
	nop
	b L136
	nop
L137:
	mov 0,%i0
	b L132
	nop
L136:
	ld [%fp+68],%o0
	ld [%o0+16],%o1
	ld [%o1+8],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-20]
	ld [%fp+72],%o0
	ld [%o0+16],%o1
	ld [%o1+8],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-24]
	ld [%fp-20],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	st %o0,[%fp-32]
	ld [%fp-24],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	st %o0,[%fp-36]
	ld [%fp+68],%o0
	ld [%o0+16],%o1
	ld [%o1+8],%o0
	sethi %hi(_pc_rtx),%o1
	ld [%o0+8],%o0
	ld [%o1+%lo(_pc_rtx)],%o1
	cmp %o0,%o1
	bne L138
	nop
	ld [%fp-32],%o0
	call _reverse_condition,0
	nop
	st %o0,[%fp-32]
L138:
	ld [%fp+72],%o0
	ld [%o0+16],%o1
	ld [%o1+8],%o0
	sethi %hi(_pc_rtx),%o1
	ld [%o0+12],%o0
	ld [%o1+%lo(_pc_rtx)],%o1
	cmp %o0,%o1
	bne L139
	nop
	ld [%fp-36],%o0
	call _reverse_condition,0
	nop
	st %o0,[%fp-36]
L139:
	mov 0,%l0
	ld [%fp-32],%o0
	ld [%fp-36],%o1
	cmp %o0,%o1
	bne L140
	nop
	ld [%fp-20],%o0
	ld [%fp-24],%o1
	ld [%o0+4],%o0
	ld [%o1+4],%o1
	call _rtx_renumbered_equal_p,0
	nop
	cmp %o0,0
	be L140
	nop
	ld [%fp-20],%o0
	ld [%fp-24],%o1
	ld [%o0+8],%o0
	ld [%o1+8],%o1
	call _rtx_renumbered_equal_p,0
	nop
	cmp %o0,0
	be L140
	nop
	mov 1,%l0
L140:
	mov %l0,%i0
	b L132
	nop
L132:
	ret
	restore
	.align 4
	.proc	012
_reverse_condition:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o1
	add %o1,-67,%o0
	mov 9,%o1
	cmp %o1,%o0
	blu L153
	nop
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(L154),%o2
	or %o2,%lo(L154),%o1
	ld [%o0+%o1],%o0
	jmp %o0
	nop
L154:
	.word	L144
	.word	L143
	.word	L146
	.word	L145
	.word	L148
	.word	L147
	.word	L150
	.word	L149
	.word	L152
	.word	L151
L143:
	mov 67,%i0
	b L141
	nop
L144:
	mov 68,%i0
	b L141
	nop
L145:
	mov 71,%i0
	b L141
	nop
L146:
	mov 72,%i0
	b L141
	nop
L147:
	mov 69,%i0
	b L141
	nop
L148:
	mov 70,%i0
	b L141
	nop
L149:
	mov 75,%i0
	b L141
	nop
L150:
	mov 76,%i0
	b L141
	nop
L151:
	mov 73,%i0
	b L141
	nop
L152:
	mov 74,%i0
	b L141
	nop
L153:
	call _abort,0
	nop
	mov 0,%i0
	b L141
	nop
L142:
L141:
	ret
	restore
	.align 4
	.global _simplejump_p
	.proc	04
_simplejump_p:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o1
	ld [%o1+16],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o1
	cmp %o1,25
	be L156
	nop
	mov 0,%i0
	b L155
	nop
L156:
	ld [%o0+4],%o1
	lduh [%o1],%o2
	sll %o2,16,%o3
	srl %o3,16,%o1
	cmp %o1,33
	be L157
	nop
	mov 0,%i0
	b L155
	nop
L157:
	ld [%o0+8],%o1
	lduh [%o1],%o2
	sll %o2,16,%o3
	srl %o3,16,%o1
	cmp %o1,38
	be L158
	nop
	mov 0,%i0
	b L155
	nop
L158:
	mov 1,%i0
	b L155
	nop
L155:
	ret
	restore
	.align 4
	.global _condjump_p
	.proc	04
_condjump_p:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o1
	ld [%o1+16],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o1
	cmp %o1,25
	be L160
	nop
	mov 0,%i0
	b L159
	nop
L160:
	ld [%o0+4],%o1
	lduh [%o1],%o2
	sll %o2,16,%o3
	srl %o3,16,%o1
	cmp %o1,33
	be L161
	nop
	mov 0,%i0
	b L159
	nop
L161:
	ld [%o0+8],%o1
	lduh [%o1],%o2
	sll %o2,16,%o3
	srl %o3,16,%o1
	cmp %o1,38
	bne L162
	nop
	mov 1,%i0
	b L159
	nop
L162:
	ld [%o0+8],%o1
	lduh [%o1],%o2
	sll %o2,16,%o3
	srl %o3,16,%o1
	cmp %o1,42
	be L163
	nop
	mov 0,%i0
	b L159
	nop
L163:
	ld [%o0+8],%o1
	sethi %hi(_pc_rtx),%o2
	ld [%o1+12],%o1
	ld [%o2+%lo(_pc_rtx)],%o2
	cmp %o1,%o2
	bne L164
	nop
	ld [%o0+8],%o1
	ld [%o1+8],%o2
	lduh [%o2],%o1
	sll %o1,16,%o2
	srl %o2,16,%o1
	cmp %o1,38
	bne L164
	nop
	mov 1,%i0
	b L159
	nop
L164:
	ld [%o0+8],%o1
	sethi %hi(_pc_rtx),%o2
	ld [%o1+8],%o1
	ld [%o2+%lo(_pc_rtx)],%o2
	cmp %o1,%o2
	bne L165
	nop
	ld [%o0+8],%o1
	ld [%o1+12],%o2
	lduh [%o2],%o1
	sll %o1,16,%o2
	srl %o2,16,%o1
	cmp %o1,38
	bne L165
	nop
	mov 1,%i0
	b L159
	nop
L165:
	mov 0,%i0
	b L159
	nop
L159:
	ret
	restore
	.align 4
	.global _sets_cc0_p
	.proc	04
_sets_cc0_p:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,25
	bne L167
	nop
	ld [%fp+68],%o0
	sethi %hi(_cc0_rtx),%o1
	ld [%o0+4],%o0
	ld [%o1+%lo(_cc0_rtx)],%o1
	cmp %o0,%o1
	bne L167
	nop
	mov 1,%i0
	b L166
	nop
L167:
	ld [%fp+68],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,20
	bne L168
	nop
	st %g0,[%fp-24]
	st %g0,[%fp-28]
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	ld [%o1],%o0
	add %o0,-1,%o1
	st %o1,[%fp-20]
L169:
	ld [%fp-20],%o0
	cmp %o0,0
	bl L170
	nop
	ld [%fp+68],%o1
	ld [%o1+4],%o0
	ld [%fp-20],%o1
	mov %o1,%o2
	sll %o2,2,%o1
	add %o0,%o1,%o2
	ld [%o2+4],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,25
	bne L172
	nop
	ld [%fp+68],%o1
	ld [%o1+4],%o0
	ld [%fp-20],%o1
	mov %o1,%o2
	sll %o2,2,%o1
	add %o0,%o1,%o2
	ld [%o2+4],%o0
	sethi %hi(_cc0_rtx),%o1
	ld [%o0+4],%o0
	ld [%o1+%lo(_cc0_rtx)],%o1
	cmp %o0,%o1
	bne L172
	nop
	mov 1,%o0
	st %o0,[%fp-24]
	b L173
	nop
L172:
	ld [%fp+68],%o1
	ld [%o1+4],%o0
	ld [%fp-20],%o1
	mov %o1,%o2
	sll %o2,2,%o1
	add %o0,%o1,%o2
	ld [%o2+4],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,25
	bne L174
	nop
	mov 1,%o0
	st %o0,[%fp-28]
L174:
L173:
L171:
	ld [%fp-20],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	b L169
	nop
L170:
	ld [%fp-24],%o0
	cmp %o0,0
	be L175
	nop
	ld [%fp-28],%o0
	cmp %o0,0
	be L177
	nop
	mov -1,%o0
	b L178
	nop
L177:
	mov 1,%o0
L178:
	b L176
	nop
L175:
	mov 0,%o0
L176:
	mov %o0,%i0
	b L166
	nop
L168:
	mov 0,%i0
	b L166
	nop
L166:
	ret
	restore
	.align 4
	.global _no_labels_between_p
	.proc	04
_no_labels_between_p:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
L180:
	ld [%fp+72],%o1
	cmp %o0,%o1
	be L181
	nop
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o1
	cmp %o1,17
	bne L183
	nop
	mov 0,%i0
	b L179
	nop
L183:
L182:
	ld [%o0+12],%o0
	b L180
	nop
L181:
	mov 1,%i0
	b L179
	nop
L179:
	ret
	restore
	.align 4
	.global _prev_real_insn
	.proc	0110
_prev_real_insn:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o1
	ld [%o1+8],%o0
L185:
	cmp %o0,0
	bne L187
	nop
	mov 0,%i0
	b L184
	nop
L187:
	lduh [%o0],%o2
	sll %o2,16,%o3
	srl %o3,16,%o1
	cmp %o1,13
	be L189
	nop
	cmp %o1,15
	be L189
	nop
	cmp %o1,14
	be L189
	nop
	b L188
	nop
L189:
	b L186
	nop
L188:
	ld [%o0+8],%o0
	b L185
	nop
L186:
	mov %o0,%i0
	b L184
	nop
L184:
	ret
	restore
	.align 4
	.global _next_real_insn
	.proc	0110
_next_real_insn:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o1
	ld [%o1+12],%o0
L191:
	cmp %o0,0
	bne L193
	nop
	mov %o0,%i0
	b L190
	nop
L193:
	lduh [%o0],%o2
	sll %o2,16,%o3
	srl %o3,16,%o1
	cmp %o1,13
	be L195
	nop
	cmp %o1,15
	be L195
	nop
	cmp %o1,14
	be L195
	nop
	b L194
	nop
L195:
	b L192
	nop
L194:
	ld [%o0+12],%o0
	b L191
	nop
L192:
	mov %o0,%i0
	b L190
	nop
L190:
	ret
	restore
	.align 4
	.global _next_label
	.proc	0110
_next_label:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
L197:
	ld [%fp+68],%o0
	ld [%o0+12],%o1
	st %o1,[%fp+68]
L199:
	ld [%fp+68],%o0
	cmp %o0,0
	be L198
	nop
	ld [%fp+68],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,17
	be L198
	nop
	b L197
	nop
L198:
	ld [%fp+68],%i0
	b L196
	nop
L196:
	ret
	restore
	.align 4
	.proc	0110
_follow_jumps:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%l2
	mov 0,%l3
L201:
	cmp %l3,9
	bg L202
	nop
	mov %l2,%o0
	call _next_real_insn,0
	nop
	mov %o0,%l0
	mov %l0,%o0
	cmp %o0,0
	be L202
	nop
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,14
	bne L202
	nop
	ld [%l0+32],%o0
	cmp %o0,0
	be L202
	nop
	ld [%l0+12],%l1
	mov %l1,%o0
	cmp %o0,0
	be L202
	nop
	lduh [%l1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,16
	bne L202
	nop
	ld [%fp+72],%o0
	cmp %o0,0
	bne L204
	nop
	st %l2,[%fp-20]
L205:
	ld [%fp-20],%o0
	cmp %o0,%l0
	be L206
	nop
	ld [%fp-20],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,18
	bne L208
	nop
	ld [%fp-20],%o0
	ld [%o0+20],%o1
	cmp %o1,-4
	bne L208
	nop
	mov %l2,%i0
	b L200
	nop
L208:
L207:
	ld [%fp-20],%o0
	ld [%o0+12],%o1
	st %o1,[%fp-20]
	b L205
	nop
L206:
L204:
	ld [%l0+32],%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	bne L209
	nop
	b L202
	nop
L209:
	ld [%l0+32],%l2
L203:
	add %l3,1,%l3
	b L201
	nop
L202:
	mov %l2,%i0
	b L200
	nop
L200:
	ret
	restore
	.align 4
	.proc	04
_tension_vector_labels:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	mov %i0,%l0
	mov %i1,%l1
	st %i2,[%fp+76]
	st %g0,[%fp-20]
	mov %l1,%o0
	sll %o0,2,%o1
	add %l0,%o1,%o2
	ld [%o2+4],%o0
	ld [%o0],%o1
	add %o1,-1,%l2
L211:
	cmp %l2,0
	bl L212
	nop
	mov %l1,%o0
	sll %o0,2,%o1
	add %l0,%o1,%o2
	ld [%o2+4],%o0
	mov %l2,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	ld [%o0+4],%l3
	mov %l3,%o0
	ld [%fp+76],%o1
	call _follow_jumps,0
	nop
	mov %o0,%l4
	cmp %l4,%l3
	be L214
	nop
	mov %l1,%o0
	sll %o0,2,%o1
	add %l0,%o1,%o2
	ld [%o2+4],%o0
	mov %l2,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	st %l4,[%o0+4]
	ld [%l4+20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%l4+20]
	ld [%l3+20],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%l3+20]
	cmp %o1,0
	bne L215
	nop
	mov %l3,%o0
	call _delete_insn,0
	nop
L215:
	mov 1,%o0
	st %o0,[%fp-20]
L214:
L213:
	add %l2,-1,%l2
	b L211
	nop
L212:
	ld [%fp-20],%i0
	b L210
	nop
L210:
	ret
	restore
	.align 4
	.proc	020
_mark_jump_label:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	mov %i0,%l0
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%l1
	cmp %l1,38
	bne L217
	nop
	ld [%l0+4],%o0
	lduh [%o0],%o2
	sll %o2,16,%o3
	srl %o3,16,%o2
	cmp %o2,17
	be L218
	nop
	b L216
	nop
L218:
	nop
	ld [%o0+12],%o1
L219:
	cmp %o1,0
	be L220
	nop
	lduh [%o1],%o2
	sll %o2,16,%o3
	srl %o3,16,%o2
	cmp %o2,17
	bne L222
	nop
	mov %o1,%o0
	b L223
	nop
L222:
	lduh [%o1],%o2
	sll %o2,16,%o3
	srl %o3,16,%o2
	cmp %o2,18
	bne L225
	nop
	ld [%o1+20],%o2
	cmp %o2,-4
	be L225
	nop
	ld [%o1+20],%o2
	cmp %o2,-6
	be L225
	nop
	b L224
	nop
L225:
	b L220
	nop
L224:
L223:
L221:
	ld [%o1+12],%o1
	b L219
	nop
L220:
	st %o0,[%l0+4]
	ld [%o0+20],%o3
	add %o3,1,%o2
	mov %o2,%o3
	st %o3,[%o0+20]
	ld [%fp+72],%o2
	cmp %o2,0
	be L226
	nop
	ld [%fp+72],%o2
	st %o0,[%o2+32]
L226:
	b L216
	nop
L217:
	cmp %l1,23
	be L228
	nop
	cmp %l1,24
	be L228
	nop
	b L227
	nop
L228:
	st %g0,[%fp+72]
L227:
	mov %l1,%o1
	sll %o1,2,%o0
	sethi %hi(_rtx_format),%o2
	or %o2,%lo(_rtx_format),%o1
	ld [%o0+%o1],%l3
	mov %l1,%o1
	sll %o1,2,%o0
	sethi %hi(_rtx_length),%o2
	or %o2,%lo(_rtx_length),%o1
	ld [%o0+%o1],%l2
L229:
	cmp %l2,0
	bl L230
	nop
	ldub [%l3+%l2],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,101
	bne L232
	nop
	mov %l2,%o0
	sll %o0,2,%o1
	add %l0,%o1,%o2
	ld [%o2+4],%o0
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	call _mark_jump_label,0
	nop
	b L233
	nop
L232:
	ldub [%l3+%l2],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,69
	bne L234
	nop
	mov 0,%l4
L235:
	mov %l2,%o0
	sll %o0,2,%o1
	add %l0,%o1,%o2
	ld [%o2+4],%o0
	ld [%o0],%o1
	cmp %l4,%o1
	bgeu L236
	nop
	mov %l2,%o0
	sll %o0,2,%o1
	add %l0,%o1,%o2
	ld [%o2+4],%o0
	mov %l4,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	call _mark_jump_label,0
	nop
L237:
	add %l4,1,%l4
	b L235
	nop
L236:
L234:
L233:
L231:
	add %l2,-1,%l2
	b L229
	nop
L230:
L216:
	ret
	restore
	.align 4
	.proc	020
_delete_jump:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	ld [%o0+16],%l0
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,25
	bne L239
	nop
	ld [%l0+4],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,33
	bne L239
	nop
	ld [%fp+68],%o0
	ld [%o0+8],%l1
	ld [%fp+68],%o0
	call _delete_insn,0
	nop
L240:
	cmp %l1,0
	be L241
	nop
	lduh [%l1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,18
	bne L241
	nop
	ld [%l1+8],%l1
	b L240
	nop
L241:
	cmp %l1,0
	be L242
	nop
	lduh [%l1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,13
	bne L242
	nop
	ld [%l1+16],%o0
	call _sets_cc0_p,0
	nop
	cmp %o0,0
	ble L242
	nop
	mov %l1,%o0
	mov 2,%o1
	mov 0,%o2
	call _find_reg_note,0
	nop
	cmp %o0,0
	bne L242
	nop
	mov %l1,%o0
	call _delete_insn,0
	nop
L242:
L239:
L238:
	ret
	restore
	.align 4
	.global _delete_insn
	.proc	0110
_delete_insn:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	mov %i0,%l0
	ld [%l0+12],%l1
	ld [%l0+8],%l2
	ld [%l0],%o1
	and %o1,16,%o0
	cmp %o0,0
	be L244
	nop
L245:
	cmp %l1,0
	be L246
	nop
	ld [%l1],%o1
	and %o1,16,%o0
	cmp %o0,0
	be L246
	nop
	ld [%l1+12],%l1
	b L245
	nop
L246:
	mov %l1,%i0
	b L243
	nop
L244:
	ld [%l0],%o0
	or %o0,16,%o1
	st %o1,[%l0]
	cmp %l1,0
	be L247
	nop
	lduh [%l1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,16
	bne L247
	nop
	ld [%l1],%o0
	or %o0,16,%o1
	st %o1,[%l1]
	ld [%l1+12],%l1
L247:
	sethi %hi(_optimize),%o0
	ld [%o0+%lo(_optimize)],%o1
	cmp %o1,0
	be L248
	nop
	cmp %l2,0
	be L249
	nop
	st %l1,[%l2+12]
L249:
	cmp %l1,0
	be L250
	nop
	st %l2,[%l1+8]
L250:
L248:
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,14
	bne L251
	nop
	ld [%l0+32],%o0
	cmp %o0,0
	be L251
	nop
	ld [%l0+32],%o1
	ld [%l0+32],%o0
	ld [%l0+32],%o1
	ld [%o1+20],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0+20]
	cmp %o2,0
	bne L252
	nop
	ld [%l0+32],%o0
	call _delete_insn,0
	nop
L253:
	cmp %l1,0
	be L254
	nop
	ld [%l1],%o1
	and %o1,16,%o0
	cmp %o0,0
	be L254
	nop
	ld [%l1+12],%l1
	b L253
	nop
L254:
	mov %l1,%i0
	b L243
	nop
L252:
L251:
	nop
L255:
	cmp %l2,0
	be L256
	nop
	ld [%l2],%o1
	and %o1,16,%o0
	cmp %o0,0
	bne L257
	nop
	lduh [%l2],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,18
	be L257
	nop
	b L256
	nop
L257:
	ld [%l2+8],%l2
	b L255
	nop
L256:
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,17
	bne L258
	nop
	cmp %l2,0
	be L258
	nop
	lduh [%l2],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,16
	bne L258
	nop
L259:
	cmp %l1,0
	be L260
	nop
	lduh [%l1],%o0
	sll %o0,16,%o1
	srl %o1,16,%l3
	mov %l3,%o0
	cmp %o0,13
	be L261
	nop
	cmp %l3,14
	be L261
	nop
	cmp %l3,15
	be L261
	nop
	cmp %l3,18
	be L261
	nop
	b L260
	nop
L261:
	cmp %l3,18
	bne L262
	nop
	ld [%l1+20],%o0
	cmp %o0,-6
	be L262
	nop
	ld [%l1+12],%l1
	b L263
	nop
L262:
	mov %l1,%o0
	call _delete_insn,0
	nop
	mov %o0,%l1
L263:
	b L259
	nop
L260:
L258:
	mov %l1,%i0
	b L243
	nop
L243:
	ret
	restore
	.align 4
	.global _next_nondeleted_insn
	.proc	0110
_next_nondeleted_insn:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
L265:
	ld [%fp+68],%o0
	ld [%o0],%o1
	and %o1,16,%o0
	cmp %o0,0
	be L266
	nop
	ld [%fp+68],%o0
	ld [%o0+12],%o1
	st %o1,[%fp+68]
	b L265
	nop
L266:
	ld [%fp+68],%i0
	b L264
	nop
L264:
	ret
	restore
	.align 4
	.global _delete_for_peephole
	.proc	020
_delete_for_peephole:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	mov %i0,%o0
	mov %i1,%o1
	mov %o0,%o2
L268:
	ld [%o2+12],%o3
	ld [%o2+8],%o4
	lduh [%o2],%o5
	sll %o5,16,%g2
	srl %g2,16,%o5
	cmp %o5,18
	be L270
	nop
	ld [%o2],%o5
	or %o5,16,%g2
	st %g2,[%o2]
	cmp %o4,0
	be L271
	nop
	st %o3,[%o4+12]
L271:
	cmp %o3,0
	be L272
	nop
	st %o4,[%o3+8]
L272:
L270:
	cmp %o2,%o1
	bne L273
	nop
	b L269
	nop
L273:
	mov %o3,%o2
	b L268
	nop
L269:
L267:
	ret
	restore
	.align 4
	.global _invert_jump
	.proc	020
_invert_jump:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
	ld [%o0+32],%l0
	ld [%fp+68],%o1
	ld [%o1+16],%o0
	mov %l0,%o1
	ld [%fp+72],%o2
	call _invert_exp,0
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	st %o1,[%o0+32]
	ld [%fp+72],%o1
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%o1+20],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+20]
	ld [%fp+68],%o0
	mov -1,%o1
	st %o1,[%o0+20]
	ld [%l0+20],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%l0+20]
	cmp %o1,0
	bne L275
	nop
	mov %l0,%o0
	call _delete_insn,0
	nop
L275:
L274:
	ret
	restore
	.align 4
	.global _invert_exp
	.proc	020
_invert_exp:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	ld [%fp+68],%o0
	cmp %o0,0
	bne L277
	nop
	b L276
	nop
L277:
	ld [%fp+68],%o0
	lduh [%o0],%o1
	sll %o1,16,%o0
	srl %o0,16,%l0
	cmp %l0,42
	bne L278
	nop
	ld [%fp+68],%o1
	ld [%o1+8],%o0
	ld [%fp+68],%o1
	ld [%fp+68],%o2
	ld [%o2+12],%o3
	st %o3,[%o1+8]
	ld [%fp+68],%o1
	st %o0,[%o1+12]
L278:
	cmp %l0,38
	bne L279
	nop
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	ld [%fp+72],%o0
	cmp %o1,%o0
	bne L280
	nop
	ld [%fp+68],%o0
	ld [%fp+76],%o1
	st %o1,[%o0+4]
L280:
	b L276
	nop
L279:
	mov %l0,%o1
	sll %o1,2,%o0
	sethi %hi(_rtx_format),%o2
	or %o2,%lo(_rtx_format),%o1
	ld [%o0+%o1],%l2
	mov %l0,%o1
	sll %o1,2,%o0
	sethi %hi(_rtx_length),%o2
	or %o2,%lo(_rtx_length),%o1
	ld [%o0+%o1],%o0
	add %o0,-1,%l1
L281:
	cmp %l1,0
	bl L282
	nop
	ldub [%l2+%l1],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,101
	bne L284
	nop
	ld [%fp+68],%o0
	mov %l1,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	call _invert_exp,0
	nop
L284:
	ldub [%l2+%l1],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,69
	bne L285
	nop
	mov 0,%l3
L286:
	ld [%fp+68],%o0
	mov %l1,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	ld [%o0],%o1
	cmp %l3,%o1
	bgeu L287
	nop
	ld [%fp+68],%o0
	mov %l1,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	mov %l3,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	call _invert_exp,0
	nop
L288:
	add %l3,1,%l3
	b L286
	nop
L287:
L285:
L283:
	add %l1,-1,%l1
	b L281
	nop
L282:
L276:
	ret
	restore
	.align 4
	.global _redirect_jump
	.proc	020
_redirect_jump:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
	ld [%o0+32],%l0
	ld [%fp+72],%o0
	cmp %o0,%l0
	bne L290
	nop
	b L289
	nop
L290:
	ld [%fp+68],%o1
	ld [%o1+16],%o0
	mov %l0,%o1
	ld [%fp+72],%o2
	call _redirect_exp,0
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	st %o1,[%o0+32]
	ld [%fp+72],%o1
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%o1+20],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+20]
	ld [%fp+68],%o0
	mov -1,%o1
	st %o1,[%o0+20]
	ld [%l0+20],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%l0+20]
	cmp %o1,0
	bne L291
	nop
	mov %l0,%o0
	call _delete_insn,0
	nop
L291:
L289:
	ret
	restore
	.align 4
	.proc	020
_redirect_exp:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	ld [%fp+68],%o0
	lduh [%o0],%o1
	sll %o1,16,%o0
	srl %o0,16,%l0
	cmp %l0,38
	bne L293
	nop
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	ld [%fp+72],%o0
	cmp %o1,%o0
	bne L294
	nop
	ld [%fp+68],%o0
	ld [%fp+76],%o1
	st %o1,[%o0+4]
L294:
	b L292
	nop
L293:
	mov %l0,%o1
	sll %o1,2,%o0
	sethi %hi(_rtx_format),%o2
	or %o2,%lo(_rtx_format),%o1
	ld [%o0+%o1],%l2
	mov %l0,%o1
	sll %o1,2,%o0
	sethi %hi(_rtx_length),%o2
	or %o2,%lo(_rtx_length),%o1
	ld [%o0+%o1],%o0
	add %o0,-1,%l1
L295:
	cmp %l1,0
	bl L296
	nop
	ldub [%l2+%l1],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,101
	bne L298
	nop
	ld [%fp+68],%o0
	mov %l1,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	call _redirect_exp,0
	nop
L298:
	ldub [%l2+%l1],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,69
	bne L299
	nop
	mov 0,%l3
L300:
	ld [%fp+68],%o0
	mov %l1,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	ld [%o0],%o1
	cmp %l3,%o1
	bgeu L301
	nop
	ld [%fp+68],%o0
	mov %l1,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	mov %l3,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	call _redirect_exp,0
	nop
L302:
	add %l3,1,%l3
	b L300
	nop
L301:
L299:
L297:
	add %l1,-1,%l1
	b L295
	nop
L296:
L292:
	ret
	restore
	.align 4
	.global _rtx_renumbered_equal_p
	.proc	04
_rtx_renumbered_equal_p:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
	lduh [%o0],%o1
	sll %o1,16,%o0
	srl %o0,16,%l1
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	cmp %o0,%o1
	bne L304
	nop
	mov 1,%i0
	b L303
	nop
L304:
	cmp %l1,34
	be L306
	nop
	cmp %l1,35
	bne L305
	nop
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,34
	be L306
	nop
	b L305
	nop
L306:
	ld [%fp+72],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,34
	be L307
	nop
	ld [%fp+72],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,35
	bne L305
	nop
	ld [%fp+72],%o0
	ld [%o0+4],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,34
	be L307
	nop
	b L305
	nop
L307:
	ld [%fp+68],%o2
	ld [%fp+72],%o1
	ldub [%o2+2],%o3
	and %o3,0xff,%o2
	ldub [%o1+2],%o3
	and %o3,0xff,%o1
	cmp %o2,%o1
	be L308
	nop
	mov 0,%i0
	b L303
	nop
L308:
	cmp %l1,35
	bne L309
	nop
	ld [%fp+68],%o1
	ld [%o1+4],%o2
	ld [%o2+4],%l0
	mov %l0,%o2
	sll %o2,1,%o1
	sethi %hi(_reg_renumber),%o3
	ld [%o3+%lo(_reg_renumber)],%o2
	add %o1,%o2,%o1
	lduh [%o1],%o2
	sll %o2,16,%o3
	sra %o3,16,%o1
	cmp %o1,0
	bl L310
	nop
	mov %l0,%o2
	sll %o2,1,%o1
	sethi %hi(_reg_renumber),%o3
	ld [%o3+%lo(_reg_renumber)],%o2
	add %o1,%o2,%o1
	lduh [%o1],%o2
	sll %o2,16,%o1
	sra %o1,16,%l0
L310:
	ld [%fp+68],%o1
	ld [%o1+8],%o2
	add %l0,%o2,%l0
	b L311
	nop
L309:
	ld [%fp+68],%o1
	ld [%o1+4],%l0
	mov %l0,%o2
	sll %o2,1,%o1
	sethi %hi(_reg_renumber),%o3
	ld [%o3+%lo(_reg_renumber)],%o2
	add %o1,%o2,%o1
	lduh [%o1],%o2
	sll %o2,16,%o3
	sra %o3,16,%o1
	cmp %o1,0
	bl L312
	nop
	mov %l0,%o2
	sll %o2,1,%o1
	sethi %hi(_reg_renumber),%o3
	ld [%o3+%lo(_reg_renumber)],%o2
	add %o1,%o2,%o1
	lduh [%o1],%o2
	sll %o2,16,%o1
	sra %o1,16,%l0
L312:
L311:
	ld [%fp+72],%o1
	lduh [%o1],%o2
	sll %o2,16,%o3
	srl %o3,16,%o1
	cmp %o1,35
	bne L313
	nop
	ld [%fp+72],%o1
	ld [%o1+4],%o2
	ld [%o2+4],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	sethi %hi(_reg_renumber),%o3
	ld [%o3+%lo(_reg_renumber)],%o2
	add %o1,%o2,%o1
	lduh [%o1],%o2
	sll %o2,16,%o3
	sra %o3,16,%o1
	cmp %o1,0
	bl L314
	nop
	mov %o0,%o2
	sll %o2,1,%o1
	sethi %hi(_reg_renumber),%o3
	ld [%o3+%lo(_reg_renumber)],%o2
	add %o1,%o2,%o1
	lduh [%o1],%o2
	sll %o2,16,%o1
	sra %o1,16,%o0
L314:
	ld [%fp+72],%o1
	ld [%o1+8],%o2
	add %o0,%o2,%o0
	b L315
	nop
L313:
	ld [%fp+72],%o1
	ld [%o1+4],%o0
	mov %o0,%o2
	sll %o2,1,%o1
	sethi %hi(_reg_renumber),%o3
	ld [%o3+%lo(_reg_renumber)],%o2
	add %o1,%o2,%o1
	lduh [%o1],%o2
	sll %o2,16,%o3
	sra %o3,16,%o1
	cmp %o1,0
	bl L316
	nop
	mov %o0,%o2
	sll %o2,1,%o1
	sethi %hi(_reg_renumber),%o3
	ld [%o3+%lo(_reg_renumber)],%o2
	add %o1,%o2,%o1
	lduh [%o1],%o2
	sll %o2,16,%o1
	sra %o1,16,%o0
L316:
L315:
	xor %l0,%o0,%o2
	subcc %g0,%o2,%g0
	subx %g0,-1,%o1
	mov %o1,%i0
	b L303
	nop
L305:
	ld [%fp+72],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %l1,%o0
	be L317
	nop
	mov 0,%i0
	b L303
	nop
L317:
	add %l1,-23,%o0
	mov 17,%o1
	cmp %o1,%o0
	blu L327
	nop
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(L326),%o2
	or %o2,%lo(L326),%o1
	ld [%o0+%o1],%o0
	jmp %o0
	nop
L326:
	.word	L321
	.word	L321
	.word	L327
	.word	L327
	.word	L327
	.word	L327
	.word	L327
	.word	L323
	.word	L327
	.word	L327
	.word	L319
	.word	L327
	.word	L327
	.word	L327
	.word	L327
	.word	L324
	.word	L325
	.word	L320
L319:
L320:
L321:
L322:
	mov 0,%i0
	b L303
	nop
L323:
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%o0+4],%o0
	ld [%o1+4],%o1
	xor %o0,%o1,%o2
	subcc %g0,%o2,%g0
	subx %g0,-1,%o0
	mov %o0,%i0
	b L303
	nop
L324:
	ld [%fp+68],%o1
	ld [%o1+4],%o0
	call _next_real_insn,0
	nop
	mov %o0,%l3
	ld [%fp+72],%o1
	ld [%o1+4],%o0
	call _next_real_insn,0
	nop
	xor %l3,%o0,%o1
	subcc %g0,%o1,%g0
	subx %g0,-1,%o0
	mov %o0,%i0
	b L303
	nop
L325:
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%o0+4],%o0
	ld [%o1+4],%o1
	xor %o0,%o1,%o2
	subcc %g0,%o2,%g0
	subx %g0,-1,%o0
	mov %o0,%i0
	b L303
	nop
L327:
L318:
	ld [%fp+68],%o1
	ld [%fp+72],%o0
	ldub [%o1+2],%o2
	and %o2,0xff,%o1
	ldub [%o0+2],%o2
	and %o2,0xff,%o0
	cmp %o1,%o0
	be L328
	nop
	mov 0,%i0
	b L303
	nop
L328:
	mov %l1,%o1
	sll %o1,2,%o0
	sethi %hi(_rtx_format),%o2
	or %o2,%lo(_rtx_format),%o1
	ld [%o0+%o1],%l2
	mov %l1,%o1
	sll %o1,2,%o0
	sethi %hi(_rtx_length),%o2
	or %o2,%lo(_rtx_length),%o1
	ld [%o0+%o1],%o0
	add %o0,-1,%l0
L329:
	cmp %l0,0
	bl L330
	nop
	ldub [%l2+%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,101
	be L337
	nop
	cmp %o0,101
	bg L348
	nop
	cmp %o0,48
	be L339
	nop
	cmp %o0,69
	be L340
	nop
	b L346
	nop
L348:
	cmp %o0,105
	be L333
	nop
	cmp %o0,115
	be L335
	nop
	b L346
	nop
L333:
	ld [%fp+68],%o0
	mov %l0,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o0
	ld [%fp+72],%o1
	mov %l0,%o2
	sll %o2,2,%o3
	add %o1,%o3,%o1
	ld [%o0+4],%o0
	ld [%o1+4],%o1
	cmp %o0,%o1
	be L334
	nop
	mov 0,%i0
	b L303
	nop
L334:
	b L332
	nop
L335:
	ld [%fp+68],%o0
	mov %l0,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o0
	ld [%fp+72],%o1
	mov %l0,%o2
	sll %o2,2,%o3
	add %o1,%o3,%o1
	ld [%o0+4],%o0
	ld [%o1+4],%o1
	call _strcmp,0
	nop
	cmp %o0,0
	be L336
	nop
	mov 0,%i0
	b L303
	nop
L336:
	b L332
	nop
L337:
	ld [%fp+68],%o0
	mov %l0,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o0
	ld [%fp+72],%o1
	mov %l0,%o2
	sll %o2,2,%o3
	add %o1,%o3,%o1
	ld [%o0+4],%o0
	ld [%o1+4],%o1
	call _rtx_renumbered_equal_p,0
	nop
	cmp %o0,0
	bne L338
	nop
	mov 0,%i0
	b L303
	nop
L338:
	b L332
	nop
L339:
	b L332
	nop
L340:
	ld [%fp+68],%o0
	mov %l0,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	ld [%fp+72],%o1
	mov %l0,%o2
	sll %o2,2,%o3
	add %o1,%o3,%o2
	ld [%o2+4],%o1
	ld [%o0],%o0
	ld [%o1],%o1
	cmp %o0,%o1
	be L341
	nop
	mov 0,%i0
	b L303
	nop
L341:
	nop
	ld [%fp+68],%o0
	mov %l0,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	ld [%o0],%o1
	add %o1,-1,%l3
L342:
	cmp %l3,0
	bl L343
	nop
	ld [%fp+68],%o0
	mov %l0,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	mov %l3,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o0
	ld [%fp+72],%o1
	mov %l0,%o2
	sll %o2,2,%o3
	add %o1,%o3,%o2
	ld [%o2+4],%o1
	mov %l3,%o2
	sll %o2,2,%o3
	add %o1,%o3,%o1
	ld [%o0+4],%o0
	ld [%o1+4],%o1
	call _rtx_renumbered_equal_p,0
	nop
	cmp %o0,0
	bne L345
	nop
	mov 0,%i0
	b L303
	nop
L345:
L344:
	add %l3,-1,%l3
	b L342
	nop
L343:
	b L332
	nop
L346:
	call _abort,0
	nop
L332:
L331:
	add %l0,-1,%l0
	b L329
	nop
L330:
	mov 1,%i0
	b L303
	nop
L303:
	ret
	restore
	.align 4
	.global _true_regnum
	.proc	04
_true_regnum:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,34
	bne L350
	nop
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	cmp %o1,55
	ble L351
	nop
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	mov %o1,%o2
	sll %o2,1,%o0
	sethi %hi(_reg_renumber),%o2
	ld [%o2+%lo(_reg_renumber)],%o1
	add %o0,%o1,%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	sra %o2,16,%o0
	mov %o0,%i0
	b L349
	nop
L351:
	ld [%fp+68],%o0
	ld [%o0+4],%i0
	b L349
	nop
L350:
	ld [%fp+68],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,35
	bne L352
	nop
	ld [%fp+68],%o1
	ld [%o1+4],%o0
	call _true_regnum,0
	nop
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	cmp %o0,0
	bl L353
	nop
	ld [%fp-20],%o0
	cmp %o0,55
	bg L353
	nop
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	ld [%fp-20],%o2
	add %o1,%o2,%o0
	mov %o0,%i0
	b L349
	nop
L353:
L352:
	mov -1,%i0
	b L349
	nop
L349:
	ret
	restore
	.global _jump_chain
	.common _jump_chain,8,"bss"
