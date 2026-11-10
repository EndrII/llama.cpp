_ZN12_GLOBAL__N_14k_mvILi1ELb0ELb1ELb0ELb0ELb0EEEvPKhS2_PKffS2_S2_S4_fPKaPK15HIP_vector_typeIfLj2EEPfS4_S4_iiiiS4_S4_iiS4_f19ggml_cuda_conv_foldSB_iNS_11mv_jobs_devE: ; @_ZN12_GLOBAL__N_14k_mvILi1ELb0ELb1ELb0ELb0ELb0EEEvPKhS2_PKffS2_S2_S4_fPKaPK15HIP_vector_typeIfLj2EEPfS4_S4_iiiiS4_S4_iiS4_f19ggml_cuda_conv_foldSB_iNS_11mv_jobs_devE
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	v_dual_mov_b32 v24, 0x3c00 :: v_dual_and_b32 v73, 15, v0
	s_clause 0x1
	s_load_b64 s[2:3], s[0:1], 0x10
	s_load_b128 s[24:27], s[0:1], 0x50
	v_bfe_u32 v72, v0, 4, 6
	v_lshl_or_b32 v1, s15, 4, v73
	s_mov_b32 s4, 0x66666666
	s_mov_b32 s6, 0x96969696
	s_mov_b32 s8, 0x99669966
	v_bfrev_b32_e32 v20, 60
	v_ashrrev_i32_e32 v2, 31, v1
	v_bfrev_b32_e32 v21, 61
	;;#ASMSTART
	v_cndmask_b32_e64 v12, v20, v21, s4
	;;#ASMEND
	s_mov_b32 s4, 0xf0f0f0f0
	;;#ASMSTART
	v_cndmask_b32_e64 v13, v20, v21, s6
	;;#ASMEND
	v_lshlrev_b64 v[64:65], 2, v[1:2]
	s_mov_b32 s6, 0xff00ff00
	;;#ASMSTART
	v_cndmask_b32_e64 v14, v20, v21, s8
	;;#ASMEND
	s_mov_b32 s8, 0xff00ff0
	v_dual_mov_b32 v25, 0xbc00 :: v_dual_mov_b32 v74, 0
	s_mov_b32 s5, 0x5a5a5a5a
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v1, vcc_lo, s2, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s3, v65, vcc_lo
	s_mov_b32 s2, 0
	s_mov_b32 s3, 0xaaaaaaaa
	;;#ASMSTART
	v_cndmask_b32_e64 v8, v24, v25, s2
	;;#ASMEND
	global_load_b32 v71, v[1:2], off
	;;#ASMSTART
	v_cndmask_b32_e64 v1, v20, v21, s3
	;;#ASMEND
	s_mov_b32 s2, 0xcccccccc
	;;#ASMSTART
	v_cndmask_b32_e64 v9, v24, v25, s4
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v3, v20, v21, s5
	;;#ASMEND
	s_mov_b32 s4, 0x3c3c3c3c
	s_mov_b32 s7, 0x55aa55aa
	;;#ASMSTART
	v_cndmask_b32_e64 v10, v24, v25, s6
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v5, v20, v21, s7
	;;#ASMEND
	s_mov_b32 s6, 0x33cc33cc
	s_mov_b32 s9, 0xa55aa55a
	;;#ASMSTART
	v_cndmask_b32_e64 v11, v24, v25, s8
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v7, v20, v21, s9
	;;#ASMEND
	s_mov_b32 s8, 0xc33cc33c
	s_mov_b32 s10, 0x69966996
	;;#ASMSTART
	v_cndmask_b32_e64 v15, v20, v21, s10
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v17, v20, v21, s2
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v18, v20, v21, s4
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v19, v20, v21, s6
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v20, v20, v21, s8
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v21, v24, v25, s3
	;;#ASMEND
	v_cmp_eq_u32_e64 s3, 0, v72
	s_cmp_lg_u64 s[26:27], 0
	v_mov_b32_e32 v75, 0
	;;#ASMSTART
	v_cndmask_b32_e64 v2, v24, v25, s2
	;;#ASMEND
	s_cselect_b32 s2, -1, 0
	;;#ASMSTART
	v_cndmask_b32_e64 v22, v24, v25, s5
	;;#ASMEND
	s_and_b32 s5, s3, s2
	;;#ASMSTART
	v_cndmask_b32_e64 v4, v24, v25, s4
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v6, v24, v25, s6
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v16, v24, v25, s8
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v23, v24, v25, s7
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v24, v24, v25, s9
	;;#ASMEND
	s_and_saveexec_b32 s4, s5
	s_cbranch_execz .LBB15_2
; %bb.1:
	v_add_co_u32 v25, vcc_lo, s26, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v26, null, s27, v65, vcc_lo
	global_load_b32 v74, v[25:26], off
.LBB15_2:
	s_or_b32 exec_lo, exec_lo, s4
	s_load_b32 s4, s[0:1], 0x70
	v_bfe_u32 v66, v0, 10, 10
	s_mov_b32 s26, exec_lo
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s5, s4, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshr_b32 s5, s5, 24
	s_add_i32 s4, s4, s5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s27, s4, 8
	v_cmpx_gt_i32_e32 s27, v66
	s_cbranch_execz .LBB15_44
; %bb.3:
	v_dual_mov_b32 v75, 0 :: v_dual_and_b32 v76, 0x3ff, v0
	s_clause 0x1
	s_load_b256 s[16:23], s[0:1], 0x78
	s_load_b32 s28, s[0:1], 0x98
	v_or_b32_e32 v0, v1, v8
	v_or_b32_e32 v1, v12, v2
	v_or_b32_e32 v2, v3, v9
	v_or_b32_e32 v3, v13, v4
	v_or_b32_e32 v4, v5, v10
	v_or_b32_e32 v5, v14, v6
	v_or_b32_e32 v6, v7, v11
	v_or_b32_e32 v7, v15, v16
	v_lshlrev_b32_e32 v16, 6, v76
	v_or_b32_e32 v8, v17, v8
	v_or_b32_e32 v9, v18, v9
	s_load_b128 s[8:11], s[0:1], 0x0
	v_or_b32_e32 v10, v19, v10
	v_and_b32_e32 v18, 0x1c0, v16
	v_mad_u64_u32 v[16:17], null, s27, s15, v[66:67]
	v_or_b32_e32 v11, v20, v11
	v_or_b32_e32 v12, v21, v12
	v_or_b32_e32 v14, v23, v14
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v67, s4, s22, v18
	v_mad_i64_i32 v[20:21], null, s21, v66, 0
	v_lshlrev_b32_e32 v23, 3, v72
	v_ashrrev_i32_e32 v17, 31, v16
	v_or_b32_e32 v13, v22, v13
	v_dual_mov_b32 v91, v66 :: v_dual_lshlrev_b32 v22, 10, v72
	v_or_b32_e32 v15, v24, v15
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b64 v[18:19], 8, v[16:17]
	v_lshlrev_b64 v[16:17], 11, v[16:17]
	v_lshlrev_b32_e32 v24, 7, v72
	v_lshlrev_b32_e32 v78, 8, v66
	v_lshl_add_u32 v79, v66, 6, 0xf00
	v_lshlrev_b32_e32 v25, 5, v72
	v_add_co_u32 v23, vcc_lo, v18, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v26, null, 0, v19, vcc_lo
	v_lshlrev_b64 v[18:19], 2, v[20:21]
	v_add_co_u32 v80, vcc_lo, s10, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_ci_u32_e64 v81, null, s11, v26, vcc_lo
	v_and_b32_e32 v27, 15, v76
	v_add_co_u32 v83, vcc_lo, s18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v84, null, s19, v19, vcc_lo
	v_add_co_u32 v16, vcc_lo, v16, v22
	v_lshlrev_b32_e32 v18, 10, v66
	v_add_co_ci_u32_e64 v17, null, 0, v17, vcc_lo
	s_cmp_lg_u64 s[18:19], 0
	v_add_co_u32 v85, vcc_lo, s8, v16
	v_add_co_u32 v87, s6, s16, v18
	s_cselect_b32 s29, -1, 0
	s_cmp_eq_u32 s20, 3
	v_lshlrev_b32_e32 v77, 4, v73
	v_add_co_ci_u32_e64 v68, null, s23, 0, s4
	v_cmp_gt_u32_e64 s4, 16, v76
	v_cmp_eq_u32_e64 s5, 0, v76
	v_lshlrev_b32_e32 v82, 6, v27
	v_add_co_ci_u32_e64 v86, null, s9, v17, vcc_lo
	v_add_co_ci_u32_e64 v88, null, s17, 0, s6
	v_add_nc_u32_e32 v89, v78, v24
	v_add_nc_u32_e32 v90, v79, v25
	s_cselect_b32 s30, -1, 0
	s_cmp_lg_u32 s20, 2
	s_mov_b32 s12, 0
	s_cselect_b32 s20, -1, 0
	s_mul_hi_i32 s10, s21, 40
	s_mul_i32 s11, s21, 40
	s_mov_b32 s9, 0x3e1e3e1e
	s_mov_b32 s21, 0x87c387c3
	s_mov_b32 s22, 0xf078f078
	s_mov_b32 s23, 0x1f0f1f0f
	s_mov_b32 s31, 0xc1e0c1e0
	s_mov_b32 s33, 0x7c3e7c3e
	s_mov_b32 s34, 0x7830783
	s_mov_b32 s35, 0xf0f8f0f8
	s_mov_b32 s36, 0x1e0f1e0f
	s_mov_b32 s37, 0x83c183c1
	s_mov_b32 s38, 0xf87cf87c
	s_mov_b32 s39, 0xf070f07
	s_mov_b32 s40, 0xe1f0e1f0
	s_mov_b32 s41, 0x3c1e3c1e
	s_mov_b32 s13, s12
	s_mov_b32 s14, s12
	s_mov_b32 s15, s12
	s_mov_b32 s16, s12
	s_mov_b32 s17, s12
	s_mov_b32 s18, s12
	s_mov_b32 s19, s12
	s_mov_b32 s43, 0x76543210
	s_mov_b32 s42, s12
	s_branch .LBB15_5
.LBB15_4:                               ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s6
	; wave barrier
	ds_load_b128 v[32:35], v89
	ds_load_b128 v[36:39], v89 offset:16
	ds_load_b128 v[40:43], v89 offset:32
	ds_load_b128 v[44:47], v89 offset:48
	ds_load_b128 v[48:51], v89 offset:64
	ds_load_b128 v[52:55], v89 offset:80
	ds_load_b128 v[56:59], v89 offset:96
	ds_load_b128 v[60:63], v89 offset:112
	v_and_b32_e32 v92, 0xf0f0f0f, v28
	v_lshrrev_b32_e32 v28, 4, v28
	v_and_b32_e32 v93, 0xf0f0f0f, v29
	v_lshrrev_b32_e32 v29, 4, v29
	v_and_b32_e32 v94, 0xf0f0f0f, v30
	v_xor_b32_e32 v92, 0x8080808, v92
	v_and_b32_e32 v28, 0xf0f0f0f, v28
	v_xor_b32_e32 v93, 0x8080808, v93
	v_and_b32_e32 v29, 0xf0f0f0f, v29
	v_lshrrev_b32_e32 v30, 4, v30
	v_and_b32_e32 v95, 0xf0f0f0f, v31
	v_xor_b32_e32 v28, 0x8080808, v28
	s_waitcnt lgkmcnt(7)
	v_dot4_i32_iu8 v32, v92, v32, 0 neg_lo:[1,1,0]
	v_and_b32_e32 v92, 0xf0f0f0f, v24
	v_lshrrev_b32_e32 v24, 4, v24
	v_xor_b32_e32 v29, 0x8080808, v29
	v_and_b32_e32 v30, 0xf0f0f0f, v30
	v_dot4_i32_iu8 v28, v28, v33, v32 neg_lo:[1,1,0]
	v_and_b32_e32 v32, 0xf0f0f0f, v25
	v_and_b32_e32 v24, 0xf0f0f0f, v24
	v_lshrrev_b32_e32 v25, 4, v25
	v_xor_b32_e32 v30, 0x8080808, v30
	v_dot4_i32_iu8 v28, v93, v34, v28 neg_lo:[1,1,0]
	v_xor_b32_e32 v34, 0x8080808, v92
	v_xor_b32_e32 v92, 0x8080808, v94
	v_xor_b32_e32 v24, 0x8080808, v24
	v_xor_b32_e32 v32, 0x8080808, v32
	v_dot4_i32_iu8 v28, v29, v35, v28 neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(5)
	v_dot4_i32_iu8 v29, v34, v40, 0 neg_lo:[1,1,0]
	v_and_b32_e32 v25, 0xf0f0f0f, v25
	v_and_b32_e32 v33, 0xf0f0f0f, v26
	v_lshrrev_b32_e32 v26, 4, v26
	v_dot4_i32_iu8 v28, v92, v36, v28 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v24, v24, v41, v29 neg_lo:[1,1,0]
	v_xor_b32_e32 v29, 0x8080808, v95
	v_xor_b32_e32 v25, 0x8080808, v25
	v_and_b32_e32 v35, 0xf0f0f0f, v23
	v_dot4_i32_iu8 v28, v30, v37, v28 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v24, v32, v42, v24 neg_lo:[1,1,0]
	v_lshrrev_b32_e32 v36, 4, v23
	v_lshrrev_b32_e32 v31, 4, v31
	v_add_co_u32 v80, vcc_lo, 0xa00, v80
	v_dot4_i32_iu8 v28, v29, v38, v28 neg_lo:[1,1,0]
	v_xor_b32_e32 v29, 0x8080808, v33
	v_dot4_i32_iu8 v24, v25, v43, v24 neg_lo:[1,1,0]
	v_and_b32_e32 v25, 0xf0f0f0f, v26
	v_and_b32_e32 v26, 0xf0f0f0f, v16
	v_lshrrev_b32_e32 v16, 4, v16
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_waitcnt lgkmcnt(4)
	v_dot4_i32_iu8 v24, v29, v44, v24 neg_lo:[1,1,0]
	v_xor_b32_e32 v25, 0x8080808, v25
	v_xor_b32_e32 v26, 0x8080808, v26
	v_and_b32_e32 v16, 0xf0f0f0f, v16
	v_and_b32_e32 v29, 0xf0f0f0f, v27
	v_xor_b32_e32 v30, 0x8080808, v31
	v_dot4_i32_iu8 v32, v25, v45, v24 neg_lo:[1,1,0]
	v_and_b32_e32 v25, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v20, 4, v20
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_iu8 v24, v26, v48, 0 neg_lo:[1,1,0]
	v_xor_b32_e32 v16, 0x8080808, v16
	v_and_b32_e32 v26, 0xf0f0f0f, v17
	v_lshrrev_b32_e32 v17, 4, v17
	v_xor_b32_e32 v25, 0x8080808, v25
	v_and_b32_e32 v20, 0xf0f0f0f, v20
	v_dot4_i32_iu8 v16, v16, v49, v24 neg_lo:[1,1,0]
	v_xor_b32_e32 v24, 0x8080808, v26
	v_and_b32_e32 v17, 0xf0f0f0f, v17
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_iu8 v25, v25, v56, 0 neg_lo:[1,1,0]
	v_xor_b32_e32 v20, 0x8080808, v20
	v_and_b32_e32 v26, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v21, 4, v21
	v_dot4_i32_iu8 v16, v24, v50, v16 neg_lo:[1,1,0]
	v_xor_b32_e32 v17, 0x8080808, v17
	v_dot4_i32_iu8 v20, v20, v57, v25 neg_lo:[1,1,0]
	v_xor_b32_e32 v24, 0x8080808, v26
	v_and_b32_e32 v21, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v25, 4, v27
	v_dot4_i32_iu8 v16, v17, v51, v16 neg_lo:[1,1,0]
	v_and_b32_e32 v17, 0xf0f0f0f, v18
	v_dot4_i32_iu8 v20, v24, v58, v20 neg_lo:[1,1,0]
	v_xor_b32_e32 v21, 0x8080808, v21
	v_and_b32_e32 v24, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v22, 4, v22
	v_xor_b32_e32 v17, 0x8080808, v17
	v_lshrrev_b32_e32 v18, 4, v18
	v_dot4_i32_iu8 v20, v21, v59, v20 neg_lo:[1,1,0]
	v_xor_b32_e32 v21, 0x8080808, v24
	v_and_b32_e32 v22, 0xf0f0f0f, v22
	v_and_b32_e32 v33, 0xf0f0f0f, v25
	v_dot4_i32_iu8 v16, v17, v52, v16 neg_lo:[1,1,0]
	v_and_b32_e32 v17, 0xf0f0f0f, v18
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_iu8 v18, v21, v60, v20 neg_lo:[1,1,0]
	v_xor_b32_e32 v34, 0x8080808, v22
	ds_load_b128 v[20:23], v90
	ds_load_b128 v[24:27], v90 offset:16
	v_xor_b32_e32 v29, 0x8080808, v29
	v_xor_b32_e32 v17, 0x8080808, v17
	v_dot4_i32_iu8 v28, v30, v39, v28 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v18, v34, v61, v18 neg_lo:[1,1,0]
	v_xor_b32_e32 v34, 0x8080808, v35
	v_and_b32_e32 v35, 0xf0f0f0f, v36
	v_and_b32_e32 v36, 0xf0f0f0f, v19
	v_lshrrev_b32_e32 v19, 4, v19
	v_dot4_i32_iu8 v16, v17, v53, v16 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v17, v34, v62, v18 neg_lo:[1,1,0]
	v_xor_b32_e32 v18, 0x8080808, v35
	v_xor_b32_e32 v34, 0x8080808, v36
	v_and_b32_e32 v19, 0xf0f0f0f, v19
	v_dot4_i32_iu8 v29, v29, v46, v32 neg_lo:[1,1,0]
	v_xor_b32_e32 v32, 0x8080808, v33
	v_dot4_i32_iu8 v17, v18, v63, v17 neg_lo:[1,1,0]
	v_lshrrev_b32_e32 v30, 16, v70
	v_dot4_i32_iu8 v16, v34, v54, v16 neg_lo:[1,1,0]
	v_xor_b32_e32 v18, 0x8080808, v19
	v_dot4_i32_iu8 v29, v32, v47, v29 neg_lo:[1,1,0]
	v_cvt_f32_i32_e32 v17, v17
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v27, 0xc1000000, v27
	v_lshrrev_b32_e32 v19, 16, v69
	v_dot4_i32_iu8 v16, v18, v55, v16 neg_lo:[1,1,0]
	v_cvt_f32_u32_e32 v18, v30
	v_cvt_f32_i32_e32 v29, v29
	v_mul_f32_e32 v23, 0xc1000000, v23
	v_fmac_f32_e32 v27, v26, v17
	v_cvt_f32_u32_e32 v17, v19
	v_cvt_f32_i32_e32 v16, v16
	v_mul_f32_e32 v19, 0xc1000000, v25
	v_dual_fmac_f32 v23, v22, v29 :: v_dual_and_b32 v96, 0xffff, v70
	v_fmac_f32_e32 v75, v27, v18
	v_and_b32_e32 v31, 0xffff, v69
	v_cvt_f32_i32_e32 v22, v28
	v_mul_f32_e32 v21, 0xc1000000, v21
	v_fmac_f32_e32 v19, v24, v16
	v_cvt_f32_u32_e32 v18, v96
	v_fmac_f32_e32 v75, v23, v17
	v_add_nc_u32_e32 v91, 10, v91
	v_add_co_ci_u32_e64 v81, null, 0, v81, vcc_lo
	v_add_co_u32 v83, vcc_lo, v83, s11
	v_cvt_f32_u32_e32 v16, v31
	v_fmac_f32_e32 v21, v20, v22
	v_fmac_f32_e32 v75, v19, v18
	v_add_co_ci_u32_e64 v84, null, s10, v84, vcc_lo
	v_add_co_u32 v85, vcc_lo, 0x5000, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v86, null, 0, v86, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, s27, v91
	v_add_co_u32 v87, s6, 0x2800, v87
	v_fmac_f32_e32 v75, v21, v16
	v_add_co_ci_u32_e64 v88, null, 0, v88, s6
	s_or_b32 s42, vcc_lo, s42
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s42
	s_cbranch_execz .LBB15_43
.LBB15_5:                               ; =>This Inner Loop Header: Depth=1
	v_add_co_u32 v32, vcc_lo, v87, v82
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v33, null, 0, v88, vcc_lo
	v_add_co_u32 v20, vcc_lo, v85, v77
	v_add_co_ci_u32_e64 v21, null, 0, v86, vcc_lo
	v_add_co_u32 v34, vcc_lo, v80, v77
	s_clause 0x2
	global_load_b128 v[56:59], v[32:33], off
	global_load_b128 v[48:51], v[32:33], off offset:16
	global_load_b128 v[40:43], v[32:33], off offset:32
	v_add_co_ci_u32_e64 v35, null, 0, v81, vcc_lo
	s_clause 0x3
	global_load_b128 v[28:31], v[20:21], off
	global_load_b128 v[24:27], v[20:21], off offset:256
	global_load_b128 v[16:19], v[20:21], off offset:512
	global_load_b128 v[20:23], v[20:21], off offset:768
	global_load_b64 v[69:70], v[34:35], off
	global_load_b128 v[32:35], v[32:33], off offset:48
	s_and_b32 s6, s29, exec_lo
	s_cselect_b32 s6, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s6, 1
	s_waitcnt vmcnt(8)
	v_dual_mov_b32 v92, v56 :: v_dual_mov_b32 v93, v57
	v_dual_mov_b32 v94, v58 :: v_dual_mov_b32 v95, v59
	s_waitcnt vmcnt(7)
	v_dual_mov_b32 v60, v48 :: v_dual_mov_b32 v61, v49
	v_dual_mov_b32 v62, v50 :: v_dual_mov_b32 v63, v51
	s_waitcnt vmcnt(6)
	v_dual_mov_b32 v52, v40 :: v_dual_mov_b32 v53, v41
	v_dual_mov_b32 v54, v42 :: v_dual_mov_b32 v55, v43
	s_cbranch_scc1 .LBB15_25
; %bb.6:                                ;   in Loop: Header=BB15_5 Depth=1
	v_add_co_u32 v36, vcc_lo, v83, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v37, null, 0, v84, vcc_lo
	s_and_b32 s6, s30, exec_lo
	s_cselect_b32 s6, 1, 0
	s_clause 0x3
	global_load_b128 v[60:63], v[36:37], off
	global_load_b128 v[52:55], v[36:37], off offset:16
	global_load_b128 v[44:47], v[36:37], off offset:32
	global_load_b128 v[36:39], v[36:37], off offset:48
	s_cmp_lg_u32 s6, 1
	s_cbranch_scc1 .LBB15_8
; %bb.7:                                ;   in Loop: Header=BB15_5 Depth=1
	s_clause 0x3
	global_load_b128 v[92:95], v[67:68], off
	global_load_b128 v[96:99], v[67:68], off offset:16
	global_load_b128 v[100:103], v[67:68], off offset:32
	global_load_b128 v[104:107], v[67:68], off offset:48
	s_waitcnt vmcnt(7)
	v_mul_f32_e32 v108, v60, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v108, v61, v61
	v_fmac_f32_e32 v108, v62, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v108, v63, v63
	s_waitcnt vmcnt(6)
	v_fmac_f32_e32 v108, v52, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v108, v53, v53
	v_fmac_f32_e32 v108, v54, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v108, v55, v55
	s_waitcnt vmcnt(5)
	v_fmac_f32_e32 v108, v44, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v108, v45, v45
	v_fmac_f32_e32 v108, v46, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v108, v47, v47
	s_waitcnt vmcnt(4)
	v_fmac_f32_e32 v108, v36, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v108, v37, v37
	v_fmac_f32_e32 v108, v38, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v108, v39, v39
	v_add_f32_dpp v108, v108, v108 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_dpp v108, v108, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v108, v108, v108 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v108, 0x3c000000, v108, s28
	v_mul_f32_e32 v109, 0x4b800000, v108
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v108
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v108, v108, v109, vcc_lo
	v_rsq_f32_e32 v108, v108
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v109, 0x45800000, v108
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v108, v108, v109, vcc_lo
	v_mul_f32_e32 v44, v108, v44
	v_mul_f32_e32 v45, v108, v45
	v_mul_f32_e32 v46, v108, v46
	v_mul_f32_e32 v47, v108, v47
	v_mul_f32_e32 v36, v108, v36
	v_mul_f32_e32 v37, v108, v37
	v_mul_f32_e32 v38, v108, v38
	v_mul_f32_e32 v39, v108, v39
	s_waitcnt vmcnt(3)
	v_dual_mul_f32 v60, v92, v60 :: v_dual_mul_f32 v61, v93, v61
	v_dual_mul_f32 v62, v94, v62 :: v_dual_mul_f32 v63, v95, v63
	s_waitcnt vmcnt(2)
	v_dual_mul_f32 v52, v96, v52 :: v_dual_mul_f32 v53, v97, v53
	v_dual_mul_f32 v54, v98, v54 :: v_dual_mul_f32 v55, v99, v55
	s_waitcnt vmcnt(1)
	v_dual_mul_f32 v44, v44, v100 :: v_dual_mul_f32 v45, v45, v101
	v_dual_mul_f32 v46, v46, v102 :: v_dual_mul_f32 v47, v47, v103
	s_waitcnt vmcnt(0)
	v_dual_mul_f32 v36, v36, v104 :: v_dual_mul_f32 v37, v37, v105
	v_mul_f32_e32 v38, v38, v106
	v_mul_f32_e32 v60, v60, v108
	v_mul_f32_e32 v61, v61, v108
	v_mul_f32_e32 v62, v62, v108
	v_mul_f32_e32 v63, v63, v108
	v_mul_f32_e32 v52, v52, v108
	v_mul_f32_e32 v53, v53, v108
	v_mul_f32_e32 v54, v54, v108
	v_mul_f32_e32 v55, v55, v108
	v_mul_f32_e32 v39, v39, v107
.LBB15_8:                               ;   in Loop: Header=BB15_5 Depth=1
	s_and_b32 vcc_lo, exec_lo, s20
	s_mov_b32 s6, -1
                                        ; implicit-def: $vgpr92
                                        ; implicit-def: $vgpr93
                                        ; implicit-def: $vgpr94
                                        ; implicit-def: $vgpr95
	s_cbranch_vccz .LBB15_10
; %bb.9:                                ;   in Loop: Header=BB15_5 Depth=1
	v_dual_sub_f32 v92, 0x42800000, v56 :: v_dual_sub_f32 v93, 0x42800000, v57
	v_dual_sub_f32 v94, 0x42800000, v58 :: v_dual_sub_f32 v95, 0x42800000, v59
	v_cmp_lt_f32_e32 vcc_lo, 0x42aeac50, v56
	v_cmp_lt_f32_e64 s6, 0x42aeac50, v57
	v_cmp_lt_f32_e64 s7, 0x42aeac50, v58
	v_cmp_lt_f32_e64 s8, 0x42aeac50, v59
	v_cndmask_b32_e64 v92, -v56, v92, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v93, -v57, v93, s6
	v_cndmask_b32_e64 v94, -v58, v94, s7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v95, -v59, v95, s8
	v_dual_mul_f32 v92, 0x3fb8aa3b, v92 :: v_dual_mul_f32 v93, 0x3fb8aa3b, v93
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v94, 0x3fb8aa3b, v94 :: v_dual_mul_f32 v95, 0x3fb8aa3b, v95
	v_exp_f32_e32 v92, v92
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v93, v93
	v_exp_f32_e32 v94, v94
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_3)
	v_exp_f32_e32 v95, v95
	v_dual_mul_f32 v96, 0x114b4ea4, v92 :: v_dual_mul_f32 v97, 0x114b4ea4, v93
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v98, 0x114b4ea4, v94 :: v_dual_mul_f32 v99, 0x114b4ea4, v95
	v_cndmask_b32_e64 v93, v93, v97, s6
	v_cndmask_b32_e32 v92, v92, v96, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v94, v94, v98, s7
	v_cndmask_b32_e64 v95, v95, v99, s8
	s_waitcnt vmcnt(3)
	v_mul_f32_e32 v97, v61, v57
	v_dual_add_f32 v93, 1.0, v93 :: v_dual_add_f32 v92, 1.0, v92
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_add_f32 v94, 1.0, v94 :: v_dual_add_f32 v95, 1.0, v95
	v_dual_mul_f32 v96, v60, v56 :: v_dual_mul_f32 v99, v63, v59
	v_rcp_f32_e32 v92, v92
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v93, v93
	v_rcp_f32_e32 v94, v94
	v_rcp_f32_e32 v95, v95
	v_mul_f32_e32 v98, v62, v58
	s_mov_b32 s6, 0
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_dual_mul_f32 v92, v96, v92 :: v_dual_mul_f32 v93, v97, v93
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v94, v98, v94 :: v_dual_mul_f32 v95, v99, v95
	s_and_b32 s6, s6, exec_lo
	s_cselect_b32 s6, 1, 0
	s_cmp_lg_u32 s6, 1
	s_cbranch_scc0 .LBB15_11
	s_branch .LBB15_12
.LBB15_10:                              ;   in Loop: Header=BB15_5 Depth=1
	s_and_b32 s6, s6, exec_lo
	s_cselect_b32 s6, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s6, 1
	s_cbranch_scc1 .LBB15_12
.LBB15_11:                              ;   in Loop: Header=BB15_5 Depth=1
	s_waitcnt vmcnt(3)
	v_dual_sub_f32 v92, 0x42800000, v60 :: v_dual_sub_f32 v93, 0x42800000, v61
	v_cmp_lt_f32_e32 vcc_lo, 0x42aeac50, v60
	v_cmp_lt_f32_e64 s6, 0x42aeac50, v61
	v_dual_sub_f32 v94, 0x42800000, v62 :: v_dual_sub_f32 v95, 0x42800000, v63
	v_cmp_lt_f32_e64 s7, 0x42aeac50, v62
	v_cndmask_b32_e64 v60, -v60, v92, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v61, -v61, v93, s6
	v_cmp_lt_f32_e64 s8, 0x42aeac50, v63
	v_cndmask_b32_e64 v62, -v62, v94, s7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v60, 0x3fb8aa3b, v60 :: v_dual_mul_f32 v61, 0x3fb8aa3b, v61
	v_cndmask_b32_e64 v63, -v63, v95, s8
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v62, 0x3fb8aa3b, v62
	v_exp_f32_e32 v60, v60
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v61, v61
	v_mul_f32_e32 v63, 0x3fb8aa3b, v63
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v62, v62
	v_exp_f32_e32 v63, v63
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v92, 0x114b4ea4, v60 :: v_dual_mul_f32 v93, 0x114b4ea4, v61
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v94, 0x114b4ea4, v62
	v_cndmask_b32_e64 v61, v61, v93, s6
	v_dual_mul_f32 v95, 0x114b4ea4, v63 :: v_dual_cndmask_b32 v60, v60, v92
	v_cndmask_b32_e64 v62, v62, v94, s7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v61, 1.0, v61
	v_cndmask_b32_e64 v63, v63, v95, s8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v60, 1.0, v60
	v_rcp_f32_e32 v61, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v62, 1.0, v62 :: v_dual_add_f32 v63, 1.0, v63
	v_rcp_f32_e32 v60, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_3)
	v_rcp_f32_e32 v63, v63
	v_mul_f32_e32 v93, v57, v61
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v62, v62
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v92, v56, v60 :: v_dual_mul_f32 v95, v59, v63
	v_mul_f32_e32 v94, v58, v62
.LBB15_12:                              ;   in Loop: Header=BB15_5 Depth=1
	s_and_b32 s6, exec_lo, s20
	s_cselect_b32 s6, 1, 0
                                        ; implicit-def: $vgpr60
                                        ; implicit-def: $vgpr61
                                        ; implicit-def: $vgpr62
                                        ; implicit-def: $vgpr63
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s6, 1
	s_mov_b32 s6, -1
	s_cbranch_scc1 .LBB15_14
; %bb.13:                               ;   in Loop: Header=BB15_5 Depth=1
	v_dual_sub_f32 v56, 0x42800000, v48 :: v_dual_sub_f32 v57, 0x42800000, v49
	v_dual_sub_f32 v58, 0x42800000, v50 :: v_dual_sub_f32 v59, 0x42800000, v51
	v_cmp_lt_f32_e32 vcc_lo, 0x42aeac50, v48
	v_cmp_lt_f32_e64 s6, 0x42aeac50, v49
	v_cmp_lt_f32_e64 s7, 0x42aeac50, v50
	v_cmp_lt_f32_e64 s8, 0x42aeac50, v51
	v_cndmask_b32_e64 v56, -v48, v56, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v57, -v49, v57, s6
	v_cndmask_b32_e64 v58, -v50, v58, s7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v59, -v51, v59, s8
	v_dual_mul_f32 v56, 0x3fb8aa3b, v56 :: v_dual_mul_f32 v57, 0x3fb8aa3b, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v58, 0x3fb8aa3b, v58 :: v_dual_mul_f32 v59, 0x3fb8aa3b, v59
	v_exp_f32_e32 v56, v56
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v57, v57
	v_exp_f32_e32 v58, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_3)
	v_exp_f32_e32 v59, v59
	s_waitcnt vmcnt(3)
	v_dual_mul_f32 v60, 0x114b4ea4, v56 :: v_dual_mul_f32 v61, 0x114b4ea4, v57
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v62, 0x114b4ea4, v58 :: v_dual_mul_f32 v63, 0x114b4ea4, v59
	v_cndmask_b32_e64 v57, v57, v61, s6
	v_cndmask_b32_e32 v56, v56, v60, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v58, v58, v62, s7
	v_cndmask_b32_e64 v59, v59, v63, s8
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v61, v53, v49
	v_dual_add_f32 v57, 1.0, v57 :: v_dual_add_f32 v56, 1.0, v56
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_add_f32 v58, 1.0, v58 :: v_dual_add_f32 v59, 1.0, v59
	v_dual_mul_f32 v60, v52, v48 :: v_dual_mul_f32 v63, v55, v51
	v_rcp_f32_e32 v56, v56
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v57, v57
	v_rcp_f32_e32 v58, v58
	v_rcp_f32_e32 v59, v59
	v_mul_f32_e32 v62, v54, v50
	s_mov_b32 s6, 0
	s_delay_alu instid0(TRANS32_DEP_3)
	v_dual_mul_f32 v60, v60, v56 :: v_dual_mul_f32 v61, v61, v57
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v62, v62, v58 :: v_dual_mul_f32 v63, v63, v59
.LBB15_14:                              ;   in Loop: Header=BB15_5 Depth=1
	s_and_b32 s6, s6, exec_lo
	s_cselect_b32 s6, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s6, 1
	s_cbranch_scc1 .LBB15_16
; %bb.15:                               ;   in Loop: Header=BB15_5 Depth=1
	s_waitcnt vmcnt(2)
	v_dual_sub_f32 v56, 0x42800000, v52 :: v_dual_sub_f32 v57, 0x42800000, v53
	v_cmp_lt_f32_e32 vcc_lo, 0x42aeac50, v52
	v_cmp_lt_f32_e64 s6, 0x42aeac50, v53
	v_dual_sub_f32 v58, 0x42800000, v54 :: v_dual_sub_f32 v59, 0x42800000, v55
	v_cmp_lt_f32_e64 s7, 0x42aeac50, v54
	v_cndmask_b32_e64 v52, -v52, v56, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v53, -v53, v57, s6
	v_cmp_lt_f32_e64 s8, 0x42aeac50, v55
	v_cndmask_b32_e64 v54, -v54, v58, s7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v52, 0x3fb8aa3b, v52 :: v_dual_mul_f32 v53, 0x3fb8aa3b, v53
	v_cndmask_b32_e64 v55, -v55, v59, s8
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v54, 0x3fb8aa3b, v54
	v_exp_f32_e32 v52, v52
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v53, v53
	v_mul_f32_e32 v55, 0x3fb8aa3b, v55
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v56, 0x114b4ea4, v52 :: v_dual_mul_f32 v57, 0x114b4ea4, v53
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v58, 0x114b4ea4, v54
	v_cndmask_b32_e64 v53, v53, v57, s6
	v_dual_mul_f32 v59, 0x114b4ea4, v55 :: v_dual_cndmask_b32 v52, v52, v56
	v_cndmask_b32_e64 v54, v54, v58, s7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v53, 1.0, v53
	v_cndmask_b32_e64 v55, v55, v59, s8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v52, 1.0, v52
	v_rcp_f32_e32 v53, v53
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v54, 1.0, v54 :: v_dual_add_f32 v55, 1.0, v55
	v_rcp_f32_e32 v52, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_3)
	v_rcp_f32_e32 v55, v55
	v_mul_f32_e32 v61, v49, v53
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v54, v54
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v60, v48, v52 :: v_dual_mul_f32 v63, v51, v55
	v_mul_f32_e32 v62, v50, v54
.LBB15_16:                              ;   in Loop: Header=BB15_5 Depth=1
	s_and_b32 s6, exec_lo, s20
	s_cselect_b32 s6, 1, 0
                                        ; implicit-def: $vgpr52
                                        ; implicit-def: $vgpr53
                                        ; implicit-def: $vgpr54
                                        ; implicit-def: $vgpr55
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s6, 1
	s_mov_b32 s6, -1
	s_cbranch_scc1 .LBB15_18
; %bb.17:                               ;   in Loop: Header=BB15_5 Depth=1
	v_dual_sub_f32 v48, 0x42800000, v40 :: v_dual_sub_f32 v49, 0x42800000, v41
	v_dual_sub_f32 v50, 0x42800000, v42 :: v_dual_sub_f32 v51, 0x42800000, v43
	v_cmp_lt_f32_e32 vcc_lo, 0x42aeac50, v40
	v_cmp_lt_f32_e64 s6, 0x42aeac50, v41
	v_cmp_lt_f32_e64 s7, 0x42aeac50, v42
	v_cmp_lt_f32_e64 s8, 0x42aeac50, v43
	v_cndmask_b32_e64 v48, -v40, v48, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v49, -v41, v49, s6
	v_cndmask_b32_e64 v50, -v42, v50, s7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v51, -v43, v51, s8
	v_dual_mul_f32 v48, 0x3fb8aa3b, v48 :: v_dual_mul_f32 v49, 0x3fb8aa3b, v49
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v50, 0x3fb8aa3b, v50 :: v_dual_mul_f32 v51, 0x3fb8aa3b, v51
	v_exp_f32_e32 v48, v48
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v49, v49
	v_exp_f32_e32 v50, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_3)
	v_exp_f32_e32 v51, v51
	s_waitcnt vmcnt(2)
	v_dual_mul_f32 v52, 0x114b4ea4, v48 :: v_dual_mul_f32 v53, 0x114b4ea4, v49
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v54, 0x114b4ea4, v50 :: v_dual_mul_f32 v55, 0x114b4ea4, v51
	v_cndmask_b32_e64 v49, v49, v53, s6
	v_cndmask_b32_e32 v48, v48, v52, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v50, v50, v54, s7
	v_cndmask_b32_e64 v51, v51, v55, s8
	s_waitcnt vmcnt(1)
	v_mul_f32_e32 v53, v45, v41
	v_dual_add_f32 v49, 1.0, v49 :: v_dual_add_f32 v48, 1.0, v48
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_add_f32 v50, 1.0, v50 :: v_dual_add_f32 v51, 1.0, v51
	v_dual_mul_f32 v52, v44, v40 :: v_dual_mul_f32 v55, v47, v43
	v_rcp_f32_e32 v48, v48
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v49, v49
	v_rcp_f32_e32 v50, v50
	v_rcp_f32_e32 v51, v51
	v_mul_f32_e32 v54, v46, v42
	s_mov_b32 s6, 0
	s_delay_alu instid0(TRANS32_DEP_3)
	v_dual_mul_f32 v52, v52, v48 :: v_dual_mul_f32 v53, v53, v49
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v54, v54, v50 :: v_dual_mul_f32 v55, v55, v51
.LBB15_18:                              ;   in Loop: Header=BB15_5 Depth=1
	s_and_b32 s6, s6, exec_lo
	s_cselect_b32 s6, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s6, 1
	s_cbranch_scc1 .LBB15_20
; %bb.19:                               ;   in Loop: Header=BB15_5 Depth=1
	s_waitcnt vmcnt(1)
	v_dual_sub_f32 v48, 0x42800000, v44 :: v_dual_sub_f32 v49, 0x42800000, v45
	v_cmp_lt_f32_e32 vcc_lo, 0x42aeac50, v44
	v_cmp_lt_f32_e64 s6, 0x42aeac50, v45
	v_dual_sub_f32 v50, 0x42800000, v46 :: v_dual_sub_f32 v51, 0x42800000, v47
	v_cmp_lt_f32_e64 s7, 0x42aeac50, v46
	v_cndmask_b32_e64 v44, -v44, v48, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v45, -v45, v49, s6
	v_cmp_lt_f32_e64 s8, 0x42aeac50, v47
	v_cndmask_b32_e64 v46, -v46, v50, s7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v44, 0x3fb8aa3b, v44 :: v_dual_mul_f32 v45, 0x3fb8aa3b, v45
	v_cndmask_b32_e64 v47, -v47, v51, s8
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v46, 0x3fb8aa3b, v46
	v_exp_f32_e32 v44, v44
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v45, v45
	v_mul_f32_e32 v47, 0x3fb8aa3b, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v46, v46
	v_exp_f32_e32 v47, v47
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v48, 0x114b4ea4, v44 :: v_dual_mul_f32 v49, 0x114b4ea4, v45
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v50, 0x114b4ea4, v46
	v_cndmask_b32_e64 v45, v45, v49, s6
	v_dual_mul_f32 v51, 0x114b4ea4, v47 :: v_dual_cndmask_b32 v44, v44, v48
	v_cndmask_b32_e64 v46, v46, v50, s7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v45, 1.0, v45
	v_cndmask_b32_e64 v47, v47, v51, s8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v44, 1.0, v44
	v_rcp_f32_e32 v45, v45
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v46, 1.0, v46 :: v_dual_add_f32 v47, 1.0, v47
	v_rcp_f32_e32 v44, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_3)
	v_rcp_f32_e32 v47, v47
	v_mul_f32_e32 v53, v41, v45
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v46, v46
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v52, v40, v44 :: v_dual_mul_f32 v55, v43, v47
	v_mul_f32_e32 v54, v42, v46
.LBB15_20:                              ;   in Loop: Header=BB15_5 Depth=1
	s_and_b32 s6, exec_lo, s20
	s_cselect_b32 s6, 1, 0
                                        ; implicit-def: $vgpr40
                                        ; implicit-def: $vgpr41
                                        ; implicit-def: $vgpr42
                                        ; implicit-def: $vgpr43
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s6, 1
	s_mov_b32 s6, -1
	s_cbranch_scc1 .LBB15_22
; %bb.21:                               ;   in Loop: Header=BB15_5 Depth=1
	s_waitcnt vmcnt(4)
	v_dual_sub_f32 v40, 0x42800000, v32 :: v_dual_sub_f32 v41, 0x42800000, v33
	v_dual_sub_f32 v42, 0x42800000, v34 :: v_dual_sub_f32 v43, 0x42800000, v35
	v_cmp_lt_f32_e32 vcc_lo, 0x42aeac50, v32
	v_cmp_lt_f32_e64 s6, 0x42aeac50, v33
	v_cmp_lt_f32_e64 s7, 0x42aeac50, v34
	v_cmp_lt_f32_e64 s8, 0x42aeac50, v35
	v_cndmask_b32_e64 v40, -v32, v40, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v41, -v33, v41, s6
	v_cndmask_b32_e64 v42, -v34, v42, s7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v43, -v35, v43, s8
	v_dual_mul_f32 v40, 0x3fb8aa3b, v40 :: v_dual_mul_f32 v41, 0x3fb8aa3b, v41
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v42, 0x3fb8aa3b, v42 :: v_dual_mul_f32 v43, 0x3fb8aa3b, v43
	v_exp_f32_e32 v40, v40
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v41, v41
	v_exp_f32_e32 v42, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_3)
	v_exp_f32_e32 v43, v43
	s_waitcnt vmcnt(1)
	v_dual_mul_f32 v44, 0x114b4ea4, v40 :: v_dual_mul_f32 v45, 0x114b4ea4, v41
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v46, 0x114b4ea4, v42 :: v_dual_mul_f32 v47, 0x114b4ea4, v43
	v_cndmask_b32_e64 v41, v41, v45, s6
	v_cndmask_b32_e32 v40, v40, v44, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v42, v42, v46, s7
	v_cndmask_b32_e64 v43, v43, v47, s8
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v45, v37, v33
	v_dual_add_f32 v41, 1.0, v41 :: v_dual_add_f32 v40, 1.0, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_add_f32 v42, 1.0, v42 :: v_dual_add_f32 v43, 1.0, v43
	v_dual_mul_f32 v44, v36, v32 :: v_dual_mul_f32 v47, v39, v35
	v_rcp_f32_e32 v40, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v41, v41
	v_rcp_f32_e32 v42, v42
	v_rcp_f32_e32 v43, v43
	v_mul_f32_e32 v46, v38, v34
	s_mov_b32 s6, 0
	s_delay_alu instid0(TRANS32_DEP_3)
	v_dual_mul_f32 v40, v44, v40 :: v_dual_mul_f32 v41, v45, v41
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v42, v46, v42 :: v_dual_mul_f32 v43, v47, v43
.LBB15_22:                              ;   in Loop: Header=BB15_5 Depth=1
	s_and_b32 s6, s6, exec_lo
	s_cselect_b32 s6, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s6, 1
	s_cbranch_scc1 .LBB15_24
; %bb.23:                               ;   in Loop: Header=BB15_5 Depth=1
	s_waitcnt vmcnt(0)
	v_dual_sub_f32 v40, 0x42800000, v36 :: v_dual_sub_f32 v41, 0x42800000, v37
	v_cmp_lt_f32_e32 vcc_lo, 0x42aeac50, v36
	v_cmp_lt_f32_e64 s6, 0x42aeac50, v37
	v_dual_sub_f32 v42, 0x42800000, v38 :: v_dual_sub_f32 v43, 0x42800000, v39
	v_cmp_lt_f32_e64 s7, 0x42aeac50, v38
	v_cndmask_b32_e64 v36, -v36, v40, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v37, -v37, v41, s6
	v_cmp_lt_f32_e64 s8, 0x42aeac50, v39
	v_cndmask_b32_e64 v38, -v38, v42, s7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v36, 0x3fb8aa3b, v36 :: v_dual_mul_f32 v37, 0x3fb8aa3b, v37
	v_cndmask_b32_e64 v39, -v39, v43, s8
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v38, 0x3fb8aa3b, v38
	v_exp_f32_e32 v36, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v37, v37
	v_mul_f32_e32 v39, 0x3fb8aa3b, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v38, v38
	v_exp_f32_e32 v39, v39
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v40, 0x114b4ea4, v36 :: v_dual_mul_f32 v41, 0x114b4ea4, v37
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v42, 0x114b4ea4, v38
	v_cndmask_b32_e64 v37, v37, v41, s6
	v_dual_mul_f32 v43, 0x114b4ea4, v39 :: v_dual_cndmask_b32 v36, v36, v40
	v_cndmask_b32_e64 v38, v38, v42, s7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v37, 1.0, v37
	v_cndmask_b32_e64 v39, v39, v43, s8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v36, 1.0, v36
	v_rcp_f32_e32 v37, v37
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v38, 1.0, v38 :: v_dual_add_f32 v39, 1.0, v39
	v_rcp_f32_e32 v36, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_3)
	v_rcp_f32_e32 v39, v39
	v_mul_f32_e32 v41, v33, v37
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v38, v38
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v40, v32, v36 :: v_dual_mul_f32 v43, v35, v39
	v_mul_f32_e32 v42, v34, v38
.LBB15_24:                              ;   in Loop: Header=BB15_5 Depth=1
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v32, v40 :: v_dual_mov_b32 v33, v41
	v_dual_mov_b32 v34, v42 :: v_dual_mov_b32 v35, v43
.LBB15_25:                              ;   in Loop: Header=BB15_5 Depth=1
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_cndmask_b32_e64 v36, v92, -v92, s9
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v37, v93, -v93, s21
	;;#ASMEND
	v_cvt_f16_f32_e32 v36.l, v36
	v_cvt_f16_f32_e32 v36.h, v37
	;;#ASMSTART
	v_cndmask_b32_e64 v37, v94, -v94, s22
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v38, v95, -v95, s23
	;;#ASMEND
	v_cvt_f16_f32_e32 v37.l, v37
	v_cvt_f16_f32_e32 v37.h, v38
	;;#ASMSTART
	v_cndmask_b32_e64 v38, v60, -v60, s31
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v39, v61, -v61, s33
	;;#ASMEND
	v_cvt_f16_f32_e32 v38.l, v38
	v_cvt_f16_f32_e32 v38.h, v39
	;;#ASMSTART
	v_cndmask_b32_e64 v39, v62, -v62, s34
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v40, v63, -v63, s35
	;;#ASMEND
	v_cvt_f16_f32_e32 v39.l, v39
	v_cvt_f16_f32_e32 v39.h, v40
	;;#ASMSTART
	v_cndmask_b32_e64 v40, v52, -v52, s36
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v41, v53, -v53, s37
	;;#ASMEND
	v_cvt_f16_f32_e32 v40.l, v40
	v_cvt_f16_f32_e32 v40.h, v41
	;;#ASMSTART
	v_cndmask_b32_e64 v41, v54, -v54, s38
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v42, v55, -v55, s39
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v32, v32, -v32, s40
	;;#ASMEND
	v_cvt_f16_f32_e32 v41.l, v41
	v_cvt_f16_f32_e32 v41.h, v42
	v_cvt_f16_f32_e32 v42.l, v32
	;;#ASMSTART
	v_cndmask_b32_e64 v32, v33, -v33, s41
	;;#ASMEND
	v_cvt_f16_f32_e32 v42.h, v32
	;;#ASMSTART
	v_cndmask_b32_e64 v32, v34, -v34, s21
	;;#ASMEND
	v_cvt_f16_f32_e32 v43.l, v32
	;;#ASMSTART
	v_cndmask_b32_e64 v32, v35, -v35, s22
	;;#ASMEND
	v_cvt_f16_f32_e32 v43.h, v32
	v_dual_mov_b32 v51, s19 :: v_dual_mov_b32 v50, s18
	v_dual_mov_b32 v49, s17 :: v_dual_mov_b32 v48, s16
	v_dual_mov_b32 v47, s15 :: v_dual_mov_b32 v46, s14
	v_dual_mov_b32 v45, s13 :: v_dual_mov_b32 v44, s12
	v_mov_b32_e32 v34, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[52:59], v[36:43], v[0:7], v[44:51]
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v41, 0
	v_cvt_f16_f32_e32 v33.l, v52
	v_cvt_f16_f32_e32 v33.h, v53
	v_cvt_f16_f32_e32 v35.l, v54
	v_cvt_f16_f32_e32 v35.h, v55
	v_cvt_f16_f32_e32 v38.l, v56
	v_cvt_f16_f32_e32 v38.h, v57
	v_cvt_f16_f32_e32 v40.l, v58
	v_cvt_f16_f32_e32 v40.h, v59
	v_permlanex16_b32 v34, v33, s43, 0xfedcba98
	v_permlanex16_b32 v37, v35, s43, 0xfedcba98
	v_permlanex16_b32 v39, v38, s43, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_permlanex16_b32 v41, v40, s43, 0xfedcba98
	v_cndmask_b32_e64 v32, v34, v33, s4
	v_cndmask_b32_e64 v36, v33, v34, s4
	v_cndmask_b32_e64 v33, v37, v35, s4
	v_cndmask_b32_e64 v37, v35, v37, s4
	v_cndmask_b32_e64 v34, v39, v38, s4
	v_cndmask_b32_e64 v38, v38, v39, s4
	v_cndmask_b32_e64 v35, v41, v40, s4
	v_cndmask_b32_e64 v39, v40, v41, s4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[52:59], v[8:15], v[32:39], v[44:51]
	v_mov_b32_e32 v33, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v48, 0
	v_mul_f32_e32 v34, 0x3d800000, v52
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v36, 0x3d800000, v53
	v_dual_mul_f32 v38, 0x3d800000, v54 :: v_dual_mul_f32 v41, 0x3d800000, v56
	v_dual_mul_f32 v40, 0x3d800000, v55 :: v_dual_mul_f32 v39, 0x3d800000, v57
	v_and_b32_e32 v32, 0x7fffffff, v34
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v42, 0x7fffffff, v36
	v_and_b32_e32 v43, 0x7fffffff, v38
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v49, 0 :: v_dual_and_b32 v44, 0x7fffffff, v40
	v_dual_mov_b32 v50, 0 :: v_dual_and_b32 v45, 0x7fffffff, v41
	v_mov_b32_dpp v47, v32 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v48, v42 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v49, v43 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
	v_mul_f32_e32 v37, 0x3d800000, v58
	v_and_b32_e32 v46, 0x7fffffff, v39
	v_mul_f32_e32 v35, 0x3d800000, v59
	v_mov_b32_dpp v50, v45 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
	v_max_f32_e32 v43, v47, v47
	v_dual_max_f32 v47, v48, v48 :: v_dual_max_f32 v48, v49, v49
	v_dual_mov_b32 v49, 0 :: v_dual_and_b32 v32, 0x7fffffff, v37
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v45, 0 :: v_dual_max_f32 v50, v50, v50
	v_max_f32_e64 v47, |v36|, v47
	v_max_f32_e64 v43, |v34|, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_mov_b32_dpp v49, v44 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
	v_mov_b32_e32 v44, 0
	v_and_b32_e32 v42, 0x7fffffff, v35
	v_mov_b32_dpp v45, v32 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
	v_mov_b32_e32 v52, 0
	v_max_f32_e64 v48, |v38|, v48
	v_mov_b32_dpp v44, v46 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
	v_mov_b32_e32 v51, 0
	v_max_f32_e32 v45, v45, v45
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v54, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_max_f32_e32 v44, v44, v44
	v_mov_b32_dpp v51, v42 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
	v_max_f32_e32 v49, v49, v49
	v_max_f32_e64 v42, |v41|, v50
	v_mov_b32_e32 v50, 0
	v_max_f32_e64 v44, |v39|, v44
	v_max_f32_e32 v46, v51, v51
	v_max_f32_e64 v32, |v40|, v49
	v_mov_b32_e32 v49, 0
	v_mov_b32_dpp v50, v47 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	v_mov_b32_e32 v51, 0
	v_max_f32_e64 v45, |v37|, v45
	v_mov_b32_dpp v52, v32 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v49, v43 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	v_dual_max_f32 v50, v50, v50 :: v_dual_mov_b32 v53, 0
	v_mov_b32_dpp v51, v48 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	v_max_f32_e64 v46, |v35|, v46
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_max_f32_e32 v49, v49, v49
	v_dual_max_f32 v47, v47, v50 :: v_dual_max_f32 v50, v52, v52
	v_mov_b32_e32 v52, 0
	v_mov_b32_dpp v53, v42 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_max_f32_e32 v43, v43, v49
	v_dual_max_f32 v49, v51, v51 :: v_dual_max_f32 v32, v32, v50
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mov_b32_dpp v52, v44 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	v_dual_max_f32 v51, v53, v53 :: v_dual_mov_b32 v50, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_max_f32_e32 v48, v48, v49
	v_mov_b32_e32 v56, 0
	v_dual_max_f32 v52, v52, v52 :: v_dual_mov_b32 v53, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_max_f32 v42, v42, v51 :: v_dual_mov_b32 v49, 0
	v_mov_b32_dpp v50, v43 row_half_mirror row_mask:0xf bank_mask:0xf
	v_max_f32_e32 v44, v44, v52
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mov_b32_dpp v53, v45 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v51, 0
	v_max_f32_e32 v50, v50, v50
	v_mov_b32_dpp v49, v46 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_max_f32_e32 v53, v53, v53
	v_mov_b32_dpp v52, v48 row_half_mirror row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v51, v47 row_half_mirror row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max_f32_e32 v49, v49, v49
	v_dual_max_f32 v45, v45, v53 :: v_dual_max_f32 v52, v52, v52
	v_mov_b32_e32 v53, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_max_f32 v51, v51, v51 :: v_dual_max_f32 v46, v46, v49
	v_dual_mov_b32 v49, 0 :: v_dual_max_f32 v48, v48, v52
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mov_b32_dpp v53, v32 row_half_mirror row_mask:0xf bank_mask:0xf
	v_mov_b32_e32 v52, 0
	v_mov_b32_dpp v49, v42 row_half_mirror row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_max_f32_e32 v53, v53, v53
	v_dual_max_f32 v43, v43, v50 :: v_dual_mov_b32 v50, 0
	v_mov_b32_dpp v52, v46 row_half_mirror row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max_f32_e32 v49, v49, v49
	v_mov_b32_dpp v50, v44 row_half_mirror row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_max_f32 v47, v47, v51 :: v_dual_max_f32 v52, v52, v52
	v_dual_mov_b32 v51, 0 :: v_dual_max_f32 v42, v42, v49
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_max_f32 v50, v50, v50 :: v_dual_mov_b32 v49, 0
	v_max_f32_e32 v46, v46, v52
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mov_b32_dpp v51, v45 row_half_mirror row_mask:0xf bank_mask:0xf
	v_mov_b32_e32 v52, 0
	v_max_f32_e32 v44, v44, v50
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_max_f32 v32, v32, v53 :: v_dual_max_f32 v51, v51, v51
	v_mov_b32_e32 v53, 0
	v_mov_b32_dpp v52, v42 row_mirror row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max_f32_e32 v50, v45, v51
	v_mov_b32_dpp v53, v43 row_mirror row_mask:0xf bank_mask:0xf
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v45, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max_f32_e32 v53, v53, v53
	v_mov_b32_dpp v51, v32 row_mirror row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b32_dpp v45, v47 row_mirror row_mask:0xf bank_mask:0xf
	v_max_f32_e32 v43, v43, v53
	v_mov_b32_e32 v53, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mov_b32_dpp v53, v44 row_mirror row_mask:0xf bank_mask:0xf
	v_max_f32_e32 v51, v51, v51
	v_dual_max_f32 v51, v32, v51 :: v_dual_mov_b32 v32, 0
	v_mov_b32_dpp v49, v48 row_mirror row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mov_b32_dpp v32, v50 row_mirror row_mask:0xf bank_mask:0xf
	v_max_f32_e32 v45, v45, v45
	v_dual_max_f32 v49, v49, v49 :: v_dual_max_f32 v32, v32, v32
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max_f32_e32 v47, v47, v45
	v_dual_max_f32 v45, v48, v49 :: v_dual_mov_b32 v48, 0
	v_dual_max_f32 v49, v52, v52 :: v_dual_max_f32 v52, v53, v53
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_permlanex16_b32 v56, v47, s43, 0xfedcba98
	v_mov_b32_dpp v48, v46 row_mirror row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_max_f32 v49, v42, v49 :: v_dual_max_f32 v42, v44, v52
	v_permlanex16_b32 v54, v43, s43, 0xfedcba98
	v_max_f32_e32 v44, v50, v32
	v_max_f32_e32 v48, v48, v48
	v_mov_b32_e32 v50, 0
	v_permlanex16_b32 v55, v45, s43, 0xfedcba98
	v_max_f32_e32 v52, v54, v54
	v_mov_b32_e32 v54, 0
	v_max_f32_e32 v46, v46, v48
	v_mov_b32_e32 v48, 0
	v_permlanex16_b32 v50, v44, s43, 0xfedcba98
	v_max_f32_e32 v43, v43, v52
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v52, 0
	v_permlanex16_b32 v54, v51, s43, 0xfedcba98
	v_permlanex16_b32 v48, v46, s43, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v32, 0x3c010204, v43
	v_permlanex16_b32 v53, v49, s43, 0xfedcba98
	v_permlanex16_b32 v52, v42, s43, 0xfedcba98
	v_cmp_neq_f32_e32 vcc_lo, 0, v43
	v_mov_b32_e32 v43, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB15_27
; %bb.26:                               ;   in Loop: Header=BB15_5 Depth=1
	v_rcp_f32_e32 v43, v32
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v57, v34, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_trunc_f32_e32 v58, v57
	v_fma_f32 v34, v34, v43, -v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ge_f32_e64 s7, |v34|, 0.5
	v_cndmask_b32_e64 v34, 0, 1.0, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_bfi_b32 v34, 0x7fffffff, v34, v57
	v_add_f32_e32 v34, v58, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v43, v34
.LBB15_27:                              ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s6
	v_dual_max_f32 v34, v56, v56 :: v_dual_max_f32 v47, v47, v47
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v47, v47, v34
	v_mul_f32_e32 v34, 0x3c010204, v47
	v_cmpx_neq_f32_e32 0, v47
	s_cbranch_execz .LBB15_29
; %bb.28:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v33, v34
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v47, v36, v33
	v_trunc_f32_e32 v56, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v33, v36, v33, -v56
	v_cmp_ge_f32_e64 s7, |v33|, 0.5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, 0, 1.0, s7
	v_bfi_b32 v33, 0x7fffffff, v33, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v56, v33
	v_cvt_i32_f32_e32 v33, v33
.LBB15_29:                              ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s6
	v_dual_max_f32 v36, v55, v55 :: v_dual_max_f32 v45, v45, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v47, v45, v36
	v_dual_mov_b32 v45, 0 :: v_dual_mul_f32 v36, 0x3c010204, v47
	v_cmp_neq_f32_e32 vcc_lo, 0, v47
	v_mov_b32_e32 v47, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB15_31
; %bb.30:                               ;   in Loop: Header=BB15_5 Depth=1
	v_rcp_f32_e32 v47, v36
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v55, v38, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_trunc_f32_e32 v56, v55
	v_fma_f32 v38, v38, v47, -v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ge_f32_e64 s7, |v38|, 0.5
	v_cndmask_b32_e64 v38, 0, 1.0, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_bfi_b32 v38, 0x7fffffff, v38, v55
	v_add_f32_e32 v38, v56, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v47, v38
.LBB15_31:                              ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s6
	v_dual_max_f32 v38, v54, v54 :: v_dual_max_f32 v51, v51, v51
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v51, v51, v38
	v_mul_f32_e32 v38, 0x3c010204, v51
	v_cmpx_neq_f32_e32 0, v51
	s_cbranch_execz .LBB15_33
; %bb.32:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v45, v38
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v51, v40, v45
	v_trunc_f32_e32 v54, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v40, v40, v45, -v54
	v_cmp_ge_f32_e64 s7, |v40|, 0.5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v40, 0, 1.0, s7
	v_bfi_b32 v40, 0x7fffffff, v40, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v40, v54, v40
	v_cvt_i32_f32_e32 v45, v40
.LBB15_33:                              ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s6
	v_max_f32_e32 v40, v53, v53
	v_max_f32_e32 v49, v49, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v51, v49, v40
	v_dual_mov_b32 v49, 0 :: v_dual_mul_f32 v40, 0x3c010204, v51
	v_cmp_neq_f32_e32 vcc_lo, 0, v51
	v_mov_b32_e32 v51, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB15_35
; %bb.34:                               ;   in Loop: Header=BB15_5 Depth=1
	v_rcp_f32_e32 v51, v40
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v53, v41, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_trunc_f32_e32 v54, v53
	v_fma_f32 v41, v41, v51, -v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ge_f32_e64 s7, |v41|, 0.5
	v_cndmask_b32_e64 v41, 0, 1.0, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_bfi_b32 v41, 0x7fffffff, v41, v53
	v_add_f32_e32 v41, v54, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v51, v41
.LBB15_35:                              ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s6
	v_dual_max_f32 v41, v52, v52 :: v_dual_max_f32 v42, v42, v42
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v41, v42, v41
	v_mul_f32_e32 v42, 0x3c010204, v41
	v_cmpx_neq_f32_e32 0, v41
	s_cbranch_execz .LBB15_37
; %bb.36:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v41, v42
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v49, v39, v41
	v_trunc_f32_e32 v52, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v39, v39, v41, -v52
	v_cmp_ge_f32_e64 s7, |v39|, 0.5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v39, 0, 1.0, s7
	v_bfi_b32 v39, 0x7fffffff, v39, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v39, v52, v39
	v_cvt_i32_f32_e32 v49, v39
.LBB15_37:                              ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s6
	v_max_f32_e32 v39, v50, v50
	v_max_f32_e32 v41, v44, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v41, v41, v39
	v_dual_mov_b32 v39, 0 :: v_dual_mul_f32 v44, 0x3c010204, v41
	v_cmp_neq_f32_e32 vcc_lo, 0, v41
	v_mov_b32_e32 v41, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB15_39
; %bb.38:                               ;   in Loop: Header=BB15_5 Depth=1
	v_rcp_f32_e32 v41, v44
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v50, v37, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_trunc_f32_e32 v52, v50
	v_fma_f32 v37, v37, v41, -v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ge_f32_e64 s7, |v37|, 0.5
	v_cndmask_b32_e64 v37, 0, 1.0, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_bfi_b32 v37, 0x7fffffff, v37, v50
	v_add_f32_e32 v37, v52, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v41, v37
.LBB15_39:                              ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s6
	v_dual_max_f32 v37, v48, v48 :: v_dual_max_f32 v46, v46, v46
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v37, v46, v37
	v_mul_f32_e32 v46, 0x3c010204, v37
	v_cmpx_neq_f32_e32 0, v37
	s_cbranch_execz .LBB15_41
; %bb.40:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v37, v46
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v39, v35, v37
	v_trunc_f32_e32 v48, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v35, v35, v37, -v48
	v_cmp_ge_f32_e64 s7, |v35|, 0.5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v35, 0, 1.0, s7
	v_bfi_b32 v35, 0x7fffffff, v35, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v35, v48, v35
	v_cvt_i32_f32_e32 v39, v35
.LBB15_41:                              ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s6
	v_cvt_f32_i32_e32 v35, v41
	v_cvt_f32_i32_e32 v37, v49
	v_cvt_f32_i32_e32 v48, v51
	v_cvt_f32_i32_e32 v50, v43
	v_cvt_f32_i32_e32 v52, v33
	v_cvt_f32_i32_e32 v53, v47
	v_cvt_f32_i32_e32 v54, v45
	v_cvt_f32_i32_e32 v55, v39
	v_add_f32_dpp v50, v50, v50 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v52, v52, v52 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v53, v53, v53 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v54, v54, v54 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v48, v48, v48 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v37, v37, v37 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v35, v35, v35 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v55, v55, v55 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v50, v50, v50 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v52, v52, v52 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v53, v53, v53 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v54, v54, v54 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v48, v48, v48 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v37, v37, v37 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v35, v35, v35 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v55, v55, v55 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v50, v50, v50 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v52, v52, v52 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v53, v53, v53 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v54, v54, v54 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v56, v48, v48 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v57, v37, v37 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v58, v35, v35 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v59, v55, v55 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v35, v50, v50 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v37, v52, v52 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v48, v53, v53 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v50, v54, v54 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v52, v56, v56 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v55, v57, v57 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v58, v58, v58 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v57, 0
	v_add_f32_dpp v60, v59, v59 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v61, 0 :: v_dual_add_nc_u32 v92, v78, v76
	v_mov_b32_e32 v63, 0
	v_permlanex16_b32 v53, v35, s43, 0xfedcba98
	v_permlanex16_b32 v54, v37, s43, 0xfedcba98
	v_permlanex16_b32 v56, v48, s43, 0xfedcba98
	v_permlanex16_b32 v57, v50, s43, 0xfedcba98
	v_permlanex16_b32 v59, v52, s43, 0xfedcba98
	v_permlanex16_b32 v61, v55, s43, 0xfedcba98
	v_permlanex16_b32 v62, v58, s43, 0xfedcba98
	v_permlanex16_b32 v63, v60, s43, 0xfedcba98
	ds_store_b8 v92, v43
	ds_store_b8 v92, v33 offset:32
	ds_store_b8 v92, v47 offset:64
	ds_store_b8 v92, v45 offset:96
	ds_store_b8 v92, v51 offset:128
	ds_store_b8 v92, v49 offset:160
	ds_store_b8 v92, v41 offset:192
	ds_store_b8 v92, v39 offset:224
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB15_4
; %bb.42:                               ;   in Loop: Header=BB15_5 Depth=1
	v_add_f32_e32 v33, v35, v53
	v_add_f32_e32 v35, v37, v54
	v_add_f32_e32 v37, v48, v56
	v_add_f32_e32 v39, v50, v57
	v_add_f32_e32 v43, v55, v61
	v_add_f32_e32 v41, v52, v59
	v_add_f32_e32 v47, v60, v63
	v_add_f32_e32 v45, v58, v62
	v_mul_f32_e32 v33, v33, v32
	v_mul_f32_e32 v35, v35, v34
	v_mul_f32_e32 v37, v37, v36
	v_mul_f32_e32 v39, v39, v38
	v_mul_f32_e32 v41, v41, v40
	v_mul_f32_e32 v43, v43, v42
	v_mul_f32_e32 v45, v45, v44
	v_mul_f32_e32 v47, v47, v46
	ds_store_b128 v79, v[32:35]
	ds_store_b128 v79, v[36:39] offset:16
	ds_store_b128 v79, v[40:43] offset:32
	ds_store_b128 v79, v[44:47] offset:48
	s_branch .LBB15_4
.LBB15_43:
	s_or_b32 exec_lo, exec_lo, s42
.LBB15_44:
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	s_or_b32 exec_lo, exec_lo, s26
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, 0
	s_mov_b32 s4, 0x76543210
	v_lshlrev_b32_e32 v0, 2, v73
	v_permlanex16_b32 v2, v75, s4, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_3)
	v_permlanex16_b32 v1, v1, s4, 0xfedcba98
	s_and_saveexec_b32 s4, s3
; %bb.45:
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v3, v66, 7, v0
	v_dual_add_f32 v2, v75, v2 :: v_dual_add_nc_u32 v3, 0x800, v3
	ds_store_2addr_b32 v3, v2, v1 offset0:128 offset1:144
; %bb.46:
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v1, v72, v66
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_mov_b32 s3, exec_lo
	v_cmpx_eq_u32_e32 0, v1
	s_cbranch_execz .LBB15_48
; %bb.47:
	v_add_nc_u32_e32 v3, 0x800, v0
	v_add_nc_u32_e32 v7, 0xc00, v0
	s_load_b32 s0, s[0:1], 0x18
	ds_load_2addr_b32 v[1:2], v3 offset0:128 offset1:160
	ds_load_2addr_b32 v[3:4], v3 offset0:192 offset1:224
	ds_load_2addr_b32 v[5:6], v7 offset1:32
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v1
	ds_load_2addr_b32 v[0:1], v7 offset0:64 offset1:96
	v_add_f32_e32 v2, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, v4, v2
	ds_load_2addr_b32 v[2:3], v7 offset0:128 offset1:160
	v_add_f32_e32 v4, v5, v4
	v_add_f32_e32 v4, v6, v4
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v4
	v_dual_add_f32 v0, v1, v0 :: v_dual_mul_f32 v1, s0, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v2, v0
	v_add_f32_e32 v0, v3, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v2, v1, v0
	v_fmac_f32_e32 v74, v1, v0
	v_add_co_u32 v0, vcc_lo, s24, v64
	v_add_co_ci_u32_e64 v1, null, s25, v65, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v2, v2, v74, s2
	global_store_b32 v[0:1], v2, off
.LBB15_48:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
