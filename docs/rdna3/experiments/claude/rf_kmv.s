_ZN12_GLOBAL__N_14k_mvILi1ELb0ELb1ELb0ELb0ELb0EEEvPKhS2_PKffS2_S2_S4_fPKaPK15HIP_vector_typeIfLj2EEPfS4_S4_iiiiS4_S4_iiS4_f19ggml_cuda_conv_foldSB_iNS_11mv_jobs_devE: ; @_ZN12_GLOBAL__N_14k_mvILi1ELb0ELb1ELb0ELb0ELb0EEEvPKhS2_PKffS2_S2_S4_fPKaPK15HIP_vector_typeIfLj2EEPfS4_S4_iiiiS4_S4_iiS4_f19ggml_cuda_conv_foldSB_iNS_11mv_jobs_devE
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	v_dual_mov_b32 v95, 0 :: v_dual_and_b32 v16, 0x3ff, v0
	s_clause 0x1
	s_load_b64 s[2:3], s[0:1], 0x10
	s_load_b128 s[44:47], s[0:1], 0x50
	v_bfe_u32 v92, v0, 4, 6
	v_and_b32_e32 v93, 15, v16
	v_dual_mov_b32 v5, v16 :: v_dual_and_b32 v6, 13, v16
	v_mov_b32_e32 v91, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v1, s15, 4, v93
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[80:81], 2, v[1:2]
	s_waitcnt lgkmcnt(0)
	s_cmp_lg_u64 s[46:47], 0
	v_add_co_u32 v1, vcc_lo, s2, v80
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v2, null, s3, v81, vcc_lo
	v_cmp_eq_u32_e64 s3, 0, v92
	s_cselect_b32 s2, -1, 0
	global_load_b32 v90, v[1:2], off
	s_and_b32 s5, s3, s2
	s_and_saveexec_b32 s4, s5
	s_cbranch_execz .LBB15_2
; %bb.1:
	v_add_co_u32 v1, vcc_lo, s46, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s47, v81, vcc_lo
	global_load_b32 v91, v[1:2], off
.LBB15_2:
	s_or_b32 exec_lo, exec_lo, s4
	s_load_b32 s4, s[0:1], 0x70
	v_bfe_u32 v94, v0, 10, 10
	s_mov_b32 s33, exec_lo
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s5, s4, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshr_b32 s5, s5, 24
	s_add_i32 s4, s4, s5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s34, s4, 8
	v_cmpx_gt_i32_e32 s34, v94
	s_cbranch_execz .LBB15_65
; %bb.3:
	v_dual_mov_b32 v105, v94 :: v_dual_and_b32 v2, 3, v16
	v_dual_mov_b32 v95, 0 :: v_dual_and_b32 v0, 1, v16
	v_and_b32_e32 v3, 6, v16
	v_and_b32_e32 v4, 5, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_bcnt_u32_b32 v2, v2, 0
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	v_and_b32_e32 v1, 2, v16
	v_mov_b16_e32 v0.h, 0xbc00
	v_bcnt_u32_b32 v3, v3, 0
	v_bcnt_u32_b32 v4, v4, 0
	v_and_b32_e32 v2, 1, v2
	v_and_b32_e32 v7, 10, v16
	v_cmp_eq_u32_e64 s4, 0, v1
	v_cndmask_b16 v12.l, v0.h, 0x3c00, vcc_lo
	v_and_b32_e32 v3, 1, v3
	v_and_b32_e32 v4, 1, v4
	v_cmp_eq_u32_e32 vcc_lo, 0, v2
	v_bcnt_u32_b32 v2, v7, 0
	v_cndmask_b16 v1.l, v0.h, 0x3c00, s4
	v_cmp_eq_u32_e64 s4, 0, v3
	v_cmp_eq_u32_e64 s6, 0, v4
	v_and_b32_e32 v4, 8, v16
	v_and_b32_e32 v3, 1, v2
	v_and_b32_e32 v7, 7, v5
	v_and_b32_e32 v8, 4, v16
	v_cndmask_b16 v1.h, v0.h, 0x3c00, vcc_lo
	v_cndmask_b16 v9.h, v0.h, 0x3c00, s4
	v_cmp_eq_u32_e32 vcc_lo, 0, v3
	v_and_b32_e32 v3, 6, v16
	v_cmp_eq_u32_e64 s4, 0, v4
	v_bcnt_u32_b32 v4, v7, 0
	v_and_b32_e32 v7, 14, v5
	v_cmp_eq_u32_e64 s5, 0, v8
	v_bcnt_u32_b32 v3, v3, 0
	v_and_b32_e32 v8, 12, v16
	v_and_b32_e32 v4, 1, v4
	v_bcnt_u32_b32 v7, v7, 0
	v_and_b32_e32 v11, 9, v16
	v_cndmask_b16 v10.h, v0.h, 0x3c00, vcc_lo
	v_and_b32_e32 v3, 1, v3
	v_cmp_eq_u32_e32 vcc_lo, 0, v4
	v_bcnt_u32_b32 v4, v8, 0
	v_and_b32_e32 v7, 1, v7
	v_bcnt_u32_b32 v8, v11, 0
	v_cndmask_b16 v10.l, v0.h, 0x3c00, s4
	v_cmp_eq_u32_e64 s4, 0, v3
	v_cndmask_b16 v3.h, v0.h, 0x3c00, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v7
	v_and_b32_e32 v7, 1, v8
	v_and_b32_e32 v8, 11, v5
	v_and_b32_e32 v4, 1, v4
	v_cndmask_b16 v3.l, v0.h, 0x3c00, s4
	v_cndmask_b16 v11.h, v0.h, 0x3c00, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v7
	v_bcnt_u32_b32 v7, v8, 0
	v_and_b32_e32 v8, 5, v16
	v_cmp_eq_u32_e64 s4, 0, v4
	v_and_b32_e32 v4, 10, v16
	v_and_b32_e32 v5, 13, v5
	v_and_b32_e32 v7, 1, v7
	v_bcnt_u32_b32 v8, v8, 0
	v_and_b32_e32 v14, 14, v16
	v_bcnt_u32_b32 v13, v4, 0
	v_cndmask_b16 v4.h, v0.h, 0x3c00, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v7
	v_and_b32_e32 v7, 1, v8
	v_bcnt_u32_b32 v8, v5, 0
	v_and_b32_e32 v13, 1, v13
	s_clause 0x2
	s_load_b128 s[8:11], s[0:1], 0x0
	s_load_b256 s[24:31], s[0:1], 0x78
	s_load_b32 s35, s[0:1], 0x98
	v_cndmask_b16 v5.h, v0.h, 0x3c00, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v7
	v_and_b32_e32 v8, 1, v8
	v_cndmask_b16 v11.l, v0.h, 0x3c00, s4
	v_cmp_eq_u32_e64 s4, 0, v13
	v_and_b32_e32 v7, 9, v16
	v_cndmask_b16 v13.l, v0.h, 0x3c00, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v8
	v_bcnt_u32_b32 v8, v14, 0
	v_lshlrev_b32_e32 v20, 4, v16
	v_bcnt_u32_b32 v7, v7, 0
	v_lshlrev_b32_e32 v19, 6, v16
	v_cndmask_b16 v5.l, v0.h, 0x3c00, s4
	v_and_b32_e32 v8, 1, v8
	v_and_b32_e32 v20, 0xf0, v20
	v_and_b32_e32 v7, 1, v7
	v_and_b32_e32 v19, 0x1c0, v19
	v_bcnt_u32_b32 v14, v93, 0
	v_cmp_eq_u32_e64 s4, 0, v8
	v_bcnt_u32_b32 v15, v6, 0
	v_lshlrev_b32_e32 v17, 10, v92
	v_mul_lo_u32 v24, 0x9e3779b1, v20
	v_cndmask_b16 v6.h, v0.h, 0x3c00, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v7
	v_cndmask_b16 v7.l, v0.h, 0x3c00, s4
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v82, s4, s30, v19
	v_lshlrev_b32_e32 v18, 4, v93
	v_and_b32_e32 v14, 1, v14
	v_and_b32_e32 v15, 1, v15
	v_add_co_ci_u32_e64 v83, null, s31, 0, s4
	v_add_co_u32 v17, s4, s8, v17
	v_lshlrev_b32_e32 v21, 3, v92
	v_add_co_ci_u32_e64 v19, null, s9, 0, s4
	v_add_co_u32 v22, s4, s10, v18
	v_cndmask_b16 v9.l, v0.h, 0x3c00, s5
	v_cndmask_b16 v2.h, v0.h, 0x3c00, s6
	v_cmp_eq_u32_e64 s5, 0, v14
	v_cmp_eq_u32_e64 s6, 0, v15
	v_cndmask_b16 v14.l, v0.h, 0x3c00, vcc_lo
	v_add_co_u32 v96, vcc_lo, v17, v18
	v_add_nc_u32_e32 v17, 0x9e3779b0, v24
	v_add_nc_u32_e32 v18, 0x3c6ef360, v24
	v_add_co_ci_u32_e64 v97, null, 0, v19, vcc_lo
	v_add_co_u32 v98, vcc_lo, v22, v21
	v_add_nc_u32_e32 v22, 0xb54cda20, v24
	v_cndmask_b16 v15.h, v0.h, 0x3c00, s5
	v_cndmask_b16 v15.l, v0.h, 0x3c00, s6
	v_cmp_gt_i32_e64 s5, 0, v17
	v_cmp_gt_i32_e64 s6, 0, v18
	v_add_nc_u32_e32 v17, 0xdaa66d10, v24
	v_add_nc_u32_e32 v18, 0x78dde6c0, v24
	v_add_co_ci_u32_e64 v23, null, s11, 0, s4
	v_cmp_gt_i32_e64 s10, 0, v22
	v_add_nc_u32_e32 v22, 0xcc623a90, v24
	v_cmp_gt_i32_e64 s7, 0, v17
	v_cmp_gt_i32_e64 s8, 0, v18
	v_add_nc_u32_e32 v17, 0xf1bbcd80, v24
	v_add_nc_u32_e32 v18, 0x8ff34730, v24
	v_add_co_ci_u32_e64 v99, null, 0, v23, vcc_lo
	v_add_nc_u32_e32 v21, 0x17156070, v24
	v_add_nc_u32_e32 v23, 0x538453d0, v24
	s_mul_i32 s46, s34, s15
	v_cmp_gt_i32_e64 s15, 0, v22
	v_add_nc_u32_e32 v22, 0xa708a7a0, v24
	v_cmp_gt_i32_e64 s12, 0, v17
	v_cmp_gt_i32_e64 s13, 0, v18
	v_mad_i64_i32 v[17:18], null, s29, v94, 0
	v_cmp_gt_i32_e64 s9, 0, v21
	v_cmp_gt_i32_e64 s11, 0, v23
	v_add_nc_u32_e32 v21, 0x2e2ac0e0, v24
	v_add_nc_u32_e32 v23, 0x6a99b440, v24
	v_cmp_gt_i32_e64 s18, 0, v22
	v_lshlrev_b32_e32 v22, 6, v93
	v_cmp_gt_i32_e64 s4, 0, v24
	v_cmp_gt_i32_e64 s14, 0, v21
	v_cmp_gt_i32_e64 s16, 0, v23
	v_add_nc_u32_e32 v21, 0x8d12df0, v24
	v_add_nc_u32_e32 v23, 0x45402150, v24
	v_lshlrev_b32_e32 v24, 8, v94
	v_and_b32_e32 v25, 0x3f0, v16
	v_lshlrev_b64 v[17:18], 2, v[17:18]
	v_lshl_or_b32 v22, v94, 10, v22
	v_mov_b16_e32 v0.l, 0x3c00
	v_lshlrev_b32_e32 v19, 7, v92
	v_cmp_gt_i32_e64 s17, 0, v21
	v_cmp_gt_u32_e64 s20, 16, v16
	v_add_nc_u32_e32 v21, v24, v25
	v_cmp_eq_u32_e64 s21, 0, v16
	v_lshl_add_u32 v100, v94, 6, 0xf00
	v_lshlrev_b32_e32 v16, 5, v92
	s_cmp_lg_u64 s[26:27], 0
	v_add_co_u32 v84, vcc_lo, s26, v17
	v_add_co_u32 v86, s22, s24, v22
	s_cselect_b32 s47, -1, 0
	s_cmp_eq_u32 s28, 3
	v_mov_b16_e32 v2.l, v9.l
	v_mov_b16_e32 v4.l, v10.l
	v_mov_b16_e32 v12.h, v1.h
	v_mov_b16_e32 v13.h, v3.h
	v_mov_b16_e32 v6.l, v11.l
	v_mov_b16_e32 v14.h, v5.h
	v_mov_b16_e32 v0.h, v12.l
	v_mov_b16_e32 v8.l, v0.l
	v_mov_b16_e32 v8.h, v1.l
	v_mov_b16_e32 v7.h, v15.h
	v_cmp_gt_i32_e64 s19, 0, v23
	v_add_co_ci_u32_e64 v85, null, s27, v18, vcc_lo
	v_add_co_ci_u32_e64 v87, null, s25, 0, s22
	v_lshlrev_b32_e32 v101, 2, v20
	v_add_nc_u32_e32 v102, v24, v19
	v_add_nc_u32_e32 v103, v100, v16
	v_add_nc_u32_e32 v104, v21, v93
	s_mov_b32 s36, 0
	s_cselect_b32 s48, -1, 0
	s_cmp_lg_u32 s28, 2
	s_mul_hi_i32 s26, s29, 40
	s_cselect_b32 s28, -1, 0
	s_mul_i32 s27, s29, 40
	s_mov_b32 s37, s36
	s_mov_b32 s38, s36
	s_mov_b32 s39, s36
	s_mov_b32 s40, s36
	s_mov_b32 s41, s36
	s_mov_b32 s42, s36
	s_mov_b32 s43, s36
	s_mov_b32 s29, 0x76543210
	s_mov_b32 s25, s36
                                        ; implicit-def: $vgpr40_vgpr41_vgpr42_vgpr43
                                        ; implicit-def: $vgpr24_vgpr25_vgpr26_vgpr27
                                        ; implicit-def: $vgpr44_vgpr45_vgpr46_vgpr47
                                        ; implicit-def: $vgpr28_vgpr29_vgpr30_vgpr31
                                        ; implicit-def: $vgpr48_vgpr49_vgpr50_vgpr51
                                        ; implicit-def: $vgpr32_vgpr33_vgpr34_vgpr35
                                        ; implicit-def: $vgpr52_vgpr53_vgpr54_vgpr55
                                        ; implicit-def: $vgpr36_vgpr37_vgpr38_vgpr39
	s_branch .LBB15_5
.LBB15_4:                               ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	; wave barrier
	ds_load_b128 v[16:19], v102
	ds_load_b128 v[20:23], v102 offset:16
	ds_load_b128 v[72:75], v102 offset:32
	ds_load_b128 v[76:79], v102 offset:48
	ds_load_b128 v[106:109], v102 offset:64
	ds_load_b128 v[110:113], v102 offset:80
	ds_load_b128 v[114:117], v102 offset:96
	ds_load_b128 v[118:121], v102 offset:112
	s_waitcnt vmcnt(4)
	v_and_b32_e32 v122, 0xf0f0f0f, v68
	v_lshrrev_b32_e32 v68, 4, v68
	v_and_b32_e32 v123, 0xf0f0f0f, v69
	v_lshrrev_b32_e32 v69, 4, v69
	v_and_b32_e32 v124, 0xf0f0f0f, v70
	v_xor_b32_e32 v122, 0x8080808, v122
	v_and_b32_e32 v68, 0xf0f0f0f, v68
	v_xor_b32_e32 v123, 0x8080808, v123
	v_and_b32_e32 v69, 0xf0f0f0f, v69
	v_lshrrev_b32_e32 v70, 4, v70
	v_and_b32_e32 v125, 0xf0f0f0f, v71
	v_xor_b32_e32 v68, 0x8080808, v68
	s_waitcnt lgkmcnt(7)
	v_dot4_i32_iu8 v16, v122, v16, 0 neg_lo:[1,1,0]
	s_waitcnt vmcnt(3)
	v_and_b32_e32 v122, 0xf0f0f0f, v64
	v_lshrrev_b32_e32 v64, 4, v64
	v_xor_b32_e32 v69, 0x8080808, v69
	v_and_b32_e32 v70, 0xf0f0f0f, v70
	v_dot4_i32_iu8 v16, v68, v17, v16 neg_lo:[1,1,0]
	v_and_b32_e32 v17, 0xf0f0f0f, v65
	v_lshrrev_b32_e32 v65, 4, v65
	v_lshrrev_b32_e32 v71, 4, v71
	v_and_b32_e32 v68, 0xf0f0f0f, v66
	v_dot4_i32_iu8 v16, v123, v18, v16 neg_lo:[1,1,0]
	v_and_b32_e32 v18, 0xf0f0f0f, v64
	v_xor_b32_e32 v64, 0x8080808, v122
	v_xor_b32_e32 v122, 0x8080808, v124
	v_xor_b32_e32 v17, 0x8080808, v17
	v_dot4_i32_iu8 v16, v69, v19, v16 neg_lo:[1,1,0]
	v_xor_b32_e32 v18, 0x8080808, v18
	s_waitcnt lgkmcnt(5)
	v_dot4_i32_iu8 v19, v64, v72, 0 neg_lo:[1,1,0]
	v_xor_b32_e32 v64, 0x8080808, v70
	v_add_nc_u32_e32 v105, 10, v105
	v_dot4_i32_iu8 v16, v122, v20, v16 neg_lo:[1,1,0]
	v_and_b32_e32 v20, 0xf0f0f0f, v65
	v_dot4_i32_iu8 v18, v18, v73, v19 neg_lo:[1,1,0]
	v_xor_b32_e32 v19, 0x8080808, v125
	v_and_b32_e32 v65, 0xf0f0f0f, v71
	v_dot4_i32_iu8 v16, v64, v21, v16 neg_lo:[1,1,0]
	v_xor_b32_e32 v20, 0x8080808, v20
	v_dot4_i32_iu8 v17, v17, v74, v18 neg_lo:[1,1,0]
	v_lshrrev_b32_e32 v18, 4, v66
	v_xor_b32_e32 v21, 0x8080808, v65
	v_dot4_i32_iu8 v16, v19, v22, v16 neg_lo:[1,1,0]
	v_xor_b32_e32 v19, 0x8080808, v68
	v_dot4_i32_iu8 v17, v20, v75, v17 neg_lo:[1,1,0]
	v_and_b32_e32 v18, 0xf0f0f0f, v18
	v_and_b32_e32 v20, 0xf0f0f0f, v67
	v_dot4_i32_iu8 v65, v21, v23, v16 neg_lo:[1,1,0]
	s_waitcnt vmcnt(2)
	v_and_b32_e32 v16, 0xf0f0f0f, v56
	s_waitcnt lgkmcnt(4)
	v_dot4_i32_iu8 v17, v19, v76, v17 neg_lo:[1,1,0]
	v_xor_b32_e32 v18, 0x8080808, v18
	v_lshrrev_b32_e32 v19, 4, v56
	v_xor_b32_e32 v68, 0x8080808, v20
	v_xor_b32_e32 v16, 0x8080808, v16
	v_and_b32_e32 v20, 0xf0f0f0f, v57
	v_dot4_i32_iu8 v66, v18, v77, v17 neg_lo:[1,1,0]
	v_and_b32_e32 v17, 0xf0f0f0f, v19
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v18, 0xf0f0f0f, v60
	v_lshrrev_b32_e32 v19, 4, v60
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_iu8 v16, v16, v106, 0 neg_lo:[1,1,0]
	v_lshrrev_b32_e32 v21, 4, v57
	v_xor_b32_e32 v17, 0x8080808, v17
	v_xor_b32_e32 v18, 0x8080808, v18
	v_and_b32_e32 v19, 0xf0f0f0f, v19
	v_lshrrev_b32_e32 v22, 4, v61
	v_lshrrev_b32_e32 v23, 4, v58
	v_dot4_i32_iu8 v16, v17, v107, v16 neg_lo:[1,1,0]
	v_xor_b32_e32 v17, 0x8080808, v20
	v_and_b32_e32 v20, 0xf0f0f0f, v21
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_iu8 v18, v18, v114, 0 neg_lo:[1,1,0]
	v_xor_b32_e32 v19, 0x8080808, v19
	v_and_b32_e32 v21, 0xf0f0f0f, v61
	v_dot4_i32_iu8 v16, v17, v108, v16 neg_lo:[1,1,0]
	v_xor_b32_e32 v17, 0x8080808, v20
	v_and_b32_e32 v20, 0xf0f0f0f, v22
	v_dot4_i32_iu8 v18, v19, v115, v18 neg_lo:[1,1,0]
	v_xor_b32_e32 v19, 0x8080808, v21
	v_lshrrev_b32_e32 v22, 4, v62
	v_dot4_i32_iu8 v16, v17, v109, v16 neg_lo:[1,1,0]
	v_and_b32_e32 v17, 0xf0f0f0f, v58
	v_lshrrev_b32_e32 v21, 4, v67
	v_dot4_i32_iu8 v18, v19, v116, v18 neg_lo:[1,1,0]
	v_xor_b32_e32 v19, 0x8080808, v20
	v_and_b32_e32 v20, 0xf0f0f0f, v62
	v_xor_b32_e32 v17, 0x8080808, v17
	v_and_b32_e32 v57, 0xf0f0f0f, v21
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_dot4_i32_iu8 v18, v19, v117, v18 neg_lo:[1,1,0]
	v_xor_b32_e32 v19, 0x8080808, v20
	v_and_b32_e32 v20, 0xf0f0f0f, v22
	v_dot4_i32_iu8 v58, v17, v110, v16 neg_lo:[1,1,0]
	v_and_b32_e32 v67, 0xf0f0f0f, v63
	v_lshrrev_b32_e32 v63, 4, v63
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_iu8 v61, v19, v118, v18 neg_lo:[1,1,0]
	v_xor_b32_e32 v62, 0x8080808, v20
	ds_load_b128 v[16:19], v103
	ds_load_b128 v[20:23], v103 offset:16
	v_xor_b32_e32 v60, 0x8080808, v60
	v_and_b32_e32 v63, 0xf0f0f0f, v63
	v_xor_b32_e32 v57, 0x8080808, v57
	v_dot4_i32_iu8 v61, v62, v119, v61 neg_lo:[1,1,0]
	v_xor_b32_e32 v62, 0x8080808, v67
	v_and_b32_e32 v67, 0xf0f0f0f, v59
	v_lshrrev_b32_e32 v59, 4, v59
	v_dot4_i32_iu8 v58, v60, v111, v58 neg_lo:[1,1,0]
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v56, 16, v89
	v_dot4_i32_iu8 v60, v62, v120, v61 neg_lo:[1,1,0]
	v_xor_b32_e32 v61, 0x8080808, v63
	v_xor_b32_e32 v62, 0x8080808, v67
	v_and_b32_e32 v59, 0xf0f0f0f, v59
	v_dot4_i32_iu8 v63, v68, v78, v66 neg_lo:[1,1,0]
	v_cvt_f32_u32_e32 v56, v56
	v_dot4_i32_iu8 v60, v61, v121, v60 neg_lo:[1,1,0]
	v_dot4_i32_iu8 v58, v62, v112, v58 neg_lo:[1,1,0]
	v_xor_b32_e32 v59, 0x8080808, v59
	v_dot4_i32_iu8 v57, v57, v79, v63 neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v23, 0xc1000000, v23
	v_cvt_f32_i32_e32 v60, v60
	v_lshrrev_b32_e32 v61, 16, v88
	v_dot4_i32_iu8 v58, v59, v113, v58 neg_lo:[1,1,0]
	v_cvt_f32_i32_e32 v57, v57
	v_mul_f32_e32 v19, 0xc1000000, v19
	v_dual_fmac_f32 v23, v22, v60 :: v_dual_and_b32 v126, 0xffff, v89
	v_cvt_f32_u32_e32 v22, v61
	v_cvt_f32_i32_e32 v58, v58
	v_mul_f32_e32 v21, 0xc1000000, v21
	v_fmac_f32_e32 v19, v18, v57
	v_fmac_f32_e32 v95, v23, v56
	v_and_b32_e32 v64, 0xffff, v88
	v_cvt_f32_u32_e32 v18, v126
	v_cvt_f32_i32_e32 v23, v65
	v_mul_f32_e32 v17, 0xc1000000, v17
	v_fmac_f32_e32 v21, v20, v58
	v_fmac_f32_e32 v95, v19, v22
	v_cvt_f32_u32_e32 v19, v64
	v_add_co_u32 v84, vcc_lo, v84, s27
	v_fmac_f32_e32 v17, v16, v23
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v95, v21, v18
	v_add_co_ci_u32_e64 v85, null, s26, v85, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, s34, v105
	v_add_co_u32 v86, s22, 0x2800, v86
	v_fmac_f32_e32 v95, v17, v19
	v_add_co_ci_u32_e64 v87, null, 0, v87, s22
	s_or_b32 s25, vcc_lo, s25
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s25
	s_cbranch_execz .LBB15_64
.LBB15_5:                               ; =>This Inner Loop Header: Depth=1
	s_and_b32 s22, s47, exec_lo
	s_cselect_b32 s22, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s22, 1
	s_cbranch_scc1 .LBB15_7
; %bb.6:                                ;   in Loop: Header=BB15_5 Depth=1
	v_dual_mov_b32 v56, v84 :: v_dual_mov_b32 v57, v85
	s_branch .LBB15_8
.LBB15_7:                               ;   in Loop: Header=BB15_5 Depth=1
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v57, 0
.LBB15_8:                               ;   in Loop: Header=BB15_5 Depth=1
	s_clause 0x3
	global_load_b128 v[16:19], v[86:87], off offset:48
	global_load_b128 v[20:23], v[86:87], off offset:32
	global_load_b128 v[72:75], v[86:87], off offset:16
	global_load_b128 v[76:79], v[86:87], off
	s_mov_b32 s22, exec_lo
	v_cmpx_ne_u64_e32 0, v[56:57]
	s_cbranch_execz .LBB15_11
; %bb.9:                                ;   in Loop: Header=BB15_5 Depth=1
	v_add_co_u32 v52, vcc_lo, v56, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v53, null, 0, v57, vcc_lo
	s_and_b32 s23, s48, exec_lo
	s_cselect_b32 s23, 1, 0
	s_clause 0x3
	global_load_b128 v[40:43], v[52:53], off offset:48
	global_load_b128 v[44:47], v[52:53], off offset:32
	global_load_b128 v[48:51], v[52:53], off offset:16
	global_load_b128 v[52:55], v[52:53], off
	s_cmp_lg_u32 s23, 1
	s_cbranch_scc1 .LBB15_11
; %bb.10:                               ;   in Loop: Header=BB15_5 Depth=1
	s_clause 0x3
	global_load_b128 v[24:27], v[82:83], off offset:48
	global_load_b128 v[28:31], v[82:83], off offset:32
	global_load_b128 v[32:35], v[82:83], off offset:16
	global_load_b128 v[36:39], v[82:83], off
.LBB15_11:                              ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	v_add_nc_u32_e32 v56, s46, v105
	s_and_b32 s22, s47, exec_lo
	s_cselect_b32 s22, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_cmp_lg_u32 s22, 1
	v_ashrrev_i32_e32 v57, 31, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b64 v[58:59], 11, v[56:57]
	v_lshlrev_b64 v[56:57], 8, v[56:57]
	v_add_co_u32 v60, vcc_lo, v96, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v61, null, v97, v59, vcc_lo
	v_add_co_u32 v88, vcc_lo, v98, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v89, null, v99, v57, vcc_lo
	s_clause 0x3
	global_load_b128 v[68:71], v[60:61], off
	global_load_b128 v[64:67], v[60:61], off offset:256
	global_load_b128 v[56:59], v[60:61], off offset:512
	global_load_b128 v[60:63], v[60:61], off offset:768
	global_load_b64 v[88:89], v[88:89], off
	s_cbranch_scc1 .LBB15_31
; %bb.12:                               ;   in Loop: Header=BB15_5 Depth=1
	s_and_b32 s22, s48, exec_lo
	s_cselect_b32 s22, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s22, 1
	s_cbranch_scc1 .LBB15_14
; %bb.13:                               ;   in Loop: Header=BB15_5 Depth=1
	s_waitcnt vmcnt(7)
	v_mul_f32_e32 v106, v47, v47
	v_mul_f32_e32 v47, v47, v31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v106, v43, v43
	v_mul_f32_e32 v43, v43, v27
	s_waitcnt vmcnt(6)
	v_fmac_f32_e32 v106, v51, v51
	v_mul_f32_e32 v51, v51, v35
	s_waitcnt vmcnt(5)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v106, v55, v55
	v_dual_mul_f32 v55, v55, v39 :: v_dual_fmac_f32 v106, v45, v45
	v_mul_f32_e32 v45, v45, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v106, v41, v41
	v_mul_f32_e32 v41, v41, v25
	v_fmac_f32_e32 v106, v49, v49
	v_mul_f32_e32 v49, v49, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v106, v53, v53
	v_dual_mul_f32 v53, v53, v37 :: v_dual_fmac_f32 v106, v46, v46
	v_mul_f32_e32 v46, v46, v30
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v106, v42, v42
	v_mul_f32_e32 v42, v42, v26
	v_fmac_f32_e32 v106, v50, v50
	v_mul_f32_e32 v50, v50, v34
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v106, v54, v54
	v_mul_f32_e32 v54, v54, v38
	v_fmac_f32_e32 v106, v44, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v106, v40, v40
	v_fmac_f32_e32 v106, v48, v48
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v106, v52, v52
	v_mul_f32_e32 v52, v52, v36
	v_add_f32_dpp v106, v106, v106 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_dpp v106, v106, v106 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v106, v106, v106 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v106, 0x3c000000, v106, s35
	v_mul_f32_e32 v107, 0x4b800000, v106
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v106
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v106, v106, v107, vcc_lo
	v_rsq_f32_e32 v106, v106
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v48, v48, v32 :: v_dual_mul_f32 v107, 0x45800000, v106
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v106, v106, v107, vcc_lo
	v_mul_f32_e32 v44, v44, v28
	v_dual_mul_f32 v40, v40, v24 :: v_dual_mul_f32 v53, v53, v106
	v_mul_f32_e32 v55, v55, v106
	v_mul_f32_e32 v54, v54, v106
	v_mul_f32_e32 v52, v52, v106
	v_mul_f32_e32 v51, v51, v106
	v_mul_f32_e32 v50, v50, v106
	v_mul_f32_e32 v49, v49, v106
	v_mul_f32_e32 v48, v48, v106
	v_mul_f32_e32 v47, v47, v106
	v_mul_f32_e32 v46, v46, v106
	v_mul_f32_e32 v45, v45, v106
	v_mul_f32_e32 v44, v44, v106
	v_mul_f32_e32 v43, v43, v106
	v_mul_f32_e32 v42, v42, v106
	v_mul_f32_e32 v41, v41, v106
	v_mul_f32_e32 v40, v40, v106
.LBB15_14:                              ;   in Loop: Header=BB15_5 Depth=1
	s_and_b32 vcc_lo, exec_lo, s28
	s_mov_b32 s22, -1
                                        ; implicit-def: $vgpr106
                                        ; implicit-def: $vgpr107
                                        ; implicit-def: $vgpr108
                                        ; implicit-def: $vgpr109
	s_cbranch_vccz .LBB15_16
; %bb.15:                               ;   in Loop: Header=BB15_5 Depth=1
	s_waitcnt vmcnt(5)
	v_dual_sub_f32 v106, 0x42800000, v76 :: v_dual_sub_f32 v107, 0x42800000, v77
	v_dual_sub_f32 v108, 0x42800000, v78 :: v_dual_sub_f32 v109, 0x42800000, v79
	v_cmp_lt_f32_e32 vcc_lo, 0x42aeac50, v76
	v_cmp_lt_f32_e64 s22, 0x42aeac50, v77
	v_cmp_lt_f32_e64 s23, 0x42aeac50, v78
	v_cmp_lt_f32_e64 s24, 0x42aeac50, v79
	v_cndmask_b32_e64 v106, -v76, v106, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v107, -v77, v107, s22
	v_cndmask_b32_e64 v108, -v78, v108, s23
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v109, -v79, v109, s24
	v_dual_mul_f32 v106, 0x3fb8aa3b, v106 :: v_dual_mul_f32 v107, 0x3fb8aa3b, v107
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v108, 0x3fb8aa3b, v108 :: v_dual_mul_f32 v109, 0x3fb8aa3b, v109
	v_exp_f32_e32 v106, v106
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v107, v107
	v_exp_f32_e32 v108, v108
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_3)
	v_exp_f32_e32 v109, v109
	v_dual_mul_f32 v110, 0x114b4ea4, v106 :: v_dual_mul_f32 v111, 0x114b4ea4, v107
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v112, 0x114b4ea4, v108 :: v_dual_mul_f32 v113, 0x114b4ea4, v109
	v_cndmask_b32_e64 v107, v107, v111, s22
	v_cndmask_b32_e32 v106, v106, v110, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v108, v108, v112, s23
	v_cndmask_b32_e64 v109, v109, v113, s24
	v_mul_f32_e32 v111, v53, v77
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v107, 1.0, v107 :: v_dual_add_f32 v106, 1.0, v106
	v_dual_add_f32 v108, 1.0, v108 :: v_dual_add_f32 v109, 1.0, v109
	v_dual_mul_f32 v110, v52, v76 :: v_dual_mul_f32 v113, v55, v79
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v106, v106
	v_rcp_f32_e32 v107, v107
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(TRANS32_DEP_3)
	v_rcp_f32_e32 v108, v108
	v_rcp_f32_e32 v109, v109
	v_mul_f32_e32 v112, v54, v78
	s_mov_b32 s22, 0
	v_dual_mul_f32 v106, v110, v106 :: v_dual_mul_f32 v107, v111, v107
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v108, v112, v108 :: v_dual_mul_f32 v109, v113, v109
	s_and_b32 s22, s22, exec_lo
	s_cselect_b32 s22, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s22, 1
	s_cbranch_scc0 .LBB15_17
	s_branch .LBB15_18
.LBB15_16:                              ;   in Loop: Header=BB15_5 Depth=1
	s_and_b32 s22, s22, exec_lo
	s_cselect_b32 s22, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s22, 1
	s_cbranch_scc1 .LBB15_18
.LBB15_17:                              ;   in Loop: Header=BB15_5 Depth=1
	s_waitcnt vmcnt(5)
	v_dual_sub_f32 v106, 0x42800000, v52 :: v_dual_sub_f32 v107, 0x42800000, v53
	v_cmp_lt_f32_e32 vcc_lo, 0x42aeac50, v52
	v_cmp_lt_f32_e64 s22, 0x42aeac50, v53
	v_dual_sub_f32 v108, 0x42800000, v54 :: v_dual_sub_f32 v109, 0x42800000, v55
	v_cmp_lt_f32_e64 s23, 0x42aeac50, v54
	v_cndmask_b32_e64 v106, -v52, v106, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v107, -v53, v107, s22
	v_cmp_lt_f32_e64 s24, 0x42aeac50, v55
	v_cndmask_b32_e64 v108, -v54, v108, s23
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v106, 0x3fb8aa3b, v106 :: v_dual_mul_f32 v107, 0x3fb8aa3b, v107
	v_cndmask_b32_e64 v109, -v55, v109, s24
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v108, 0x3fb8aa3b, v108
	v_exp_f32_e32 v106, v106
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v107, v107
	v_mul_f32_e32 v109, 0x3fb8aa3b, v109
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v108, v108
	v_exp_f32_e32 v109, v109
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v110, 0x114b4ea4, v106 :: v_dual_mul_f32 v111, 0x114b4ea4, v107
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v112, 0x114b4ea4, v108
	v_cndmask_b32_e64 v107, v107, v111, s22
	v_dual_mul_f32 v113, 0x114b4ea4, v109 :: v_dual_cndmask_b32 v106, v106, v110
	v_cndmask_b32_e64 v108, v108, v112, s23
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v107, 1.0, v107
	v_cndmask_b32_e64 v109, v109, v113, s24
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v106, 1.0, v106
	v_rcp_f32_e32 v107, v107
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v108, 1.0, v108 :: v_dual_add_f32 v109, 1.0, v109
	v_rcp_f32_e32 v106, v106
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_3)
	v_rcp_f32_e32 v109, v109
	v_mul_f32_e32 v107, v77, v107
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v108, v108
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v106, v76, v106 :: v_dual_mul_f32 v109, v79, v109
	v_mul_f32_e32 v108, v78, v108
.LBB15_18:                              ;   in Loop: Header=BB15_5 Depth=1
	s_and_b32 s22, exec_lo, s28
	s_cselect_b32 s22, 1, 0
                                        ; implicit-def: $vgpr77
                                        ; implicit-def: $vgpr78
                                        ; implicit-def: $vgpr76
                                        ; implicit-def: $vgpr79
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s22, 1
	s_mov_b32 s22, -1
	s_cbranch_scc1 .LBB15_20
; %bb.19:                               ;   in Loop: Header=BB15_5 Depth=1
	s_waitcnt vmcnt(5)
	v_dual_sub_f32 v76, 0x42800000, v72 :: v_dual_sub_f32 v77, 0x42800000, v73
	v_dual_sub_f32 v78, 0x42800000, v74 :: v_dual_sub_f32 v79, 0x42800000, v75
	v_cmp_lt_f32_e32 vcc_lo, 0x42aeac50, v72
	v_cmp_lt_f32_e64 s22, 0x42aeac50, v73
	v_cmp_lt_f32_e64 s23, 0x42aeac50, v74
	v_cmp_lt_f32_e64 s24, 0x42aeac50, v75
	v_cndmask_b32_e64 v76, -v72, v76, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v77, -v73, v77, s22
	v_cndmask_b32_e64 v78, -v74, v78, s23
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v79, -v75, v79, s24
	v_dual_mul_f32 v76, 0x3fb8aa3b, v76 :: v_dual_mul_f32 v77, 0x3fb8aa3b, v77
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v78, 0x3fb8aa3b, v78 :: v_dual_mul_f32 v79, 0x3fb8aa3b, v79
	v_exp_f32_e32 v76, v76
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v77, v77
	v_exp_f32_e32 v78, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_3)
	v_exp_f32_e32 v79, v79
	v_dual_mul_f32 v110, 0x114b4ea4, v76 :: v_dual_mul_f32 v111, 0x114b4ea4, v77
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v112, 0x114b4ea4, v78 :: v_dual_mul_f32 v113, 0x114b4ea4, v79
	v_cndmask_b32_e64 v77, v77, v111, s22
	v_cndmask_b32_e32 v76, v76, v110, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v78, v78, v112, s23
	v_cndmask_b32_e64 v79, v79, v113, s24
	s_mov_b32 s22, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v77, 1.0, v77 :: v_dual_add_f32 v76, 1.0, v76
	v_dual_add_f32 v78, 1.0, v78 :: v_dual_add_f32 v79, 1.0, v79
	v_mul_f32_e32 v113, v51, v75
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v110, v77
	v_rcp_f32_e32 v76, v76
	v_mul_f32_e32 v77, v48, v72
	v_rcp_f32_e32 v111, v78
	v_rcp_f32_e32 v79, v79
	v_mul_f32_e32 v78, v49, v73
	v_mul_f32_e32 v112, v50, v74
	s_delay_alu instid0(TRANS32_DEP_3) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v78, v78, v110 :: v_dual_mul_f32 v77, v77, v76
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v76, v112, v111
	v_mul_f32_e32 v79, v113, v79
.LBB15_20:                              ;   in Loop: Header=BB15_5 Depth=1
	s_and_b32 s22, s22, exec_lo
	s_cselect_b32 s22, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s22, 1
	s_cbranch_scc1 .LBB15_22
; %bb.21:                               ;   in Loop: Header=BB15_5 Depth=1
	s_waitcnt vmcnt(5)
	v_dual_sub_f32 v76, 0x42800000, v48 :: v_dual_sub_f32 v77, 0x42800000, v49
	v_cmp_lt_f32_e32 vcc_lo, 0x42aeac50, v48
	v_cmp_lt_f32_e64 s22, 0x42aeac50, v49
	v_dual_sub_f32 v78, 0x42800000, v50 :: v_dual_sub_f32 v79, 0x42800000, v51
	v_cmp_lt_f32_e64 s23, 0x42aeac50, v50
	v_cndmask_b32_e64 v76, -v48, v76, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v77, -v49, v77, s22
	v_cmp_lt_f32_e64 s24, 0x42aeac50, v51
	v_cndmask_b32_e64 v78, -v50, v78, s23
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v76, 0x3fb8aa3b, v76 :: v_dual_mul_f32 v77, 0x3fb8aa3b, v77
	v_cndmask_b32_e64 v79, -v51, v79, s24
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v78, 0x3fb8aa3b, v78
	v_exp_f32_e32 v76, v76
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v77, v77
	v_mul_f32_e32 v79, 0x3fb8aa3b, v79
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v78, v78
	v_exp_f32_e32 v79, v79
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v110, 0x114b4ea4, v76 :: v_dual_mul_f32 v111, 0x114b4ea4, v77
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v112, 0x114b4ea4, v78
	v_cndmask_b32_e64 v77, v77, v111, s22
	v_dual_mul_f32 v113, 0x114b4ea4, v79 :: v_dual_cndmask_b32 v76, v76, v110
	v_cndmask_b32_e64 v78, v78, v112, s23
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v77, 1.0, v77
	v_cndmask_b32_e64 v79, v79, v113, s24
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v76, 1.0, v76
	v_rcp_f32_e32 v110, v77
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v79, 1.0, v79
	v_rcp_f32_e32 v76, v76
	v_add_f32_e32 v78, 1.0, v78
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v79, v79
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v77, v72, v76
	v_rcp_f32_e32 v111, v78
	v_dual_mul_f32 v78, v73, v110 :: v_dual_mul_f32 v79, v75, v79
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v76, v74, v111
.LBB15_22:                              ;   in Loop: Header=BB15_5 Depth=1
	s_and_b32 s22, exec_lo, s28
	s_cselect_b32 s22, 1, 0
                                        ; implicit-def: $vgpr72
                                        ; implicit-def: $vgpr73
                                        ; implicit-def: $vgpr75
                                        ; implicit-def: $vgpr74
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s22, 1
	s_mov_b32 s22, -1
	s_cbranch_scc1 .LBB15_24
; %bb.23:                               ;   in Loop: Header=BB15_5 Depth=1
	s_waitcnt vmcnt(6)
	v_dual_sub_f32 v72, 0x42800000, v20 :: v_dual_sub_f32 v73, 0x42800000, v21
	v_dual_sub_f32 v74, 0x42800000, v22 :: v_dual_sub_f32 v75, 0x42800000, v23
	v_cmp_lt_f32_e32 vcc_lo, 0x42aeac50, v20
	v_cmp_lt_f32_e64 s22, 0x42aeac50, v21
	v_cmp_lt_f32_e64 s23, 0x42aeac50, v22
	v_cmp_lt_f32_e64 s24, 0x42aeac50, v23
	v_cndmask_b32_e64 v72, -v20, v72, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v73, -v21, v73, s22
	v_cndmask_b32_e64 v74, -v22, v74, s23
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v75, -v23, v75, s24
	v_dual_mul_f32 v72, 0x3fb8aa3b, v72 :: v_dual_mul_f32 v73, 0x3fb8aa3b, v73
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v74, 0x3fb8aa3b, v74 :: v_dual_mul_f32 v75, 0x3fb8aa3b, v75
	v_exp_f32_e32 v72, v72
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v73, v73
	v_exp_f32_e32 v74, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_3)
	v_exp_f32_e32 v75, v75
	v_dual_mul_f32 v110, 0x114b4ea4, v72 :: v_dual_mul_f32 v111, 0x114b4ea4, v73
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v112, 0x114b4ea4, v74 :: v_dual_mul_f32 v113, 0x114b4ea4, v75
	v_cndmask_b32_e64 v73, v73, v111, s22
	v_cndmask_b32_e32 v72, v72, v110, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v74, v74, v112, s23
	v_cndmask_b32_e64 v75, v75, v113, s24
	v_dual_mul_f32 v111, v45, v21 :: v_dual_mul_f32 v112, v46, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v73, 1.0, v73 :: v_dual_add_f32 v72, 1.0, v72
	v_dual_add_f32 v75, 1.0, v75 :: v_dual_add_f32 v74, 1.0, v74
	v_mul_f32_e32 v113, v47, v23
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v73, v73
	v_rcp_f32_e32 v72, v72
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v110, v75
	v_mul_f32_e32 v75, v44, v20
	v_rcp_f32_e32 v74, v74
	s_mov_b32 s22, 0
	v_mul_f32_e32 v73, v111, v73
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v72, v75, v72 :: v_dual_mul_f32 v75, v112, v74
	v_mul_f32_e32 v74, v113, v110
.LBB15_24:                              ;   in Loop: Header=BB15_5 Depth=1
	s_and_b32 s22, s22, exec_lo
	s_cselect_b32 s22, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s22, 1
	s_cbranch_scc1 .LBB15_26
; %bb.25:                               ;   in Loop: Header=BB15_5 Depth=1
	s_waitcnt vmcnt(6)
	v_dual_sub_f32 v72, 0x42800000, v44 :: v_dual_sub_f32 v73, 0x42800000, v45
	v_cmp_lt_f32_e32 vcc_lo, 0x42aeac50, v44
	v_cmp_lt_f32_e64 s22, 0x42aeac50, v45
	v_dual_sub_f32 v74, 0x42800000, v46 :: v_dual_sub_f32 v75, 0x42800000, v47
	v_cmp_lt_f32_e64 s23, 0x42aeac50, v46
	v_cndmask_b32_e64 v72, -v44, v72, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v73, -v45, v73, s22
	v_cmp_lt_f32_e64 s24, 0x42aeac50, v47
	v_cndmask_b32_e64 v74, -v46, v74, s23
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v72, 0x3fb8aa3b, v72 :: v_dual_mul_f32 v73, 0x3fb8aa3b, v73
	v_cndmask_b32_e64 v75, -v47, v75, s24
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v74, 0x3fb8aa3b, v74
	v_exp_f32_e32 v72, v72
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v73, v73
	v_mul_f32_e32 v75, 0x3fb8aa3b, v75
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v74, v74
	v_exp_f32_e32 v75, v75
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v110, 0x114b4ea4, v72 :: v_dual_mul_f32 v111, 0x114b4ea4, v73
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v112, 0x114b4ea4, v74
	v_cndmask_b32_e64 v73, v73, v111, s22
	v_dual_mul_f32 v113, 0x114b4ea4, v75 :: v_dual_cndmask_b32 v72, v72, v110
	v_cndmask_b32_e64 v74, v74, v112, s23
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v73, 1.0, v73
	v_cndmask_b32_e64 v75, v75, v113, s24
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v72, 1.0, v72
	v_rcp_f32_e32 v73, v73
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v74, 1.0, v74 :: v_dual_add_f32 v75, 1.0, v75
	v_rcp_f32_e32 v72, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_3)
	v_rcp_f32_e32 v110, v75
	v_mul_f32_e32 v73, v21, v73
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v74, v74
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v72, v20, v72 :: v_dual_mul_f32 v75, v22, v74
	v_mul_f32_e32 v74, v23, v110
.LBB15_26:                              ;   in Loop: Header=BB15_5 Depth=1
	s_and_b32 s22, exec_lo, s28
	s_cselect_b32 s22, 1, 0
                                        ; implicit-def: $vgpr20
                                        ; implicit-def: $vgpr21
                                        ; implicit-def: $vgpr22
                                        ; implicit-def: $vgpr23
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s22, 1
	s_mov_b32 s22, -1
	s_cbranch_scc1 .LBB15_28
; %bb.27:                               ;   in Loop: Header=BB15_5 Depth=1
	s_waitcnt vmcnt(7)
	v_dual_sub_f32 v20, 0x42800000, v16 :: v_dual_sub_f32 v21, 0x42800000, v17
	v_dual_sub_f32 v22, 0x42800000, v18 :: v_dual_sub_f32 v23, 0x42800000, v19
	v_cmp_lt_f32_e32 vcc_lo, 0x42aeac50, v16
	v_cmp_lt_f32_e64 s22, 0x42aeac50, v17
	v_cmp_lt_f32_e64 s23, 0x42aeac50, v18
	v_cmp_lt_f32_e64 s24, 0x42aeac50, v19
	v_cndmask_b32_e64 v20, -v16, v20, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v21, -v17, v21, s22
	v_cndmask_b32_e64 v22, -v18, v22, s23
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v23, -v19, v23, s24
	v_dual_mul_f32 v20, 0x3fb8aa3b, v20 :: v_dual_mul_f32 v21, 0x3fb8aa3b, v21
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v22, 0x3fb8aa3b, v22 :: v_dual_mul_f32 v23, 0x3fb8aa3b, v23
	v_exp_f32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v21, v21
	v_exp_f32_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_3)
	v_exp_f32_e32 v23, v23
	v_dual_mul_f32 v110, 0x114b4ea4, v20 :: v_dual_mul_f32 v111, 0x114b4ea4, v21
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v112, 0x114b4ea4, v22 :: v_dual_mul_f32 v113, 0x114b4ea4, v23
	v_cndmask_b32_e64 v21, v21, v111, s22
	v_cndmask_b32_e32 v20, v20, v110, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v22, v22, v112, s23
	v_cndmask_b32_e64 v23, v23, v113, s24
	v_mul_f32_e32 v111, v41, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v21, 1.0, v21 :: v_dual_add_f32 v20, 1.0, v20
	v_dual_add_f32 v22, 1.0, v22 :: v_dual_add_f32 v23, 1.0, v23
	v_dual_mul_f32 v110, v40, v16 :: v_dual_mul_f32 v113, v43, v19
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v20, v20
	v_rcp_f32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(TRANS32_DEP_3)
	v_rcp_f32_e32 v22, v22
	v_rcp_f32_e32 v23, v23
	v_mul_f32_e32 v112, v42, v18
	s_mov_b32 s22, 0
	v_dual_mul_f32 v20, v110, v20 :: v_dual_mul_f32 v21, v111, v21
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v22, v112, v22 :: v_dual_mul_f32 v23, v113, v23
.LBB15_28:                              ;   in Loop: Header=BB15_5 Depth=1
	s_and_b32 s22, s22, exec_lo
	s_cselect_b32 s22, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s22, 1
	s_cbranch_scc1 .LBB15_30
; %bb.29:                               ;   in Loop: Header=BB15_5 Depth=1
	s_waitcnt vmcnt(7)
	v_dual_sub_f32 v20, 0x42800000, v40 :: v_dual_sub_f32 v21, 0x42800000, v41
	v_cmp_lt_f32_e32 vcc_lo, 0x42aeac50, v40
	v_cmp_lt_f32_e64 s22, 0x42aeac50, v41
	v_dual_sub_f32 v22, 0x42800000, v42 :: v_dual_sub_f32 v23, 0x42800000, v43
	v_cmp_lt_f32_e64 s23, 0x42aeac50, v42
	v_cndmask_b32_e64 v20, -v40, v20, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v21, -v41, v21, s22
	v_cmp_lt_f32_e64 s24, 0x42aeac50, v43
	v_cndmask_b32_e64 v22, -v42, v22, s23
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v20, 0x3fb8aa3b, v20 :: v_dual_mul_f32 v21, 0x3fb8aa3b, v21
	v_cndmask_b32_e64 v23, -v43, v23, s24
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v22, 0x3fb8aa3b, v22
	v_exp_f32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v21, v21
	v_mul_f32_e32 v23, 0x3fb8aa3b, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v110, 0x114b4ea4, v20 :: v_dual_mul_f32 v111, 0x114b4ea4, v21
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v112, 0x114b4ea4, v22
	v_cndmask_b32_e64 v21, v21, v111, s22
	v_dual_mul_f32 v113, 0x114b4ea4, v23 :: v_dual_cndmask_b32 v20, v20, v110
	v_cndmask_b32_e64 v22, v22, v112, s23
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v21, 1.0, v21
	v_cndmask_b32_e64 v23, v23, v113, s24
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v20, 1.0, v20
	v_rcp_f32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v22, 1.0, v22 :: v_dual_add_f32 v23, 1.0, v23
	v_rcp_f32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_3)
	v_rcp_f32_e32 v23, v23
	v_mul_f32_e32 v21, v17, v21
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v22, v22
	s_waitcnt_depctr depctr_va_vdst(0)
	v_dual_mul_f32 v20, v16, v20 :: v_dual_mul_f32 v23, v19, v23
	v_mul_f32_e32 v22, v18, v22
.LBB15_30:                              ;   in Loop: Header=BB15_5 Depth=1
	s_waitcnt vmcnt(7)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v16, v20 :: v_dual_mov_b32 v17, v21
	v_dual_mov_b32 v18, v22 :: v_dual_mov_b32 v19, v23
	s_branch .LBB15_32
.LBB15_31:                              ;   in Loop: Header=BB15_5 Depth=1
	s_waitcnt vmcnt(5)
	v_dual_mov_b32 v106, v76 :: v_dual_mov_b32 v107, v77
	v_dual_mov_b32 v108, v78 :: v_dual_mov_b32 v109, v79
	v_dual_mov_b32 v77, v72 :: v_dual_mov_b32 v78, v73
	v_dual_mov_b32 v76, v74 :: v_dual_mov_b32 v79, v75
	v_dual_mov_b32 v72, v20 :: v_dual_mov_b32 v73, v21
	v_dual_mov_b32 v75, v22 :: v_dual_mov_b32 v74, v23
.LBB15_32:                              ;   in Loop: Header=BB15_5 Depth=1
	v_cndmask_b32_e64 v20, v106, -v106, s4
	v_cndmask_b32_e64 v21, v107, -v107, s5
	v_cndmask_b32_e64 v22, v108, -v108, s6
	v_cndmask_b32_e64 v23, v109, -v109, s7
	s_waitcnt vmcnt(5)
	v_cndmask_b32_e64 v77, v77, -v77, s8
	v_cvt_f16_f32_e32 v106.l, v20
	v_cndmask_b32_e64 v20, v78, -v78, s9
	v_cvt_f16_f32_e32 v106.h, v21
	v_cvt_f16_f32_e32 v107.l, v22
	v_cvt_f16_f32_e32 v107.h, v23
	v_cndmask_b32_e64 v21, v79, -v79, s11
	v_cvt_f16_f32_e32 v108.h, v20
	v_cndmask_b32_e64 v20, v76, -v76, s10
	v_cndmask_b32_e64 v22, v72, -v72, s12
	v_cndmask_b32_e64 v23, v73, -v73, s13
	v_cndmask_b32_e64 v72, v75, -v75, s14
	v_cndmask_b32_e64 v16, v16, -v16, s16
	v_cvt_f16_f32_e32 v109.l, v20
	v_cndmask_b32_e64 v20, v74, -v74, s15
	v_cndmask_b32_e64 v17, v17, -v17, s17
	v_cndmask_b32_e64 v18, v18, -v18, s18
	v_cndmask_b32_e64 v19, v19, -v19, s19
	v_cvt_f16_f32_e32 v108.l, v77
	v_cvt_f16_f32_e32 v109.h, v21
	v_cvt_f16_f32_e32 v110.l, v22
	v_cvt_f16_f32_e32 v110.h, v23
	v_cvt_f16_f32_e32 v111.l, v72
	v_cvt_f16_f32_e32 v111.h, v20
	v_cvt_f16_f32_e32 v112.l, v16
	v_cvt_f16_f32_e32 v112.h, v17
	v_cvt_f16_f32_e32 v113.l, v18
	v_cvt_f16_f32_e32 v113.h, v19
	v_dual_mov_b32 v79, s43 :: v_dual_mov_b32 v78, s42
	v_dual_mov_b32 v77, s41 :: v_dual_mov_b32 v76, s40
	v_dual_mov_b32 v75, s39 :: v_dual_mov_b32 v74, s38
	v_dual_mov_b32 v73, s37 :: v_dual_mov_b32 v72, s36
	s_mov_b32 s22, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[16:23], v[106:113], v[0:7], v[72:79]
	v_cvt_f16_f32_e32 v16.l, v16
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v16.h, v17
	v_mov_b32_e32 v17, 0
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	v_mov_b32_e32 v19, 0
	v_cvt_f16_f32_e32 v20.l, v20
	v_cvt_f16_f32_e32 v20.h, v21
	v_mov_b32_e32 v21, 0
	v_cvt_f16_f32_e32 v22.l, v22
	v_cvt_f16_f32_e32 v22.h, v23
	v_mov_b32_e32 v23, 0
	v_permlanex16_b32 v17, v16, s29, 0xfedcba98
	v_permlanex16_b32 v19, v18, s29, 0xfedcba98
	v_permlanex16_b32 v21, v20, s29, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_permlanex16_b32 v23, v22, s29, 0xfedcba98
	v_cndmask_b32_e64 v106, v17, v16, s20
	v_cndmask_b32_e64 v110, v16, v17, s20
	v_cndmask_b32_e64 v107, v19, v18, s20
	v_cndmask_b32_e64 v111, v18, v19, s20
	v_cndmask_b32_e64 v108, v21, v20, s20
	v_cndmask_b32_e64 v112, v20, v21, s20
	v_cndmask_b32_e64 v109, v23, v22, s20
	v_cndmask_b32_e64 v113, v22, v23, s20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[16:23], v[8:15], v[106:113], v[72:79]
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v74, 0
	v_mul_f32_e32 v16, 0x3d800000, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v72, 0x7fffffff, v16
	v_mov_b32_dpp v73, v72 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_max_f32 v72, v73, v73 :: v_dual_mov_b32 v73, 0
	v_max_f32_e64 v72, |v16|, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_dpp v73, v72 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	v_max_f32_e32 v73, v73, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v72, v72, v73
	v_mov_b32_dpp v74, v72 row_half_mirror row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_max_f32 v73, v74, v74 :: v_dual_mov_b32 v74, 0
	v_max_f32_e32 v72, v72, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_dpp v74, v72 row_mirror row_mask:0xf bank_mask:0xf
	v_dual_max_f32 v73, v74, v74 :: v_dual_mov_b32 v74, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v72, v72, v73
	v_permlanex16_b32 v74, v72, s29, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v73, v74, v74
	v_dual_max_f32 v74, v72, v73 :: v_dual_mov_b32 v73, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v72, 0x3c010204, v74
	v_cmpx_neq_f32_e32 0, v74
	s_cbranch_execz .LBB15_34
; %bb.33:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v73, v72
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v74, v16, v73
	v_trunc_f32_e32 v75, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v16, v16, v73, -v75
	v_cmp_ge_f32_e64 s23, |v16|, 0.5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v16, 0, 1.0, s23
	v_bfi_b32 v16, 0x7fffffff, v16, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v16, v75, v16
	v_cvt_i32_f32_e32 v73, v16
.LBB15_34:                              ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v16, v73
	v_mov_b32_e32 v74, 0
	ds_store_b8 v104, v73
	v_add_f32_dpp v16, v16, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v16, v16 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_dpp v16, v16, v16 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v16, v16 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v74, v16, s29, 0xfedcba98
	s_and_saveexec_b32 s22, s21
; %bb.35:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v74
	v_mul_f32_e32 v73, v16, v72
	ds_store_b64 v100, v[72:73]
; %bb.36:                               ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	v_dual_mul_f32 v17, 0x3d800000, v17 :: v_dual_mov_b32 v16, 0
	s_mov_b32 s22, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v73, 0 :: v_dual_and_b32 v72, 0x7fffffff, v17
	v_mov_b32_dpp v16, v72 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
	v_mov_b32_e32 v72, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v16
	v_max_f32_e64 v16, |v17|, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_dpp v72, v16 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	v_max_f32_e32 v72, v72, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v72
	v_mov_b32_dpp v73, v16 row_half_mirror row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_max_f32 v72, v73, v73 :: v_dual_mov_b32 v73, 0
	v_max_f32_e32 v16, v16, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_dpp v73, v16 row_mirror row_mask:0xf bank_mask:0xf
	v_dual_max_f32 v72, v73, v73 :: v_dual_mov_b32 v73, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v72
	v_permlanex16_b32 v73, v16, s29, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v72, v73, v73
	v_dual_max_f32 v73, v16, v72 :: v_dual_mov_b32 v72, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v16, 0x3c010204, v73
	v_cmpx_neq_f32_e32 0, v73
	s_cbranch_execz .LBB15_38
; %bb.37:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v72, v16
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v73, v17, v72
	v_trunc_f32_e32 v74, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v17, v17, v72, -v74
	v_cmp_ge_f32_e64 s23, |v17|, 0.5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v17, 0, 1.0, s23
	v_bfi_b32 v17, 0x7fffffff, v17, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v74, v17
	v_cvt_i32_f32_e32 v72, v17
.LBB15_38:                              ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v17, v72
	v_mov_b32_e32 v73, 0
	ds_store_b8 v104, v72 offset:32
	v_add_f32_dpp v17, v17, v17 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v17, v17 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_dpp v17, v17, v17 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v17, v17 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v73, v17, s29, 0xfedcba98
	s_and_saveexec_b32 s22, s21
; %bb.39:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v73
	v_mul_f32_e32 v17, v17, v16
	ds_store_b64 v100, v[16:17] offset:8
; %bb.40:                               ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	v_dual_mul_f32 v17, 0x3d800000, v18 :: v_dual_mov_b32 v16, 0
	v_mov_b32_e32 v72, 0
	s_mov_b32 s22, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v18, 0x7fffffff, v17
	v_mov_b32_dpp v16, v18 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
	v_mov_b32_e32 v18, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v16
	v_max_f32_e64 v16, |v17|, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_dpp v18, v16 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	v_max_f32_e32 v18, v18, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v18
	v_mov_b32_dpp v72, v16 row_half_mirror row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_max_f32_e32 v18, v72, v72
	v_mov_b32_e32 v72, 0
	v_max_f32_e32 v16, v16, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_dpp v72, v16 row_mirror row_mask:0xf bank_mask:0xf
	v_max_f32_e32 v18, v72, v72
	v_mov_b32_e32 v72, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v18
	v_permlanex16_b32 v72, v16, s29, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v18, v72, v72
	v_max_f32_e32 v72, v16, v18
	v_mov_b32_e32 v18, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v16, 0x3c010204, v72
	v_cmpx_neq_f32_e32 0, v72
	s_cbranch_execz .LBB15_42
; %bb.41:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v18, v16
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v72, v17, v18
	v_trunc_f32_e32 v73, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v17, v17, v18, -v73
	v_cmp_ge_f32_e64 s23, |v17|, 0.5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v17, 0, 1.0, s23
	v_bfi_b32 v17, 0x7fffffff, v17, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v73, v17
	v_cvt_i32_f32_e32 v18, v17
.LBB15_42:                              ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v17, v18
	v_mov_b32_e32 v72, 0
	ds_store_b8 v104, v18 offset:64
	v_add_f32_dpp v17, v17, v17 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v17, v17 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_dpp v17, v17, v17 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v17, v17 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v72, v17, s29, 0xfedcba98
	s_and_saveexec_b32 s22, s21
; %bb.43:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v72
	v_mul_f32_e32 v17, v17, v16
	ds_store_b64 v100, v[16:17] offset:16
; %bb.44:                               ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	v_dual_mul_f32 v17, 0x3d800000, v19 :: v_dual_mov_b32 v16, 0
	s_mov_b32 s22, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v19, 0 :: v_dual_and_b32 v18, 0x7fffffff, v17
	v_mov_b32_dpp v16, v18 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
	v_mov_b32_e32 v18, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v16
	v_max_f32_e64 v16, |v17|, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_dpp v18, v16 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	v_max_f32_e32 v18, v18, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v18
	v_mov_b32_dpp v19, v16 row_half_mirror row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_max_f32 v18, v19, v19 :: v_dual_mov_b32 v19, 0
	v_max_f32_e32 v16, v16, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_dpp v19, v16 row_mirror row_mask:0xf bank_mask:0xf
	v_dual_max_f32 v18, v19, v19 :: v_dual_mov_b32 v19, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v18
	v_permlanex16_b32 v19, v16, s29, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v18, v19, v19
	v_dual_max_f32 v19, v16, v18 :: v_dual_mov_b32 v18, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v16, 0x3c010204, v19
	v_cmpx_neq_f32_e32 0, v19
	s_cbranch_execz .LBB15_46
; %bb.45:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v18, v16
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v19, v17, v18
	v_trunc_f32_e32 v72, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v17, v17, v18, -v72
	v_cmp_ge_f32_e64 s23, |v17|, 0.5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v17, 0, 1.0, s23
	v_bfi_b32 v17, 0x7fffffff, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v72, v17
	v_cvt_i32_f32_e32 v18, v17
.LBB15_46:                              ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v17, v18
	v_mov_b32_e32 v19, 0
	ds_store_b8 v104, v18 offset:96
	v_add_f32_dpp v17, v17, v17 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v17, v17 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_dpp v17, v17, v17 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v17, v17 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v19, v17, s29, 0xfedcba98
	s_and_saveexec_b32 s22, s21
; %bb.47:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v19
	v_mul_f32_e32 v17, v17, v16
	ds_store_b64 v100, v[16:17] offset:24
; %bb.48:                               ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	v_dual_mul_f32 v17, 0x3d800000, v20 :: v_dual_mov_b32 v16, 0
	s_mov_b32 s22, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v19, 0 :: v_dual_and_b32 v18, 0x7fffffff, v17
	v_mov_b32_dpp v16, v18 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
	v_mov_b32_e32 v18, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v16
	v_max_f32_e64 v16, |v17|, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_dpp v18, v16 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	v_max_f32_e32 v18, v18, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v18
	v_mov_b32_dpp v19, v16 row_half_mirror row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_max_f32 v18, v19, v19 :: v_dual_mov_b32 v19, 0
	v_max_f32_e32 v16, v16, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_dpp v19, v16 row_mirror row_mask:0xf bank_mask:0xf
	v_dual_max_f32 v18, v19, v19 :: v_dual_mov_b32 v19, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v18
	v_permlanex16_b32 v19, v16, s29, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v18, v19, v19
	v_dual_max_f32 v19, v16, v18 :: v_dual_mov_b32 v18, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v16, 0x3c010204, v19
	v_cmpx_neq_f32_e32 0, v19
	s_cbranch_execz .LBB15_50
; %bb.49:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v18, v16
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v19, v17, v18
	v_trunc_f32_e32 v20, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v17, v17, v18, -v20
	v_cmp_ge_f32_e64 s23, |v17|, 0.5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v17, 0, 1.0, s23
	v_bfi_b32 v17, 0x7fffffff, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v20, v17
	v_cvt_i32_f32_e32 v18, v17
.LBB15_50:                              ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v17, v18
	v_mov_b32_e32 v19, 0
	ds_store_b8 v104, v18 offset:128
	v_add_f32_dpp v17, v17, v17 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v17, v17 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_dpp v17, v17, v17 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v17, v17 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v19, v17, s29, 0xfedcba98
	s_and_saveexec_b32 s22, s21
; %bb.51:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v19
	v_mul_f32_e32 v17, v17, v16
	ds_store_b64 v100, v[16:17] offset:32
; %bb.52:                               ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	v_dual_mul_f32 v17, 0x3d800000, v21 :: v_dual_mov_b32 v16, 0
	s_mov_b32 s22, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v19, 0 :: v_dual_and_b32 v18, 0x7fffffff, v17
	v_mov_b32_dpp v16, v18 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
	v_mov_b32_e32 v18, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v16
	v_max_f32_e64 v16, |v17|, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_dpp v18, v16 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	v_max_f32_e32 v18, v18, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v18
	v_mov_b32_dpp v19, v16 row_half_mirror row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_max_f32 v18, v19, v19 :: v_dual_mov_b32 v19, 0
	v_max_f32_e32 v16, v16, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_dpp v19, v16 row_mirror row_mask:0xf bank_mask:0xf
	v_dual_max_f32 v18, v19, v19 :: v_dual_mov_b32 v19, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v18
	v_permlanex16_b32 v19, v16, s29, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v18, v19, v19
	v_dual_max_f32 v19, v16, v18 :: v_dual_mov_b32 v18, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v16, 0x3c010204, v19
	v_cmpx_neq_f32_e32 0, v19
	s_cbranch_execz .LBB15_54
; %bb.53:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v18, v16
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v19, v17, v18
	v_trunc_f32_e32 v20, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v17, v17, v18, -v20
	v_cmp_ge_f32_e64 s23, |v17|, 0.5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v17, 0, 1.0, s23
	v_bfi_b32 v17, 0x7fffffff, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v20, v17
	v_cvt_i32_f32_e32 v18, v17
.LBB15_54:                              ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v17, v18
	v_mov_b32_e32 v19, 0
	ds_store_b8 v104, v18 offset:160
	v_add_f32_dpp v17, v17, v17 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v17, v17 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_dpp v17, v17, v17 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v17, v17 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v19, v17, s29, 0xfedcba98
	s_and_saveexec_b32 s22, s21
; %bb.55:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v19
	v_mul_f32_e32 v17, v17, v16
	ds_store_b64 v100, v[16:17] offset:40
; %bb.56:                               ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	v_dual_mul_f32 v17, 0x3d800000, v22 :: v_dual_mov_b32 v16, 0
	s_mov_b32 s22, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v19, 0 :: v_dual_and_b32 v18, 0x7fffffff, v17
	v_mov_b32_dpp v16, v18 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
	v_mov_b32_e32 v18, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v16
	v_max_f32_e64 v16, |v17|, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_dpp v18, v16 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	v_max_f32_e32 v18, v18, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v18
	v_mov_b32_dpp v19, v16 row_half_mirror row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_max_f32 v18, v19, v19 :: v_dual_mov_b32 v19, 0
	v_max_f32_e32 v16, v16, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_dpp v19, v16 row_mirror row_mask:0xf bank_mask:0xf
	v_dual_max_f32 v18, v19, v19 :: v_dual_mov_b32 v19, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v18
	v_permlanex16_b32 v19, v16, s29, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v18, v19, v19
	v_dual_max_f32 v19, v16, v18 :: v_dual_mov_b32 v18, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v16, 0x3c010204, v19
	v_cmpx_neq_f32_e32 0, v19
	s_cbranch_execz .LBB15_58
; %bb.57:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v18, v16
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v19, v17, v18
	v_trunc_f32_e32 v20, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v17, v17, v18, -v20
	v_cmp_ge_f32_e64 s23, |v17|, 0.5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v17, 0, 1.0, s23
	v_bfi_b32 v17, 0x7fffffff, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v20, v17
	v_cvt_i32_f32_e32 v18, v17
.LBB15_58:                              ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v17, v18
	v_mov_b32_e32 v19, 0
	ds_store_b8 v104, v18 offset:192
	v_add_f32_dpp v17, v17, v17 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v17, v17 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_dpp v17, v17, v17 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v17, v17 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v19, v17, s29, 0xfedcba98
	s_and_saveexec_b32 s22, s21
; %bb.59:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v19
	v_mul_f32_e32 v17, v17, v16
	ds_store_b64 v100, v[16:17] offset:48
; %bb.60:                               ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	v_dual_mul_f32 v17, 0x3d800000, v23 :: v_dual_mov_b32 v16, 0
	s_mov_b32 s22, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v19, 0 :: v_dual_and_b32 v18, 0x7fffffff, v17
	v_mov_b32_dpp v16, v18 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
	v_mov_b32_e32 v18, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v16
	v_max_f32_e64 v16, |v17|, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_dpp v18, v16 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	v_max_f32_e32 v18, v18, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v18
	v_mov_b32_dpp v19, v16 row_half_mirror row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_max_f32 v18, v19, v19 :: v_dual_mov_b32 v19, 0
	v_max_f32_e32 v16, v16, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_dpp v19, v16 row_mirror row_mask:0xf bank_mask:0xf
	v_dual_max_f32 v18, v19, v19 :: v_dual_mov_b32 v19, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v16, v16, v18
	v_permlanex16_b32 v19, v16, s29, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_f32_e32 v18, v19, v19
	v_dual_max_f32 v19, v16, v18 :: v_dual_mov_b32 v18, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v16, 0x3c010204, v19
	v_cmpx_neq_f32_e32 0, v19
	s_cbranch_execz .LBB15_62
; %bb.61:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v18, v16
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v19, v17, v18
	v_trunc_f32_e32 v20, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v17, v17, v18, -v20
	v_cmp_ge_f32_e64 s23, |v17|, 0.5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v17, 0, 1.0, s23
	v_bfi_b32 v17, 0x7fffffff, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v20, v17
	v_cvt_i32_f32_e32 v18, v17
.LBB15_62:                              ;   in Loop: Header=BB15_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v17, v18
	v_mov_b32_e32 v19, 0
	ds_store_b8 v104, v18 offset:224
	v_add_f32_dpp v17, v17, v17 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v17, v17 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_dpp v17, v17, v17 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v17, v17 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v19, v17, s29, 0xfedcba98
	s_and_saveexec_b32 s22, s21
	s_cbranch_execz .LBB15_4
; %bb.63:                               ;   in Loop: Header=BB15_5 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v19
	v_mul_f32_e32 v17, v17, v16
	ds_store_b64 v100, v[16:17] offset:56
	s_branch .LBB15_4
.LBB15_64:
	s_or_b32 exec_lo, exec_lo, s25
.LBB15_65:
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	s_or_b32 exec_lo, exec_lo, s33
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, 0
	s_mov_b32 s4, 0x76543210
	v_lshlrev_b32_e32 v0, 2, v93
	v_permlanex16_b32 v2, v95, s4, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_3)
	v_permlanex16_b32 v1, v1, s4, 0xfedcba98
	s_and_saveexec_b32 s4, s3
; %bb.66:
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v3, v94, 7, v0
	v_dual_add_f32 v2, v95, v2 :: v_dual_add_nc_u32 v3, 0x800, v3
	ds_store_2addr_b32 v3, v2, v1 offset0:128 offset1:144
; %bb.67:
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v1, v92, v94
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_mov_b32 s3, exec_lo
	v_cmpx_eq_u32_e32 0, v1
	s_cbranch_execz .LBB15_69
; %bb.68:
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
	v_dual_add_f32 v0, v1, v0 :: v_dual_mul_f32 v1, s0, v90
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v2, v0
	v_add_f32_e32 v0, v3, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v2, v1, v0
	v_fmac_f32_e32 v91, v1, v0
	v_add_co_u32 v0, vcc_lo, s44, v80
	v_add_co_ci_u32_e64 v1, null, s45, v81, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v2, v2, v91, s2
	global_store_b32 v[0:1], v2, off
.LBB15_69:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
