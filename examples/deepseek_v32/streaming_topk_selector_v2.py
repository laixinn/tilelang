import torch
import tilelang
import tilelang.language as T

import os, sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from deepseek_v32.utils import generate_random_cu_seqlens, per_custom_dims_cast_to_fp8

def ref_fp8_mqa_logits(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor, cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor, topk: int):
    k = kv
    q = q.float()
    k = k.float()

    seq_len_kv = kv.shape[0]
    mask_lo = torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    mask_hi = torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    cost = mask.sum()
    topk_logits, topk_indices = logits.topk(topk)
    return logits, cost, topk_indices, topk_logits

def display_error_message(msg):
    print(f"\033[31mWARNING: {msg}\033[0m")

def compute_correlation(a, b, label="tensor"):
    a, b = a.data.double(), b.data.double()
    norm_sum = (a * a + b * b).sum()
    if norm_sum == 0:
        display_error_message(f"{label} all zero")
        return 1
    correlation = 2 * (a * b).sum() / norm_sum
    return correlation

def validate_tensor_match(a, b, tolerance=1e-8, tensor_name="tensor", should_raise=True):
    a_finite = torch.isfinite(a)
    b_finite = torch.isfinite(b)
    if not torch.all(a_finite == b_finite):
        display_error_message(f"{tensor_name} Error: isfinite mask mismatch")
        if should_raise:
            assert False
    if not torch.isclose(
        a.masked_fill(a_finite, 0),
        b.masked_fill(b_finite, 0),
        rtol=0,
        atol=0,
        equal_nan=True,
    ).all():
        display_error_message(f"{tensor_name} Error: nonfinite value mismatch")
        if should_raise:
            assert False
    a = a.masked_fill(~a_finite, 0)
    b = b.masked_fill(~b_finite, 0)
    correlation = compute_correlation(a, b, tensor_name)
    difference = 1.0 - correlation
    if not (0 <= difference <= tolerance):
        display_error_message(f"{tensor_name} Error: {difference}")
        if should_raise:
            assert False
    return difference

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True, # logits computation needs
}


def convert_to_uint16(x):
    hval = T.Cast(T.float16, x)
    bits_uint = T.reinterpret(T.uint16, hval)
    bits_uint = T.if_then_else(x < 0, ~bits_uint & (0xFFFF), bits_uint | (0x8000))
    return bits_uint >> 8


def convert_to_uint32(x):
    bits_uint = T.reinterpret(T.uint32, x)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.Cast(T.uint32, (0xFFFFFFFF)),
        bits_uint | T.Cast(T.uint32, (0x80000000)),
    )
    return bits_uint


@tilelang.jit(pass_configs=pass_configs)
def tl_topk_impl(
    heads,
    index_dim,
    topk,
    num_stages=2,
    threads=512,
    debug=False,
    dtype=T.float8_e4m3fn,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    RADIX = 1 << 8
    BLOCK_SIZE = threads
    SMEM_INPUT_SIZE = 4096  # assume the threshold bucket size after first pass is less than 4K

    # logits compute
    block_Q = 1 # restricted
    block_N = 256
    # dtype=T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    index_k_scale_shape = [seq_len_kv]
    logits_shape = [seq_len, topk]

    block_TOPK = topk + block_N

    @T.prim_func
    def tl_topk_kernel(
        # input: T.Tensor[(seq_len, seq_len_kv), accum_dtype],
        topk_index: T.Tensor[(seq_len, topk), index_dtype],
        topk_logits: T.Tensor[(seq_len, topk), accum_dtype],
        starts: T.Tensor[(seq_len), index_dtype],
        ends: T.Tensor[(seq_len), index_dtype],
        # logits compute
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        IndexKScale: T.Tensor(index_k_scale_shape, accum_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        LogitsIdx: T.Tensor(logits_shape, index_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as (bx):
            # logits compute
            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            index_k_scale_fragment = T.alloc_fragment([block_N], accum_dtype)
            s_shared = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.reshape(s_shared, (block_N, block_Q, heads))
            logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            cu_k_s_min = T.alloc_var(index_dtype)
            cu_k_e_max = T.alloc_var(index_dtype)

            cu_k_s_min = 2147483647
            cu_k_e_max = -2147483648

            for bq_i in T.serial(block_Q):
                cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv))
            for bq_i in T.serial(block_Q):
                cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv))

            s_threshold_bin_id = T.alloc_shared([1], T.int32)
            s_histogram = T.alloc_shared([2, RADIX + 1], T.int32)
            s_num_input = T.alloc_shared([2], T.int32)
            s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)

            l_threshold_bin_id = T.alloc_var(T.int32)
            l_new_topk = T.alloc_var(T.int32)
            l_num_input = T.alloc_var(T.int32)
            l_bin_id32 = T.alloc_var(T.int32)
            l_val = T.alloc_var(T.int32)
            l_start_pos = T.alloc_var(T.int32)
            l_out_pos = T.alloc_var(T.int32)

            l_new_topk = topk

            # sync
            tx = T.get_thread_binding()
            copy_done = T.alloc_barrier(arrive_count=512)
            gemm_done = T.alloc_barrier(arrive_count=512)

            T.fill(s_histogram[0, :], 0)
            T.fill(s_num_input[0], 0)

            nbn_i = T.alloc_var(T.int32)
            pos = T.alloc_var(T.int32)
            s_val = T.alloc_var(accum_dtype)
            s_idx = T.alloc_var(index_dtype)
            input_idx = T.alloc_var(T.int32)

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            # fill block_TOPK logits
            fill_size = T.min(topk, cu_k_e_max - cu_k_s_min)
            T.barrier_arrive(gemm_done)

            for nbn_i in T.serial(T.ceildiv(fill_size, block_N)):
                T.barrier_wait(gemm_done, nbn_i % 2)

                T.copy(IndexK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)
                T.copy(IndexKScale[cu_k_s_min + nbn_i * block_N], index_k_scale_fragment)

                if debug and bx == 0 and tx == 0:
                    T.print(nbn_i, "copy done")

                T.barrier_arrive(copy_done)
                T.barrier_wait(copy_done, nbn_i % 2)

                if debug and bx == 0 and tx == 0:
                    T.print(nbn_i, "start gemm")

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s_shared,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                if debug and bx == 0 and tx == 0:
                    T.print(nbn_i, "gemm done")

                T.barrier_arrive(gemm_done)

                if debug and bx == 0 and tx == 0:
                    T.print(nbn_i, "pass gemm barrier")

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (T.max(s_shared[bn_i, bq_i * heads + h_i], 0) * weights[bq_i, h_i]) * index_k_scale_fragment[
                        bn_i
                    ]

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                # update histogram for topk stage 1
                for s in T.Parallel(block_N):
                    input_idx = cu_k_s_min + nbn_i * block_N + s
                    if input_idx < cu_k_e_max and input_idx >= cu_k_s_min and s < block_N:
                        inval_int16 = convert_to_uint16(logits[s, 0])
                        T.atomic_add(s_histogram[0, inval_int16], 1)

                # store topk logits and index first
                for s in T.Parallel(block_N):
                    if input_idx < cu_k_e_max and input_idx >= cu_k_s_min and cu_k_s_min + nbn_i * block_N + s < fill_size:
                        Logits[bx, cu_k_s_min + nbn_i * block_N + s] = logits[s, 0]
                        LogitsIdx[bx, cu_k_s_min + nbn_i * block_N + s] = cu_k_s_min + nbn_i * block_N + s

            T.sync_threads(1, 512)

            if cu_k_e_max - cu_k_s_min > topk:
                # update topk for each logits block compute
                cu_k_s_min = cu_k_s_min + fill_size
                T.fill(s_histogram[1, :], 0)

                for nbn_i in T.serial(T.ceildiv(cu_k_e_max - cu_k_s_min, block_N)):
                    T.fill(s_num_input[0], 0)

                    # logits compute
                    T.barrier_wait(gemm_done, nbn_i % 2)

                    T.copy(IndexK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)
                    T.copy(IndexKScale[cu_k_s_min + nbn_i * block_N], index_k_scale_fragment)

                    if debug and bx == 0 and tx == 0:
                        T.print(nbn_i, "copy done")

                    T.barrier_arrive(copy_done)
                    T.barrier_wait(copy_done, nbn_i % 2)

                    if debug and bx == 0 and tx == 0:
                        T.print(nbn_i, "start gemm")

                    T.gemm(
                        index_k_shared,
                        index_q_shared,
                        s_shared,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    if debug and bx == 0 and tx == 0:
                        T.print(nbn_i, "gemm done")

                    T.barrier_arrive(gemm_done)

                    if debug and bx == 0 and tx == 0:
                        T.print(nbn_i, "pass gemm barrier")

                    for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                        s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i]) * index_k_scale_fragment[
                            bn_i
                        ]

                    T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                    # block_Q is restricted to 1
                    for s in T.Parallel(block_N):
                        input_idx = cu_k_s_min + nbn_i * block_N + s
                        if input_idx < cu_k_e_max and input_idx >= cu_k_s_min and s < block_N:
                            inval_int16 = convert_to_uint16(logits[s, 0])
                            T.atomic_add(s_histogram[nbn_i % 2, inval_int16], 1)

                    # maintain s_histogram for the next block
                    T.copy(s_histogram[nbn_i % 2, :], s_histogram[(nbn_i % 2) ^ 1, :])

                    # topk compute

                    # cumsum
                    s_threshold_bin_id[0] = -1
                    T.sync_threads(1, 512)
                    if tx < RADIX:
                        for i in T.serial(8):
                            offset = 1 << i
                            T.sync_threads(3, RADIX)
                            if tx < RADIX - offset:
                                l_val = s_histogram[nbn_i % 2, tx] + s_histogram[nbn_i % 2, tx + offset]
                            T.sync_threads(3, RADIX)
                            if tx < RADIX - offset:
                                s_histogram[nbn_i % 2, tx] = l_val

                        # find threshold bin id
                        T.sync_threads(3, RADIX)
                        if s_histogram[nbn_i % 2, tx] > l_new_topk and s_histogram[nbn_i % 2, tx + 1] <= l_new_topk:
                            s_threshold_bin_id[0] = tx
                    T.sync_threads(1, 512)
                    l_threshold_bin_id = s_threshold_bin_id[0]
                    l_new_topk = l_new_topk - s_histogram[nbn_i % 2, l_threshold_bin_id + 1]
                    T.sync_threads(1, 512)

                    if debug and bx == 0 and tx == 0 and l_threshold_bin_id < 0:
                        T.print(l_threshold_bin_id, "stage 1l_threshold_bin_id < 0")

                    if debug and bx == 0 and tx == 0:
                        T.print(l_new_topk, "l_new_topk 0")

                    # reset counter greater than topk to topk
                    s_histogram[(nbn_i % 2) ^ 1, l_threshold_bin_id] = topk - s_histogram[nbn_i % 2, l_threshold_bin_id + 1]
                    T.fill(s_histogram[(nbn_i % 2) ^ 1, 0 : l_threshold_bin_id], 0)
                    if debug and bx == 0 and tx == 0:
                        T.print(s_histogram[(nbn_i % 2) ^ 1, l_threshold_bin_id], "s_histogram[(nbn_i % 2) ^ 1, l_threshold_bin_id]")

                    # collect previous topk elements with exponent ≥ threshold
                    for s in T.serial(T.ceildiv(topk, BLOCK_SIZE)):
                        T.sync_threads(1, 512)
                        input_idx = s * BLOCK_SIZE + tx
                        if input_idx < topk:
                            bin_id = convert_to_uint16(Logits[bx, input_idx])
                            l_bin_id32 = T.Cast(T.int32, bin_id)
                            if l_bin_id32 > l_threshold_bin_id:
                                # need a pos = T.atomic_add(s_histogram[bin_id32+1], 1)
                                pos = T.atomic_add(s_histogram[nbn_i % 2, l_bin_id32 + 1], 1, return_prev=True)
                                topk_index[bx, pos] = LogitsIdx[bx, input_idx]
                                topk_logits[bx, pos] = Logits[bx, input_idx]

                            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                                s_input_idx[0, pos] = input_idx

                    # collect current block elements with exponent ≥ threshold
                    for s in T.Parallel(block_N):
                        input_idx = cu_k_s_min + nbn_i * block_N + s
                        if input_idx < cu_k_e_max and input_idx >= cu_k_s_min and s < block_N:
                            inval_int16 = convert_to_uint16(logits[s, 0])
                            l_bin_id32 = T.Cast(T.int32, inval_int16)
                            if l_bin_id32 > l_threshold_bin_id:
                                pos = T.atomic_add(s_histogram[nbn_i % 2, l_bin_id32 + 1], 1, return_prev=True)
                                topk_index[bx, pos] = input_idx
                                topk_logits[bx, pos] = logits[s, 0]

                            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                                s_input_idx[0, pos] = input_idx

                    # stage 2: tail pass
                    for round in T.serial(4):
                        if l_new_topk <= 0:
                            T.loop_break()

                        r_idx = round % 2
                        l_start_pos = topk - l_new_topk

                        T.sync_threads(1, 512)
                        T.fill(s_histogram[nbn_i % 2, :], 0)
                        if tx == 0:
                            s_num_input[r_idx ^ 1] = 0
                        T.sync_threads(1, 512)

                        if debug and bx == 0 and tx == 0:
                            T.print(s_num_input[r_idx], "s_num_input[r_idx]")

                        l_num_input = s_num_input[r_idx]
                        for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                            T.sync_threads(1, 512)
                            if s * BLOCK_SIZE + tx < l_num_input:
                                input_idx = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                                if input_idx < cu_k_s_min + nbn_i * block_N:
                                    s_val = Logits[bx, input_idx]
                                else:
                                    s_val = logits[input_idx - cu_k_s_min - nbn_i * block_N, 0]
                                l_bin_id32 = T.Cast(
                                    T.int32, ((convert_to_uint32(s_val) >> (24 - round * 8)) & 0xFF)
                                )
                                T.atomic_add(s_histogram[nbn_i % 2, l_bin_id32], 1)
                        T.sync_threads(1, 512)

                        # cumsum
                        s_threshold_bin_id[0] = -1
                        if tx < RADIX:
                            for i in T.serial(8):
                                offset = 1 << i
                                T.sync_threads(3, RADIX)
                                if tx < RADIX - offset:
                                    l_val = s_histogram[nbn_i % 2, tx] + s_histogram[nbn_i % 2, tx + offset]
                                T.sync_threads(3, RADIX)
                                if tx < RADIX - offset:
                                    s_histogram[nbn_i % 2, tx] = l_val

                            # find threshold bin id
                            T.sync_threads(3, RADIX)
                            if s_histogram[nbn_i % 2, tx] > l_new_topk and s_histogram[nbn_i % 2, tx + 1] <= l_new_topk:
                                s_threshold_bin_id[0] = tx
                        T.sync_threads(1, 512)

                        l_threshold_bin_id = s_threshold_bin_id[0]
                        l_new_topk = l_new_topk - s_histogram[nbn_i % 2, l_threshold_bin_id + 1]
                        T.sync_threads(1, 512)

                        for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                            T.sync_threads(1, 512)
                            if s * BLOCK_SIZE + tx < l_num_input:
                                input_idx = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                                if input_idx < cu_k_s_min + nbn_i * block_N:
                                    s_val = Logits[bx, input_idx]
                                    s_idx = LogitsIdx[bx, input_idx]
                                else:
                                    s_val = logits[input_idx - cu_k_s_min - nbn_i * block_N, 0]
                                    s_idx = input_idx

                                l_bin_id32 = T.Cast(
                                    T.int32, ((convert_to_uint32(s_val) >> (24 - round * 8)) & 0xFF)
                                )

                                if l_bin_id32 > l_threshold_bin_id:
                                    pos = T.atomic_add(s_histogram[nbn_i % 2, l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                    topk_logits[bx, pos] = s_val
                                    topk_index[bx, pos] = s_idx
                                elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                    if round == 3:
                                        l_out_pos = T.atomic_add(s_histogram[nbn_i % 2, l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                        if l_out_pos < topk:
                                            topk_logits[bx, l_out_pos] = s_val
                                            topk_index[bx, l_out_pos] = s_idx
                                    else:
                                        pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                        s_input_idx[r_idx ^ 1, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]



                    # dump topk to Logits
                    T.copy(topk_index[bx, :], LogitsIdx[bx, :])
                    T.copy(topk_logits[bx, :], Logits[bx, :])
            else:
                T.copy(LogitsIdx[bx, :], topk_index[bx, :])
                T.copy(Logits[bx, :], topk_logits[bx, :])


    return tl_topk_kernel


def tl_topk(
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke,
    starts, ends, topk, debug=False, kv_scales=None, input=None,
):
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]

    topk_indexes = torch.zeros(seq_len, topk, device=q.device, dtype=torch.int32)
    topk_logits = torch.empty([seq_len, topk], device=q.device, dtype=torch.float32)
    logits = torch.empty([seq_len, topk], device=q.device, dtype=torch.float32)
    logits_idx = torch.empty([seq_len, topk], device=q.device, dtype=torch.int32)

    if kv_scales is None:
        kv_scales = torch.ones(seq_len_kv, device=q.device, dtype=torch.float32)

    kernel = tl_topk_impl(heads=heads, index_dim=index_dim, topk=topk, debug=debug, dtype=q.dtype)
    kernel(
        # input, 
        topk_indexes, 
        topk_logits,
        starts, 
        ends,
        # logits
        q.view(seq_len * heads, index_dim),
        kv,
        kv_scales,
        logits,
        logits_idx,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
    return topk_indexes, topk_logits, logits


def test_topk_selector(S=4096, SKV=16384, H=32, HKV=1, D=64, kv_stride=1, topk=2048, test_accuracy=True):
    torch.manual_seed(0)
    input = torch.randn(S, SKV, dtype=torch.float32).cuda()
    starts = torch.zeros(S, dtype=torch.int32).cuda()
    ends = torch.ones(S, dtype=torch.int32).cuda() * SKV

    q = torch.randn(S, H, D, device="cuda", dtype=torch.bfloat16).to(torch.bfloat16)
    kv = torch.randn(SKV, D, device="cuda", dtype=torch.bfloat16).to(torch.bfloat16)
    weights = torch.randn(S, H, device="cuda", dtype=torch.float32)
    p = (torch.randn(S, SKV, device="cuda", dtype=torch.float32) * 4).softmax(dim=-1)

    # TODO: use mask
    # ks, ke = generate_random_cu_seqlens(per_cp_seqlen=S, cp_size=4, cp_rank=3, kv_stride=kv_stride, average_q_len=2048)
    ks, ke = starts, ends

    print(f"{ks=}, {ke=}")

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_scales = per_custom_dims_cast_to_fp8(kv, (0,), False)

    # topk_indexes, topk_logits, original_logits = tl_topk(
    #     q=q_fp8, kv=kv_fp8, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke,
    #     starts=starts, ends=ends, topk=topk,
    #     debug=True, kv_scales=kv_scales, 
    # )
    topk_indexes, topk_logits, original_logits = tl_topk(
        q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke,
        starts=starts, ends=ends, topk=topk,
        debug=True, 
    )

    torch.cuda.synchronize()

    logits_ref, cost_ref, topk_indices_ref, topk_logits_ref = ref_fp8_mqa_logits(q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke, topk=topk)

    torch.cuda.synchronize()
    
    print(topk_indexes)

    print(topk_indices_ref)

    # # torch.testing.assert_close(original_logits, logits_ref)
    # acc_diff = validate_tensor_match(original_logits, logits_ref, tolerance=1e-14, tensor_name="logits", should_raise=False)
    # print(f"accuracy difference: {acc_diff}")

    # indexes_ref = fast_topk(input, topk)
    # print(indexes_ref)

    # Calculate intersection of out_ref and out_trt
    if test_accuracy:
        for i in range(S):
            ref_np = topk_indices_ref[i][torch.isfinite(topk_logits_ref[i])].cpu().to(torch.int32).numpy()

            trt_np = topk_indexes[i][( topk_indexes[i] >= 0) & (topk_indexes[i] < SKV)].cpu().to(torch.int32).numpy()

            # gt_len = (ke[i] - ks[i]).item()
            gt_len = ref_np.shape[0]

            set_ref = set(ref_np)
            set_trt = set(trt_np)
            intersection = set_ref & set_trt
            # print(set_ref - set_trt)
            print("selected/all:", len(intersection), "/", gt_len, "=", len(intersection) / gt_len)

    # Performance test with CUDA events

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        _ = tl_topk(
            q=q_fp8, kv=kv_fp8, kv_scales=kv_scales, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke,
            input=logits_ref, starts=starts, ends=ends, topk=topk
        )
        _ = ref_fp8_mqa_logits(q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke, topk=topk)
    torch.cuda.synchronize()

    n_iters = 20
    start_event.record()
    for _ in range(n_iters):
        _ = tl_topk(
            q=q_fp8, kv=kv_fp8, kv_scales=kv_scales, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke,
            input=logits_ref, starts=starts, ends=ends, topk=topk
        )
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Average tl_topk time: {elapsed_time_ms / n_iters:.3f} ms")

    # Torch topk time
    start_event.record()
    for _ in range(n_iters):
        # _ = torch.topk(input, topk, dim=-1)[1]
        _ = ref_fp8_mqa_logits(q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke, topk=topk)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Average torch.topk time: {elapsed_time_ms / n_iters:.3f} ms")


def run_regression_perf(batch=64, seq_len=32 * 1024, topk=2048):
    batch = 64
    seq_len = 32 * 1024
    topk = 2048
    torch.manual_seed(1)
    input = torch.randn(batch, seq_len, dtype=torch.float32).cuda()
    starts = torch.zeros(batch, dtype=torch.int32).cuda()
    ends = torch.ones(batch, dtype=torch.int32).cuda() * seq_len

    indexes = tl_topk(input, starts, ends, topk)

    indexes_ref = torch.topk(input, topk, dim=-1)[1]

    for i in range(batch):
        ref_np = indexes_ref[i].cpu().to(torch.int32).numpy()
        trt_np = indexes[i].cpu().to(torch.int32).numpy()

        set_ref = set(ref_np)
        set_trt = set(trt_np)
        intersection = set_ref & set_trt
        print("selected/all:", len(intersection), "/", len(set_ref), "=", len(intersection) / len(set_ref))

    from tilelang.profiler import do_bench

    def run_kernel_only():
        tl_topk(input, starts, ends, topk)

    return do_bench(run_kernel_only, warmup=10, rep=100, backend="cupti")


if __name__ == "__main__":
    test_topk_selector(S=2048+256, SKV=2048+256)
    # test_topk_selector(S=16384, SKV=16384)