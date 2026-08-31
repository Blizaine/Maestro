# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from importlib.metadata import version
from mmgp import offload
import torch.nn.functional as F
import warnings
from importlib.metadata import version

major, minor = torch.cuda.get_device_capability(None)
bfloat16_supported =  major >= 8 

try:
    import triton
    triton_installed = True
except:
    triton_installed = False


try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except (ImportError, OSError):
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except (ImportError, OSError):
    FLASH_ATTN_2_AVAILABLE = False
    flash_attn = None

try:
    from sageattention import sageattn_varlen
    def sageattn_varlen_wrapper(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        ):
        return sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
    
except ImportError:
    sageattn_varlen_wrapper = None

try:
    from spas_sage_attn import block_sparse_sage2_attn_cuda
except ImportError:
    block_sparse_sage2_attn_cuda = None
    if not triton_installed: 
        try:
            sg2_version = version("sageattention")
            print("Sage Attention has been detected but it won't work until Triton is installed.")
        except ImportError:
            pass

try:
    from .sage2_core import sageattn as sageattn2, is_sage2_supported
    sage2_supported =  is_sage2_supported()
except ImportError:
    sageattn2 = None
    sage2_supported = False
    if not triton_installed: 
        try:
            sg2_version = version("sageattention")
            if not triton_installed: print("Sage Attention 2 has been detected but it won't work until Triton is installed.")
        except ImportError:
            pass

@torch.compiler.disable()
def sageattn2_wrapper(
        qkv_list,
        attention_length,
        recycle_q = False,
    ):
    q,k, v = qkv_list
    qkv_list = [q,k,v]
    del q, k ,v
    o = sageattn2(qkv_list, tensor_layout="NHD", recycle_q = recycle_q)
    qkv_list.clear()

    return o

try:
    from sageattn import sageattn_blackwell as sageattn3
    if not triton_installed:
        print("Sage Attention 3 is installed but it won't be supported until Triton is installed.")
except ImportError:
    sageattn3 = None
    if not triton_installed: 
        try:
            sg3_version = version("sageattn_blackwell")
            print("Sage Attention 3 has been detected but it won't work until Triton is installed.")
        except ImportError:
            pass

if sageattn3 is None:
    try:
        from sageattn3 import sageattn3_blackwell as sageattn3 #word0 windows version
    except ImportError:
        sageattn3 = None
        if not triton_installed: 
            try:
                sg3_version = version("sageattn3_blackwell")
                print("Sage Attention 3 has been detected but it won't work until Triton is installed.")
            except ImportError:
                pass

@torch.compiler.disable()
def sageattn3_wrapper(
        qkv_list,
        attention_length
    ):
    q,k, v = qkv_list
    # qkv_list = [q,k,v]
    # del q, k ,v
    # o = sageattn3(qkv_list, tensor_layout="NHD")
    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)
    o = sageattn3(q, k, v)
    o = o.transpose(1,2)
    qkv_list.clear()

    return o

     


# try:
# if True:
    # from .sage2_core import sageattn_qk_int8_pv_fp8_window_cuda
    # @torch.compiler.disable()
    # def sageattn_window_wrapper(
    #         qkv_list,
    #         attention_length,
    #         window
    #     ):
    #     q,k, v = qkv_list
    #     padding_length = q.shape[0] -attention_length
    #     q = q[:attention_length, :, : ].unsqueeze(0)
    #     k = k[:attention_length, :, : ].unsqueeze(0)
    #     v = v[:attention_length, :, : ].unsqueeze(0)
    #     qkvl_list = [q, k , v]
    #     del q, k ,v
    #     o = sageattn_qk_int8_pv_fp8_window_cuda(qkvl_list, tensor_layout="NHD", window = window).squeeze(0)
    #     qkv_list.clear()

    #     if padding_length > 0:
    #         o = torch.cat([o, torch.empty( (padding_length, *o.shape[-2:]), dtype= o.dtype, device=o.device  ) ], 0)

    #     return o
# except ImportError:
#     sageattn2 = sageattn_qk_int8_pv_fp8_window_cuda

@torch.compiler.disable()
def sdpa_wrapper(
        qkv_list,
        attention_length,
        attention_mask = None,
        causal = False,
    ):
    q, k, v = qkv_list

    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)
    if attention_mask is not None:
        # PyTorch SDPA accepts a boolean mask, a float32 mask, or a floating
        # mask matching the query dtype. Quantized/offloaded LTX layers can
        # promote Q/K/V to FP32 while their reference-conditioning mask stays
        # BF16, so normalize that otherwise-invalid combination at the final
        # attention boundary. Keep float32 masks untouched because SDPA
        # supports them natively and they may contain weighted attention bias.
        if (
            torch.is_floating_point(attention_mask)
            and attention_mask.dtype not in (torch.float32, q.dtype)
        ):
            attention_mask = attention_mask.to(
                device=q.device,
                dtype=q.dtype,
            )
        attention_mask = attention_mask.transpose(1,2)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=causal).transpose(1,2)
    del q, k ,v
    qkv_list.clear()

    return o


_flash_attn_kernels = None


def flash_attn_kernels_available():
    """Return whether the imported FlashAttention wheel runs on this GPU.

    Importing the Python extension only validates its ABI. Architecture-
    specific wheels can still import on a GPU for which they contain no CUDA
    kernel, failing later with ``no kernel image is available``. Probe one
    minimal varlen call and cache the result so every model path shares the
    same runtime verdict.
    """

    global _flash_attn_kernels
    if _flash_attn_kernels is not None:
        return _flash_attn_kernels
    if flash_attn is None or not torch.cuda.is_available():
        _flash_attn_kernels = False
        return False
    try:
        probe = torch.zeros(
            1,
            1,
            32,
            dtype=torch.float16,
            device="cuda",
        )
        cu_seqlens = torch.tensor(
            [0, 1],
            dtype=torch.int32,
            device="cuda",
        )
        flash_attn.flash_attn_varlen_func(
            probe,
            probe,
            probe,
            cu_seqlens,
            cu_seqlens,
            1,
            1,
        )
        torch.cuda.synchronize()
        _flash_attn_kernels = True
    except Exception as exc:
        _flash_attn_kernels = False
        print(
            "[Runtime] FlashAttention imports but has no usable kernel for "
            f"this GPU; using SageAttention/SDPA instead ({exc})."
        )
    return _flash_attn_kernels


def get_attention_modes():
    ret = ["sdpa", "auto"]
    if flash_attn != None:
        ret.append("flash")
    if memory_efficient_attention != None:
        ret.append("xformers")
    if sageattn_varlen_wrapper != None:
        ret.append("sage")
    if sageattn2 != None and version("sageattention").startswith("2") :
        ret.append("sage2")
    if block_sparse_sage2_attn_cuda != None and version("sageattention").startswith("2") :
        ret.append("radial")

    if sageattn3 != None: # and version("sageattention").startswith("3") :
        ret.append("sage3")
        
    return ret

def get_supported_attention_modes():
    ret = get_attention_modes()
    major, minor = torch.cuda.get_device_capability()
    if  major < 10 or not triton_installed:
        if "sage3" in ret:
            ret.remove("sage3")

    if not sage2_supported or not triton_installed:
        if "sage2" in ret:
            ret.remove("sage2")
        if "radial" in ret:
            ret.remove("radial")

    if major < 7 or not triton_installed:
        if "sage" in ret:
            ret.remove("sage")

    if "flash" in ret and not flash_attn_kernels_available():
        ret.remove("flash")

    return ret


SOL_ATTENTION_CAPABILITIES = ((8, 9), (9, 0), (10, 0), (12, 0))
SOL_ATTENTION_MIN_TRITON = (3, 6)


def _triton_version_tuple():
    if not triton_installed:
        return ()
    raw = str(getattr(triton, "__version__", "0"))
    parts = []
    for value in raw.split(".")[:2]:
        digits = "".join(character for character in value if character.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


def get_sol_attention_status():
    """Describe whether the bundled H3-only Sol backend can run here."""

    capability = tuple(torch.cuda.get_device_capability())
    triton_version = str(getattr(triton, "__version__", "")) if triton_installed else None
    if not triton_installed:
        reason = "Sol Engine requires Triton 3.6 or newer."
        installed = supported = False
    elif _triton_version_tuple() < SOL_ATTENTION_MIN_TRITON:
        reason = (
            "Sol Engine requires Triton 3.6 or newer "
            f"(this runtime has {triton_version})."
        )
        installed = supported = False
    elif capability not in SOL_ATTENTION_CAPABILITIES:
        supported_names = ", ".join(
            f"SM{item[0]}{item[1]}" for item in SOL_ATTENTION_CAPABILITIES
        )
        reason = (
            f"Sol Engine supports {supported_names}; this GPU is "
            f"SM{capability[0]}{capability[1]}."
        )
        installed, supported = True, False
    else:
        reason = None
        installed = supported = True
    return {
        "installed": installed,
        "supported": supported,
        "reason": reason,
        "capability": f"SM{capability[0]}{capability[1]}",
        "triton_version": triton_version,
        "minimum_triton": "3.6",
        "first_run_compiles_kernels": True,
    }


def get_override_attention_modes():
    """Generation overrides include model-specific backends such as Sol."""

    modes = get_attention_modes()
    if get_sol_attention_status()["installed"]:
        modes.append("sol")
    return modes


def get_supported_override_attention_modes():
    modes = get_supported_attention_modes()
    if get_sol_attention_status()["supported"]:
        modes.append("sol")
    return modes


def get_default_attention_mode():
    for mode in ("sage2", "sage", "sdpa"):
        if mode in get_supported_attention_modes():
            return mode
    return "sdpa"

__all__ = [
    'pay_attention',
    'attention',
    'get_attention_modes',
    'get_supported_attention_modes',
    'get_override_attention_modes',
    'get_supported_override_attention_modes',
    'get_sol_attention_status',
    'get_default_attention_mode',
]

def get_cu_seqlens(batch_size, lens, max_len):
    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = lens[i] 
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens

@torch.compiler.disable()
def pay_attention(
    qkv_list,
    dropout_p=0.,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    version=None,
    force_attention= None,
    attention_mask = None,
    recycle_q = False,
    q_lens = None,
    k_lens = None,
):
    # format : torch.Size([batches, tokens, heads, head_features])
    # assume if q_lens is non null, each q is padded up to lq (one q out of two will need to be discarded or ignored)
    # assume if k_lens is non null, each k is padded up to lk (one k out of two will need to be discarded or ignored)
    if attention_mask != None:
        force_attention = "sdpa"
        if  attention_mask.dtype == torch.bfloat16 and not bfloat16_supported:
            attention_mask = attention_mask.to(torch.float16)
    attn = offload.shared_state["_attention"] if force_attention== None else force_attention

    # Sol is a MiniMax-H3 main-DiT override, not a general attention mode.
    # H3's short token-refiner calls and any fail-safe recovery come through
    # this shared function and should use the fastest supported dense backend.
    if attn == "sol":
        attn = get_default_attention_mode()

    q,k,v = qkv_list
    qkv_list.clear()
    out_dtype = q.dtype
    if q.dtype == torch.bfloat16 and not bfloat16_supported:
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
    final_padding = 0
    b, lq, lk = q.size(0), q.size(1), k.size(1)

    q = q.to(v.dtype)
    k = k.to(v.dtype)
    batch = len(q)
    if len(k) != batch: k = k.expand(batch, -1, -1, -1)
    if len(v) != batch: v = v.expand(batch, -1, -1, -1)
    if attn == "chipmunk":
        from src.chipmunk.modules import SparseDiffMlp, SparseDiffAttn
        from src.chipmunk.util import LayerCounter, GLOBAL_CONFIG
    if attn == "radial": attn ="sage2"

    if b > 1 and k_lens != None and attn in ("sage2", "sage3", "sdpa"):
        assert attention_mask == None
        # Poor's man var k len attention
        assert q_lens == None
        chunk_sizes = []
        k_sizes = []
        current_size = k_lens[0]
        current_count= 1
        for k_len in k_lens[1:]:
            if k_len == current_size:
                current_count += 1
            else:
                chunk_sizes.append(current_count)
                k_sizes.append(current_size)
                current_count = 1
                current_size = k_len
        chunk_sizes.append(current_count)
        k_sizes.append(k_len)
        if len(chunk_sizes) > 1 or k_lens[0] != k.shape[1]:
            q_chunks =torch.split(q, chunk_sizes)
            k_chunks =torch.split(k, chunk_sizes)
            v_chunks =torch.split(v, chunk_sizes)
            q, k, v = None, None, None
            k_chunks = [ u[:, :sz] for u, sz in zip(k_chunks, k_sizes)]
            v_chunks = [ u[:, :sz] for u, sz in zip(v_chunks, k_sizes)]
            o = []
            for sub_q, sub_k, sub_v in zip(q_chunks, k_chunks, v_chunks): 
                qkv_list = [sub_q, sub_k, sub_v]
                sub_q, sub_k, sub_v = None, None, None
                o.append( pay_attention(qkv_list) )
            q_chunks, k_chunks, v_chunks = None, None, None
            o = torch.cat(o, dim = 0)
            return o
    elif (q_lens != None or k_lens != None) and attn in ("sage2", "sage3", "sdpa"):
        assert b == 1
        szq = q_lens[0].item() if q_lens != None else lq
        szk = k_lens[0].item() if k_lens != None else lk
        final_padding = lq - szq
        q = q[:, :szq]
        k = k[:, :szk]
        v = v[:, :szk]

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    if attn=="sage" or attn=="flash":
        if b != 1 :
            if k_lens == None:
                k_lens = torch.tensor( [lk] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
            if q_lens == None:
                q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
            k = k.reshape(-1, *k.shape[-2:])
            v = v.reshape(-1, *v.shape[-2:])
            q = q.reshape(-1, *q.shape[-2:])
            cu_seqlens_q=get_cu_seqlens(b, q_lens, lq) 
            cu_seqlens_k=get_cu_seqlens(b, k_lens, lk) 
        else:
            szq = q_lens[0].item() if q_lens != None else lq
            szk = k_lens[0].item() if k_lens != None else lk
            if szq != lq or szk != lk:
                cu_seqlens_q = torch.tensor([0, szq, lq], dtype=torch.int32, device="cuda")
                cu_seqlens_k = torch.tensor([0, szk, lk], dtype=torch.int32, device="cuda")
            else:
                cu_seqlens_q = torch.tensor([0, lq], dtype=torch.int32, device="cuda")
                cu_seqlens_k = torch.tensor([0, lk], dtype=torch.int32, device="cuda")
            q = q.squeeze(0)
            k = k.squeeze(0)
            v = v.squeeze(0)


    # apply attention
    if attn=="sage":
        x = sageattn_varlen_wrapper(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q= cu_seqlens_q,
            cu_seqlens_kv= cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_kv=lk,
        ).unflatten(0, (b, lq))
    elif attn=="sage3":
        qkv_list = [q,k,v]
        del q,k,v
        x = sageattn3_wrapper(qkv_list, lq)
    elif attn=="sage2":
        qkv_list = [q,k,v]
        del q,k,v
        x = sageattn2_wrapper(qkv_list, lq, recycle_q = recycle_q)
    elif attn=="sdpa":
        qkv_list = [q, k, v]
        del q ,k ,v
        x = sdpa_wrapper(qkv_list, lq, attention_mask=attention_mask, causal=causal) #.unsqueeze(0)
    elif attn=="flash" and version == 3:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q= cu_seqlens_q,
            cu_seqlens_k= cu_seqlens_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    elif attn=="flash":
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q= cu_seqlens_q,
            cu_seqlens_k= cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output

    elif attn=="xformers":
        from xformers.ops.fmha.attn_bias import BlockDiagonalPaddedKeysMask
        if k_lens == None and q_lens == None:
            x = memory_efficient_attention(q, k, v )
        elif k_lens != None and q_lens == None:
            attn_mask = BlockDiagonalPaddedKeysMask.from_seqlens([lq] * b , lk , list(k_lens) ) 
            x = memory_efficient_attention(q, k, v, attn_bias= attn_mask )
        elif b == 1:
            szq = q_lens[0].item() if q_lens != None else lq
            szk = k_lens[0].item() if k_lens != None else lk
            attn_mask = BlockDiagonalPaddedKeysMask.from_seqlens([szq, lq - szq ] , lk , [szk, 0] ) 
            x = memory_efficient_attention(q, k, v, attn_bias= attn_mask )
        else:
            assert False
    x = x.type(out_dtype)
    if final_padding > 0:
        x = torch.cat([x, torch.empty( (x.shape[0], final_padding, *x.shape[-2:]), dtype= x.dtype, device=x.device  ) ], 1)


    return x 
