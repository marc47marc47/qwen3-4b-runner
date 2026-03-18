use anyhow::{Result, bail};
use candle_core::quantized::{QMatMul, gguf_file};
use candle_core::{D, DType, Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Module, kv_cache::ConcatKvCache};
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;
use std::collections::HashMap;
use std::io::{Read, Seek};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizedArchitecture {
    Qwen3,
    Qwen3_5,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    FullAttention,
    StateSpace,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct MetadataSummary {
    pub architecture: QuantizedArchitecture,
    pub metadata_prefix: Option<String>,
    pub general_architecture: Option<String>,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_freq_base: f64,
}

#[derive(Debug, Clone)]
pub struct TensorNameMap {
    pub token_embeddings: String,
    pub output_norm: String,
    pub output: String,
    pub layers: Vec<LayerTensorNames>,
}

#[derive(Debug, Clone)]
pub struct LayerTensorNames {
    pub layer_idx: usize,
    pub kind: LayerKind,
    #[allow(dead_code)]
    pub prefix: String,
    pub attn_norm: String,
    pub ffn_norm: String,
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: String,
    pub o_proj: String,
    pub q_norm: Option<String>,
    pub k_norm: Option<String>,
    pub q_bias: Option<String>,
    pub k_bias: Option<String>,
    pub v_bias: Option<String>,
    pub ffn_gate: String,
    pub ffn_up: String,
    pub ffn_down: String,
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> CandleResult<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct MlpWeights {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl MlpWeights {
    fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        names: &LayerTensorNames,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: QMatMul::from_qtensor(ct.tensor(reader, &names.ffn_gate, device)?)?,
            up_proj: QMatMul::from_qtensor(ct.tensor(reader, &names.ffn_up, device)?)?,
            down_proj: QMatMul::from_qtensor(ct.tensor(reader, &names.ffn_down, device)?)?,
        })
    }
}

impl Module for MlpWeights {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        Ok(self.down_proj.forward(&(gate * up)?)?)
    }
}

#[derive(Debug, Clone)]
struct AttentionWeights {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    q_bias: Option<Tensor>,
    k_bias: Option<Tensor>,
    v_bias: Option<Tensor>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    q_out_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: ConcatKvCache,
}

impl AttentionWeights {
    fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        names: &LayerTensorNames,
        summary: &MetadataSummary,
        rotary_emb: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let q_norm = match &names.q_norm {
            Some(name) => Some(RmsNorm::from_qtensor(
                ct.tensor(reader, name, device)?,
                summary.rms_norm_eps,
            )?),
            None => None,
        };
        let k_norm = match &names.k_norm {
            Some(name) => Some(RmsNorm::from_qtensor(
                ct.tensor(reader, name, device)?,
                summary.rms_norm_eps,
            )?),
            None => None,
        };
        let q_bias = match &names.q_bias {
            Some(name) => Some(ct.tensor(reader, name, device)?.dequantize(device)?),
            None => None,
        };
        let k_bias = match &names.k_bias {
            Some(name) => Some(ct.tensor(reader, name, device)?.dequantize(device)?),
            None => None,
        };
        let v_bias = match &names.v_bias {
            Some(name) => Some(ct.tensor(reader, name, device)?.dequantize(device)?),
            None => None,
        };
        let q_out_dim = ct
            .tensor_infos
            .get(&names.q_proj)
            .ok_or_else(|| anyhow::anyhow!("missing tensor info for {}", names.q_proj))?
            .shape
            .dims2()?
            .0;

        Ok(Self {
            q_proj: QMatMul::from_qtensor(ct.tensor(reader, &names.q_proj, device)?)?,
            k_proj: QMatMul::from_qtensor(ct.tensor(reader, &names.k_proj, device)?)?,
            v_proj: QMatMul::from_qtensor(ct.tensor(reader, &names.v_proj, device)?)?,
            o_proj: QMatMul::from_qtensor(ct.tensor(reader, &names.o_proj, device)?)?,
            q_norm,
            k_norm,
            q_bias,
            k_bias,
            v_bias,
            num_heads: summary.num_attention_heads,
            num_kv_heads: summary.num_kv_heads,
            num_kv_groups: summary.num_attention_heads / summary.num_kv_heads,
            head_dim: summary.head_dim,
            q_out_dim,
            rotary_emb,
            kv_cache: ConcatKvCache::new(2),
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
    ) -> CandleResult<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        let mut q = self.q_proj.forward(x)?;
        let mut k = self.k_proj.forward(x)?;
        let mut v = self.v_proj.forward(x)?;

        if let Some(bias) = &self.q_bias {
            q = q.broadcast_add(bias)?;
        }
        if let Some(bias) = &self.k_bias {
            k = k.broadcast_add(bias)?;
        }
        if let Some(bias) = &self.v_bias {
            v = v.broadcast_add(bias)?;
        }

        let mut attn_gate = None;
        let q = if self.q_out_dim == self.num_heads * self.head_dim * 2 {
            let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim * 2))?;
            let q_states = q.narrow(D::Minus1, 0, self.head_dim)?;
            let gate = q.narrow(D::Minus1, self.head_dim, self.head_dim)?;
            attn_gate = Some(gate);
            let q_states = match &self.q_norm {
                Some(norm) => norm.forward(&q_states.flatten(0, 2)?)?.reshape((
                    batch,
                    seq_len,
                    self.num_heads,
                    self.head_dim,
                ))?,
                None => q_states,
            };
            q_states.transpose(1, 2)?.contiguous()?
        } else {
            let q = q
                .reshape((batch, seq_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            match &self.q_norm {
                Some(norm) => norm.forward(&q.flatten(0, 2)?)?.reshape((
                    batch,
                    self.num_heads,
                    seq_len,
                    self.head_dim,
                ))?,
                None => q,
            }
        };
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k.flatten(0, 2)?)?.reshape((
                batch,
                self.num_kv_heads,
                seq_len,
                self.head_dim,
            ))?,
            None => k,
        };

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(mask) = attn_mask {
            let mask = if mask.dtype() != scores.dtype() {
                mask.to_dtype(scores.dtype())?
            } else {
                mask.clone()
            };
            scores = scores.broadcast_add(&mask)?;
        }

        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let context = probs.matmul(&v)?.transpose(1, 2)?;
        let context = match attn_gate {
            Some(gate) => {
                let gate = candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?
                    .to_dtype(context.dtype())?;
                context.broadcast_mul(&gate)?
            }
            None => context,
        };
        let context = context.reshape((batch, seq_len, self.num_heads * self.head_dim))?;
        Ok(self.o_proj.forward(&context)?)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
struct StateSpaceWeights {
    in_proj_qkv: QMatMul,
    attn_gate: QMatMul,
    ssm_alpha: QMatMul,
    ssm_beta: QMatMul,
    ssm_conv1d_weight: Tensor,
    ssm_dt_bias: Tensor,
    ssm_a: Tensor,
    ssm_norm_weight: Tensor,
    ssm_out: QMatMul,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    conv_dim: usize,
    norm_eps: f64,
    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
}

impl StateSpaceWeights {
    fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        names: &LayerTensorNames,
        summary: &MetadataSummary,
    ) -> Result<Self> {
        let qkv_shape = ct
            .tensor_infos
            .get(&names.q_proj)
            .ok_or_else(|| anyhow::anyhow!("missing tensor info for {}", names.q_proj))?
            .shape
            .dims2()?;
        let gate_shape = ct
            .tensor_infos
            .get(&names.o_proj)
            .ok_or_else(|| anyhow::anyhow!("missing tensor info for {}", names.o_proj))?
            .shape
            .dims2()?;
        let alpha_name = format!("{}.ssm_alpha.weight", names.prefix);
        let alpha_shape = ct
            .tensor_infos
            .get(&alpha_name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor info for {alpha_name}"))?
            .shape
            .dims2()?;
        let conv_name = format!("{}.ssm_conv1d.weight", names.prefix);
        let conv_shape = ct
            .tensor_infos
            .get(&conv_name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor info for {conv_name}"))?
            .shape
            .dims2()?;
        let norm_name = format!("{}.ssm_norm.weight", names.prefix);
        let norm_dim = ct
            .tensor_infos
            .get(&norm_name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor info for {norm_name}"))?
            .shape
            .dims1()?;

        let conv_dim = qkv_shape.0;
        let value_dim = gate_shape.0;
        let num_v_heads = alpha_shape.0;
        let head_v_dim = norm_dim;
        if value_dim % num_v_heads != 0 {
            bail!(
                "invalid qwen3.5 state-space value_dim {value_dim} for {} heads",
                num_v_heads
            );
        }
        if conv_dim < value_dim || (conv_dim - value_dim) % 2 != 0 {
            bail!("invalid qwen3.5 state-space conv_dim {conv_dim} and value_dim {value_dim}");
        }
        let key_dim = (conv_dim - value_dim) / 2;
        let num_k_heads = num_v_heads;
        if key_dim % num_k_heads != 0 {
            bail!(
                "invalid qwen3.5 state-space key_dim {key_dim} for {} heads",
                num_k_heads
            );
        }
        let head_k_dim = key_dim / num_k_heads;
        let conv_kernel_size = conv_shape.1;
        let head_v_dim_from_value = value_dim / num_v_heads;
        if head_v_dim != head_v_dim_from_value {
            bail!(
                "qwen3.5 state-space head_v_dim mismatch: norm={head_v_dim}, derived={head_v_dim_from_value}"
            );
        }

        let state = Self {
            in_proj_qkv: QMatMul::from_qtensor(ct.tensor(reader, &names.q_proj, device)?)?,
            attn_gate: QMatMul::from_qtensor(ct.tensor(reader, &names.o_proj, device)?)?,
            ssm_alpha: QMatMul::from_qtensor(ct.tensor(reader, &alpha_name, device)?)?,
            ssm_beta: QMatMul::from_qtensor(ct.tensor(
                reader,
                &format!("{}.ssm_beta.weight", names.prefix),
                device,
            )?)?,
            ssm_conv1d_weight: ct.tensor(reader, &conv_name, device)?.dequantize(device)?,
            ssm_dt_bias: ct
                .tensor(reader, &format!("{}.ssm_dt.bias", names.prefix), device)?
                .dequantize(device)?
                .to_dtype(DType::F32)?,
            ssm_a: ct
                .tensor(reader, &format!("{}.ssm_a", names.prefix), device)?
                .dequantize(device)?
                .to_dtype(DType::F32)?,
            ssm_norm_weight: ct
                .tensor(reader, &norm_name, device)?
                .dequantize(device)?
                .to_dtype(DType::F32)?,
            ssm_out: QMatMul::from_qtensor(ct.tensor(
                reader,
                &format!("{}.ssm_out.weight", names.prefix),
                device,
            )?)?,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            conv_dim,
            norm_eps: summary.rms_norm_eps,
            conv_state: None,
            recurrent_state: None,
        };
        state.validate()?;
        Ok(state)
    }

    fn l2_norm(x: &Tensor) -> CandleResult<Tensor> {
        let inv_norm = (x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?.recip()?;
        x.broadcast_mul(&inv_norm)
    }

    fn repeat_interleave(x: &Tensor, n: usize, dim: usize) -> CandleResult<Tensor> {
        if n == 1 {
            return Ok(x.clone());
        }

        let mut dims = x.dims().to_vec();
        dims.insert(dim + 1, n);
        let expanded = x.unsqueeze(dim + 1)?.broadcast_as(dims.as_slice())?;
        let mut final_dims = x.dims().to_vec();
        final_dims[dim] *= n;
        expanded.reshape(final_dims.as_slice())
    }

    fn rms_norm_gated(&self, hidden_states: &Tensor, gate: &Tensor) -> CandleResult<Tensor> {
        let input_dtype = hidden_states.dtype();
        let hidden_states = hidden_states.to_dtype(DType::F32)?;
        let variance = hidden_states.sqr()?.mean_keepdim(D::Minus1)?;
        let hidden_states =
            hidden_states.broadcast_mul(&(&variance + self.norm_eps)?.sqrt()?.recip()?)?;
        let hidden_states = hidden_states.broadcast_mul(&self.ssm_norm_weight)?;
        let gate = candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?;
        hidden_states.broadcast_mul(&gate)?.to_dtype(input_dtype)
    }

    fn apply_depthwise_conv_silu(&mut self, mixed_qkv: &Tensor) -> CandleResult<Tensor> {
        let (batch_size, conv_dim, seq_len) = mixed_qkv.dims3()?;
        let kernel_size = self.conv_kernel_size;
        let use_precomputed_state = self.conv_state.is_some() && seq_len == 1;

        let conv_input = if use_precomputed_state {
            let conv_state = self.conv_state.as_mut().expect("validated above");
            let conv_state_data = Tensor::cat(&[conv_state.as_ref(), mixed_qkv], 2)?;
            *conv_state = conv_state_data.narrow(2, 1, kernel_size - 1)?;
            conv_state_data
        } else {
            let pad = kernel_size - 1;
            let padding = Tensor::zeros(
                (batch_size, conv_dim, pad),
                mixed_qkv.dtype(),
                mixed_qkv.device(),
            )?;
            let padded_qkv = Tensor::cat(&[&padding, mixed_qkv], 2)?;
            self.conv_state = Some(padded_qkv.narrow(2, seq_len, pad)?);
            padded_qkv
        };

        let weight = self.ssm_conv1d_weight.to_dtype(conv_input.dtype())?;
        let weight = match weight.rank() {
            2 => weight,
            3 => weight.squeeze(1)?,
            rank => {
                candle_core::bail!("unsupported qwen3.5 ssm_conv1d rank {rank}");
            }
        };
        let mut outputs = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let window = conv_input.narrow(2, i, kernel_size)?;
            let out = window.broadcast_mul(&weight.unsqueeze(0)?)?.sum(2)?;
            outputs.push(out.unsqueeze(2)?);
        }
        let out = Tensor::cat(&outputs, 2)?;
        candle_nn::ops::silu(&out)
    }

    fn torch_recurrent_gated_delta_rule(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        initial_state: Option<&Tensor>,
    ) -> CandleResult<(Tensor, Tensor)> {
        let query = Self::l2_norm(query)?;
        let key = Self::l2_norm(key)?;
        let query = query.transpose(1, 2)?;
        let key = key.transpose(1, 2)?;
        let value = value.transpose(1, 2)?;
        let beta = beta.transpose(1, 2)?;
        let g = g.transpose(1, 2)?;

        let (batch_size, num_heads, seq_len, k_head_dim) = key.dims4()?;
        let v_head_dim = value.dim(3)?;
        let scale = 1.0 / (query.dim(3)? as f64).sqrt();
        let query = (query * scale)?;

        let mut outputs = Vec::with_capacity(seq_len);
        let mut last_recurrent_state = match initial_state {
            Some(state) => state.to_dtype(DType::F32)?,
            None => Tensor::zeros(
                (batch_size, num_heads, k_head_dim, v_head_dim),
                DType::F32,
                query.device(),
            )?,
        };

        let query_f32 = query.to_dtype(DType::F32)?;
        let key_f32 = key.to_dtype(DType::F32)?;
        let value_f32 = value.to_dtype(DType::F32)?;
        let g_exp = g.to_dtype(DType::F32)?.exp()?.unsqueeze(3)?;
        let beta_f32 = beta.to_dtype(DType::F32)?.unsqueeze(3)?;

        for i in 0..seq_len {
            let q_t = query_f32.narrow(2, i, 1)?;
            let k_t = key_f32.narrow(2, i, 1)?;
            let v_t = value_f32.narrow(2, i, 1)?;
            let g_t = g_exp.narrow(2, i, 1)?;
            let beta_t = beta_f32.narrow(2, i, 1)?;

            last_recurrent_state = last_recurrent_state.broadcast_mul(&g_t)?;
            let kv_mem = k_t.matmul(&last_recurrent_state)?;
            let delta = (&v_t - &kv_mem)?.broadcast_mul(&beta_t)?;
            let delta_k = k_t.transpose(2, 3)?.matmul(&delta)?;
            last_recurrent_state = (&last_recurrent_state + &delta_k)?;
            outputs.push(q_t.matmul(&last_recurrent_state)?);
        }

        let core_attn_out = Tensor::cat(&outputs, 2)?.transpose(1, 2)?;
        Ok((core_attn_out, last_recurrent_state))
    }

    fn forward(&mut self, x: &Tensor, _offset: usize) -> CandleResult<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let hidden_states = x.contiguous()?;
        let input_dtype = hidden_states.dtype();

        let mixed_qkv = self.in_proj_qkv.forward(&hidden_states)?.transpose(1, 2)?;
        let z = self.attn_gate.forward(&hidden_states)?.reshape((
            batch_size,
            seq_len,
            self.num_v_heads,
            self.head_v_dim,
        ))?;
        let beta = candle_nn::ops::sigmoid(&self.ssm_beta.forward(&hidden_states)?)?;
        let a = self.ssm_alpha.forward(&hidden_states)?;

        let mixed_qkv = self
            .apply_depthwise_conv_silu(&mixed_qkv)?
            .transpose(1, 2)?;
        let q = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?.reshape((
            batch_size,
            seq_len,
            self.num_k_heads,
            self.head_k_dim,
        ))?;
        let k = mixed_qkv
            .narrow(D::Minus1, self.key_dim, self.key_dim)?
            .reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let v = mixed_qkv
            .narrow(D::Minus1, self.key_dim * 2, self.value_dim)?
            .reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        let g = {
            let a_f32 = a.to_dtype(DType::F32)?;
            let a_plus_dt = a_f32.broadcast_add(&self.ssm_dt_bias)?;
            let softplus = (a_plus_dt.exp()? + 1.0)?.log()?;
            softplus.broadcast_mul(&self.ssm_a)?
        };

        let repeat_n = self.num_v_heads / self.num_k_heads;
        let q = Self::repeat_interleave(&q, repeat_n, 2)?;
        let k = Self::repeat_interleave(&k, repeat_n, 2)?;
        let initial_state = if seq_len == 1 {
            self.recurrent_state.as_ref()
        } else {
            None
        };
        let (core_attn_out, new_state) =
            self.torch_recurrent_gated_delta_rule(&q, &k, &v, &g, &beta, initial_state)?;
        self.recurrent_state = Some(new_state);

        let core_attn_out = core_attn_out.to_dtype(input_dtype)?;
        let core_attn_out = core_attn_out.reshape(((), self.head_v_dim))?;
        let z_flat = z.reshape(((), self.head_v_dim))?;
        let core_attn_out = self.rms_norm_gated(&core_attn_out, &z_flat)?;
        let core_attn_out = core_attn_out.reshape((batch_size, seq_len, self.value_dim))?;
        self.ssm_out.forward(&core_attn_out)
    }

    fn clear_cache(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }

    fn validate(&self) -> CandleResult<()> {
        let _ = (
            &self.in_proj_qkv,
            &self.attn_gate,
            &self.ssm_alpha,
            &self.ssm_beta,
            &self.ssm_conv1d_weight,
            &self.ssm_dt_bias,
            &self.ssm_a,
            &self.ssm_norm_weight,
            &self.ssm_out,
        );
        if self.conv_dim != self.key_dim * 2 + self.value_dim {
            candle_core::bail!("invalid qwen3.5 state-space dims");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
enum LayerWeightsKind {
    FullAttention(AttentionWeights),
    StateSpace(StateSpaceWeights),
}

#[derive(Debug, Clone)]
struct LayerWeights {
    kind: LayerWeightsKind,
    mlp: MlpWeights,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl LayerWeights {
    fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        names: &LayerTensorNames,
        summary: &MetadataSummary,
        rotary: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let kind = match names.kind {
            LayerKind::FullAttention => LayerWeightsKind::FullAttention(AttentionWeights::new(
                ct, reader, device, names, summary, rotary,
            )?),
            LayerKind::StateSpace => LayerWeightsKind::StateSpace(StateSpaceWeights::new(
                ct, reader, device, names, summary,
            )?),
            LayerKind::Unknown => {
                bail!(
                    "layer {} uses unsupported quantized qwen3.5 layout ({:?})",
                    names.layer_idx,
                    names.kind
                );
            }
        };

        Ok(Self {
            kind,
            mlp: MlpWeights::new(ct, reader, device, names)?,
            ln1: RmsNorm::from_qtensor(
                ct.tensor(reader, &names.attn_norm, device)?,
                summary.rms_norm_eps,
            )?,
            ln2: RmsNorm::from_qtensor(
                ct.tensor(reader, &names.ffn_norm, device)?,
                summary.rms_norm_eps,
            )?,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> CandleResult<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = match &mut self.kind {
            LayerWeightsKind::FullAttention(attn) => attn.forward(&h, mask, offset)?,
            LayerWeightsKind::StateSpace(ssm) => ssm.forward(&h, offset)?,
        };
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        Ok((x + h2)?)
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.kind {
            LayerWeightsKind::FullAttention(attn) => attn.clear_kv_cache(),
            LayerWeightsKind::StateSpace(ssm) => ssm.clear_cache(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    dtype: DType,
    masks: HashMap<usize, Tensor>,
}

impl MetadataSummary {
    pub fn detect(metadata: &HashMap<String, gguf_file::Value>) -> Self {
        let general_architecture = metadata.get("general.architecture").and_then(as_string);
        let metadata_prefix = detect_metadata_prefix(metadata, general_architecture.as_deref());
        let architecture = match metadata_prefix.as_deref() {
            Some("qwen3") => QuantizedArchitecture::Qwen3,
            Some("qwen2") | Some("qwen35") => QuantizedArchitecture::Qwen3_5,
            _ => QuantizedArchitecture::Unknown,
        };
        let prefix = metadata_prefix.clone();
        Self {
            architecture,
            metadata_prefix,
            general_architecture,
            num_attention_heads: metadata_value(
                metadata,
                prefix.as_deref(),
                "attention.head_count",
            )
            .and_then(as_u64)
            .map(|v| v as usize)
            .unwrap_or(0),
            num_kv_heads: metadata_value(metadata, prefix.as_deref(), "attention.head_count_kv")
                .and_then(as_u64)
                .map(|v| v as usize)
                .unwrap_or(0),
            head_dim: metadata_value(metadata, prefix.as_deref(), "attention.key_length")
                .and_then(as_u64)
                .map(|v| v as usize)
                .unwrap_or(0),
            num_layers: metadata_value(metadata, prefix.as_deref(), "block_count")
                .and_then(as_u64)
                .map(|v| v as usize)
                .unwrap_or(0),
            hidden_size: metadata_value(metadata, prefix.as_deref(), "embedding_length")
                .and_then(as_u64)
                .map(|v| v as usize)
                .unwrap_or(0),
            max_position_embeddings: metadata_value(metadata, prefix.as_deref(), "context_length")
                .and_then(as_u64)
                .map(|v| v as usize)
                .unwrap_or(0),
            rms_norm_eps: metadata_value(
                metadata,
                prefix.as_deref(),
                "attention.layer_norm_rms_epsilon",
            )
            .and_then(as_f64)
            .unwrap_or(1e-6),
            rope_freq_base: metadata_value(metadata, prefix.as_deref(), "rope.freq_base")
                .and_then(as_f64)
                .unwrap_or(10000.0),
        }
    }
}

impl TensorNameMap {
    pub fn detect(ct: &gguf_file::Content, summary: &MetadataSummary) -> Result<Self> {
        let token_embeddings = require_tensor(ct, &["token_embd.weight"])?;
        let output_norm = require_tensor(ct, &["output_norm.weight"])?;
        let output = first_existing_tensor(ct, &["output.weight", "token_embd.weight"])
            .unwrap_or_else(|| "token_embd.weight".to_string());

        let mut layers = Vec::with_capacity(summary.num_layers);
        for layer_idx in 0..summary.num_layers {
            layers.push(LayerTensorNames::detect(ct, layer_idx)?);
        }

        Ok(Self {
            token_embeddings,
            output_norm,
            output,
            layers,
        })
    }
}

impl LayerTensorNames {
    fn detect(ct: &gguf_file::Content, layer_idx: usize) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let kind = if ct
            .tensor_infos
            .contains_key(&format!("{prefix}.attn_q.weight"))
        {
            LayerKind::FullAttention
        } else if ct
            .tensor_infos
            .contains_key(&format!("{prefix}.attn_qkv.weight"))
        {
            LayerKind::StateSpace
        } else {
            LayerKind::Unknown
        };

        Ok(Self {
            layer_idx,
            kind,
            prefix: prefix.clone(),
            attn_norm: require_tensor(ct, &[&format!("{prefix}.attn_norm.weight")])?,
            ffn_norm: require_tensor(
                ct,
                &[
                    &format!("{prefix}.ffn_norm.weight"),
                    &format!("{prefix}.post_attention_norm.weight"),
                ],
            )?,
            q_proj: require_tensor(
                ct,
                &[
                    &format!("{prefix}.attn_q.weight"),
                    &format!("{prefix}.attn_qkv.weight"),
                ],
            )?,
            k_proj: require_tensor(
                ct,
                &[
                    &format!("{prefix}.attn_k.weight"),
                    &format!("{prefix}.attn_qkv.weight"),
                ],
            )?,
            v_proj: require_tensor(
                ct,
                &[
                    &format!("{prefix}.attn_v.weight"),
                    &format!("{prefix}.attn_qkv.weight"),
                ],
            )?,
            o_proj: require_tensor(
                ct,
                &[
                    &format!("{prefix}.attn_output.weight"),
                    &format!("{prefix}.attn_gate.weight"),
                ],
            )?,
            q_norm: first_existing_tensor(ct, &[&format!("{prefix}.attn_q_norm.weight")]),
            k_norm: first_existing_tensor(ct, &[&format!("{prefix}.attn_k_norm.weight")]),
            q_bias: first_existing_tensor(ct, &[&format!("{prefix}.attn_q.bias")]),
            k_bias: first_existing_tensor(ct, &[&format!("{prefix}.attn_k.bias")]),
            v_bias: first_existing_tensor(ct, &[&format!("{prefix}.attn_v.bias")]),
            ffn_gate: require_tensor(ct, &[&format!("{prefix}.ffn_gate.weight")])?,
            ffn_up: require_tensor(ct, &[&format!("{prefix}.ffn_up.weight")])?,
            ffn_down: require_tensor(ct, &[&format!("{prefix}.ffn_down.weight")])?,
        })
    }
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let summary = MetadataSummary::detect(&ct.metadata);
        if summary.architecture != QuantizedArchitecture::Qwen3_5 {
            bail!(
                "quantized_qwen3_5 received unsupported architecture {:?}",
                summary.architecture
            );
        }
        if summary.num_attention_heads == 0
            || summary.num_kv_heads == 0
            || summary.head_dim == 0
            || summary.num_layers == 0
            || summary.hidden_size == 0
        {
            bail!("incomplete qwen3.5 gguf metadata: {:?}", summary);
        }

        let tensor_names = TensorNameMap::detect(&ct, &summary)?;
        let embed_tokens = Embedding::new(
            ct.tensor(reader, &tensor_names.token_embeddings, device)?
                .dequantize(device)?,
            summary.hidden_size,
        );
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, &tensor_names.output_norm, device)?,
            summary.rms_norm_eps,
        )?;
        let lm_head = QMatMul::from_qtensor(ct.tensor(reader, &tensor_names.output, device)?)?;
        let dtype = match ct.metadata.get("general.dtype") {
            Some(value) => match value.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };
        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            summary.head_dim,
            summary.max_position_embeddings,
            summary.rope_freq_base,
            device,
        )?);

        let mut layers = Vec::with_capacity(summary.num_layers);
        for layer in &tensor_names.layers {
            layers.push(LayerWeights::new(
                &ct,
                reader,
                device,
                layer,
                &summary,
                rotary.clone(),
            )?);
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            masks: HashMap::new(),
        })
    }

    fn causal_mask(&mut self, seq_len: usize) -> CandleResult<Tensor> {
        if let Some(mask) = self.masks.get(&seq_len) {
            return Ok(mask.clone());
        }

        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { minf } else { 0.0 }))
            .collect();
        let mask = Tensor::from_slice(&mask, (1, 1, seq_len, seq_len), &self.device)?
            .to_dtype(self.dtype)?;
        self.masks.insert(seq_len, mask.clone());
        Ok(mask)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> CandleResult<Tensor> {
        let (_, seq_len) = input.dims2()?;
        let mut hidden = self.embed_tokens.forward(input)?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.causal_mask(seq_len)?)
        };

        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, mask.as_ref(), offset)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        let last_hidden = hidden.i((.., seq_len - 1, ..))?;
        Ok(self.lm_head.forward(&last_hidden)?)
    }

    #[allow(dead_code)]
    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

fn detect_metadata_prefix(
    metadata: &HashMap<String, gguf_file::Value>,
    general_architecture: Option<&str>,
) -> Option<String> {
    if metadata.contains_key("qwen3.attention.head_count") {
        return Some("qwen3".to_string());
    }
    if metadata.contains_key("qwen2.attention.head_count") {
        return Some("qwen2".to_string());
    }
    if metadata.contains_key("qwen35.attention.head_count") {
        return Some("qwen35".to_string());
    }
    if let Some(arch) = general_architecture {
        if metadata.contains_key(&format!("{arch}.attention.head_count")) {
            return Some(arch.to_string());
        }
    }
    metadata.keys().find_map(|key| {
        key.strip_suffix(".attention.head_count")
            .map(ToString::to_string)
    })
}

fn as_string(value: &gguf_file::Value) -> Option<String> {
    match value {
        gguf_file::Value::String(value) => Some(value.clone()),
        _ => None,
    }
}

fn metadata_value<'a>(
    metadata: &'a HashMap<String, gguf_file::Value>,
    prefix: Option<&str>,
    suffix: &str,
) -> Option<&'a gguf_file::Value> {
    let key = format!("{}.{suffix}", prefix?);
    metadata.get(&key)
}

fn require_tensor(ct: &gguf_file::Content, candidates: &[&str]) -> Result<String> {
    first_existing_tensor(ct, candidates).ok_or_else(|| {
        anyhow::anyhow!(
            "missing tensor; tried candidates: {}",
            candidates.join(", ")
        )
    })
}

fn first_existing_tensor(ct: &gguf_file::Content, candidates: &[&str]) -> Option<String> {
    candidates.iter().find_map(|name| {
        ct.tensor_infos
            .contains_key(*name)
            .then(|| (*name).to_string())
    })
}

fn as_u64(value: &gguf_file::Value) -> Option<u64> {
    value.to_u64().ok()
}

fn as_f64(value: &gguf_file::Value) -> Option<f64> {
    value
        .to_f64()
        .ok()
        .or_else(|| value.to_f32().ok().map(|value| value as f64))
}
