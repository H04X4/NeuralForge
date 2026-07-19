#[allow(clippy::too_many_arguments)]
pub fn causal_attention(
    output: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    if seq_len == 0 {
        return;
    }
    for h in 0..n_heads {
        let h_off = h * head_dim;
        for i in 0..seq_len {
            let qi_off = h_off + i * n_heads * head_dim;
            let mut scores = vec![0.0f32; seq_len];
            for (j, s) in scores.iter_mut().enumerate().take(i + 1) {
                let kj_off = h_off + j * n_heads * head_dim;
                let mut sum = 0.0f32;
                for d in 0..head_dim {
                    sum += q[qi_off + d] * k[kj_off + d];
                }
                *s = sum * scale;
            }
            for s in scores.iter_mut().take(seq_len).skip(i + 1) {
                *s = f32::NEG_INFINITY;
            }

            let mut max_val = scores[0];
            for &s in &scores {
                if s > max_val {
                    max_val = s;
                }
            }
            let mut sum_exp = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_val).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0 / sum_exp;
            for s in scores.iter_mut() {
                *s *= inv_sum;
            }

            let out_off = h_off + i * n_heads * head_dim;
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for (j, s) in scores.iter().enumerate().take(i + 1) {
                    let vj_off = h_off + j * n_heads * head_dim;
                    val += s * v[vj_off + d];
                }
                output[out_off + d] = val;
            }
        }
    }
}


