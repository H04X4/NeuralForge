pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn swiglu(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..out.len().min(a.len()).min(b.len()) {
        out[i] = silu(a[i]) * b[i];
    }
}

pub fn rope(x: &mut [f32], pos: usize, theta: f32) {
    let n = x.len();
    let half = n / 2;
    for i in 0..half {
        let freq = pos as f32 * theta.powf(-2.0 * i as f32 / n as f32);
        let (sinc, cosc) = freq.sin_cos();
        let xi = x[i];
        let xi_half = x[i + half];
        x[i] = xi * cosc - xi_half * sinc;
        x[i + half] = xi * sinc + xi_half * cosc;
    }
}

pub fn rope_qwen(x: &mut [f32], pos: usize, head_dim: usize, rope_theta: f32) {
    for chunk in x.chunks_mut(head_dim) {
        rope(chunk, pos, rope_theta);
    }
}

pub fn softmax_inplace(x: &mut [f32]) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let mut max_val = x[0];
    for &v in x.iter() {
        if v > max_val {
            max_val = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

pub fn softmax_causal(x: &mut [f32]) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let mut max_val = x[0];
    for &v in x.iter() {
        if v > max_val {
            max_val = v;
        }
    }
    let mut sum = 0.0f32;
    for (i, v) in x.iter_mut().enumerate() {
        let val = if i > n - 1 {
            *v = 0.0;
            continue;
        } else {
            (*v - max_val).exp()
        };
        *v = val;
        sum += val;
    }
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}


