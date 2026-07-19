pub fn rms_norm(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let rms = (ss / n as f32 + eps).sqrt();
    let inv = 1.0 / rms;
    for i in 0..n {
        x[i] = x[i] * inv * w[i];
    }
}


