use astra_core::Result;

pub fn matmul(c: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<()> {
    if c.len() < m * n || a.len() < m * k || b.len() < k * n {
        return Err(astra_core::Error::TensorShape {
            expected: vec![m, n],
            got: vec![c.len()],
        });
    }
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for t in 0..k {
                sum += a[i * k + t] * b[t * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    Ok(())
}

pub fn matmul_tiled(c: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<()> {
    if c.len() < m * n || a.len() < m * k || b.len() < k * n {
        return Err(astra_core::Error::TensorShape {
            expected: vec![m, n],
            got: vec![c.len()],
        });
    }
    const TM: usize = 32;
    const TN: usize = 32;
    const TK: usize = 256;

    c.fill(0.0f32);
    for i0 in (0..m).step_by(TM) {
        let imax = (i0 + TM).min(m);
        for j0 in (0..n).step_by(TN) {
            let jmax = (j0 + TN).min(n);
            for k0 in (0..k).step_by(TK) {
                let kmax = (k0 + TK).min(k);
                for i in i0..imax {
                    for kk in k0..kmax {
                        let aik = a[i * k + kk];
                        let b_row = &b[kk * n + j0..kk * n + jmax];
                        let c_row = &mut c[i * n + j0..i * n + jmax];
                        for j in 0..(jmax - j0) {
                            c_row[j] += aik * b_row[j];
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 31) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0 {
        sign << 15
    } else if exp >= 0x8f {
        (sign << 15) | 0x7c00 | if mant != 0 { 0x0200 } else { 0 }
    } else if exp < 0x70 {
        sign << 15
    } else {
        let fe = (exp - 127 + 15) as u16;
        let fm = (mant >> 13) as u16;
        (sign << 15) | (fe << 10) | fm
    }
}

fn f16_to_f32(x: u16) -> f32 {
    let sign = ((x >> 15) as u32) << 31;
    let exp = (x >> 10) & 0x1f;
    let mantissa = (x & 0x3ff) as u32;
    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        f32::from_bits(sign | ((127 - 15 - 10) << 23) | (mantissa << 13))
    } else if exp == 31 {
        f32::from_bits(sign | 0x7f800000 | (mantissa << 13))
    } else {
        f32::from_bits(sign | ((exp as u32 + 127 - 15) << 23) | (mantissa << 13))
    }
}

const Q8_0_BLOCK: usize = 32;

pub fn quantize_q8_0(out: &mut [u8], src: &[f32]) {
    assert!(out.len() >= src.len() + src.len() / Q8_0_BLOCK * 2);
    let blocks = src.len().div_ceil(Q8_0_BLOCK);
    for b in 0..blocks {
        let start = b * Q8_0_BLOCK;
        let end = (start + Q8_0_BLOCK).min(src.len());
        let mut amax = 0.0f32;
        for &sv in src[start..end].iter() {
            let v = sv.abs();
            if v > amax {
                amax = v;
            }
        }
        let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        let out_off = b * (2 + Q8_0_BLOCK);
        out[out_off..out_off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for (i, &sv) in src[start..end].iter().enumerate() {
            let q = (sv * id).round().clamp(-128.0, 127.0) as i8;
            out[out_off + 2 + i] = q as u8;
        }
    }
}

pub fn matmul_q8_0(c: &mut [f32], a: &[f32], w: &[u8], m: usize, n: usize, k: usize) -> Result<()> {
    if c.len() < m * n || a.len() < m * k {
        return Err(astra_core::Error::TensorShape {
            expected: vec![m, n],
            got: vec![c.len()],
        });
    }
    let blocks_per_row = k.div_ceil(Q8_0_BLOCK);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            let w_base = j * blocks_per_row * (2 + Q8_0_BLOCK);
            for blk in 0..blocks_per_row {
                let off = w_base + blk * (2 + Q8_0_BLOCK);
                let scale = f16_to_f32(u16::from_le_bytes([w[off], w[off + 1]]));
                let blk_end = ((blk + 1) * Q8_0_BLOCK).min(k);
                for v in 0..(blk_end - blk * Q8_0_BLOCK) {
                    let k_idx = blk * Q8_0_BLOCK + v;
                    let qv = w[off + 2 + v] as i8 as f32;
                    sum += qv * scale * a[i * k + k_idx];
                }
            }
            c[i * n + j] = sum;
        }
    }
    Ok(())
}

const Q4_0_BLOCK: usize = 32;

pub fn matmul_q4_0(c: &mut [f32], a: &[f32], w: &[u8], m: usize, n: usize, k: usize) -> Result<()> {
    if c.len() < m * n || a.len() < m * k {
        return Err(astra_core::Error::TensorShape {
            expected: vec![m, n],
            got: vec![c.len()],
        });
    }
    let blocks_per_row = k.div_ceil(Q4_0_BLOCK);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            let w_base = j * blocks_per_row * (2 + 16);
            for blk in 0..blocks_per_row {
                let off = w_base + blk * (2 + 16);
                let scale = f16_to_f32(u16::from_le_bytes([w[off], w[off + 1]]));
                let blk_vals = &w[off + 2..off + 18];
                for v in 0..Q4_0_BLOCK.min(k - blk * Q4_0_BLOCK) {
                    let byte_idx = v / 2;
                    let nibble = if v % 2 == 0 {
                        blk_vals[byte_idx] & 0x0f
                    } else {
                        blk_vals[byte_idx] >> 4
                    };
                    let qv = (nibble as i8 - 8) as f32;
                    let k_idx = blk * Q4_0_BLOCK + v;
                    sum += qv * scale * a[i * k + k_idx];
                }
            }
            c[i * n + j] = sum;
        }
    }
    Ok(())
}


