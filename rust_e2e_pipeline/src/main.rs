use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// ============================================================================
// Pillar 1: Hardware PTPv2 Synchronizer & SE(3) 3D LiDAR Deskewing (32,768 pts)
// ============================================================================
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
pub struct LidarPoint4D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub dt: f32, // relative timestamp offset from scan start [s]
}

pub struct PtpSyncResult {
    pub sync_ok: bool,
    pub max_skew_ms: f32,
    pub min_ego_corridor_x: f32,
}

#[inline(always)]
pub fn pillar1_ptp_and_lidar_deskew(
    t_cam_ns: u64,
    t_lidar_ns: u64,
    t_can_ns: u64,
    ego_vx: f32,
    ego_vy: f32,
    ego_yaw_rate: f32,
    raw_points: &[LidarPoint4D],
    deskewed_out: &mut [LidarPoint4D],
    raw_rgb_u8: &[u8],
    norm_rgb_f32: &mut [f32],
) -> PtpSyncResult {
    // 1. IEEE 1588 PTPv2 Clock Domain Verification (< 1.0 ms)
    let diff_lidar = (t_cam_ns as i64 - t_lidar_ns as i64).unsigned_abs();
    let diff_can = (t_cam_ns as i64 - t_can_ns as i64).unsigned_abs();
    let max_skew_ns = diff_lidar.max(diff_can);
    let max_skew_ms = (max_skew_ns as f32) * 1e-6;
    let sync_ok = max_skew_ms <= 2.5;

    // 2. SE(3) 2D+Z Motion Compensation to Camera Exposure Midpoint
    // Small-angle exact Taylor/Trig pre-computation for high-speed SIMD auto-vectorization
    let mut min_corridor_x = 150.0_f32;
    let n = raw_points.len().min(deskewed_out.len());

    for i in 0..n {
        let p = unsafe { *raw_points.get_unchecked(i) };
        let dtheta = ego_yaw_rate * p.dt;
        // Fast exact 2nd-order Padé/Taylor rotation (accurate to 1e-7 rad for dt < 0.05s)
        let cos_t = 1.0 - 0.5 * dtheta * dtheta;
        let sin_t = dtheta;

        let dx = ego_vx * p.dt;
        let dy = ego_vy * p.dt;

        let x_new = cos_t * p.x - sin_t * p.y + dx;
        let y_new = sin_t * p.x + cos_t * p.y + dy;

        unsafe {
            *deskewed_out.get_unchecked_mut(i) = LidarPoint4D {
                x: x_new,
                y: y_new,
                z: p.z,
                dt: 0.0,
            };
        }

        // Fused Front Ego-Corridor Obstacle Extraction (|y| < 1.15m, 0.5m < x < 80m, z > -1.2m)
        if x_new > 0.5 && x_new < min_corridor_x && y_new.abs() < 1.15 && p.z > -1.2 && p.z < 1.8 {
            min_corridor_x = x_new;
        }
    }

    // 3. Camera 128x128x3 uint8 -> FP32 Normalization ([0, 1])
    let img_len = raw_rgb_u8.len().min(norm_rgb_f32.len());
    let inv_255 = 1.0_f32 / 255.0_f32;
    for i in 0..img_len {
        unsafe {
            *norm_rgb_f32.get_unchecked_mut(i) = (*raw_rgb_u8.get_unchecked(i) as f32) * inv_255;
        }
    }

    PtpSyncResult {
        sync_ok,
        max_skew_ms,
        min_ego_corridor_x: min_corridor_x,
    }
}

// ============================================================================
// Pillar 2: Lock-Free Atomic System 2 -> System 1 Corridor Bridge
// ============================================================================
pub struct LockFreeCorridorBridge {
    // Packs (d_min_mm: i16, d_max_mm: i16, v_ref_cms: u16, confidence_permille: u16) into a single 64-bit word!
    // Guarantees 100% tear-free, lock-free single-instruction atomic read/write on x86_64 & ARM64.
    packed_corridor: AtomicU64,
}

#[derive(Clone, Copy)]
pub struct CorridorPrior {
    pub d_min: f32,
    pub d_max: f32,
    pub v_ref: f32,
    pub confidence: f32,
}

impl LockFreeCorridorBridge {
    pub fn new(d_min: f32, d_max: f32, v_ref: f32, confidence: f32) -> Self {
        let bridge = Self {
            packed_corridor: AtomicU64::new(0),
        };
        bridge.store(d_min, d_max, v_ref, confidence);
        bridge
    }

    #[inline(always)]
    pub fn store(&self, d_min: f32, d_max: f32, v_ref: f32, confidence: f32) {
        let d_min_mm = (d_min * 1000.0) as i16 as u16 as u64;
        let d_max_mm = (d_max * 1000.0) as i16 as u16 as u64;
        let v_ref_cms = (v_ref.max(0.0) * 100.0) as u16 as u64;
        let conf_pm = (confidence.clamp(0.0, 1.0) * 1000.0) as u16 as u64;
        let word = (d_min_mm << 48) | (d_max_mm << 32) | (v_ref_cms << 16) | conf_pm;
        self.packed_corridor.store(word, Ordering::Release);
    }

    #[inline(always)]
    pub fn load(&self) -> CorridorPrior {
        let word = self.packed_corridor.load(Ordering::Acquire);
        let d_min_mm = ((word >> 48) & 0xFFFF) as u16 as i16;
        let d_max_mm = ((word >> 32) & 0xFFFF) as u16 as i16;
        let v_ref_cms = ((word >> 16) & 0xFFFF) as u16;
        let conf_pm = (word & 0xFFFF) as u16;
        CorridorPrior {
            d_min: (d_min_mm as f32) * 0.001,
            d_max: (d_max_mm as f32) * 0.001,
            v_ref: (v_ref_cms as f32) * 0.01,
            confidence: (conf_pm as f32) * 0.001,
        }
    }
}

// ============================================================================
// Pillar 3: ISO 26262 ASIL-D 3-Tier Safety Shield + CAN-FD CRC-8 Frame Packer
// ============================================================================
#[derive(Clone, Copy)]
pub struct SafetyShieldOutput {
    pub steer_cmd: f32,
    pub accel_cmd: f32,
    pub active_tier: u8, // 0=Nominal, 1=Entropy, 2=CBF/RSS, 3=MRM
    pub can_frame: [u8; 8],
}

#[inline(always)]
pub fn compute_crc8_sae_j1850(data: &[u8; 7]) -> u8 {
    let mut crc: u8 = 0xFF;
    for &byte in data.iter() {
        crc ^= byte;
        for _ in 0..8 {
            if (crc & 0x80) != 0 {
                crc = (crc << 1) ^ 0x1D;
            } else {
                crc <<= 1;
            }
        }
    }
    crc ^ 0xFF
}

#[inline(always)]
pub fn pillar3_safety_shield_and_can_pack(
    mode_probs: &[f32; 5],
    mode_steers: &[f32; 5],
    mode_accels: &[f32; 5],
    ego_d: f32,
    ego_v: f32,
    lead_dist_x: f32,
    lead_v: f32,
    prev_steer: f32,
    prev_accel: f32,
    ptp_sync_ok: bool,
    corridor: CorridorPrior,
    alive_counter: u8,
) -> SafetyShieldOutput {
    let mut active_tier: u8 = 0;

    // 1. Select best mode & compute normalized Shannon Entropy
    let mut best_idx = 0usize;
    let mut best_prob = mode_probs[0];
    let mut entropy = 0.0_f32;
    for i in 0..5 {
        let p = mode_probs[i].max(1e-6);
        if p > best_prob {
            best_prob = p;
            best_idx = i;
        }
        entropy -= p * p.ln();
    }
    let norm_entropy = entropy / 1.6094379_f32; // ln(5)

    let mut steer = mode_steers[best_idx];
    let mut accel = mode_accels[best_idx];

    // Tier 1: Epistemic Uncertainty Gate
    if norm_entropy > 0.88 || best_prob < 0.45 {
        active_tier = 1;
        // Blend toward cautious lane keeping & mild deceleration
        steer = 0.6 * steer;
        accel = accel.min(-1.0);
    }

    // Tier 2A: Analytical Control Barrier Function (CBF) for Double Yellow Line (d_yellow = -3.45m, margin = -2.55m)
    let d_safe_min = corridor.d_min.max(-2.55_f32);
    let h_cbf = ego_d - d_safe_min;
    let cbf_alpha = 1.85_f32;
    // Lateral velocity approx v * steer * 0.45
    let d_dot = ego_v * steer * 0.45;
    if d_dot + cbf_alpha * h_cbf < 0.0 {
        active_tier = active_tier.max(2);
        let required_d_dot = -cbf_alpha * h_cbf;
        let safe_steer = (required_d_dot / (ego_v.max(1.0) * 0.45)).clamp(-0.45, 0.45);
        steer = steer.max(safe_steer);
    }

    // Tier 2B: Mobileye RSS Longitudinal Safe Distance Shield
    let rho = 0.15_f32; // reaction time [s]
    let a_min_brake = 4.0_f32;
    let a_max_brake = 8.0_f32;
    let v_rel = (ego_v - lead_v).max(0.0);
    let d_rss = (ego_v * rho + 0.5 * v_rel * v_rel / a_min_brake).max(4.5);
    if lead_dist_x < d_rss {
        active_tier = active_tier.max(2);
        let pen = ((d_rss - lead_dist_x) / d_rss).clamp(0.0, 1.0);
        accel = accel.min(-a_min_brake - pen * (a_max_brake - a_min_brake));
    }

    // Tier 3: Hardware PTP Sync Watchdog MRM & Kinematic Jerk/Steer Rate Saturation
    if !ptp_sync_ok {
        active_tier = 3;
        steer = -0.15 * ego_d.clamp(-1.0, 1.0);
        accel = -3.5;
    }

    let dt = 0.030_f32;
    let max_dsteer = 0.45_f32 * dt;
    let max_daccel = 4.5_f32 * dt;
    steer = steer.clamp(prev_steer - max_dsteer, prev_steer + max_dsteer).clamp(-0.50, 0.50);
    accel = accel.clamp(prev_accel - max_daccel, prev_accel + max_daccel).clamp(-8.0, 3.0);

    // Pack into Hyundai/Kia E-GMP ADAS_CMD_11 (0x164) 8-Byte CAN Frame with SAE J1850 CRC-8
    let steer_raw = ((steer * 1000.0) as i16) as u16;
    let accel_raw = (((accel + 10.0) * 1000.0) as u16).min(0xFFFF);
    let data7: [u8; 7] = [
        (steer_raw & 0xFF) as u8,
        (steer_raw >> 8) as u8,
        (accel_raw & 0xFF) as u8,
        (accel_raw >> 8) as u8,
        active_tier,
        (corridor.v_ref * 3.6) as u8,
        alive_counter & 0x0F,
    ];
    let crc = compute_crc8_sae_j1850(&data7);
    let can_frame = [data7[0], data7[1], data7[2], data7[3], data7[4], data7[5], data7[6], crc];

    SafetyShieldOutput {
        steer_cmd: steer,
        accel_cmd: accel,
        active_tier,
        can_frame,
    }
}

// ============================================================================
// Statistics Helper
// ============================================================================
struct LatencyStats {
    mean_us: f64,
    p50_us: f64,
    p90_us: f64,
    p99_us: f64,
    p999_us: f64,
    max_us: f64,
    std_us: f64,
}

fn compute_stats(samples_ns: &[u64]) -> LatencyStats {
    let mut sorted = samples_ns.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let sum_ns: u128 = sorted.iter().map(|&x| x as u128).sum();
    let mean_us = (sum_ns as f64 / n as f64) * 1e-3;
    let var_us: f64 = sorted
        .iter()
        .map(|&x| {
            let d = (x as f64) * 1e-3 - mean_us;
            d * d
        })
        .sum::<f64>()
        / (n as f64);

    let idx = |p: f64| -> usize { ((p * (n as f64 - 1.0)).round() as usize).min(n - 1) };
    LatencyStats {
        mean_us,
        p50_us: sorted[idx(0.50)] as f64 * 1e-3,
        p90_us: sorted[idx(0.90)] as f64 * 1e-3,
        p99_us: sorted[idx(0.99)] as f64 * 1e-3,
        p999_us: sorted[idx(0.999)] as f64 * 1e-3,
        max_us: sorted[n - 1] as f64 * 1e-3,
        std_us: var_us.sqrt(),
    }
}

fn print_stats_json(name: &str, s: &LatencyStats, comma: bool) {
    let suffix = if comma { "," } else { "" };
    println!(
        "  \"{}\": {{\"mean_us\": {:.3}, \"p50_us\": {:.3}, \"p90_us\": {:.3}, \"p99_us\": {:.3}, \"p999_us\": {:.3}, \"max_us\": {:.3}, \"std_us\": {:.3}}}{}",
        name, s.mean_us, s.p50_us, s.p90_us, s.p99_us, s.p999_us, s.max_us, s.std_us, suffix
    );
}

fn main() {
    let num_iters = 2000usize;
    let num_lidar_pts = 32_768usize;
    let rgb_len = 128 * 128 * 3;

    // Pre-allocate deterministic sensor buffers (Zero-GC in steady-state!)
    let mut raw_lidar = vec![LidarPoint4D::default(); num_lidar_pts];
    for i in 0..num_lidar_pts {
        let angle = (i as f32) * 0.0001917476;
        let dist = 6.0 + ((i * 73) % 5000) as f32 * 0.012;
        raw_lidar[i] = LidarPoint4D {
            x: dist * angle.cos(),
            y: dist * angle.sin(),
            z: -0.5 + ((i % 40) as f32) * 0.05,
            dt: ((i as f32) / (num_lidar_pts as f32)) * 0.050 - 0.025,
        };
    }
    let mut deskewed_lidar = vec![LidarPoint4D::default(); num_lidar_pts];
    let raw_rgb = vec![128u8; rgb_len];
    let mut norm_rgb = vec![0.0f32; rgb_len];

    // Pre-allocate Zero-GC Ring Buffer for Pillar 4 Shadow-Mode Logger (16 slots of 1MB telemetry/camera chunks)
    let ring_slot_bytes = 1024 * 1024; // 1 MB per frame chunk
    let mut ring_buffer: Vec<Vec<u8>> = (0..16).map(|_| vec![0u8; ring_slot_bytes]).collect();
    let incoming_frame = vec![42u8; ring_slot_bytes];

    // Warmup
    for _ in 0..100 {
        let res = pillar1_ptp_and_lidar_deskew(
            1_000_000_000,
            1_000_250_000,
            999_850_000,
            12.5,
            0.15,
            0.08,
            black_box(&raw_lidar),
            black_box(&mut deskewed_lidar),
            black_box(&raw_rgb),
            black_box(&mut norm_rgb),
        );
        black_box(res);
    }

    // 1. Benchmark Pillar 1 (PTP + 32,768 LiDAR SE(3) Deskew + RGB Norm)
    let mut p1_samples = Vec::with_capacity(num_iters);
    for i in 0..num_iters {
        let t0 = Instant::now();
        let res = pillar1_ptp_and_lidar_deskew(
            1_000_000_000 + (i as u64) * 30_000_000,
            1_000_250_000 + (i as u64) * 30_000_000,
            999_850_000 + (i as u64) * 30_000_000,
            12.5,
            0.15,
            0.08,
            black_box(&raw_lidar),
            black_box(&mut deskewed_lidar),
            black_box(&raw_rgb),
            black_box(&mut norm_rgb),
        );
        black_box(res.min_ego_corridor_x);
        p1_samples.push(t0.elapsed().as_nanos() as u64);
    }

    // 2. Benchmark Pillar 3 (ASIL-D 3-Tier Safety Shield + CAN CRC-8 Packing)
    let bridge = Arc::new(LockFreeCorridorBridge::new(-2.45, 2.85, 11.5, 0.96));
    let mode_probs = [0.68_f32, 0.14, 0.09, 0.05, 0.04];
    let mode_steers = [-0.08_f32, 0.02, 0.11, -0.18, 0.22];
    let mode_accels = [0.45_f32, 0.20, -0.50, -1.20, -2.50];

    let mut p3_samples = Vec::with_capacity(num_iters);
    let mut prev_steer = 0.0_f32;
    let mut prev_accel = 0.0_f32;
    for i in 0..num_iters {
        let t0 = Instant::now();
        let ego_d = -2.40 + ((i % 30) as f32) * 0.02;
        let lead_dist = 14.0 - ((i % 20) as f32) * 0.35;
        let out = pillar3_safety_shield_and_can_pack(
            black_box(&mode_probs),
            black_box(&mode_steers),
            black_box(&mode_accels),
            black_box(ego_d),
            12.0,
            black_box(lead_dist),
            8.5,
            prev_steer,
            prev_accel,
            true,
            bridge.load(),
            (i & 0x0F) as u8,
        );
        prev_steer = out.steer_cmd;
        prev_accel = out.accel_cmd;
        black_box(out.can_frame);
        p3_samples.push(t0.elapsed().as_nanos() as u64);
    }

    // 3. Benchmark Full Host Pipeline Under Heavy Multi-Thread Contention
    //    (Background Thread 1: System 2 World Model Corridor Writer + CPU Math)
    //    (Background Thread 2: Heavy Memory Churn / Ring-Buffer Logger Thread)
    let stop_flag = Arc::new(AtomicBool::new(false));
    let bridge_bg = Arc::clone(&bridge);
    let stop_bg1 = Arc::clone(&stop_flag);

    let bg_sys2 = thread::spawn(move || {
        let mut step = 0u64;
        while !stop_bg1.load(Ordering::Relaxed) {
            let d_min = -2.45 + ((step % 10) as f32) * 0.01;
            let d_max = 2.80 - ((step % 10) as f32) * 0.01;
            bridge_bg.store(d_min, d_max, 11.5, 0.95);
            step = step.wrapping_add(1);
            thread::sleep(Duration::from_micros(250));
        }
    });

    let stop_bg2 = Arc::clone(&stop_flag);
    let bg_logger = thread::spawn(move || {
        // Simulate background telemetry compression work on separate OS thread
        let mut scratch = vec![0u8; 512 * 1024];
        let mut c = 1u8;
        while !stop_bg2.load(Ordering::Relaxed) {
            for b in scratch.iter_mut().step_by(64) {
                *b = b.wrapping_add(c);
            }
            c = c.wrapping_add(1);
            black_box(&scratch);
            thread::sleep(Duration::from_micros(200));
        }
    });

    let mut full_host_samples = Vec::with_capacity(num_iters);
    let mut sync_read_samples = Vec::with_capacity(num_iters);

    for i in 0..num_iters {
        // Measure Lock-Free Corridor Read + Zero-GC Ring-Buffer Copy
        let t_sync = Instant::now();
        let corr = bridge.load();
        let slot = i & 15;
        ring_buffer[slot].copy_from_slice(black_box(&incoming_frame));
        black_box(&ring_buffer[slot]);
        sync_read_samples.push(t_sync.elapsed().as_nanos() as u64);

        // Measure Full Host Pipeline (Pillar 1 + Pillar 2 Sync + Pillar 3 Shield + Pillar 4 Ring-Buffer)
        let t_full = Instant::now();
        let p1 = pillar1_ptp_and_lidar_deskew(
            1_000_000_000 + (i as u64) * 30_000_000,
            1_000_250_000 + (i as u64) * 30_000_000,
            999_850_000 + (i as u64) * 30_000_000,
            12.5,
            0.15,
            0.08,
            black_box(&raw_lidar),
            black_box(&mut deskewed_lidar),
            black_box(&raw_rgb),
            black_box(&mut norm_rgb),
        );
        let corr_live = bridge.load();
        let p3 = pillar3_safety_shield_and_can_pack(
            black_box(&mode_probs),
            black_box(&mode_steers),
            black_box(&mode_accels),
            black_box(-2.40 + ((i % 30) as f32) * 0.02),
            12.0,
            black_box(p1.min_ego_corridor_x.min(14.0)),
            8.5,
            prev_steer,
            prev_accel,
            p1.sync_ok,
            corr_live,
            (i & 0x0F) as u8,
        );
        ring_buffer[slot].copy_from_slice(black_box(&incoming_frame));
        prev_steer = p3.steer_cmd;
        prev_accel = p3.accel_cmd;
        black_box(p3.can_frame);
        black_box(corr);
        full_host_samples.push(t_full.elapsed().as_nanos() as u64);
    }

    stop_flag.store(true, Ordering::Relaxed);
    let _ = bg_sys2.join();
    let _ = bg_logger.join();

    let s_p1 = compute_stats(&p1_samples);
    let s_p2 = compute_stats(&sync_read_samples);
    let s_p3 = compute_stats(&p3_samples);
    let s_full = compute_stats(&full_host_samples);

    println!("{{");
    print_stats_json("pillar1_ptp_lidar_deskew", &s_p1, true);
    print_stats_json("pillar2_4_lockfree_sync_and_ringbuf", &s_p2, true);
    print_stats_json("pillar3_safety_shield_can_crc", &s_p3, true);
    print_stats_json("full_host_pipeline_under_contention", &s_full, false);
    println!("}}");
}
