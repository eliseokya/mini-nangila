use clap::Parser;
use chrono::Utc;
use nangila_ai::MomentumPredictor;
use nangila_core::{Predictor, Quantizer};
use nangila_hpc::ErrorBoundedQuantizer;
use nangila_math::FixedPointBuffer;
use std::{thread, time::Duration};

/// Simulated rotor twin demo: 6-DOF IMU at 100 Hz with predictive compression.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Duration in seconds
    #[arg(long, default_value_t = 10)]
    duration: u64,

    /// Sampling rate (Hz)
    #[arg(long, default_value_t = 100)]
    rate_hz: u64,

    /// Error bound epsilon
    #[arg(long, default_value_t = 0.05)]
    epsilon: f32,
}

fn main() {
    let args = Args::parse();
    let dt = Duration::from_millis(1000 / args.rate_hz.max(1));
    let total_ticks = args.duration * args.rate_hz;

    // Edge and Cloud nodes (closed loop)
    let edge_predictor = MomentumPredictor::new(0.9);
    let edge_quantizer = ErrorBoundedQuantizer::new(args.epsilon);
    let mut edge = nangila_twin::EdgeNode::new(edge_predictor, edge_quantizer);

    let cloud_predictor = MomentumPredictor::new(0.9);
    let cloud_quantizer = ErrorBoundedQuantizer::new(args.epsilon);
    let mut cloud = nangila_twin::CloudNode::new(cloud_predictor, cloud_quantizer);

    println!("Rotor Twin Streaming Demo — {}s @ {} Hz", args.duration, args.rate_hz);
    println!("Start: {}", Utc::now());

    let mut bytes_raw = 0usize;
    let mut bytes_tx = 0usize;
    let mut rms_err_accum = 0.0f64;
    let mut rms_count = 0usize;

    for t in 0..total_ticks {
        let tick = t as f32 * (1.0 / args.rate_hz as f32);
        // 6-DOF IMU-like signal (ax, ay, az, gx, gy, gz)
        let imu = [
            (tick * 2.0 * std::f32::consts::PI * 1.67).sin() * 1.0, // ax
            (tick * 2.0 * std::f32::consts::PI * 1.67).cos() * 1.0, // ay
            0.01,                                                   // az bias
            (tick * 2.0 * std::f32::consts::PI * 1.67).sin() * 5.0, // gx
            (tick * 2.0 * std::f32::consts::PI * 1.67).cos() * 5.0, // gy
            0.0,                                                    // gz
        ];
        let sensor = FixedPointBuffer::from_f32(&imu);

        bytes_raw += imu.len() * 4; // FP32

        let (packet, scale) = edge.send(&sensor).expect("edge send");
        bytes_tx += packet.len() + 4; // + f32 scale

        let recon = cloud.receive(&packet, scale).expect("cloud receive");

        // Compute RMS error for monitoring
        let orig = sensor.to_f32();
        let rec = recon.to_f32();
        for i in 0..orig.len() {
            let e = (orig[i] - rec[i]) as f64;
            rms_err_accum += e * e;
            rms_count += 1;
        }

        if t % args.rate_hz == 0 {
            let ratio = bytes_raw as f32 / bytes_tx as f32;
            let rms = (rms_err_accum / rms_count.max(1) as f64).sqrt();
            println!(
                "t={:4}s  compression={:6.2}×  rms_err={:.5}",
                t / args.rate_hz,
                ratio,
                rms
            );
        }

        thread::sleep(dt);
    }

    let ratio = bytes_raw as f32 / bytes_tx as f32;
    let rms = (rms_err_accum / rms_count.max(1) as f64).sqrt();
    println!("Done: {}", Utc::now());
    println!("Total raw: {:.2} KB, tx: {:.2} KB, ratio: {:.2}×, RMS err: {:.5}",
        bytes_raw as f32 / 1024.0,
        bytes_tx as f32 / 1024.0,
        ratio,
        rms
    );
}

