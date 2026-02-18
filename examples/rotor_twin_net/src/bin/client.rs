use clap::Parser;
use nangila_ai::MomentumPredictor;
use nangila_core::{Predictor, Quantizer};
use nangila_hpc::ErrorBoundedQuantizer;
use nangila_math::FixedPointBuffer;
use std::time::{Duration, Instant};

#[path = "../generated/rotor.rs"]
mod rotor_pb;

use rotor_pb::rotor::rotor_service_client::RotorServiceClient;
use rotor_pb::rotor::Frame;

#[derive(Parser, Debug, Clone)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:50051")]
    server: String,
    #[arg(long, default_value_t = 50)]
    rate_hz: u64,
    #[arg(long, default_value_t = 5)]
    duration: u64,
    #[arg(long, default_value = "client_metrics.csv")]
    metrics: String,
    /// Epsilon (error bound) for quantization
    #[arg(long, default_value_t = 0.05)]
    epsilon: f32,
    /// TopK fraction (0.0..1.0). If > 0, enables RLE over zeroed residuals
    #[arg(long, default_value_t = 0.0)]
    topk: f32,
}

fn imu_sample(t: f32) -> Vec<f32> {
    vec![
        (t * 2.0 * std::f32::consts::PI * 1.67).sin() * 1.0,
        (t * 2.0 * std::f32::consts::PI * 1.67).cos() * 1.0,
        0.01,
        (t * 2.0 * std::f32::consts::PI * 1.67).sin() * 5.0,
        (t * 2.0 * std::f32::consts::PI * 1.67).cos() * 5.0,
        0.0,
    ]
}

// Simple RLE encoder for i16 array (same format as HPC example)
fn rle_encode(qvals: &[i16]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < qvals.len() {
        if qvals[i] == 0 {
            let start = i;
            while i < qvals.len() && qvals[i] == 0 && (i - start) < 128 { i += 1; }
            let count = i - start;
            let opcode = (128 + count - 1) as u8;
            out.push(opcode);
        } else {
            let start = i;
            while i < qvals.len() && qvals[i] != 0 && (i - start) < 128 { i += 1; }
            let count = i - start;
            let opcode = (count - 1) as u8;
            out.push(opcode);
            for j in 0..count {
                out.extend_from_slice(&qvals[start + j].to_le_bytes());
            }
        }
    }
    out
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut client = RotorServiceClient::connect(args.server.clone()).await?;
    let mut predictor = MomentumPredictor::new(0.9);
    let quant = ErrorBoundedQuantizer::new(args.epsilon);
    let mut fh = std::fs::File::create(&args.metrics)?;
    writeln!(fh, "tick,len,raw,comp,ratio").ok();
    let dt = Duration::from_millis(1000 / args.rate_hz);
    let start = Instant::now();
    let mut tick: u64 = 0;
    loop {
        let t = start.elapsed().as_secs_f32();
        if t > args.duration as f32 { break; }
        let data = imu_sample(t);
        let buf = FixedPointBuffer::from_f32(&data);
        let pred = predictor.predict().unwrap();
        let residual = if pred.is_empty() { buf.clone() } else { buf.sub(&pred).unwrap() };
        let (bytes, scale, used_rle) = if args.topk > 0.0 {
            // TopK + RLE path: create i16 residual vector with zeros for dropped entries
            let res_f = residual.to_f32();
            let n = res_f.len();
            let k = ((n as f32) * args.topk).max(1.0) as usize;
            // Indices of top |residual|
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_unstable_by(|&a, &b| res_f[b].abs().partial_cmp(&res_f[a].abs()).unwrap());
            let keep_idx = &idx[..k];
            // Quantization scale for i16
            let scale = 2.0 * args.epsilon;
            let mut qvals: Vec<i16> = vec![0; n];
            for &i in keep_idx {
                let q = (res_f[i] / scale).round();
                let clamped = q.max(i16::MIN as f32).min(i16::MAX as f32) as i16;
                qvals[i] = clamped;
            }
            let rle = rle_encode(&qvals);
            (rle, -scale, true) // negative scale flags RLE on the wire
        } else {
            let (bytes, scale) = quant.quantize(&residual);
            (bytes, scale, false)
        };
        // Closed-loop update at edge
        let deq = quant.dequantize(&bytes, scale);
        let recon = if pred.is_empty() { deq } else { pred.add(&deq).unwrap() };
        predictor.update(&recon).unwrap();

        let req = tonic::Request::new(Frame {
            tick,
            length: data.len() as u32,
            scale,
            residual: bytes.clone(),
        });
        let _ = client.send_frame(req).await?;
        let raw = data.len() * 4;
        let comp = bytes.len() + 4;
        let ratio = raw as f32 / comp as f32;
        writeln!(fh, "{},{},{},{},{}", tick, data.len(), raw, comp, ratio).ok();
        tick += 1;
        tokio::time::sleep(dt).await;
    }
    Ok(())
}

use std::io::Write;
