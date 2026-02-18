use clap::Parser;
use nangila_ai::MomentumPredictor;
use nangila_core::{Predictor, Quantizer};
use nangila_hpc::ErrorBoundedQuantizer;
use nangila_math::FixedPointBuffer;
use std::sync::Arc;

#[path = "../generated/rotor.rs"]
mod rotor_pb;

use rotor_pb::rotor::rotor_service_server::{RotorService, RotorServiceServer};
use rotor_pb::rotor::{Ack, Frame};
use tonic::{transport::Server, Request, Response, Status};

#[derive(Parser, Debug, Clone)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:50051")]
    listen: String,
    #[arg(long, default_value = "server_metrics.csv")]
    metrics: String,
    #[arg(long, default_value_t = 0.9)]
    beta: f32,
}

struct RotorServer {
    predictor: tokio::sync::Mutex<MomentumPredictor>,
    quant: ErrorBoundedQuantizer,
    metrics: tokio::sync::Mutex<std::fs::File>,
}

#[tonic::async_trait]
impl RotorService for RotorServer {
    async fn send_frame(&self, request: Request<Frame>) -> Result<Response<Ack>, Status> {
        let frame = request.into_inner();
        let len = frame.length as usize;
        let scale = frame.scale;
        let bytes = frame.residual;

        // Predictor prediction
        let pred = {
            let p = self.predictor.lock().await;
            p.predict().map_err(|e| Status::internal(format!("predict error: {}", e)))?
        };
        // Dequantize residual (scale<0 => RLE+i16 format)
        let deq = if scale < 0.0 {
            let vals = rle_decode(&bytes);
            let scale = -scale;
            let mut f = Vec::with_capacity(vals.len());
            for v in vals { f.push((v as f32) * scale); }
            FixedPointBuffer::from_f32(&f)
        } else {
            self.quant.dequantize(&bytes, scale)
        };
        // Reconstruct
        let recon = if pred.is_empty() { deq } else { pred.add(&deq).map_err(|e| Status::internal(format!("add error: {}", e)))? };
        // Trim to length (in case of padding)
        let mut f = recon.to_f32();
        f.truncate(len);
        let recon_buf = FixedPointBuffer::from_f32(&f);

        // Update predictor with reconstruction
        {
            let mut p = self.predictor.lock().await;
            p.update(&recon_buf).map_err(|e| Status::internal(format!("update error: {}", e)))?;
        }

        // Log metrics (raw vs comp)
        let raw = len * 4; // FP32
        let comp = bytes.len() + 4; // bytes + scale(f32) approx
        let ratio = raw as f32 / comp as f32;
        {
            use std::io::Write;
            let mut fh = self.metrics.lock().await;
            writeln!(fh, "tick,{},len,{},raw,{},comp,{},ratio,{:.4}", frame.tick, len, raw, comp, ratio).ok();
        }

        Ok(Response::new(Ack { ok: true }))
    }
}

// RLE decoder matching client encoding
fn rle_decode(bytes: &[u8]) -> Vec<i16> {
    let mut out = Vec::new();
    let mut ptr = 0;
    while ptr < bytes.len() {
        let opcode = bytes[ptr];
        ptr += 1;
        if opcode >= 128 {
            let count = (opcode - 128 + 1) as usize;
            out.extend(std::iter::repeat(0i16).take(count));
        } else {
            let count = (opcode + 1) as usize;
            for _ in 0..count {
                if ptr + 2 > bytes.len() { break; }
                let v = i16::from_le_bytes([bytes[ptr], bytes[ptr+1]]);
                ptr += 2;
                out.push(v);
            }
        }
    }
    out
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let addr = args.listen.parse()?;
    let predictor = MomentumPredictor::new(args.beta);
    let quant = ErrorBoundedQuantizer::new(0.05);
    let fh = std::fs::File::create(&args.metrics)?;
    let svc = RotorServer {
        predictor: tokio::sync::Mutex::new(predictor),
        quant,
        metrics: tokio::sync::Mutex::new(fh),
    };

    println!("Rotor server listening on {}", args.listen);
    Server::builder()
        .add_service(RotorServiceServer::new(svc))
        .serve(addr)
        .await?;
    Ok(())
}
