use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use clap::Parser;
use nangila_core::{Predictor, Quantizer};
use nangila_hpc::{ErrorBoundedQuantizer, LinearPredictor, RunLengthQuantizer, BlockSparseQuantizer, BlockSparseRleQuantizer};
use nangila_math::FixedPointBuffer;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to input LAMMPS dump (optional). If omitted, generates synthetic data.
    #[arg(long)]
    input: Option<PathBuf>,

    /// Output compressed file
    #[arg(long, default_value = "compressed.nz")] 
    output: PathBuf,

    /// Decompress an existing file (overrides other flags)
    #[arg(long)]
    decompress: bool,

    /// Epsilon error bound
    #[arg(long, default_value_t = 1e-3)]
    epsilon: f32,

    /// Enable RLE on error-bounded residuals (often 2-10x extra on smooth data)
    #[arg(long)]
    use_rle: bool,

    /// Use block-sparse (mask+payload) encoding with error-bounded INT16
    #[arg(long)]
    use_block_sparse: bool,

    /// Use block-sparse + mask-RLE encoding with error-bounded INT16
    #[arg(long)]
    use_block_sparse_rle: bool,

    /// Number of particles for synthetic data
    #[arg(long, default_value_t = 100_000)]
    particles: usize,

    /// Number of steps for synthetic data
    #[arg(long, default_value_t = 100)]
    steps: usize,

    /// Generate a synthetic LAMMPS ASCII dump to this path and exit
    #[arg(long)]
    generate: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.decompress {
        let recon = read_and_decompress(&args.output)?;
        println!("Decompressed {} values", recon.len());
        return Ok(());
    }

    if let Some(gen_path) = args.generate.as_ref() {
        println!("Generating synthetic LAMMPS dump: {:?} ({} atoms, {} steps)", gen_path, args.particles, args.steps);
        write_lammps_dump(gen_path, args.particles, args.steps)?;
        println!("✓ Wrote synthetic dump");
        return Ok(());
    }

    // Load LAMMPS dump if provided; otherwise synthetic
    let (data, particles, steps) = if let Some(path) = args.input.as_ref() {
        println!("Reading LAMMPS dump: {}", path.display());
        let snaps = read_lammps_dump(path).context("parse lammps dump")?;
        if snaps.is_empty() { anyhow::bail!("No snapshots parsed from dump") }
        let particles = snaps[0].len();
        let steps = snaps.len();
        let mut flat = Vec::with_capacity(particles * steps);
        for s in &snaps { flat.extend_from_slice(s); }
        (flat, particles, steps)
    } else {
        let vec = generate_synthetic(args.particles, args.steps);
        (vec, args.particles, args.steps)
    };

    // Predictor and quantizer
    let mut predictor = LinearPredictor::new();

    // Compress sequential snapshots; write a simple container format
    // Header: b"NZCP" [u8 version=1] [u8 codec] [u32 particles] [u32 steps]
    //   codec: 1 = ErrorBoundedINT16, 2 = ErrorBoundedINT16_RLE, 3 = ErrorBoundedINT16_BlockSparse, 4 = ErrorBoundedINT16_BlockSparse_RLE
    // For each step: [f32 scale] [u32 nbytes] [bytes...]
    let mut f = File::create(&args.output).context("create output")?;
    f.write_all(b"NZCP").context("write magic")?;
    f.write_u8(1).context("write version")?;
    let codec_id: u8 = if args.use_block_sparse_rle { 4 } else if args.use_block_sparse { 3 } else if args.use_rle { 2 } else { 1 };
    f.write_u8(codec_id).context("write codec")?; // 1=EB, 2=EB+RLE
    f.write_u32::<LittleEndian>(particles as u32).context("write particles")?;
    f.write_u32::<LittleEndian>(steps as u32).context("write steps")?;

    let mut total_raw = 0usize;
    let mut total_comp = 0usize;
    let mut max_abs_err = 0.0f32;

    if args.use_block_sparse_rle {
        let quantizer = BlockSparseRleQuantizer::new(args.epsilon);
        println!("Codec: ErrorBoundedINT16 + BlockSparse+RLE (ε = {:.3e})", args.epsilon);
        for (step_idx, chunk) in data.chunks(particles).enumerate() {
            let buf = FixedPointBuffer::from_f32(chunk);
            total_raw += particles * 4;

            let pred = predictor.predict().expect("predict");
            let residual = if pred.is_empty() { buf.clone() } else { buf.sub(&pred).expect("sub") };
            let (bytes, scale) = quantizer.quantize(&residual);
            total_comp += bytes.len() + 4 + 4; // scale + length

            let deq = quantizer.dequantize(&bytes, scale);
            let recon = if pred.is_empty() { deq } else { pred.add(&deq).expect("add") };
            predictor.update(&recon).expect("update");

            let orig = buf.to_f32();
            let rec = recon.to_f32();
            for i in 0..orig.len() { let e = (orig[i] - rec[i]).abs(); if e > max_abs_err { max_abs_err = e; } }

            f.write_f32::<LittleEndian>(scale).context("write scale")?;
            f.write_u32::<LittleEndian>(bytes.len() as u32).context("write len")?;
            f.write_all(&bytes).context("write bytes")?;

            if step_idx % 10 == 0 { let ratio = total_raw as f32 / total_comp as f32; println!("step {:4}  ratio {:6.2}×  max_err {:.6}", step_idx, ratio, max_abs_err); }
        }
    } else if args.use_block_sparse {
        let quantizer = BlockSparseQuantizer::new(args.epsilon);
        println!("Codec: ErrorBoundedINT16 + BlockSparse (ε = {:.3e})", args.epsilon);
        for (step_idx, chunk) in data.chunks(particles).enumerate() {
            let buf = FixedPointBuffer::from_f32(chunk);
            total_raw += particles * 4;

            let pred = predictor.predict().expect("predict");
            let residual = if pred.is_empty() { buf.clone() } else { buf.sub(&pred).expect("sub") };
            let (bytes, scale) = quantizer.quantize(&residual);
            total_comp += bytes.len() + 4 + 4; // scale + length

            // Update predictor with reconstruction
            let deq = quantizer.dequantize(&bytes, scale);
            let recon = if pred.is_empty() { deq } else { pred.add(&deq).expect("add") };
            predictor.update(&recon).expect("update");

            // Track error bound
            let orig = buf.to_f32();
            let rec = recon.to_f32();
            for i in 0..orig.len() {
                let e = (orig[i] - rec[i]).abs();
                if e > max_abs_err { max_abs_err = e; }
            }

            // Write this step
            f.write_f32::<LittleEndian>(scale).context("write scale")?;
            f.write_u32::<LittleEndian>(bytes.len() as u32).context("write len")?;
            f.write_all(&bytes).context("write bytes")?;

            if step_idx % 10 == 0 {
                let ratio = total_raw as f32 / total_comp as f32;
                println!("step {:4}  ratio {:6.2}×  max_err {:.6}", step_idx, ratio, max_abs_err);
            }
        }
    } else if args.use_rle {
        let quantizer = RunLengthQuantizer::new(args.epsilon);
        println!("Codec: ErrorBoundedINT16 + RLE (ε = {:.3e})", args.epsilon);
        for (step_idx, chunk) in data.chunks(particles).enumerate() {
            let buf = FixedPointBuffer::from_f32(chunk);
            total_raw += particles * 4;

            let pred = predictor.predict().expect("predict");
            let residual = if pred.is_empty() { buf.clone() } else { buf.sub(&pred).expect("sub") };
            let (bytes, scale) = quantizer.quantize(&residual);
            total_comp += bytes.len() + 4 + 4; // scale + length

            // Update predictor with reconstruction
            let deq = quantizer.dequantize(&bytes, scale);
            let recon = if pred.is_empty() { deq } else { pred.add(&deq).expect("add") };
            predictor.update(&recon).expect("update");

            // Track error bound
            let orig = buf.to_f32();
            let rec = recon.to_f32();
            for i in 0..orig.len() {
                let e = (orig[i] - rec[i]).abs();
                if e > max_abs_err { max_abs_err = e; }
            }

            // Write this step
            f.write_f32::<LittleEndian>(scale).context("write scale")?;
            f.write_u32::<LittleEndian>(bytes.len() as u32).context("write len")?;
            f.write_all(&bytes).context("write bytes")?;

            if step_idx % 10 == 0 {
                let ratio = total_raw as f32 / total_comp as f32;
                println!("step {:4}  ratio {:6.2}×  max_err {:.6}", step_idx, ratio, max_abs_err);
            }
        }
    } else {
        let quantizer = ErrorBoundedQuantizer::new(args.epsilon);
        println!("Codec: ErrorBoundedINT16 (ε = {:.3e})", args.epsilon);
        for (step_idx, chunk) in data.chunks(particles).enumerate() {
            let buf = FixedPointBuffer::from_f32(chunk);
            total_raw += particles * 4;

            let pred = predictor.predict().expect("predict");
            let residual = if pred.is_empty() { buf.clone() } else { buf.sub(&pred).expect("sub") };
            let (bytes, scale) = quantizer.quantize(&residual);
            total_comp += bytes.len() + 4 + 4; // scale + length

            // Update predictor with reconstruction
            let deq = quantizer.dequantize(&bytes, scale);
            let recon = if pred.is_empty() { deq } else { pred.add(&deq).expect("add") };
            predictor.update(&recon).expect("update");

            // Track error bound
            let orig = buf.to_f32();
            let rec = recon.to_f32();
            for i in 0..orig.len() {
                let e = (orig[i] - rec[i]).abs();
                if e > max_abs_err { max_abs_err = e; }
            }

            // Write this step
            f.write_f32::<LittleEndian>(scale).context("write scale")?;
            f.write_u32::<LittleEndian>(bytes.len() as u32).context("write len")?;
            f.write_all(&bytes).context("write bytes")?;

            if step_idx % 10 == 0 {
                let ratio = total_raw as f32 / total_comp as f32;
                println!("step {:4}  ratio {:6.2}×  max_err {:.6}", step_idx, ratio, max_abs_err);
            }
        }
    }

    let ratio = total_raw as f32 / total_comp as f32;
    println!("DONE  compressed={:.2} MB  raw={:.2} MB  ratio={:.2}×  max_err={:.6}",
        total_comp as f32 / 1e6,
        total_raw as f32 / 1e6,
        ratio,
        max_abs_err,
    );

    Ok(())
}

fn read_and_decompress(path: &PathBuf) -> Result<Vec<f32>> {
    let mut f = File::open(path).context("open compressed")?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).context("read magic")?;
    if &magic != b"NZCP" { anyhow::bail!("bad magic") }

    let _version = f.read_u8().context("read ver")?;
    let codec = f.read_u8().context("read codec")?; // 1=EB, 2=EB+RLE
    let particles = f.read_u32::<LittleEndian>().context("read particles")? as usize;
    let steps = f.read_u32::<LittleEndian>().context("read steps")? as usize;

    let mut all = Vec::with_capacity(particles * steps);
    match codec {
        1 => {
            let quant = ErrorBoundedQuantizer::new(1e-3); // scale read per step
            for _ in 0..steps {
                let scale = f.read_f32::<LittleEndian>().context("read scale")?;
                let len = f.read_u32::<LittleEndian>().context("read len")? as usize;
                let mut bytes = vec![0u8; len];
                f.read_exact(&mut bytes).context("read bytes")?;
                let rec = quant.dequantize(&bytes, scale).to_f32();
                all.extend_from_slice(&rec);
            }
        }
        2 => {
            let quant = RunLengthQuantizer::new(1e-3);
            for _ in 0..steps {
                let scale = f.read_f32::<LittleEndian>().context("read scale")?;
                let len = f.read_u32::<LittleEndian>().context("read len")? as usize;
                let mut bytes = vec![0u8; len];
                f.read_exact(&mut bytes).context("read bytes")?;
                let rec = quant.dequantize(&bytes, scale).to_f32();
                all.extend_from_slice(&rec);
            }
        }
        3 => {
            let quant = BlockSparseQuantizer::new(1e-3);
            for _ in 0..steps {
                let scale = f.read_f32::<LittleEndian>().context("read scale")?;
                let len = f.read_u32::<LittleEndian>().context("read len")? as usize;
                let mut bytes = vec![0u8; len];
                f.read_exact(&mut bytes).context("read bytes")?;
                let rec = quant.dequantize(&bytes, scale).to_f32();
                all.extend_from_slice(&rec);
            }
        }
        4 => {
            let quant = BlockSparseRleQuantizer::new(1e-3);
            for _ in 0..steps {
                let scale = f.read_f32::<LittleEndian>().context("read scale")?;
                let len = f.read_u32::<LittleEndian>().context("read len")? as usize;
                let mut bytes = vec![0u8; len];
                f.read_exact(&mut bytes).context("read bytes")?;
                let rec = quant.dequantize(&bytes, scale).to_f32();
                all.extend_from_slice(&rec);
            }
        }
        _ => anyhow::bail!("unknown codec {}", codec),
    }
    Ok(all)
}

fn generate_synthetic(particles: usize, steps: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(particles * steps);
    for t in 0..steps {
        let tt = t as f32 * 0.1;
        for i in 0..particles {
            let phase = tt + (i as f32) * 0.01;
            let noise = (i % 100) as f32 * 0.0001;
            out.push(phase.cos() + noise);
        }
    }
    out
}

/// Minimal LAMMPS dump reader (ASCII) for format:
/// ITEM: TIMESTEP
/// <int>
/// ITEM: NUMBER OF ATOMS
/// <int>
/// ITEM: BOX BOUNDS ... (3 lines)
/// ITEM: ATOMS id type x y z ... (at least x y z present)
/// <id> <type> <x> <y> <z> ... repeated for N atoms
///
/// Returns: Vec of snapshots, each snapshot is flattened [x0,y0,z0,x1,y1,z1,...] length = 3*N
fn read_lammps_dump(path: &std::path::Path) -> Result<Vec<Vec<f32>>> {
    use std::io::{BufRead, BufReader};
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut rdr = BufReader::new(f);
    let mut line = String::new();
    let mut snapshots: Vec<Vec<f32>> = Vec::new();
    loop {
        line.clear();
        if rdr.read_line(&mut line)? == 0 { break; }
        if !line.trim().starts_with("ITEM: TIMESTEP") { continue; }
        // timestep
        line.clear();
        rdr.read_line(&mut line)?;
        // number of atoms
        line.clear();
        rdr.read_line(&mut line)?; // expect ITEM: NUMBER OF ATOMS
        if !line.starts_with("ITEM: NUMBER OF ATOMS") {
            anyhow::bail!("unexpected format, expected NUMBER OF ATOMS line")
        }
        line.clear();
        rdr.read_line(&mut line)?;
        let natoms: usize = line.trim().parse().context("parse natoms")?;
        // box bounds (3 lines after ITEM: BOX BOUNDS)
        line.clear();
        rdr.read_line(&mut line)?; // ITEM: BOX BOUNDS
        if !line.starts_with("ITEM: BOX BOUNDS") {
            anyhow::bail!("unexpected format, expected BOX BOUNDS line")
        }
        for _ in 0..3 { line.clear(); rdr.read_line(&mut line)?; }
        // atoms header
        line.clear();
        rdr.read_line(&mut line)?; // ITEM: ATOMS ...
        if !line.starts_with("ITEM: ATOMS") { anyhow::bail!("expected ATOMS header") }
        // parse column indices for x,y,z
        let cols: Vec<&str> = line.trim().split_whitespace().collect();
        // Fields start after 'ITEM:' and 'ATOMS'
        let fields: Vec<&str> = cols.iter().skip(2).copied().collect();
        let mut x_idx = None; let mut y_idx = None; let mut z_idx = None;
        for (i, &col) in fields.iter().enumerate() {
            match col { "x" => x_idx = Some(i), "y" => y_idx = Some(i), "z" => z_idx = Some(i), _ => {} }
        }
        let (xi, yi, zi) = (
            x_idx.ok_or_else(|| anyhow::anyhow!("x column not found"))?,
            y_idx.ok_or_else(|| anyhow::anyhow!("y column not found"))?,
            z_idx.ok_or_else(|| anyhow::anyhow!("z column not found"))?,
        );
        // read natoms lines
        let mut snap = Vec::with_capacity(natoms * 3);
        for _ in 0..natoms {
            line.clear();
            if rdr.read_line(&mut line)? == 0 { anyhow::bail!("unexpected EOF reading atoms") }
            let toks: Vec<&str> = line.split_whitespace().collect();
            // Tokens correspond to fields exactly
            let x: f32 = toks.get(xi).ok_or_else(|| anyhow::anyhow!("x missing"))?.parse().context("x parse")?;
            let y: f32 = toks.get(yi).ok_or_else(|| anyhow::anyhow!("y missing"))?.parse().context("y parse")?;
            let z: f32 = toks.get(zi).ok_or_else(|| anyhow::anyhow!("z missing"))?.parse().context("z parse")?;
            snap.push(x); snap.push(y); snap.push(z);
        }
        snapshots.push(snap);
    }
    Ok(snapshots)
}

/// Generate a simple ASCII LAMMPS dump with x,y,z for `atoms` across `steps` timesteps.
fn write_lammps_dump(path: &std::path::Path, atoms: usize, steps: usize) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    for t in 0..steps {
        writeln!(f, "ITEM: TIMESTEP")?;
        writeln!(f, "{}", t)?;
        writeln!(f, "ITEM: NUMBER OF ATOMS")?;
        writeln!(f, "{}", atoms)?;
        writeln!(f, "ITEM: BOX BOUNDS xx yy zz")?;
        writeln!(f, "0 10")?; writeln!(f, "0 10")?; writeln!(f, "0 10")?;
        writeln!(f, "ITEM: ATOMS id type x y z")?;
        let tt = t as f32 * 0.1;
        for i in 0..atoms {
            let phase = tt + (i as f32) * 0.001;
            let x = phase.cos();
            let y = phase.sin();
            let z = (phase*0.5).cos();
            writeln!(f, "{} {} {:.6} {:.6} {:.6}", i+1, 1, x, y, z)?;
        }
    }
    Ok(())
}
