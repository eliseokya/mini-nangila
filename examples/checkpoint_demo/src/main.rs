use clap::Parser;
use anyhow::{Result, Context};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Output HDF5 file
    #[arg(long, default_value = "traj_ckpt.h5")]
    output: String,

    /// Dataset name under root
    #[arg(long, default_value = "traj")]
    dataset: String,

    /// Total length of the 1D signal
    #[arg(long, default_value_t = 50_000)]
    len: usize,

    /// Chunk size
    #[arg(long, default_value_t = 1024)]
    chunk: usize,

    /// Error bound epsilon
    #[arg(long, default_value_t = 1e-3)]
    epsilon: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let data: Vec<f32> = (0..args.len)
        .map(|i| {
            let t = i as f32 * 0.01;
            (t * 0.5).cos() + ((i % 100) as f32) * 1e-4
        })
        .collect();

    #[cfg(feature = "hdf5")]
    {
        use nangila_checkpoint::h5;
        h5::write_chunked_dataset(&args.output, &args.dataset, &data, args.chunk, args.epsilon, "ErrorBoundedINT16")
            .context("write chunked compressed")?;
        let out = h5::read_compressed_1d(&args.output, &args.dataset).context("read compressed")?;

        let mut max_err = 0.0f32;
        for i in 0..args.len { max_err = max_err.max((out[i] - data[i]).abs()); }
        println!(
            "Wrote {} (dataset '{}'), len={} chunk={} eps={}  max_err={:.6}",
            args.output, args.dataset, args.len, args.chunk, args.epsilon, max_err
        );
    }

    #[cfg(not(feature = "hdf5"))]
    {
        println!("This demo requires HDF5. Rebuild with: --features hdf5");
    }

    Ok(())
}

