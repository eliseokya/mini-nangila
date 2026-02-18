fn main() {
    // Only attempt to compile protos if building binaries (user-invoked)
    // This keeps CI happy since this crate is outside the workspace and not built by default.
    if std::env::var("CARGO_FEATURE_NET").is_ok() || std::env::var("BUILD_PROTO").is_ok() {
        tonic_build::configure()
            .build_server(true)
            .build_client(true)
            .compile(&["proto/rotor.proto"], &["proto"]) // proto out dir
            .expect("failed to compile protos");
    }
}

