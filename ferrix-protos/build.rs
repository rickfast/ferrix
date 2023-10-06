use std::io::Result;

fn main() -> Result<()> {
    let protoc_path = protoc_bin_vendored::protoc_bin_path().unwrap();

    std::env::set_var("PROTOC", protoc_path);
    tonic_build::configure()
        .build_server(true)
        .compile(&["api/kserve_v2_grpc.proto"], &["api/"])?;

    Ok(())
}
