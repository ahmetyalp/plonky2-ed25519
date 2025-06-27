FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 as builder

# Install curl
RUN apt-get update && \
  apt-get install -y curl build-essential git libssl-dev pkg-config

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup override set nightly-2024-06-23

WORKDIR /app
COPY . .

RUN RUSTFLAGS="-Ctarget-cpu=native -Awarnings" cargo build -r --all-features

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

WORKDIR /app
COPY --from=builder /app/target/release/plonky2_ed25519 /app/plonky2_ed25519

ENTRYPOINT [ "/app/plonky2_ed25519", "--msg", "0123456789ABCDEF", "--pk", "9DBB279277D4EFE2E5F114A9AAB25C83FC9509D3B3D3B90929854F5A243AEBCD", "--sig", "2EF7A1AA2FC58D40691236664418ADC903C153ABC0C95D02AC45B436C02081C2B93891B37B17F57C7CDE97B52BBB8F1865C14A92ADA4DC34ED0DE7935346E40E"]
