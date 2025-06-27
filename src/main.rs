#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(allocator_api)]

use anyhow::Result;
use clap::Parser;
use core::num::ParseIntError;
use log::{info, Level};
use plonky2::gates::noop::NoopGate;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{
    CircuitConfig, CommonCircuitData, VerifierCircuitTarget, VerifierOnlyCircuitData,
};
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig, Hasher, PoseidonGoldilocksConfig};
use plonky2::plonk::proof::{CompressedProofWithPublicInputs, ProofWithPublicInputs};
use plonky2::plonk::prover::prove;
use plonky2::util::timing::TimingTree;
use plonky2_ed25519::curve::eddsa::{
    SAMPLE_MSG1, SAMPLE_MSG2, SAMPLE_PK1, SAMPLE_SIG1, SAMPLE_SIG2,
};
use plonky2_ed25519::gadgets::eddsa::{fill_circuits, make_verify_circuits};
use plonky2_field::extension::Extendable;
use plonky2_field::goldilocks_field::GoldilocksField;
use std::alloc::{AllocError, Allocator, Layout};
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;

use plonky2::fri::oracle::MyAllocator;
use plonky2_field::fft::fft_root_table;
use plonky2_field::zero_poly_coset::ZeroPolyOnCoset;
use plonky2_util::{log2_ceil, log2_strict};

use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::mpsc::{self, Sender, UnboundedReceiver};
use tokio::sync::Notify;
use tokio::time::{sleep, timeout, Duration};

use jemallocator::Jemalloc;
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[cfg(feature = "cuda")]
use plonky2::plonk::prover::my_prove;

#[cfg(feature = "cuda")]
use plonky2::fri::oracle::CudaInnerContext;

#[cfg(feature = "cuda")]
use rustacuda::memory::DeviceBuffer;
#[cfg(feature = "cuda")]
use rustacuda::memory::{cuda_malloc, DeviceBox};
#[cfg(feature = "cuda")]
use rustacuda::prelude::*;

// #[macro_use]
// extern crate rustacuda;
// extern crate rustacuda_core;

// extern crate cuda;
// use cuda::runtime::{CudaError, cudaMalloc, cudaMemcpy, cudaFree};
// use cuda::runtime::raw::{cudaError_t, cudaError_enum};

type ProofTuple<F, C, const D: usize> = (
    ProofWithPublicInputs<F, C, D>,
    VerifierOnlyCircuitData<C, D>,
    CommonCircuitData<F, D>,
);

#[cfg(feature = "cuda")]
fn prove_ed25519_cuda<
    F: RichField + Extendable<D> + rustacuda::memory::DeviceCopy,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    msg: &[u8],
    sigv: &[u8],
    pkv: &[u8],
) -> Result<()>
where
    [(); C::Hasher::HASH_SIZE]:,
{
    let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::wide_ecc_config());

    let targets = make_verify_circuits(&mut builder, msg.len());
    let mut pw = PartialWitness::new();
    fill_circuits::<F, D>(&mut pw, msg, sigv, pkv, &targets);

    println!(
        "Constructing inner proof with {} gates",
        builder.num_gates()
    );
    println!("index num: {}", builder.virtual_target_index);

    let data = builder.build::<C>();

    let mut ctx;
    {
        rustacuda::init(CudaFlags::empty()).unwrap();
        let device_index = 0;
        let device = rustacuda::prelude::Device::get_device(device_index).unwrap();
        let _ctx =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                .unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let stream2 = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let poly_num: usize = 234;
        let values_num_per_poly = 1 << 18;
        let blinding = false;
        const SALT_SIZE: usize = 4;
        let rate_bits = 3;
        let cap_height = 4;

        let lg_n = log2_strict(values_num_per_poly);
        let n_inv = F::inverse_2exp(lg_n);
        let _n_inv_ptr: *const F = &n_inv;

        let fft_root_table_max = fft_root_table(1 << (lg_n + rate_bits)).concat();
        let fft_root_table_deg = fft_root_table(1 << lg_n).concat();

        let salt_size = if blinding { SALT_SIZE } else { 0 };
        let values_flatten_len = poly_num * values_num_per_poly;
        let ext_values_flatten_len =
            (values_flatten_len + salt_size * values_num_per_poly) * (1 << rate_bits);
        let mut ext_values_flatten: Vec<F> = Vec::with_capacity(ext_values_flatten_len);
        unsafe {
            ext_values_flatten.set_len(ext_values_flatten_len);
        }

        let mut values_flatten: Vec<F, MyAllocator> =
            Vec::with_capacity_in(values_flatten_len, MyAllocator {});
        unsafe {
            values_flatten.set_len(values_flatten_len);
        }

        let (values_flatten2, ext_values_flatten2) = {
            let poly_num = 20;
            let values_flatten_len = poly_num * values_num_per_poly;
            let ext_values_flatten_len =
                (values_flatten_len + salt_size * values_num_per_poly) * (1 << rate_bits);
            let mut ext_values_flatten: Vec<F> = Vec::with_capacity(ext_values_flatten_len);
            unsafe {
                ext_values_flatten.set_len(ext_values_flatten_len);
            }

            let mut values_flatten: Vec<F, MyAllocator> =
                Vec::with_capacity_in(values_flatten_len, MyAllocator {});
            unsafe {
                values_flatten.set_len(values_flatten_len);
            }
            (values_flatten, ext_values_flatten)
        };

        let (values_flatten3, ext_values_flatten3) = {
            let poly_num = data.common.config.num_challenges * (1 << rate_bits);
            let values_flatten_len = poly_num * values_num_per_poly;
            let ext_values_flatten_len =
                (values_flatten_len + salt_size * values_num_per_poly) * (1 << rate_bits);
            let mut ext_values_flatten: Vec<F> = Vec::with_capacity(ext_values_flatten_len);
            unsafe {
                ext_values_flatten.set_len(ext_values_flatten_len);
            }

            let mut values_flatten: Vec<F, MyAllocator> =
                Vec::with_capacity_in(values_flatten_len, MyAllocator {});
            unsafe {
                values_flatten.set_len(values_flatten_len);
            }
            (values_flatten, ext_values_flatten)
        };

        let len_cap = 1 << cap_height;
        let num_digests = 2 * (values_num_per_poly * (1 << rate_bits) - len_cap);
        let num_digests_and_caps = num_digests + len_cap;
        let mut digests_and_caps_buf: Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash> =
            Vec::with_capacity(num_digests_and_caps);
        unsafe {
            digests_and_caps_buf.set_len(num_digests_and_caps);
        }

        let digests_and_caps_buf2 = digests_and_caps_buf.clone();
        let digests_and_caps_buf3 = digests_and_caps_buf.clone();

        // let mut values_device = unsafe{
        //     DeviceBuffer::<F>::uninitialized(values_flatten_len)?
        // };

        let pad_extvalues_len = ext_values_flatten.len();
        // let mut ext_values_device = {
        //         let mut values_device = unsafe {
        //             DeviceBuffer::<F>::uninitialized(
        //             pad_extvalues_len
        //                 + ext_values_flatten_len
        //                 + digests_and_caps_buf.len()*4
        //             )
        //         }.unwrap();
        //
        //         values_device
        // };

        let cache_mem_device = {
            let cache_mem_device = unsafe {
                DeviceBuffer::<F>::uninitialized(
                    // values_flatten_len +
                    pad_extvalues_len + ext_values_flatten_len + digests_and_caps_buf.len() * 4,
                )
            }
            .unwrap();

            cache_mem_device
        };

        let root_table_device = {
            let root_table_device = DeviceBuffer::from_slice(&fft_root_table_deg).unwrap();
            root_table_device
        };

        let root_table_device2 = {
            let root_table_device = DeviceBuffer::from_slice(&fft_root_table_max).unwrap();
            root_table_device
        };

        let constants_sigmas_commitment_leaves_device = DeviceBuffer::from_slice(
            &data
                .prover_only
                .constants_sigmas_commitment
                .merkle_tree
                .leaves
                .concat(),
        )
        .unwrap();

        let shift_powers = F::coset_shift()
            .powers()
            .take(1 << (lg_n))
            .collect::<Vec<F>>();
        let shift_powers_device = {
            let shift_powers_device = DeviceBuffer::from_slice(&shift_powers).unwrap();
            shift_powers_device
        };

        let shift_inv_powers = F::coset_shift()
            .powers()
            .take(1 << (lg_n + rate_bits))
            .map(|f| f.inverse())
            .collect::<Vec<F>>();
        let shift_inv_powers_device = {
            let shift_inv_powers_device = DeviceBuffer::from_slice(&shift_inv_powers).unwrap();
            shift_inv_powers_device
        };

        // unsafe
        // {
        //     let mut file = File::create("inv-powers.bin").unwrap();
        //     let v = shift_inv_powers;
        //     file.write_all(std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()*8));
        // }

        let quotient_degree_bits = log2_ceil(data.common.quotient_degree_factor);
        let points = F::two_adic_subgroup(data.common.degree_bits() + quotient_degree_bits);

        let z_h_on_coset = ZeroPolyOnCoset::new(data.common.degree_bits(), quotient_degree_bits);

        let points_device = DeviceBuffer::from_slice(&points).unwrap();
        let z_h_on_coset_evals_device = DeviceBuffer::from_slice(&z_h_on_coset.evals).unwrap();
        let z_h_on_coset_inverses_device =
            DeviceBuffer::from_slice(&z_h_on_coset.inverses).unwrap();
        let k_is_device = DeviceBuffer::from_slice(&data.common.k_is).unwrap();

        ctx = plonky2::fri::oracle::CudaInvContext {
            inner: CudaInnerContext { stream, stream2 },
            ext_values_flatten: Arc::new(ext_values_flatten),
            values_flatten: Arc::new(values_flatten),
            digests_and_caps_buf: Arc::new(digests_and_caps_buf),

            ext_values_flatten2: Arc::new(ext_values_flatten2),
            values_flatten2: Arc::new(values_flatten2),
            digests_and_caps_buf2: Arc::new(digests_and_caps_buf2),

            ext_values_flatten3: Arc::new(ext_values_flatten3),
            values_flatten3: Arc::new(values_flatten3),
            digests_and_caps_buf3: Arc::new(digests_and_caps_buf3),

            cache_mem_device,
            second_stage_offset: ext_values_flatten_len,
            root_table_device,
            root_table_device2,
            constants_sigmas_commitment_leaves_device,
            shift_powers_device,
            shift_inv_powers_device,

            points_device,
            z_h_on_coset_evals_device,
            z_h_on_coset_inverses_device,
            k_is_device,

            ctx: _ctx,
        };
    }

    tokio::task::spawn_blocking(move || loop {
        let mut timing = TimingTree::new("prove gpu", Level::Debug);
        println!(
            "num_gate_constraints: {}, num_constraints: {}, selectors_info: {:?}",
            data.common.num_gate_constraints, data.common.num_constants, data.common.selectors_info,
        );
        let proof = my_prove(
            &data.prover_only,
            &data.common,
            pw.clone(),
            &mut timing,
            &mut ctx,
        )?;

        timing.print();

        let timing = TimingTree::new("verify", Level::Info);
        data.verify(proof.clone()).expect("verify error");
        timing.print();
    });

    tokio::task::spawn_blocking(move || loop {
        let mut timing = TimingTree::new("prove gpu", Level::Debug);
        println!(
            "num_gate_constraints: {}, num_constraints: {}, selectors_info: {:?}",
            data.common.num_gate_constraints, data.common.num_constants, data.common.selectors_info,
        );
        let proof = my_prove(
            &data.prover_only,
            &data.common,
            pw.clone(),
            &mut timing,
            &mut ctx,
        )?;

        timing.print();

        let timing = TimingTree::new("verify", Level::Info);
        data.verify(proof.clone()).expect("verify error");
        timing.print();
    });

    Ok(())
}

fn prove_ed25519<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    msg: &[u8],
    sigv: &[u8],
    pkv: &[u8],
) -> Result<ProofTuple<F, C, D>>
where
    [(); C::Hasher::HASH_SIZE]:,
{
    let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::wide_ecc_config());

    let targets = make_verify_circuits(&mut builder, msg.len());
    let mut pw = PartialWitness::new();
    fill_circuits::<F, D>(&mut pw, msg, sigv, pkv, &targets);

    println!(
        "Constructing inner proof with {} gates",
        builder.num_gates()
    );
    println!("index num: {}", builder.virtual_target_index);

    let data = builder.build::<C>();

    let mut timing = TimingTree::new("prove cpu", Level::Debug);
    println!(
        "num_gate_constraints: {}, num_constraints: {}, selectors_info: {:?}",
        data.common.num_gate_constraints, data.common.num_constants, data.common.selectors_info,
    );
    let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
    data.verify(proof.clone()).expect("verify error");

    timing.print();

    Ok((proof, data.verifier_only, data.common))
}

#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    msg: Option<String>,
    #[arg(short, long)]
    pk: Option<String>,
    #[arg(short, long)]
    sig: Option<String>,
}

pub fn decode_hex(s: &String) -> Result<Vec<u8>, ParseIntError> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
        .collect()
}

fn main() -> Result<()> {
    // Initialize logging
    let mut builder = env_logger::Builder::from_default_env();
    builder.format_timestamp(None);
    // builder.filter_level(LevelFilter::Debug);
    builder.try_init()?;

    let args = Cli::parse();

    if args.sig.is_none() || args.pk.is_none() || args.msg.is_none() {
        println!("The required arguments were not provided: --msg MSG_IN_HEX  --pk PUBLIC_KEY_IN_HEX  --sig SIGNATURE_IN_HEX");
        return Ok(());
    }

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
    // unsafe {
    //     let f = std::mem::transmute::<u64, F>(0xfffffffeffe00001);
    //     // let inv = std::mem::transmute::<u64, F>(0xbfa99fe2edeb56f5);
    //
    //     println!("inv: {:016X}", std::mem::transmute::<F, u64>(f.inverse()));
    //     // println!("n  : {:016X}", std::mem::transmute::<F, u64>(f));
    //     // println!("res: {:016X}", std::mem::transmute::<F, u64>(f * inv));
    //     //
    //     // fn split(x: u128) -> (u64, u64) {
    //     //     (x as u64, (x >> 64) as u64)
    //     // }
    //     //
    //     // println!("{:?}", split(f.0 as u128 * inv.0 as u128));
    // }

    // let inputs = [
    //     GoldilocksField(12057761340118092379),
    //     GoldilocksField(6921394802928742357),
    //     GoldilocksField(401572749463996457),
    //     GoldilocksField(8075242603528285606),
    //     GoldilocksField(16383556155787439553),
    //     GoldilocksField(18045582516498195573),
    //     GoldilocksField(7296969412159674050),
    //     GoldilocksField(8317318176954617326)
    // ];
    // // let res = PoseidonHash::hash_no_pad(&inputs);
    // // let res = hash_n_to_m_no_pad::<F, PoseidonPermutation>(&inputs, 4);
    //
    // let mut state = [F::ZERO; 12];
    //
    // // Absorb all input chunks.
    // state[..inputs.len()].copy_from_slice(&inputs);
    // state = F::poseidon(state);
    //
    // let res = state.into_iter().take(4).collect::<Vec<_>>();
    //
    // // let res = HashOut::from_vec(res);
    // let hex_string: String = unsafe{*std::mem::transmute::<*const _, *const [u8;32]>(res.as_ptr())}.iter().map(|byte| format!("{:02x}", byte)).collect();
    // let result: String = hex_string.chars()
    //     .collect::<Vec<char>>()
    //     .chunks(16)
    //     .map(|chunk| chunk.iter().collect::<String>())
    //     .collect::<Vec<String>>()
    //     .join(", ");
    // println!("cpu hash: {}", result);
    //
    // exit(0);

    #[cfg(feature = "cuda")]
    {
        prove_ed25519_cuda::<F, C, D>(
            decode_hex(&args.msg.unwrap())?.as_slice(),
            decode_hex(&args.sig.unwrap())?.as_slice(),
            decode_hex(&args.pk.unwrap())?.as_slice(),
        )?;
    }

    #[cfg(not(feature = "cuda"))]
    {
        prove_ed25519::<F, C, D>(
            decode_hex(&args.msg.unwrap())?.as_slice(),
            decode_hex(&args.sig.unwrap())?.as_slice(),
            decode_hex(&args.pk.unwrap())?.as_slice(),
        )?;
    }
    return Ok(());
}
