use mini_tensorflow::{ParallelOps, SIMDOps, Tensor};
use std::time::Instant;

fn main() {
    println!("SIMD and Parallel Operations Benchmark");

    // Create large tensors for benchmarking
    let size = 1_000_000;
    let a = Tensor::random(vec![size]);
    let b = Tensor::random(vec![size]);

    println!("Tensor size: {} elements", size);

    // Benchmark regular addition
    println!("\n=== Addition Benchmark ===");
    let start = Instant::now();
    let _regular_add = a.add(&b);
    let regular_time = start.elapsed();

    // Benchmark SIMD addition
    let start = Instant::now();
    let _simd_add = a.simd_add(&b);
    let simd_time = start.elapsed();

    // Benchmark parallel addition
    let start = Instant::now();
    let _parallel_add = a.parallel_add(&b);
    let parallel_time = start.elapsed();

    println!("Regular:  {:?}", regular_time);
    println!("SIMD:     {:?}", simd_time);
    println!("Parallel: {:?}", parallel_time);

    if simd_time < regular_time {
        let speedup = regular_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
        println!("SIMD speedup: {:.2}x", speedup);
    }

    if parallel_time < regular_time {
        let speedup = regular_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
        println!("Parallel speedup: {:.2}x", speedup);
    }

    // Benchmark multiplication
    println!("\n=== Multiplication Benchmark ===");
    let start = Instant::now();
    let _regular_mul = a.mul(&b);
    let regular_mul_time = start.elapsed();

    let start = Instant::now();
    let _simd_mul = a.simd_mul(&b);
    let simd_mul_time = start.elapsed();

    let start = Instant::now();
    let _parallel_mul = a.parallel_mul(&b);
    let parallel_mul_time = start.elapsed();

    println!("Regular:  {:?}", regular_mul_time);
    println!("SIMD:     {:?}", simd_mul_time);
    println!("Parallel: {:?}", parallel_mul_time);

    // Benchmark matrix multiplication
    let matrix_size = 500; // Reduced for faster testing
    let matrix_a = Tensor::random(vec![matrix_size, matrix_size]);
    let matrix_b = Tensor::random(vec![matrix_size, matrix_size]);

    println!("\n=== Matrix Multiplication Benchmark ===");
    println!("Matrix size: {}x{}", matrix_size, matrix_size);

    let start = Instant::now();
    let _regular_matmul = matrix_a.matmul(&matrix_b);
    let regular_matmul_time = start.elapsed();

    let start = Instant::now();
    let _parallel_matmul = matrix_a.parallel_matmul(&matrix_b);
    let parallel_matmul_time = start.elapsed();

    println!("Regular:  {:?}", regular_matmul_time);
    println!("Parallel: {:?}", parallel_matmul_time);

    if parallel_matmul_time < regular_matmul_time {
        let speedup =
            regular_matmul_time.as_nanos() as f64 / parallel_matmul_time.as_nanos() as f64;
        println!("Parallel matmul speedup: {:.2}x", speedup);
    }

    // Benchmark ReLU
    println!("\n=== ReLU Benchmark ===");
    let relu_tensor = Tensor::new(
        (0..size).map(|i| (i as f64 / 1000.0) - 500.0).collect(),
        vec![size],
    );

    let start = Instant::now();
    let _regular_relu = relu_tensor.relu();
    let regular_relu_time = start.elapsed();

    let start = Instant::now();
    let _simd_relu = relu_tensor.simd_relu();
    let simd_relu_time = start.elapsed();

    println!("Regular: {:?}", regular_relu_time);
    println!("SIMD:    {:?}", simd_relu_time);

    if simd_relu_time < regular_relu_time {
        let speedup = regular_relu_time.as_nanos() as f64 / simd_relu_time.as_nanos() as f64;
        println!("SIMD ReLU speedup: {:.2}x", speedup);
    }

    // Benchmark sum operations
    println!("\n=== Sum Benchmark ===");
    let start = Instant::now();
    let regular_sum = a.sum();
    let regular_sum_time = start.elapsed();

    let start = Instant::now();
    let simd_sum = a.simd_sum();
    let simd_sum_time = start.elapsed();

    println!(
        "Regular: {:?} (result: {:.2})",
        regular_sum_time, regular_sum
    );
    println!("SIMD:    {:?} (result: {:.2})", simd_sum_time, simd_sum);

    if simd_sum_time < regular_sum_time {
        let speedup = regular_sum_time.as_nanos() as f64 / simd_sum_time.as_nanos() as f64;
        println!("SIMD sum speedup: {:.2}x", speedup);
    }

    // Test correctness
    println!("\n=== Correctness Verification ===");
    let small_a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![8]);
    let small_b = Tensor::new(vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], vec![8]);

    let regular_result = small_a.add(&small_b);
    let simd_result = small_a.simd_add(&small_b);
    let parallel_result = small_a.parallel_add(&small_b);

    println!("Regular result:  {}", regular_result);
    println!("SIMD result:     {}", simd_result);
    println!("Parallel result: {}", parallel_result);

    let max_diff_simd = regular_result
        .data
        .iter()
        .zip(&simd_result.data)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f64::max);

    let max_diff_parallel = regular_result
        .data
        .iter()
        .zip(&parallel_result.data)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f64::max);

    println!("Max difference (SIMD): {:.2e}", max_diff_simd);
    println!("Max difference (Parallel): {:.2e}", max_diff_parallel);

    if max_diff_simd < 1e-10 && max_diff_parallel < 1e-10 {
        println!("✅ All optimizations produce correct results!");
    } else {
        println!("❌ Some optimizations have accuracy issues");
    }

    // Memory bandwidth test
    println!("\n=== Memory Bandwidth Test ===");
    let large_tensor = Tensor::random(vec![10_000_000]);

    let start = Instant::now();
    let _sum1 = large_tensor.sum();
    let bandwidth_time1 = start.elapsed();

    let start = Instant::now();
    let _sum2 = large_tensor.simd_sum();
    let bandwidth_time2 = start.elapsed();

    let data_size_mb = (large_tensor.size() * 8) as f64 / (1024.0 * 1024.0); // 8 bytes per f64
    let bandwidth1 = data_size_mb / bandwidth_time1.as_secs_f64();
    let bandwidth2 = data_size_mb / bandwidth_time2.as_secs_f64();

    println!("Data size: {:.1} MB", data_size_mb);
    println!("Regular bandwidth:  {:.1} MB/s", bandwidth1);
    println!("SIMD bandwidth:     {:.1} MB/s", bandwidth2);

    // Platform-specific performance notes
    println!("\n=== Platform Information ===");
    #[cfg(target_arch = "x86_64")]
    println!("Architecture: x86_64 (SIMD optimizations enabled)");

    #[cfg(not(target_arch = "x86_64"))]
    println!(
        "Architecture: {} (SIMD optimizations disabled)",
        std::env::consts::ARCH
    );

    println!("Rust version: {:?}", std::env::var("RUSTC_VERSION"));

    // Performance recommendations
    println!("\n=== Performance Recommendations ===");

    let total_simd_faster = [
        simd_time < regular_time,
        simd_mul_time < regular_mul_time,
        simd_relu_time < regular_relu_time,
    ]
    .iter()
    .filter(|&&x| x)
    .count();

    let total_parallel_faster = [
        parallel_time < regular_time,
        parallel_mul_time < regular_mul_time,
        parallel_matmul_time < regular_matmul_time,
    ]
    .iter()
    .filter(|&&x| x)
    .count();

    if total_simd_faster >= 2 {
        println!("✅ SIMD operations show significant speedup - use for element-wise ops");
    } else {
        println!("⚠️  SIMD operations may not be beneficial on this platform");
    }

    if total_parallel_faster >= 2 {
        println!("✅ Parallel operations show significant speedup - use for large computations");
    } else {
        println!("⚠️  Parallel operations may not be beneficial - overhead might dominate");
    }

    println!("\nBenchmark completed! 🦀");
}
