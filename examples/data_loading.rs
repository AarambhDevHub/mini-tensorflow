use mini_tensorflow::{
    DataLoader, Dataset, data::make_classification_dataset, data::make_regression_dataset,
};
use std::fs;

fn main() {
    println!("Data Loading Example");

    // Create synthetic classification dataset
    println!("\n=== Classification Dataset ===");
    let mut classification_data = make_classification_dataset(1000, 20, 3);
    classification_data.summary();

    // Normalize features
    classification_data.normalize_features();
    println!("Features normalized");

    // Split into train/test
    let (train_data, test_data) = classification_data.split(0.8);
    println!(
        "Train size: {}, Test size: {}",
        train_data.len(),
        test_data.len()
    );

    // Create data loader
    let mut train_loader = DataLoader::new(train_data, 32)
        .with_shuffle(true)
        .with_drop_last(false);

    println!("\n=== Batch Processing ===");
    for (batch_idx, (batch_data, batch_labels)) in train_loader.iter().enumerate() {
        println!("Batch {}: {} samples", batch_idx + 1, batch_data.len());

        if batch_idx >= 2 {
            // Show only first 3 batches
            break;
        }

        // Show batch shapes
        if !batch_data.is_empty() {
            println!("  Data shape: {:?}", batch_data[0].shape);
            println!("  Label shape: {:?}", batch_labels[0].shape);
        }
    }

    // Create synthetic regression dataset
    println!("\n=== Regression Dataset ===");
    let regression_data = make_regression_dataset(500, 10, 0.1);
    regression_data.summary();

    // Demonstrate CSV loading capability
    println!("\n=== CSV Dataset Creation ===");
    create_sample_csv();

    // Load from CSV (this would work if CSV exists)
    if let Ok(csv_dataset) = Dataset::from_csv(
        "sample_data.csv",
        vec![0, 1, 2], // feature columns
        3,             // target column
        true,          // has header
    ) {
        println!("Loaded dataset from CSV:");
        csv_dataset.summary();

        // Create data loader for CSV data
        let mut csv_loader = DataLoader::new(csv_dataset, 16).with_shuffle(true);

        let mut total_batches = 0;
        for (_batch_data, _batch_labels) in csv_loader.iter() {
            total_batches += 1;
        }
        println!("Total batches in CSV dataset: {}", total_batches);
    } else {
        println!("Could not load CSV (this is expected in demo)");
    }
}

fn create_sample_csv() {
    let csv_content = r#"feature1,feature2,feature3,target
1.0,2.0,3.0,6.0
2.0,3.0,4.0,9.0
3.0,4.0,5.0,12.0
4.0,5.0,6.0,15.0
5.0,6.0,7.0,18.0"#;

    if let Err(_) = fs::write("sample_data.csv", csv_content) {
        println!("Could not create sample CSV file");
    } else {
        println!("Created sample_data.csv");
    }
}
