// main.rs

mod controller;

use controller::{initialize_model, load_data, setup_qdrant_collection, generate_embeddings, upsert_points, search_qdrant};
use qdrant_client::{Qdrant, QdrantError};
use tokio;

#[tokio::main]
async fn main() -> Result<(), QdrantError> {
    // Initialize model and client
    let model = initialize_model();
    let client = Qdrant::from_url("http://localhost:6334").build()?;

    // Setup Qdrant collection
    setup_qdrant_collection(&client).await?;

    // Load data from file
    let (documents, payloads) = load_data("data.jsonl");

    // Generate embeddings
    let embeddings = generate_embeddings(&model, documents);

    // Upsert points into Qdrant
    upsert_points(&client, "real_estate", embeddings, payloads).await?;

    // Search in Qdrant
    search_qdrant(&client, &model, "real_estate", "detached house in cul de sac").await?;

    Ok(())
}

