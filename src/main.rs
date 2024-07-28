use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, ScalarQuantizationBuilder, SearchParamsBuilder,
    SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant, QdrantError};
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use tokio;

#[tokio::main]
async fn main() -> Result<(), QdrantError> {
    // Initialize the FastEmbed model
    let model = TextEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::AllMiniLML6V2,
        show_download_progress: true,
        ..Default::default()
    })
    .expect("Failed to initialize FastEmbed model");

    // Example of top level client
    let client = Qdrant::from_url("http://localhost:6334").build()?;

    let collections_list = client.list_collections().await?;
    dbg!(collections_list);

    let collection_name = "real_estate";
    client.delete_collection(collection_name).await?;

    client
        .create_collection(
            CreateCollectionBuilder::new(collection_name)
                .vectors_config(VectorParamsBuilder::new(384, Distance::Cosine)) // Adjust vector size to 384
                .quantization_config(ScalarQuantizationBuilder::default()),
        )
        .await?;

    let collection_info = client.collection_info(collection_name).await?;
    dbg!(collection_info);

    // Reading the JSONL file
    let file = File::open("data.jsonl").expect("Unable to open file - data.jsonl");
    let reader = BufReader::new(file);

    let mut documents = Vec::new();
    let mut payloads = Vec::new();

    for (index, line) in reader.lines().enumerate() {
        let line = line.expect("Unable to read line");
        let json: Value = serde_json::from_str(&line).expect("Unable to parse JSON");

        // Assuming we want to embed the 'description' field
        if let Some(description) = json.get("description").and_then(|d| d.as_str()) {
            documents.push(description.to_string());
        } else {
            eprintln!("No description found for entry: {}", index);
            continue;
        }

        // Save the entire JSON object as payload
        payloads.push(json);
    }

    // Generate embeddings
    let embeddings = model
        .embed(documents, None)
        .expect("Failed to generate embeddings");
    println!("Embeddings length: {}", embeddings.len());
    println!("Embedding dimension: {}", embeddings[0].len());

    // Prepare points with embeddings and payloads
    let points: Vec<PointStruct> = embeddings
        .into_iter()
        .enumerate()
        .map(|(i, embedding)| {
            let payload: Payload = payloads[i]
                .clone()
                .try_into()
                .expect("Failed to convert payload");
            PointStruct::new(i as u64, embedding, payload)
        })
        .collect();

    // Upsert points into Qdrant
    client
        .upsert_points(UpsertPointsBuilder::new(collection_name, points))
        .await?;

    // Example search using one of the entries
    let search_document = vec!["detached house period features".to_string()];
    let embedding_for_search = model
        .embed(search_document, None)
        .expect("Failed to generate search embedding")[0]
        .clone();

    let search_result = client
        .search_points(
            SearchPointsBuilder::new(collection_name, embedding_for_search, 10)
                .with_payload(true)
                .params(SearchParamsBuilder::default().exact(true)),
        )
        .await?;
    dbg!(&search_result);


    if let Some(found_point) = search_result.result.into_iter().next() {
        let payload = found_point.payload;
        if let Some(description) = payload.get("description") {
            println!("Found description: {}", description.clone().into_json());
        } else {
            println!("Key 'description' not found in payload: {:?}", payload);
        }

        if let Some(link) = payload.get("link") {
            println!("Found link: {}", link.clone().into_json());
        } else {
            println!("Key 'link' not found in payload: {:?}", payload);
        }
    }

    Ok(())
}


