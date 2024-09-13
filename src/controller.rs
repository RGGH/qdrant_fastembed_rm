use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, ScalarQuantizationBuilder, SearchParamsBuilder,
    SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant, QdrantError};
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

/// Initializes the FastEmbed model for generating text embeddings.
///
/// This function initializes the `TextEmbedding` model using the `AllMiniLML6V2` model
/// with progress feedback.
///
/// # Example
///
/// ```
/// let model = initialize_model();
/// ```
pub fn initialize_model() -> TextEmbedding {
    let start_time = Instant::now();
    let model = TextEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::AllMiniLML6V2,
        show_download_progress: true,
        ..Default::default()
    })
    .expect("Failed to initialize FastEmbed model");
    println!("Model initialization time: {:?}", start_time.elapsed());

    model
}

/// Sets up a collection in Qdrant, including deleting the existing collection and
/// creating a new one with a vector size of 384 and cosine distance for embeddings.
///
/// # Arguments
///
/// * `client` - A reference to the `Qdrant` client.
///
/// # Returns
///
/// A result that indicates if the operation was successful or an error occurred.
///
/// # Example
///
/// ```
/// let client = Qdrant::from_url("http://localhost:6334").build()?;
/// setup_qdrant_collection(&client).await?;
/// ```
pub async fn setup_qdrant_collection(client: &Qdrant) -> Result<(), QdrantError> {
    let collection_name = "real_estate";
    client.delete_collection(collection_name).await?;

    client
        .create_collection(
            CreateCollectionBuilder::new(collection_name)
                .vectors_config(VectorParamsBuilder::new(384, Distance::Cosine))
                .quantization_config(ScalarQuantizationBuilder::default()),
        )
        .await?;

    Ok(())
}

/// Loads data from a JSONL file and extracts documents and payloads.
///
/// This function reads the specified JSONL file and collects all documents with
/// descriptions into a vector of strings. It also collects the corresponding JSON
/// payloads for each document.
///
/// # Arguments
///
/// * `filename` - The path to the JSONL file.
///
/// # Returns
///
/// A tuple of two vectors: one containing document descriptions (as strings)
/// and the other containing the corresponding payloads (as `serde_json::Value`).
///
/// # Example
///
/// ```
/// let (documents, payloads) = load_data("data.jsonl");
/// ```
pub fn load_data(filename: &str) -> (Vec<String>, Vec<Value>) {
    let file = File::open(filename).expect("Unable to open file - data.jsonl");
    let reader = BufReader::new(file);

    let mut documents = Vec::new();
    let mut payloads = Vec::new();

    for (index, line) in reader.lines().enumerate() {
        let line = line.expect("Unable to read line");
        let json: Value = serde_json::from_str(&line).expect("Unable to parse JSON");

        if let Some(description) = json.get("description").and_then(|d| d.as_str()) {
            documents.push(description.to_string());
            payloads.push(json); // Push payload only if description is present
        } else {
            eprintln!("No description found for entry: {}", index);
        }
    }

    (documents, payloads)
}

/// Generates embeddings for a list of documents using the FastEmbed model.
///
/// This function takes in the FastEmbed model and a list of document strings,
/// then returns a vector of embeddings, where each embedding corresponds to a document.
///
/// # Arguments
///
/// * `model` - A reference to the initialized `TextEmbedding` model.
/// * `documents` - A vector of document strings to generate embeddings for.
///
/// # Returns
///
/// A vector of embeddings where each embedding is a vector of 32-bit floats.
///
/// # Example
///
/// ```
/// let embeddings = generate_embeddings(&model, documents);
/// ```
pub fn generate_embeddings(
    model: &TextEmbedding,
    documents: Vec<String>,
) -> Vec<Vec<f32>> {
    let start_time = Instant::now();
    let embeddings = model
        .embed(documents, None)
        .expect("Failed to generate embeddings");
    println!("Embeddings length: {}", embeddings.len());
    println!("Embedding dimension: {}", embeddings[0].len());
    println!("Embedding generation time: {:?}", start_time.elapsed());

    embeddings
}

/// Upserts points (documents with embeddings and payloads) into a Qdrant collection.
///
/// This function takes a Qdrant client, the collection name, the generated embeddings, and the
/// corresponding payloads, and upserts the points into the specified collection.
///
/// # Arguments
///
/// * `client` - A reference to the `Qdrant` client.
/// * `collection_name` - The name of the Qdrant collection.
/// * `embeddings` - A vector of embeddings for the documents.
/// * `payloads` - A vector of payloads corresponding to the documents.
///
/// # Returns
///
/// A result indicating the success or failure of the operation.
///
/// # Example
///
/// ```
/// upsert_points(&client, "real_estate", embeddings, payloads).await?;
/// ```
pub async fn upsert_points(
    client: &Qdrant,
    collection_name: &str,
    embeddings: Vec<Vec<f32>>,
    payloads: Vec<Value>,
) -> Result<(), QdrantError> {
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

    let start_time = Instant::now();
    client
        .upsert_points(UpsertPointsBuilder::new(collection_name, points))
        .await?;
    println!("Upsert points time: {:?}", start_time.elapsed());

    Ok(())
}

/// Performs a search on the Qdrant collection for documents similar to the given query.
///
/// This function generates an embedding for the query using the FastEmbed model, and searches the Qdrant
/// collection for the top 10 closest matches based on cosine distance. The results include the associated payloads.
///
/// # Arguments
///
/// * `client` - A reference to the `Qdrant` client.
/// * `model` - A reference to the `TextEmbedding` model.
/// * `collection_name` - The name of the Qdrant collection.
/// * `query` - The search query string to find similar documents.
///
/// # Returns
///
/// A result indicating the success or failure of the search operation.
///
/// # Example
///
/// ```
/// search_qdrant(&client, &model, "real_estate", "detached house in cul de sac").await?;
/// ```
pub async fn search_qdrant(
    client: &Qdrant,
    model: &TextEmbedding,
    collection_name: &str,
    query: &str,
) -> Result<(), QdrantError> {
    let search_document = vec![query.to_string()];
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

#[cfg(test)]
mod tests {
    use super::*;
    use qdrant_client::Qdrant;
    use serde_json::json;
    use std::fs::File;
    use std::io::Write;
    use tokio;

    #[tokio::test]
    async fn test_initialize_model() {
        let model = initialize_model();
        // Perform an actual check to see if the model is working or initialized
        assert!(model.embed(vec!["test".to_string()], None).is_ok());
    }

    #[tokio::test]
    async fn test_setup_qdrant_collection() {
        let client = Qdrant::from_url("http://localhost:6334").build().expect("Failed to build Qdrant client");
        let result = setup_qdrant_collection(&client).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_data() {
        let filename = "test_data.jsonl";
        let mut file = File::create(filename).expect("Unable to create file");
        writeln!(file, r#"{{"description": "Test document", "key": "value"}}"#).expect("Unable to write to file");

        let (documents, payloads) = load_data(filename);
        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0], "Test document");
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0], json!({"description": "Test document", "key": "value"}));
    }

    #[tokio::test]
    async fn test_generate_embeddings() {
        let model = initialize_model();
        let documents = vec!["Document 1".to_string(), "Document 2".to_string()];
        let embeddings = generate_embeddings(&model, documents);
        assert_eq!(embeddings.len(), 2);
        assert!(embeddings[0].len() > 0);
    }

    #[tokio::test]
    async fn test_upsert_points() {
        let client = Qdrant::from_url("http://localhost:6334").build().expect("Failed to build Qdrant client");
        let collection_name = "real_estate";
        let model = initialize_model();

        let documents = vec!["Document 1".to_string(), "Document 2".to_string()];
        let embeddings = generate_embeddings(&model, documents);
        let payloads = vec![
            json!({"description": "Document 1", "key": "value1"}),
            json!({"description": "Document 2", "key": "value2"})
        ];

        let result = upsert_points(&client, collection_name, embeddings, payloads).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_search_qdrant() {
        let client = Qdrant::from_url("http://localhost:6334").build().expect("Failed to build Qdrant client");
        let model = initialize_model();
        let collection_name = "real_estate";
        let query = "test query";

        let result = search_qdrant(&client, &model, collection_name, query).await;
        assert!(result.is_ok());
    }
}

