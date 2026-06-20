# Import the official Ollama library
import ollama

def generate_local_embedding(text_chunk):
    """
    Takes a string of text and returns its vector embedding using a local Ollama model.
    """
    print(f"Generating embedding for: '{text_chunk}'...")
    

    response = ollama.embeddings(
        model='nomic-embed-text:v1.5',
        prompt=text_chunk
    )
    
    vector = response['embedding']
    return vector

if __name__ == "__main__":
    # Example transcript chunk from your YouTube video
    sample_transcript_chunk = "Machine learning is a branch of artificial intelligence."
    
    # Generate the vector
    my_vector = generate_local_embedding(sample_transcript_chunk)
    
    # Print the results to verify it worked
    print(my_vector[:2])
    # print(f"Total dimensions in this vector: {len(my_vector)}")
    # print(f"First 5 numbers of the vector array: {my_vector[:5]}")