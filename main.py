from embeddings import (
    generate_embedding,
    compare_sentences
)


def main():
    print("AI TEXT EMBEDDING EXPLORER")

    text1 = input("Enter first sentence: ")
    text2 = input("Enter second sentence: ")

    try:
        embedding = generate_embedding(text1)

        print("\nEmbedding dimensions:", len(embedding))
        print("First 10 values:", embedding[:10])

        similarity = compare_sentences(
            text1,
            text2
        )

        print("\nSimilarity score:", round(similarity, 4))

    except ValueError as error:
        print("Error:", error)


if __name__ == "__main__":
    main()