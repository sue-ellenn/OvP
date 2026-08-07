from pipeline.update_database import update_database
from pipeline.update_embeddings import update_embeddings

if __name__ == "__main__":
    update_database()
    update_embeddings()
    print("Update klaar")