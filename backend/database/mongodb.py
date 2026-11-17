
import os

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "nexus_ai")

client = None
database = None


def connect_to_mongodb():
    """Connect to MongoDB Atlas and verify the connection."""
    global client, database

    if not MONGODB_URI:
        raise ValueError(
            "MONGODB_URI is missing. Check your .env file."
        )

    try:
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000
        )

        # Verify that MongoDB is reachable.
        client.admin.command("ping")

        database = client[MONGODB_DATABASE]

        print("MongoDB connection successful!")
        print(f"Database: {MONGODB_DATABASE}")

        return database

    except PyMongoError as error:
        client = None
        database = None
        raise ConnectionError(
            f"MongoDB connection failed: {error}"
        ) from error


def get_database():
    """Return the active database instance."""
    if database is None:
        raise RuntimeError(
            "Database is not connected. Connect first."
        )

    return database


def close_mongodb_connection():
    """Close the MongoDB client."""
    global client, database

    if client is not None:
        client.close()

    client = None
    database = None

    print("MongoDB connection closed.")


def get_collection(collection_name):
    """Return a collection from the active database."""
    return get_database()[collection_name]
