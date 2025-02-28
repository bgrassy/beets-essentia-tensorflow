from beetsplug.essentia_tensorflow.model_manager import ModelManager
model_manager = ModelManager(
    {
        "models": {
            "embeddings": {
                "discogs": {
                    "model_path": "/home/brock/Projects/beets-essentia-tensorflow/discogs-effnet-bs64-1.pb"
                },
            },
            "classification": {
                "genre": {
                    "model_path": "/home/brock/Projects/beets-essentia-tensorflow/genre_discogs400-discogs-effnet-1.pb",
                    "metadata_path": "/home/brock/Downloads/genre_discogs400-discogs-effnet-1.json",
                    "embedding_model": "discogs",
                }
                "mood": {
                    "model_path": "/home/brock/Projects/beets-essentia-tensorflow/genre_discogs400-discogs-effnet-1.pb",
                    "metadata_path": "/home/brock/Downloads/mood_happy-discogs-effnet-1.json",
                    "embedding_model": "discogs",
                }
            },
        }
    }
)


from beetsplug.essentia_tensorflow.audio_processor import AudioProcessor
processor = AudioProcessor(model_manager, {})
print(processor._get_available_features())
processed = processor.process_file("/mnt/media/audio/music/L/Lenderman, MJ/Manning Fireworks/05 She's Leaving You.flac")
