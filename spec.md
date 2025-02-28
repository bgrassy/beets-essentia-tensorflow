# Beets-Essentia Plugin Technical Specification

## 1. Overview

The Beets-Essentia plugin integrates Essentia's TensorFlow-based music analysis capabilities into the Beets music library manager. It extracts musical features using pre-trained models and stores results in both the Beets database and audio file tags.

### 1.1 Core Features

- Genre and style recognition
- Mood/emotion detection
- Danceability estimation
- Voice/instrumental classification
- BPM and beat tracking
- Key detection and chord progression analysis

### 1.2 Design Principles

- Configurable model paths and storage options
- Efficient parallel processing with thread management
- Resumable operations
- Robust error handling
- Minimal memory footprint

## 2. Dependencies

### 2.1 Required Packages

```python
from essentia.standard import *
import essentia.streaming as ess
import tensorflow as tf
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from beetsplug.base import BeetsPlugin
```

### 2.2 External Dependencies

- Essentia library with TensorFlow support
- Pre-trained TensorFlow models (user-provided)
- Beets 1.6.0 or higher

## 3. Configuration

### 3.1 Plugin Configuration

The plugin supports two approaches for model configuration:

1. Using metadata JSON files for comprehensive model information
2. Direct parameter configuration for more control

Both approaches can be mixed, with direct parameters overriding metadata values.

```yaml
essentia:
  # Global configuration
  auto: no # Run automatically on import
  dry-run: no # Test run without making changes
  write: yes # Write to audio file tags
  threads: 1 # Number of parallel processing threads
  force: no # Force reanalysis of previously analyzed files
  quiet: no # Reduce output verbosity

  # Optional: base directory for models (relative paths in config or metadata will be resolved against this)
  models_directory: /path/to/models/base/directory

  models:
    embeddings:
      discogs:
        # Option 1: Using metadata JSON
        metadata_path: /path/to/discogs_model_metadata.json
        # Option 2: Direct configuration (overrides metadata)
        model_path: /path/to/override/discogs_model.pb  # Optional override
        model_output: "PartitionedCall:1"  # Optional override
      musicnn:
        metadata_path: /path/to/musicnn_model_metadata.json
        # Can override specific parameters
        model_output: "custom_output"
    classification:
      genre:
        # Option 1: Using metadata JSON
        metadata_path: /path/to/genre_model_metadata.json
        # Option 2: Direct configuration (overrides metadata)
        model_path: /path/to/override/genre_model.pb  # Optional override
        model_input: "serving_default_model_Placeholder"  # Optional override
        model_output: "PartitionedCall:0"  # Optional override
        embedding_model: "discogs"  # Optional override - reference to embedding model
      style:
        # Option 2: Direct configuration only (no metadata)
        model_path: /path/to/style_model.pb
        model_input: "serving_default_model_Placeholder"
        model_output: "model/Softmax"
        embedding_model: "discogs"
      mood:
        metadata_path: /path/to/mood_model_metadata.json
      danceability:
        metadata_path: /path/to/dance_model_metadata.json
      voice_instrumental:
        metadata_path: /path/to/voice_model_metadata.json
    rhythm:
      tempocnn:
        metadata_path: /path/to/tempo_model_metadata.json
      beats:
        metadata_path: /path/to/beats_model_metadata.json
    harmony:
      key:
        metadata_path: /path/to/key_model_metadata.json
      chords:
        metadata_path: /path/to/chords_model_metadata.json

  storage:
    tags:
      write: yes
      update_existing: no
      formats:
        id3: yes
        vorbis: yes
        mp4: yes
        asf: yes
      fields:
        bpm: yes
        key: yes
        genre: no
        mood: no
        dance: no
        voice: no

    database:
      store_probabilities: yes
      beat_resolution: 0.001
      chord_format: "simple"
```

### 3.2 Model Metadata JSON Format

Each model can have a corresponding metadata JSON file that describes the model's properties, inputs/outputs, and class labels.

#### Example Model Metadata:

```json
{
    "name": "Genre Discogs400",
    "type": "Music genre classification",
    "link": "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
    "model_path": "/path/to/model/genre_discogs400-discogs-effnet-1.pb",
    "version": "1",
    "description": "Prediction of 400 music styles from the Discogs taxonomy",
    "author": "Author Name",
    "email": "author@example.com",
    "release_date": "2023-05-04",
    "framework": "tensorflow",
    "framework_version": "2.8.0",
    "classes": [
        "Class1",
        "Class2",
        "..."
    ],
    "schema": {
        "inputs": [
            {
                "name": "serving_default_model_Placeholder",
                "type": "float",
                "shape": ["batch_size", 1280]
            }
        ],
        "outputs": [
            {
                "name": "PartitionedCall:0",
                "type": "float",
                "shape": ["batch_size", 400],
                "op": "Sigmoid",
                "output_purpose": "predictions"
            }
        ]
    },
    "inference": {
        "sample_rate": 16000,
        "algorithm": "TensorflowPredict2D",
        "embedding_model": {
            "algorithm": "TensorflowPredictEffnetDiscogs",
            "model_name": "discogs-effnet-bs64-1",
            "link": "https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.pb"
        }
    }
}
```

### 3.3 Configuration Resolution Process

When loading a model, the plugin will:

1. Check if a metadata file is specified
2. If yes, load model parameters from the metadata
3. Check for override parameters in the configuration
4. Apply any override parameters, replacing metadata values
5. Validate the final configuration
6. Load the model

Relative paths in both configuration and metadata can use the optional global `models_directory` as a base path.

## 4. Implementation Structure

### 4.1 Main Plugin Class

```python
class EssentiaPlugin(BeetsPlugin):
    def __init__(self):
        super().__init__()
        self.setup_plugin()

    def setup_plugin(self):
        """Initialize plugin configuration and models"""
        self.load_config()
        self.validate_model_paths()
        self.initialize_processors()

    def commands(self):
        """Register plugin commands"""
        cmd = ui.Subcommand('essentia', help='Analyze music using Essentia')
        cmd.parser.add_option('--restart', action='store_true',
                            help='Force fresh start')
        cmd.parser.add_option('--status', action='store_true',
                            help='Show processing status')
        return [cmd]

    def handle_album(self, lib, opts, args):
        """Process albums from command line"""
        pass

    def handle_item(self, lib, opts, args):
        """Process individual tracks from command line"""
        pass

    def import_items(self, session, task):
        """Import hook for automatic processing"""
        pass
```

### 4.2 Processing Pipeline

See the detailed pipeline specification in the previously created processing-pipeline artifact, including:

- Audio loading
- Parallel feature extraction groups
- Thread management
- Memory management
- Error handling
- Progress reporting

### 4.3 Resume Functionality

See the detailed resume specification in the previously created resume-spec artifact, including:

- State tracking
- Resume logic
- Error handling
- User interface

## 5. Data Storage

### 5.1 Database Fields

```python
class EssentiaPlugin(BeetsPlugin):
    item_fields = {
        'essentia_genre': types.STRING,
        'essentia_genre_probs': types.FLOAT,
        'essentia_style': types.STRING,
        'essentia_style_probs': types.FLOAT,
        'essentia_mood': types.STRING,
        'essentia_mood_probs': types.FLOAT,
        'essentia_dance': types.FLOAT,
        'essentia_voice': types.STRING,
        'essentia_voice_prob': types.FLOAT,
        'essentia_bpm': types.FLOAT,
        'essentia_beats': types.LIST,
        'essentia_beat_count': types.INTEGER,
        'essentia_key': types.STRING,
        'essentia_key_prob': types.FLOAT,
        'essentia_chords': types.LIST,
        'essentia_scale': types.STRING,
    }
```

### 5.2 Tag Mapping

Detailed tag mapping for different audio formats as specified in the storage-mapping artifact.

## 6. Error Handling

### 6.1 Model Loading Errors

```python
def validate_model_paths(self):
    """Ensure all required models exist and are valid"""
    for category, models in self.config['models'].items():
        for name, path in models.items():
            if not os.path.exists(path):
                raise ModelNotFoundError(f"Model not found: {path}")
            try:
                self.load_model(path)
            except Exception as e:
                raise InvalidModelError(f"Invalid model {path}: {str(e)}")
```

### 6.2 Processing Errors

```python
def process_safe(self, func, *args, **kwargs):
    """Wrapper for safe processing with timeout"""
    try:
        with timeout(self.config['timeout']):
            return func(*args, **kwargs)
    except TimeoutError:
        self._log.warning(f"Timeout processing {func.__name__}")
        return None
    except Exception as e:
        self._log.error(f"Error in {func.__name__}: {str(e)}")
        return None
```

### 6.3 Storage Errors

- Validation before storage
- Rollback capabilities
- Partial result storage
- Detailed error logging

## 7. Testing Plan

### 7.1 Unit Tests

- Model loading and validation
- Audio file processing
- Feature extraction accuracy
- Tag writing/reading
- Database operations
- Resume functionality
- Error handling

### 7.2 Integration Tests

- Full pipeline processing
- Multiple file formats
- Various configuration combinations
- Plugin commands
- Import hooks
- Database queries

### 7.3 Performance Tests

- Memory usage monitoring
- Processing speed benchmarks
- Thread scaling tests
- Large library processing

### 7.4 Test Data Requirements

- Sample audio files in various formats
- Pre-trained models for testing
- Known-good feature extraction results
- Corrupted files for error handling tests

## 8. Documentation

### 8.1 Required Documentation

- Installation guide
- Configuration reference
- Model download instructions
- Command line interface
- Feature descriptions
- Troubleshooting guide
- API reference

### 8.2 Example Usage

```bash
# Install plugin
pip install beets-essentia

# Configure plugin
beet config -e

# Download models
# (Instructions for model download)

# Run analysis
beet essentia [options] [query]

# Show status
beet essentia --status

# Force restart
beet essentia --restart
```

## 9. Performance Considerations

### 9.1 Memory Management

- Model sharing between threads
- Audio buffer reuse
- Garbage collection hints
- Memory monitoring

### 9.2 Processing Optimization

- Batch processing capabilities
- Caching mechanisms
- Thread pool management
- Resource usage limits

## 10. Future Considerations

### 10.1 Potential Enhancements

- Additional Essentia features
- Custom model support
- Alternative embedding models
- Extended tag mapping
- Web API integration

### 10.2 Maintenance

- Version compatibility tracking
- Dependency updates
- Model updates
- Performance monitoring
- User feedback channels
