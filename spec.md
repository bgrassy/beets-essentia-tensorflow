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
```yaml
essentia:
    auto: no                          # Run automatically on import
    dry-run: no                       # Test run without making changes
    write: yes                        # Write to audio file tags
    threads: 1                        # Number of parallel processing threads
    force: no                         # Force reanalysis of previously analyzed files
    quiet: no                         # Reduce output verbosity

    models:
        embeddings:
            musicnn: /path/to/musicnn_model
            vggish: /path/to/vggish_model
        classification:
            genre: /path/to/genre_model
            style: /path/to/style_model
            mood: /path/to/mood_model
            danceability: /path/to/dance_model
            voice_instrumental: /path/to/voice_model
        rhythm:
            tempocnn: /path/to/tempo_model
            beats: /path/to/beats_model
        harmony:
            key: /path/to/key_model
            chords: /path/to/chords_model

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
