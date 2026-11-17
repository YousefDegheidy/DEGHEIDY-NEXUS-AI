# DEGHEIDY Multilingual AI Chatbot
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YousefDegheidy/multilingual-ai-chatbot/blob/main/DEGHEIDY_Chatbot_V10_Colab.ipynb)

**A multilingual, context-aware AI chatbot with Arabic-English conversation support, speech recognition, and streaming responses.**

Built with Python, Qwen, Hugging Face Transformers, Faster-Whisper, and Gradio.

## Project Overview

DEGHEIDY Multilingual AI Chatbot is an AI application designed to provide natural conversations in Arabic, English, and mixed-language text.

The project combines large language models, context-aware dialogue management, and automatic speech-to-text transcription in an interactive Gradio interface.

Its main goal is to explore how multilingual language models, conversation memory, and speech-processing components can work together in a practical conversational AI system.

## Key Features

### 1. Multilingual Conversations
- Supports Arabic and English text.
- Handles mixed Arabic-English conversations (code-switching).
- Includes typo-handling and question-understanding heuristics.
- Generates responses progressively using streaming.

### 2. Context-Aware Conversation Memory
- Maintains conversational context within the active session.
- Connects related follow-up questions to earlier messages.
- Tracks conversation topics and user corrections.
- Attempts to reduce irrelevant or repetitive responses.

### 3. Speech Recognition
- Uses Faster-Whisper for multilingual speech-to-text.
- Supports automatic language detection without manually switching between Arabic and English.
- Uses Voice Activity Detection (VAD) to identify speech segments.
- Includes handling for overlapping audio and repeated transcription.
- Allows users to review and edit recognized text before sending it.

### 4. Response Quality Checks
- Checks for some incomplete or inconsistent answers.
- Attempts to identify missed parts of multi-part questions.
- Includes selected consistency checks for follow-up conversations.
- Supports regression testing using previously problematic prompts.

### 5. Interactive Gradio Interface
- Chat interface with an **Ask ??** text input.
- Microphone input for speech recognition.
- Streaming chatbot responses.
- Conversation reset functionality.
- Google Colab execution support.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core development |
| Qwen3-4B-Instruct | Main conversational language model |
| Qwen2.5-3B-Instruct | Fallback language model |
| Hugging Face Transformers | Model loading and inference |
| Faster-Whisper | Multilingual speech recognition |
| WebRTC VAD | Speech activity detection |
| Gradio | Interactive user interface |
| PyTorch | Deep learning inference |
| Google Colab | GPU-enabled experimentation |

## Project Structure

- **app.py** — Main application and Gradio interface
- **src/dialogue.py** — Conversation memory, response generation, and quality checks
- **src/speech.py** — Speech recognition and audio processing
- **tests/** — Automated tests
- **DEGHEIDY_Chatbot_V10_Colab.ipynb** — Google Colab notebook
- **requirements.txt** — Core Python dependencies
- **regression_prompts.md** — Conversation evaluation examples
- **README_AR.md** — Arabic documentation

## Running the Project

### Google Colab

1. Open the included Google Colab notebook.
2. Select a T4 GPU runtime when available.
3. Run the notebook's code cell.
4. Open the temporary Gradio link generated in the output.
5. Type your question or use the microphone.
6. Review recognized speech in **Ask ??** before sending.

The full chatbot requires a CUDA-compatible GPU in the current configuration. A lightweight speech-only mode is available on CPU.

## Testing

The project includes automated tests for selected conversation handling, response checks, and speech-processing logic.

Run the tests using:

`python -m unittest discover -s tests -v`

These tests evaluate code behavior but do not establish a measured real-world accuracy score for the language or speech recognition models.

## Current Limitations

- Language-model responses may contain factual or reasoning errors.
- Speech recognition accuracy can vary with accent, background noise, and Arabic-English code-switching.
- Conversation memory is session-based and is not persistent across restarts.
- GPU availability affects inference performance.
- Gradio share links are temporary and are not permanent hosting.

## Future Improvements

- More extensive Arabic-English speech recognition benchmarks
- Better long-term conversation memory
- Additional response-quality evaluation
- Reduced latency and memory usage
- Deployment outside Google Colab

## Author

**Yousef Amr Fahmy Degheidy**

AI / Machine Learning Developer

[GitHub Profile](https://github.com/YousefDegheidy)

---

*This project is an evolving AI engineering portfolio project focused on multilingual conversational AI, context management, and speech processing.*
