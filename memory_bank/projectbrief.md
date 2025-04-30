# OCR Benchmarking POC

## Project Overview
This is a proof of concept (POC) for benchmarking various Optical Character Recognition (OCR) techniques on scanned document datasets.

## Project Goals
1. Compare traditional OCR methods (Tesseract, EasyOCR, PaddleOCR)
2. Evaluate Vision Language Models (VLMs) for OCR capabilities (Qwen-2.5-VL, OlmOCR, Mistral OCR, Gemini Flash)
3. Consider data privacy implications of each solution
4. Analyze performance metrics (accuracy, speed, resource requirements)
5. Assess deployment feasibility on local vs. cloud environments

## Key Constraints
- Data privacy is critical - prefer solutions where data doesn't leave premises
- Local environment is macOS without CUDA support
- Alternative GPU access available through Google Colab
- Need to balance accuracy with implementation cost and complexity

## Implementation Approach
- Phased exploration of OCR methods (one method per phase)
- Progressive benchmarking and comparison
- Focus on practical implementation and real-world usability 