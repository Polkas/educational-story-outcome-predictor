<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Educational Story Analysis ML Project

This project analyzes educational stories to predict outcomes and detect self-fulfilling prophecy patterns using the MU-NLPC/Edustories-en dataset.

## Project Context

- Optimized for Apple M3 chip with MPS acceleration
- Uses efficient models like DistilBERT and Flan-T5-small
- Implements PEFT/LoRA for efficient fine-tuning
- Focuses on anamnesis-outcome correlations and educational bias detection

## Code Generation Guidelines

- Always use torch.device("mps") for Apple Silicon optimization
- Prefer efficient transformers models that fit in 36GB RAM
- Use gradient checkpointing and mixed precision training
- Implement proper story segmentation (30% anamnesis, 70% outcome)
- Include bias detection and fairness metrics
- Follow Hugging Face best practices for model deployment
- Use PEFT/LoRA adapters for memory-efficient fine-tuning
- Include comprehensive logging and evaluation metrics
