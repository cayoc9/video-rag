# Experimentos — Video RAG

Este diretório armazenará os notebooks, scripts e resultados dos experimentos do laboratório.

## Estrutura Planejada

```
experiments/
├── 01_baseline_poc/
│   ├── README.md           # Instruções do experimento
│   ├── extract.py          # Extração ASR/OCR/Frames
│   ├── index.py            # Indexação no vector store
│   ├── query.py            # Consulta RAG
│   └── results/
│       └── metrics.json
│
├── 02_citation_faithfulness/
│   ├── README.md
│   ├── dataset/            # 50 QA pairs anotados manualmente
│   ├── faithfulness_checker.py
│   └── results/
│
├── 03_cross_video/
│   ├── README.md
│   ├── multi_index.py      # Índice unificado multi-vídeo
│   └── results/
│
└── 04_adaptive_sampling/
    ├── README.md
    ├── motion_detector.py
    └── results/
```

## Como Reproduzir os Experimentos

> Instruções serão adicionadas à medida que os experimentos forem implementados.

### Pré-requisitos Gerais

```bash
# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependências base
pip install openai-whisper easyocr chromadb faiss-cpu langchain ffmpeg-python

# Para embeddings visuais
pip install transformers sentence-transformers pillow

# Para avaliação
pip install ragas deepeval
```

### Variáveis de Ambiente Necessárias

```bash
export OPENAI_API_KEY="sk-..."          # Para GPT-4o-mini (consulta)
export RUNPOD_API_KEY="..."             # Para indexação em cloud (opcional)
```

---

*Experimentos a serem adicionados — Fase 1 iniciando em Julho/2026*
