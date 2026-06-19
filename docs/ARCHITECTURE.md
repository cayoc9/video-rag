# Arquitetura do Pipeline Video RAG — Laboratório

> Proposta de arquitetura para o modelo híbrido (cloud indexação + local consulta)

---

## Visão Geral

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FASE 1: INDEXAÇÃO (Cloud)                    │
│                         RunPod RTX 3090/4090                        │
│                                                                     │
│  Vídeo → [Extrator] → ASR/OCR/Frames → [Embeddings] → [Índice]    │
│             ↓                                ↓              ↓       │
│          Whisper                        CLIP/BLIP        FAISS     │
│          EasyOCR                      Sentence-T         Qdrant    │
│         GroundingDINO                                    Chroma    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Download do Índice (~GB)
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   FASE 2: PERSISTÊNCIA (Local)                      │
│                      /mnt/large-memory                              │
│                                                                     │
│           Índice Vetorial + Grafo de Conhecimento                   │
│           (metadados de timestamps + embeddings)                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Query
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    FASE 3: CONSULTA (Local + API)                   │
│                       CPU Intel Xeon                                │
│                                                                     │
│  Pergunta → [Retriever] → Top-K Segmentos → [LLM] → Resposta       │
│                 ↓                                ↓                  │
│           Busca vetorial                   GPT-4o-mini              │
│           (FAISS CPU)                    (~centavos/query)          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Componentes Técnicos

### Extração (Fase 1)

```python
# Stack de extração recomendado

EXTRACTION_STACK = {
    "audio_transcription": "openai/whisper-large-v3",  # ASR
    "frame_sampling": "ffmpeg",                         # 1 frame/segundo (adaptável)
    "ocr": "EasyOCR",                                   # Texto em frames
    "object_detection": "GroundingDINO",                # Descrição visual
    "scene_detection": "PySceneDetect"                  # Segmentação por cena
}

# Configuração de amostragem adaptativa
FRAME_SAMPLING = {
    "base_fps": 1,           # 1 frame/segundo padrão
    "high_motion_fps": 4,    # 4 frames/segundo quando há movimento
    "motion_threshold": 0.3  # Threshold de mudança entre frames
}
```

### Indexação (Fase 1)

```python
# Embeddings multimodais
EMBEDDING_MODELS = {
    "visual": "openai/clip-vit-large-patch14",
    "text": "sentence-transformers/all-MiniLM-L6-v2",
    "multimodal": "BLIP-2"  # Para embeddings conjuntos
}

# Vector store
VECTOR_STORE = "qdrant"  # Alternativa: FAISS (mais simples, CPU-friendly)

# Estrutura de chunk (segmento de vídeo)
CHUNK_STRUCTURE = {
    "video_id": "str",
    "start_time": "float",   # segundos
    "end_time": "float",     # segundos  
    "transcript": "str",     # ASR texto
    "ocr_text": "str",       # Texto extraído visualmente
    "visual_description": "str",  # GroundingDINO output
    "embedding": "list[float]",   # 768 dims
    "scene_id": "int"
}
```

### Consulta (Fase 3)

```python
# Pipeline de consulta
QUERY_PIPELINE = """
1. Receber pergunta do usuário
2. Gerar embedding da pergunta (mesmos modelos da indexação)
3. Busca vetorial → Top-10 segmentos mais similares
4. Re-ranking contextual (cross-encoder se disponível)
5. Montar contexto: [timestamp, transcript, descrição visual]
6. Enviar para GPT-4o-mini com prompt estruturado
7. Retornar resposta + timestamps das fontes
"""

RETRIEVAL_CONFIG = {
    "top_k": 10,
    "reranking": False,      # Desabilitar se CPU lenta
    "context_window": 5,     # Segmentos vizinhos ao recuperado
}
```

---

## Tecnologias por Componente

| Componente | Opção Gratuita/Local | Opção Cloud (Paga) |
|-----------|---------------------|-------------------|
| **ASR** | Whisper (local) | Deepgram, AssemblyAI |
| **OCR** | EasyOCR, Tesseract | Google Vision API |
| **Embeddings** | CLIP, SentenceT | OpenAI ada-002 |
| **Vector DB** | FAISS, ChromaDB | Pinecone, Qdrant Cloud |
| **LLM (Query)** | Ollama/Llama3 local | GPT-4o-mini, Gemini Flash |
| **Orquestração** | LangChain, LlamaIndex | — |

---

## Estimativas de Custo e Tempo

### Indexação (RunPod, RTX 3090)

| Volume | Tempo Estimado | Custo (@$0.39/h) |
|--------|---------------|------------------|
| 10h de vídeo | ~2h de GPU | ~$0.78 |
| 100h de vídeo | ~15h de GPU | ~$5.85 |
| 1000h de vídeo | ~120h de GPU | ~$46.80 |

### Consulta (Local ou API)

| Modo | Latência/Query | Custo/1000 Queries |
|------|---------------|-------------------|
| CPU Local (Intel Xeon) | 2–8s | $0 |
| GPT-4o-mini | 1–3s | ~$0.15 |
| GPT-4o | 2–5s | ~$2.50 |

---

## Diagrama de Dados

```
Vídeo Original (MP4/MKV)
├── Audio Stream
│   └── Whisper ASR → Transcript chunks [00:00:00 → 00:02:30]: "texto..."
├── Video Stream
│   ├── Frame @ 1fps → CLIP Embedding [768-dim]
│   ├── EasyOCR → Texto detectado
│   └── GroundingDINO → "pessoa segurando capacete, máquina CNC ao fundo"
└── Metadata
    ├── Duration, FPS, Resolution
    └── Scene boundaries [0, 45, 132, 287, ...] segundos

         ↓ Indexação

Vector DB (Qdrant/FAISS)
└── Collections:
    ├── visual_embeddings [clip-vit-L/14, 768-dim]
    ├── text_embeddings [all-MiniLM-L6, 384-dim]
    └── multimodal_embeddings [BLIP-2, 768-dim]

Knowledge Graph (opcional, para STAR-RAG)
└── Nodes: entidades (pessoas, objetos, locais)
└── Edges: relações temporais e causais
```

---

*Arquitetura proposta — Junho/2026*
