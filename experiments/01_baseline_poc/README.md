# Experimento 01 — Baseline PoC (Pipeline Completo)

Pipeline ponta-a-ponta de Video RAG usando o servidor `avancos` (192.168.1.9:11434).

## Estrutura

```
01_baseline_poc/
├── config.py           # Configurações centralizadas (servidor, modelos, paths)
├── extract_visual.py   # Descreve frames via llava/qwen3-vl no avancos
├── index.py            # Indexa chunks (transcript + visual) no ChromaDB
├── query.py            # Consulta RAG: retrieval + geração via avancos
├── requirements.txt    # Dependências Python
└── README.md           # Este arquivo
```

## Pré-requisitos

```bash
# Instalar dependências
pip install -r requirements.txt

# Verificar avancos
curl http://192.168.1.9:11434/api/tags | python -m json.tool | grep '"name"'
```

## Pipeline Completo

### Passo 1 — Extrair frames do vídeo

```bash
# Extrair 1 frame por segundo
ffmpeg -i SEU_VIDEO.mp4 -vf fps=1 frames/frame_%04d.jpg

# Para vídeo de 30min: ~1800 frames (~400MB em JPG)
```

### Passo 2 — Descrever frames via avancos

```bash
# Usar llava:latest (padrão, testado ✅)
python extract_visual.py --input frames/ --output descriptions.jsonl

# Usar moondream (mais rápido para grandes volumes)
python extract_visual.py --input frames/ --model moondream --output descriptions.jsonl

# Usar qwen3-vl (mais detalhado para slides/OCR)
python extract_visual.py --input frames/ --model qwen3-vl --output descriptions.jsonl
```

### Passo 3 — Indexar no ChromaDB

```bash
# Com transcrição Whisper (.srt ou .txt)
python index.py \
  --video-id "aula_piloto" \
  --transcript transcript.srt \
  --visual descriptions.jsonl

# Sem descrições visuais (apenas transcrição)
python index.py --video-id "aula_piloto" --transcript transcript.txt
```

### Passo 4 — Consultar

```bash
# Pergunta única
python query.py --question "Em que minuto o professor explicou redes neurais?"

# Modo interativo
python query.py --interactive

# Com raciocínio mais profundo
python query.py --question "..." --model deepseek-r1:7b

# Filtrar por vídeo específico
python query.py --interactive --video-id "aula_piloto"

# Salvar resultado
python query.py --question "..." --output resultado.json
```

## Configuração via Variáveis de Ambiente

```bash
# Sobrescrever configurações sem editar código
export VIDEORAG_VISION_MODEL=qwen3-vl           # usar qwen3-vl ao invés de llava
export VIDEORAG_RAG_MODEL=deepseek-r1:14b       # usar deepseek para respostas
export VIDEORAG_RETRIEVAL_TOP_K=8               # mais chunks recuperados
export VIDEORAG_CHROMA_PERSIST_DIR=/mnt/large-memory/videorag_index
```

## Modelos Disponíveis no avancos

### Vision (para describe_frame)
| Modelo | Velocidade | Qualidade | Melhor para |
|--------|-----------|-----------|------------|
| `moondream` | ⚡⚡⚡ | ⭐⭐ | Pré-filtro, escala |
| `llava:latest` | ⚡⚡ | ⭐⭐⭐ | Uso geral ✅ |
| `minicpm-v` | ⚡⚡ | ⭐⭐⭐ | Alternativa ao llava |
| `glm-ocr` | ⚡⚡ | ⭐⭐⭐⭐ | Slides, texto |
| `qwen3-vl` | ⚡ | ⭐⭐⭐⭐⭐ | Análise detalhada |

### Texto (para geração RAG)
| Modelo | Velocidade | Qualidade | Melhor para |
|--------|-----------|-----------|------------|
| `qwen2.5:7b-instruct` | ⚡⚡⚡ | ⭐⭐⭐ | Consultas rápidas |
| `gemma3:12b` | ⚡⚡ | ⭐⭐⭐ | Alternativa |
| `deepseek-r1:7b` | ⚡⚡ | ⭐⭐⭐⭐ | Raciocínio |
| `deepseek-r1:70b` | ⚡ | ⭐⭐⭐⭐⭐ | Máxima qualidade |

## Estimativa de Tempo (30min de vídeo)

| Etapa | Duração | Observações |
|-------|---------|-------------|
| Extração frames (ffmpeg) | ~30s | 1800 frames @ 1fps |
| Transcrição (Whisper CPU) | ~15–20min | Intel Xeon |
| Descrição visual (llava) | ~60–90min | ~2–3s/frame |
| Descrição visual (moondream) | ~15–30min | ~0.5–1s/frame |
| Indexação ChromaDB | ~2–5min | |
| Query RAG | ~5–15s | por pergunta |

> **Dica:** Use `--model moondream` para primeiro teste de pipeline, depois `llava:latest` para qualidade.

## Troubleshooting

```bash
# avancos não responde?
ping 192.168.1.9
curl http://192.168.1.9:11434/api/version

# ChromaDB com problemas?
python -c "import chromadb; c = chromadb.PersistentClient('/mnt/large-memory/videorag_index'); print(c.list_collections())"

# Ver modelos disponíveis no avancos
curl http://192.168.1.9:11434/api/tags | python -m json.tool
```
