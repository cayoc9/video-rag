# 🎬 Video RAG — Pesquisa & Laboratório

> **Retrieval-Augmented Generation aplicado a vídeos de longa duração**  
> Um repositório de pesquisa, experimentos e implementações para busca semântica e raciocínio sobre vídeos.

[![Status](https://img.shields.io/badge/status-pesquisa_ativa-brightgreen)](.)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](.)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

---

## 📋 Visão Geral

Este repositório documenta a pesquisa aprofundada sobre **Video RAG** — sistemas capazes de responder perguntas em linguagem natural sobre horas de vídeo, recuperando os trechos visuais e auditivos relevantes para gerar respostas precisas.

O projeto é voltado ao **contexto universitário/laboratorial**, mapeando o estado da arte (SOTA), lacunas abertas de pesquisa e uma estratégia de implementação viável com o hardware disponível.

---

## 📁 Estrutura do Repositório

```
video-rag/
├── README.md                        # Este arquivo
├── research/
│   ├── SOTA_VIDEORAG_RESEARCH.md    # Estado da arte: modelos e frameworks
│   ├── VIDEORAG_REPORT.md           # Viabilidade técnica e custos
│   ├── RESEARCH_GAPS_DEEP.md        # Lacunas aprofundadas (deep research)
│   └── BENCHMARKS.md                # Benchmarks e métricas de avaliação
├── docs/
│   ├── ARCHITECTURE.md              # Arquitetura do pipeline proposto
│   └── ROADMAP.md                   # Plano de implementação faseada
└── experiments/
    └── README.md                    # Guia para reprodução dos experimentos
```

---

## 🔬 Principais Documentos de Pesquisa

| Documento | Descrição |
|-----------|-----------|
| [Estado da Arte](research/SOTA_VIDEORAG_RESEARCH.md) | Mapeamento de soluções SOTA 2025–2026 |
| [Relatório de Viabilidade](research/VIDEORAG_REPORT.md) | Hardware, custos e estratégia híbrida |
| [Lacunas de Pesquisa](research/RESEARCH_GAPS_DEEP.md) | Deep research: problemas em aberto |
| [Benchmarks](research/BENCHMARKS.md) | Métricas e datasets de avaliação |
| [Arquitetura](docs/ARCHITECTURE.md) | Pipeline proposto para o laboratório |
| [Roadmap](docs/ROADMAP.md) | Fases de implementação |

---

## 🗺️ Estado da Arte Resumido

### Modelos Acadêmicos (2025–2026)

| Modelo | Conferência | Inovação |
|--------|------------|---------|
| **Video-RAG** | NeurIPS 2025 | Textos auxiliares visualmente alinhados (OCR+ASR) |
| **STAR-RAG** | ICLR 2026 | Temporal Graph Summarization |
| **E²RAG** | Jun 2025 | Dual-Graph (Entidade + Evento) |
| **TMRL** | Jan 2026 | Matryoshka Representation Learning |
| **LVAgent** | 2025 | Framework multi-agente colaborativo |
| **T\* (T-star)** | 2025 | Busca temporal como busca espacial |

### Soluções Comerciais

- **Twelve Labs (Pegasus 1.5)** → Indexação nativa, ideal para 10k+ horas
- **Gemini 1.5/2.0 Pro** → Janela de 2M tokens, raciocínio narrativo profundo

---

## 🔭 Lacunas Abertas de Pesquisa

> Oportunidades para artigos e patentes identificadas nesta pesquisa:

1. **Raciocínio Espacial 3D** — Integrar SLAM ao RAG para localização real de objetos
2. **Audio-Visual Grounding** — Vincular sons específicos a regiões da imagem
3. **Streaming VideoRAG** — Poda dinâmica de grafo em tempo real com gestão de memória
4. **Amostragem Adaptativa de FPS** — Aumentar análise apenas quando há movimento relevante
5. **Cross-Video Knowledge** — Síntese de conhecimento entre múltiplos vídeos concorrentes
6. **Citation Faithfulness** — Rastrear e validar quais segmentos sustentam cada afirmação gerada

---

## 💻 Estratégia de Implementação (Modelo Híbrido)

```
Fase 1: Indexação → Cloud (RunPod RTX 3090/4090 @ $0.39/h)
         ↓
Fase 2: Download do Índice → /mnt/large-memory (persistência local)
         ↓
Fase 3: Consulta RAG → CPU local (Intel Xeon) + GPT-4o-mini
```

**Custo estimado:** ~$2–10 para 100h de vídeo | ~$15–80 para 1000h de vídeo

---

## 📚 Referências Principais

1. Luo et al. (2025). *Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension*. NeurIPS 2025.
2. Jeong et al. (2025). *VideoRAG: Retrieval-Augmented Generation over Video Corpus*. arXiv:2501.05874.
3. Zhu et al. (2026). *STAR-RAG: Right Answer at the Right Time*. ICLR 2026.
4. LV-HAYSTACK Benchmark (2025). 480 horas de vídeo, 15k+ instâncias anotadas.
5. VidHalluc (CVPR 2025). Benchmark de alucinação em vídeo.

---

## 🤝 Contribuindo

Este repositório está em fase de pesquisa ativa. Contribuições são bem-vindas via Pull Requests.

---

*Pesquisa iniciada em Junho/2026 — Laboratório Universitário de IA*
