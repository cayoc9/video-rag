# Estado da Arte: VideoRAG e RAG Temporal (Relatório 2025-2026)

Este documento sintetiza a pesquisa profunda sobre as soluções mais avançadas para o problema de busca e raciocínio em vídeos de longa duração, comparando abordagens acadêmicas e comerciais.

---

## 1. Mapeamento de Soluções SOTA (State-of-the-Art)

### 1.1. Modelos Acadêmicos e Open-Source
| Modelo / Framework | Origem | Inovação Principal | Foco Técnico |
| :--- | :--- | :--- | :--- |
| **Video-RAG** | NeurIPS 2025 | Visually-aligned auxiliary texts | Extração de OCR/ASR antes da inferência. |
| **Qwen3-VL** | Alibaba 2026 | Unified Multimodal Space | Espaço de vetores único para som, imagem e texto. |
| **STAR-RAG** | ICLR 2026 | Temporal Graph Summarization | Uso de Grafos de Regras para evitar alucinação temporal. |
| **TMRL** | Jan 2026 | Matryoshka Rep. Learning | Vetores "aninhados" para busca multiescala rápida. |
| **E²RAG** | Jun 2025 | Dual-Graph (Entity-Event) | Separação de personagens e eventos em grafos distintos. |

### 1.2. Soluções Comerciais (APIs Gerenciadas)
*   **Twelve Labs (Pegasus 1.5):** Focada em **indexação nativa**. Ideal para buscar em arquivos de milhares de horas em milissegundos.
*   **Gemini 1.5 Pro (Google):** Focada em **janela de contexto (2M+ tokens)**. Ideal para raciocínio narrativo profundo em um único vídeo sem precisar de banco de dados.

---

## 2. Análise Comparativa por Casos de Uso

| Caso de Uso | Solução Recomendada | Justificativa |
| :--- | :--- | :--- |
| **Busca em Arquivo Morto (10k+ horas)** | **Twelve Labs** | A indexação nativa permite busca instantânea por palavras-chave visuais/áudio. |
| **Análise de Aula/Reunião (1-2h)** | **Gemini 1.5 Pro** | Raciocínio superior sobre diálogos e slides sem setup de infraestrutura. |
| **Monitoramento de Segurança (Live)** | **Qwen3-VL (Local)** | Baixa latência e privacidade total rodando em GPU própria. |
| **Documentários e Narrativas Longas** | **STAR-RAG / E²RAG** | Mantém a consistência de quem fez o quê ao longo de horas de filme. |
| **Dispositivos de Borda (Edge/Robótica)** | **TMRL** | Alta eficiência de memória e velocidade de recuperação. |

---

## 3. Lacunas de Pesquisa e Oportunidades de Inovação

Para o laboratório universitário, estas são as fronteiras não resolvidas (Oportunidades de Artigos/Patentes):

1.  **Raciocínio Espacial 3D:** Integrar SLAM ao RAG para entender a profundidade e posição real de objetos.
2.  **Audio-Visual Grounding:** Vincular sons específicos a regiões da imagem (ex: saber qual máquina está apitando em um painel cheio).
3.  **Poda Dinâmica de Grafo:** Algoritmos para RAG em tempo real (Streaming) que decidem o que "esquecer" para economizar RAM.
4.  **Amostragem Adaptativa:** Aumentar o FPS da análise automaticamente apenas quando há movimento relevante.

---

## 4. Referências Selecionadas

1.  **Luo et al. (2025).** *Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension*. NeurIPS 2025.
2.  **Jeong et al. (2025).** *VideoRAG: Retrieval-Augmented Generation over Video Corpus*. arXiv:2501.05874.
3.  **Zhu et al. (2026).** *Right Answer at the Right Time: Temporal Retrieval-Augmented Generation via Graph Summarization*. ICLR 2026 (Sub).
4.  **Alibaba Qwen Team (2026).** *Qwen3-VL: A Unified Representation Space for Multimodal Understanding*. Official Technical Report.
5.  **Twelve Labs Documentation (2026).** *Marengo 3.0 & Pegasus 1.5: Video-Native Foundation Models*.
6.  **Huynh et al. (2026).** *Temporal-aware Matryoshka Representation Learning*. Jan 2026.

---
*Compilado por Gemini CLI em 03/05/2026.*
