# Benchmarks e Métricas de Avaliação — Video RAG

> Referência consolidada dos principais datasets, benchmarks e métricas usados para avaliar sistemas Video RAG em 2025–2026.

---

## 1. Benchmarks Principais

### 1.1 Benchmarks de Busca Temporal

| Benchmark | Tamanho | Foco | Limitação |
|-----------|---------|------|-----------|
| **LV-HAYSTACK** | 480h / 15k+ instâncias | Busca temporal em vídeos longos | Não avalia geração de respostas |
| **LongerVideos** | 134h | RAG em vídeos extremamente longos | Pouca diversidade de domínio |
| **NExT-GQA** | ~10k QA pairs | Grounded Question Answering | Vídeos curtos (< 3 min) |

### 1.2 Benchmarks de Alucinação Visual

| Benchmark | Conferência | O que avalia |
|-----------|------------|-------------|
| **VidHalluc** | CVPR 2025 | Alucinação em ação, sequência temporal e transição de cena |
| **ARGUS** | ICCV 2025 | Omissão e alucinação em Video-LLMs |
| **VideoHallu** | 2025 | Cenários fisicamente impossíveis (Sora, Kling, Veo2) |

### 1.3 Benchmarks de Grounding Espaço-Temporal

| Benchmark | Tamanho | Foco |
|-----------|---------|------|
| **OmniGround** | 3.475 vídeos / 81 categorias | Grounding espacial e temporal complexo |
| **GaRAGe** | Large-scale | RAG com anotações humanas de grounding |

---

## 2. Métricas de Avaliação

### 2.1 Métricas de Grounding Temporal

```
mIoU (Mean Intersection over Union)
  → Sobreposição entre janela temporal prevista e ground truth
  → Principal métrica para localização temporal
  
R@k, IoU=α (Recall at k com threshold α)
  → % de instâncias onde o top-k inclui segmento com IoU ≥ α
  → Ex: R@1 IoU=0.5 = % de casos onde o melhor resultado tem ≥ 50% de sobreposição
  
mIoP (Mean Intersection over Precision)
  → Mais restritivo que mIoU — foca em grounding fino
  
Hit@1
  → Usado especificamente para highlight detection
```

### 2.2 Métricas de Qualidade de Resposta

```
Acc@GQA (Grounded Question Answering Accuracy)
  → Avalia SIMULTANEAMENTE: grounding correto + resposta correta
  → Métrica mais completa para Video RAG end-to-end
  
Acc@QA
  → Apenas a resposta — ignora qual segmento foi recuperado
  
Faithfulness Score (Ragas)
  → % de afirmações na resposta que são suportadas pelo contexto recuperado
  → Escala 0–1
  
Answer Completeness
  → A resposta cobre todos os aspectos relevantes da pergunta?
```

### 2.3 Métricas de Avaliação por LLM-as-Judge

```python
# Framework recomendado: Ragas + GPT-4.1 ou Gemini 2.5 Pro como juiz

from ragas.metrics import (
    faithfulness,        # Afirmações suportadas pelo contexto
    answer_relevancy,    # Relevância da resposta para a pergunta
    context_precision,   # Precisão do contexto recuperado
    context_recall       # Recall do contexto em relação ao ground truth
)
```

---

## 3. Datasets de Vídeo para Experimentos

### 3.1 Disponíveis Gratuitamente

| Dataset | Horas | Domínio | Uso recomendado |
|---------|-------|---------|----------------|
| **ActivityNet Captions** | ~849h | Atividades gerais | Baseline de grounding temporal |
| **Kinetics-700** | ~700h | Ações humanas | Classificação de atividades |
| **YouTube-8M** | >350k vídeos | Multi-domínio | Recuperação em escala |
| **EgoSchema** | ~250h | Primeira pessoa | Raciocínio egocêntrico |

### 3.2 Para Domínios Específicos (Laboratório)

| Domínio | Dataset Recomendado | Fonte |
|---------|---------------------|-------|
| **Educacional** | Lecture recordings (MIT OpenCourseWare) | Gratuito |
| **Industrial** | MVTec Anomaly Detection (vídeos) | Kaggle |
| **Segurança** | VIRAT Ground Dataset | NIST |
| **Médico** | Cholec80 (cirurgia laparoscópica) | GitHub |

---

## 4. Configuração de Avaliação Recomendada

### Para o Laboratório (Modelo Híbrido)

```python
# Avaliação mínima recomendada para publicação

EVALUATION_SUITE = {
    "temporal_grounding": {
        "primary_metric": "mIoU",
        "secondary_metric": "R@1_IoU=0.5",
        "benchmark": "LV-HAYSTACK (subset)"
    },
    "qa_quality": {
        "primary_metric": "Acc@GQA",  
        "secondary_metric": "Faithfulness",
        "benchmark": "NExT-GQA"
    },
    "hallucination": {
        "primary_metric": "VidHalluc Score",
        "judge": "GPT-4o-mini (custo reduzido)"
    },
    "efficiency": {
        "metric": "Latência (ms) × Acc@GQA",  # Pareto
        "hardware_target": "CPU Intel Xeon (laboratório)"
    }
}
```

---

## 5. Referências

- LV-HAYSTACK: arXiv (2025) — 480h, 15k instâncias
- VidHalluc: CVPR 2025
- ARGUS: ICCV 2025
- OmniGround: CVPR 2025
- NExT-GQA: Benchmark padrão da comunidade
- Ragas: ragas.io — framework de avaliação de RAG
- DeepEval: deepeval.com — avaliação multimodal (2025)

---

*Compilado em Junho/2026*
