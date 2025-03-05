---
title: "Visual Intelligence: Classificazione Istologica"
author: Your Name
date: March 2025
theme: gaia
class:
  - lead
  - invert
paginate: true
backgroundColor: "#ffffff"
marp: false
style: |
  @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&family=Source+Sans+Pro:wght@400;700&display=swap');
  section {
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 1.6rem;
    color: #333333;
    background-color: #ffffff;
    padding: 2rem;
  }
  h1, h2 {
    font-family: 'Roboto', sans-serif;
    color: #1e40af;
    font-weight: 700;
  }
  h3, h4 {
    font-family: 'Roboto', sans-serif;
    color: #2563eb;
  }
  strong {
    color: #dc2626;
    font-weight: 700;
  }
  ul li, ol li {
    margin-bottom: 0.5rem;
  }
  ul li::marker {
    color: #2563eb;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 1rem 0;
  }
  th {
    background-color: #2563eb;
    color: white;
    padding: 0.5rem;
  }
  td {
    border: 1px solid #d1d5db;
    padding: 0.5rem;
    text-align: center;
  }
  tr:nth-child(even) {
    background-color: #f3f4f6;
  }
  blockquote {
    border-left: 5px solid #2563eb;
    padding-left: 1rem;
    color: #4b5563;
    font-style: italic;
  }
  code {
    font-family: 'Courier New', monospace;
    background-color: #f3f4f6;
    padding: 0.2rem 0.4rem;
    border-radius: 3px;
  }
  section.lead {
    background-color: #1e40af;
    color: white;
  }
  section.lead h1, section.lead h2 {
    color: white;
  }
---

<!-- _class: lead -->

# Visual Intelligence Project

## Classificazione Automatica di Immagini Istologiche

#### Your Name | March 2025

---

<!-- header: '📊 Visual Intelligence - ML Project' -->

## 🔍 Introduzione

- **Obiettivo**: Classificazione automatica di campioni istologici polmonari
- **Sfida**: Distinguere tra diversi tipi cellulari (pneumociti I/II, macrofagi alveolari)
- **Motivazione**: Assistere i patologi nell'analisi rapida di campioni istologici

> _"L'intelligenza artificiale può migliorare la precisione diagnostica e ridurre il carico di lavoro nell'analisi istologica"_

---

## 📦 Dataset

- **Origine**: Campioni istologici polmonari con colorazione ematossilina-eosina
- **Composizione**: 3 classi principali (pneumociti tipo I, tipo II, macrofagi alveolari)
- **Dimensioni**: 1000 immagini ad alta risoluzione (768×768 px)

**Preprocessing applicato**:

- Normalizzazione del contrasto
- Augmentation (rotazioni, flipping, variazioni di luminosità)
- Ridimensionamento ottimizzato (analizzato da 768×768 a 5×5 px)

---

## 🧠 Architettura del Modello

```mermaid
graph TD
    A[Input Image] --> B[Conv Layer 1]
    B --> C[MaxPool]
    C --> D[Conv Layer 2]
    D --> E[MaxPool]
    E --> F[Fully Connected]
    F --> G[Output Classification]
```

- **Base Model**: Architettura ispirata a ResNet ma ottimizzata per dimensione
- **Alternative testate**: CNN tradizionale vs. modello leggero personalizzato
- **Efficienza**: Implementazione di mixed precision training per accelerare

---

## 💻 Dettagli Implementativi

- **Framework**: PyTorch con fastai per l'addestramento rapido
- **Hardware**: NVIDIA GeForce RTX 3090 (riduzione tempi da 4h a 45min)
- **Strategie di ottimizzazione**:
  - Data loading efficiente con `num_workers` ottimizzato
  - Batch size calibrata (32-64 immagini)
  - Learning rate finder per determinare LR ottimale

---

## 📈 Risultati & Analisi

| Modello         | Accuracy | F1-Score | Tempo Training |
| --------------- | -------- | -------- | -------------- |
| CNN base        | 87.5%    | 0.86     | 4h             |
| CNN ottimizzata | 91.2%    | 0.90     | 45min          |
| Grayscale       | 83.3%    | 0.82     | 30min          |
| Mini (5×5)      | 89.7%    | 0.88     | 10min          |

**Osservazione chiave**: Il modello funziona sorprendentemente bene anche con immagini molto ridotte (5×5 px)

---

## 🧪 Esperimenti & Scoperte

- **Riduzione dimensionale**: Anche a 5×5 pixel il modello mantiene performance elevate
- **Ipotesi**: Il modello potrebbe basarsi principalmente su pattern di colore
- **Verifica**:
  - Test con immagini in grayscale → risultati inferiori
  - Test con rimozione sfondo → ancora buoni risultati
  - **Conclusione**: Il modello impara pattern di colore associati alle classi

---

## 🚧 Sfide & Miglioramenti

- **Sfide affrontate**:

  - Sovradimensionamento iniziale del modello
  - Tempi di training eccessivi
  - Rischio di apprendimento di features non generalizzabili

- **Miglioramenti futuri**:
  - Implementare tecniche di interpretabilità (Grad-CAM, LIME)
  - Sviluppare architetture attention-based per migliore focalizzazione
  - Validazione con dataset esterni per verificare generalizzazione

---

## 🎓 Concetti ML Applicati

- **Bias-variance trade-off**: Bilanciamento complessità del modello
- **Data augmentation**: Tecniche per arricchire il dataset
- **Transfer learning vs. training from scratch**: Analisi comparativa
- **Problemi di overfitting**: Strategie implementate (dropout, regolarizzazione)

---

<!-- _class: lead -->

## 📝 Conclusioni

- Il modello leggero ottimizzato raggiunge **91.2% di accuracy**
- La scoperta dell'apprendimento basato sul colore evidenzia l'importanza dell'**interpretabilità**
- L'applicazione pratica potrebbe assistere i patologi nella **classificazione rapida** di campioni

---

<!-- _class: lead -->

# Grazie per l'attenzione!

## Domande?

Contatti:
📧 your.email@university.edu
🔗 github.com/yourusername
