## CL_GSAN

Code and data for Knowledge-Based Systems 2023 paper: Fine-grained biomedical knowledge negation detection via contrastive learning.

## Introduction

Negation is a common and complex phenomenon in natural language, which will reverse the truth value of knowledge extracted from sentences and make knowledge validation indispensable. Negation detection at the granularity of knowledge triples is still a challenging and minimally explored task, especially in the biomedical domain where multiple knowledge triples frequently exist in one sentence and the same triple can express quite distinct meanings under different contexts. In this paper, we illustrate and define the label conflict problem in biomedical negation detection and propose CL-GSAN, a novel Contrastive Learning-based Gated Structured Attention Network for identifying the negated knowledge triples from the biomedical text. CL-GSAN jointly performs fine-grained knowledge-context modeling within a sentence via a gating mechanism and learns the target triple’s context representations by considering the semantic divergence of claims via contrastive learning. Experimental results demonstrate that CL-GSAN obtains promising performance on three biomedical corpora, including the Chinese Medical Knowledge Negation (CMKN) corpus constructed by us to facilitate fine-grained, i.e., triple-level, knowledge negation detection.

## Requirements

This repo was tested on Pytorch 1.4.0 and allennlp 0.9.0 The main requirements are:

- allennlp = 0.9.0
- pytorch = 1.4.0+cu100
- cuda version = 10.0
