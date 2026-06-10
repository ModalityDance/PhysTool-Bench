<a name="readme-top"></a>

<div align="center">
  <img src="./assets/LOGO.png" alt="Project Logo" width="250">
  <h1 align="center">Beyond APIs: Probing the Limits of MLLMs in Physical Tool Use</h1>
</div>

<div align="center">

  <!-- Project Page -->
  <a href="https://modalitydance.github.io/PhysTool-Bench/">
    <img src="https://img.shields.io/badge/Project-Page-6a5acd?style=for-the-badge" alt="Project Page">
  </a>

  <!-- Paper Link -->
  <a href="https://arxiv.org/abs/2606.10803">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv" alt="Paper">
  </a>

  <!-- HuggingFace Papers -->
  <a href="{huggingface_papers_url}">
    <img src="https://img.shields.io/badge/HuggingFace-Papers-fcc21b?style=for-the-badge&logo=huggingface&logoColor=white" alt="HF Papers">
  </a>

  <!-- HuggingFace Datasets -->
  <a href="https://huggingface.co/datasets/ModalityDance/PhysTool-Bench">
    <img src="https://img.shields.io/badge/HuggingFace-Datasets-fcc21b?style=for-the-badge&logo=huggingface&logoColor=white" alt="HF Datasets">
  </a>

</div>



Welcome to _PhysTool-Bench_! 👋 

**_PhysTool-Bench_ is a benchmark that evaluates how well MLLMs perceive, select, and sequence physical tools in real-world scenes.**

It consists of 2 tasks that separate **visual recognition** from **functional planning**:

- **Task I – Tool Recognition**: List all visible tools in a cluttered scene (image only).  
- **Task II – Tool Selection & Planning**: Given an **real scenario image** + a **brief task instruction**, output the **ordered sequence** of required tools.

## 📊 Dataset at a Glance

<div align="center">

| Key Numbers | Value |
|-------------|-------|
| Total queries (image+task+answers) | 2,510 |
| Unique physical tools | 2,678 |
| Tools per scene | 8.6 (3.1 required, 5.5 distractors) |

</div>


## 🪐 Key Features

- **Two‑Task Design:** Decouples *recognition* (all visible tools) from *planning* (select + order).
- **Real‑World Tool Variety:** Across 57 categories (manufacturing, healthcare, farming, etc.).
- **Challenging Distractors:** 3–10 visually/functionally similar decoys per scene.
- **Rich Evaluation Metrics:** EM, TCR, SR@k, and fine‑grained error analysis.


<div align="center">
  <figure>
    <img src="./assets/overview.png" alt="Overview" style="max-width: 70%; height: auto;">
    <br>
    <figcaption><em>Quick Overview of PhysTool-Bench.</em></figcaption>
  </figure>
</div>


## 📑 Table of Contents <span id="table-of-contents"></span>


* <a href='#quick-start'>🚀 Quick Start</a>
  * <a href='#installation'>Installation</a>
  * <a href='#data'>Download Data</a>
  * <a href='#infer'>Inference</a>
  * <a href='#eval'>Evaluation</a>

* <a href='#how-it-works'>✨ How It Works</a>
* <a href='#acknowledgements'>🌱 Acknowledgements</a>
* <a href='#citation'>📚 Citation</a>

<!-- * <a href='#examples'>⬇️ Examples</a> -->
<!-- * * <a href='#documentation'>📖 Documentation</a> -->
<!-- * <a href='#todo'>📝 TODO List</a> -->



## 🚀 Quick Start <span id="quick-start"></span>


### 1. Installation <span id="installation"></span>

Since different models require conflicting versions of transformers and other libraries, we provide **separate** Conda environments for running **different model families**. Choose the one that matches the model you want to evaluate.

If you prefer using models via **API** (e.g., GPT-4o), you can **skip** the environment setup and directly run the inference scripts with your API key.

#### **Conda (recommended)**

For Open-Flamingo
```
conda create -n flamingo_env python=3.10 -y
eval "$(conda shell.bash hook)" && conda activate flamingo_env

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.28.1
pip install open-flamingo==2.0.1 --no-deps
pip install einops einops-exts open_clip_torch huggingface-hub Pillow accelerate sentencepiece
```

For mPLUG-Owl3
```
conda create -n mPLUG_env python=3.10 -y
eval "$(conda shell.bash hook)" && conda activate mPLUG_env

pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.40.2
pip install icecream einops 'accelerate>=0.26.0' pillow
```

For MiniCPM
```
conda create -n minicpm_env python=3.10 -y
eval "$(conda shell.bash hook)" && conda activate minicpm_env

pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "transformers<5.0.0" "timm>=1.0.0" "accelerate>=1.0.0"
pip install sentencepiece pillow decord einops minicpmo hyperpyyaml speechbrain librosa onnx onnxruntime-gpu
```

#### **Hardware Requirements (recommended to fill)**

* GPU: Recommended for training and inference (CUDA-compatible)
* Python: **3.10**
* CUDA: **12.1 / 12.4**
* Frameworks: **PyTorch 2.4.0, Transformers (version varies per model), Accelerate**


### 2. Download datasets <span id="data"></span>

```bash
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

or download manually from:

* https://huggingface.co/datasets/ModalityDance/PhysTool-Bench


### 3. Inference <span id="infer"></span>

#### **Task I Inference**

Throught API:
```bash
python scripts/inference/task_i_api.py -t YOUR_API_KEY -m <model_name> -i data/generation_checkpoint.json -o results --delay 1

# For example:
python scripts/inference/task_i_api.py -t sk-xxxx -m gpt-4o -i data/generation_checkpoint.json --delay 1
```

For local models:
```bash
python scripts/inference/task_i_<model_name>.py

# For example:
python scripts/inference/task_i_Openflamingo9B.py
```

#### **Task II Inference**

Throught API:
```bash
python scripts/inference/task_ii_api.py -t YOUR_API_KEY -m <model_name> -i data/generation_checkpoint.json -o results --delay 2

# For example:
python scripts/inference/task_ii_api.py -t sk-xxxx -m gpt-4o -i data/generation_checkpoint.json -o results --delay 2
```

For local models:
```bash
python scripts/inference/task_ii_<model_name>.py

# For example:
python scripts/inference/task_ii_Openflamingo9B.py
```


### 4. Evaluation <span id="eval"></span>

#### **Task I**

```bash
python scripts/evaluation/eval_tool_finding.py \
    --model <model_name> \
    --ground-truth data/corrected_tools.json \
    --predictions results/all_tools_identified_<model_name>.json \    # predicted tool lists from Task I inference
    --output-json results/eval_tool_finding_<model_name>.json \       # where evaluation results will be saved
    --match-method {fuzzy|strict}

# For example:
python scripts/evaluation/eval_tool_finding.py \
    --model gpt-4o \
    --ground-truth data/corrected_tools.json \
    --predictions results/all_tools_identified_gpt-4o.json \
    --output-json results/eval_tool_finding_gpt-4o.json \
    --match-method fuzzy
```

#### **Task II**

Use Gemini as a Judge (by API):
```bash
python scripts/evaluation/eval_gemini.py -t YOUR_API_KEY -m <model_name> \
    -r results/task_ii_results_<model_name>.json \
    -o results/evaluation_of_<model_name>_with_gemini.json \
    -k 1,2,3

# For example:
python scripts/evaluation/eval_gemini.py -t sk-xxxx -m MiniCPM \
    -r results/task_ii_results_MiniCPM.json \
    -o results/evaluation_of_MiniCPM_with_gemini.json \
    -k 1,2,3
```

Use exsiting matching pairs:
```bash
python scripts/evaluation/eval_offline.py -m <model_name> \
    -r results/task_ii_results_<model_name>.json \
    -o results/evaluation_of_<model_name>_with_gemini.json \
    -k 1,2,3

# For example:
python scripts/evaluation/eval_offline.py -m MiniCPM \
    -r results/task_ii_results_MiniCPM.json \
    -o results/evaluation_of_MiniCPM_with_gemini.json \
    -k 1,2,3
```

## ✨ How It Works <span id="how-it-works"></span>

_PhysTool-Bench_ is built through **controlled expansion and iterative refinement** — starting from a seed set of tools, growing organically, and verifying every step.

<div align="center">
  <figure>
    <img src="./assets/physTool_bench_pipeline.png" alt="Pipeline" style="max-width: 90%; height: auto;">
    <br>
    <figcaption><em>Three-stage construction (left) and two-task evaluation (right).</em></figcaption>
  </figure>
</div>


### 🏗️ 1. Tool Bank: Grow from Seeds, Not Everything at Once

- Start with **310 manually curated tools**, then iteratively expand.
- **Recycle novel distractors** generated during query creation back into the bank.
- → Covers **2,678 tools** across 57 categories, avoids artificial “tool spotting”, and ensures **broad + balanced** coverage.

### 🔍 2. Query Generation + QC: Relentless Refinement

- **Distractors**: 3–10 per scene, visually *or* functionally similar to targets. 86.9% tasks require strict order.
- **Three QC stages** (LLM necessity audit → programmatic alignment → human visual review) to remove ambiguity, artificial cues, or physically unrealistic images.
- → Every query has a **clear, verifiable ground truth** (humans reach 75% EM on familiar tasks).

### 🧪 3. Two‑Task Evaluation: Pinpoint Where Models Fail

- **Task I (Recognition)** – image only → list *all* visible tools. Measures pure visual enumeration.
- **Task II (Planning)** – image + instruction → ordered required tools. Measures functional mapping + sequencing.
- If a model sees correctly (Task I) but plans poorly (Task II), the bottleneck is **physical commonsense**, not vision.



## 🌱 **Acknowledgements** <span id="acknowledgements"></span>

An example: We would like to thank the contributors, open-source projects, and research communities whose work made **_PhysTool-Bench_** possible. This project builds upon ideas, tools, and datasets developed by the broader machine learning and information retrieval ecosystem. 

- 🖼️ **Image Generation** – [Nano Banana Pro](https://aistudio.google.com/models/nano-banana) (synthetic scene rendering)  
- 🧠 **Open‑weight Models**  
  - [MiniCPM‑V](https://github.com/OpenBMB/MiniCPM-V)  
  - [mPLUG‑Owl3](https://github.com/X-PLUG/mPLUG-Owl)  
  - [OpenFlamingo](https://github.com/mlfoundations/open_flamingo)  
  - [InternVL](https://github.com/OpenGVLab/InternVL)  
  - [DeepSeek‑VL](https://github.com/deepseek-ai/DeepSeek-VL)  
  - [Kimi‑VL](https://github.com/MoonshotAI/Kimi-VL)  
  - [Ovis](https://github.com/AIDC-AI/Ovis)  
- 💻 **Code & Libraries** – [🤗 Transformers](https://github.com/huggingface/transformers), [vLLM](https://github.com/vllm-project/vllm), [PyTorch](https://pytorch.org), [PIL](https://pypi.org/project/pillow/), [requests](https://requests.readthedocs.io)  
- 📚 **Dataset & Classification** – [UNSPSC](https://www.ungm.org/Public/UNSPSC), manual annotation & QC team  
- 📊 **Inference & Evaluation** – vLLM, custom evaluation scripts (offline, Gemini‑based, fuzzy matching)  

This project is licensed under the **MIT License**. Please refer to the `LICENSE` file for full details.


## 📚 **Citation** <span id="citation"></span>

If you use **_PhysTool-Bench_** in your research or applications, please consider citing:

```bibtex
@article{PhysTool-Bench2026,
  title        = {Beyond APIs: Probing the Limits of MLLMs in Physical Tool Use},
  author       = {Zhixin Ma and Yutong Zhou and Yongqi Li and Chong Wah Ngo and Wenjie Li},
  journal      = {arXiv preprint arXiv:{xxxx.xxxxx}},
  year         = {2026}
}
```

<!-- Modify the repository URL accordingly. -->

<div align="center">

<a href="https://github.com/ModalityDance/PhysTool-Bench">
  <img src="https://img.shields.io/badge/⭐ Star%20us%20on%20GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />
</a>

<a href="https://github.com/ModalityDance/PhysTool-Bench/issues">
  <img src="https://img.shields.io/badge/🐞 Report%20Issues-e74c3c?style=for-the-badge&logo=github" />
</a>

<a href="https://github.com/ModalityDance/PhysTool-Bench/discussions">
  <img src="https://img.shields.io/badge/💬 Discussions-20c997?style=for-the-badge&logo=github" />
</a>
<br/>
⭐ <b>Thank you for visiting <em>PhysTool-Bench</em>!</b> ⭐

</div>
