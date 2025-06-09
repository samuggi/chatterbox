<img width="1200" alt="cb-big2" src="https://github.com/user-attachments/assets/bd8c5f03-e91d-4ee5-b680-57355da204d1" />

# Chatterbox TTS

[![Alt Text](https://img.shields.io/badge/listen-demo_samples-blue)](https://resemble-ai.github.io/chatterbox_demopage/)
[![Alt Text](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ResembleAI/Chatterbox)
[![Alt Text](https://static-public.podonos.com/badges/insight-on-pdns-sm-dark.svg)](https://podonos.com/resembleai/chatterbox)
[![Discord](https://img.shields.io/discord/1377773249798344776?label=join%20discord&logo=discord&style=flat)](https://discord.gg/XqS7RxUp)

_♥️ <a href="https://resemble.ai" target="_blank"><img width="100" alt="resemble-logo-horizontal" src="https://github.com/user-attachments/assets/35cf756b-3506-4943-9c72-c05ddfa4e525" /></a>가 만들었습니다._

[Resemble AI](https://resemble.ai)의 첫 번째 프로덕션 등급 오픈 소스 TTS 모델인 Chatterbox를 소개하게 되어 기쁩니다. MIT 라이선스로 제공되는 Chatterbox는 ElevenLabs와 같은 주요 비공개 소스 시스템과 비교하여 벤치마크되었으며, 일대일 평가에서 지속적으로 선호되고 있습니다.

밈, 비디오, 게임, AI 에이전트 등 어떤 작업을 하든 Chatterbox는 콘텐츠에 생명을 불어넣습니다. 또한 음성을 돋보이게 하는 강력한 기능인 **감정 과장 제어**를 지원하는 최초의 오픈 소스 TTS 모델이기도 합니다. 지금 바로 [Hugging Face Gradio 앱](https://huggingface.co/spaces/ResembleAI/Chatterbox)에서 사용해 보세요.

모델이 마음에 들지만 더 높은 정확도를 위해 확장하거나 조정해야 하는 경우 경쟁력 있는 가격의 TTS 서비스(<a href="https://resemble.ai">링크</a>)를 확인해 보세요. 이 서비스는 200ms 미만의 매우 짧은 지연 시간으로 안정적인 성능을 제공하여 에이전트, 애플리케이션 또는 대화형 미디어의 프로덕션 사용에 이상적입니다.

# 주요 세부 정보
- SoTA 제로샷 TTS
- 0.5B Llama 백본
- 고유한 과장/강도 제어
- 정렬 정보 추론을 통한 매우 안정적인 성능
- 0.5M 시간의 정리된 데이터로 학습
- 워터마크 처리된 출력
- 간편한 음성 변환 스크립트
- [ElevenLabs보다 뛰어난 성능](https://podonos.com/resembleai/chatterbox)

# 팁
- **일반 사용 (TTS 및 음성 에이전트):**
  - 기본 설정(`exaggeration=0.5`, `cfg_weight=0.5`)은 대부분의 프롬프트에 적합합니다.
  - 참조 화자의 말하기 스타일이 빠른 경우 `cfg_weight`를 약 `0.3`으로 낮추면 페이스를 개선할 수 있습니다.

- **표현력이 풍부하거나 극적인 음성:**
  - 낮은 `cfg_weight` 값(예: `~0.3`)을 시도하고 `exaggeration`을 약 `0.7` 이상으로 높입니다.
  - `exaggeration`이 높을수록 음성 속도가 빨라지는 경향이 있습니다. `cfg_weight`를 줄이면 더 느리고 신중한 페이스로 보정하는 데 도움이 됩니다.


# 설치
```
pip install chatterbox-tts
```


# 사용법
```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
ta.save("test-1.wav", wav, model.sr)

# 다른 음성으로 합성하려면 오디오 프롬프트를 지정합니다.
AUDIO_PROMPT_PATH="YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav, model.sr)
```
더 많은 예제는 `example_tts.py`를 참조하세요.

## 모델 아키텍처 및 핵심 로직

Chatterbox TTS는 여러 구성 요소가 함께 작동하여 텍스트로부터 고품질 음성을 생성합니다. 주요 구성 요소는 다음과 같습니다.

1.  **`ChatterboxTTS` 클래스 (`chatterbox.tts.Py`):**
    *   TTS 작업을 위한 기본 인터페이스입니다.
    *   모델 로딩 (`from_pretrained`, `from_local`), 조건 준비, 음성 생성을 관리합니다.

2.  **텍스트 처리:**
    *   `punc_norm` 함수: 입력 텍스트의 구두점을 정리하고 표준화합니다.
    *   `EnTokenizer`: 텍스트를 토큰 시퀀스로 변환합니다.

3.  **`T3` (Token-To-Token) 모델 (`chatterbox.models.t3.t3.Py`):**
    *   핵심 텍스트-음성 토큰 변환 모델입니다.
    *   Hugging Face Transformers의 Llama 아키텍처를 백본으로 사용합니다.
    *   입력: 텍스트 토큰, 음성 컨디셔닝 프롬프트 토큰, 화자 임베딩, 감정 과장 파라미터.
    *   출력: 음성 토큰 시퀀스.
    *   학습된 위치 임베딩을 사용하며, Classifier-Free Guidance (CFG)를 활용하여 생성 품질을 제어합니다.

4.  **`VoiceEncoder` 모델 (`chatterbox.models.voice_encoder.voice_encoder.Py`):**
    *   참조 오디오에서 화자 임베딩을 추출합니다.
    *   LSTM 기반 네트워크를 사용하여 오디오의 멜 스펙트로그램으로부터 화자의 음성 특성을 인코딩합니다.
    *   `prepare_conditionals` 과정에서 T3 모델에 전달될 화자 임베딩(`ve_embed`)을 생성하는 데 사용됩니다.

5.  **`S3Gen` (Speech Token to Waveform) 모델 (`chatterbox.models.s3gen.s3gen.Py`):**
    *   T3 모델이 생성한 음성 토큰을 실제 오디오 파형으로 변환하는 보코더입니다.
    *   두 단계로 구성됩니다:
        *   **`S3Token2Mel`**: 음성 토큰을 멜 스펙트로그램으로 변환합니다.
            *   `S3Tokenizer`를 사용하여 음성 토큰을 처리합니다.
            *   참조 오디오로부터 화자 특성(x-vector)을 추출하기 위해 `CAMPPlus` 화자 인코더를 사용합니다.
            *   핵심 로직은 `CausalMaskedDiffWithXvec` 모듈로, 이는 `UpsampleConformerEncoder`와 `CausalConditionalCFM` (Conditional Flow Matching)을 사용하는 `ConditionalDecoder`를 포함합니다. 이는 멜 스펙트로그램 생성을 위해 플로우 매칭 또는 확산 기반 접근 방식을 사용함을 시사합니다.
        *   **`S3Token2Wav`** (S3Token2Mel 상속): `S3Token2Mel`에서 생성된 멜 스펙트로그램을 최종 오디오 파형으로 변환합니다.
            *   `HiFTGenerator` (HiFi-GAN 변형)를 보코더로 사용합니다.
            *   F0(기본 주파수) 예측을 위한 `ConvRNNF0Predictor`를 포함합니다.
    *   `embed_ref` 메소드는 참조 오디오를 처리하여 화자 특성 및 참조 멜/토큰을 추출하는 데 중요합니다.

6.  **컨디셔닝 (`prepare_conditionals` 메소드):**
    *   음성 클로닝을 위해 참조 오디오(`audio_prompt_path`)와 `exaggeration` 파라미터를 사용하여 컨디셔닝 정보를 준비합니다.
    *   T3 모델을 위한 `t3_cond` (화자 임베딩, 참조 오디오의 음성 토큰, 감정 수준 포함)와 S3Gen 모델을 위한 `gen` (참조 오디오의 x-vector 및 멜 스펙트로그램/토큰 포함)을 생성합니다.

7.  **음성 생성 (`generate` 메소드):**
    *   입력 텍스트, (선택적) 참조 오디오 경로, 그리고 다음과 같은 주요 파라미터를 받습니다:
        *   `exaggeration` (기본값: 0.5): 음성의 표현력/과장 정도를 제어합니다.
        *   `cfg_weight` (기본값: 0.5): Classifier-Free Guidance 가중치로, 모델이 텍스트나 화자 특성을 얼마나 엄격하게 따를지 조절합니다. 낮은 값은 더 빠르고 다양한 스타일을, 높은 값은 더 안정적인 스타일을 유도할 수 있습니다.
        *   `temperature` (기본값: 0.8): 생성 과정의 무작위성을 제어합니다. 높은 값은 더 다양하지만 일관성이 떨어질 수 있는 출력을, 낮은 값은 더 결정적이고 일관된 출력을 생성합니다.
    *   준비된 컨디셔닝 정보를 사용하여 T3 모델로 음성 토큰을 생성한 후, S3Gen 모델로 최종 오디오 파형을 생성합니다.

8.  **워터마킹:**
    *   생성된 오디오에는 `perth.PerthImplicitWatermarker`를 사용하여 감지 불가능한 신경망 워터마크가 삽입됩니다.

## 주요 파라미터 설명

Chatterbox TTS 사용 시 `generate` 함수에서 다음과 같은 주요 파라미터를 조정하여 출력 음성을 제어할 수 있습니다.

*   `text` (str): 음성으로 변환할 텍스트입니다.
*   `audio_prompt_path` (str, 선택 사항): 음성 클로닝을 위한 참조 오디오 파일 경로입니다. 제공되면 이 오디오의 음성 스타일로 텍스트가 발화됩니다. 제공되지 않으면 기본 음성 또는 이전에 `prepare_conditionals`로 설정된 음성을 사용합니다.
*   `exaggeration` (float, 기본값: 0.5): 감정 표현의 과장 정도를 조절합니다. 0.0에 가까울수록 차분한 톤이 되며, 1.0 이상으로 설정하면 더 과장된 표현이 됩니다.
*   `cfg_weight` (float, 기본값: 0.5): Classifier-Free Guidance의 가중치입니다. 이 값이 높을수록 모델은 프롬프트(텍스트 및/또는 오디오 프롬프트의 특성)를 더 충실히 따르려고 합니다. 일반적으로 0.3 ~ 0.7 사이의 값을 사용합니다.
    *   참조 화자의 말하기 스타일이 빠른 경우, `cfg_weight`를 약 `0.3`으로 낮추면 페이스 조절에 도움이 될 수 있습니다.
    *   표현력이 풍부하거나 극적인 음성을 원할 경우, `cfg_weight`를 낮추고(`~0.3`) `exaggeration`을 높이는(`~0.7` 이상) 조합을 시도해볼 수 있습니다.
*   `temperature` (float, 기본값: 0.8): 생성 과정의 무작위성을 조절합니다. 값이 높을수록 더 다양하고 예기치 않은 출력이 나올 수 있으며, 낮을수록 더 일관되고 결정적인 출력이 생성됩니다.

# 감사의 말
- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)

# 책임감 있는 AI를 위한 내장 PerTh 워터마킹

Chatterbox에서 생성된 모든 오디오 파일에는 [Resemble AI의 Perth (Perceptual Threshold) 워터마커](https://github.com/resemble-ai/perth)가 포함되어 있습니다. 이는 MP3 압축, 오디오 편집 및 일반적인 조작에도 거의 100% 감지 정확도를 유지하면서 감지할 수 없는 신경망 워터마크입니다.

# 공식 Discord

👋 [Discord](https://discord.gg/XqS7RxUp)에서 저희와 함께 멋진 것을 만들어 보세요!

# 면책 조항
이 모델을 나쁜 일에 사용하지 마세요. 프롬프트는 인터넷에서 자유롭게 사용할 수 있는 데이터에서 가져옵니다.
