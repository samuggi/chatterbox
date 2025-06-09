# 한국어 TTS 모델 파인튜닝 가이드 (ESPnet 기준)

이 가이드는 ESPnet2 도구모음을 사용하여 기존에 학습된 TTS 모델(예: 영어 모델)을 한국어 음성 데이터셋에 파인튜닝하는 방법론을 설명합니다. ESPnet의 레시피 구조와 일반적인 파인튜닝 절차를 기반으로 합니다.

## 1. 기본 개념 및 준비 사항

### 파인튜닝이란?
파인튜닝은 이미 대규모 데이터셋으로 학습된 모델(pre-trained model)을 가져와, 특정 작업이나 도메인, 또는 새로운 언어(여기서는 한국어)에 맞게 소규모의 특정 데이터셋으로 추가 학습하는 과정을 의미합니다. 이를 통해 적은 데이터와 시간으로도 높은 품질의 모델을 얻을 수 있습니다.

### 필요한 것
*   **ESPnet2 설치:** ESPnet2가 설치되어 있고, 기본적인 `run.sh` 스크립트 실행 방법을 숙지하고 있어야 합니다.
*   **사전 학습된 모델 (Pre-trained Model):** 파인튜닝의 시작점으로 사용할 강력한 TTS 모델이 필요합니다. ESPnet Model Zoo에서 영어 또는 다국어 TTS 모델을 선택할 수 있습니다. (예: Tacotron2, Transformer-TTS, FastSpeech2, VITS 등)
*   **한국어 음성 데이터셋:** 파인튜닝에 사용할 한국어 음성 데이터와 해당 텍스트 스크립트가 필요합니다.
*   **GPU:** TTS 모델 학습에는 GPU가 필수적입니다.

## 2. 한국어 음성 데이터셋 준비

### 데이터셋 종류 및 크기
*   **단일 화자 vs 다중 화자:** 파인튜닝하려는 모델의 목표에 따라 단일 화자 데이터셋 또는 다중 화자 데이터셋을 준비합니다.
*   **데이터 크기:** 파인튜닝은 적은 데이터로도 가능하지만, 일반적으로 고품질의 데이터가 많을수록 좋습니다. 수 시간 분량의 깨끗한 단일 화자 데이터로도 괜찮은 결과를 얻을 수 있으며, 다중 화자의 경우 화자당 데이터 양과 총 화자 수를 고려해야 합니다.
*   **음질:** 깨끗하고 소음이 적은 22.05kHz 또는 16kHz (모델에 따라 다름, VITS는 주로 22.05kHz 사용) 샘플링 속도의 WAV 파일 형식을 권장합니다.

### 데이터셋 형식 (ESPnet Kaldi 스타일)
ESPnet은 Kaldi 스타일의 데이터 디렉토리를 사용합니다. 각 데이터셋 분할(예: `train`, `dev`, `eval`)에 대해 다음 파일들을 준비해야 합니다.

*   `wav.scp`: 각 발화의 ID와 해당 WAV 파일의 전체 경로를 매핑합니다.
    ```
    utt_id_001 /path/to/audio/utt_id_001.wav
    utt_id_002 /path/to/audio/utt_id_002.wav
    ...
    ```
*   `text`: 각 발화 ID와 해당 한국어 텍스트 스크립트를 매핑합니다.
    ```
    utt_id_001 안녕하세요.
    utt_id_002 만나서 반갑습니다.
    ...
    ```
*   `utt2spk`: 각 발화 ID와 해당 화자 ID를 매핑합니다. 단일 화자 데이터셋의 경우 모든 발화가 동일한 화자 ID를 가집니다.
    ```
    utt_id_001 speaker_A
    utt_id_002 speaker_A
    ...
    ```
*   `spk2utt` (선택 사항): 각 화자 ID와 해당 화자의 발화 ID 목록을 매핑합니다. `utils/utt2spk_to_spk2utt.pl` 스크립트로 생성할 수 있습니다.

### 데이터 전처리 및 정제
*   **음성 파일 정제:**
    *   음성 파일 시작과 끝의 긴 묵음 구간을 제거합니다.
    *   발화 중간의 긴 묵음은 필요에 따라 분리하거나, 텍스트에 `<pau>` (pause)와 같은 특수 토큰을 추가하여 모델이 학습하도록 유도할 수 있습니다.
*   **텍스트 정규화 (Text Normalization):**
    *   숫자, 약어, 특수 문자, 외국어 등을 한국어 발음에 맞게 변환합니다. (예: "123" -> "백이십삼")
    *   ESPnet의 `textcleaner` 옵션을 사용하거나, 한국어에 맞는 별도의 전처리 스크립트를 작성할 수 있습니다.
*   **G2P (Grapheme-to-Phoneme) 변환:**
    *   한국어 텍스트(한글)를 발음 기호(phoneme) 또는 자소(jaso) 시퀀스로 변환하는 과정입니다. 이는 모델이 음성 표현을 더 쉽게 학습하도록 돕습니다.
    *   ESPnet은 다양한 G2P 도구를 지원하며, 한국어의 경우 `g2pk` (Kyubyong/g2pK 기반) 또는 `korean_jaso` (jdongian/python-jamo 기반) 등을 사용할 수 있습니다.
    *   `run.sh` 실행 시 `--g2p` 옵션 (예: `--g2p g2pk_no_space`)과 `--token_type phn` (또는 jaso 사용 시 `char`)을 지정합니다.

## 3. ESPnet TTS 레시피를 이용한 파인튜닝 절차

ESPnet의 TTS 레시피(`egs2/<your_chosen_recipe>/tts1/run.sh`)를 기반으로 설명합니다. JVS 레시피의 파인튜닝 예시를 많이 참조합니다.

### 단계 1-5: 한국어 데이터 준비 및 토큰화
1.  **ESPnet TTS 레시피 선택:** `egs2/TEMPLATE/tts1`을 복사하여 새로운 한국어 데이터셋 레시피를 만듭니다. (예: `egs2/kss/tts1`)
2.  `db.sh` 파일에 한국어 데이터셋의 경로를 설정합니다.
3.  `run.sh` 스크립트를 **stage 5 (토큰 리스트 생성)까지** 실행합니다.
    ```bash
    ./run.sh --stage 1 --stop-stage 5 \
             --g2p g2pk_no_space \ # 또는 korean_jaso 등 선택
             --token_type phn \    # G2P 사용 시 phn, 자소 직접 사용 시 char
             --cleaner none \      # 필요시 한국어용 cleaner 설정
             --train_set train \
             --dev_set dev \
             --test_sets "eval1"
    ```
    *   이 과정에서 `dump/token_list/<token_type>_<cleaner>_<g2p>/tokens.txt` 파일이 한국어 데이터 기준으로 생성됩니다. **이 한국어 `tokens.txt` 파일을 사용해야 합니다.**

### 단계 2: 사전 학습된 모델 다운로드
파인튜닝의 기반이 될 사전 학습된 모델을 다운로드합니다. (예: 영어 Tacotron2 모델)
```bash
. ./path.sh
espnet_model_zoo_download --unpack true --cachedir downloads <hugging_face_user/model_name_or_zenodo_link>
# 예: kan-bayashi/ljspeech_tacotron2 (이 모델은 영어 모델입니다)
```
다운로드된 모델의 경로를 확인합니다 (예: `downloads/<model_name_dir>/exp/.../train.loss.best.pth`).

### 단계 3: 파인튜닝 학습 설정
1.  **파인튜닝용 설정 파일 준비:** 기존의 학습 설정 파일(예: `conf/train.yaml` 또는 특정 모델 설정 파일 `conf/tuning/train_tacotron2.yaml`)을 복사하여 파인튜닝용 설정 파일(예: `conf/tuning/finetune_tacotron2_korean.yaml`)을 만듭니다.
    *   **학습률 (learning rate):** 원본 학습률보다 훨씬 작게 설정합니다. (예: `1.0e-4` 또는 `1.0e-5`)
    *   **에폭 수 (max_epoch):** 적은 에폭으로도 충분할 수 있습니다. (예: 10-50 에폭)
    *   **배치 크기 (batch_size 또는 batch_bins):** GPU 메모리에 맞게 조정합니다.
    *   **사전 학습된 모델 경로 지정 (`init_param`):** 이 부분이 가장 중요합니다.
        ```yaml
        # conf/tuning/finetune_tacotron2_korean.yaml 내에 추가 또는 수정
        # ... (다른 파라미터들) ...
        init_param: "/path/to/your/pretrained_model.pth:tts:tts:tts.enc.embed,tts.dec.embed"
        # 또는 tts.embed.weight 등 정확한 임베딩 레이어 파라미터 명칭 확인 필요
        # 위 예시는 인코더와 디코더의 토큰 임베딩 레이어를 제외하고 나머지 tts 모듈의 가중치를 로드합니다.
        # 사전 학습된 모델과 현재 모델의 모듈 이름이 다르면 (예: model.enc -> tts.enc) :<src_key>:<dst_key> 형식으로 매핑합니다.
        # 예: "/path/to/model.pth:model.encoder:tts.enc"
        ```
        *   **중요:** 영어 모델을 한국어로 파인튜닝할 때, 영어 음소와 한국어 음소/자소는 다르므로 **토큰 임베딩 레이어는 사전 학습된 모델의 가중치를 사용하지 않아야 합니다.** `init_param`의 `exclude_keys` 부분 (마지막 콜론 뒤)에 해당 임베딩 레이어의 파라미터 이름을 지정하여 제외합니다. (정확한 파라미터 이름은 모델 아키텍처와 `state_dict()`를 확인하여 찾아야 합니다. 일반적인 예로 `tts.enc.embed.weight`, `tts.dec.embed.weight` 등이 될 수 있습니다.)

### 단계 4: 파인튜닝 학습 실행
수정된 설정과 함께 `run.sh`를 stage 6 (또는 FastSpeech류의 경우 stage 5부터 시작하여 duration 생성 후 stage 7)부터 실행합니다.
```bash
./run.sh --stage 6 \
         --g2p g2pk_no_space \ # stage 1-5와 동일한 G2P 설정 유지
         --token_type phn \
         --train_config conf/tuning/finetune_tacotron2_korean.yaml \
         --train_args "--init_param /path/to/your/pretrained_model.pth:tts:tts:tts.enc.embed,tts.dec.embed" \
         --tag finetune_korean_tacotron2 \
         # (기타 필요한 옵션들: fs, n_fft, n_shift 등은 데이터와 모델에 맞게 설정)
```
*   `--train_args` 내의 `--init_param` 경로는 실제 다운로드 받거나 준비한 사전학습 모델 경로로 수정해야 합니다. yaml 파일 내에 `init_param`을 명시했다면 `--train_args`에서 생략 가능합니다.
*   `--tag` 옵션으로 실험 디렉토리 이름을 지정하여 관리합니다.

### 단계 5: 디코딩 및 평가
학습이 완료되면 stage 8 (디코딩) 및 stage 9 (평가)를 실행하여 생성된 음성을 확인하고 평가합니다.
```bash
./run.sh --stage 8 --stop-stage 9 --tag finetune_korean_tacotron2 \
         --tts_exp exp/<your_user_name>/tts_finetune_korean_tacotron2 # 학습된 모델 경로 지정
```

## 4. 주요 하이퍼파라미터 및 학습 전략 (파인튜닝 시)

*   **Learning Rate:** 사전 학습 시 사용된 학습률보다 1/10 ~ 1/100 정도로 낮게 설정하는 것이 일반적입니다.
*   **Optimizer:** Adam optimizer가 주로 사용됩니다. 사전 학습된 모델의 옵티마이저 상태를 이어받을지, 새로 시작할지 여부도 고려할 수 있습니다 (보통 새로 시작).
*   **Batch Size:** GPU 메모리 한도 내에서 가능한 크게 설정하는 것이 좋지만, 데이터셋 크기가 작을 경우 너무 큰 배치 크기는 불안정할 수 있습니다. ESPnet은 `batch_type="numel"`과 `batch_bins`를 사용하여 동적 배치를 지원합니다.
*   **Number of Epochs:** 전체 데이터셋을 몇 번 반복 학습할지 결정합니다. 파인튜닝은 적은 에폭으로도 빠르게 수렴하는 경향이 있습니다. 검증 세트의 성능을 모니터링하며 조기 종료(early stopping)를 활용할 수 있습니다.
*   **Weight Decay:** 과적합을 방지하기 위한 정규화 파라미터입니다.
*   **Layer Freezing (선택 사항):** 초기 몇 에폭 동안은 사전 학습된 모델의 일부 레이어(예: 인코더)를 고정(freeze)하고, 디코더와 새로운 임베딩 레이어만 학습시키는 전략을 사용할 수 있습니다. 이후 전체 레이어를 함께 파인튜닝합니다. ESPnet 설정에서 특정 레이어의 학습률을 0으로 설정하거나 `requires_grad=False`로 설정하여 구현할 수 있습니다 (모델 코드 수정 또는 `optim_conf` 조정 필요).
*   **Token Embedding Layer:** 언어가 다른 경우 (예: 영어 -> 한국어) 반드시 새로 학습하거나, 매우 작은 학습률로 파인튜닝해야 합니다. `--init_param`에서 해당 레이어를 제외하는 것이 중요합니다.
*   **데이터 증강 (Data Augmentation):** 데이터가 매우 적을 경우, 속도 변화, 볼륨 변화 등의 간단한 데이터 증강을 고려할 수 있으나, TTS에서는 음질에 민감하므로 주의해야 합니다.

## 5. 일반적인 팁 및 문제 해결

*   **작은 데이터셋으로 시작:** 전체 데이터셋으로 바로 학습하기 전에, 작은 서브셋으로 전체 파인튜닝 파이프라인이 정상적으로 동작하는지 확인합니다.
*   **로그 및 TensorBoard 모니터링:** 학습 로그(`exp/.../train.log`)와 TensorBoard를 통해 학습 손실, 검증 손실, 어텐션 맵 등을 주의 깊게 모니터링합니다. 어텐션이 잘 학습되는지(특히 Tacotron2, Transformer 모델의 경우)가 중요합니다.
*   **음성 품질 평가:** 생성된 음성을 직접 들어보고, 필요한 경우 MCD, MOS 등의 객관적/주관적 평가를 수행합니다.
*   **한국어 G2P 선택:** `g2pk` 또는 `korean_jaso` 등 다양한 한국어 G2P/자소분리기를 테스트해보고, 데이터와 모델에 가장 적합한 것을 선택합니다. 생성되는 음소/자소의 일관성과 정확성이 중요합니다.
*   **메모리 부족:** `batch_size` 또는 `batch_bins`를 줄여보거나, `num_workers`를 조정합니다.
*   **학습이 잘 안될 때 (`[327]` ESPnet FAQ 참조):**
    *   데이터 정제 (묵음, 노이즈)가 잘 되었는지 확인합니다.
    *   텍스트 정규화가 충분한지 확인합니다.
    *   어텐션이 깨지는 경우, 학습률을 더 낮추거나, reduction factor를 조정하거나 (Tacotron2), 데이터 품질을 점검합니다.

이 가이드라인은 ESPnet을 사용한 한국어 TTS 모델 파인튜닝의 시작점입니다. 실제 적용 시에는 사용하는 특정 모델 아키텍처, 데이터셋의 특성, 가용한 컴퓨팅 자원 등을 고려하여 세부적인 설정을 조정해야 합니다. ESPnet의 공식 문서와 각 레시피의 README, 그리고 커뮤니티의 논의를 참고하는 것이 좋습니다.
