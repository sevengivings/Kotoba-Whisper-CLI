# Third-Party Notices

언어: **한국어**

이 문서는 Kotoba-Whisper CLI와 standalone 버전을 배포하거나 다른 컴퓨터에 설치할 때 확인해야 할 주요 제3자 모델, 도구, 라이브러리 고지입니다. 법률 자문은 아니며, 실제 공개 배포나 상업 배포 전에는 각 프로젝트의 최신 라이선스와 약관을 다시 확인하세요.

확인일: 2026-07-31

## 핵심 요약

- 현재 프로젝트는 기본적으로 모델 파일을 저장소에 포함하지 않고, 실행 시 Hugging Face 또는 Ollama를 통해 사용자가 내려받는 흐름입니다.
- 이 방식은 모델 가중치를 직접 재배포하는 것보다 라이선스 부담이 작습니다.
- 나중에 Windows exe, ZIP, 설치 프로그램 등에 모델 가중치나 변환 모델을 포함하면 재배포에 해당할 수 있으므로, 라이선스 전문과 출처 고지를 반드시 포함하세요.
- 사용자가 처리하는 영상/음성 자체의 저작권, 개인정보, 성인물 관련 법규 준수 책임은 사용자에게 있습니다.

## 모델

### Kotoba-Whisper v2.2

- 사용 위치: 일본어 음성 인식/자막 추출
- 모델 ID: `kotoba-tech/kotoba-whisper-v2.2`
- 라이선스: Apache License 2.0으로 표시됨
- 출처: https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2
- 참고: NVIDIA NGC의 Kotoba Whisper 설명도 상업적 사용 가능 및 Apache 2.0 추가 정보를 표시합니다.
  - https://catalog.ngc.nvidia.com/orgs/nvidia/riva/models/kotoba_whisper

Apache 2.0 모델을 재배포하는 경우 일반적으로 다음을 지켜야 합니다.

- Apache License 2.0 전문 포함
- 원 저작권/저작자/라이선스 표시 보존
- 원본에 `NOTICE` 파일이 있으면 해당 고지 포함
- 수정 또는 변환한 모델을 배포하면 변경 사실 표시

현재 프로젝트처럼 모델을 코드 저장소에 넣지 않고 사용자가 직접 다운로드하게 하는 경우에도 README 또는 배포 문서에 모델 출처와 라이선스를 표시하는 것을 권장합니다.

### OpenAI Whisper

- 관계: Kotoba-Whisper v2.2의 기반 모델로 설명됨
- 라이선스: OpenAI Whisper의 코드와 모델 가중치는 MIT License로 공개되어 있음
- 출처: https://github.com/openai/whisper

### Hy-MT2 GGUF 번역 모델

- 사용 위치: Ollama 기반 일본어 SRT -> 한국어 SRT 번역
- 기본 고품질 모델: `hf.co/mradermacher/Hy-MT2-30B-A3B-GGUF:Q4_K_M`
- 권장 최소 모델: `hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M`
- 출처: Ollama에서 `hf.co/...` 형식으로 Hugging Face 모델을 내려받아 사용

주의:

- 이 프로젝트는 Hy-MT2 GGUF 모델 파일을 저장소에 포함하지 않습니다.
- 사용자가 `ollama pull ...`로 직접 내려받는 방식입니다.
- 나중에 번역 모델을 설치 파일에 포함하려면 해당 Hugging Face 저장소의 원본 라이선스, 변환 모델 라이선스, 원 모델 라이선스를 별도로 확인해야 합니다.

### pyannote diarization models

- 현재 프로젝트의 기본 전사 흐름에서는 사용하지 않습니다.
- Kotoba-Whisper 모델 카드의 화자분리 예제에는 `pyannote/segmentation-3.0`, `pyannote/speaker-diarization-3.1` 사용이 포함되어 있으며, 별도 약관 동의와 Hugging Face 토큰이 필요할 수 있습니다.
- 향후 화자분리 기능을 추가한다면 pyannote 계열 모델의 약관을 별도로 검토하세요.

## 주요 런타임과 라이브러리

아래 항목은 현재 Docker판 또는 standalone판에서 직접 사용하는 주요 구성요소입니다. 정확한 전체 목록은 `requirements.txt`, `standalone/pyproject.toml`, `standalone/uv.lock`, Docker base image의 구성요소를 확인하세요.

| 구성요소 | 사용 위치 | 확인된 라이선스/표시 |
| --- | --- | --- |
| Hugging Face Transformers | 모델 로딩/추론 | Apache 2.0 |
| Hugging Face Hub | 모델 다운로드 | Apache |
| PyTorch | CUDA 추론 | BSD-3-Clause |
| torchaudio | 오디오/추론 보조 | BSD 계열로 표시 |
| safetensors | 모델 파일 로딩 | Apache 2.0 계열로 표시 |
| sentencepiece | 토크나이저 | Apache |
| protobuf | 모델/설정 처리 | 3-Clause BSD |
| tqdm | 진행률 표시 | MPL-2.0 AND MIT |
| imageio-ffmpeg | standalone ffmpeg 바이너리 확보 | BSD-2-Clause |
| PyYAML | 설정 파일 처리 | MIT |
| Docker base image `nvidia/cuda` | Docker판 CUDA 런타임 | NVIDIA CUDA 이미지 약관 확인 필요 |
| ffmpeg | Docker판 오디오 추출, standalone은 imageio-ffmpeg 번들 사용 | 빌드 구성에 따라 LGPL/GPL 조건 확인 필요 |
| Ollama | 로컬 번역 모델 실행 | Ollama 자체 라이선스 및 모델별 라이선스 확인 필요 |

## 배포 형태별 권장 사항

### 소스 코드만 배포

- 이 문서를 저장소에 포함합니다.
- README에서 모델과 주요 라이선스 고지로 연결합니다.
- 모델 가중치, Ollama 모델, Hugging Face 캐시, 사용자 입력/출력 파일은 저장소에 포함하지 않습니다.

### Docker 이미지 배포

- 이미지가 포함하는 OS 패키지, CUDA 런타임, Python 패키지, ffmpeg 조건을 함께 확인합니다.
- 모델 캐시를 이미지에 굽지 않는 구성이 가장 단순합니다.
- 모델 캐시를 이미지에 포함하면 Kotoba-Whisper 모델 재배포 조건을 충족해야 합니다.

### Windows exe 또는 ZIP 배포

- 모델을 번들하지 않고 첫 실행 시 다운로드하게 하면 배포 크기와 라이선스 부담이 줄어듭니다.
- 모델을 포함하는 “완전 오프라인 배포판”을 만들 경우 다음을 포함하세요.
  - `THIRD_PARTY_NOTICES.md`
  - Apache 2.0, MIT, BSD, MPL 등 관련 라이선스 전문
  - 모델 출처와 버전 또는 commit hash
  - 수정/변환 여부
- `ffmpeg` 실행 파일을 직접 포함하면 해당 빌드의 라이선스 조건을 별도로 확인하세요.

## 프로젝트에서 지켜야 할 실무 원칙

- `.gitignore`로 모델 캐시, 입력 영상, 출력 자막, 사용자 설정 파일을 계속 제외합니다.
- README에는 “모델은 사용자가 직접 다운로드한다”는 흐름을 유지합니다.
- 배포 패키지를 만들 때는 실제 포함된 파일 기준으로 `LICENSES/`와 `NOTICE`를 다시 생성합니다.
- 새 모델을 기본값으로 추가할 때는 모델 ID, 출처, 라이선스, 권장 VRAM을 함께 문서화합니다.
