# FITNESS Java ONNX Web

브라우저는 카메라와 MediaPipe 랜드마크 추출만 담당합니다.  
Good/Bad 판정, 카운트, 점수, 피드백은 Java Spring Boot 서버가 ONNX 모델로 처리합니다.

## 실행 방법

```bash
cd Fitness_Java_ONNX_Web
mvn spring-boot:run
```

접속:

```text
http://localhost:8080/exercise.html
```

## 구조

```text
src/main/java/com/fitness/
  FitnessApplication.java
  WebSocketConfig.java
  WorkoutWebSocketHandler.java
  WorkoutAnalyzer.java

src/main/resources/static/
  index.html
  exercise.html
  models/*.onnx
  static/MediaPipe, ONNX Runtime Web assets
```

## 기준 로직

Java의 `WorkoutAnalyzer.java`는 Python 원본 기준으로 이식했습니다.

- `Pose_detect.py`
  - 33개 landmark
  - x, y, z, visibility
  - 골반 중심 기준 이동
  - 어깨-골반 거리 스케일 정규화
- `Classifier.py`
  - 30프레임 버퍼
  - 30프레임 미만이면 Good
  - ONNX 출력 0=Good, 1=Bad
- `Count_module.py`
  - 각도 상태 머신으로 카운트
- `Score_manager.py`
  - Good/Bad 패널티
  - 안정성/정확도 점수
  - 피드백 문구 생성


## Render 배포 방법

1. 이 폴더 전체를 GitHub 저장소에 업로드합니다.
2. Render에서 `New +` → `Blueprint` 또는 `Web Service`를 선택합니다.
3. 저장소를 연결합니다.
4. Docker 환경으로 배포합니다.
5. 배포 후 접속 주소는 보통 아래 형태입니다.

```text
https://fitness-java-onnx-web.onrender.com
```

운동 페이지:

```text
https://fitness-java-onnx-web.onrender.com/exercise.html
```

## 중요

이 버전은 B 방식입니다.

```text
브라우저 카메라
→ MediaPipe landmark 추출
→ Java WebSocket 전송
→ Java ONNX Runtime 추론
→ 카운트/점수/피드백 반환
```

Render 무료 플랜은 CPU가 약하고 sleep이 있어 첫 접속이 느릴 수 있습니다.


## Fix

Spring Boot Maven plugin `repackage` is enabled so Render can run `java -jar app.jar`.
