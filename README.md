# FITNESS Java Render Backend

이 버전은 Render 배포용 B 방식입니다.

## 역할

- 브라우저: 카메라 실행, MediaPipe landmark 추출, 화면 표시
- Java Spring Boot 서버: 전처리, ONNX Good/Bad 추론, 카운트, 점수, 피드백 판단

## 실행

```bash
mvn spring-boot:run
```

접속:

```text
http://localhost:8080/exercise.html
```

## Render

GitHub에 이 폴더 전체를 올린 뒤 Render Web Service에서 Docker로 배포합니다.

```text
https://배포주소.onrender.com/exercise.html
```

## 중요

원본 Python 로직 기준으로 Java 이식:
- Pose_detect.py get_feature_vector
- Classifier.py predict
- Count_module.py update
- Score_manager.py calculate / generate_feedback
