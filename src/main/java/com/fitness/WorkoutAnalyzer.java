package com.fitness;

import ai.onnxruntime.*;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class WorkoutAnalyzer {
    private static final int SEQ_LENGTH = 30;
    private static final int FEATURES = 132;
    private static final double VIS_THRESH = 0.65;

    private final OrtEnvironment env;
    private final Map<String, OrtSession> sessions = new ConcurrentHashMap<>();
    private final Map<String, ClientState> states = new ConcurrentHashMap<>();

    public WorkoutAnalyzer() throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
    }

    public Map<String, Object> analyze(String clientId, String exercise, List<Landmark> lm) throws Exception {
        if (lm == null || lm.size() < 33) {
            return Map.of("error", "no_pose");
        }

        ClientState state = states.computeIfAbsent(clientId + ":" + exercise, k -> new ClientState(exercise));

        Map<String, Double> angles = getExerciseAngles(lm, exercise);
        float[] feature = getFeatureVector(lm);
        String poseLabel = predict(exercise, state, feature);

        state.landmarkHistory.add(Map.of(
                "left_shoulder_x", lm.get(11).x,
                "right_shoulder_x", lm.get(12).x
        ));
        if (state.landmarkHistory.size() > 30) state.landmarkHistory.remove(0);

        Map<String, Object> stateData = state.update(angles, 0.0);
        String currentState = (String) stateData.get("state");

        Map<String, Object> score = calculateScore(angles, exercise, poseLabel, state.landmarkHistory, currentState);
        String feedback = generateFeedback(angles, exercise, score, currentState);

        List<List<Double>> landmarks2d = new ArrayList<>();
        for (int i = 0; i < 33; i++) {
            landmarks2d.add(List.of(lm.get(i).x, lm.get(i).y));
        }

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("angles", angles);
        result.put("state", stateData);
        result.put("score", score);
        result.put("feedback", feedback);
        result.put("pose_label", poseLabel);
        result.put("landmarks", landmarks2d);
        return result;
    }

    public void resetClient(String clientId) {
        states.keySet().removeIf(k -> k.startsWith(clientId + ":"));
    }

    private OrtSession getSession(String exercise) throws Exception {
        if (sessions.containsKey(exercise)) return sessions.get(exercise);

        // ONNX가 외부 데이터 파일(.onnx.data)을 참조할 수 있으므로
        // .onnx와 .onnx.data를 같은 임시 폴더에 함께 복사한 뒤 로드합니다.
        File tempDir = Files.createTempDirectory("fitness_onnx_" + exercise + "_").toFile();
        tempDir.deleteOnExit();

        String onnxName = exercise + ".onnx";
        String dataName = exercise + ".onnx.data";

        ClassPathResource onnxRes = new ClassPathResource("static/models/" + onnxName);
        File onnxFile = new File(tempDir, onnxName);
        try (InputStream in = onnxRes.getInputStream()) {
            Files.copy(in, onnxFile.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }
        onnxFile.deleteOnExit();

        ClassPathResource dataRes = new ClassPathResource("static/models/" + dataName);
        if (dataRes.exists()) {
            File dataFile = new File(tempDir, dataName);
            try (InputStream in = dataRes.getInputStream()) {
                Files.copy(in, dataFile.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
            dataFile.deleteOnExit();
        }

        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        OrtSession session = env.createSession(onnxFile.getAbsolutePath(), opts);
        sessions.put(exercise, session);
        return session;
    }

    // Pose_detect.py get_feature_vector와 동일:
    // 33개 landmark × x,y,z,visibility = 132
    // 골반 중심 기준 이동 + 어깨-골반 거리 스케일 정규화
    private float[] getFeatureVector(List<Landmark> lm) {
        double cx = (lm.get(23).x + lm.get(24).x) / 2.0;
        double cy = (lm.get(23).y + lm.get(24).y) / 2.0;
        double cz = (lm.get(23).z + lm.get(24).z) / 2.0;

        double sx = (lm.get(11).x + lm.get(12).x) / 2.0;
        double sy = (lm.get(11).y + lm.get(12).y) / 2.0;
        double sz = (lm.get(11).z + lm.get(12).z) / 2.0;

        double dist = Math.sqrt(Math.pow(cx - sx, 2) + Math.pow(cy - sy, 2) + Math.pow(cz - sz, 2));
        double scale = dist > 0.01 ? dist : 1.0;

        float[] out = new float[FEATURES];
        int k = 0;
        for (Landmark p : lm) {
            out[k++] = (float) ((p.x - cx) / scale);
            out[k++] = (float) ((p.y - cy) / scale);
            out[k++] = (float) ((p.z - cz) / scale);
            out[k++] = (float) p.visibility;
        }
        return out;
    }

    // Classifier.py와 동일: buffer 30프레임, 30개 미만이면 Good
    private String predict(String exercise, ClientState state, float[] feature) throws Exception {
        state.featureBuffer.add(feature);
        if (state.featureBuffer.size() > SEQ_LENGTH) state.featureBuffer.remove(0);

        if (state.featureBuffer.size() < SEQ_LENGTH) return "Good";

        float[][][] input = new float[1][SEQ_LENGTH][FEATURES];
        for (int i = 0; i < SEQ_LENGTH; i++) {
            System.arraycopy(state.featureBuffer.get(i), 0, input[0][i], 0, FEATURES);
        }

        OrtSession session = getSession(exercise);
        String inputName = session.getInputNames().iterator().next();

        try (OnnxTensor tensor = OnnxTensor.createTensor(env, input);
             OrtSession.Result result = session.run(Map.of(inputName, tensor))) {
            Object value = result.get(0).getValue();

            float[] logits;
            if (value instanceof float[][]) {
                float[][] arr = (float[][]) value;
                logits = arr[0];
            } else if (value instanceof float[]) {
                float[] arr = (float[]) value;
                logits = arr;
            } else {
                return "Good";
            }

            int predicted = logits[0] >= logits[1] ? 0 : 1;
            return predicted == 0 ? "Good" : "Bad";
        }
    }

    private Map<String, Double> getExerciseAngles(List<Landmark> lm, String exercise) {
        Map<String, Double> angles = new LinkedHashMap<>();
        if ("squat".equals(exercise)) {
            Double knee = bestAngle(lm, new int[]{23,25,27}, new int[]{24,26,28});
            Double trunk = bestAngle(lm, new int[]{11,23,25}, new int[]{12,24,26});
            if (knee != null) angles.put("knee", knee);
            if (trunk != null) angles.put("trunk", trunk);
        } else if ("pushup".equals(exercise)) {
            Double elbow = bestAngle(lm, new int[]{11,13,15}, new int[]{12,14,16});
            Double bodyLine = bestAngle(lm, new int[]{11,23,27}, new int[]{12,24,28});
            if (elbow != null) angles.put("elbow", elbow);
            if (bodyLine != null) angles.put("body_line", bodyLine);
        } else if ("pullup".equals(exercise)) {
            if (visible(lm, 11,13,15)) angles.put("left_elbow", calculateAngle(lm.get(11), lm.get(13), lm.get(15)));
            if (visible(lm, 12,14,16)) angles.put("right_elbow", calculateAngle(lm.get(12), lm.get(14), lm.get(16)));
        }
        return angles;
    }

    private boolean visible(List<Landmark> lm, int... idxs) {
        for (int idx : idxs) {
            if (lm.get(idx).visibility < VIS_THRESH) return false;
        }
        return true;
    }

    private Double bestAngle(List<Landmark> lm, int[] left, int[] right) {
        double lv = 1.0;
        for (int idx : left) lv = Math.min(lv, lm.get(idx).visibility);
        double rv = 1.0;
        for (int idx : right) rv = Math.min(rv, lm.get(idx).visibility);
        if (lv < VIS_THRESH && rv < VIS_THRESH) return null;

        int[] chosen = lv >= rv ? left : right;
        return calculateAngle(lm.get(chosen[0]), lm.get(chosen[1]), lm.get(chosen[2]));
    }

    private double calculateAngle(Landmark a, Landmark b, Landmark c) {
        double[] ba = {a.x - b.x, a.y - b.y};
        double[] bc = {c.x - b.x, c.y - b.y};
        double dot = ba[0] * bc[0] + ba[1] * bc[1];
        double normBa = Math.sqrt(ba[0] * ba[0] + ba[1] * ba[1]);
        double normBc = Math.sqrt(bc[0] * bc[0] + bc[1] * bc[1]);
        double cos = dot / (normBa * normBc + 1e-6);
        cos = Math.max(-1.0, Math.min(1.0, cos));
        return Math.toDegrees(Math.acos(cos));
    }

    private Map<String, Object> calculateScore(Map<String, Double> angles, String exercise,
                                               String poseLabel, List<Map<String, Double>> history, String state) {
        Map<String, Double> ideal = new HashMap<>();
        if ("squat".equals(exercise)) {
            ideal.put("knee", 90.0); ideal.put("trunk", 75.0);
        } else if ("pushup".equals(exercise)) {
            ideal.put("elbow", 90.0); ideal.put("body_line", 180.0);
        } else if ("pullup".equals(exercise)) {
            ideal.put("left_elbow", 60.0); ideal.put("right_elbow", 60.0);
        }

        Double accuracy = null;
        if ("down".equals(state) && !angles.isEmpty()) {
            List<Double> errors = new ArrayList<>();
            for (Map.Entry<String, Double> e : ideal.entrySet()) {
                if (angles.containsKey(e.getKey())) {
                    double err = Math.abs(angles.get(e.getKey()) - e.getValue());
                    errors.add(Math.min(err, 90.0));
                }
            }
            if (!errors.isEmpty()) {
                double mean = errors.stream().mapToDouble(Double::doubleValue).average().orElse(50.0);
                accuracy = 100.0 - (mean * (100.0 / 90.0));
            } else {
                accuracy = 50.0;
            }
        }

        double stability = 100.0;
        if (history.size() >= 10) {
            List<Double> xs = new ArrayList<>();
            for (int i = Math.max(0, history.size() - 10); i < history.size(); i++) {
                xs.add(history.get(i).get("left_shoulder_x"));
            }
            double mean = xs.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            double variance = xs.stream().mapToDouble(x -> Math.pow(x - mean, 2)).average().orElse(0.0);
            double std = Math.sqrt(variance);
            stability = Math.max(0, 100 - std * 1000);
        }

        double penalty = "Bad".equals(poseLabel) ? 30.0 : 0.0;
        double total = (accuracy != null) ? accuracy * 0.6 + stability * 0.4 - penalty : stability - penalty;
        total = Math.max(0, Math.min(100, total));

        Map<String, Object> out = new LinkedHashMap<>();
        out.put("total", round1(total));
        out.put("accuracy", accuracy == null ? null : round1(accuracy));
        out.put("stability", round1(stability));
        out.put("pose_label", poseLabel);
        out.put("measuring", "down".equals(state));
        return out;
    }

    private String generateFeedback(Map<String, Double> angles, String exercise, Map<String, Object> score, String state) {
        if (angles.isEmpty()) return "";
        if (!"down".equals(state)) return "";

        List<FeedbackItem> feedbacks = new ArrayList<>();

        if ("squat".equals(exercise)) {
            double knee = angles.getOrDefault("knee", 180.0);
            double trunk = angles.getOrDefault("trunk", 90.0);

            if (knee > 120) feedbacks.add(new FeedbackItem(0, "무릎을 더 깊이 굽혀주세요"));
            else if (knee > 105) feedbacks.add(new FeedbackItem(1, "조금만 더 내려가세요"));

            if (trunk < 50) feedbacks.add(new FeedbackItem(0, "상체가 너무 앞으로 쏠렸어요, 등을 세워주세요"));
            else if (trunk < 60) feedbacks.add(new FeedbackItem(1, "상체를 조금 더 세워주세요"));

        } else if ("pushup".equals(exercise)) {
            double elbow = angles.getOrDefault("elbow", 180.0);
            double bodyLine = angles.getOrDefault("body_line", 180.0);

            if (elbow > 120) feedbacks.add(new FeedbackItem(0, "팔을 더 굽혀주세요"));
            else if (elbow > 100) feedbacks.add(new FeedbackItem(1, "조금만 더 내려가세요"));

            if (bodyLine < 150) feedbacks.add(new FeedbackItem(0, "엉덩이가 너무 올라갔어요, 몸을 일직선으로 유지해주세요"));
            else if (bodyLine < 160) feedbacks.add(new FeedbackItem(1, "엉덩이를 살짝 내려주세요"));

        } else if ("pullup".equals(exercise)) {
            double le = angles.getOrDefault("left_elbow", 180.0);
            double re = angles.getOrDefault("right_elbow", 180.0);
            List<Double> vals = new ArrayList<>();
            if (le < 170) vals.add(le);
            if (re < 170) vals.add(re);
            double avg = vals.isEmpty() ? 180.0 : vals.stream().mapToDouble(Double::doubleValue).average().orElse(180.0);

            if (100 < avg && avg < 140) feedbacks.add(new FeedbackItem(0, "턱이 바에 닿을 때까지 올려주세요"));
            else if (80 < avg && avg <= 100) feedbacks.add(new FeedbackItem(1, "조금만 더 올려주세요"));

            if (Math.abs(le - re) > 20) feedbacks.add(new FeedbackItem(1, "양팔 힘을 균등하게 써주세요"));
        }

        double stability = ((Number) score.get("stability")).doubleValue();
        if (stability < 50) feedbacks.add(new FeedbackItem(0, "몸이 많이 흔들려요, 균형을 잡아주세요"));
        else if (stability < 70) feedbacks.add(new FeedbackItem(2, "좌우 균형을 유지해주세요"));

        if (feedbacks.isEmpty()) {
            Object accObj = score.get("accuracy");
            double acc = accObj instanceof Number ? ((Number) accObj).doubleValue() : 0.0;
            if (acc >= 90) return "자세가 아주 좋아요!";
            if (acc >= 75) return "좋아요, 그대로 유지해주세요";
            return "";
        }

        feedbacks.sort(Comparator.comparingInt(f -> f.priority));
        return feedbacks.get(0).message;
    }

    private double round1(double x) {
        return Math.round(x * 10.0) / 10.0;
    }

    private static class FeedbackItem {
        int priority;
        String message;
        FeedbackItem(int priority, String message) {
            this.priority = priority;
            this.message = message;
        }
    }

    private static class ClientState {
        String exercise;
        String state = "ready";
        int count = 0;
        List<Float> downScores = new ArrayList<>();
        List<float[]> featureBuffer = new ArrayList<>();
        List<Map<String, Double>> landmarkHistory = new ArrayList<>();

        ClientState(String exercise) {
            this.exercise = exercise;
        }

        Map<String, Object> update(Map<String, Double> angles, double qualityScore) {
            Threshold cfg = Threshold.forExercise(exercise);
            String prevState = state;

            double angle;
            if (cfg.key2 != null) {
                List<Double> vals = new ArrayList<>();
                if (angles.containsKey(cfg.key)) vals.add(angles.get(cfg.key));
                if (angles.containsKey(cfg.key2)) vals.add(angles.get(cfg.key2));
                angle = vals.isEmpty() ? (cfg.inverted ? 0.0 : 180.0)
                        : vals.stream().mapToDouble(Double::doubleValue).average().orElse(cfg.inverted ? 0.0 : 180.0);
            } else {
                angle = angles.getOrDefault(cfg.key, cfg.inverted ? 0.0 : 180.0);
            }

            boolean countIncreased = false;

            if (!cfg.inverted) {
                if ("ready".equals(state) && angle < cfg.downAngle) {
                    state = "down";
                    downScores.clear();
                } else if ("down".equals(state)) {
                    downScores.add((float) qualityScore);
                    if (angle > cfg.upAngle) {
                        state = "ready";
                        count += 1;
                        downScores.clear();
                        countIncreased = true;
                    }
                }
            } else {
                if ("ready".equals(state) && angle > cfg.downAngle) {
                    state = "down";
                    downScores.clear();
                } else if ("down".equals(state)) {
                    downScores.add((float) qualityScore);
                    if (angle < cfg.upAngle) {
                        state = "ready";
                        count += 1;
                        downScores.clear();
                        countIncreased = true;
                    }
                }
            }

            Map<String, Object> out = new LinkedHashMap<>();
            out.put("state", state);
            out.put("count", count);
            out.put("state_changed", !Objects.equals(prevState, state));
            out.put("changed", !Objects.equals(prevState, state));
            out.put("measuring", "down".equals(state));
            out.put("countIncreased", countIncreased);
            out.put("repDisallowed", false);
            return out;
        }
    }

    private static class Threshold {
        String key;
        String key2;
        double downAngle;
        double upAngle;
        boolean inverted;

        static Threshold forExercise(String exercise) {
            Threshold t = new Threshold();
            if ("squat".equals(exercise)) {
                t.key = "knee"; t.downAngle = 95; t.upAngle = 155; t.inverted = false;
            } else if ("pushup".equals(exercise)) {
                t.key = "elbow"; t.downAngle = 85; t.upAngle = 150; t.inverted = false;
            } else if ("pullup".equals(exercise)) {
                t.key = "left_elbow"; t.key2 = "right_elbow"; t.downAngle = 150; t.upAngle = 65; t.inverted = true;
            } else {
                t.key = "knee"; t.downAngle = 95; t.upAngle = 155; t.inverted = false;
            }
            return t;
        }
    }
}
