package com.fitness;

import ai.onnxruntime.*;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.*;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class WorkoutService {
    private static final int SEQ_LENGTH = 30;
    private static final int FEATURES = 132;
    private static final double VIS_THRESH = 0.65;

    private final OrtEnvironment env;
    private final Map<String, OrtSession> sessions = new ConcurrentHashMap<>();
    private final Map<String, ClientState> clients = new ConcurrentHashMap<>();

    public WorkoutService() throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
    }

    public void reset(String sessionId) {
        clients.keySet().removeIf(k -> k.startsWith(sessionId + ":"));
    }

    public Map<String, Object> analyze(String sessionId, String exercise, List<Landmark> lm) {
        try {
            if (lm == null || lm.size() < 33) return Map.of("error", "no_pose");

            String key = sessionId + ":" + exercise;
            ClientState client = clients.computeIfAbsent(key, k -> new ClientState(exercise));

            Map<String, Double> angles = getExerciseAngles(lm, exercise);
            float[] feature = getFeatureVector(lm);
            String poseLabel = predict(exercise, client, feature);

            client.landmarkHistory.add(Map.of(
                    "left_shoulder_x", lm.get(11).x,
                    "right_shoulder_x", lm.get(12).x
            ));
            if (client.landmarkHistory.size() > 30) client.landmarkHistory.remove(0);

            Map<String, Object> state = client.update(angles, 0.0);
            Map<String, Object> score = calculateScore(angles, exercise, poseLabel, client.landmarkHistory, (String) state.get("state"));
            String feedback = generateFeedback(angles, exercise, score, (String) state.get("state"));

            List<List<Double>> points = new ArrayList<>();
            for (int i = 0; i < 33; i++) points.add(List.of(lm.get(i).x, lm.get(i).y));

            Map<String, Object> out = new LinkedHashMap<>();
            out.put("angles", angles);
            out.put("state", state);
            out.put("score", score);
            out.put("feedback", feedback);
            out.put("pose_label", poseLabel);
            out.put("landmarks", points);
            return out;
        } catch (Exception e) {
            e.printStackTrace();
            return Map.of("error", "server_error", "message", String.valueOf(e.getMessage()));
        }
    }

    private OrtSession session(String exercise) throws Exception {
        OrtSession cached = sessions.get(exercise);
        if (cached != null) return cached;

        File dir = Files.createTempDirectory("onnx_" + exercise + "_").toFile();
        dir.deleteOnExit();

        copyResource("static/models/" + exercise + ".onnx", new File(dir, exercise + ".onnx"));
        // external data 파일이 있으면 반드시 같은 폴더에 있어야 함
        copyResourceIfExists("static/models/" + exercise + ".onnx.data", new File(dir, exercise + ".onnx.data"));

        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        OrtSession created = env.createSession(new File(dir, exercise + ".onnx").getAbsolutePath(), opts);
        sessions.put(exercise, created);
        return created;
    }

    private void copyResource(String path, File target) throws IOException {
        ClassPathResource res = new ClassPathResource(path);
        try (InputStream in = res.getInputStream()) {
            Files.copy(in, target.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }
        target.deleteOnExit();
    }

    private void copyResourceIfExists(String path, File target) throws IOException {
        ClassPathResource res = new ClassPathResource(path);
        if (!res.exists()) return;
        try (InputStream in = res.getInputStream()) {
            Files.copy(in, target.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }
        target.deleteOnExit();
    }

    // Pose_detect.py get_feature_vector와 동일
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
        int idx = 0;
        for (Landmark p : lm) {
            out[idx++] = (float) ((p.x - cx) / scale);
            out[idx++] = (float) ((p.y - cy) / scale);
            out[idx++] = (float) ((p.z - cz) / scale);
            out[idx++] = (float) p.visibility;
        }
        return out;
    }

    // Classifier.py predict와 동일: 30프레임 미만이면 Good
    private String predict(String exercise, ClientState client, float[] feature) throws Exception {
        client.buffer.add(feature);
        if (client.buffer.size() > SEQ_LENGTH) client.buffer.remove(0);
        if (client.buffer.size() < SEQ_LENGTH) return "Good";

        float[][][] input = new float[1][SEQ_LENGTH][FEATURES];
        for (int t = 0; t < SEQ_LENGTH; t++) {
            System.arraycopy(client.buffer.get(t), 0, input[0][t], 0, FEATURES);
        }

        OrtSession s = session(exercise);
        String inputName = s.getInputNames().iterator().next();

        try (OnnxTensor tensor = OnnxTensor.createTensor(env, input);
             OrtSession.Result result = s.run(Map.of(inputName, tensor))) {
            Object value = result.get(0).getValue();

            float[] logits;
            if (value instanceof float[][]) {
                logits = ((float[][]) value)[0];
            } else if (value instanceof float[]) {
                logits = (float[]) value;
            } else {
                return "Good";
            }

            return logits[0] >= logits[1] ? "Good" : "Bad";
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
            Double body = bestAngle(lm, new int[]{11,23,27}, new int[]{12,24,28});
            if (elbow != null) angles.put("elbow", elbow);
            if (body != null) angles.put("body_line", body);
        } else if ("pullup".equals(exercise)) {
            if (visible(lm, 11,13,15)) angles.put("left_elbow", angle(lm.get(11), lm.get(13), lm.get(15)));
            if (visible(lm, 12,14,16)) angles.put("right_elbow", angle(lm.get(12), lm.get(14), lm.get(16)));
        }

        return angles;
    }

    private boolean visible(List<Landmark> lm, int... ids) {
        for (int id : ids) if (lm.get(id).visibility < VIS_THRESH) return false;
        return true;
    }

    private Double bestAngle(List<Landmark> lm, int[] left, int[] right) {
        double lv = 1.0, rv = 1.0;
        for (int id : left) lv = Math.min(lv, lm.get(id).visibility);
        for (int id : right) rv = Math.min(rv, lm.get(id).visibility);
        if (lv < VIS_THRESH && rv < VIS_THRESH) return null;
        int[] use = lv >= rv ? left : right;
        return angle(lm.get(use[0]), lm.get(use[1]), lm.get(use[2]));
    }

    private double angle(Landmark a, Landmark b, Landmark c) {
        double bax = a.x - b.x, bay = a.y - b.y;
        double bcx = c.x - b.x, bcy = c.y - b.y;
        double dot = bax * bcx + bay * bcy;
        double n1 = Math.sqrt(bax*bax + bay*bay);
        double n2 = Math.sqrt(bcx*bcx + bcy*bcy);
        double cos = dot / (n1 * n2 + 1e-6);
        cos = Math.max(-1.0, Math.min(1.0, cos));
        return Math.toDegrees(Math.acos(cos));
    }

    private Map<String, Object> calculateScore(Map<String, Double> angles, String exercise, String poseLabel,
                                               List<Map<String, Double>> history, String state) {
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
                    errors.add(Math.min(Math.abs(angles.get(e.getKey()) - e.getValue()), 90.0));
                }
            }
            accuracy = errors.isEmpty() ? 50.0 : 100.0 - (mean(errors) * (100.0 / 90.0));
        }

        double stability = 100.0;
        if (history.size() >= 10) {
            List<Double> xs = new ArrayList<>();
            for (int i = history.size() - 10; i < history.size(); i++) {
                xs.add(history.get(i).get("left_shoulder_x"));
            }
            double m = mean(xs);
            double var = 0;
            for (double x : xs) var += Math.pow(x - m, 2);
            var /= xs.size();
            stability = Math.max(0, 100 - Math.sqrt(var) * 1000);
        }

        double penalty = "Bad".equals(poseLabel) ? 30.0 : 0.0;
        double total = accuracy != null ? accuracy * 0.6 + stability * 0.4 - penalty : stability - penalty;

        Map<String, Object> out = new LinkedHashMap<>();
        out.put("total", round1(Math.max(0, Math.min(100, total))));
        out.put("accuracy", accuracy == null ? null : round1(accuracy));
        out.put("stability", round1(stability));
        out.put("pose_label", poseLabel);
        out.put("measuring", "down".equals(state));
        return out;
    }

    private String generateFeedback(Map<String, Double> angles, String exercise, Map<String, Object> score, String state) {
        if (angles.isEmpty()) return "";
        if (!"down".equals(state)) return "";

        List<Feedback> feedbacks = new ArrayList<>();

        if ("squat".equals(exercise)) {
            double knee = angles.getOrDefault("knee", 180.0);
            double trunk = angles.getOrDefault("trunk", 90.0);
            if (knee > 120) feedbacks.add(new Feedback(0, "무릎을 더 깊이 굽혀주세요"));
            else if (knee > 105) feedbacks.add(new Feedback(1, "조금만 더 내려가세요"));
            if (trunk < 50) feedbacks.add(new Feedback(0, "상체가 너무 앞으로 쏠렸어요, 등을 세워주세요"));
            else if (trunk < 60) feedbacks.add(new Feedback(1, "상체를 조금 더 세워주세요"));
        } else if ("pushup".equals(exercise)) {
            double elbow = angles.getOrDefault("elbow", 180.0);
            double body = angles.getOrDefault("body_line", 180.0);
            if (elbow > 120) feedbacks.add(new Feedback(0, "팔을 더 굽혀주세요"));
            else if (elbow > 100) feedbacks.add(new Feedback(1, "조금만 더 내려가세요"));
            if (body < 150) feedbacks.add(new Feedback(0, "엉덩이가 너무 올라갔어요, 몸을 일직선으로 유지해주세요"));
            else if (body < 160) feedbacks.add(new Feedback(1, "엉덩이를 살짝 내려주세요"));
        } else if ("pullup".equals(exercise)) {
            double le = angles.getOrDefault("left_elbow", 180.0);
            double re = angles.getOrDefault("right_elbow", 180.0);
            List<Double> vals = new ArrayList<>();
            if (le < 170) vals.add(le);
            if (re < 170) vals.add(re);
            double avg = vals.isEmpty() ? 180.0 : mean(vals);
            if (100 < avg && avg < 140) feedbacks.add(new Feedback(0, "턱이 바에 닿을 때까지 올려주세요"));
            else if (80 < avg && avg <= 100) feedbacks.add(new Feedback(1, "조금만 더 올려주세요"));
            if (Math.abs(le - re) > 20) feedbacks.add(new Feedback(1, "양팔 힘을 균등하게 써주세요"));
        }

        double stability = ((Number) score.get("stability")).doubleValue();
        if (stability < 50) feedbacks.add(new Feedback(0, "몸이 많이 흔들려요, 균형을 잡아주세요"));
        else if (stability < 70) feedbacks.add(new Feedback(2, "좌우 균형을 유지해주세요"));

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

    private double mean(List<Double> xs) {
        if (xs.isEmpty()) return 0.0;
        double s = 0;
        for (double x : xs) s += x;
        return s / xs.size();
    }

    private double round1(double x) { return Math.round(x * 10.0) / 10.0; }

    private static class Feedback {
        final int priority;
        final String message;
        Feedback(int priority, String message) { this.priority = priority; this.message = message; }
    }

    private static class ClientState {
        final String exercise;
        String state = "ready";
        int count = 0;
        final List<Float> downScores = new ArrayList<>();
        final List<float[]> buffer = new ArrayList<>();
        final List<Map<String, Double>> landmarkHistory = new ArrayList<>();

        ClientState(String exercise) { this.exercise = exercise; }

        Map<String, Object> update(Map<String, Double> angles, double qualityScore) {
            Threshold cfg = Threshold.of(exercise);
            String prev = state;

            double angle;
            if (cfg.key2 != null) {
                List<Double> vals = new ArrayList<>();
                if (angles.containsKey(cfg.key)) vals.add(angles.get(cfg.key));
                if (angles.containsKey(cfg.key2)) vals.add(angles.get(cfg.key2));
                angle = vals.isEmpty() ? (cfg.inverted ? 0 : 180) : vals.stream().mapToDouble(Double::doubleValue).average().orElse(cfg.inverted ? 0 : 180);
            } else {
                angle = angles.getOrDefault(cfg.key, cfg.inverted ? 0.0 : 180.0);
            }

            boolean countIncreased = false;

            if (!cfg.inverted) {
                if ("ready".equals(state) && angle < cfg.down) {
                    state = "down";
                    downScores.clear();
                } else if ("down".equals(state)) {
                    downScores.add((float) qualityScore);
                    if (angle > cfg.up) {
                        state = "ready";
                        count += 1;
                        downScores.clear();
                        countIncreased = true;
                    }
                }
            } else {
                if ("ready".equals(state) && angle > cfg.down) {
                    state = "down";
                    downScores.clear();
                } else if ("down".equals(state)) {
                    downScores.add((float) qualityScore);
                    if (angle < cfg.up) {
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
            out.put("state_changed", !Objects.equals(prev, state));
            out.put("changed", !Objects.equals(prev, state));
            out.put("measuring", "down".equals(state));
            out.put("countIncreased", countIncreased);
            out.put("repDisallowed", false);
            return out;
        }
    }

    private static class Threshold {
        String key, key2;
        double down, up;
        boolean inverted;
        static Threshold of(String exercise) {
            Threshold t = new Threshold();
            if ("pushup".equals(exercise)) {
                t.key = "elbow"; t.down = 85; t.up = 150;
            } else if ("pullup".equals(exercise)) {
                t.key = "left_elbow"; t.key2 = "right_elbow"; t.down = 150; t.up = 65; t.inverted = true;
            } else {
                t.key = "knee"; t.down = 95; t.up = 155;
            }
            return t;
        }
    }
}
