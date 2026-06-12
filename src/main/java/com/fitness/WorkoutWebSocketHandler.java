package com.fitness;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.*;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.net.URI;
import java.util.*;

@Component
public class WorkoutWebSocketHandler extends TextWebSocketHandler {
    private final ObjectMapper mapper = new ObjectMapper();
    private final WorkoutAnalyzer analyzer;

    public WorkoutWebSocketHandler(WorkoutAnalyzer analyzer) {
        this.analyzer = analyzer;
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
        analyzer.resetClient(session.getId());
    }

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
        String exercise = getExercise(session.getUri());

        Map<String, Object> payload = mapper.readValue(message.getPayload(), new TypeReference<Map<String, Object>>() {});
        Object rawLandmarks = payload.get("landmarks");

        if (!(rawLandmarks instanceof List<?> list)) {
            session.sendMessage(new TextMessage(mapper.writeValueAsString(Map.of("error", "no_landmarks"))));
            return;
        }

        List<Landmark> landmarks = new ArrayList<>();
        for (Object item : list) {
            if (item instanceof Map<?, ?> m) {
                double x = number(m.get("x"), 0);
                double y = number(m.get("y"), 0);
                double z = number(m.get("z"), 0);
                double v = number(m.get("visibility"), number(m.get("visibilityScore"), 1));
                landmarks.add(new Landmark(x, y, z, v));
            }
        }

        Map<String, Object> result = analyzer.analyze(session.getId(), exercise, landmarks);
        session.sendMessage(new TextMessage(mapper.writeValueAsString(result)));
    }

    private String getExercise(URI uri) {
        if (uri == null) return "squat";
        String path = uri.getPath();
        String[] parts = path.split("/");
        String ex = parts.length > 0 ? parts[parts.length - 1] : "squat";
        if (!Set.of("squat", "pushup", "pullup").contains(ex)) return "squat";
        return ex;
    }

    private double number(Object o, double fallback) {
        if (o instanceof Number n) return n.doubleValue();
        return fallback;
    }
}
