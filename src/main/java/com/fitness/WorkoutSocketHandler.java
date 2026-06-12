package com.fitness;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.*;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.net.URI;
import java.util.*;

@Component
public class WorkoutSocketHandler extends TextWebSocketHandler {
    private final ObjectMapper mapper = new ObjectMapper();
    private final WorkoutService workoutService;

    public WorkoutSocketHandler(WorkoutService workoutService) {
        this.workoutService = workoutService;
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
        workoutService.reset(session.getId());
    }

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
        String exercise = exerciseFromUri(session.getUri());
        Map<String, Object> body = mapper.readValue(message.getPayload(), new TypeReference<Map<String, Object>>() {});
        Object raw = body.get("landmarks");

        if (!(raw instanceof List<?>)) {
            session.sendMessage(new TextMessage(mapper.writeValueAsString(Map.of("error", "no_landmarks"))));
            return;
        }

        List<?> rawList = (List<?>) raw;
        List<Landmark> landmarks = new ArrayList<>();

        for (Object item : rawList) {
            if (item instanceof Map<?, ?>) {
                Map<?, ?> m = (Map<?, ?>) item;
                double x = num(m.get("x"), 0);
                double y = num(m.get("y"), 0);
                double z = num(m.get("z"), 0);
                double visibility = num(m.get("visibility"), num(m.get("visibilityScore"), 1));
                landmarks.add(new Landmark(x, y, z, visibility));
            }
        }

        Map<String, Object> result = workoutService.analyze(session.getId(), exercise, landmarks);
        session.sendMessage(new TextMessage(mapper.writeValueAsString(result)));
    }

    private String exerciseFromUri(URI uri) {
        if (uri == null) return "squat";
        String[] parts = uri.getPath().split("/");
        String exercise = parts.length == 0 ? "squat" : parts[parts.length - 1];
        if (!Set.of("squat", "pushup", "pullup").contains(exercise)) return "squat";
        return exercise;
    }

    private double num(Object value, double fallback) {
        return value instanceof Number ? ((Number) value).doubleValue() : fallback;
    }
}
