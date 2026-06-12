package com.fitness;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {
    private final WorkoutWebSocketHandler workoutWebSocketHandler;

    public WebSocketConfig(WorkoutWebSocketHandler workoutWebSocketHandler) {
        this.workoutWebSocketHandler = workoutWebSocketHandler;
    }

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(workoutWebSocketHandler, "/ws/{exercise}")
                .setAllowedOrigins("*");
    }
}
