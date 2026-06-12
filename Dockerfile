FROM maven:3.9.9-eclipse-temurin-17 AS build
WORKDIR /app
COPY pom.xml .
COPY src ./src
RUN mvn -DskipTests package

FROM eclipse-temurin:17-jre
WORKDIR /app
COPY --from=build /app/target/fitness-render-java-backend-1.0.0.jar app.jar
EXPOSE 8080
CMD ["java", "-jar", "app.jar"]
