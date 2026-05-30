package spring.project.thesis.logicbackend.config;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.persistence.AttributeConverter;
import jakarta.persistence.Converter;
import spring.project.thesis.logicbackend.dto.SampleGradCam;

import java.util.List;

@Converter
public class SampleGradCamListConverter implements AttributeConverter<List<SampleGradCam>, String> {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Override
    public String convertToDatabaseColumn(List<SampleGradCam> attribute) {
        if (attribute == null) return null;
        try {
            return MAPPER.writeValueAsString(attribute);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Failed to serialize SampleGradCam list", e);
        }
    }

    @Override
    public List<SampleGradCam> convertToEntityAttribute(String dbData) {
        if (dbData == null || dbData.isBlank()) return null;
        try {
            return MAPPER.readValue(dbData, new TypeReference<List<SampleGradCam>>() {});
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Failed to deserialize SampleGradCam list", e);
        }
    }
}
