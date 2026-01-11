package com.mineagent;

public record Observation(
    double reward,
    byte[] frame
) {
    public byte[] serialize() {
        return frame;
    }
}
