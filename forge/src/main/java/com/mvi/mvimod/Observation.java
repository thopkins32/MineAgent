package com.mvi.mvimod;

public record Observation(
    double reward,
    ActionState actionState,
    byte[] frame
) {
    public byte[] serialize() {
        return frame;
    }
}
