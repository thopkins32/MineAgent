package com.mvi.mvimod;

public record Observation(
    int reward,
    ActionState actionState,
    byte[] frame
) {
    public byte[] serialize() {
        // For now, we'll keep the existing protocol that sends reward + frame
        // The action state will be serialized separately later
        // TODO: Implement full serialization including action state
        return frame;
    }
}
