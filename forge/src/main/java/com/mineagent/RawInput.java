package com.mineagent;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

public record RawInput(
    int[] keyCodes,
    float mouseDx,
    float mouseDy,
    byte mouseButtons,
    float scrollDelta,
    String text
) {
    public static RawInput fromBytes(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        buffer.order(ByteOrder.BIG_ENDIAN); // Match Python protocol (struct.pack with '>')
        
        // Keys
        int numKeysPressed = buffer.get() & 0xFF;
        int[] keyCodes = new int[numKeysPressed];
        for (int i = 0; i < numKeysPressed; i++) {
            keyCodes[i] = buffer.getShort();
        }

        // Mouse
        float mouseDx = buffer.getFloat();
        float mouseDy = buffer.getFloat();
        byte mouseButtons = buffer.get();
        float scrollDelta = buffer.getFloat();

        // Text (for typing)
        int textLength = buffer.getShort() & 0xFFFF;
        byte[] textBytes = new byte[textLength];
        buffer.get(textBytes);
        String text = new String(textBytes, StandardCharsets.UTF_8);

        return new RawInput(
            keyCodes,
            mouseDx,
            mouseDy,
            mouseButtons,
            scrollDelta,
            text
        );
    }
};
