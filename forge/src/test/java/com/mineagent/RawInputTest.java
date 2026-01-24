package com.mineagent;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class RawInputTest {

    /**
     * Helper method to create a byte array representing a RawInput.
     *
     * <p>Binary format:
     *
     * <ul>
     *   <li>1 byte: number of keys pressed (unsigned)
     *   <li>N × 2 bytes: key codes as shorts
     *   <li>4 bytes: mouseDx as float
     *   <li>4 bytes: mouseDy as float
     *   <li>1 byte: mouseButtons
     *   <li>4 bytes: scrollDelta as float
     *   <li>2 bytes: text length (unsigned short)
     *   <li>M bytes: UTF-8 encoded text
     * </ul>
     */
    private byte[] createRawInputBytes(
            int[] keyCodes,
            float mouseDx,
            float mouseDy,
            byte mouseButtons,
            float scrollDelta,
            String text) {
        byte[] textBytes = text.getBytes(StandardCharsets.UTF_8);
        int bufferSize = 1 + (keyCodes.length * 2) + 4 + 4 + 1 + 4 + 2 + textBytes.length;
        ByteBuffer buffer = ByteBuffer.allocate(bufferSize);
        buffer.order(ByteOrder.BIG_ENDIAN); // Match Python protocol (struct.pack with '>')

        buffer.put((byte) keyCodes.length);
        for (int keyCode : keyCodes) {
            buffer.putShort((short) keyCode);
        }
        buffer.putFloat(mouseDx);
        buffer.putFloat(mouseDy);
        buffer.put(mouseButtons);
        buffer.putFloat(scrollDelta);
        buffer.putShort((short) textBytes.length);
        buffer.put(textBytes);

        return buffer.array();
    }

    @Test
    void fromBytes_withTypicalInput_parsesCorrectly() {
        int[] keyCodes = {87, 32}; // W key and Space
        float mouseDx = 10.5f;
        float mouseDy = -5.25f;
        byte mouseButtons = 0b00000001; // Left button pressed
        float scrollDelta = 1.0f;
        String text = "";

        byte[] bytes = createRawInputBytes(keyCodes, mouseDx, mouseDy, mouseButtons, scrollDelta, text);
        RawInput result = RawInput.fromBytes(bytes);

        assertArrayEquals(keyCodes, result.keyCodes());
        assertEquals(mouseDx, result.mouseDx(), 0.0001f);
        assertEquals(mouseDy, result.mouseDy(), 0.0001f);
        assertEquals(mouseButtons, result.mouseButtons());
        assertEquals(scrollDelta, result.scrollDelta(), 0.0001f);
        assertEquals(text, result.text());
    }

    @Test
    void fromBytes_withNoKeysPressed_parsesEmptyKeyArray() {
        int[] keyCodes = {};
        float mouseDx = 0.0f;
        float mouseDy = 0.0f;
        byte mouseButtons = 0;
        float scrollDelta = 0.0f;
        String text = "";

        byte[] bytes = createRawInputBytes(keyCodes, mouseDx, mouseDy, mouseButtons, scrollDelta, text);
        RawInput result = RawInput.fromBytes(bytes);

        assertArrayEquals(keyCodes, result.keyCodes());
        assertEquals(0, result.keyCodes().length);
    }

    @Test
    void fromBytes_withMultipleKeys_parsesAllKeys() {
        int[] keyCodes = {87, 65, 83, 68, 340}; // WASD + Left Shift
        float mouseDx = 0.0f;
        float mouseDy = 0.0f;
        byte mouseButtons = 0;
        float scrollDelta = 0.0f;
        String text = "";

        byte[] bytes = createRawInputBytes(keyCodes, mouseDx, mouseDy, mouseButtons, scrollDelta, text);
        RawInput result = RawInput.fromBytes(bytes);

        assertArrayEquals(keyCodes, result.keyCodes());
        assertEquals(5, result.keyCodes().length);
    }

    @Test
    void fromBytes_withNegativeMouseDeltas_parsesCorrectly() {
        int[] keyCodes = {};
        float mouseDx = -100.75f;
        float mouseDy = -200.5f;
        byte mouseButtons = 0;
        float scrollDelta = -3.0f;
        String text = "";

        byte[] bytes = createRawInputBytes(keyCodes, mouseDx, mouseDy, mouseButtons, scrollDelta, text);
        RawInput result = RawInput.fromBytes(bytes);

        assertEquals(mouseDx, result.mouseDx(), 0.0001f);
        assertEquals(mouseDy, result.mouseDy(), 0.0001f);
        assertEquals(scrollDelta, result.scrollDelta(), 0.0001f);
    }

    @Test
    void fromBytes_withAllMouseButtonsPressed_parsesCorrectly() {
        int[] keyCodes = {};
        float mouseDx = 0.0f;
        float mouseDy = 0.0f;
        byte mouseButtons = (byte) 0xFF; // All buttons pressed
        float scrollDelta = 0.0f;
        String text = "";

        byte[] bytes = createRawInputBytes(keyCodes, mouseDx, mouseDy, mouseButtons, scrollDelta, text);
        RawInput result = RawInput.fromBytes(bytes);

        assertEquals((byte) 0xFF, result.mouseButtons());
    }

    @Test
    void fromBytes_withText_parsesTextCorrectly() {
        int[] keyCodes = {};
        float mouseDx = 0.0f;
        float mouseDy = 0.0f;
        byte mouseButtons = 0;
        float scrollDelta = 0.0f;
        String text = "Hello World!";

        byte[] bytes = createRawInputBytes(keyCodes, mouseDx, mouseDy, mouseButtons, scrollDelta, text);
        RawInput result = RawInput.fromBytes(bytes);

        assertEquals(text, result.text());
    }

    @Test
    void fromBytes_withUnicodeText_parsesUtf8Correctly() {
        int[] keyCodes = {};
        float mouseDx = 0.0f;
        float mouseDy = 0.0f;
        byte mouseButtons = 0;
        float scrollDelta = 0.0f;
        String text = "日本語テスト 🎮";

        byte[] bytes = createRawInputBytes(keyCodes, mouseDx, mouseDy, mouseButtons, scrollDelta, text);
        RawInput result = RawInput.fromBytes(bytes);

        assertEquals(text, result.text());
    }

    @Test
    void fromBytes_withCompleteInput_parsesAllFieldsCorrectly() {
        int[] keyCodes = {69, 256}; // E key and Escape
        float mouseDx = 42.0f;
        float mouseDy = -17.5f;
        byte mouseButtons = 0b00000101; // Left and middle buttons
        float scrollDelta = 2.5f;
        String text = "test command";

        byte[] bytes = createRawInputBytes(keyCodes, mouseDx, mouseDy, mouseButtons, scrollDelta, text);
        RawInput result = RawInput.fromBytes(bytes);

        assertArrayEquals(keyCodes, result.keyCodes());
        assertEquals(mouseDx, result.mouseDx(), 0.0001f);
        assertEquals(mouseDy, result.mouseDy(), 0.0001f);
        assertEquals(mouseButtons, result.mouseButtons());
        assertEquals(scrollDelta, result.scrollDelta(), 0.0001f);
        assertEquals(text, result.text());
    }

    @Test
    void fromBytes_withMaxKeyCount_parsesAllKeys() {
        // Test with 255 keys (maximum for unsigned byte)
        int[] keyCodes = new int[255];
        for (int i = 0; i < 255; i++) {
            keyCodes[i] = i;
        }
        float mouseDx = 0.0f;
        float mouseDy = 0.0f;
        byte mouseButtons = 0;
        float scrollDelta = 0.0f;
        String text = "";

        byte[] bytes = createRawInputBytes(keyCodes, mouseDx, mouseDy, mouseButtons, scrollDelta, text);
        RawInput result = RawInput.fromBytes(bytes);

        assertEquals(255, result.keyCodes().length);
        for (int i = 0; i < 255; i++) {
            assertEquals(i, result.keyCodes()[i]);
        }
    }

    @Test
    void fromBytes_withNegativeKeyCode_parsesAsSignedShort() {
        // Key codes are stored as shorts, so negative values are possible
        int[] keyCodes = {-1, -100};
        float mouseDx = 0.0f;
        float mouseDy = 0.0f;
        byte mouseButtons = 0;
        float scrollDelta = 0.0f;
        String text = "";

        byte[] bytes = createRawInputBytes(keyCodes, mouseDx, mouseDy, mouseButtons, scrollDelta, text);
        RawInput result = RawInput.fromBytes(bytes);

        assertEquals(-1, result.keyCodes()[0]);
        assertEquals(-100, result.keyCodes()[1]);
    }

    @Test
    void fromBytes_withLargeFloatValues_parsesCorrectly() {
        int[] keyCodes = {};
        float mouseDx = Float.MAX_VALUE;
        float mouseDy = Float.MIN_VALUE;
        byte mouseButtons = 0;
        float scrollDelta = Float.POSITIVE_INFINITY;
        String text = "";

        byte[] bytes = createRawInputBytes(keyCodes, mouseDx, mouseDy, mouseButtons, scrollDelta, text);
        RawInput result = RawInput.fromBytes(bytes);

        assertEquals(Float.MAX_VALUE, result.mouseDx(), 0.0f);
        assertEquals(Float.MIN_VALUE, result.mouseDy(), 0.0f);
        assertEquals(Float.POSITIVE_INFINITY, result.scrollDelta(), 0.0f);
    }

    @Test
    void recordEquality_withSameValues_areEqual() {
        int[] keyCodes = {65};
        RawInput input1 = new RawInput(keyCodes, 1.0f, 2.0f, (byte) 1, 0.5f, "test");
        RawInput input2 = new RawInput(keyCodes, 1.0f, 2.0f, (byte) 1, 0.5f, "test");

        // Records use reference equality for arrays, so these won't be equal
        // unless they share the same array reference
        assertEquals(input1, input2);
    }

    @Test
    void recordEquality_withDifferentValues_areNotEqual() {
        RawInput input1 = new RawInput(new int[] {65}, 1.0f, 2.0f, (byte) 1, 0.5f, "test");
        RawInput input2 = new RawInput(new int[] {66}, 1.0f, 2.0f, (byte) 1, 0.5f, "test");

        assertNotEquals(input1, input2);
    }
}
