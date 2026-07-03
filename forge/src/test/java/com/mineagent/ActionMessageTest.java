package com.mineagent;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

/**
 * Tests for the v2 {@link ActionMessage} wire format. Mirrors the Python protocol tests: header
 * flags, the worked byte examples from the IPC spec, round-trip parsing, and validation errors.
 */
class ActionMessageTest {

  // GLFW key codes used in the tests (must match the Python constants).
  private static final int KEY_W = 87;
  private static final int KEY_S = 83;
  private static final int KEY_A = 65;
  private static final int KEY_SPACE = 32;

  /** Build a raw v2 byte stream for an ACTION message. */
  private byte[] buildActionBytes(
      int[] keyPress,
      int[] keyRelease,
      boolean hasMouse,
      float mouseDx,
      float mouseDy,
      boolean hasButtons,
      int buttonPress,
      int buttonRelease,
      boolean hasScroll,
      float scroll) {
    int size = 1; // flags
    int flags = ActionMessage.MSG_TYPE_ACTION;
    if (hasMouse) {
      flags |= ActionMessage.FLAG_HAS_MOUSE;
      size += 8;
    }
    if (hasButtons) {
      flags |= ActionMessage.FLAG_HAS_BUTTONS;
      size += 1;
    }
    if (hasScroll) {
      flags |= ActionMessage.FLAG_HAS_SCROLL;
      size += 4;
    }
    if (keyPress.length > 0 || keyRelease.length > 0) {
      flags |= ActionMessage.FLAG_HAS_KEYS;
      size += 2 + (keyPress.length + keyRelease.length) * 2;
    }

    ByteBuffer buffer = ByteBuffer.allocate(size);
    buffer.order(ByteOrder.BIG_ENDIAN);
    buffer.put((byte) flags);
    if ((flags & ActionMessage.FLAG_HAS_KEYS) != 0) {
      buffer.put((byte) keyPress.length);
      buffer.put((byte) keyRelease.length);
      for (int c : keyPress) {
        buffer.putShort((short) c);
      }
      for (int c : keyRelease) {
        buffer.putShort((short) c);
      }
    }
    if (hasMouse) {
      buffer.putFloat(mouseDx);
      buffer.putFloat(mouseDy);
    }
    if (hasButtons) {
      buffer.put((byte) ((buttonPress & 0x7) | ((buttonRelease & 0x7) << 3)));
    }
    if (hasScroll) {
      buffer.putFloat(scroll);
    }
    return buffer.array();
  }

  private byte[] buildTextBytes(String text) {
    byte[] textBytes = text.getBytes(StandardCharsets.UTF_8);
    ByteBuffer buffer =
        ByteBuffer.allocate(1 + 2 + textBytes.length).order(ByteOrder.BIG_ENDIAN);
    buffer.put((byte) ActionMessage.MSG_TYPE_TEXT);
    buffer.putShort((short) textBytes.length);
    buffer.put(textBytes);
    return buffer.array();
  }

  // --- header / message types ---

  @Test
  void pureHold_isOneByte() {
    ActionMessage msg = ActionMessage.fromBytes(new byte[] {0x00});
    assertEquals(ActionMessage.MSG_TYPE_ACTION, msg.msgType());
    assertEquals(0, msg.keyPress().length);
    assertEquals(0, msg.keyRelease().length);
    assertFalse(msg.hasMouse());
    assertFalse(msg.hasButtons());
    assertFalse(msg.hasScroll());
  }

  @Test
  void reset_isOneByte() {
    ActionMessage msg = ActionMessage.fromBytes(new byte[] {(byte) ActionMessage.MSG_TYPE_RESET});
    assertEquals(ActionMessage.MSG_TYPE_RESET, msg.msgType());
  }

  @Test
  void ping_isOneByte() {
    ActionMessage msg = ActionMessage.fromBytes(new byte[] {(byte) ActionMessage.MSG_TYPE_PING});
    assertEquals(ActionMessage.MSG_TYPE_PING, msg.msgType());
  }

  // --- worked examples ---

  @Test
  void lookOnly_isNineBytes() {
    byte[] bytes =
        buildActionBytes(
            new int[0], new int[0], true, 1.5f, -2.25f, false, 0, 0, false, 0f);
    assertEquals(9, bytes.length);
    ActionMessage msg = ActionMessage.fromBytes(bytes);
    assertTrue(msg.hasMouse());
    assertEquals(1.5f, msg.mouseDx(), 0.0001f);
    assertEquals(-2.25f, msg.mouseDy(), 0.0001f);
  }

  @Test
  void pressW_isFiveBytes() {
    byte[] bytes =
        buildActionBytes(
            new int[] {KEY_W}, new int[0], false, 0f, 0f, false, 0, 0, false, 0f);
    assertEquals(5, bytes.length);
    ActionMessage msg = ActionMessage.fromBytes(bytes);
    assertArrayEquals(new int[] {KEY_W}, msg.keyPress());
    assertEquals(0, msg.keyRelease().length);
  }

  @Test
  void releaseLeftAndScroll() {
    byte[] bytes =
        buildActionBytes(
            new int[0], new int[0], false, 0f, 0f, true, 0, 0b001, true, -1.0f);
    assertEquals(6, bytes.length);
    ActionMessage msg = ActionMessage.fromBytes(bytes);
    assertTrue(msg.hasButtons());
    assertEquals(0, msg.buttonPress());
    assertEquals(0b001, msg.buttonRelease());
    assertTrue(msg.hasScroll());
    assertEquals(-1.0f, msg.scroll(), 0.0001f);
  }

  @Test
  void buttonBytePacksPressAndReleaseNibbles() {
    byte[] bytes =
        buildActionBytes(
            new int[0], new int[0], false, 0f, 0f, true, 0b001, 0b100, false, 0f);
    ActionMessage msg = ActionMessage.fromBytes(bytes);
    assertEquals(0b001, msg.buttonPress());
    assertEquals(0b100, msg.buttonRelease());
  }

  @Test
  void keysPressAndReleaseLists() {
    byte[] bytes =
        buildActionBytes(
            new int[] {KEY_W, KEY_SPACE}, new int[] {KEY_A}, false, 0f, 0f, false, 0, 0,
            false, 0f);
    ActionMessage msg = ActionMessage.fromBytes(bytes);
    assertArrayEquals(new int[] {KEY_W, KEY_SPACE}, msg.keyPress());
    assertArrayEquals(new int[] {KEY_A}, msg.keyRelease());
  }

  // --- round trip ---

  @Test
  void roundTrip_fullAction() {
    byte[] bytes =
        buildActionBytes(
            new int[] {KEY_W, KEY_S}, new int[] {KEY_A}, true, 12.5f, -7.25f, true,
            0b010, 0b001, true, 2.5f);
    ActionMessage msg = ActionMessage.fromBytes(bytes);
    assertArrayEquals(new int[] {KEY_W, KEY_S}, msg.keyPress());
    assertArrayEquals(new int[] {KEY_A}, msg.keyRelease());
    assertTrue(msg.hasMouse());
    assertEquals(12.5f, msg.mouseDx(), 0.0001f);
    assertEquals(-7.25f, msg.mouseDy(), 0.0001f);
    assertEquals(0b010, msg.buttonPress());
    assertEquals(0b001, msg.buttonRelease());
    assertEquals(2.5f, msg.scroll(), 0.0001f);
  }

  @Test
  void roundTrip_textUnicode() {
    String text = "héllo 🎮";
    ActionMessage msg = ActionMessage.fromBytes(buildTextBytes(text));
    assertEquals(ActionMessage.MSG_TYPE_TEXT, msg.msgType());
    assertEquals(text, msg.text());
  }

  // --- validation ---

  @Test
  void fromBytes_rejectsReservedBits() {
    assertThrows(
        IllegalArgumentException.class,
        () -> ActionMessage.fromBytes(new byte[] {(byte) 0xC0}));
  }

  @Test
  void fromBytes_rejectsTrailingBytes() {
    byte[] reset = new byte[] {(byte) ActionMessage.MSG_TYPE_RESET, 0x00};
    assertThrows(IllegalArgumentException.class, () -> ActionMessage.fromBytes(reset));
  }

  @Test
  void staticFactories_buildCorrectTypes() {
    assertEquals(ActionMessage.MSG_TYPE_RESET, ActionMessage.reset().msgType());
    assertEquals(ActionMessage.MSG_TYPE_PING, ActionMessage.ping().msgType());
  }
}
