package com.mineagent;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * v2 event-based action message exchanged between the Python agent and the Forge mod.
 *
 * <p>Wire format (big-endian):
 *
 * <ul>
 *   <li>1 byte flags:
 *       bits 0-1 = message type (ACTION / RESET / TEXT / PING), bit 2 = has key events,
 *       bit 3 = has mouse move, bit 4 = has button events, bit 5 = has scroll,
 *       bits 6-7 reserved (must be 0).
 *   <li>key events (if bit 2): u8 numPress, u8 numRelease, numPress x i16 key codes to PRESS,
 *       numRelease x i16 key codes to RELEASE. Keys in neither list are HOLD.
 *   <li>mouse move (if bit 3): f32 dx, f32 dy.
 *   <li>button events (if bit 4): 1 byte, low 3 bits = buttons to PRESS, next 3 bits = buttons
 *       to RELEASE. Buttons in neither nibble are HOLD. Bit 0 = left, 1 = right, 2 = middle.
 *   <li>scroll (if bit 5): f32 delta.
 *   <li>TEXT body: u16 UTF-8 length + bytes.
 *   <li>RESET / PING: no body.
 * </ul>
 *
 * <p>Java maintains the held key/button state across messages; PRESS adds, RELEASE removes,
 * and HOLD (unlisted) leaves state unchanged.
 */
public record ActionMessage(
    int msgType,
    int[] keyPress,
    int[] keyRelease,
    boolean hasMouse,
    float mouseDx,
    float mouseDy,
    boolean hasButtons,
    int buttonPress,
    int buttonRelease,
    boolean hasScroll,
    float scroll,
    String text) {

  public static final int MSG_TYPE_ACTION = 0;
  public static final int MSG_TYPE_RESET = 1;
  public static final int MSG_TYPE_TEXT = 2;
  public static final int MSG_TYPE_PING = 3;

  public static final int FLAG_HAS_KEYS = 1 << 2;
  public static final int FLAG_HAS_MOUSE = 1 << 3;
  public static final int FLAG_HAS_BUTTONS = 1 << 4;
  public static final int FLAG_HAS_SCROLL = 1 << 5;
  public static final int FLAG_MASK_RESERVED = 0xC0;

  public static ActionMessage reset() {
    return new ActionMessage(
        MSG_TYPE_RESET, new int[0], new int[0], false, 0f, 0f, false, 0, 0, false, 0f, "");
  }

  public static ActionMessage ping() {
    return new ActionMessage(
        MSG_TYPE_PING, new int[0], new int[0], false, 0f, 0f, false, 0, 0, false, 0f, "");
  }

  /** Parse a complete v2 message from a byte array. Used by tests and replay. */
  public static ActionMessage fromBytes(byte[] bytes) {
    ByteBuffer buffer = ByteBuffer.wrap(bytes);
    buffer.order(ByteOrder.BIG_ENDIAN);
    return read(buffer);
  }

  /** Parse a complete v2 message from a ByteBuffer positioned at the flags byte. */
  public static ActionMessage read(ByteBuffer buffer) {
    int flags = buffer.get() & 0xFF;
    if ((flags & FLAG_MASK_RESERVED) != 0) {
      throw new IllegalArgumentException(
          "Reserved flag bits set: " + Integer.toHexString(flags));
    }
    int msgType = flags & 0x3;

    int[] keyPress = new int[0];
    int[] keyRelease = new int[0];
    boolean hasMouse = false;
    float mouseDx = 0f;
    float mouseDy = 0f;
    boolean hasButtons = false;
    int buttonPress = 0;
    int buttonRelease = 0;
    boolean hasScroll = false;
    float scroll = 0f;
    String text = "";

    if (msgType == MSG_TYPE_ACTION) {
      if ((flags & FLAG_HAS_KEYS) != 0) {
        int numPress = buffer.get() & 0xFF;
        int numRelease = buffer.get() & 0xFF;
        keyPress = new int[numPress];
        for (int i = 0; i < numPress; i++) {
          keyPress[i] = buffer.getShort();
        }
        keyRelease = new int[numRelease];
        for (int i = 0; i < numRelease; i++) {
          keyRelease[i] = buffer.getShort();
        }
      }
      if ((flags & FLAG_HAS_MOUSE) != 0) {
        mouseDx = buffer.getFloat();
        mouseDy = buffer.getFloat();
        hasMouse = true;
      }
      if ((flags & FLAG_HAS_BUTTONS) != 0) {
        int b = buffer.get() & 0xFF;
        buttonPress = b & 0x7;
        buttonRelease = (b >> 3) & 0x7;
        hasButtons = true;
      }
      if ((flags & FLAG_HAS_SCROLL) != 0) {
        scroll = buffer.getFloat();
        hasScroll = true;
      }
    } else if (msgType == MSG_TYPE_TEXT) {
      int textLen = buffer.getShort() & 0xFFFF;
      byte[] textBytes = new byte[textLen];
      buffer.get(textBytes);
      text = new String(textBytes, StandardCharsets.UTF_8);
    }
    // RESET and PING carry no body.

    if (buffer.hasRemaining()) {
      throw new IllegalArgumentException("Trailing bytes in ActionMessage");
    }

    return new ActionMessage(
        msgType,
        keyPress,
        keyRelease,
        hasMouse,
        mouseDx,
        mouseDy,
        hasButtons,
        buttonPress,
        buttonRelease,
        hasScroll,
        scroll,
        text);
  }
}
