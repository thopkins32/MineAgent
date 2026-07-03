package com.mineagent;

import com.mojang.logging.LogUtils;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;
import net.minecraft.client.Minecraft;
import org.lwjgl.glfw.GLFW;
import org.slf4j.Logger;

/**
 * Injects input events directly into Minecraft's GLFW callback handlers.
 *
 * <p>The v2 protocol is edge-based: {@link ActionMessage} carries PRESS / RELEASE edges for keys
 * and mouse buttons, and anything not listed is HOLD. This injector maintains the held key set and
 * held mouse-button bitmask, applies edges by firing {@code GLFW_PRESS} / {@code GLFW_RELEASE}
 * through Minecraft's handlers, and relies on {@link #maintainButtonState()} (called every tick) to
 * keep continuous actions like mining alive while buttons are held.
 */
public class InputInjector {
  private static final Logger LOGGER = LogUtils.getLogger();

  // Modifier keys
  private static final Set<Integer> MODIFIER_KEYS =
      new HashSet<>(
          Arrays.asList(
              GLFW.GLFW_KEY_LEFT_SHIFT,
              GLFW.GLFW_KEY_RIGHT_SHIFT,
              GLFW.GLFW_KEY_LEFT_CONTROL,
              GLFW.GLFW_KEY_RIGHT_CONTROL,
              GLFW.GLFW_KEY_LEFT_ALT,
              GLFW.GLFW_KEY_RIGHT_ALT,
              GLFW.GLFW_KEY_LEFT_SUPER,
              GLFW.GLFW_KEY_RIGHT_SUPER,
              GLFW.GLFW_KEY_MENU));

  // Held key state (the set of keys currently considered down).
  private Set<Integer> heldKeys = new HashSet<>();
  private int currentModifiers = 0;

  // Held mouse button state (bits: 0=left, 1=right, 2=middle).
  private byte heldMouseButtons = 0;

  // Virtual mouse position (absolute coordinates from accumulated deltas, used in menus).
  private double virtualMouseX = 0.0;
  private double virtualMouseY = 0.0;
  private boolean mouseInitialized = false;

  /** Injects an {@link ActionMessage} by applying its edges in order: keys, mouse, buttons, scroll, text. */
  public void inject(ActionMessage message) {
    Minecraft mc = Minecraft.getInstance();
    if (mc == null || mc.getWindow() == null) {
      LOGGER.warn("Cannot inject input - Minecraft not initialized");
      return;
    }

    long window = mc.getWindow().getWindow();

    // 1. Key edges (PRESS adds, RELEASE removes, unlisted = HOLD).
    for (int code : message.keyPress()) {
      pressKey(mc, window, code);
    }
    for (int code : message.keyRelease()) {
      releaseKey(mc, window, code);
    }

    // 2. Mouse movement.
    if (message.hasMouse()) {
      handleMouseMovement(mc, window, message.mouseDx(), message.mouseDy());
    }

    // 3. Mouse button edges.
    if (message.hasButtons()) {
      pressButtons(mc, window, message.buttonPress());
      releaseButtons(mc, window, message.buttonRelease());
    }

    // 4. Scroll wheel.
    if (message.hasScroll()) {
      handleScrollWheel(mc, window, message.scroll());
    }

    // 5. Text input (for chat / signs).
    handleTextInput(mc, message.text());
  }

  /** Presses a key: fires GLFW_PRESS for newly-held keys, updates held state and modifiers. */
  private void pressKey(Minecraft mc, long window, int keyCode) {
    if (heldKeys.contains(keyCode)) {
      return; // already held -> HOLD, no event
    }
    Set<Integer> newHeld = new HashSet<>(heldKeys);
    newHeld.add(keyCode);
    int modifiers = computeModifiers(findModifierKeys(newHeld));
    fireKeyEvent(mc, window, keyCode, GLFW.GLFW_PRESS, modifiers);
    heldKeys.add(keyCode);
    currentModifiers = computeModifiers(findModifierKeys(heldKeys));
  }

  /** Releases a key: fires GLFW_RELEASE for previously-held keys, updates held state and modifiers. */
  private void releaseKey(Minecraft mc, long window, int keyCode) {
    if (!heldKeys.contains(keyCode)) {
      return; // not held -> nothing to release
    }
    fireKeyEvent(mc, window, keyCode, GLFW.GLFW_RELEASE, currentModifiers);
    heldKeys.remove(keyCode);
    currentModifiers = computeModifiers(findModifierKeys(heldKeys));
  }

  /** Fires a key event through Minecraft's KeyboardHandler. */
  private void fireKeyEvent(Minecraft mc, long window, int keyCode, int action, int modifiers) {
    int scanCode = GLFW.glfwGetKeyScancode(keyCode);

    LOGGER.debug(
        "Firing key event: keyCode={}, scanCode={}, action={}, mods={}",
        keyCode,
        scanCode,
        action == GLFW.GLFW_PRESS ? "PRESS" : "RELEASE",
        modifiers);

    mc.keyboardHandler.keyPress(window, keyCode, scanCode, action, modifiers);
  }

  /**
   * Finds all modifier keys in a set of keys.
   *
   * <p>If every key in the set is a modifier, returns an empty set so that a lone modifier press is
   * treated as a plain key press (mods = 0) rather than self-modifying.
   */
  private Set<Integer> findModifierKeys(Set<Integer> keys) {
    Set<Integer> modifiers =
        keys.stream().filter(MODIFIER_KEYS::contains).collect(Collectors.toSet());
    if (!modifiers.isEmpty() && modifiers.size() == keys.size()) {
      return Collections.emptySet();
    }
    return modifiers;
  }

  /** Computes the GLFW modifier bitmask for a set of modifier key codes. */
  private int computeModifiers(Set<Integer> modifierKeys) {
    int mods = 0;
    if (modifierKeys.contains(GLFW.GLFW_KEY_LEFT_SHIFT)
        || modifierKeys.contains(GLFW.GLFW_KEY_RIGHT_SHIFT)) {
      mods |= GLFW.GLFW_MOD_SHIFT;
    }
    if (modifierKeys.contains(GLFW.GLFW_KEY_LEFT_CONTROL)
        || modifierKeys.contains(GLFW.GLFW_KEY_RIGHT_CONTROL)) {
      mods |= GLFW.GLFW_MOD_CONTROL;
    }
    if (modifierKeys.contains(GLFW.GLFW_KEY_LEFT_ALT)
        || modifierKeys.contains(GLFW.GLFW_KEY_RIGHT_ALT)) {
      mods |= GLFW.GLFW_MOD_ALT;
    }
    return mods;
  }

  /** Presses the mouse buttons set in {@code pressMask} (bits 0-2: left, right, middle). */
  private void pressButtons(Minecraft mc, long window, int pressMask) {
    for (int button = 0; button < 3; button++) {
      if ((pressMask & (1 << button)) == 0) {
        continue;
      }
      if ((heldMouseButtons & (1 << button)) != 0) {
        continue; // already held -> HOLD
      }
      LOGGER.debug("Mouse button: button={}, action=PRESS", button);
      mc.mouseHandler.onPress(window, button, GLFW.GLFW_PRESS, currentModifiers);
      heldMouseButtons |= (byte) (1 << button);
    }
  }

  /** Releases the mouse buttons set in {@code releaseMask} (bits 0-2: left, right, middle). */
  private void releaseButtons(Minecraft mc, long window, int releaseMask) {
    for (int button = 0; button < 3; button++) {
      if ((releaseMask & (1 << button)) == 0) {
        continue;
      }
      if ((heldMouseButtons & (1 << button)) == 0) {
        continue; // not held -> nothing to release
      }
      LOGGER.debug("Mouse button: button={}, action=RELEASE", button);
      mc.mouseHandler.onPress(window, button, GLFW.GLFW_RELEASE, currentModifiers);
      heldMouseButtons &= (byte) ~(1 << button);
    }
  }

  /**
   * Handles mouse movement by directly rotating the player.
   *
   * <p>MouseHandler.onMove() only works when the mouse is "grabbed" (captured for gameplay), so we
   * bypass it and call player.turn() directly, which is what Minecraft ultimately does.
   *
   * <p>The delta values are in "pixel" units and get scaled by mouse sensitivity.
   */
  private void handleMouseMovement(Minecraft mc, long window, float deltaX, float deltaY) {
    if (deltaX == 0 && deltaY == 0) {
      return;
    }

    // Only rotate the player when in-game (not in menus)
    if (mc.player == null || mc.screen != null) {
      // If in a menu, we could use onMove for menu interaction
      if (mc.screen != null) {
        if (!mouseInitialized) {
          virtualMouseX = mc.getWindow().getWidth() / 2.0;
          virtualMouseY = mc.getWindow().getHeight() / 2.0;
          mouseInitialized = true;
        }
        virtualMouseX += deltaX;
        virtualMouseY += deltaY;
        mc.mouseHandler.onMove(window, virtualMouseX, virtualMouseY);
      }
      return;
    }

    // Get mouse sensitivity from options (default 0.5, range 0-1)
    double sensitivity = mc.options.sensitivity().get() * 0.6 + 0.2;
    double sensitivityCubed = sensitivity * sensitivity * sensitivity * 8.0;

    // Convert pixel deltas to rotation deltas (matching Minecraft's turnPlayer logic)
    double yawDelta = deltaX * sensitivityCubed;
    double pitchDelta = deltaY * sensitivityCubed;

    LOGGER.debug(
        "Mouse move: delta=({}, {}), yaw={}, pitch={}", deltaX, deltaY, yawDelta, pitchDelta);

    mc.player.turn(yawDelta, pitchDelta);
  }

  /** Handles scroll wheel input by calling MouseHandler.onScroll(). */
  private void handleScrollWheel(Minecraft mc, long window, float scrollDelta) {
    if (scrollDelta == 0) {
      return;
    }

    LOGGER.debug("Scroll: delta={}", scrollDelta);
    mc.mouseHandler.onScroll(window, 0.0, (double) scrollDelta);
  }

  /**
   * Handles text input for chat, signs, and other text fields. Only processes when a screen is
   * open.
   */
  private void handleTextInput(Minecraft mc, String text) {
    if (text == null || text.isEmpty()) {
      return;
    }

    if (mc.screen == null) {
      LOGGER.debug("Ignoring text input - no screen open: '{}'", text);
      return;
    }

    LOGGER.debug("Text input: '{}'", text);

    for (char c : text.toCharArray()) {
      mc.screen.charTyped(c, 0);
    }
  }

  /** Resets all input state. Call this on RESET, disconnect, or cleanup. */
  public void reset() {
    Minecraft mc = Minecraft.getInstance();
    if (mc != null && mc.getWindow() != null) {
      long window = mc.getWindow().getWindow();

      // Release all held keys
      for (int key : heldKeys) {
        fireKeyEvent(mc, window, key, GLFW.GLFW_RELEASE, currentModifiers);
      }

      // Release all held mouse buttons
      for (int button = 0; button < 3; button++) {
        if ((heldMouseButtons & (1 << button)) != 0) {
          mc.mouseHandler.onPress(window, button, GLFW.GLFW_RELEASE, currentModifiers);
        }
      }
    }

    heldKeys.clear();
    currentModifiers = 0;
    heldMouseButtons = 0;
    mouseInitialized = false;

    LOGGER.info("InputInjector reset");
  }

  /** Gets the current virtual mouse X position. */
  public double getVirtualMouseX() {
    return virtualMouseX;
  }

  /** Gets the current virtual mouse Y position. */
  public double getVirtualMouseY() {
    return virtualMouseY;
  }

  /** Gets the set of currently held key codes. */
  public Set<Integer> getHeldKeys() {
    return new HashSet<>(heldKeys);
  }

  /** Gets the current modifier key state. */
  public int getModifiers() {
    return currentModifiers;
  }

  /** Gets the current held mouse button bitmask (bits 0-2: left, right, middle). */
  public byte getHeldMouseButtons() {
    return heldMouseButtons;
  }

  /**
   * Maintains continuous button state by firing press events every tick. This must be called every
   * tick to simulate holding a mouse button (e.g. for mining).
   */
  public void maintainButtonState() {
    Minecraft mc = Minecraft.getInstance();
    if (mc == null || mc.getWindow() == null || mc.mouseHandler == null) {
      return;
    }

    // If any buttons are held, fire press events to maintain the state
    if (heldMouseButtons != 0) {
      long window = mc.getWindow().getWindow();

      for (int button = 0; button < 3; button++) {
        if ((heldMouseButtons & (1 << button)) != 0) {
          mc.mouseHandler.onPress(window, button, GLFW.GLFW_PRESS, currentModifiers);
        }
      }
    }

    // Also set KeyMapping states as backup
    if (mc.options != null) {
      boolean leftDown = (heldMouseButtons & 1) != 0;
      boolean rightDown = (heldMouseButtons & 2) != 0;
      mc.options.keyAttack.setDown(leftDown);
      mc.options.keyUse.setDown(rightDown);
    }
  }
}
