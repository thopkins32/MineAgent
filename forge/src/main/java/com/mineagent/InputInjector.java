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
 * Injects raw input events directly into Minecraft's GLFW callback handlers. This provides a
 * unified architecture where both keyboard and mouse input go through the same handlers that real
 * hardware input uses.
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

  // Key state tracking for press/release detection
  private Set<Integer> previouslyPressedKeys = new HashSet<>();
  private int previousModifiers = 0;

  // Mouse button state tracking (bits: 0=left, 1=right, 2=middle)
  private byte previousMouseButtons = 0;

  // Virtual mouse position (absolute coordinates from accumulated deltas)
  private double virtualMouseX = 0.0;
  private double virtualMouseY = 0.0;
  private boolean mouseInitialized = false;

  /**
   * Injects a RawInput into Minecraft's input handlers.
   *
   * @param input The raw input containing key codes, mouse data, and text
   */
  public void inject(RawInput input) {
    Minecraft mc = Minecraft.getInstance();
    if (mc == null || mc.getWindow() == null) {
      LOGGER.warn("Cannot inject input - Minecraft not initialized");
      return;
    }

    long window = mc.getWindow().getWindow();

    // 1. Handle key state changes via KeyboardHandler
    handleKeyboardInput(mc, window, input.keyCodes());

    // 2. Handle mouse movement via MouseHandler.onMove
    handleMouseMovement(mc, window, input.mouseDx(), input.mouseDy());

    // 3. Handle mouse buttons via MouseHandler.onPress
    handleMouseButtons(mc, window, input.mouseButtons());

    // 4. Handle scroll wheel via MouseHandler.onScroll
    handleScrollWheel(mc, window, input.scrollDelta());

    // 5. Handle text input (for chat/signs)
    handleTextInput(mc, input.text());
  }

  /**
   * Handles keyboard input by detecting press/release transitions and calling
   * KeyboardHandler.keyPress() for each event.
   */
  private void handleKeyboardInput(Minecraft mc, long window, int[] keyCodes) {
    Set<Integer> currentKeys = Arrays.stream(keyCodes).boxed().collect(Collectors.toSet());

    // Find modifier keys and non-modifier keys
    Set<Integer> modifierKeys = findModifierKeys(currentKeys);
    Set<Integer> nonModifierKeys =
        currentKeys.stream().filter(key -> !modifierKeys.contains(key)).collect(Collectors.toSet());
    int modifiers = computeModifiers(modifierKeys);

    // Release keys that were pressed but are no longer
    for (int key : previouslyPressedKeys) {
      if (!nonModifierKeys.contains(key)) {
        fireKeyEvent(mc, window, key, GLFW.GLFW_RELEASE, previousModifiers);
      }
    }

    // Press keys that are newly pressed
    for (int key : nonModifierKeys) {
      if (!previouslyPressedKeys.contains(key)) {
        fireKeyEvent(mc, window, key, GLFW.GLFW_PRESS, modifiers);
      }
    }

    previouslyPressedKeys = nonModifierKeys;
    previousModifiers = modifiers;
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

    // Call the same handler that GLFW callbacks use
    mc.keyboardHandler.keyPress(window, keyCode, scanCode, action, modifiers);
  }

  /*
   * Finds all modifier keys in a set of keys.
   *
   * @param keys The set of keys to search
   * @return The set of modifier keys, or an empty set if every key is a modifier
   */
  private Set<Integer> findModifierKeys(Set<Integer> keys) {
    Set<Integer> modifiers =
        keys.stream().filter(MODIFIER_KEYS::contains).collect(Collectors.toSet());
    // If every key is a modifier, treat as "not used as modifier" so return empty set
    if (!modifiers.isEmpty() && modifiers.size() == keys.size()) {
      return Collections.emptySet();
    }
    return modifiers;
  }

  /** Computes current modifier key state based on pressed keys. */
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
    // deltaX affects yaw (horizontal), deltaY affects pitch (vertical)
    double yawDelta = deltaX * sensitivityCubed;
    double pitchDelta = deltaY * sensitivityCubed;

    LOGGER.debug(
        "Mouse move: delta=({}, {}), yaw={}, pitch={}", deltaX, deltaY, yawDelta, pitchDelta);

    // Directly rotate the player
    // turn(yRot, xRot) where yRot is yaw change and xRot is pitch change
    mc.player.turn(yawDelta, pitchDelta);
  }

  /**
   * Handles mouse button state changes.
   *
   * <p>For continuous actions like mining, we fire press events every tick while held, and only
   * fire release events on actual release transitions.
   */
  private void handleMouseButtons(Minecraft mc, long window, byte currentButtons) {
    int modifiers = previousModifiers;

    // Check each button (0=left, 1=right, 2=middle)
    for (int button = 0; button < 3; button++) {
      boolean wasDown = (previousMouseButtons & (1 << button)) != 0;
      boolean isDown = (currentButtons & (1 << button)) != 0;

      if (isDown) {
        // Fire press event every tick while held (for continuous actions)
        LOGGER.debug("Mouse button: button={}, action=PRESS (continuous)", button);
        mc.mouseHandler.onPress(window, button, GLFW.GLFW_PRESS, modifiers);
      } else if (wasDown) {
        // Only fire release when transitioning from down to up
        LOGGER.debug("Mouse button: button={}, action=RELEASE", button);
        mc.mouseHandler.onPress(window, button, GLFW.GLFW_RELEASE, modifiers);
      }
    }

    previousMouseButtons = currentButtons;
  }

  /** Handles scroll wheel input by calling MouseHandler.onScroll(). */
  private void handleScrollWheel(Minecraft mc, long window, float scrollDelta) {
    if (scrollDelta == 0) {
      return;
    }

    LOGGER.debug("Scroll: delta={}", scrollDelta);

    // Call the same handler that GLFW scroll callbacks use
    // xOffset is typically 0 for vertical scrolling, yOffset is the scroll amount
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

  /** Resets all input state. Call this when disconnecting or cleaning up. */
  public void reset() {
    Minecraft mc = Minecraft.getInstance();
    if (mc != null && mc.getWindow() != null) {
      long window = mc.getWindow().getWindow();

      // Release all pressed keys
      for (int key : previouslyPressedKeys) {
        fireKeyEvent(mc, window, key, GLFW.GLFW_RELEASE, previousModifiers);
      }

      // Release all pressed mouse buttons
      for (int button = 0; button < 3; button++) {
        if ((previousMouseButtons & (1 << button)) != 0) {
          mc.mouseHandler.onPress(window, button, GLFW.GLFW_RELEASE, previousModifiers);
        }
      }
    }

    previouslyPressedKeys.clear();
    previousModifiers = 0;
    previousMouseButtons = 0;
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

  /** Gets the set of currently pressed key codes. */
  public Set<Integer> getPressedKeys() {
    return new HashSet<>(previouslyPressedKeys);
  }

  /** Gets the current modifier key state. */
  public int getModifiers() {
    return previousModifiers;
  }

  /** Gets the current mouse button state. */
  public byte getMouseButtons() {
    return previousMouseButtons;
  }

  /**
   * Maintains continuous button state by firing press events every tick. This must be called every
   * tick to simulate holding a mouse button.
   */
  public void maintainButtonState() {
    Minecraft mc = Minecraft.getInstance();
    if (mc == null || mc.getWindow() == null || mc.mouseHandler == null) {
      return;
    }

    // If any buttons are held, fire press events to maintain the state
    if (previousMouseButtons != 0) {
      long window = mc.getWindow().getWindow();
      int modifiers = previousModifiers;

      for (int button = 0; button < 3; button++) {
        if ((previousMouseButtons & (1 << button)) != 0) {
          mc.mouseHandler.onPress(window, button, GLFW.GLFW_PRESS, modifiers);
        }
      }
    }

    // Also set KeyMapping states as backup
    if (mc.options != null) {
      boolean leftDown = (previousMouseButtons & 1) != 0;
      boolean rightDown = (previousMouseButtons & 2) != 0;
      mc.options.keyAttack.setDown(leftDown);
      mc.options.keyUse.setDown(rightDown);
    }
  }
}
