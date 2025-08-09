package com.mvi.mvimod;

import com.mojang.logging.LogUtils;
import net.minecraft.client.Minecraft;
import net.minecraft.client.KeyMapping;
import org.slf4j.Logger;
import org.lwjgl.glfw.GLFW;

public class ActionHandler {
  private static final Logger LOGGER = LogUtils.getLogger();
  private static boolean rightMouseDown = false;
  private static boolean leftMouseDown = false;
  private static boolean virtualMouseInitialized = false;
  private static double virtualMouseX = 0.0;
  private static double virtualMouseY = 0.0;
  private static boolean suppressSystemMouseInput = true;

  public static ActionState getActionState(Minecraft mc) {
    final ActionState actionState = new ActionState(
      mc.options.keyUp.isDown(),
      mc.options.keyDown.isDown(),
      mc.options.keyLeft.isDown(),
      mc.options.keyRight.isDown(),
      mc.options.keyJump.isDown(),
      mc.options.keyShift.isDown(),
      mc.options.keySprint.isDown(),
      mc.options.keyInventory.isDown(),
      mc.options.keyDrop.isDown(),
      mc.options.keySwapOffhand.isDown(),
      mc.options.keyUse.isDown(),
      mc.options.keyAttack.isDown(),
      mc.options.keyPickItem.isDown(),
      mc.options.keyHotbarSlots[0].isDown(),
      mc.options.keyHotbarSlots[1].isDown(),
      mc.options.keyHotbarSlots[2].isDown(),
      mc.options.keyHotbarSlots[3].isDown(),
      mc.options.keyHotbarSlots[4].isDown(),
      mc.options.keyHotbarSlots[5].isDown(),
      mc.options.keyHotbarSlots[6].isDown(),
      mc.options.keyHotbarSlots[7].isDown(),
      rightMouseDown,
      leftMouseDown
    );
    return actionState;
  }

  public static void clickKeyMapping(KeyMapping keyMapping, boolean click) {
    if (click) {
      KeyMapping.click(keyMapping.getKey());
    }
  }

  public static void pressKeyMapping(KeyMapping keyMapping, boolean down) {
    if (down && !keyMapping.isDown()) {
      LOGGER.info("Pressing key mapping: {}", keyMapping.getName());
      keyMapping.setDown(true);
    } else if (!down && keyMapping.isDown()) {
      LOGGER.info("Releasing key mapping: {}", keyMapping.getName());
      keyMapping.setDown(false);
    }
  }

  public static void exitMenu(Minecraft mc, boolean exit) {
    if (exit) {
      mc.setScreen(null);
    }
  }

  public static void turnPlayer(Minecraft mc, float deltaX, float deltaY) {
    if (deltaX != 0.0f || deltaY != 0.0f) {
      mc.player.turn(deltaX, deltaY);
    }
  }

  public static void handleMouseControl(Minecraft mc, float deltaX, float deltaY) {
    if (isScreenOpen(mc)) {
      moveMouseInGUI(mc, deltaX, deltaY);
    } else {
      // Re-enable OS cursor when not in GUI
      if (suppressSystemMouseInput) {
        long windowHandle = mc.getWindow().getWindow();
        GLFW.glfwSetInputMode(windowHandle, GLFW.GLFW_CURSOR, GLFW.GLFW_CURSOR_NORMAL);
      }
      turnPlayer(mc, deltaX, deltaY);
    }
  }

  public static boolean isScreenOpen(Minecraft mc) {
    return mc.screen != null;
  }

  public static void moveMouseInGUI(Minecraft mc, float deltaX, float deltaY) {
    if (mc.screen != null) {
      // Detach OS mouse by disabling the cursor when suppressing system input
      suppressSystemMouseInput = Config.SUPPRESS_SYSTEM_MOUSE_INPUT.get();
      if (suppressSystemMouseInput) {
        long windowHandle = mc.getWindow().getWindow();
        GLFW.glfwSetInputMode(windowHandle, GLFW.GLFW_CURSOR, GLFW.GLFW_CURSOR_DISABLED);
      }
      // Initialize virtual mouse at window center if needed
      if (!virtualMouseInitialized) {
        virtualMouseX = mc.getWindow().getWidth() / 2.0;
        virtualMouseY = mc.getWindow().getHeight() / 2.0;
        virtualMouseInitialized = true;
      }

      // Update internal virtual mouse position
      virtualMouseX += deltaX;
      virtualMouseY += deltaY;

      // Clamp to window bounds
      double maxX = mc.getWindow().getWidth() - 1;
      double maxY = mc.getWindow().getHeight() - 1;
      if (virtualMouseX < 0) virtualMouseX = 0;
      if (virtualMouseY < 0) virtualMouseY = 0;
      if (virtualMouseX > maxX) virtualMouseX = maxX;
      if (virtualMouseY > maxY) virtualMouseY = maxY;

      // Notify current screen about mouse movement without moving OS cursor
      mc.screen.mouseMoved(virtualMouseX, virtualMouseY);
      LOGGER.debug("GUI Virtual mouse moved by ({}, {}) to ({}, {})", deltaX, deltaY, virtualMouseX, virtualMouseY);
    }
  }

  public static void rightMouseControl(Minecraft mc, boolean down) {
    if (mc.screen != null && rightMouseDown != down) {
      suppressSystemMouseInput = Config.SUPPRESS_SYSTEM_MOUSE_INPUT.get();
      if (suppressSystemMouseInput) {
        long windowHandle = mc.getWindow().getWindow();
        GLFW.glfwSetInputMode(windowHandle, GLFW.GLFW_CURSOR, GLFW.GLFW_CURSOR_DISABLED);
      }

      int button = GLFW.GLFW_MOUSE_BUTTON_RIGHT;

      if (down) {
        mc.screen.mouseClicked(virtualMouseX, virtualMouseY, button);
        rightMouseDown = true;
      } else {
        mc.screen.mouseReleased(virtualMouseX, virtualMouseY, button);
        rightMouseDown = false;
      }

      LOGGER.debug("GUI Virtual right click at ({}, {})", virtualMouseX, virtualMouseY);
    }
  }

  public static void leftMouseControl(Minecraft mc, boolean down) {
    if (mc.screen != null && leftMouseDown != down) {
      suppressSystemMouseInput = Config.SUPPRESS_SYSTEM_MOUSE_INPUT.get();
      if (suppressSystemMouseInput) {
        long windowHandle = mc.getWindow().getWindow();
        GLFW.glfwSetInputMode(windowHandle, GLFW.GLFW_CURSOR, GLFW.GLFW_CURSOR_DISABLED);
      }

      int button = GLFW.GLFW_MOUSE_BUTTON_LEFT;

      if (down) {
        mc.screen.mouseClicked(virtualMouseX, virtualMouseY, button);
        leftMouseDown = true;
      } else {
        mc.screen.mouseReleased(virtualMouseX, virtualMouseY, button);
        leftMouseDown = false;
      }

      LOGGER.debug("GUI Virtual left click at ({}, {})", virtualMouseX, virtualMouseY);
    }
  }
}
