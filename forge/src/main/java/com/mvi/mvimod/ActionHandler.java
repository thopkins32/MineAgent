package com.mvi.mvimod;

import com.mojang.logging.LogUtils;
import net.minecraft.client.Minecraft;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.client.gui.screens.inventory.AbstractContainerScreen;
import org.slf4j.Logger;
import org.lwjgl.glfw.GLFW;

public class ActionHandler {
  private static final Logger LOGGER = LogUtils.getLogger();

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
      mc.options.keyHotbarSlots[7].isDown()
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

  /**
   * Enhanced mouse control that handles both world and GUI interaction
   */
  public static void handleMouseControl(Minecraft mc, float deltaX, float deltaY) {
    if (isScreenOpen(mc)) {
      // Handle GUI mouse movement
      moveMouseInGUI(mc, deltaX, deltaY);
    } else {
      // Handle world mouse movement (player turning)
      turnPlayer(mc, deltaX, deltaY);
    }
  }

  /**
   * Check if any screen is currently open that the agent can observe
   */
  public static boolean isScreenOpen(Minecraft mc) {
    return mc.screen != null;
  }

  /**
   * Move mouse cursor within GUI screens
   */
  public static void moveMouseInGUI(Minecraft mc, float deltaX, float deltaY) {
    if (mc.screen != null) {
      long windowHandle = mc.getWindow().getWindow();
      
      // Get current mouse position
      double[] currentX = new double[1];
      double[] currentY = new double[1];
      GLFW.glfwGetCursorPos(windowHandle, currentX, currentY);
      
      // Calculate new position
      double newX = currentX[0] + deltaX;
      double newY = currentY[0] + deltaY;
      
      // Clamp to screen bounds
      int screenWidth = mc.getWindow().getScreenWidth();
      int screenHeight = mc.getWindow().getScreenHeight();
      newX = Math.max(0, Math.min(screenWidth - 1, newX));
      newY = Math.max(0, Math.min(screenHeight - 1, newY));
      
      // Set new mouse position
      GLFW.glfwSetCursorPos(windowHandle, newX, newY);
      
      LOGGER.debug("GUI Mouse moved by ({}, {}) to ({}, {})", deltaX, deltaY, newX, newY);
    }
  }

  public static void clickMouse(Minecraft mc, boolean leftClick) {
    if (mc.screen != null) {
      long windowHandle = mc.getWindow().getWindow();
      
      // Get current mouse position
      double[] mouseX = new double[1];
      double[] mouseY = new double[1];
      GLFW.glfwGetCursorPos(windowHandle, mouseX, mouseY);
      
      int button = leftClick ? GLFW.GLFW_MOUSE_BUTTON_LEFT : GLFW.GLFW_MOUSE_BUTTON_RIGHT;
      
      // Simulate mouse press and release
      mc.screen.mouseClicked(mouseX[0], mouseY[0], button);
      mc.screen.mouseReleased(mouseX[0], mouseY[0], button);
      
      LOGGER.debug("GUI Mouse {} clicked at ({}, {})", 
        leftClick ? "left" : "right", mouseX[0], mouseY[0]);
    }
  }
}
