package com.mineagent;

import com.mojang.blaze3d.platform.Window;
import com.mojang.logging.LogUtils;
import java.nio.ByteBuffer;
import net.minecraft.client.Minecraft;
import net.minecraftforge.event.TickEvent;
import net.minecraftforge.event.entity.living.LivingDeathEvent;
import net.minecraftforge.event.entity.living.LivingHurtEvent;
import net.minecraftforge.event.server.ServerStartingEvent;
import net.minecraftforge.event.server.ServerStoppingEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import org.lwjgl.glfw.GLFW;
import org.lwjgl.opengl.GL11;
import org.slf4j.Logger;

/** Handles client-side game events and coordinates input injection with observations. */
public class ClientEventHandler {
  private static final Logger LOGGER = LogUtils.getLogger();
  private static final DataBridge dataBridge = DataBridge.getInstance();

  @SubscribeEvent
  public static void onServerStarting(ServerStartingEvent event) {
    LOGGER.info("MineAgent Mod Server Starting - Network handler is managed on client side");
  }

  @SubscribeEvent
  public static void onServerStopping(ServerStoppingEvent event) {
    LOGGER.info("MineAgent Mod Server Stopping");
  }

  /** Main game tick handler. Processes raw input and captures observations. */
  @SubscribeEvent
  public static void onClientTick(TickEvent.ClientTickEvent event) {
    Minecraft mc = Minecraft.getInstance();
    if (mc.level != null && mc.player != null && event.phase == TickEvent.Phase.END) {
      // Handle input suppression when client is connected
      handleInputSuppression(mc);

      // Process any pending raw input
      final RawInput rawInput = dataBridge.getLatestRawInput();
      if (rawInput != null) {
        dataBridge.getInputInjector().inject(rawInput);
      }

      // IMPORTANT: Maintain button state every tick for continuous actions
      // This fires press events and sets KeyMapping states for held buttons
      dataBridge.getInputInjector().maintainButtonState();

      // Capture and send observation
      final byte[] frame = captureFrame();
      dataBridge.sendObservation(new Observation(0.0, frame));
    }
  }

  /**
   * Handles input suppression when a Python client is connected. Disables the system cursor to
   * prevent real mouse input from interfering.
   */
  private static void handleInputSuppression(Minecraft mc) {
    boolean clientConnected = dataBridge.isClientConnected();
    boolean suppressMouse = Config.SUPPRESS_SYSTEM_MOUSE_INPUT.get();
    boolean suppressKeyboard = Config.SUPPRESS_SYSTEM_KEYBOARD_INPUT.get();

    if (clientConnected && (suppressMouse || suppressKeyboard)) {
      long windowHandle = mc.getWindow().getWindow();

      // Suppress mouse by hiding/disabling cursor
      if (suppressMouse) {
        GLFW.glfwSetInputMode(windowHandle, GLFW.GLFW_CURSOR, GLFW.GLFW_CURSOR_DISABLED);
      }

      // Note: Keyboard suppression would require intercepting at a lower level
      // For now, the agent's input will override via the GLFW handlers
    }
  }

  @SubscribeEvent
  public static void onPlayerHurt(LivingHurtEvent event) {
    // Future: Calculate reward based on damage
    // if (event.getEntity() instanceof Player) {
    //     dataBridge.sendEvent("PLAYER_HURT", String.valueOf(event.getAmount()));
    // }
  }

  @SubscribeEvent
  public static void onPlayerDeath(LivingDeathEvent event) {
    // Future: Calculate negative reward on death
    // if (event.getEntity() instanceof Player) {
    //     dataBridge.sendEvent("PLAYER_DEATH", "-100.0");
    // }
  }

  private static byte[] captureFrame() {
    Minecraft mc = Minecraft.getInstance();
    return captureScreenshot(mc.getWindow());
  }

  private static byte[] captureScreenshot(Window window) {
    int width = window.getWidth();
    int height = window.getHeight();

    ByteBuffer buffer = ByteBuffer.allocateDirect(width * height * 3);
    GL11.glReadPixels(0, 0, width, height, GL11.GL_RGB, GL11.GL_UNSIGNED_BYTE, buffer);

    byte[] bytes = new byte[buffer.capacity()];
    buffer.get(bytes);
    return bytes;
  }
}
