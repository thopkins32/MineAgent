package com.mineagent;

import com.mojang.blaze3d.platform.Window;
import com.mojang.logging.LogUtils;
import java.nio.ByteBuffer;
import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.client.gui.screens.TitleScreen;
import net.minecraft.client.gui.screens.DeathScreen;
import net.minecraft.client.gui.screens.PauseScreen;
import net.minecraft.world.entity.LivingEntity;
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
    if (event.phase != TickEvent.Phase.END) {
      return;
    }
    Minecraft mc = Minecraft.getInstance();
    boolean inWorld = mc.level != null && mc.player != null;
    boolean onTitleScreen = mc.screen instanceof TitleScreen;
    boolean onDeathScreen = mc.screen instanceof DeathScreen;
    boolean onPauseScreen = mc.screen instanceof PauseScreen;
    boolean isMenu = mc.screen != null;
    boolean inWorldWithOverlay = mc.level != null && mc.screen != null;

    if (onTitleScreen) {
      mc.createWorldOpenFlows().openWorld("New World", () -> {
        mc.forceSetScreen(new TitleScreen());
      });
      // final ResetInput resetInput = dataBridge.getLatestResetInput();
      // if (resetInput != null) {
      //   // TODO: Open new world? This currently only opens existing world
      //   // Minecraft mc = Minecraft.getInstance();
      //   // createFreshLevel opens the world creation flow.
      //   // For a quick programmatic world with defaults:
      //   // mc.createWorldOpenFlows().createFreshLevel(
      //   //     saveName,           // String: folder name for the save
      //   //     displayName,        // String: human-readable world name
      //   //     WorldPresets.NORMAL, // or FLAT, AMPLIFIED, etc. -- ResourceKey<WorldPreset>
      //   //     null,               // @Nullable Screen to return to on cancel
      //   //     null                // @Nullable WorldCreationContext overrides
      //   // );
      // }
      return;
    }
    else if (onPauseScreen) {
      mc.setScreen(null);
      return;
    }

    if (!inWorld) {
      return;
    }

    if (onDeathScreen) {
      mc.player.respawn();
      return;
    }

    enforceWindowSize(mc);

    handleInputSuppression(mc);

    // Process any pending raw input
    final RawInput rawInput = dataBridge.getLatestRawInput();
    if (rawInput != null) {
      dataBridge.getInputInjector().inject(rawInput);
    }

    // IMPORTANT: Maintain button state every tick for continuous actions
    // This fires press events and sets KeyMapping states for held buttons
    dataBridge.getInputInjector().maintainButtonState();

    // Observations only while the local player exists (no capture on death screen). Extrinsic
    // reward queued during death therefore attaches to the first frame after respawn if needed.
    double reward = dataBridge.takeExtrinsicReward();
    dataBridge.sendObservation(new Observation(reward, captureFrame()));
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
    if (!isClientControlledPlayer(event.getEntity())) {
      return;
    }
    double perPoint = Config.EXTRINSIC_DAMAGE_PER_POINT.get();
    if (perPoint != 0.0) {
      dataBridge.addExtrinsicReward(-perPoint * event.getAmount());
    }
  }

  @SubscribeEvent
  public static void onPlayerDeath(LivingDeathEvent event) {
    if (!isClientControlledPlayer(event.getEntity())) {
      return;
    }
    double penalty = Config.DEATH_PENALTY.get();
    if (penalty != 0.0) {
      dataBridge.addExtrinsicReward(-penalty);
    }
  }

  /** The player the agent controls on this machine (not other players or mobs). */
  private static boolean isClientControlledPlayer(LivingEntity entity) {
    return entity instanceof LocalPlayer p && p == Minecraft.getInstance().player;
  }

  private static void enforceWindowSize(Minecraft mc) {
    Window window = mc.getWindow();
    int targetW = Config.WINDOW_WIDTH.get();
    int targetH = Config.WINDOW_HEIGHT.get();
    if (window.getWidth() != targetW || window.getHeight() != targetH) {
      window.setWindowed(targetW, targetH);
    }
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
