package com.mineagent;

import com.mojang.blaze3d.platform.Window;
import com.mojang.logging.LogUtils;
import java.nio.ByteBuffer;
import net.minecraft.client.Minecraft;
import net.minecraft.world.entity.player.Player;
import net.minecraftforge.event.TickEvent;
import net.minecraftforge.event.entity.living.LivingDeathEvent;
import net.minecraftforge.event.entity.living.LivingHurtEvent;
import net.minecraftforge.event.server.ServerStartingEvent;
import net.minecraftforge.event.server.ServerStoppingEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import org.lwjgl.opengl.GL11;
import org.slf4j.Logger;

public class ClientEventHandler {
  private static final Logger LOGGER = LogUtils.getLogger();
  private static DataBridge dataBridge = DataBridge.getInstance();

  @SubscribeEvent
  public static void onServerStarting(ServerStartingEvent event) {
    LOGGER.info("MineAgent Mod Server Starting - Network handler is managed on client side");
  }

  @SubscribeEvent
  public static void onServerStopping(ServerStoppingEvent event) {
    LOGGER.info("MineAgent Mod Server Stopping");
  }

  @SubscribeEvent
  public static void onClientTick(TickEvent.ClientTickEvent event) {
    if (event.phase == TickEvent.Phase.END) {
      captureAndSendFrame();
    }
  }

  @SubscribeEvent
  public static void onPlayerHurt(LivingHurtEvent event) {
    if (event.getEntity() instanceof Player) {
      dataBridge.sendEvent("PLAYER_HURT", String.valueOf(event.getAmount()));
    }
  }

  @SubscribeEvent
  public static void onPlayerDeath(LivingDeathEvent event) {
    if (event.getEntity() instanceof Player) {
      dataBridge.sendEvent("PLAYER_DEATH", "-100.0");
    }
  }

  private static void captureAndSendFrame() {
    Minecraft mc = Minecraft.getInstance();
    if (mc.level != null && mc.player != null) {
      LOGGER.info("Capturing screenshot");
      byte[] frameData = captureScreenshot(mc.getWindow());
      LOGGER.info("Sending frame data");
      dataBridge.sendFrame(frameData);
    }
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
