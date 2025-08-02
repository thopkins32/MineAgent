package com.mvi.mvimod;

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
import org.lwjgl.opengl.GL11;
import org.slf4j.Logger;

public class ClientEventHandler {
  private static final Logger LOGGER = LogUtils.getLogger();
  private static DataBridge dataBridge = DataBridge.getInstance();
  
  @SubscribeEvent
  public static void onServerStarting(ServerStartingEvent event) {
    LOGGER.info("MVI Mod Server Starting - Network handler is managed on client side");
  }

  @SubscribeEvent
  public static void onServerStopping(ServerStoppingEvent event) {
    LOGGER.info("MVI Mod Server Stopping");
  }

  private static void processAction(Action action) {
    Minecraft mc = Minecraft.getInstance();
    ActionHandler.pressKeyMapping(mc.options.keyUp, action.up());
    ActionHandler.pressKeyMapping(mc.options.keyDown, action.down());
    ActionHandler.pressKeyMapping(mc.options.keyLeft, action.left());
    ActionHandler.pressKeyMapping(mc.options.keyRight, action.right());
    ActionHandler.pressKeyMapping(mc.options.keyJump, action.jump());
    ActionHandler.pressKeyMapping(mc.options.keyShift, action.sneak());
    ActionHandler.pressKeyMapping(mc.options.keySprint, action.sprint());
    ActionHandler.pressKeyMapping(mc.options.keyInventory, action.inventory());
    ActionHandler.pressKeyMapping(mc.options.keyDrop, action.drop());
    ActionHandler.pressKeyMapping(mc.options.keySwapOffhand, action.swap());
    ActionHandler.pressKeyMapping(mc.options.keyUse, action.use());
    ActionHandler.pressKeyMapping(mc.options.keyAttack, action.attack());
    ActionHandler.pressKeyMapping(mc.options.keyPickItem, action.pick_item());
    ActionHandler.pressKeyMapping(mc.options.keyHotbarSlots[0], action.hotbar1());
    ActionHandler.pressKeyMapping(mc.options.keyHotbarSlots[1], action.hotbar2());
    ActionHandler.pressKeyMapping(mc.options.keyHotbarSlots[2], action.hotbar3());
    ActionHandler.pressKeyMapping(mc.options.keyHotbarSlots[3], action.hotbar4());
    ActionHandler.pressKeyMapping(mc.options.keyHotbarSlots[4], action.hotbar5());
    ActionHandler.pressKeyMapping(mc.options.keyHotbarSlots[5], action.hotbar6());
    ActionHandler.pressKeyMapping(mc.options.keyHotbarSlots[6], action.hotbar7());
    ActionHandler.pressKeyMapping(mc.options.keyHotbarSlots[7], action.hotbar8());
    ActionHandler.exitMenu(mc, action.exitMenu());
    ActionHandler.turnPlayer(mc, action.mouseControlX(), action.mouseControlY());
  }

  @SubscribeEvent
  public static void onClientTick(TickEvent.ClientTickEvent event) {
    Minecraft mc = Minecraft.getInstance();
    if (mc.level != null && mc.player != null) {
      final Action action = dataBridge.getLatestAction();
      if (action != null) {
        processAction(action);
      }

      if (event.phase == TickEvent.Phase.END) {
        // int reward = packageReward();
        final ActionState currentActionState = ActionHandler.getActionState(Minecraft.getInstance());
        final byte[] frame = captureFrame();
        dataBridge.sendObservation(new Observation(0.0, currentActionState, frame));
      }
    }
  }

  @SubscribeEvent
  public static void onPlayerHurt(LivingHurtEvent event) {
    // if (event.getEntity() instanceof Player) {
    //   dataBridge.sendEvent("PLAYER_HURT", String.valueOf(event.getAmount()));
    // }
  }

  @SubscribeEvent
  public static void onPlayerDeath(LivingDeathEvent event) {
    // if (event.getEntity() instanceof Player) {
    //   dataBridge.sendEvent("PLAYER_DEATH", "-100.0");
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
  
  /**
   * Release all currently pressed keys (for cleanup on disconnect)
   */
  public static void releaseAllKeys() {
    Minecraft mc = Minecraft.getInstance();
    ActionHandler.pressKeyMapping(mc.options.keyUp, false);
    ActionHandler.pressKeyMapping(mc.options.keyDown, false);
    ActionHandler.pressKeyMapping(mc.options.keyLeft, false);
    ActionHandler.pressKeyMapping(mc.options.keyRight, false);
    ActionHandler.pressKeyMapping(mc.options.keyJump, false);
    ActionHandler.pressKeyMapping(mc.options.keyShift, false);
    ActionHandler.pressKeyMapping(mc.options.keySprint, false);
    ActionHandler.pressKeyMapping(mc.options.keyInventory, false);
    ActionHandler.pressKeyMapping(mc.options.keyDrop, false);
    ActionHandler.pressKeyMapping(mc.options.keySwapOffhand, false);
    ActionHandler.pressKeyMapping(mc.options.keyUse, false);
    ActionHandler.pressKeyMapping(mc.options.keyAttack, false);
    ActionHandler.pressKeyMapping(mc.options.keyPickItem, false);
    ActionHandler.pressKeyMapping(mc.options.keyHotbarSlots[0], false);
    ActionHandler.pressKeyMapping(mc.options.keyHotbarSlots[1], false);
    ActionHandler.pressKeyMapping(mc.options.keyHotbarSlots[2], false);
    ActionHandler.pressKeyMapping(mc.options.keyHotbarSlots[3], false);
    ActionHandler.pressKeyMapping(mc.options.keyHotbarSlots[4], false);
    ActionHandler.pressKeyMapping(mc.options.keyHotbarSlots[5], false);
    ActionHandler.pressKeyMapping(mc.options.keyHotbarSlots[6], false);
    ActionHandler.pressKeyMapping(mc.options.keyHotbarSlots[7], false);
  }
}
