package com.mvi.mvimod;

import com.mojang.blaze3d.platform.Window;
import com.mojang.logging.LogUtils;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import net.minecraft.client.Minecraft;
import net.minecraft.client.KeyMapping;
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
  
  // Track action state across ticks
  private static ActionState actionState = new ActionState();
  
  // Map string keys to Minecraft KeyMapping objects
  private static final Map<String, KeyMapping> keyMappings = new HashMap<>();
  private static boolean keyMappingsInitialized = false;

  @SubscribeEvent
  public static void onServerStarting(ServerStartingEvent event) {
    LOGGER.info("MVI Mod Server Starting - Network handler is managed on client side");
    initializeKeyMappings();
  }

  @SubscribeEvent
  public static void onServerStopping(ServerStoppingEvent event) {
    LOGGER.info("MVI Mod Server Stopping");
  }

  @SubscribeEvent
  public static void onClientTick(TickEvent.ClientTickEvent event) {
    
    ArrayList<String> commands = dataBridge.emptyCommandQueue();
    for (String command : commands) {
      processCommand(command);
    }

    if (event.phase == TickEvent.Phase.END) {
      // Increment action state timings
      actionState.incrementTimings();
      
      // TODO: Move to data bridge?
      int reward = packageReward();
      ActionState currentActionState = captureActionState();
      byte[] frame = captureFrame();
      dataBridge.sendObservation(new Observation(reward, currentActionState, frame));
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
  
  private static void initializeKeyMappings() {
    Minecraft mc = Minecraft.getInstance();
    if (mc.options == null) return;
    
    // Movement keys
    keyMappings.put("UP", mc.options.keyUp);
    keyMappings.put("DOWN", mc.options.keyDown);
    keyMappings.put("LEFT", mc.options.keyLeft);
    keyMappings.put("RIGHT", mc.options.keyRight);
    
    // Action keys
    keyMappings.put("JUMP", mc.options.keyJump);
    keyMappings.put("SNEAK", mc.options.keyShift);
    keyMappings.put("SPRINT", mc.options.keySprint);
    keyMappings.put("INVENTORY", mc.options.keyInventory);
    keyMappings.put("DROP", mc.options.keyDrop);
    keyMappings.put("SWAP", mc.options.keySwapOffhand);
    keyMappings.put("USE", mc.options.keyUse);
    keyMappings.put("ATTACK", mc.options.keyAttack);
    keyMappings.put("PICK_ITEM", mc.options.keyPickItem);
    
    // Number keys for hotbar
    KeyMapping[] hotbarKeys = mc.options.keyHotbarSlots;
    for (int i = 0; i < hotbarKeys.length && i < 9; i++) {
      keyMappings.put("HOTBAR_" + String.valueOf(i + 1), hotbarKeys[i]);
    }
    
    keyMappingsInitialized = true;
    LOGGER.info("Key mappings initialized with {} keys", keyMappings.size());
  }
  
  private static void processCommand(String command) {
    LOGGER.debug("Processing command: {}", command);
    
    if (command.startsWith("PRESS_")) {
      String keyName = command.substring(6);
      pressKey(keyName);
    } else if (command.startsWith("RELEASE_")) {
      String keyName = command.substring(8);
      releaseKey(keyName);
    } else if (command.startsWith("MOUSE_MOVE_")) {
      // Format: MOUSE_MOVE_X_Y (e.g., MOUSE_MOVE_5.0_-2.5)
      String coords = command.substring(11);
      String[] parts = coords.split("_");
      if (parts.length == 2) {
        try {
          float deltaX = Float.parseFloat(parts[0]);
          float deltaY = Float.parseFloat(parts[1]);
          moveMouseForHeadControl(deltaX, deltaY);
        } catch (NumberFormatException e) {
          LOGGER.warn("Invalid mouse movement coordinates: {}", coords);
        }
      }
    } else if (command.equals("ESCAPE")) {
      pressEscapeKey();
    } else {
      LOGGER.warn("Unknown command format: {}", command);
    }
  }
  
  private static void pressKey(String keyName) {
    KeyMapping keyMapping = keyMappings.get(keyName);
    if (keyMapping != null) {
      if (!keyMapping.isDown()) {
        keyMapping.setDown(true);
        actionState.setKeyPressed(keyName, true);
        LOGGER.debug("Pressed key: {}", keyName);
      }
    } else {
      LOGGER.warn("Unknown key: {}", keyName);
    }
  }
  
  private static void releaseKey(String keyName) {
    KeyMapping keyMapping = keyMappings.get(keyName);
    if (keyMapping != null) {
      if (keyMapping.isDown()) {
        keyMapping.setDown(false);
        actionState.setKeyPressed(keyName, false);
        LOGGER.debug("Released key: {}", keyName);
      }
    } else {
      LOGGER.warn("Unknown key: {}", keyName);
    }
  }
  
  private static void moveMouseForHeadControl(float deltaX, float deltaY) {
    Minecraft mc = Minecraft.getInstance();
    if (mc.player != null) {
      // Store mouse deltas in action state
      actionState.setMouseDelta(deltaX, deltaY);
      
      // Apply mouse movement to player rotation
      // deltaX affects yaw (horizontal looking)
      // deltaY affects pitch (vertical looking) 
      mc.player.turn(deltaX * 0.15, deltaY * 0.15);
      
      LOGGER.debug("Applied mouse movement: deltaX={}, deltaY={}", deltaX, deltaY);
    }
  }
  
  private static void pressEscapeKey() {
    Minecraft mc = Minecraft.getInstance();
    if (mc.screen != null) {
      // Close current screen/menu
      mc.setScreen(null);
    }
    actionState.setKeyPressed("MENU_EXIT", true);
    LOGGER.debug("Pressed ESC key - closing menu");
  }

  private static int packageReward() {
    // TODO: Get rewards from the reward queue
    return 0; // Placeholder
  }

  private static ActionState captureActionState() {
    // Return a copy of the current action state
    ActionState currentState = new ActionState();
    for (Map.Entry<String, Boolean> entry : actionState.getKeyStates().entrySet()) {
      currentState.setKeyPressed(entry.getKey(), entry.getValue());
    }
    return currentState;
  }

  private static byte[] captureFrame() {
    Minecraft mc = Minecraft.getInstance();
    if (mc.level != null && mc.player != null) {
      return captureScreenshot(mc.getWindow());
    }
    return null;
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
    if (!keyMappingsInitialized) return;
    
    LOGGER.info("Releasing all pressed keys for cleanup");
    for (Map.Entry<String, KeyMapping> entry : keyMappings.entrySet()) {
      KeyMapping keyMapping = entry.getValue();
      if (keyMapping.isDown()) {
        keyMapping.setDown(false);
        LOGGER.debug("Released key during cleanup: {}", entry.getKey());
      }
    }
    actionState.releaseAllKeys();
  }
}
