package com.mvi.mvimod;

import com.mojang.logging.LogUtils;
import net.minecraft.client.Minecraft;
import net.minecraft.client.KeyMapping;
import org.slf4j.Logger;

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

  public static void pressKeyMapping(KeyMapping keyMapping, boolean down) {
    if (down && !keyMapping.isDown()) {
      keyMapping.setDown(true);
    } else if (!down && keyMapping.isDown()) {
      keyMapping.setDown(false);
    }
  }

  public static void exitMenu(Minecraft mc, boolean exit) {
    if (exit) {
      mc.setScreen(null);
    }
  }

  public static void turnPlayer(Minecraft mc, float deltaX, float deltaY) {
    if (deltaX != 0.0f && deltaY != 0.0f) {
      mc.player.turn(deltaX, deltaY);
    }
  }
}
