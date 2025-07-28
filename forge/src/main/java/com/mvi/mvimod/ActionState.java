package com.mvi.mvimod;

import java.util.HashMap;
import java.util.Map;

public class ActionState {
  private final Map<String, Boolean> keyStates;
  private float mouseDeltaX = 0.0f;
  private float mouseDeltaY = 0.0f;
  
  public ActionState() {
    this.keyStates = new HashMap<>();
    
    // Initialize all tracked keys to false
    initializeKeyStates();
  }
  
  private void initializeKeyStates() {
    // Movement keys
    keyStates.put("UP", false);
    keyStates.put("DOWN", false);
    keyStates.put("LEFT", false);
    keyStates.put("RIGHT", false);
    
    // Action keys
    keyStates.put("JUMP", false);     // Jump
    keyStates.put("SNEAK", false);    // Sneak
    keyStates.put("SPRINT", false);     // Sprint
    keyStates.put("INVENTORY", false);         // Inventory
    keyStates.put("DROP", false);         // Drop
    keyStates.put("CANCEL", false);    // Menu/Cancel
    keyStates.put("SWAP", false); // Swap off-hand
    keyStates.put("USE", false);
    keyStates.put("ATTACK", false);
    keyStates.put("PICK_ITEM", false);
    
    // Menu control
    keyStates.put("MENU_EXIT", false);  // ESC key for exiting menus
    
    // Number keys for hotbar
    for (int i = 1; i <= 9; i++) {
      keyStates.put("HOTBAR_" +String.valueOf(i), false);
    }
  }
  
  public boolean isKeyPressed(String key) {
    return keyStates.getOrDefault(key, false);
  }
  
  public void setKeyPressed(String key, boolean pressed) {
    keyStates.put(key, pressed);
  }
  
  public Map<String, Boolean> getKeyStates() {
    return new HashMap<>(keyStates);
  }
  
  public void releaseAllKeys() {
    for (String key : keyStates.keySet()) {
      keyStates.put(key, false);
    }
    // Reset mouse deltas when releasing all keys
    mouseDeltaX = 0.0f;
    mouseDeltaY = 0.0f;
  }
  
  /**
   * Set mouse movement delta for head/camera control
   * @param deltaX Horizontal mouse movement (positive = right)
   * @param deltaY Vertical mouse movement (positive = up)
   */
  public void setMouseDelta(float deltaX, float deltaY) {
    this.mouseDeltaX = deltaX;
    this.mouseDeltaY = deltaY;
  }
  
  /**
   * Get horizontal mouse movement delta
   * @return Mouse delta X value
   */
  public float getMouseDeltaX() {
    return mouseDeltaX;
  }
  
  /**
   * Get vertical mouse movement delta
   * @return Mouse delta Y value  
   */
  public float getMouseDeltaY() {
    return mouseDeltaY;
  }
  
  /**
   * Reset mouse deltas (typically called after processing movement)
   */
  public void resetMouseDeltas() {
    mouseDeltaX = 0.0f;
    mouseDeltaY = 0.0f;
  }
}
