package com.mineagent;

import com.mojang.blaze3d.platform.Window;
import net.minecraft.client.KeyboardHandler;
import net.minecraft.client.Minecraft;
import net.minecraft.client.MouseHandler;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Options;
import net.minecraft.client.gui.screens.Screen;
import java.lang.reflect.Field;
import static org.mockito.Mockito.*;
import org.mockito.quality.Strictness;

/**
 * Test fixture for setting up Minecraft mocks with all necessary fields configured.
 * This handles the reflection needed to set public fields on Minecraft and Options classes.
 * 
 * Usage:
 * <pre>
 * MinecraftTestFixture fixture = new MinecraftTestFixture();
 * Minecraft mc = fixture.getMinecraft();
 * // Use mc in your tests
 * </pre>
 */
public class MinecraftTestFixture {
  private final Minecraft mc;
  private final Window window;
  private final KeyboardHandler keyboardHandler;
  private final MouseHandler mouseHandler;
  private final Options options;
  private final KeyMapping keyAttack;
  private final KeyMapping keyUse;
  private final Screen screen;
  
  private static final long DEFAULT_WINDOW_HANDLE = 12345L;

  /**
   * Creates a new test fixture with all mocks configured.
   * The window handle defaults to 12345L.
   */
  public MinecraftTestFixture() {
    this(DEFAULT_WINDOW_HANDLE);
  }

  /**
   * Creates a new test fixture with a custom window handle.
   * 
   * @param windowHandle The GLFW window handle to use
   */
  public MinecraftTestFixture(long windowHandle) {
    // Create all mocks with lenient settings
    this.mc = mock(Minecraft.class, withSettings().strictness(Strictness.LENIENT));
    this.window = mock(Window.class, withSettings().strictness(Strictness.LENIENT));
    this.keyboardHandler = mock(KeyboardHandler.class, withSettings().strictness(Strictness.LENIENT));
    this.mouseHandler = mock(MouseHandler.class, withSettings().strictness(Strictness.LENIENT));
    this.options = mock(Options.class, withSettings().strictness(Strictness.LENIENT));
    this.keyAttack = mock(KeyMapping.class, withSettings().strictness(Strictness.LENIENT));
    this.keyUse = mock(KeyMapping.class, withSettings().strictness(Strictness.LENIENT));
    this.screen = mock(Screen.class, withSettings().strictness(Strictness.LENIENT));
    
    // Setup common mock behavior
    when(window.getWindow()).thenReturn(windowHandle);
    when(mc.getWindow()).thenReturn(window);
    
    // Set fields using reflection
    setupMinecraftFields();
    setupOptionsFields();
  }

  /**
   * Sets up all Minecraft class fields using reflection.
   */
  private void setupMinecraftFields() {
    setField(Minecraft.class, mc, "keyboardHandler", keyboardHandler);
    setField(Minecraft.class, mc, "mouseHandler", mouseHandler);
    setField(Minecraft.class, mc, "options", options);
  }

  /**
   * Sets up all Options class fields using reflection.
   */
  private void setupOptionsFields() {
    setField(Options.class, options, "keyAttack", keyAttack);
    setField(Options.class, options, "keyUse", keyUse);
  }

  /**
   * Sets a field on an object using reflection.
   * Silently fails if the field doesn't exist or can't be accessed.
   * 
   * @param clazz The class containing the field
   * @param target The object instance to set the field on
   * @param fieldName The name of the field
   * @param value The value to set
   */
  private static void setField(Class<?> clazz, Object target, String fieldName, Object value) {
    try {
      Field field = clazz.getField(fieldName);
      field.setAccessible(true);
      field.set(target, value);
    } catch (NoSuchFieldException | IllegalAccessException e) {
      // Field doesn't exist or can't be accessed - this might happen if
      // Minecraft structure changes, but we continue anyway
      // Tests will fail if they actually need this field
    }
  }

  /**
   * Sets a field on the Minecraft instance.
   * Useful for setting fields like "player" or "screen" in specific tests.
   * 
   * @param fieldName The name of the field
   * @param value The value to set
   */
  public void setMinecraftField(String fieldName, Object value) {
    setField(Minecraft.class, mc, fieldName, value);
  }

  /**
   * Sets a field on the Options instance.
   * 
   * @param fieldName The name of the field
   * @param value The value to set
   */
  public void setOptionsField(String fieldName, Object value) {
    setField(Options.class, options, fieldName, value);
  }

  // Getters for all mocked components
  
  public Minecraft getMinecraft() {
    return mc;
  }

  public Window getWindow() {
    return window;
  }

  public KeyboardHandler getKeyboardHandler() {
    return keyboardHandler;
  }

  public MouseHandler getMouseHandler() {
    return mouseHandler;
  }

  public Options getOptions() {
    return options;
  }

  public KeyMapping getKeyAttack() {
    return keyAttack;
  }

  public KeyMapping getKeyUse() {
    return keyUse;
  }

  public Screen getScreen() {
    return screen;
  }
}
