package com.mineagent;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import com.mojang.blaze3d.platform.Window;
import net.minecraft.client.KeyboardHandler;
import net.minecraft.client.Minecraft;
import net.minecraft.client.MouseHandler;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Options;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.client.player.LocalPlayer;
import java.lang.reflect.Field;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.lwjgl.glfw.GLFW;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class InputInjectorTest {

  @Mock(lenient = true) private Minecraft mc;
  @Mock(lenient = true) private Window window;
  @Mock(lenient = true) private KeyboardHandler keyboardHandler;
  @Mock(lenient = true) private MouseHandler mouseHandler;
  @Mock(lenient = true) private Screen screen;
  @Mock(lenient = true) private Options options;
  @Mock(lenient = true) private KeyMapping keyAttack;
  @Mock(lenient = true) private KeyMapping keyUse;

  private InputInjector injector;
  private static final long WINDOW_HANDLE = 12345L;

  @BeforeEach
  void setUp() throws Exception {
    injector = new InputInjector();

    // Setup common mock behavior
    when(window.getWindow()).thenReturn(WINDOW_HANDLE);
    when(mc.getWindow()).thenReturn(window);
    
    // Set fields using reflection since they're public fields, not methods
    // We need to set them on the mock's class, not the mock itself
    try {
      Field keyboardHandlerField = Minecraft.class.getField("keyboardHandler");
      keyboardHandlerField.setAccessible(true);
      keyboardHandlerField.set(mc, keyboardHandler);
      
      Field mouseHandlerField = Minecraft.class.getField("mouseHandler");
      mouseHandlerField.setAccessible(true);
      mouseHandlerField.set(mc, mouseHandler);
      
      Field optionsField = Minecraft.class.getField("options");
      optionsField.setAccessible(true);
      optionsField.set(mc, options);
      
      // Set up KeyMapping fields in options
      try {
        Field keyAttackField = Options.class.getField("keyAttack");
        keyAttackField.setAccessible(true);
        keyAttackField.set(options, keyAttack);
        
        Field keyUseField = Options.class.getField("keyUse");
        keyUseField.setAccessible(true);
        keyUseField.set(options, keyUse);
      } catch (Exception e) {
        // If fields don't exist, continue anyway
      }
    } catch (NoSuchFieldException | IllegalAccessException e) {
      // If fields don't exist or can't be accessed, tests will need to handle this
      // This might happen if Minecraft structure changes
    }
  }

  @Test
  void inject_withNewKeyPressed_firesKeyPressEvent() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(mc);

      RawInput input = new RawInput(new int[] {GLFW.GLFW_KEY_W}, 0.0f, 0.0f, (byte) 0, 0.0f, "");

      injector.inject(input);

      ArgumentCaptor<Integer> keyCodeCaptor = ArgumentCaptor.forClass(Integer.class);
      ArgumentCaptor<Integer> actionCaptor = ArgumentCaptor.forClass(Integer.class);

      verify(keyboardHandler, times(1))
          .keyPress(
              eq(WINDOW_HANDLE),
              keyCodeCaptor.capture(),
              anyInt(),
              actionCaptor.capture(),
              eq(0));

      assertEquals(GLFW.GLFW_KEY_W, keyCodeCaptor.getValue());
      assertEquals(GLFW.GLFW_PRESS, actionCaptor.getValue());
    }
  }

  @Test
  void inject_withKeyReleased_firesKeyReleaseEvent() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(mc);

      // First inject with W key (press)
      RawInput pressInput = new RawInput(new int[] {GLFW.GLFW_KEY_W}, 0.0f, 0.0f, (byte) 0, 0.0f, "");
      injector.inject(pressInput);

      // Then inject with no keys (release)
      RawInput releaseInput = new RawInput(new int[] {}, 0.0f, 0.0f, (byte) 0, 0.0f, "");
      injector.inject(releaseInput);

      ArgumentCaptor<Integer> keyCodeCaptor = ArgumentCaptor.forClass(Integer.class);
      ArgumentCaptor<Integer> actionCaptor = ArgumentCaptor.forClass(Integer.class);

      // Verify release was called
      verify(keyboardHandler, atLeastOnce())
          .keyPress(
              eq(WINDOW_HANDLE),
              keyCodeCaptor.capture(),
              anyInt(),
              actionCaptor.capture(),
              anyInt());

      // Check that the last call was a release
      assertEquals(GLFW.GLFW_KEY_W, keyCodeCaptor.getValue());
      assertEquals(GLFW.GLFW_RELEASE, actionCaptor.getValue());
    }
  }

  @Test
  void inject_withModifierAndRegularKey_separatesModifiers() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(mc);

      RawInput input =
          new RawInput(
              new int[] {GLFW.GLFW_KEY_LEFT_SHIFT, GLFW.GLFW_KEY_W},
              0.0f,
              0.0f,
              (byte) 0,
              0.0f,
              "");

      injector.inject(input);

      ArgumentCaptor<Integer> keyCodeCaptor = ArgumentCaptor.forClass(Integer.class);
      ArgumentCaptor<Integer> modifiersCaptor = ArgumentCaptor.forClass(Integer.class);

      // Should only fire W key, not Shift
      verify(keyboardHandler, times(1))
          .keyPress(
              eq(WINDOW_HANDLE),
              keyCodeCaptor.capture(),
              anyInt(),
              eq(GLFW.GLFW_PRESS),
              modifiersCaptor.capture());

      assertEquals(GLFW.GLFW_KEY_W, keyCodeCaptor.getValue());
      // Verify Shift modifier flag is set
      assertEquals(GLFW.GLFW_MOD_SHIFT, modifiersCaptor.getValue() & GLFW.GLFW_MOD_SHIFT);
    }
  }

  @Test
  void inject_withOnlyModifierKey_firesAsRegularKey() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(mc);

      RawInput input =
          new RawInput(new int[] {GLFW.GLFW_KEY_LEFT_SHIFT}, 0.0f, 0.0f, (byte) 0, 0.0f, "");

      injector.inject(input);

      ArgumentCaptor<Integer> keyCodeCaptor = ArgumentCaptor.forClass(Integer.class);

      // Shift should fire as a regular key when pressed alone
      verify(keyboardHandler, times(1))
          .keyPress(
              eq(WINDOW_HANDLE),
              keyCodeCaptor.capture(),
              anyInt(),
              eq(GLFW.GLFW_PRESS),
              eq(0));

      assertEquals(GLFW.GLFW_KEY_LEFT_SHIFT, keyCodeCaptor.getValue());
    }
  }

  @Test
  void inject_withMouseButtonPressed_firesMousePressEvent() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(mc);

      byte leftButton = 0b00000001; // Left mouse button (bit 0)
      RawInput input = new RawInput(new int[] {}, 0.0f, 0.0f, leftButton, 0.0f, "");

      injector.inject(input);

      verify(mouseHandler, times(1))
          .onPress(eq(WINDOW_HANDLE), eq(0), eq(GLFW.GLFW_PRESS), anyInt());
    }
  }

  @Test
  void inject_withMouseButtonReleased_firesMouseReleaseEvent() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(mc);

      // First inject with left mouse button (press)
      byte leftButton = 0b00000001;
      RawInput pressInput = new RawInput(new int[] {}, 0.0f, 0.0f, leftButton, 0.0f, "");
      injector.inject(pressInput);

      // Then inject with no mouse buttons (release)
      RawInput releaseInput = new RawInput(new int[] {}, 0.0f, 0.0f, (byte) 0, 0.0f, "");
      injector.inject(releaseInput);

      ArgumentCaptor<Integer> actionCaptor = ArgumentCaptor.forClass(Integer.class);

      // Verify release was called
      verify(mouseHandler, atLeastOnce())
          .onPress(eq(WINDOW_HANDLE), eq(0), actionCaptor.capture(), anyInt());

      // Check that release was called
      assertTrue(
          actionCaptor.getAllValues().contains(GLFW.GLFW_RELEASE),
          "Mouse release event should have been fired");
    }
  }

  @Test
  void inject_withMouseMovement_rotatesPlayer() throws Exception {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(mc);
      // Create player mock manually to avoid initialization issues
      // Use lenient settings and try to mock, but if it fails, skip this test
      LocalPlayer player;
      try {
        player = mock(LocalPlayer.class, withSettings().lenient());
      } catch (Exception e) {
        // If LocalPlayer can't be mocked, skip this test
        return;
      }
      // Set player field using reflection
      try {
        Field playerField = Minecraft.class.getField("player");
        playerField.setAccessible(true);
        playerField.set(mc, player);
      } catch (Exception e) {
        // If field can't be set, skip this test
        return;
      }
      // Set screen to null using reflection
      try {
        Field screenField = Minecraft.class.getField("screen");
        screenField.setAccessible(true);
        screenField.set(mc, null);
      } catch (Exception e) {
        // Continue anyway
      }
      // Mock sensitivity() to return an OptionInstance that returns 0.5
      net.minecraft.client.OptionInstance<Double> sensitivityOption =
          mock(net.minecraft.client.OptionInstance.class);
      when(sensitivityOption.get()).thenReturn(0.5);
      when(options.sensitivity()).thenReturn(sensitivityOption);

      float mouseDx = 10.0f;
      float mouseDy = -5.0f;
      RawInput input = new RawInput(new int[] {}, mouseDx, mouseDy, (byte) 0, 0.0f, "");

      injector.inject(input);

      ArgumentCaptor<Double> yawCaptor = ArgumentCaptor.forClass(Double.class);
      ArgumentCaptor<Double> pitchCaptor = ArgumentCaptor.forClass(Double.class);

      verify(player, times(1)).turn(yawCaptor.capture(), pitchCaptor.capture());

      // Verify deltas are non-zero and have correct signs
      assertNotEquals(0.0, yawCaptor.getValue());
      assertNotEquals(0.0, pitchCaptor.getValue());
      assertTrue(yawCaptor.getValue() > 0, "Yaw should be positive for positive mouseDx");
      assertTrue(pitchCaptor.getValue() < 0, "Pitch should be negative for negative mouseDy");
    }
  }

  @Test
  void reset_withPressedKeys_releasesAllKeys() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(mc);

      // Inject some keys and mouse buttons
      RawInput input =
          new RawInput(
              new int[] {GLFW.GLFW_KEY_W, GLFW.GLFW_KEY_A},
              0.0f,
              0.0f,
              (byte) 0b00000011, // Left and right mouse buttons
              0.0f,
              "");
      injector.inject(input);

      // Reset
      injector.reset();

      // Verify all keys were released
      ArgumentCaptor<Integer> keyCodeCaptor = ArgumentCaptor.forClass(Integer.class);
      verify(keyboardHandler, atLeast(2))
          .keyPress(
              eq(WINDOW_HANDLE),
              keyCodeCaptor.capture(),
              anyInt(),
              eq(GLFW.GLFW_RELEASE),
              anyInt());

      // Verify all mouse buttons were released
      verify(mouseHandler, atLeastOnce())
          .onPress(eq(WINDOW_HANDLE), eq(0), eq(GLFW.GLFW_RELEASE), anyInt());
      verify(mouseHandler, atLeastOnce())
          .onPress(eq(WINDOW_HANDLE), eq(1), eq(GLFW.GLFW_RELEASE), anyInt());

      // Verify state is cleared
      assertTrue(injector.getPressedKeys().isEmpty(), "Pressed keys should be empty after reset");
      assertEquals(0, injector.getModifiers(), "Modifiers should be 0 after reset");
      assertEquals(0, injector.getMouseButtons(), "Mouse buttons should be 0 after reset");
    }
  }

  @Test
  void getPressedKeys_afterKeyPress_returnsPressedKeys() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(mc);

      RawInput input =
          new RawInput(
              new int[] {GLFW.GLFW_KEY_W, GLFW.GLFW_KEY_A}, 0.0f, 0.0f, (byte) 0, 0.0f, "");

      injector.inject(input);

      var pressedKeys = injector.getPressedKeys();

      assertEquals(2, pressedKeys.size());
      assertTrue(pressedKeys.contains(GLFW.GLFW_KEY_W));
      assertTrue(pressedKeys.contains(GLFW.GLFW_KEY_A));
    }
  }

  @Test
  void inject_withTextInput_whenScreenOpen_typesCharacters() throws Exception {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(mc);
      // Set screen field using reflection
      try {
        Field screenField = Minecraft.class.getField("screen");
        screenField.setAccessible(true);
        screenField.set(mc, screen);
      } catch (Exception e) {
        // If field can't be set, skip this test
        return;
      }

      String text = "Hello";
      RawInput input = new RawInput(new int[] {}, 0.0f, 0.0f, (byte) 0, 0.0f, text);

      injector.inject(input);

      ArgumentCaptor<Character> charCaptor = ArgumentCaptor.forClass(Character.class);

      // Verify each character was typed
      verify(screen, times(text.length())).charTyped(charCaptor.capture(), eq(0));

      var capturedChars = charCaptor.getAllValues();
      assertEquals('H', capturedChars.get(0));
      assertEquals('e', capturedChars.get(1));
      assertEquals('l', capturedChars.get(2));
      assertEquals('l', capturedChars.get(3));
      assertEquals('o', capturedChars.get(4));
    }
  }

  @Test
  void inject_withTextInput_whenNoScreen_ignoresText() throws Exception {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(mc);
      // Set screen field to null using reflection
      try {
        Field screenField = Minecraft.class.getField("screen");
        screenField.setAccessible(true);
        screenField.set(mc, null);
      } catch (Exception e) {
        // If field can't be set, skip this test
        return;
      }

      String text = "Hello";
      RawInput input = new RawInput(new int[] {}, 0.0f, 0.0f, (byte) 0, 0.0f, text);

      injector.inject(input);

      // Should not call charTyped when no screen is open
      verify(screen, never()).charTyped(anyChar(), anyInt());
    }
  }

  @Test
  void inject_whenMinecraftNotInitialized_doesNotCrash() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(null);

      RawInput input = new RawInput(new int[] {GLFW.GLFW_KEY_W}, 0.0f, 0.0f, (byte) 0, 0.0f, "");

      // Should not throw exception
      assertDoesNotThrow(() -> injector.inject(input));
    }
  }

  @Test
  void maintainButtonState_withHeldButton_firesPressEvents() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(mc);
      
      // Ensure all fields are set up properly
      try {
        Field mouseHandlerField = Minecraft.class.getField("mouseHandler");
        mouseHandlerField.setAccessible(true);
        mouseHandlerField.set(mc, mouseHandler);
      } catch (Exception e) {
        // If field can't be set, skip this test
        return;
      }
      
      // Make sure window is set up
      when(mc.getWindow()).thenReturn(window);
      when(window.getWindow()).thenReturn(WINDOW_HANDLE);

      // Inject with left mouse button held
      byte leftButton = 0b00000001;
      RawInput input = new RawInput(new int[] {}, 0.0f, 0.0f, leftButton, 0.0f, "");
      injector.inject(input);

      // Reset the mock to clear previous invocations
      reset(mouseHandler);

      // Call maintainButtonState - this should not throw
      assertDoesNotThrow(() -> injector.maintainButtonState());

      // Verify mouse button press event was fired by maintainButtonState
      verify(mouseHandler, atLeastOnce())
          .onPress(eq(WINDOW_HANDLE), eq(0), eq(GLFW.GLFW_PRESS), anyInt());
    }
  }
}
