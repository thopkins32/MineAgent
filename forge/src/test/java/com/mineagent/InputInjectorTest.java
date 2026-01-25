package com.mineagent;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.lwjgl.glfw.GLFW;
import org.mockito.ArgumentCaptor;
import org.mockito.MockedStatic;
import org.mockito.quality.Strictness;

class InputInjectorTest {

  private MinecraftTestFixture fixture;
  private InputInjector injector;
  private static final long WINDOW_HANDLE = 12345L;

  @BeforeEach
  void setUp() {
    fixture = new MinecraftTestFixture(WINDOW_HANDLE);
    injector = new InputInjector();
  }

  @Test
  void inject_withNewKeyPressed_firesKeyPressEvent() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

      RawInput input = new RawInput(new int[] {GLFW.GLFW_KEY_W}, 0.0f, 0.0f, (byte) 0, 0.0f, "");

      injector.inject(input);

      ArgumentCaptor<Integer> keyCodeCaptor = ArgumentCaptor.forClass(Integer.class);
      ArgumentCaptor<Integer> actionCaptor = ArgumentCaptor.forClass(Integer.class);

      verify(fixture.getKeyboardHandler(), times(1))
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
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

      // First inject with W key (press)
      RawInput pressInput = new RawInput(new int[] {GLFW.GLFW_KEY_W}, 0.0f, 0.0f, (byte) 0, 0.0f, "");
      injector.inject(pressInput);

      // Then inject with no keys (release)
      RawInput releaseInput = new RawInput(new int[] {}, 0.0f, 0.0f, (byte) 0, 0.0f, "");
      injector.inject(releaseInput);

      ArgumentCaptor<Integer> keyCodeCaptor = ArgumentCaptor.forClass(Integer.class);
      ArgumentCaptor<Integer> actionCaptor = ArgumentCaptor.forClass(Integer.class);

      // Verify release was called
      verify(fixture.getKeyboardHandler(), atLeastOnce())
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
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

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
      verify(fixture.getKeyboardHandler(), times(1))
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
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

      RawInput input =
          new RawInput(new int[] {GLFW.GLFW_KEY_LEFT_SHIFT}, 0.0f, 0.0f, (byte) 0, 0.0f, "");

      injector.inject(input);

      ArgumentCaptor<Integer> keyCodeCaptor = ArgumentCaptor.forClass(Integer.class);

      // Shift should fire as a regular key when pressed alone
      verify(fixture.getKeyboardHandler(), times(1))
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
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

      byte leftButton = 0b00000001; // Left mouse button (bit 0)
      RawInput input = new RawInput(new int[] {}, 0.0f, 0.0f, leftButton, 0.0f, "");

      injector.inject(input);

      verify(fixture.getMouseHandler(), times(1))
          .onPress(eq(WINDOW_HANDLE), eq(0), eq(GLFW.GLFW_PRESS), anyInt());
    }
  }

  @Test
  void inject_withMouseButtonReleased_firesMouseReleaseEvent() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

      // First inject with left mouse button (press)
      byte leftButton = 0b00000001;
      RawInput pressInput = new RawInput(new int[] {}, 0.0f, 0.0f, leftButton, 0.0f, "");
      injector.inject(pressInput);

      // Then inject with no mouse buttons (release)
      RawInput releaseInput = new RawInput(new int[] {}, 0.0f, 0.0f, (byte) 0, 0.0f, "");
      injector.inject(releaseInput);

      ArgumentCaptor<Integer> actionCaptor = ArgumentCaptor.forClass(Integer.class);

      // Verify release was called
      verify(fixture.getMouseHandler(), atLeastOnce())
          .onPress(eq(WINDOW_HANDLE), eq(0), actionCaptor.capture(), anyInt());

      // Check that release was called
      assertTrue(
          actionCaptor.getValue().equals(GLFW.GLFW_RELEASE),
          "Mouse release event should have been fired last");
    }
  }

  /*
  * TODO: Can't mock LocalPlayer due to static initialization issues.
  * Need to rework code to use dependency injection to allow for easier testing.
  @Test
  void inject_withMouseMovement_rotatesPlayer() throws Exception {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());
      // Create player mock manually to avoid initialization issues
      LocalPlayer player = mock(LocalPlayer.class, withSettings().strictness(Strictness.LENIENT));
      // Set player field using fixture helper
      fixture.setMinecraftField("player", player);
      // Set screen to null using fixture helper
      fixture.setMinecraftField("screen", null);
      
      // Mock sensitivity() to return an OptionInstance that returns 0.5
      net.minecraft.client.OptionInstance<Double> sensitivityOption =
          mock(net.minecraft.client.OptionInstance.class, withSettings().strictness(Strictness.LENIENT));
      when(sensitivityOption.get()).thenReturn(0.5);
      when(fixture.getOptions().sensitivity()).thenReturn(sensitivityOption);

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
  */

  @Test
  void reset_withPressedKeys_releasesAllKeys() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

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
      verify(fixture.getKeyboardHandler(), atLeast(2))
          .keyPress(
              eq(WINDOW_HANDLE),
              keyCodeCaptor.capture(),
              anyInt(),
              eq(GLFW.GLFW_RELEASE),
              anyInt());

      // Verify all mouse buttons were released
      verify(fixture.getMouseHandler(), atLeastOnce())
          .onPress(eq(WINDOW_HANDLE), eq(0), eq(GLFW.GLFW_RELEASE), anyInt());
      verify(fixture.getMouseHandler(), atLeastOnce())
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
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

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
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());
      // Set screen field using fixture helper
      fixture.setMinecraftField("screen", fixture.getScreen());

      String text = "Hello";
      RawInput input = new RawInput(new int[] {}, 0.0f, 0.0f, (byte) 0, 0.0f, text);

      injector.inject(input);

      ArgumentCaptor<Character> charCaptor = ArgumentCaptor.forClass(Character.class);

      // Verify each character was typed
      verify(fixture.getScreen(), times(text.length())).charTyped(charCaptor.capture(), eq(0));

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
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());
      // Set screen field to null using fixture helper
      fixture.setMinecraftField("screen", null);

      String text = "Hello";
      RawInput input = new RawInput(new int[] {}, 0.0f, 0.0f, (byte) 0, 0.0f, text);

      injector.inject(input);

      // Should not call charTyped when no screen is open
      verify(fixture.getScreen(), never()).charTyped(anyChar(), anyInt());
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
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

      // Inject with left mouse button held
      byte leftButton = 0b00000001;
      RawInput input = new RawInput(new int[] {}, 0.0f, 0.0f, leftButton, 0.0f, "");
      injector.inject(input);

      // Reset the mock to clear previous invocations
      reset(fixture.getMouseHandler());

      // Call maintainButtonState - this should not throw
      assertDoesNotThrow(() -> injector.maintainButtonState());

      // Verify mouse button press event was fired by maintainButtonState
      verify(fixture.getMouseHandler(), atLeastOnce())
          .onPress(eq(WINDOW_HANDLE), eq(0), eq(GLFW.GLFW_PRESS), anyInt());
    }
  }
}
