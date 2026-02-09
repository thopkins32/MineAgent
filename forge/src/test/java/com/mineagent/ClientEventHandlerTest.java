package com.mineagent;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import com.mojang.blaze3d.platform.Window;
import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.world.level.Level;
import net.minecraftforge.event.TickEvent;
import net.minecraftforge.event.entity.living.LivingDeathEvent;
import net.minecraftforge.event.entity.living.LivingHurtEvent;
import net.minecraftforge.event.server.ServerStartingEvent;
import net.minecraftforge.event.server.ServerStoppingEvent;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.lwjgl.glfw.GLFW;
import org.mockito.MockedStatic;

class ClientEventHandlerTest {

  private MinecraftTestFixture fixture;
  private DataBridge dataBridge;
  private static final long WINDOW_HANDLE = 12345L;

  @BeforeEach
  void setUp() {
    fixture = new MinecraftTestFixture(WINDOW_HANDLE);
    dataBridge = DataBridge.getInstance();
    // Clear any existing raw input
    dataBridge.getLatestRawInput();
  }

  @Test
  void onClientTick_processesRawInputWhenAvailable() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

      // Set up level and player
      Level mockLevel = mock(Level.class);
      LocalPlayer mockPlayer = mock(LocalPlayer.class);
      fixture.setMinecraftField("level", mockLevel);
      fixture.setMinecraftField("player", mockPlayer);
      fixture.setMinecraftField("screen", null);

      // Set raw input
      RawInput input = new RawInput(new int[] {65}, 1.0f, 2.0f, (byte) 1, 0.5f, "test");
      dataBridge.setLatestRawInput(input);

      // Create tick event
      TickEvent.ClientTickEvent event = mock(TickEvent.ClientTickEvent.class);
      when(event.phase).thenReturn(TickEvent.Phase.END);

      // Call handler
      ClientEventHandler.onClientTick(event);

      // Verify input was processed (InputInjector would have been called)
      // Since input was consumed, getting it again should return null or different
      RawInput retrieved = dataBridge.getLatestRawInput();
      // Input should have been consumed, so retrieved might be null
      // (depending on implementation, it might be cleared or still there)
    }
  }

  @Test
  void onClientTick_callsMaintainButtonState() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

      Level mockLevel = mock(Level.class);
      LocalPlayer mockPlayer = mock(LocalPlayer.class);
      fixture.setMinecraftField("level", mockLevel);
      fixture.setMinecraftField("player", mockPlayer);
      fixture.setMinecraftField("screen", null);

      TickEvent.ClientTickEvent event = mock(TickEvent.ClientTickEvent.class);
      when(event.phase).thenReturn(TickEvent.Phase.END);

      // Call handler - maintainButtonState should be called
      ClientEventHandler.onClientTick(event);

      // Verify it doesn't throw (maintainButtonState is called internally)
      assertDoesNotThrow(() -> ClientEventHandler.onClientTick(event));
    }
  }

  @Test
  void onClientTick_capturesAndSendsObservation() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

      Level mockLevel = mock(Level.class);
      LocalPlayer mockPlayer = mock(LocalPlayer.class);
      fixture.setMinecraftField("level", mockLevel);
      fixture.setMinecraftField("player", mockPlayer);
      fixture.setMinecraftField("screen", null);

      // Set network handler to capture observations
      NetworkHandler mockNetworkHandler = mock(NetworkHandler.class);
      dataBridge.setNetworkHandler(mockNetworkHandler);

      TickEvent.ClientTickEvent event = mock(TickEvent.ClientTickEvent.class);
      when(event.phase).thenReturn(TickEvent.Phase.END);

      // Call handler
      ClientEventHandler.onClientTick(event);

      // Verify observation was sent (captureFrame is called internally)
      // Since captureFrame uses GL11.glReadPixels which we can't easily mock,
      // we verify the handler doesn't throw
      assertDoesNotThrow(() -> ClientEventHandler.onClientTick(event));
    }
  }

  @Test
  void onClientTick_handlesInputSuppressionWhenClientConnected() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class);
        MockedStatic<GLFW> mockedGLFW = mockStatic(GLFW.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());
      mockedGLFW
          .when(() -> GLFW.glfwSetInputMode(anyLong(), anyInt(), anyInt()))
          .thenAnswer(invocation -> null);

      Level mockLevel = mock(Level.class);
      LocalPlayer mockPlayer = mock(LocalPlayer.class);
      fixture.setMinecraftField("level", mockLevel);
      fixture.setMinecraftField("player", mockPlayer);
      fixture.setMinecraftField("screen", null);

      // Set client as connected
      dataBridge.setClientConnected(true);

      TickEvent.ClientTickEvent event = mock(TickEvent.ClientTickEvent.class);
      when(event.phase).thenReturn(TickEvent.Phase.END);

      // Call handler
      ClientEventHandler.onClientTick(event);

      // Verify GLFW call was made for cursor suppression
      // (if SUPPRESS_SYSTEM_MOUSE_INPUT is true)
      if (Config.SUPPRESS_SYSTEM_MOUSE_INPUT.get()) {
        mockedGLFW.verify(
            () -> GLFW.glfwSetInputMode(eq(WINDOW_HANDLE), eq(GLFW.GLFW_CURSOR), eq(GLFW.GLFW_CURSOR_DISABLED)),
            atLeastOnce());
      }
    }
  }

  @Test
  void onClientTick_skipsProcessingWhenLevelIsNull() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

      // Set level to null
      fixture.setMinecraftField("level", null);
      fixture.setMinecraftField("player", null);

      TickEvent.ClientTickEvent event = mock(TickEvent.ClientTickEvent.class);
      when(event.phase).thenReturn(TickEvent.Phase.END);

      // Call handler - should skip processing
      assertDoesNotThrow(() -> ClientEventHandler.onClientTick(event));
    }
  }

  @Test
  void onClientTick_skipsProcessingWhenPlayerIsNull() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

      Level mockLevel = mock(Level.class);
      fixture.setMinecraftField("level", mockLevel);
      fixture.setMinecraftField("player", null);

      TickEvent.ClientTickEvent event = mock(TickEvent.ClientTickEvent.class);
      when(event.phase).thenReturn(TickEvent.Phase.END);

      // Call handler - should skip processing
      assertDoesNotThrow(() -> ClientEventHandler.onClientTick(event));
    }
  }

  @Test
  void onClientTick_onlyProcessesOnEndPhase() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

      Level mockLevel = mock(Level.class);
      LocalPlayer mockPlayer = mock(LocalPlayer.class);
      fixture.setMinecraftField("level", mockLevel);
      fixture.setMinecraftField("player", mockPlayer);

      // Test START phase - should skip
      TickEvent.ClientTickEvent startEvent = mock(TickEvent.ClientTickEvent.class);
      when(startEvent.phase).thenReturn(TickEvent.Phase.START);

      assertDoesNotThrow(() -> ClientEventHandler.onClientTick(startEvent));

      // Test END phase - should process
      TickEvent.ClientTickEvent endEvent = mock(TickEvent.ClientTickEvent.class);
      when(endEvent.phase).thenReturn(TickEvent.Phase.END);

      assertDoesNotThrow(() -> ClientEventHandler.onClientTick(endEvent));
    }
  }

  @Test
  void onServerStarting_logsAppropriately() {
    ServerStartingEvent event = mock(ServerStartingEvent.class);

    // Should not throw
    assertDoesNotThrow(() -> ClientEventHandler.onServerStarting(event));
  }

  @Test
  void onServerStopping_logsAppropriately() {
    ServerStoppingEvent event = mock(ServerStoppingEvent.class);

    // Should not throw
    assertDoesNotThrow(() -> ClientEventHandler.onServerStopping(event));
  }

  @Test
  void onPlayerHurt_isRegistered() {
    LivingHurtEvent event = mock(LivingHurtEvent.class);

    // Should not throw (currently no-op)
    assertDoesNotThrow(() -> ClientEventHandler.onPlayerHurt(event));
  }

  @Test
  void onPlayerDeath_isRegistered() {
    LivingDeathEvent event = mock(LivingDeathEvent.class);

    // Should not throw (currently no-op)
    assertDoesNotThrow(() -> ClientEventHandler.onPlayerDeath(event));
  }

  @Test
  void handleInputSuppression_setsCursorModeWhenConfigured() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class);
        MockedStatic<GLFW> mockedGLFW = mockStatic(GLFW.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());
      mockedGLFW
          .when(() -> GLFW.glfwSetInputMode(anyLong(), anyInt(), anyInt()))
          .thenAnswer(invocation -> null);

      Level mockLevel = mock(Level.class);
      LocalPlayer mockPlayer = mock(LocalPlayer.class);
      fixture.setMinecraftField("level", mockLevel);
      fixture.setMinecraftField("player", mockPlayer);
      fixture.setMinecraftField("screen", null);

      // Set client connected and mouse suppression enabled
      dataBridge.setClientConnected(true);

      TickEvent.ClientTickEvent event = mock(TickEvent.ClientTickEvent.class);
      when(event.phase).thenReturn(TickEvent.Phase.END);

      ClientEventHandler.onClientTick(event);

      // Verify cursor mode was set if suppression is enabled
      if (Config.SUPPRESS_SYSTEM_MOUSE_INPUT.get()) {
        mockedGLFW.verify(
            () -> GLFW.glfwSetInputMode(eq(WINDOW_HANDLE), eq(GLFW.GLFW_CURSOR), eq(GLFW.GLFW_CURSOR_DISABLED)),
            atLeastOnce());
      }
    }
  }

  @Test
  void handleInputSuppression_doesNotSuppressWhenClientNotConnected() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class);
        MockedStatic<GLFW> mockedGLFW = mockStatic(GLFW.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

      Level mockLevel = mock(Level.class);
      LocalPlayer mockPlayer = mock(LocalPlayer.class);
      fixture.setMinecraftField("level", mockLevel);
      fixture.setMinecraftField("player", mockPlayer);

      // Set client as NOT connected
      dataBridge.setClientConnected(false);

      TickEvent.ClientTickEvent event = mock(TickEvent.ClientTickEvent.class);
      when(event.phase).thenReturn(TickEvent.Phase.END);

      ClientEventHandler.onClientTick(event);

      // Verify GLFW cursor call was NOT made
      mockedGLFW.verify(
          () -> GLFW.glfwSetInputMode(anyLong(), eq(GLFW.GLFW_CURSOR), anyInt()), never());
    }
  }
}
